use super::{
    index_allocator::{MemoryInModule, ModuleAffinityIndexAllocator, SlotId},
    MemoryAllocationIndex,
};
use crate::{
    mpk::{self, PkeyRef},
    AutoEnabled, CompiledModuleId, InstanceAllocationRequest, Memory, MemoryImageSlot, Mmap,
    PoolingInstanceAllocatorConfig,
};
use anyhow::{anyhow, bail, Context, Result};
use libc::c_void;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};
use wasmtime_environ::{
    DefinedMemoryIndex, MemoryPlan, MemoryStyle, Module, Tunables, WASM_PAGE_SIZE,
};

/// A set of allocator slots.
///
/// The allocated slots can be split by striping them: e.g., with two stripe
/// colors 0 and 1, we would allocate all even slots using stripe 0 and all odd
/// slots using stripe 1.
///
/// This is helpful for the use of protection keys: (a) if a request comes to
/// allocate multiple instances, we can allocate them all from the same stripe
/// and (b) if a store wants to allocate more from the same stripe it can.
#[derive(Debug)]
struct Stripe {
    allocator: ModuleAffinityIndexAllocator,
    pkey: Option<PkeyRef>,
}

/// Represents a pool of WebAssembly linear memories.
///
/// A linear memory is divided into accessible pages and guard pages.
///
/// A diagram for this struct's fields is:
///
/// ```text
///                       memory_size
///                           /
///         max_accessible   /                    memory_and_guard_size
///                 |       /                               |
///              <--+--->  /                    <-----------+---------->
///              <--------+->
///
/// +-----------+--------+---+-----------+     +--------+---+-----------+
/// | PROT_NONE |            | PROT_NONE | ... |            | PROT_NONE |
/// +-----------+--------+---+-----------+     +--------+---+-----------+
/// |           |<------------------+---------------------------------->
/// \           |                    \
/// mapping     |            `max_total_memories` memories
///            /
///    pre_slab_guard_size
/// ```
#[derive(Debug)]
pub struct MemoryPool {
    mapping: Mmap,
    /// This memory pool is stripe-aware. If using  memory protection keys, this
    /// will contain one stripe per available key; otherwise, a single stripe
    /// with an empty key.
    stripes: Vec<Stripe>,
    // If using a copy-on-write allocation scheme, the slot management. We
    // dynamically transfer ownership of a slot to a Memory when in
    // use.
    image_slots: Vec<Mutex<Option<MemoryImageSlot>>>,
    // The size, in bytes, of each linear memory's reservation, not including
    // any guard region.
    memory_size: usize,
    // The size, in bytes, of each linear memory's reservation plus the trailing
    // guard region allocated for it.
    memory_and_guard_size: usize,
    // The maximum size that can become accessible, in bytes, of each linear
    // memory. Guaranteed to be a whole number of wasm pages.
    max_accessible: usize,
    // The size, in bytes, of the offset to the first linear memory in this
    // pool. This is here to help account for the first region of guard pages,
    // if desired, before the first linear memory.
    pre_slab_guard_size: usize,
    // The maximum number of memories that can be allocated concurrently, aka
    // our pool's capacity.
    max_total_memories: usize,
    // The maximum number of memories that a single core module instance may
    // use.
    //
    // NB: this is needed for validation but does not affect the pool's size.
    memories_per_instance: usize,
    // How much linear memory, in bytes, to keep resident after resetting for
    // use with the next instance. This much memory will be `memset` to zero
    // when a linear memory is deallocated.
    //
    // Memory exceeding this amount in the wasm linear memory will be released
    // with `madvise` back to the kernel.
    //
    // Only applicable on Linux.
    keep_resident: usize,
    // Keep track of protection keys handed out to initialized stores; this
    // allows us to round-robin the assignment of stores to stripes.
    next_available_pkey: AtomicUsize,
}

impl MemoryPool {
    /// Create a new `MemoryPool`.
    pub fn new(config: &PoolingInstanceAllocatorConfig, tunables: &Tunables) -> Result<Self> {
        // The maximum module memory page count cannot exceed 65536 pages
        if config.limits.memory_pages > 0x10000 {
            bail!(
                "module memory page limit of {} exceeds the maximum of 65536",
                config.limits.memory_pages
            );
        }

        let pkeys = match config.memory_protection_keys {
            AutoEnabled::Auto => {
                if mpk::is_supported() {
                    mpk::keys()
                } else {
                    &[]
                }
            }
            AutoEnabled::Enable => {
                if mpk::is_supported() {
                    mpk::keys()
                } else {
                    bail!("mpk is disabled on this system")
                }
            }
            AutoEnabled::Disable => &[],
        };

        // Interpret the larger of the maximal size of memory or the static
        // memory bound as the size of the virtual address space reservation for
        // memory itself. Typically `static_memory_bound` is 4G which helps
        // elide most bounds checks in wasm. If `memory_pages` is larger,
        // though, then this is a non-moving pooling allocator so we create
        // larger reservations to account for that.
        let max_memory_bytes = config.limits.memory_pages.max(tunables.static_memory_bound)
            * u64::from(WASM_PAGE_SIZE);

        // Create a slab layout and allocate it as completely inaccessible
        // region to start--`PROT_NONE`.
        let constraints = SlabConstraints {
            max_memory_bytes: max_memory_bytes as usize,
            num_memory_slots: config.limits.total_memories as usize,
            num_pkeys_available: pkeys.len(),
            guard_bytes: tunables.static_memory_offset_guard_size as usize,
            guard_before_slots: tunables.guard_before_linear_memory,
        };
        let layout = calculate(&constraints)?;
        dbg!(&constraints);
        dbg!(&layout);
        log::debug!("creating memory pool: {constraints:?} -> {layout:?}");
        let mut mapping = Mmap::accessible_reserved(0, layout.total_slab_bytes)
            .context("failed to create memory pool mapping")?;

        // Then, stripe the memory with the available protection keys. This is
        // unnecessary if there is only one stripe color.
        if layout.num_stripes >= 2 {
            let mut cursor = layout.pre_slab_guard_bytes;
            let pkeys = &pkeys[..layout.num_stripes];
            for i in 0..constraints.num_memory_slots {
                let pkey = &pkeys[i % pkeys.len()];
                let region = unsafe { mapping.slice_mut(cursor..cursor + layout.slot_bytes) };
                pkey.as_ref().mark(region)?;
                cursor += layout.slot_bytes;
            }
            debug_assert_eq!(
                cursor + layout.post_slab_guard_bytes,
                layout.total_slab_bytes
            );
        }

        let image_slots: Vec<_> = std::iter::repeat_with(|| Mutex::new(None))
            .take(constraints.num_memory_slots)
            .collect();

        let create_stripe = |(i, pkey): (usize, &PkeyRef)| {
            let num_slots = constraints.num_memory_slots / layout.num_stripes
                + usize::from(constraints.num_memory_slots % layout.num_stripes > i);
            let allocator = ModuleAffinityIndexAllocator::new(
                num_slots.try_into().unwrap(),
                config.max_unused_warm_slots,
            );
            Stripe {
                allocator,
                pkey: Some(pkey.clone()),
            }
        };

        let stripes: Vec<_> = pkeys
            .into_iter()
            .take(layout.num_stripes)
            .enumerate()
            .map(create_stripe)
            .collect();

        let pool = Self {
            stripes,
            mapping,
            image_slots,
            memory_size: constraints.max_memory_bytes,
            memory_and_guard_size: layout.slot_bytes,
            pre_slab_guard_size: layout.pre_slab_guard_bytes,
            max_total_memories: constraints.num_memory_slots,
            memories_per_instance: usize::try_from(config.limits.max_memories_per_module).unwrap(),
            max_accessible: (config.limits.memory_pages as usize) * (WASM_PAGE_SIZE as usize),
            keep_resident: config.linear_memory_keep_resident,
            next_available_pkey: AtomicUsize::new(0),
        };

        Ok(pool)
    }

    /// Return a protection key that stores can use for requesting new
    pub fn get_next_pkey(&self) -> Option<PkeyRef> {
        let index = self.next_available_pkey.fetch_add(1, Ordering::SeqCst) % self.stripes.len();
        debug_assert!(
            self.stripes.len() < 2 || self.stripes[index].pkey.is_some(),
            "if we are using stripes, we cannot have an empty protection key"
        );
        self.stripes[index].pkey.clone()
    }

    /// Validate whether this memory pool supports the given module.
    pub fn validate(&self, module: &Module) -> Result<()> {
        let memories = module.memory_plans.len() - module.num_imported_memories;
        if memories > usize::try_from(self.memories_per_instance).unwrap() {
            bail!(
                "defined memories count of {} exceeds the per-instance limit of {}",
                memories,
                self.memories_per_instance,
            );
        }

        for (i, plan) in module
            .memory_plans
            .iter()
            .skip(module.num_imported_memories)
        {
            match plan.style {
                MemoryStyle::Static { bound } => {
                    if u64::try_from(self.memory_size).unwrap() < bound {
                        bail!(
                            "memory size allocated per-memory is too small to \
                             satisfy static bound of {bound:#x} pages"
                        );
                    }
                }
                MemoryStyle::Dynamic { .. } => {}
            }
            let max = self.max_accessible / (WASM_PAGE_SIZE as usize);
            if plan.memory.minimum > u64::try_from(max).unwrap() {
                bail!(
                    "memory index {} has a minimum page size of {} which exceeds the limit of {}",
                    i.as_u32(),
                    plan.memory.minimum,
                    max,
                );
            }
        }
        Ok(())
    }

    /// Are zero slots in use right now?
    pub fn is_empty(&self) -> bool {
        for stripe in &self.stripes {
            if !stripe.allocator.is_empty() {
                return false;
            }
        }
        true
    }

    /// Allocate a single memory for the given instance allocation request.
    pub fn allocate(
        &self,
        request: &mut InstanceAllocationRequest,
        memory_plan: &MemoryPlan,
        memory_index: DefinedMemoryIndex,
    ) -> Result<(MemoryAllocationIndex, Memory)> {
        let stripe_index = if let Some(pkey) = &request.pkey {
            pkey.as_ref().as_stripe()
        } else {
            debug_assert!(self.stripes.len() < 2);
            0
        };

        let striped_allocation_index = self.stripes[stripe_index]
            .allocator
            .alloc(
                request
                    .runtime_info
                    .unique_id()
                    .map(|id| MemoryInModule(id, memory_index)),
            )
            .map(|slot| StripedAllocationIndex(u32::try_from(slot.index()).unwrap()))
            .ok_or_else(|| {
                anyhow!(
                    "maximum concurrent memory limit of {} reached for stripe {}",
                    self.stripes[stripe_index].allocator.len(),
                    stripe_index
                )
            })?;
        let allocation_index =
            striped_allocation_index.as_unstriped_slot_index(stripe_index, self.stripes.len());

        match (|| {
            // Double-check that the runtime requirements of the memory are
            // satisfied by the configuration of this pooling allocator. This
            // should be returned as an error through `validate_memory_plans`
            // but double-check here to be sure.
            match memory_plan.style {
                MemoryStyle::Static { bound } => {
                    let bound = bound * u64::from(WASM_PAGE_SIZE);
                    assert!(bound <= u64::try_from(self.memory_size).unwrap());
                }
                MemoryStyle::Dynamic { .. } => {}
            }

            let base_ptr = self.get_base(allocation_index);
            let base_capacity = self.max_accessible;

            let mut slot = self.take_memory_image_slot(allocation_index);
            let image = request.runtime_info.memory_image(memory_index)?;
            let initial_size = memory_plan.memory.minimum * WASM_PAGE_SIZE as u64;

            // If instantiation fails, we can propagate the error
            // upward and drop the slot. This will cause the Drop
            // handler to attempt to map the range with PROT_NONE
            // memory, to reserve the space while releasing any
            // stale mappings. The next use of this slot will then
            // create a new slot that will try to map over
            // this, returning errors as well if the mapping
            // errors persist. The unmap-on-drop is best effort;
            // if it fails, then we can still soundly continue
            // using the rest of the pool and allowing the rest of
            // the process to continue, because we never perform a
            // mmap that would leave an open space for someone
            // else to come in and map something.
            slot.instantiate(initial_size as usize, image, memory_plan)?;

            Memory::new_static(
                memory_plan,
                base_ptr,
                base_capacity,
                slot,
                self.memory_and_guard_size,
                unsafe { &mut *request.store.get().unwrap() },
            )
        })() {
            Ok(memory) => Ok((allocation_index, memory)),
            Err(e) => {
                self.stripes[stripe_index]
                    .allocator
                    .free(SlotId(striped_allocation_index.0));
                Err(e)
            }
        }
    }

    /// Deallocate a previously-allocated memory.
    ///
    /// # Safety
    ///
    /// The memory must have been previously allocated from this pool and
    /// assigned the given index, must currently be in an allocated state, and
    /// must never be used again.
    pub unsafe fn deallocate(&self, allocation_index: MemoryAllocationIndex, memory: Memory) {
        let mut image = memory.unwrap_static_image();

        // Reset the image slot. If there is any error clearing the
        // image, just drop it here, and let the drop handler for the
        // slot unmap in a way that retains the address space
        // reservation.
        if image.clear_and_remain_ready(self.keep_resident).is_ok() {
            self.return_memory_image_slot(allocation_index, image);
        }

        let stripe_index = allocation_index.index() % self.stripes.len();
        self.stripes[stripe_index]
            .allocator
            .free(SlotId(allocation_index.0));
    }

    /// Purging everything related to `module`.
    pub fn purge_module(&self, module: CompiledModuleId) {
        // This primarily means clearing out all of its memory images present in
        // the virtual address space. Go through the index allocator for slots
        // affine to `module` and reset them, freeing up the index when we're
        // done.
        //
        // Note that this is only called when the specified `module` won't be
        // allocated further (the module is being dropped) so this shouldn't hit
        // any sort of infinite loop since this should be the final operation
        // working with `module`.
        //
        // TODO: We are given a module id, but key affinity by pair of module id
        // and defined memory index. We are missing any defined memory index or
        // count of how many memories the module defines here. Therefore, we
        // probe up to the maximum number of memories per instance. This is fine
        // because that maximum is generally relatively small. If this method
        // somehow ever gets hot because of unnecessary probing, we should
        // either pass in the actual number of defined memories for the given
        // module to this method, or keep a side table of all slots that are
        // associated with a module (not just module and memory). The latter
        // would require care to make sure that its maintenance wouldn't be too
        // expensive for normal allocation/free operations.
        for stripe in &self.stripes {
            for i in 0..self.memories_per_instance {
                use wasmtime_environ::EntityRef;
                let memory_index = DefinedMemoryIndex::new(i);
                while let Some(id) = stripe
                    .allocator
                    .alloc_affine_and_clear_affinity(module, memory_index)
                {
                    // Clear the image from the slot and, if successful, return it back
                    // to our state. Note that on failure here the whole slot will get
                    // paved over with an anonymous mapping.
                    let index = MemoryAllocationIndex(id.0);
                    let mut slot = self.take_memory_image_slot(index);
                    if slot.remove_image().is_ok() {
                        self.return_memory_image_slot(index, slot);
                    }

                    stripe.allocator.free(id);
                }
            }
        }
    }

    fn get_base(&self, allocation_index: MemoryAllocationIndex) -> *mut u8 {
        assert!(allocation_index.index() < self.max_total_memories);
        let offset =
            self.pre_slab_guard_size + allocation_index.index() * self.memory_and_guard_size;
        unsafe { self.mapping.as_ptr().offset(offset as isize).cast_mut() }
    }

    /// Take ownership of the given image slot. Must be returned via
    /// `return_memory_image_slot` when the instance is done using it.
    fn take_memory_image_slot(&self, allocation_index: MemoryAllocationIndex) -> MemoryImageSlot {
        let maybe_slot = self.image_slots[allocation_index.index()]
            .lock()
            .unwrap()
            .take();

        maybe_slot.unwrap_or_else(|| {
            MemoryImageSlot::create(
                self.get_base(allocation_index) as *mut c_void,
                0,
                self.max_accessible,
            )
        })
    }

    /// Return ownership of the given image slot.
    fn return_memory_image_slot(
        &self,
        allocation_index: MemoryAllocationIndex,
        slot: MemoryImageSlot,
    ) {
        assert!(!slot.is_dirty());
        *self.image_slots[allocation_index.index()].lock().unwrap() = Some(slot);
    }
}

impl Drop for MemoryPool {
    fn drop(&mut self) {
        // Clear the `clear_no_drop` flag (i.e., ask to *not* clear on
        // drop) for all slots, and then drop them here. This is
        // valid because the one `Mmap` that covers the whole region
        // can just do its one munmap.
        for mut slot in std::mem::take(&mut self.image_slots) {
            if let Some(slot) = slot.get_mut().unwrap() {
                slot.no_clear_on_drop();
            }
        }
    }
}

/// The index of a memory allocation within an `InstanceAllocator`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub struct StripedAllocationIndex(u32);

impl StripedAllocationIndex {
    fn as_unstriped_slot_index(self, stripe: usize, num_stripes: usize) -> MemoryAllocationIndex {
        let num_stripes: u32 = num_stripes.try_into().unwrap();
        let stripe: u32 = stripe.try_into().unwrap();
        MemoryAllocationIndex(self.0 * num_stripes + stripe)
    }
}

#[derive(Clone, Debug)]
struct SlabConstraints {
    max_memory_bytes: usize,
    num_memory_slots: usize,
    num_pkeys_available: usize,
    guard_bytes: usize,
    guard_before_slots: bool,
}

#[derive(Debug)]
struct SlabLayout {
    /// The total number of bytes to allocate for the memory pool slab.
    total_slab_bytes: usize,
    /// If necessary, the number of bytes to reserve as a guard region at the
    /// beginning of the slab.
    pre_slab_guard_bytes: usize,
    /// If necessary, the number of bytes to reserve as a guard region at the
    /// beginning of the slab.
    post_slab_guard_bytes: usize,
    /// The size of each slot in the memory pool; this comprehends the maximum
    /// memory size (i.e., from WebAssembly or Wasmtime configuration) plus any
    /// guard region after the memory to catch OOB access. On these guard
    /// regions, note that:
    /// - users can configure how aggressively (or not) to elide bounds checks
    ///   via `Config::static_memory_guard_size`
    /// - memory protection keys can compress the size of the guard region by
    ///   placing slots from a different key (i.e., a stripe) in the guard
    ///   region
    slot_bytes: usize,
    /// The number of stripes needed in the slab layout.
    num_stripes: usize,
}

fn calculate(constraints: &SlabConstraints) -> Result<SlabLayout> {
    let SlabConstraints {
        max_memory_bytes,
        num_memory_slots,
        num_pkeys_available,
        guard_bytes,
        guard_before_slots,
    } = *constraints;

    // If the user specifies a guard region, we always need to allocate a
    // `PROT_NONE` region for it before any memory slots. Recall that we can
    // avoid bounds checks for loads and stores with immediates up to
    // `guard_bytes`, but we rely on Wasmtime to emit bounds checks for any
    // accesses greater than this.
    let pre_slab_guard_bytes = if guard_before_slots { guard_bytes } else { 0 };

    let (num_stripes, needed_guard_bytes) = if guard_bytes == 0 || max_memory_bytes == 0 {
        // In the uncommon case where the memory or guard regions are empty, we
        // will not need any stripes: we just lay out the slots back-to-back
        // using a single stripe.
        (1, guard_bytes)
    } else if num_pkeys_available < 2 {
        // If we do not have enough protection keys to stripe the memory, we do
        // the same. We can't elide any of the guard bytes because we aren't
        // overlapping guard regions with other stripes...
        (1, guard_bytes)
    } else {
        // ...but if we can create at least two stripes, we can use another
        // stripe (i.e., a different pkey) as this slot's guard region--this
        // reduces the guard bytes each slot has to allocate. We must make sure,
        // though, that if the size of that other stripe(s) does not fully cover
        // `guard_bytes`, we keep those around to prevent OOB access.
        //
        // We first calculate the number of stripes we need: we want to minimize
        // this so that there is less chance of a single store running out of
        // slots with its stripe--we need at least two, though. But this is not
        // just an optimization; we need to handle the case when there are fewer
        // slots than stripes. E.g., if our pool is configured with only three
        // slots (`num_memory_slots = 3`), we will run into failures if we
        // attempt to set up more than three stripes.
        let needed_num_stripes =
            guard_bytes / max_memory_bytes + usize::from(guard_bytes % max_memory_bytes != 0) + 1;
        let num_stripes = num_pkeys_available.min(needed_num_stripes);
        let next_slots_overlapping_bytes = max_memory_bytes
            .checked_mul(num_stripes - 1)
            .unwrap_or(usize::MAX);
        let needed_guard_bytes = guard_bytes
            .checked_sub(next_slots_overlapping_bytes)
            .unwrap_or(0);
        (num_stripes, needed_guard_bytes)
    };

    // The page-aligned slot size; equivalent to `memory_and_guard_size`.
    let page_alignment = crate::page_size() - 1;
    let slot_bytes = max_memory_bytes
        .checked_add(needed_guard_bytes)
        .and_then(|slot_bytes| slot_bytes.checked_add(page_alignment))
        .and_then(|slot_bytes| Some(slot_bytes & !page_alignment))
        .ok_or_else(|| anyhow!("slot size is too large"))?;

    // We may need another guard region (like `pre_slab_guard_bytes`) at the end
    // of our slab. We could be conservative and just create it as large as
    // `guard_bytes`, but because we know that the last slot already has a
    // region as large as `needed_guard_bytes`, we can reduce the final guard
    // region by that much.
    let post_slab_guard_bytes = guard_bytes - needed_guard_bytes;

    // The final layout (where `n = num_memory_slots`):
    // ┌────────────────────┬──────┬──────┬───┬──────┬─────────────────────┐
    // │pre_slab_guard_bytes│slot 1│slot 2│...│slot n│post_slab_guard_bytes│
    // └────────────────────┴──────┴──────┴───┴──────┴─────────────────────┘
    let total_slab_bytes = slot_bytes
        .checked_mul(num_memory_slots)
        .and_then(|c| c.checked_add(pre_slab_guard_bytes))
        .and_then(|c| c.checked_add(post_slab_guard_bytes))
        .ok_or_else(|| anyhow!("total size of memory reservation exceeds addressable memory"))?;

    Ok(SlabLayout {
        total_slab_bytes,
        pre_slab_guard_bytes,
        post_slab_guard_bytes,
        slot_bytes,
        num_stripes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{InstanceLimits, PoolingInstanceAllocator};
    use proptest::prelude::*;
    use wasmtime_environ::WASM_PAGE_SIZE;

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn test_memory_pool() -> Result<()> {
        let pool = MemoryPool::new(
            &PoolingInstanceAllocatorConfig {
                limits: InstanceLimits {
                    total_memories: 5,
                    max_tables_per_module: 0,
                    max_memories_per_module: 3,
                    table_elements: 0,
                    memory_pages: 1,
                    ..Default::default()
                },
                ..Default::default()
            },
            &Tunables {
                static_memory_bound: 1,
                static_memory_offset_guard_size: 0,
                ..Tunables::default()
            },
        )?;

        assert_eq!(pool.memory_and_guard_size, WASM_PAGE_SIZE as usize);
        assert_eq!(pool.max_total_memories, 5);
        assert_eq!(pool.max_accessible, WASM_PAGE_SIZE as usize);

        let base = pool.mapping.as_ptr() as usize;

        for i in 0..5 {
            let index = MemoryAllocationIndex(i);
            let ptr = pool.get_base(index);
            assert_eq!(ptr as usize - base, i as usize * pool.memory_and_guard_size);
        }

        Ok(())
    }

    #[test]
    fn test_pooling_allocator_with_reservation_size_exceeded() {
        let config = PoolingInstanceAllocatorConfig {
            limits: InstanceLimits {
                total_memories: 1,
                memory_pages: 2,
                ..Default::default()
            },
            ..PoolingInstanceAllocatorConfig::default()
        };
        let pool = PoolingInstanceAllocator::new(
            &config,
            &Tunables {
                static_memory_bound: 1,
                static_memory_offset_guard_size: 0,
                ..Tunables::default()
            },
        )
        .unwrap();
        assert_eq!(pool.memories.memory_size, 2 * 65536);
    }

    #[test]
    fn test_pooling_allocator_striping() {
        if !mpk::is_supported() {
            println!("skipping `test_pooling_allocator_striping` test; mpk is not supported");
            return;
        }

        // Force the use of MPK.
        let config = PoolingInstanceAllocatorConfig {
            memory_protection_keys: AutoEnabled::Enable,
            ..PoolingInstanceAllocatorConfig::default()
        };
        let pool = MemoryPool::new(&config, &Tunables::default()).unwrap();
        assert!(pool.stripes.len() >= 2);

        let max_memory_slots = config.limits.total_memories;
        dbg!(pool.stripes[0].allocator.num_empty_slots());
        dbg!(pool.stripes[1].allocator.num_empty_slots());
        let available_memory_slots: usize = pool
            .stripes
            .iter()
            .map(|s| s.allocator.num_empty_slots())
            .sum();
        assert_eq!(max_memory_slots, available_memory_slots.try_into().unwrap());
    }

    #[test]
    fn check_known_layout_calculations() {
        for num_pkeys_available in 0..16 {
            for num_memory_slots in [0, 1, 10, 64] {
                for max_memory_bytes in
                    [0, 1 * WASM_PAGE_SIZE as usize, 10 * WASM_PAGE_SIZE as usize]
                {
                    for guard_bytes in [0, 2 << 30 /* 2GB */] {
                        for guard_before_slots in [true, false] {
                            let constraints = SlabConstraints {
                                max_memory_bytes,
                                num_memory_slots,
                                num_pkeys_available,
                                guard_bytes,
                                guard_before_slots,
                            };
                            let layout = calculate(&constraints);
                            assert_slab_layout_invariants(constraints, layout.unwrap());
                        }
                    }
                }
            }
        }
    }

    proptest! {
        #[test]
        fn check_random_layout_calculations(c in constraints()) {
            if let Ok(l) = calculate(&c) {
                assert_slab_layout_invariants(c, l);
            }
        }
    }

    fn constraints() -> impl Strategy<Value = SlabConstraints> {
        (
            any::<usize>(),
            any::<usize>(),
            any::<usize>(),
            any::<usize>(),
            any::<bool>(),
        )
            .prop_map(
                |(
                    max_memory_bytes,
                    num_memory_slots,
                    num_pkeys_available,
                    guard_bytes,
                    guard_before_slots,
                )| {
                    SlabConstraints {
                        max_memory_bytes,
                        num_memory_slots,
                        num_pkeys_available,
                        guard_bytes,
                        guard_before_slots,
                    }
                },
            )
    }

    fn assert_slab_layout_invariants(c: SlabConstraints, s: SlabLayout) {
        // Check that all the sizes add up.
        assert_eq!(
            s.total_slab_bytes,
            s.pre_slab_guard_bytes + s.slot_bytes * c.num_memory_slots + s.post_slab_guard_bytes,
            "the slab size does not add up: {c:?} => {s:?}"
        );

        // Check that the memory slot size is page-aligned.
        assert!(
            s.slot_bytes % crate::page_size() == 0,
            "slot size is not page-aligned: {c:?} => {s:?}",
        );

        // Check that we use no more or less stripes than needed.
        assert!(s.num_stripes >= 1, "not enough stripes: {c:?} => {s:?}");
        if c.num_pkeys_available == 0 {
            assert_eq!(
                s.num_stripes, 1,
                "expected at least one stripe: {c:?} => {s:?}"
            );
        } else {
            assert!(
                s.num_stripes <= c.num_pkeys_available,
                "layout has more stripes than available pkeys: {c:?} => {s:?}"
            );
        }

        // Check that we use the minimum number of stripes/protection keys.
        // - if the next slot is bigger
        if c.num_pkeys_available > 1 && c.max_memory_bytes > 0 {
            assert!(
                s.num_stripes <= (c.guard_bytes / c.max_memory_bytes) + 2,
                "calculated more stripes than needed: {c:?} => {s:?}"
            );
        }

        // Check that the memory-striping will not allow OOB access.
        if s.num_stripes > 1 {
            assert!(
                s.slot_bytes * (s.num_stripes - 1) >= c.guard_bytes,
                "layout may allow OOB access: {c:?} => {s:?}"
            );
        }
    }
}
