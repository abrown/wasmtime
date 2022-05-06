use crate::memory::{LinearMemory, MemoryCreator};
use crate::module::BareModuleInfo;
use crate::store::{InstanceId, StoreOpaque};
use crate::MemoryType;
use anyhow::{anyhow, Result};
use std::convert::TryFrom;
use std::ptr;
use std::sync::Arc;
use wasmtime_environ::{EntityIndex, MemoryIndex, MemoryPlan, MemoryStyle, Module, WASM_PAGE_SIZE};
use wasmtime_runtime::{
    allocate_single_memory_instance, DefaultMemoryCreator, Imports, InstanceAllocationRequest,
    Memory, MemoryImage, RuntimeLinearMemory, RuntimeMemoryCreator, SharedMemory, StorePtr,
    VMMemoryDefinition, VMMemoryImport,
};

pub fn create_memory(
    store: &mut StoreOpaque,
    memory_ty: &MemoryType,
    preallocation: Option<SharedMemory>,
) -> Result<InstanceId> {
    let mut module = Module::new();
    let mut imports = Imports::default();
    let memory = match &preallocation {
        // If we are passing in a pre-allocated memory, we need to import it into
        // the "frankenstein" instance and immediately export it. We know this
        // memory *must* be shared--it could be used by several instances.
        Some(shared_memory) => {
            // Construct an empty `VMMemoryImport` that we *must* fill in once
            // we have an associated `VMContext` (i.e., after allocation below).
            imports.memories = &[VMMemoryImport {
                from: ptr::null_mut(),
                vmctx: ptr::null_mut(),
            }];
            module.num_imported_memories += 1;
            shared_memory.clone().as_memory()
        }
        // If we do not have a pre-allocated memory, then we create it here and
        // associate it with the "frankenstein" instance, which now owns it.
        None => {
            let plan = wasmtime_environ::MemoryPlan::for_memory(
                memory_ty.wasmtime_memory().clone(),
                &store.engine().config().tunables,
            );
            let creator = &DefaultMemoryCreator;
            let store = unsafe {
                store
                    .traitobj()
                    .as_mut()
                    .expect("the store pointer cannot be null here")
            };
            Memory::new_dynamic(&plan, creator, store, None)?
        }
    };

    // Since we have only associated a single memory with the "frankenstein"
    // instance, it must be exported at index 0.
    module
        .exports
        .insert(String::new(), EntityIndex::Memory(MemoryIndex::from_u32(0)));

    // We create an instance in the on-demand allocator when creating handles
    // associated with external objects. The configured instance allocator
    // should only be used when creating module instances as we don't want host
    // objects to count towards instance limits.
    let runtime_info = &BareModuleInfo::maybe_imported_func(Arc::new(module), None).into_traitobj();
    let host_state = Box::new(());
    let request = InstanceAllocationRequest {
        imports,
        host_state,
        store: StorePtr::new(store.traitobj()),
        runtime_info,
    };

    unsafe {
        let handle = allocate_single_memory_instance(request, memory)?;
        let instance_id = store.add_instance(handle.clone(), true);

        // TODO: the issue here is that we have already copied the `imports` pointers
        // over into the new instance `VMContext`. See the call chain:
        // `allocate_single_memory_instance` -> `Instance::new_at` ->
        // `Instance::initialize_vmctx`.
        if let Some(mut shared_memory) = preallocation {
            todo!();
            let definition = shared_memory.vmmemory();
            let vmctx = handle.vmctx_ptr().clone();
        }

        Ok(instance_id)
    }
}

struct LinearMemoryProxy {
    mem: Box<dyn LinearMemory>,
}

impl RuntimeLinearMemory for LinearMemoryProxy {
    fn byte_size(&self) -> usize {
        self.mem.byte_size()
    }

    fn maximum_byte_size(&self) -> Option<usize> {
        self.mem.maximum_byte_size()
    }

    fn grow_to(&mut self, new_size: usize) -> Result<()> {
        self.mem.grow_to(new_size)
    }

    fn vmmemory(&mut self) -> VMMemoryDefinition {
        VMMemoryDefinition {
            base: self.mem.as_ptr(),
            current_length: self.mem.byte_size(),
        }
    }

    fn needs_init(&self) -> bool {
        true
    }

    #[cfg(feature = "pooling-allocator")]
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[derive(Clone)]
pub(crate) struct MemoryCreatorProxy(pub Arc<dyn MemoryCreator>);

impl RuntimeMemoryCreator for MemoryCreatorProxy {
    fn new_memory(
        &self,
        plan: &MemoryPlan,
        minimum: usize,
        maximum: Option<usize>,
        _: Option<&Arc<MemoryImage>>,
    ) -> Result<Box<dyn RuntimeLinearMemory>> {
        let ty = MemoryType::from_wasmtime_memory(&plan.memory);
        let reserved_size_in_bytes = match plan.style {
            MemoryStyle::Static { bound } => {
                Some(usize::try_from(bound * (WASM_PAGE_SIZE as u64)).unwrap())
            }
            MemoryStyle::Dynamic { .. } => None,
        };
        self.0
            .new_memory(
                ty,
                minimum,
                maximum,
                reserved_size_in_bytes,
                usize::try_from(plan.offset_guard_size).unwrap(),
            )
            .map(|mem| Box::new(LinearMemoryProxy { mem }) as Box<dyn RuntimeLinearMemory>)
            .map_err(|e| anyhow!(e))
    }
}
