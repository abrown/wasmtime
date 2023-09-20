//! Memory Protection Keys (MPK) implementation for use in striped memory
//! allocation.
//!
//! MPK is an x86 feature available on relatively recent versions of Intel and
//! AMD CPUs. In Linux, this feature is named `pku` (protection keys userspace)
//! and consists of three new system calls: `pkey_alloc`, `pkey_free`, and
//! `pkey_mprotect` (see the [Linux documentation]). This crate provides an
//! abstraction, [`Pkey`], that the [pooling allocator] applies to contiguous
//! memory allocations, allowing it to avoid guard pages in some cases and more
//! efficiently use memory. This technique was first presented in a 2022 paper:
//! [Segue and ColorGuard: Optimizing SFI Performance and Scalability on Modern
//! x86][colorguard].
//!
//! [pooling allocator]: crate::PoolingInstanceAllocator
//! [Linux documentation]:
//!     https://www.kernel.org/doc/html/latest/core-api/protection-keys.html
//! [colorguard]: https://plas2022.github.io/files/pdf/SegueColorGuard.pdf
//!
//! This module implements the various parts necessary to use MPK in Wasmtime:
//! - [`is_supported`] indicates whether the feature is available at runtime
//! - [`Pkey`] provides safe access to the kernel-allocated protection keys
//! - the `sys` module bridges the gap to Linux's `pkey_*` system calls
//! - the `pkru` module controls the x86 `PKRU` register (and other CPU state)

use anyhow::{Context, Result};
use std::sync::OnceLock;

/// Check if the MPK feature is supported.
pub fn is_supported() -> bool {
    cfg!(target_os = "linux") && cfg!(target_arch = "x86_64") && pkru::has_cpuid_bit_set()
    // TODO: we cannot check CR4 due to privilege
}

/// Allocate all protection keys available to this process.
///
/// This asks the kernel for all available keys (we expect 1-15; 0 is
/// kernel-reserved) in a thread-safe way. This avoids interference when
/// multiple threads try to allocate keys at the same time (e.g., during
/// testing). It also ensures that a single copy of the keys are reserved for
/// the lifetime of the process.
///
/// TODO: this is not the best-possible design. This creates global state that
/// would prevent any other code in the process from using protection keys; the
/// `KEYS` are never deallocated from the system with `pkey_dealloc`.
pub fn keys() -> &'static [ProtectionKey] {
    let keys = KEYS.get_or_init(|| {
        let mut allocated = vec![];
        if is_supported() {
            while let Ok(key_id) = sys::pkey_alloc(0, 0) {
                debug_assert!(key_id < 16);
                // UNSAFETY: here we unsafely assume that the system-allocated pkey
                // will exist forever.
                let pkey = ProtectionKey(key_id);
                debug_assert_eq!(pkey.as_stripe(), allocated.len());
                allocated.push(pkey);
            }
        }
        allocated
    });
    &keys
}
static KEYS: OnceLock<Vec<ProtectionKey>> = OnceLock::new();

/// Only allow access to pages marked by the keys set in `mask`.
///
/// Any accesses to pages marked by another key will result in a `SIGSEGV`
/// fault.
pub fn allow(mask: ProtectionMask) {
    let mut allowed = 0;
    for i in 0..16 {
        if mask.0 & (1 << i) == 1 {
            allowed |= 0b11 << (i * 2);
        }
    }

    let previous = pkru::read();
    pkru::write(pkru::DISABLE_ACCESS ^ allowed);
    log::debug!("PKRU change: {:#034b} => {:#034b}", previous, pkru::read());
}

/// An MPK protection key.
///
/// The expected usage is:
/// - allocate a new key with [`Pkey::new`]
/// - mark some regions of memory as accessible with [`Pkey::protect`]
/// - [`allow`] or disallow access to the memory regions using a
///   [`ProtectionMask`]; any accesses to unmarked pages result in a fault
/// - drop the key
///
/// Since this kernel is allocated from the kernel, we must inform the kernel
/// when it is dropped. Similarly, to retrieve all available protection keys,
/// one must request them from the kernel (e.g., call [`Pkey::new`] until it
/// fails).
///
/// Because MPK may not be available on all systems, [`Pkey`] wraps an `Option`
/// that will always be `None` if MPK is not supported. The idea here is that
/// the API can remain the same regardless of MPK support.
#[derive(Clone, Copy, Debug)]
pub struct ProtectionKey(u32);

impl ProtectionKey {
    /// Mark a page as protected by this [`Pkey`].
    ///
    /// This "colors" the pages of `region` via a kernel `pkey_mprotect` call to
    /// only allow reads and writes when this [`Pkey`] is activated (see
    /// [`Pkey::activate`]).
    ///
    /// # Errors
    ///
    /// This will fail if the region is not page aligned or for some unknown
    /// kernel reason.
    pub fn protect(&self, region: &mut [u8]) -> Result<()> {
        let addr = region.as_mut_ptr() as usize;
        let len = region.len();
        let prot = sys::PROT_READ | sys::PROT_WRITE;
        sys::pkey_mprotect(addr, len, prot, self.0).with_context(|| {
            format!(
                "failed to mark region with pkey (addr = {addr:#x}, len = {len}, prot = {prot:#b})"
            )
        })
    }

    /// Convert the [`Pkey`] to its 0-based index; this is useful for
    /// determining which allocation "stripe" a key belongs to.
    ///
    /// This function assumes that the kernel has allocated key 0 for itself.
    pub fn as_stripe(&self) -> usize {
        debug_assert!(self.0 != 0);
        self.0 as usize - 1
    }
}

/// A bit field indicating which protection keys should be *allowed*.
///
/// When bit `n` is set, it means the protection key is allowed--conversely,
/// protection is disabled for this key.
pub struct ProtectionMask(u16);
impl ProtectionMask {
    /// Allow access from all protection keys.
    pub fn all() -> Self {
        Self(u16::MAX)
    }

    /// Only allow access to memory protected with protection key 0; note that
    /// this does not mean "none" but rather allows access from the default
    /// kernel protection key.
    pub fn zero() -> Self {
        Self(1)
    }

    /// Include `pkey` as another allowed protection key in the mask.
    pub fn or(self, pkey: ProtectionKey) -> Self {
        Self(self.0 | 1 << pkey.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Mmap;
    use std::ffi::CStr;

    #[test]
    fn check_is_supported() {
        println!("is pku supported = {}", is_supported());
    }

    #[test]
    fn check_initialized_keys() {
        if is_supported() {
            assert!(!keys().is_empty())
        }
    }

    #[test]
    fn check_invalid_mark() {
        let pkey = keys()[0];
        let unaligned_region = unsafe {
            let addr = 1 as *mut u8; // this is not page-aligned!
            let len = 1;
            std::slice::from_raw_parts_mut(addr, len)
        };
        let result = pkey.protect(unaligned_region);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "failed to mark region with pkey (addr = 0x1, len = 1, prot = 0b11)"
        );
    }

    static mut tripped: bool = false;

    unsafe extern "C" fn handle_protection_fault(
        sig: libc::c_int,
        _: *mut libc::siginfo_t,
        c: *mut libc::c_void,
    ) {
        // assert!(c.is_null());
        println!("pkru: {:#034b}", pkru::read());
        let pkey_two = keys()[2];
        allow(ProtectionMask::all());

        unsafe {
            tripped = true;
        }

        let name_ptr = unsafe { libc::strsignal(sig) };
        let name = CStr::from_ptr(name_ptr).to_str().unwrap();
        println!("received signal: {name} ({sig})");
        panic!("...");
    }

    #[test]
    #[ignore]
    fn check_protection_fault() {
        // Set up signal handler.
        let mut old: libc::sigaction = unsafe { std::mem::zeroed() };
        let mut new: libc::sigaction = unsafe { std::mem::zeroed() };
        new.sa_sigaction = handle_protection_fault as usize;
        //new.sa_flags = libc::SA_SIGINFO | libc::SA_RESTART | libc::SA_RESETHAND;
        // new.sa_flags = libc::SA_SIGINFO | libc::SA_NODEFER | libc::SA_ONSTACK | libc::SA_RESETHAND;
        new.sa_flags = libc::SA_SIGINFO | libc::SA_NODEFER | libc::SA_ONSTACK;
        // if unsafe { libc::sigaction(libc::SIGSEGV, &new, &mut old) } != 0 {
        if unsafe { libc::sigaction(libc::SIGSEGV, &new, std::ptr::null_mut()) } != 0 {
            panic!("unable to set up signal handler");
            // return Err(Error::last_os_error());
        }

        let mut mmap = Mmap::with_at_least(crate::page_size()).unwrap();
        let region = unsafe { mmap.slice_mut(0..mmap.len()) };
        region[0] = 42;
        assert_eq!(42, region[0]);

        let pkey_one = keys()[1];
        let pkey_two = keys()[2];
        pkey_one.protect(region).unwrap();
        println!("pkru: {:#034b}", pkru::read());

        allow(ProtectionMask::zero().or(pkey_one));
        println!("pkru: {:#034b}", pkru::read());
        assert_eq!(42, region[0]);

        allow(ProtectionMask::zero().or(pkey_two));
        let caught = std::panic::catch_unwind(|| {
            println!("pkru: {:#034b}", pkru::read());
            assert_eq!(42, region[0]);
        });

        allow(ProtectionMask::all());
        println!("pkru: {:#034b}", pkru::read());

        assert_eq!(42, region[0]);

        // TODO: Tear down signal handler.

        // // See https://docs.rs/signal-hook-registry/1.4.0/src/signal_hook_registry/lib.rs.html#158
        // let mut sig_action = libc::sigaction {
        //     sa_sigaction: handle_sigint as libc::sighandler_t,
        //     sa_mask: todo!(),
        //     sa_flags: libc::SA_SIGINFO | libc::SA_RESTART,
        //     sa_restorer: todo!(),
        // };

        // let pkey = Pkey::new().unwrap();
        // let region = unsafe {
        //     let addr = 1 as *mut u8; // this is not page-aligned!
        //     let len = 1;
        //     std::slice::from_raw_parts_mut(addr, len)
        // };
        // let result = pkey.mark(region);
        // assert!(result.is_err());
        // assert_eq!(
        //     result.unwrap_err().to_string(),
        //     "failed to mark region with pkey (addr = 0x1, len = 1, prot = 0b11)"
        // );
    }
}

/// Expose the `pkey_*` Linux system calls. See the kernel documentation for
/// more information:
/// - [`pkeys`] overview
/// - [`pkey_alloc`] (with `pkey_free`)
/// - [`pkey_mprotect`]
/// - `pkey_set` is implemented directly in assembly.
///
/// [`pkey_alloc`]: https://man7.org/linux/man-pages/man2/pkey_alloc.2.html
/// [`pkey_mprotect`]: https://man7.org/linux/man-pages/man2/pkey_mprotect.2.html
/// [`pkeys`]: https://man7.org/linux/man-pages/man7/pkeys.7.html
pub mod sys {
    use crate::page_size;
    use anyhow::{anyhow, Result};

    /// Protection mask allowing reads of pkey-protected memory (see `prot` in
    /// [`pkey_mprotect`]).
    pub const PROT_READ: u32 = libc::PROT_READ as u32; // == 0b0001.

    /// Protection mask allowing writes of pkey-protected memory (see `prot` in
    /// [`pkey_mprotect`]).
    pub const PROT_WRITE: u32 = libc::PROT_WRITE as u32; // == 0b0010;

    /// Allocate a new protection key in the Linux kernel ([docs]); returns the
    /// key ID.
    ///
    /// [docs]: https://man7.org/linux/man-pages/man2/pkey_alloc.2.html
    ///
    /// Each process has its own separate pkey index; e.g., if process `m`
    /// allocates key 1, process `n` can as well.
    pub fn pkey_alloc(flags: u32, access_rights: u32) -> Result<u32> {
        debug_assert_eq!(flags, 0); // reserved for future use--must be 0.
        let result = unsafe { libc::syscall(libc::SYS_pkey_alloc, flags, access_rights) };
        if result >= 0 {
            Ok(result.try_into().expect("TODO"))
        } else {
            debug_assert_eq!(result, -1); // only this error result is expected.
            Err(anyhow!(unsafe { errno_as_string() }))
        }
    }

    /// Free a kernel protection key ([docs]).
    ///
    /// [docs]: https://man7.org/linux/man-pages/man2/pkey_alloc.2.html
    pub fn pkey_free(key: u32) -> Result<()> {
        let result = unsafe { libc::syscall(libc::SYS_pkey_free, key) };
        if result == 0 {
            Ok(())
        } else {
            debug_assert_eq!(result, -1); // only this error result is expected.
            Err(anyhow!(unsafe { errno_as_string() }))
        }
    }

    /// Change the access protections for a page-aligned memory region ([docs]).
    ///
    /// [docs]: https://man7.org/linux/man-pages/man2/pkey_mprotect.2.html
    pub fn pkey_mprotect(addr: usize, len: usize, prot: u32, key: u32) -> Result<()> {
        let page_size = page_size();
        if addr % page_size != 0 {
            log::warn!(
                "memory must be page-aligned for MPK (addr = {addr:#x}, page size = {page_size}"
            );
        }
        let result = unsafe { libc::syscall(libc::SYS_pkey_mprotect, addr, len, prot, key) };
        if result == 0 {
            Ok(())
        } else {
            debug_assert_eq!(result, -1); // only this error result is expected.
            Err(anyhow!(unsafe { errno_as_string() }))
        }
    }

    /// Helper function for retrieving the libc error message for the current
    /// error (see GNU libc's ["Checking for Errors"] documentation).
    ///
    /// ["Checking for Errors"]: https://www.gnu.org/software/libc/manual/html_node/Checking-for-Errors.html
    unsafe fn errno_as_string() -> String {
        let errno = *libc::__errno_location();
        let err_ptr = libc::strerror(errno);
        std::ffi::CStr::from_ptr(err_ptr)
            .to_string_lossy()
            .into_owned()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[ignore = "cannot be run when keys() has already allocated all keys"]
        #[test]
        fn check_allocate_and_free() {
            let key = pkey_alloc(0, 0).unwrap();
            assert_eq!(key, 1);
            // It may seem strange to assert the key ID here, but we already
            // make some assumptions:
            //  1. we are running on Linux with `pku` enabled
            //  2. Linux will allocate key 0 for itself
            //  3. we are running this test in non-MPK mode and no one else is
            //     using pkeys
            // If these assumptions are incorrect, this test can be removed.
            pkey_free(key).unwrap()
        }

        #[test]
        fn check_invalid_free() {
            let result = pkey_free(42);
            assert!(result.is_err());
            assert_eq!(result.unwrap_err().to_string(), "Invalid argument");
        }

        #[test]
        #[should_panic]
        fn check_invalid_alloc_flags() {
            pkey_alloc(42, 0).unwrap();
        }

        #[test]
        fn check_invalid_alloc_rights() {
            assert!(pkey_alloc(0, 42).is_err());
        }
    }
}

/// Control access to the x86 `PKRU` register.
///
/// As documented in the Intel Software Development Manual, vol 3a, section 2.7,
/// the 32 bits of the `PKRU` register laid out as follows (note the
/// little-endianness):
///
/// ```text
/// ┌───┬───┬───┬───┬───┬───┐
/// │...│AD2│WD1│AD1│WD0│AD0│
/// └───┴───┴───┴───┴───┴───┘
/// ```
///
/// - `ADn = 1` means "access disable key `n`"--no reads or writes allowed to
///   pages marked with key `n`.
/// - `WDn = 1` means "write disable key `n`"--only reads are prevented to pages
///   marked with key `n`
/// - it is unclear what it means to have both `ADn` and `WDn` set
///
/// Note that this only handles the user-mode `PKRU` register; there is an
/// equivalent supervisor-mode MSR, `IA32_PKRS`.
///
/// TODO: thread-safety? what if multiple keys try to access the PKRU register
/// for this CPU?
mod pkru {
    use core::arch::asm;

    /// This `PKRU` register mask allows access to any pages marked with any
    /// key--in other words, reading and writing is permitted to all pages.
    pub const ALLOW_ACCESS: u32 = 0;

    /// This `PKRU` register mask disables access to any page marked with any
    /// key--in other words, no reading or writing to all pages.
    //pub const DISABLE_ACCESS: u32 = 0b01010101_01010101_01010101_01010101;
    pub const DISABLE_ACCESS: u32 = 0b11111111_11111111_11111111_11111111;
    // TODO: set to all 1s?

    /// Read the value of the `PKRU` register.
    // #[cfg(test)]
    pub(crate) fn read() -> u32 {
        // ECX must be 0 to prevent a general protection exception (#GP).
        let ecx: u32 = 0;
        let pkru: u32;
        unsafe {
            asm!("rdpkru", in("ecx") ecx, out("eax") pkru, out("edx") _,
                options(nomem, nostack, preserves_flags));
        }
        return pkru;
    }

    /// Write a value to the `PKRU` register.
    pub fn write(pkru: u32) {
        // Both ECX and EDX must be 0 to prevent a general protection exception
        // (#GP).
        let ecx: u32 = 0;
        let edx: u32 = 0;
        unsafe {
            asm!("wrpkru", in("eax") pkru, in("ecx") ecx, in("edx") edx,
                options(nomem, nostack, preserves_flags));
        }
    }

    /// Check the `ECX.PKU` flag (bit 3) of the `07h` `CPUID` leaf; see the
    /// Intel Software Development Manual, vol 3a, section 2.7.
    pub fn has_cpuid_bit_set() -> bool {
        let result = unsafe { std::arch::x86_64::__cpuid(0x07) };
        (result.ecx & 0b100) != 0
    }

    /// Check that the `CR4.PKE` flag (bit 22) is set; see the Intel Software
    /// Development Manual, vol 3a, section 2.7. This register can only be
    /// accessed from privilege level 0.
    #[cfg(test)]
    pub fn has_cr4_bit_set() -> bool {
        let cr4: u64;
        unsafe {
            asm!("mov {}, cr4", out(reg) cr4, options(nomem, nostack, preserves_flags));
        }
        (cr4 & (1 << 22)) != 0
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        #[ignore]
        fn check_read() {
            assert_eq!(read(), DISABLE_ACCESS ^ 1);
            // Accepting all the assumptions of the `check_allocate_and_free`
            // test, we should see that we are only able to access pages marked
            // by the kernel pkey, key 0.
        }

        #[test]
        fn check_roundtrip() {
            let pkru = read();
            // Allow access to pages marked with any key.
            write(0);
            assert_eq!(read(), 0);
            // Restore the original value.
            write(pkru);
            assert_eq!(read(), pkru);
        }
    }
}
