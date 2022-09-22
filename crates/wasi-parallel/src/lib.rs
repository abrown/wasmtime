//! Implement [`wasi-parallel`].
//!
//! [`wasi-parallel`]: https://github.com/WebAssembly/wasi-parallel

mod context;
mod device;
mod r#impl;
mod witx;

use anyhow::Result;
use context::WasiParallelContext;
use std::cell::RefCell;
use wasmtime::{Caller, Extern, SharedMemory, Trap};
use wiggle::GuestError;

/// This struct solely wraps [context::WasiParallelContext] in a `RefCell`.
pub struct WasiParallel {
    pub(crate) ctx: RefCell<WasiParallelContext>,
}

impl WasiParallel {
    pub fn new() -> Self {
        Self {
            ctx: RefCell::new(WasiParallelContext::new()),
        }
    }
}

/// Define the ways wasi-parallel can fail.
pub type WasiParallelError = anyhow::Error;

/// Re-export the Wiggle-generated `add_to_linker` function. Because
/// `WasiParallelContext` needs access to `Caller` (i.e., to retrieve the
/// `Engine`) and the wiggle infrastructure does not support this, this API call
/// is skipped during wiggle generation (see `skip` in `witx.rs`) and manually
/// implemented here.
pub fn add_to_linker<T>(
    linker: &mut wasmtime::Linker<T>,
    get_cx: impl Fn(&mut T) -> &mut WasiParallel + Send + Sync + Copy + 'static,
) -> anyhow::Result<()> {
    witx::wasi_ephemeral_parallel::add_to_linker(linker, get_cx)?;

    // At one time, this code was auto-generated by
    // `wiggle_generate::wasmtime::generate_func`.
    linker.func_wrap(
        "wasi_ephemeral_parallel",
        "parallel_exec",
        move |mut caller: Caller<'_, T>,
              arg0: i32,
              arg1: i32,
              arg2: i32,
              arg3: i32,
              arg4: i32,
              arg5: i32,
              arg6: i32,
              arg7: i32,
              arg8: i32|
              -> Result<i32, Trap> {
            // A module using wasi-parallel must export a shared memory named
            // 'memory'. This is what is passed to Wiggle.
            let shared_memory = get_exported_shared_memory(&mut caller)?;
            let mem = unsafe {
                std::slice::from_raw_parts_mut(
                    shared_memory.data() as *mut u8,
                    shared_memory.data_size(),
                )
            };
            let mem = wiggle::wasmtime::WasmtimeGuestMemory::new(mem);

            // Parse the arguments using wiggle. Ideally this would happen
            // directly in wiggle-generated code but our need for `Caller`
            // forces us to do it here. Changes from the auto-generated
            // are marked with TODOs.
            let device_id = arg0;
            let kernel = wiggle::GuestPtr::<[u8]>::new(&mem, (arg1 as u32, arg2 as u32));
            let num_threads = arg3; // TODO: as u32
            let block_size = arg4; // TODO: as u32
            let in_buffers = wiggle::GuestPtr::<[i32]>::new(&mem, (arg5 as u32, arg6 as u32)); // TODO ...<[Buffer]>
            let out_buffers = wiggle::GuestPtr::<[i32]>::new(&mem, (arg7 as u32, arg8 as u32)); // TODO: ...<[Buffer]>

            // To properly compile the function we need to use the same engine
            // as the caller.
            let engine = &caller.engine().clone();

            // Call the wasi-parallel context with all the right Rust types.
            let outer_ctx = get_cx(caller.data_mut());
            let mut inner_ctx = outer_ctx.ctx.borrow_mut();
            let kernel = &*kernel.as_slice().map_err(stringify_wiggle_err)?;
            let in_buffers = &*in_buffers.as_slice().map_err(stringify_wiggle_err)?;
            let out_buffers = &*out_buffers.as_slice().map_err(stringify_wiggle_err)?;
            let result = inner_ctx.invoke_parallel_for(
                device_id,
                kernel,
                engine,
                shared_memory,
                num_threads,
                block_size,
                in_buffers,
                out_buffers,
            );
            match result {
                Ok(_) => Ok(0), // TODO: <ParErrno as wiggle::GuestErrorType>::success() as i32
                Err(e) => Err(Trap::new(e.to_string())),
            }
        },
    )?;

    Ok(())
}

/// Retrieve the exported `"memory"` from the `Caller`; this is a helper
/// implementation for simplifying the `parallel_for` closure. Usually this is
/// auto-generated inline by `wiggle_generate::wasmtime::generate_func`.
pub(crate) fn get_exported_shared_memory<T>(caller: &mut Caller<T>) -> Result<SharedMemory, Trap> {
    match caller.get_export("memory") {
        Some(Extern::SharedMemory(m)) => Ok(m),
        _ => Err(Trap::new("missing required shared memory export: 'memory'")),
    }
}

/// Construct a Wasmtime [`Trap`] from a Wiggle [`GuestError`]; this uses the
/// same logic as `From<GuestError> for wiggle::Trap` but with one less
/// conversion.
fn stringify_wiggle_err(err: GuestError) -> Trap {
    Trap::new(err.to_string())
}
