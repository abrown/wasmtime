//! This module translates the calls from the Wasm program (running inside a
//! Wasm engine) into something usable by [WasiParallelContext] (running inside
//! the Wasm runtime). Wiggle performs most of the conversions using
//! `from_witx!` in `witx.rs` but the special nature of `parallel_for` (it can
//! call back into a function in the Wasm module) involves a manual
//! implementation of this glue code.
use std::convert::TryInto;

use crate::context::{Kernel, WasiParallelContext, WasmRunnable};
use crate::witx::types::{
    Buffer, BufferAccessKind, BufferData, BufferSize, DeviceKind, Function, ParallelDevice,
};
use crate::witx::wasi_ephemeral_parallel::WasiEphemeralParallel;
use crate::{WasiParallel, WasiParallelError};
use wasmtime::{Caller, Extern, Func, Memory, Trap, Val};

impl WasiEphemeralParallel for WasiParallel {
    fn get_device(&mut self, hint: DeviceKind) -> Result<ParallelDevice, WasiParallelError> {
        let id = self.ctx.borrow().get_device(hint)?;
        Ok(ParallelDevice::from(id))
    }

    fn create_buffer(
        &mut self,
        device: ParallelDevice,
        size: BufferSize,
        kind: BufferAccessKind,
    ) -> Result<Buffer, super::WasiParallelError> {
        let id = self
            .ctx
            .borrow_mut()
            .create_buffer(device.into(), size as i32, kind)?;
        Ok(Buffer::from(id))
        // TODO change type to `u32`: parameters `device_d` and `size`, returned
        // buffer ID
    }

    fn write_buffer<'a>(
        &mut self,
        data: &BufferData<'a>,
        buffer: Buffer,
    ) -> Result<(), super::WasiParallelError> {
        let mut ctx = self.ctx.borrow_mut();
        let buffer = ctx.get_buffer_mut(buffer.into())?;
        buffer.write(*data)?;
        Ok(())
    }

    fn read_buffer<'a>(
        &mut self,
        buffer: Buffer,
        data: &BufferData<'a>,
    ) -> Result<(), super::WasiParallelError> {
        let ctx = self.ctx.borrow_mut();
        let buffer = ctx.get_buffer(buffer.into())?;
        buffer.read(*data)
    }

    fn parallel_for<'a>(
        &mut self,
        worker: &Function<'a>,
        num_threads: u32,
        block_size: u32,
        in_buffers: &wiggle::GuestPtr<'a, [Buffer]>,
        out_buffers: &wiggle::GuestPtr<'a, [Buffer]>,
    ) -> Result<(), super::WasiParallelError> {
        let ctx = self.ctx.borrow_mut();
        // TODO this seems a bit inefficient...
        let in_buffers: Vec<i32> = in_buffers.as_slice()?.iter().map(|&b| b.into()).collect();
        let out_buffers: Vec<i32> = out_buffers.as_slice()?.iter().map(|&b| b.into()).collect();
        ctx.invoke_parallel_for(
            &worker.as_slice()?,
            num_threads.try_into()?,
            block_size.try_into()?,
            &in_buffers,
            &out_buffers,
        )
    }

    // `parallel_for` is implemented below and manually linked in `lib.rs`.
}

/// Retrieve the exported `"memory"` from the `Caller`; this is a helper
/// implementation for simplifying the `parallel_for` closure. Usually this is
/// auto-generated inline by `wiggle_generate::wasmtime::generate_func`.
pub(crate) fn get_exported_memory<T>(caller: &mut Caller<T>) -> Result<Memory, Trap> {
    match caller.get_export("memory") {
        Some(Extern::Memory(m)) => Ok(m),
        _ => Err(Trap::new("missing required memory export")),
    }
}

/// Retrieve a function from the exported `"table"`; this is a helper
/// implementation for simplifying the `parallel_for` closure.
pub(crate) fn get_exported_table_function<T>(
    caller: &mut Caller<T>,
    function_index: u32,
) -> Result<Func, Trap> {
    let table = match caller.get_export("__indirect_function_table") {
        Some(Extern::Table(t)) => t,
        _ => {
            return Err(Trap::new(
                "wasi-parallel requires a '__indirect_function_table' export",
            ))
        }
    };

    match table.get(caller, function_index) {
        Some(Val::FuncRef(Some(f))) => Ok(f),
        _ => {
            return Err(Trap::new(
                "the '__indirect_function_table' export does not contain a funcref at the given index",
            ));
        }
    }
}
