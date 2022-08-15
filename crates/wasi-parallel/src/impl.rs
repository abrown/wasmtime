//! This module translates the calls from the Wasm program (running inside a
//! Wasm engine) into something usable by [WasiParallelContext] (running inside
//! the Wasm runtime). Wiggle performs most of the conversions using
//! `from_witx!` in `witx.rs` but the special nature of `parallel_for` (it can
//! call back into a function in the Wasm module) involves a manual
//! implementation of this glue code.
use crate::witx::types::{
    Buffer, BufferAccessKind, BufferData, BufferSize, DeviceKind, ParallelDevice,
};
use crate::witx::wasi_ephemeral_parallel::WasiEphemeralParallel;
use crate::{WasiParallel, WasiParallelError};

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

    // Note: `parallel_for` is manually linked in `lib.rs`.
}
