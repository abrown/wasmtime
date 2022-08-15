pub mod cpu;
pub mod cpu_single_threaded;
pub mod wasm_memory_buffer;

#[cfg(feature = "opencl")]
mod opencl;

use crate::witx::types::{BufferAccessKind, DeviceKind};
use crate::{
    context::Kernel,
    device::{cpu::CpuDevice, cpu_single_threaded::CpuSingleThreadedDevice},
};
use anyhow::Result;
use std::{any::Any, fmt::Debug};
use wiggle::GuestPtr;

/// Discover available devices.
pub fn discover() -> Vec<Box<dyn Device>> {
    #[allow(unused_mut)]
    let mut devices = vec![CpuDevice::new(), CpuSingleThreadedDevice::new()];
    #[cfg(feature = "opencl")]
    devices.append(&mut opencl::discover());
    devices
}

/// Define the operations possible on a device.
pub trait Device: Send + Sync {
    /// Return the device kind.
    fn kind(&self) -> DeviceKind;

    /// Return the device name.
    fn name(&self) -> String;

    /// Create a buffer associated with this device. The created buffer is held
    /// in `WasiParallelContext`, which must guarantee that buffers are sent to
    /// the correct devices.
    fn create_buffer(&self, size: i32, access: BufferAccessKind) -> Box<dyn Buffer>;

    /// Invoke a parallel for on the device.
    fn invoke_for(
        &self,
        kernel: Kernel,
        num_threads: i32,
        block_size: i32,
        in_buffers: Vec<&Box<dyn Buffer>>,
        out_buffers: Vec<&Box<dyn Buffer>>,
    ) -> Result<()>;
}

impl Debug for dyn Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.kind(), self.name())
    }
}

/// Define the operations possible on a buffer.
pub trait Buffer: Send + Sync {
    /// Returns the size of the buffer in bytes.
    fn len(&self) -> u32;

    /// Describes how the buffer can be used by the device.
    fn access(&self) -> BufferAccessKind;

    /// Write the given slice of Wasm memory into the buffer.
    fn write(&mut self, data: GuestPtr<[u8]>) -> Result<()>;

    /// Read the buffer into a slice.
    fn read(&self, slice: GuestPtr<[u8]>) -> Result<()>;

    /// Allow for downcasting the buffer: `buffer.as_any().downcast_ref::<...>()`.
    fn as_any(&self) -> &dyn Any;
}

impl Debug for dyn Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Buffer: size: {} access: {:?}",
            self.len(),
            self.access()
        )
    }
}
