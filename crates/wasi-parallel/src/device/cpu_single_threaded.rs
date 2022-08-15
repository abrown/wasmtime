use super::{wasm_memory_buffer::WasmMemoryBuffer, Buffer, Device};
use crate::context::Kernel;
use crate::witx::types::{BufferAccessKind, DeviceKind};
use anyhow::Result;
use log::info;

pub struct CpuSingleThreadedDevice;

impl CpuSingleThreadedDevice {
    pub fn new() -> Box<dyn Device> {
        Box::new(Self)
    }
}

impl Device for CpuSingleThreadedDevice {
    fn kind(&self) -> DeviceKind {
        DeviceKind::Cpu
    }

    fn name(&self) -> String {
        "single-threaded implementation".into() // TODO retrieve CPU name from system.
    }

    fn create_buffer(&self, size: i32, access: BufferAccessKind) -> Box<dyn Buffer> {
        Box::new(WasmMemoryBuffer::new(size as u32, access))
    }

    fn invoke_for(
        &self,
        kernel: Kernel,
        num_threads: i32,
        block_size: i32,
        _in_buffers: Vec<&Box<dyn Buffer>>,
        _out_buffers: Vec<&Box<dyn Buffer>>,
    ) -> Result<()> {
        let kernel_func = &kernel.wasm;
        for thread_id in 0..num_threads {
            info!("Running thread {}", thread_id);
            kernel_func(&[thread_id.into(), num_threads.into(), block_size.into()]).expect("TODO");
        }
        Ok(())
    }
}
