use super::{wasm_memory_buffer::WasmMemoryBuffer, Buffer, Device};
use crate::context::Kernel;
use crate::witx::types::{BufferAccessKind, DeviceKind};
use anyhow::{Context, Result};
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

    fn parallelize(
        &mut self,
        kernel: Kernel,
        num_threads: i32,
        block_size: i32,
        _in_buffers: Vec<&Box<dyn Buffer>>,  // TODO
        _out_buffers: Vec<&Box<dyn Buffer>>, // TODO
    ) -> Result<()> {
        // JIT-compile and instantiate the parallel kernel.
        let module = wasmtime::Module::new(kernel.engine(), kernel.module())
            .context("unable to compile kernel module")?;
        let mut store = wasmtime::Store::new(kernel.engine(), ());
        let imports = vec![kernel.memory().clone().into()];
        let instance = wasmtime::Instance::new(&mut store, &module, &imports)
            .context("failed to construct kernel instance")?;
        let kernel_fn = instance
            .get_typed_func::<(i32, i32, i32), (), _>(&mut store, Kernel::NAME)
            .expect("failed to find the kernel function");

        // Run each iteration of the parallel kernel sequentially.
        for thread_id in 0..num_threads {
            info!("Running thread {}", thread_id);
            kernel_fn
                .call(&mut store, (thread_id, num_threads, block_size))
                .context("failed to run kernel")?;
        }

        Ok(())
    }
}
