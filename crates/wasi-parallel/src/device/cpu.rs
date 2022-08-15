use std::convert::TryInto;

use super::wasm_memory_buffer::WasmMemoryBuffer;
use super::{Buffer, Device};
use crate::context::Kernel;
use crate::witx::types::{BufferAccessKind, DeviceKind};
use anyhow::Result;

pub struct CpuDevice {
    pool: scoped_threadpool::Pool,
}

impl CpuDevice {
    pub fn new() -> Box<dyn Device> {
        let pool = scoped_threadpool::Pool::new(num_cpus::get().try_into().unwrap());
        Box::new(Self { pool })
    }
}

impl Device for CpuDevice {
    fn kind(&self) -> DeviceKind {
        DeviceKind::Cpu
    }

    fn name(&self) -> String {
        "thread pool implementation".into() // TODO retrieve CPU name from system.
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
        self.pool.scoped(|scoped| {
            let module = wasmtime::Module::new(kernel.engine(), kernel.module())
                .expect("unable to compile module");
            for thread_id in 0..num_threads {
                let engine = kernel.engine().clone();
                let module = module.clone();
                let memory = kernel.memory().clone();
                scoped.execute(move || {
                    let mut store = wasmtime::Store::new(&engine, ());
                    let imports = vec![memory.into()];
                    let instance = wasmtime::Instance::new(&mut store, &module, &imports)
                        .expect("failed to construct thread instance");
                    let kernel_fn = instance
                        .get_typed_func::<(i32, i32, i32), (), _>(&mut store, Kernel::NAME)
                        .expect("failed to find kernel function");
                    log::info!("Running thread {}", thread_id);
                    kernel_fn
                        .call(&mut store, (thread_id, num_threads, block_size))
                        .expect("failed to run kernel");
                });
            }
        });
        Ok(())
    }
}
