use super::wasm_memory_buffer::WasmMemoryBuffer;
use super::{Buffer, Device};
use crate::context::Kernel;
use crate::witx::types::{BufferAccessKind, DeviceKind};
use anyhow::Result;
use log::info;
use scoped_threadpool::Pool;
use wasmtime::Val;

pub struct CpuDevice;

impl CpuDevice {
    pub fn new() -> Box<dyn Device> {
        Box::new(Self)
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

    fn invoke_for(
        &self,
        kernel: Kernel,
        num_threads: i32,
        block_size: i32,
        in_buffers: Vec<&Box<dyn Buffer>>,
        out_buffers: Vec<&Box<dyn Buffer>>,
    ) -> Result<()> {
        let kernel_func = &kernel.wasm;
        let mut params: Vec<Val> = vec![num_threads.into(), block_size.into()];
        for buffer in in_buffers.iter().chain(out_buffers.iter()) {
            let buffer = buffer.as_any().downcast_ref::<WasmMemoryBuffer>().expect(
                "Unable to downcast to a WasmMemoryBuffer: this error means that the \
                WasiParallelContext has failed to filter out non-CPU buffers.",
            );
            let offset = buffer
                .offset
                .expect("Offset not set on buffer: has write_buffer been called on it?");
            params.push((offset as i32).into());
            params.push((buffer.len() as i32).into());
        }
        let params = &params[..];

        // Setup the thread pool using the same number of threads as CPUs. Note
        // that we use the scoped_threadpool here to avoid lifetime issues:
        // `threadpool`'s `pool::execute` method forces the passed closure to be
        // `'static` but both `Caller` and `Func` above cannot match that
        // lifetime.
        let mut pool = Pool::new(num_cpus::get() as u32);
        pool.scoped(|scoped| {
            for thread_id in 0..num_threads {
                scoped.execute(move || {
                    info!("Running thread {}", thread_id);
                    let mut thread_params = params.to_vec();
                    thread_params.insert(0, thread_id.into());
                    kernel_func(&thread_params).expect("kernel failed");
                });
            }
        });

        Ok(())
    }
}
