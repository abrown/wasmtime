extern crate ocl_core as core;

use std::{
    any::Any,
    borrow::{Borrow, BorrowMut},
    collections::{hash_map::DefaultHasher, HashMap},
    ffi::CString,
    hash::{Hash, Hasher},
    sync::{Arc, Mutex},
};

use crate::witx::types::{BufferAccessKind, DeviceKind};
use anyhow::{anyhow, Context, Error, Result};
use ocl_core::{ArgVal, DeviceType, MemFlags};
use wiggle::GuestPtr;

pub fn discover() -> Vec<Box<dyn super::Device>> {
    let mut devices = vec![];

    for plat in core::get_platform_ids().unwrap_or(vec![]) {
        for dev in core::get_device_ids(plat, None, None).unwrap_or(vec![]) {
            if let Ok(device) = Device::new(dev) {
                devices.push(device)
            }
        }
    }
    devices
}

struct OpenClContext {
    pub context: core::Context,
    pub device: core::DeviceId,
    pub queue: core::CommandQueue,
}

struct Device {
    context: Arc<OpenClContext>,
    program_cache: Mutex<HashMap<u64, core::Program>>,
}

impl Device {
    pub fn new(device: core::DeviceId) -> Result<Box<dyn super::Device>, core::Error> {
        let ctx_props = core::ContextProperties::new();
        let device_ids = [device];
        let context = core::create_context(Some(&ctx_props), &device_ids, None, None)?;

        let queue_props = core::CommandQueueProperties::new();
        let queue = core::create_command_queue(&context, device, Some(queue_props))?;

        Ok(Box::new(Self {
            context: Arc::new(OpenClContext {
                context,
                device,
                queue,
            }),
            program_cache: Mutex::new(HashMap::new()),
        }))
    }
}

impl super::Device for Device {
    fn kind(&self) -> DeviceKind {
        // If we don't know (or there was an error getting the device type) we default to DGPU
        let mut device_kind = DeviceKind::DiscreteGpu;
        if let Ok(cl_type_info) = core::get_device_info(self.context.device, core::DeviceInfo::Type)
        {
            if let core::DeviceInfoResult::Type(cl_type) = cl_type_info {
                if cl_type.contains(DeviceType::CPU) {
                    device_kind = DeviceKind::Cpu
                } else if cl_type.contains(DeviceType::GPU) {
                    // TODO: Figure out how to differentiate between dedicated and integrated GPUs
                    device_kind = DeviceKind::DiscreteGpu
                }
            }
        }
        device_kind
    }

    fn name(&self) -> String {
        let name_info_res = core::get_device_info(self.context.device, core::DeviceInfo::Name);

        match name_info_res {
            Ok(name_info) => name_info.to_string(),
            Err(_) => format!("OpenCL Device {:?}", self.context.device),
        }
    }

    fn create_buffer(&self, size: i32, access: BufferAccessKind) -> Box<dyn super::Buffer> {
        Buffer::new(self.context.clone(), size, access)
    }

    fn invoke_for(
        &self,
        kernel: super::Kernel,
        num_threads: i32,
        block_size: i32,
        in_buffers: Vec<&Box<dyn super::Buffer>>,
        out_buffers: Vec<&Box<dyn super::Buffer>>,
    ) -> anyhow::Result<()> {
        let spirv = &kernel
            .spirv
            .context("SPIR-V not found in kernel function")?;

        let devices = [self.context.device.clone()];

        let mut hasher = DefaultHasher::new();
        spirv.hash(&mut hasher);
        let spirv_hash = hasher.finish();

        let kernel;
        {
            // TODO: evaluate using an LRU instead of a HashMap
            let mut cache = self
                .program_cache
                .lock()
                .map_err(|e| Error::msg(e.to_string()))?;
            let program = cache.entry(spirv_hash).or_insert_with(|| {
                let p = core::create_program_with_il(&self.context.context, &spirv, None).unwrap();

                let options = CString::new("").expect("CString::new failed");

                core::build_program(&p, Some(&devices), &options, None, None).unwrap();
                p
            });

            // TODO: we might need a better way to determine the entrypoint name
            kernel = core::create_kernel(&program, "spir_main").map_err(Error::msg)?;
        }
        let mut index = 0;

        // TODO: figure out a better way pass arguments to kernel
        for buf in in_buffers {
            let buf = buf
                .as_any()
                .downcast_ref::<Buffer>()
                .context(format!("In buffer {} is not owned by this device", index))?;
            core::set_kernel_arg(&kernel, index, ArgVal::mem(buf.mem()?.borrow()))
                .map_err(Error::msg)?;
            index += 1;
        }

        for buf in out_buffers {
            let buf = buf
                .as_any()
                .downcast_ref::<Buffer>()
                .context(format!("Out buffer {} is not owned by this device", index))?;
            core::set_kernel_arg(&kernel, index, ArgVal::mem(buf.mem()?.borrow()))
                .map_err(Error::msg)?;
            index += 1;
        }

        unsafe {
            core::enqueue_kernel(
                &self.context.queue,
                &kernel,
                1,
                None,
                &[(block_size * num_threads) as usize, 0, 0],
                Some([num_threads as usize, 0, 0]),
                None::<core::Event>,
                None::<&mut core::Event>,
            )
            .map_err(Error::msg)?
        }

        // TODO: we should be able to use events to wait only when we need to
        core::finish(&self.context.queue).map_err(Error::msg)?;

        Ok(())
    }
}
struct Buffer {
    context: Arc<OpenClContext>,
    len: u32,
    access: BufferAccessKind,
    // Using a mutex + options because we only create buffers when we need
    // them/have enough information. Using Arc because we can't pass data
    // behind the mutex outside the mutex.
    memory: Mutex<Option<Arc<core::Mem>>>,
}

impl Buffer {
    pub fn new(
        context: Arc<OpenClContext>,
        len: i32,
        access: BufferAccessKind,
    ) -> Box<dyn super::Buffer> {
        Box::new(Self {
            len: len as u32,
            access,
            context,
            memory: Mutex::new(None),
        })
    }

    pub fn mem(&self) -> Result<Arc<core::Mem>> {
        let mut guard = self.memory.lock().map_err(|e| Error::msg(e.to_string()))?;

        match &*guard {
            Some(mem) => Ok(mem.clone()),
            None => {
                let memory = self.create_memory(None)?;
                guard.borrow_mut().replace(memory.clone());
                Ok(memory)
            }
        }
    }

    fn create_memory(&self, data: Option<&[u8]>) -> Result<Arc<core::Mem>> {
        use BufferAccessKind::*;

        Ok(Arc::new(unsafe {
            core::create_buffer(
                &self.context.context,
                match self.access {
                    Write => MemFlags::WRITE_ONLY,
                    Read => MemFlags::READ_ONLY | MemFlags::COPY_HOST_PTR,
                    ReadWrite => MemFlags::READ_WRITE | MemFlags::COPY_HOST_PTR,
                },
                self.len as usize,
                data,
            )
            .map_err(Error::msg)?
        }))
    }
}

impl super::Buffer for Buffer {
    fn len(&self) -> u32 {
        self.len
    }

    fn access(&self) -> BufferAccessKind {
        self.access
    }

    fn write(&mut self, source: GuestPtr<[u8]>) -> anyhow::Result<()> {
        let mut guard = self.memory.lock().map_err(|e| Error::msg(e.to_string()))?;
        let source = &source.as_slice()?;
        match &*guard {
            Some(mem) => unsafe {
                // TODO: this doesn't need to block
                core::enqueue_write_buffer(
                    &self.context.queue,
                    mem.as_ref(),
                    true,
                    0,
                    source,
                    None::<core::Event>,
                    None::<&mut core::Event>,
                )
                .map_err(Error::msg)?;
            },
            None => {
                let memory = self.create_memory(Some(source))?;
                guard.borrow_mut().replace(memory.clone());
            }
        };
        Ok(())
    }

    fn read(&self, destination: GuestPtr<[u8]>) -> anyhow::Result<()> {
        let guard = self.memory.lock().map_err(|e| Error::msg(e.to_string()))?;
        match &*guard {
            Some(mem) => unsafe {
                core::enqueue_read_buffer(
                    &self.context.queue,
                    mem.as_ref(),
                    true,
                    0,
                    &mut destination.as_slice_mut()?,
                    None::<core::Event>,
                    None::<&mut core::Event>,
                )
                .map_err(Error::msg)?;
                Ok(())
            },
            None => Err(anyhow!("Write buffer has not been written to.")),
        }
    }

    fn as_any(&self) -> &dyn Any {
        self as &dyn Any
    }
}
