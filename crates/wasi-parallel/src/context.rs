//! This module implements the state held by wasi-parallel. E.g., when
//! wasi-parallel returns a handle to a device, it must keep a mapping of which
//! device was returned.

use crate::device::{discover, Buffer, Device};
use crate::witx::types::{Buffer, BufferAccessKind, DeviceKind};
use crate::KernelSection;
use anyhow::{anyhow, Result};
use indexmap::IndexMap;
use log::{info, warn};
use rand::Rng;
use std::collections::HashMap;
use wasmtime::Val;

#[derive(Debug)]
pub struct WasiParallelContext {
    pub spirv: HashMap<i32, Vec<u8>>,
    pub devices: IndexMap<i32, Box<dyn Device>>,
    pub buffers: HashMap<i32, Box<dyn Buffer>>,
    pub device_for_buffers: HashMap<i32, i32>,
}

impl WasiParallelContext {
    pub fn new(sections: Vec<KernelSection>) -> Self {
        // Perform some rudimentary device discovery.
        let mut devices = IndexMap::new();
        for device in discover() {
            devices.insert(Self::random_id(), device);
        }

        let mut context = Self {
            spirv: HashMap::new(),
            devices,
            buffers: HashMap::new(),
            device_for_buffers: HashMap::new(),
        };

        // Insert any SPIR-V found as custom sections to the Wasm binary.
        for section in sections {
            context.insert_spirv(section.0 as i32, section.1)
        }

        context
    }

    /// Insert a SPIR-V section for a given function index. Note that this is
    /// only necessary for the current "fat binary" mechanism--this could change
    /// in the future (TODO).
    pub fn insert_spirv(&mut self, index: i32, bytes: Vec<u8>) {
        if self.spirv.contains_key(&index) {
            warn!(
                "context already contains SPIR-V for function index: {}",
                index
            );
        }
        self.spirv.insert(index, bytes);
    }

    /// Retrieve a SPIR-V section for a given function index. Note that this is
    /// only necessary for implementing the current "fat binary" mechanism--this
    /// could change in the future (TODO).
    pub fn get_spirv(&self, index: i32) -> Option<&Vec<u8>> {
        self.spirv.get(&index)
    }

    /// Retrieve a device based on a hint, using the default device if the hint
    /// cannot be satisfied.
    pub fn get_device(&self, hint: DeviceKind) -> Result<i32> {
        match self
            .devices
            .iter()
            .find(|(_, device)| device.kind() == hint)
        {
            // If we can find a device matching the hint, return it...
            Some((&id, _)) => Ok(id),
            // ...otherwise, use the default device.
            None => self.get_default_device(),
        }
    }

    // Retrieve the default device, which currently is the first registered
    // device (TODO).
    pub fn get_default_device(&self) -> Result<i32> {
        match self.devices.iter().next().as_ref() {
            // Use the first available device (TODO: implicit default)...
            Some((&id, _)) => Ok(id),
            // ...or fail if none are available.
            None => Err(anyhow!("no devices available")),
        }
    }

    /// Create a buffer linked to a device.
    pub fn create_buffer(
        &mut self,
        device_id: i32,
        size: i32,
        access: BufferAccessKind,
    ) -> Result<i32> {
        let device = match self.devices.get(&device_id) {
            Some(val) => val,
            None => return Err(anyhow!("unrecognized device")),
        };

        if size < 0 {
            return Err(anyhow!("invalid size (less than 0)"));
        }

        let id = Self::random_id();
        self.buffers
            .insert(id, device.as_ref().create_buffer(size, access));
        self.device_for_buffers.insert(id, device_id);
        Ok(id)
    }

    /// Retrieve a created buffer by its ID.
    pub fn get_buffer(&self, buffer_id: i32) -> Result<&dyn Buffer> {
        match self.buffers.get(&buffer_id) {
            Some(buffer) => Ok(buffer.as_ref()),
            None => Err(anyhow!("invalid buffer ID")),
        }
    }

    /// Retrieve a created buffer by its ID.
    pub fn get_buffer_mut(&mut self, buffer_id: i32) -> Result<&mut dyn Buffer> {
        match self.buffers.get_mut(&buffer_id) {
            Some(buffer) => Ok(buffer.as_mut()),
            None => Err(anyhow!("invalid buffer ID")),
        }
    }

    /// Invoke the `kernel` in parallel on the devices indicated by the input
    /// and output buffers.
    pub fn invoke_parallel_for(
        &mut self,
        kernel: &[u8],
        num_threads: i32,
        block_size: i32,
        in_buffers: &[i32],
        out_buffers: &[i32],
    ) -> Result<()> {
        // Collect the input buffers.
        let mut in_buffers_ = Vec::new();
        for (i, b) in in_buffers.iter().enumerate() {
            match self.buffers.get(b) {
                Some(b) => in_buffers_.push(b),
                None => return Err(anyhow!("in buffer {} has an invalid ID", i)),
            }
        }

        // Collect the output buffers.
        let mut out_buffers_ = Vec::new();
        for (i, b) in out_buffers.iter().enumerate() {
            match self.buffers.get(b) {
                Some(b) => out_buffers_.push(b),
                None => return Err(anyhow!("out buffer {} has an invalid ID", i)),
            }
        }

        // Check that all buffers have the same device.
        let mut device = None;
        for dev in in_buffers
            .iter()
            .chain(out_buffers.iter())
            .map(|b| *self.device_for_buffers.get(b).unwrap())
        {
            if let Some(device) = device {
                if device != dev {
                    return Err(anyhow!("buffers are assigned to different devices"));
                }
            } else {
                device = Some(dev)
            }
        }

        // If no device is found, use the default one.
        let device = device.unwrap_or(self.get_default_device()?);

        // Check that the device is valid.
        if let Some(device) = self.devices.get(&device) {
            info!(
                "Calling invoke_for on {:?} with number of threads = {}, block_size = {}",
                device, num_threads, block_size
            );
            device.invoke_for(
                Kernel(kernel.to_owned()),
                num_threads,
                block_size,
                in_buffers_,
                out_buffers_,
            )?
        } else {
            return Err(anyhow!("invalid device ID"));
        }

        Ok(())
    }

    fn random_id() -> i32 {
        rand::thread_rng().gen()
    }
}

// We pass around a closure here because we want to close over Caller instead of
// threading it throughout the context. The closure must be `Fn` so we can send
// it to a thread pool more than once; the `Send + Sync` are necessary for the
// same reason.
pub(crate) type WasmRunnable = Box<dyn Fn(&[Val]) -> Result<Box<[Val]>> + Send + Sync>;

pub struct Kernel(Vec<u8>);
