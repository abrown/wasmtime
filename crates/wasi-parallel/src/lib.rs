//! Implement wasi-parallel.
mod context;
mod device;
mod r#impl;
mod witx;

use anyhow::Result;
use context::WasiParallelContext;
use r#impl::{get_exported_memory, get_exported_table_function, parallel_for};
use std::{cell::RefCell, convert::TryInto};
use wasmparser::Parser;
use wasmtime::{Caller, Trap, Val};

/// This struct solely wraps [context::WasiParallelContext] in a `RefCell`.
pub struct WasiParallel {
    pub(crate) ctx: RefCell<WasiParallelContext>,
}

impl WasiParallel {
    pub fn new(sections: Vec<KernelSection>) -> Self {
        Self {
            ctx: RefCell::new(WasiParallelContext::new(sections)),
        }
    }
}

/// Define the ways wasi-parallel can fail.
pub type WasiParallelError = anyhow::Error;

pub use witx::wasi_ephemeral_parallel::add_to_linker;

pub struct KernelSection(u32, Vec<u8>);

/// Find any SPIR-V custom sections. These sections should fulfill the following
/// requirements:
/// - the name must be "wasi-parallel"
/// - the first four bytes should correspond to the table index of the Wasm
///   kernel function
/// - the remaining bytes must be the encoded SPIR-V bytes corresponding to the
///   kernel function code.
pub fn find_custom_spirv_sections(bytes: &[u8]) -> Result<Vec<KernelSection>> {
    let mut found_sections = Vec::new();
    for payload in Parser::new(0).parse_all(bytes) {
        match payload? {
            wasmparser::Payload::CustomSection {
                name: "wasi-parallel",
                data,
                ..
            } => {
                let function_index = u32::from_le_bytes(data[0..4].try_into()?);
                let spirv = data[4..].to_vec();
                found_sections.push(KernelSection(function_index, spirv))
            }
            // Ignore other sections.
            _ => {}
        }
    }
    Ok(found_sections)
}
