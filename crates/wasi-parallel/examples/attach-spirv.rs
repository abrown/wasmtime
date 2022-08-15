//! Add an example application that attaches a wasi-parallel custom section to an existing Wasm or
//! WAT file. Usage:
//! ```ignore
//! cargo run --example attach-spirv -- \
//!   --input tests/wasm/buffer.wat \
//!   --id 42 \
//!   --data tests/spirv/nstream32.spv \
//!   --output /tmp/test-wasi-parallel.wasm
//! ```
//! See the `cli.rs` tests also.

use anyhow::Result;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::PathBuf;
use structopt::{clap::AppSettings, StructOpt};
use wasm_encoder::{CustomSection, Section};

/// Attach a custom section to a Wasm module.
#[derive(StructOpt)]
#[structopt(
    name = "attach-spirv",
    version = env!("CARGO_PKG_VERSION"),
    global_settings = &[AppSettings::ColoredHelp],
)]
pub struct AttachFlags {
    /// The path of the WebAssembly module to append to.
    #[structopt(short, long, required = true, value_name = "PATH")]
    pub input: PathBuf,

    /// The path at which to save the modified WebAssembly.
    #[structopt(short, long, required = true, value_name = "PATH")]
    pub output: PathBuf,

    /// The name to give the custom section.
    #[structopt(short, long, default_value = "wasi-parallel")]
    pub name: String,

    /// If provided, prepend the given ID as a `u32` in front of the custom section data.
    #[structopt(long)]
    pub id: Option<u32>,

    /// If provided, retrieve the data to append into the custom section from this file; if not
    /// provided, the application will use stdin.
    #[structopt(short, long, value_name = "PATH")]
    pub data: Option<PathBuf>,
}

fn main() -> Result<()> {
    let flags = AttachFlags::from_iter_safe(std::env::args()).unwrap_or_else(|e| e.exit());

    // Get the current module in Wasm-encoded form (not WAT).
    let mut wasm_bytes = wat::parse_bytes(&fs::read(&flags.input)?)?.to_vec();

    // Construct the new custom section.
    let mut new_section_bytes = flags.id.map_or(Vec::new(), |id| id.to_le_bytes().to_vec());
    if let Some(path) = flags.data {
        File::open(path)?.read_to_end(&mut new_section_bytes)?;
    } else {
        io::stdin().read_to_end(&mut new_section_bytes)?;
    };
    let custom_section = CustomSection {
        name: &flags.name,
        data: &new_section_bytes,
    };

    // Append the new custom section at the end. Note that the encoded Wasm module accepts a custom
    // section pretty much anywhere, unlike other sections:
    // https://webassembly.github.io/spec/core/binary/modules.html#binary-module.
    wasm_bytes.push(custom_section.id());
    custom_section.encode(&mut wasm_bytes);

    // Write out the modified module.
    File::create(flags.output)?.write_all(&wasm_bytes)?;

    Ok(())
}
