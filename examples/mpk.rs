//! This example demonstrates:
//! - how to enable memory protection keys (MPK) in a Wasmtime embedding (see
//!   [`build_engine`])
//! - the expected memory compression from using MPK: it will probe the system
//!   by creating larger and larger memory pools until system memory is
//!   exhausted (see [`probe_engine_size`]). Then, it prints a comparison of the
//!   memory used in both the MPK enabled and MPK disabled configurations.
//!
//! You can execute this example with:
//!
//! ```console
//! $ cargo run --example mpk
//! ```
//!
//! Append `-- --help` for details about the configuring the memory size of the
//! pool. Also, to inspect interesting configuration values used for
//! constructing the pool, turn on logging:
//!
//! ```console
//! $ RUST_LOG=debug cargo run --example mpk -- --static-memory-maximum-size 512MiB
//! ```

use anyhow::{anyhow, Result};
use bytesize::ByteSize;
use clap::Parser;
use log::{info, warn};
use std::str::FromStr;
use wasmtime::*;

fn main() -> Result<()> {
    drop(env_logger::try_init());
    let args = Args::parse();
    println!("args: {:?}", args);

    let without_mpk = probe_engine_size(&args, MpkEnabled::Disable)?;
    println!("without MPK: {:?}", without_mpk);

    let with_mpk = probe_engine_size(&args, MpkEnabled::Enable)?;
    println!("with MPK: {:?}", with_mpk);

    Ok(())
}

#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The maximum number of bytes for each WebAssembly linear memory in the
    /// pool.
    #[arg(long, default_value = "128MiB", value_parser = parse_byte_size)]
    memory_size: u64,

    /// The maximum number of bytes a memory is considered static; see
    /// `Config::static_memory_maximum_size` for more details and the default
    /// value if unset.
    #[arg(long, value_parser = parse_byte_size)]
    static_memory_maximum_size: Option<u64>,

    /// The size in bytes of the guard region to expect between static memory
    /// slots; see [`Config::static_memory_guard_size`] for more details and the
    /// default value if unset.
    #[arg(long, value_parser = parse_byte_size)]
    static_memory_guard_size: Option<u64>,
}

/// Parse a human-readable byte size--e.g., "512 MiB"--into the correct number
/// of bytes.
fn parse_byte_size(value: &str) -> Result<u64> {
    let size = ByteSize::from_str(value).map_err(|e| anyhow!(e))?;
    Ok(size.as_u64())
}

/// Find the engine with the largest number of memories we can create on this
/// machine.
fn probe_engine_size(args: &Args, mpk: MpkEnabled) -> Result<Pool> {
    let mut search = ExponentialSearch::new();
    let mut mapped_bytes = 0;
    while !search.done() {
        match build_engine(&args, search.next(), mpk) {
            Ok(rb) => {
                // TODO: assert!(rb >= mapped_bytes);
                mapped_bytes = rb;
                search.record(true)
            }
            Err(e) => {
                warn!("failed engine allocation, continuing search: {:?}", e);
                search.record(false)
            }
        }
    }
    Ok(Pool {
        num_memories: search.next(),
        mapped_bytes,
    })
}

#[derive(Debug)]
#[allow(dead_code)]
struct Pool {
    num_memories: u32,
    mapped_bytes: usize,
}

/// Exponentially increase the `next` value until the attempts fail, then
/// perform a binary search to find the maximum attempted value that still
/// succeeds.
#[derive(Debug)]
struct ExponentialSearch {
    /// Determines if we are in the growth phase.
    growing: bool,
    /// The last successful value tried; this is the algorithm's lower bound.
    last: u32,
    /// The next value to try; this is the algorithm's upper bound.
    next: u32,
}
impl ExponentialSearch {
    fn new() -> Self {
        Self {
            growing: true,
            last: 0,
            next: 1,
        }
    }
    fn next(&self) -> u32 {
        self.next
    }
    fn record(&mut self, success: bool) {
        if !success {
            self.growing = false
        }
        let diff = if self.growing {
            (self.next - self.last) * 2
        } else {
            (self.next - self.last + 1) / 2
        };
        if success {
            self.last = self.next;
            self.next = self.next + diff;
        } else {
            self.next = self.next - diff;
        }
    }
    fn done(&self) -> bool {
        self.last == self.next
    }
}

/// Build a pool-allocated engine with `num_memories` slots.
fn build_engine(args: &Args, num_memories: u32, enable_mpk: MpkEnabled) -> Result<usize> {
    // Configure the memory pool.
    let mut pool = PoolingAllocationConfig::default();
    let memory_pages = args.memory_size / u64::from(wasmtime_environ::WASM_PAGE_SIZE);
    pool.memory_pages(memory_pages);
    pool.total_memories(num_memories)
        .memory_protection_keys(enable_mpk);

    // Configure the engine itself.
    let mut config = Config::new();
    if let Some(static_memory_maximum_size) = args.static_memory_maximum_size {
        config.static_memory_maximum_size(static_memory_maximum_size);
    }
    if let Some(static_memory_guard_size) = args.static_memory_guard_size {
        config.static_memory_guard_size(static_memory_guard_size);
    }
    config.allocation_strategy(InstanceAllocationStrategy::Pooling(pool));

    // Measure memory use before and after the engine is built.
    let mapped_bytes_before = num_bytes_mapped()?;
    let engine = Engine::new(&config)?;
    let mapped_bytes_after = num_bytes_mapped()?;

    // Ensure we actually use the engine somehow.
    engine.increment_epoch();

    let mapped_bytes = mapped_bytes_after - mapped_bytes_before;
    info!(
        "{}-slot pool ({:?}): {} bytes mapped",
        num_memories, enable_mpk, mapped_bytes
    );
    Ok(mapped_bytes)
}

/// Add up the sizes of all the mapped virtual memory regions for the current
/// process. On Linux, this is parsed from `/proc/PID/maps` ([reference]).
///
/// [reference]: https://github.com/rbspy/proc-maps/blob/93cdede6e626c272badfe51f0062f87befb0c024/src/linux_maps.rs#L52
fn num_bytes_mapped() -> Result<usize> {
    let pid = std::process::id().try_into().unwrap();
    let maps = proc_maps::get_process_maps(pid)?;
    let total = maps.iter().map(|m| m.size()).sum();
    Ok(total)
}
