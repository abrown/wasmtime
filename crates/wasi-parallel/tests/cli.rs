use anyhow::Result;
use clap::Parser;
use wasmtime_cli::commands::RunCommand;

// Run the Wasm file through the Wasmtime CLI; this is the closest we can get
// to running `wasmtime --wasi-modules experimental-wasi-parallel` without
// having the binary available.
fn cli(file: &str) -> Result<()> {
    let command = RunCommand::parse_from(&[
        "run",
        "--wasi-modules",
        "experimental-wasi-parallel",
        "--wasm-features",
        "threads",
        file,
    ]);
    command.execute()
    // TODO capture output and check that the last line printed is "0".
}

#[test]
fn run_parallel_for() {
    cli("tests/wat/parallel-for.wat").unwrap();
}

#[test]
fn run_buffer() {
    cli("tests/wasm/buffer.wasm").unwrap();
}

#[cfg(feature = "opencl")]
#[test]
fn run_fill() {
    cli("tests/wasm/fill.wasm").unwrap();
}
