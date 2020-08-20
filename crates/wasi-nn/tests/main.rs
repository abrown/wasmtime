use anyhow::Result;
use std::fs::File;
use std::path::{Path, PathBuf};
use wasmtime::{Linker, Module, Store};
use wasmtime_wasi::{Wasi, WasiCtxBuilder};
use wasmtime_wasi_nn::{WasiNn, WasiNnCtx};

/// Execute the Wasm `file` with the given `preopen` using both the wasi and wasi-nn APIs.
fn execute_wasi_nn_program(preopen: impl AsRef<Path>, file: impl AsRef<Path>) -> Result<()> {
    println!("Executing: {}", file.as_ref().display());
    let store = Store::default();
    let mut linker = Linker::new(&store);

    // Build a Wasi context that pre-opens the `tests` directory.
    let mut cx = WasiCtxBuilder::new();
    cx.inherit_stdio().inherit_env();
    cx.preopened_dir(File::open(preopen).unwrap(), ".");
    let cx = cx.build()?;

    // Create an instance of `Wasi` which contains a `WasiCtx`. Note that
    // `WasiCtx` provides a number of ways to configure what the target program
    // will have access to.
    let wasi = Wasi::new(&store, cx);
    wasi.add_to_linker(&mut linker)?;

    // Do the same for a `WasiNn` instance.
    let wasi_nn = WasiNn::new(&store, WasiNnCtx::new()?);
    wasi_nn.add_to_linker(&mut linker)?;
    println!("load: {:?}", wasi_nn.get_export("load").unwrap().ty());

    linker.func("env", "foo", |x: u32| x * 2)?;
    linker.func("wasi_nn", "bar", |x: u32| x * 3)?;

    // Instantiate our module with the imports we've created, and run it.
    let module = Module::from_file(store.engine(), file)?;
    linker.module("", &module)?;
    linker.get_default("")?.get0::<()>()?()?;

    Ok(())
}

#[test]
fn simple() {
    assert_eq!(1, 1);
}

// See build.rs:
include!(concat!(env!("OUT_DIR"), "/tests.rs"));
