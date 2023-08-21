use anyhow::Result;
use std::{path::Path, path::PathBuf, process::Command};
use wasmtime::{Engine, Linker, Module, Store};
use wasmtime_wasi::{ambient_authority, Dir, WasiCtx, WasiCtxBuilder};
use wasmtime_wasi_nn::{backend, InMemoryRegistry, WasiNnCtx};

#[test]
fn image_classification() -> Result<()> {
    wasmtime_wasi_nn::test_check!();

    // Set up a WASI environment that includes wasi-nn and opens the MobileNet
    // artifacts directory as `fixture` in the guest.
    let engine = Engine::default();
    let (mut store, mut linker) = embed_wasi_nn(&engine, WasiNnCtx::default())?;

    // Build and run the example crate.
    let wasm_file = cargo_build("examples/image-classification");
    let module = Module::from_file(&engine, wasm_file)?;
    linker.module(&mut store, "", &module)?;
    linker
        .get_default(&mut store, "")?
        .typed::<(), ()>(&store)?
        .call(&mut store, ())?;

    Ok(())
}

#[test]
fn image_classification_with_names() -> Result<()> {
    wasmtime_wasi_nn::test_check!();

    // Set up a WASI environment that includes wasi-nn and opens the MobileNet
    // artifacts directory as `fixture` in the guest.
    let engine = Engine::default();
    let mut openvino = backend::openvino::OpenvinoBackend::default();
    let mut registry = InMemoryRegistry::new();
    let mobilenet_dir = wasmtime_wasi_nn::test_check::artifacts_dir();
    registry.load(&mut openvino, &mobilenet_dir)?;
    let wasi_nn = WasiNnCtx::new([Box::new(openvino) as _], Box::new(registry));
    let (mut store, mut linker) = embed_wasi_nn(&engine, wasi_nn)?;

    // Build and run the example crate.
    let wasm_file = cargo_build("examples/image-classification-named");
    let module = Module::from_file(&engine, wasm_file)?;
    linker.module(&mut store, "", &module)?;
    linker
        .get_default(&mut store, "")?
        .typed::<(), ()>(&store)?
        .call(&mut store, ())?;

    Ok(())
}

struct Host {
    wasi: WasiCtx,
    wasi_nn: WasiNnCtx,
}

fn embed_wasi_nn(engine: &Engine, wasi_nn: WasiNnCtx) -> Result<(Store<Host>, Linker<Host>)> {
    let mut linker = Linker::new(&engine);
    let host_dir = Dir::open_ambient_dir(
        wasmtime_wasi_nn::test_check::artifacts_dir(),
        ambient_authority(),
    )?;
    let wasi = WasiCtxBuilder::new()
        .inherit_stdio()
        .preopened_dir(host_dir, "fixture")?
        .build();
    let store = Store::<Host>::new(&engine, Host { wasi, wasi_nn });
    wasmtime_wasi_nn::witx::add_to_linker(&mut linker, |s: &mut Host| &mut s.wasi_nn)?;
    wasmtime_wasi::add_to_linker(&mut linker, |s: &mut Host| &mut s.wasi)?;
    Ok((store, linker))
}

fn cargo_build(crate_dir: impl AsRef<Path>) -> PathBuf {
    let crate_dir = crate_dir.as_ref();
    let crate_name = crate_dir.file_name().unwrap().to_str().unwrap();
    let cargo_toml = crate_dir.join("Cargo.toml");
    let wasm = crate_dir.join(format!("target/wasm32-wasi/release/{}.wasm", crate_name));
    let result = Command::new("cargo")
        .arg("build")
        .arg("--release")
        .arg("--target=wasm32-wasi")
        .arg("--manifest-path")
        .arg(cargo_toml)
        .output()
        .unwrap();
    if !wasm.is_file() {
        panic!("no file found at: {}", wasm.display());
    }
    if !result.status.success() {
        panic!(
            "cargo build failed: {}\n{}",
            result.status,
            String::from_utf8_lossy(&result.stderr)
        );
    }
    wasm
}
