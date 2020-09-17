# wasmtime-wasi-nn

This crate enables support for the [wasi-nn] API in Wasmtime. Currently it contains an implementation of [wasi-nn] using
OpenVINO™ but in the future it could support multiple machine learning backends. Since the [wasi-nn] API is expected
to be an optional feature of WASI, this crate is currently separate from the [wasi-common] crate. This crate is
experimental and its API, functionality, and location could quickly change.

[fixture README]: tests/fixture/README.md
[tests]: tests
[openvino]: https://crates.io/crates/openvino
[wasi-nn]: https://github.com/WebAssembly/wasi-nn
[wasi-common]: ../wasi-common

### Use

Use the Wasmtime APIs to instantiate a Wasm module and link in the `WasiNn` implementation as follows:

```
let wasi_nn = WasiNn::new(&store, WasiNnCtx::new()?);
wasi_nn.add_to_linker(&mut linker)?;
```

See [tests/main.rs](tests/main.rs) for an example.

### Build

This crate should build as usual (i.e. `cargo build`) but note that using an existing installation of OpenVINO™, rather
than building from source, will drastically improve the build times. See the [openvino] crate for more information

### Test

An end-to-end test demonstrating ML classification is included in [tests]:
 - `tests/fixture` contains the ML model and a prepared tensor; instructions for creating these files are in the 
   [fixture README]
 - `tests/example` contains a standalone Rust project that uses the [wasi-nn] APIs and is compiled to the `wasm32-wasi`
   target
 - `build.rs` is responsible for compiling `tests/example` and generating a test to run the example
 - `tests/main.rs` includes the generated test and runs by setting up a Wasm instance and linking in the wasi-nn
   implementation in this crate
