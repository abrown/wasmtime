# wasmtime-wasi-parallel

This crate enables experimental support for the [wasi-parallel] API in Wasmtime.
__WARNING__: _this implementation is highly experimental, is subject to change,
and abuses the Wasmtime API!_ It is published as a proof-of-concept to discuss
the design of the wasi-parallel specification. Please create any issues related
to this implementation in the [wasi-parallel] repository.

The main idea is to expose a "parallel for" mechanism using WASI (see the
[explainer] for more details). The "parallel for" call is not limited to CPU
execution; this proof-of-concept implementation can execute parallel code on
both the CPU (using Wasmtime's JIT compiled functions) and the GPU (using
OpenCL). If you plan to experiment with this crate, see the "Use" section below.

This branch is an executable RFC--your feedback is appreciated!

[wasi-parallel]: https://github.com/abrown/wasi-parallel-spec
[explainer]: https://github.com/abrown/wasi-parallel-spec/blob/main/docs/Explainer.md

### Build

```
cargo build
```

To optionally include GPU support, add the `--features opencl` flag.

### Test

```
cargo test
```

Note: the Rust code in `tests/rust` is compiled by `build.rs` to `tests/wasm`.

### Benchmark

```
cargo bench
```

### Use + Gotchas

This crate/branch is usable from the Wasmtime CLI:

```
wasmtime run --wasi-modules experimental-wasi-parallel <module>
```

The input Wasm module has several requirements:
 - use the `"wasi_ephemeral_parallel"` namespace; see this [WAT example] for a
   manually-written module or this [Rust example] using some bindings
 - the first parameter of `wasi_ephemeral_parallel::parallel_for` defines what
   code executes in parallel--for CPU execution, this is an index into a funcref
   table exported as `"table"`
 - to run code on a GPU, this crate expects SPIR-V for the function to exist in
   a Wasm custom section; the [`attach-spirv`] example binary provides a way to
   attach SPIR-V to a Wasm module--for GPU execution the first
   `wasi_ephemeral_parallel::parallel_for` parameter is the custom section ID
   field
 - ideally, the Wasm function index (CPU) and SPIR-V ID (GPU) should match but
   this is difficult to enforce with the current "fat binary"
   mechanism--remember, this is a proof-of-concept and this part of the design
   highly unstable

[WAT example]: tests/wat/parallel-for.wat
[Rust example]: tests/rust/buffer.rs
[`attach-spirv`]: examples/attach-spirv.rs
