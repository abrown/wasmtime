This example project demonstrates using the `wasi-nn` API. It consists of Rust code that is built using the
`wasm32-wasi` target by the enclosing project's `build.rs` file. The enclosing project contains a Rust test that
instantiates the appropriate wasi-nn context and runs the compiled Wasm.
