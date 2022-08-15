//! Compare the advantage of running a hand-coded nstream implementation using
//! wasi-parallel vs sequentially. WARNING: the benchmark times are not
//! representative of the parallel speedup because the amount of parallelized
//! work, the `$nstream` kernel, is small (~10%) compared to the total benchmark
//! run time, which includes Cranelift JIT compilation and the sequential buffer
//! initialization (`$initialize`). This could be fixed by adding a method of
//! starting and stopping measurement right before and after the kernel like
//! most other implementations do (TODO).
mod run;

use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, Criterion,
};
use run::run;
use std::path::Path;

fn bench_nstream(c: &mut Criterion) {
    let mut group = c.benchmark_group("nstream");
    measure_wasm(&mut group, "benches/nstream/parallel.wat");
    measure_wasm(&mut group, "benches/nstream/sequential.wat");
}

fn measure_wasm<P: AsRef<Path>>(group: &mut BenchmarkGroup<WallTime>, path: P) {
    let path = path.as_ref();
    group.bench_function(path.to_string_lossy(), |b| {
        b.iter(|| run(path));
    });
}

criterion_group!(benches, bench_nstream);
criterion_main!(benches);
