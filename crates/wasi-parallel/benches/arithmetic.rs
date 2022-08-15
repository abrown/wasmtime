//! Compare adding 1 billion integers when running a hand-coded nstream
//! implementation using wasi-parallel vs sequentially.
mod run;

use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, Criterion,
};
use run::run;
use std::path::Path;

fn bench_arithmetic(c: &mut Criterion) {
    let mut group = c.benchmark_group("arithmetic");
    // In order to not spend multiple minutes running this benchmark, we reduce
    // the number of samples. To get statistically significant results, remove
    // this line.
    group.sample_size(10);
    measure_wasm(&mut group, "benches/arithmetic/parallel.wat");
    measure_wasm(&mut group, "benches/arithmetic/sequential.wat");
}

fn measure_wasm<P: AsRef<Path>>(group: &mut BenchmarkGroup<WallTime>, path: P) {
    let path = path.as_ref();
    group.bench_function(path.to_string_lossy(), |b| {
        b.iter(|| run(path));
    });
}

criterion_group!(benches, bench_arithmetic);
criterion_main!(benches);
