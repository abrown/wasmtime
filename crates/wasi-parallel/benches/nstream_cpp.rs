mod run;
use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, Criterion,
};
use run::BenchContext;
use std::path::Path;
use wasmtime::Val;

#[derive(Debug, Clone, Copy)]
enum ExecMode {
    Sequential = 0,
    CPU = 1,
    GPU = 2,
}

fn bench_nstream(c: &mut Criterion) {
    let mut group = c.benchmark_group("nstream_cpp");
    measure_wasm(
        &mut group,
        "benches/wasm/nstream_cpp.wasm",
        ExecMode::Sequential,
    );
    measure_wasm(&mut group, "benches/wasm/nstream_cpp.wasm", ExecMode::CPU);
    measure_wasm(&mut group, "benches/wasm/nstream_cpp.wasm", ExecMode::GPU);
}

fn measure_wasm<P: AsRef<Path>>(group: &mut BenchmarkGroup<WallTime>, path: P, mode: ExecMode) {
    let path = path.as_ref();
    let mut ctx = BenchContext::new(path).unwrap();
    ctx.invoke("setup", Some(vec![Val::I32(mode as i32)]))
        .unwrap();

    group.bench_function(
        format!("{} (seq: {:?})", path.to_string_lossy(), mode),
        |b| {
            b.iter(|| ctx.invoke("run", None));
        },
    );
}

criterion_group!(benches, bench_nstream);
criterion_main!(benches);
