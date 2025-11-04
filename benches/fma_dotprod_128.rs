use criterion::{criterion_group, criterion_main, Criterion};
use matmul::fma_dotprod;
use std::hint::black_box;

fn benchmark_fma_dotprod_128(c: &mut Criterion) {
    let a: Vec<f64> = (0..128).map(|i| i as f64 * 0.1).collect();
    let b: Vec<f64> = (0..128).map(|i| (i as f64 + 1.0) * 0.2).collect();

    c.bench_function("fma_dotprod_128", |bencher| {
        bencher.iter(|| black_box(fma_dotprod(&a, &b)))
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().measurement_time(std::time::Duration::from_secs(3));
    targets = benchmark_fma_dotprod_128
}
criterion_main!(benches);
