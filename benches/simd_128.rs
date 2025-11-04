use criterion::{criterion_group, criterion_main, Criterion};
use matmul::{generate_test_matrices, simd_matmul};
use std::hint::black_box;

fn benchmark_simd_128(c: &mut Criterion) {
    let test_data = generate_test_matrices(128);
    c.bench_function("simd_128", |b| {
        b.iter(|| black_box(simd_matmul(&test_data.a, &test_data.b)))
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().measurement_time(std::time::Duration::from_secs(3));
    targets = benchmark_simd_128
}
criterion_main!(benches);
