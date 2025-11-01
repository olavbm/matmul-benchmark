use criterion::{criterion_group, criterion_main, Criterion};
use matmul::{generate_test_matrices, naive_matmul};
use std::hint::black_box;

fn benchmark_naive_64(c: &mut Criterion) {
    let test_data = generate_test_matrices(64);
    c.bench_function("naive_64", |b| {
        b.iter(|| black_box(naive_matmul(&test_data.a, &test_data.b)))
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().measurement_time(std::time::Duration::from_secs(10));
    targets = benchmark_naive_64
}
criterion_main!(benches);
