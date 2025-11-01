use criterion::{criterion_group, criterion_main, Criterion};
use matmul::{blocked_matmul, generate_test_matrices};
use std::hint::black_box;

fn benchmark_blocked_64(c: &mut Criterion) {
    let test_data = generate_test_matrices(64);
    c.bench_function("blocked_64", |b| {
        b.iter(|| black_box(blocked_matmul(&test_data.a, &test_data.b, 64)))
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().measurement_time(std::time::Duration::from_secs(10));
    targets = benchmark_blocked_64
}
criterion_main!(benches);
