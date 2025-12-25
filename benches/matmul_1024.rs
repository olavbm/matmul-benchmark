use criterion::{criterion_group, criterion_main, Criterion};
use matmul::*;
use std::hint::black_box;

const N: usize = 1024;

fn bench_1024(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_1024");
    group.measurement_time(std::time::Duration::from_secs(10));

    let a = Matrix::random(N, N);
    let b = Matrix::random(N, N);

    group.bench_function("naive", |bencher| {
        bencher.iter(|| {
            black_box(naive_matmul(black_box(&a), black_box(&b)))
        })
    });

    group.bench_function("blocked", |bencher| {
        bencher.iter(|| {
            black_box(blocked_matmul(black_box(&a), black_box(&b), 64))
        })
    });

    group.bench_function("simd", |bencher| {
        bencher.iter(|| {
            black_box(simd_matmul(black_box(&a), black_box(&b)))
        })
    });

    group.bench_function("simd_blocked", |bencher| {
        bencher.iter(|| {
            black_box(simd_blocked_matmul_optimized(black_box(&a), black_box(&b)))
        })
    });

    // Reference implementation
    let a_nalg = nalgebra::DMatrix::from_fn(N, N, |i, j| a.get(i, j));
    let b_nalg = nalgebra::DMatrix::from_fn(N, N, |i, j| b.get(i, j));

    group.bench_function("nalgebra", |bencher| {
        bencher.iter(|| {
            black_box(&a_nalg * &b_nalg)
        })
    });

    group.finish();
}

criterion_group!(benches, bench_1024);
criterion_main!(benches);
