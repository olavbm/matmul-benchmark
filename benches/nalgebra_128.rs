use criterion::{criterion_group, criterion_main, Criterion};
use nalgebra::DMatrix;
use rand::{thread_rng, Rng};
use std::hint::black_box;

fn benchmark_nalgebra_128(c: &mut Criterion) {
    let mut rng = thread_rng();
    let na_a = DMatrix::<f64>::from_fn(128, 128, |_, _| rng.gen());
    let na_b = DMatrix::<f64>::from_fn(128, 128, |_, _| rng.gen());

    c.bench_function("nalgebra_128", |b| {
        b.iter(|| black_box(&na_a * &na_b))
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().measurement_time(std::time::Duration::from_secs(3));
    targets = benchmark_nalgebra_128
}
criterion_main!(benches);
