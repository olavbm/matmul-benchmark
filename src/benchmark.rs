// Simplified benchmarking - just basic timing for now
use crate::Matrix;
use std::time::{Duration, Instant};

pub fn time_matmul<F>(matmul_fn: F, a: &Matrix, b: &Matrix) -> (Matrix, Duration) 
where 
    F: Fn(&Matrix, &Matrix) -> Matrix
{
    let start = Instant::now();
    let result = matmul_fn(a, b);
    let duration = start.elapsed();
    (result, duration)
}

pub fn benchmark_gflops(size: usize, duration: Duration) -> f64 {
    let ops = 2.0 * (size as f64).powi(3); // 2n³ operations for matmul
    ops / duration.as_secs_f64() / 1e9 // GFLOP/s
}