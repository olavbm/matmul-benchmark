use arrayfire::*;
use std::time::Instant;

fn main() {
    println!("ArrayFire GPU Matrix Multiplication Demo\n");

    // Print available devices
    info();
    println!();

    // Set backend to OpenCL for GPU
    set_backend(Backend::OPENCL);

    let device_info = device_info();
    println!("Using device: {}", device_info);
    println!();

    // Matrix sizes to benchmark
    let sizes = vec![128, 256, 512, 1024];

    println!("{:<10} {:<15} {:<15} {:<15}", "Size", "CPU Time (ms)", "GPU Time (ms)", "Speedup");
    println!("{:-<60}", "");

    for size in sizes {
        benchmark_matmul(size);
    }
}

fn benchmark_matmul(n: u64) {
    let dims = Dim4::new(&[n, n, 1, 1]);

    // Create random matrices on GPU
    let a = randu::<f32>(dims);
    let b = randu::<f32>(dims);

    // Warm-up run
    let _ = matmul(&a, &b, MatProp::NONE, MatProp::NONE);
    sync(0);

    // GPU benchmark
    let start = Instant::now();
    let c_gpu = matmul(&a, &b, MatProp::NONE, MatProp::NONE);
    sync(0); // Wait for GPU to finish
    let gpu_time = start.elapsed();

    // CPU backend benchmark
    set_backend(Backend::CPU);
    let a_cpu = randu::<f32>(dims);
    let b_cpu = randu::<f32>(dims);

    // Warm-up
    let _ = matmul(&a_cpu, &b_cpu, MatProp::NONE, MatProp::NONE);
    sync(0);

    let start = Instant::now();
    let c_cpu = matmul(&a_cpu, &b_cpu, MatProp::NONE, MatProp::NONE);
    sync(0);
    let cpu_time = start.elapsed();

    // Switch back to GPU
    set_backend(Backend::OPENCL);

    let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();

    println!(
        "{:<10} {:<15.2} {:<15.2} {:<15.2}x",
        format!("{}x{}", n, n),
        cpu_time.as_secs_f64() * 1000.0,
        gpu_time.as_secs_f64() * 1000.0,
        speedup
    );

    // Verify correctness (small example)
    if n == 128 {
        let diff = sub(&c_cpu, &c_gpu, false);
        let max_diff = max_all(&abs(&diff)).0;
        println!("  └─ Max difference between CPU and GPU: {:.2e}", max_diff);
    }
}
