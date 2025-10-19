use arrayfire::*;
use std::time::Instant;

fn main() {
    println!("ArrayFire GPU vs CPU Matrix Multiplication\n");

    // Show available devices
    info();
    println!();

    let sizes = vec![256, 512, 1024];

    println!("{:<12} {:<15} {:<15} {:<10}", "Matrix Size", "CPU (ms)", "GPU (ms)", "Speedup");
    println!("{:-<55}", "");

    for n in sizes {
        benchmark_size(n);
    }
}

fn benchmark_size(n: u64) {
    let dims = Dim4::new(&[n, n, 1, 1]);

    // GPU Benchmark (OpenCL backend)
    set_backend(Backend::OPENCL);
    let a_gpu = randu::<f32>(dims);
    let b_gpu = randu::<f32>(dims);

    // Warm-up
    let _ = matmul(&a_gpu, &b_gpu, MatProp::NONE, MatProp::NONE);
    sync(0);

    let start = Instant::now();
    let _c_gpu = matmul(&a_gpu, &b_gpu, MatProp::NONE, MatProp::NONE);
    sync(0);
    let gpu_time = start.elapsed();

    // CPU Benchmark (with oneAPI MKL)
    set_backend(Backend::CPU);
    let a_cpu = randu::<f32>(dims);
    let b_cpu = randu::<f32>(dims);

    // Warm-up
    let _ = matmul(&a_cpu, &b_cpu, MatProp::NONE, MatProp::NONE);
    sync(0);

    let start = Instant::now();
    let _c_cpu = matmul(&a_cpu, &b_cpu, MatProp::NONE, MatProp::NONE);
    sync(0);
    let cpu_time = start.elapsed();

    let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
    let gflops = (2.0 * (n as f64).powi(3)) / (gpu_time.as_secs_f64() * 1e9);

    println!(
        "{:<12} {:<15.2} {:<15.2} {:<10.2}x",
        format!("{}×{}", n, n),
        cpu_time.as_secs_f64() * 1000.0,
        gpu_time.as_secs_f64() * 1000.0,
        speedup
    );

    if n == 1024 {
        println!("\n✓ GPU achieved {:.2} GFLOPS for 1024×1024", gflops);
    }
}
