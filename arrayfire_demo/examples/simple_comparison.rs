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

    // Verify correctness: GPU vs CPU
    println!("\n{:-<55}", "");
    println!("Verifying Correctness (GPU vs CPU)");
    println!("{:-<55}", "");
    verify_correctness();
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

fn verify_correctness() {
    let n = 512;
    let dims = Dim4::new(&[n, n, 1, 1]);

    // Create same input matrices (using same seed for reproducibility)
    set_backend(Backend::OPENCL);
    set_seed(42);
    let a_gpu = randu::<f32>(dims);
    let b_gpu = randu::<f32>(dims);

    // Compute on GPU
    let c_gpu = matmul(&a_gpu, &b_gpu, MatProp::NONE, MatProp::NONE);
    sync(0);

    // Copy GPU result to host BEFORE switching backend
    let mut gpu_result = vec![0.0f32; (n * n) as usize];
    c_gpu.host(&mut gpu_result);

    // Switch to CPU and recreate same matrices
    set_backend(Backend::CPU);
    set_seed(42);
    let a_cpu = randu::<f32>(dims);
    let b_cpu = randu::<f32>(dims);

    // Compute on CPU
    let c_cpu = matmul(&a_cpu, &b_cpu, MatProp::NONE, MatProp::NONE);
    sync(0);

    // Copy CPU result to host
    let mut cpu_result = vec![0.0f32; (n * n) as usize];
    c_cpu.host(&mut cpu_result);

    // Compute statistics
    let mut max_error = 0.0f32;
    let mut sum_sq_error = 0.0f64;

    for i in 0..(n * n) as usize {
        let error = (gpu_result[i] - cpu_result[i]).abs();
        max_error = max_error.max(error);
        sum_sq_error += (error as f64).powi(2);
    }

    let rmse = (sum_sq_error / ((n * n) as f64)).sqrt();
    let relative_error = max_error / cpu_result.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

    println!("Test size: {}×{}", n, n);
    println!("Max absolute error: {:.2e}", max_error);
    println!("RMSE: {:.2e}", rmse);
    println!("Relative error: {:.2e}", relative_error);

    // Floating point tolerance (should be very small for identical implementations)
    if max_error < 1e-3 {
        println!("\n✓ GPU and CPU implementations match!");
    } else {
        println!("\n⚠ Warning: Significant difference between GPU and CPU results");
    }
}
