use af_opencl_interop as afcl;
use arrayfire::*;
use ocl_core::{self as core};
use std::ffi::CString;
use std::time::Instant;

// Custom OpenCL kernel (naive implementation)
const MATMUL_KERNEL: &str = r#"
__kernel void matmul_naive(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int N)
{
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row + k * N] * B[k + col * N];
        }
        C[row + col * N] = sum;
    }
}
"#;

fn main() {
    println!("GPU Matrix Multiplication Demo\n");

    // Print available devices
    info();
    println!();

    // Set backend to OpenCL for GPU
    set_backend(Backend::OPENCL);

    let device_info = device_info();
    println!("Using device: {:?}", device_info);
    println!();

    // Compile custom kernel once
    let kernel = setup_custom_kernel();

    // Matrix sizes to benchmark
    let sizes = vec![128, 256, 512, 1024];

    println!(
        "{:<12} {:<15} {:<15} {:<15} {:<12}",
        "Size", "Custom (ms)", "ArrayFire (ms)", "GFLOPS", "Speedup"
    );
    println!("{:-<75}", "");

    for size in sizes {
        benchmark_matmul(size, &kernel);
    }
}

fn setup_custom_kernel() -> core::Kernel {
    let dev_id = afcl::get_device_id();
    let ctx = afcl::get_context(false);

    let src = CString::new(MATMUL_KERNEL).unwrap();
    let program =
        unsafe { core::create_program_with_source(ctx, &[src]).expect("Failed to create program") };

    let device = unsafe { core::DeviceId::from_raw(dev_id) };

    unsafe {
        core::build_program(
            &program,
            Some(&[device]),
            &CString::new("").unwrap(),
            None,
            None,
        )
        .expect("Failed to build program");
    }

    unsafe { core::create_kernel(&program, "matmul_naive").expect("Failed to create kernel") }
}

fn benchmark_matmul(n: u64, kernel: &core::Kernel) {
    let dims = Dim4::new(&[n, n, 1, 1]);

    // Create random matrices on GPU
    let a = randu::<f32>(dims);
    let b = randu::<f32>(dims);
    let c_custom = constant(0.0f32, dims);

    // Get OpenCL objects
    let queue = afcl::get_queue(false);
    let cmd_queue = unsafe { core::CommandQueue::from_raw_copied_ptr(queue) };

    // Get memory pointers
    let a_ptr = unsafe { core::Mem::from_raw_copied_ptr(a.device_ptr() as *mut std::ffi::c_void) };
    let b_ptr = unsafe { core::Mem::from_raw_copied_ptr(b.device_ptr() as *mut std::ffi::c_void) };
    let c_ptr =
        unsafe { core::Mem::from_raw_copied_ptr(c_custom.device_ptr() as *mut std::ffi::c_void) };

    // Set kernel arguments
    unsafe {
        core::set_kernel_arg(kernel, 0, core::ArgVal::mem(&a_ptr)).unwrap();
        core::set_kernel_arg(kernel, 1, core::ArgVal::mem(&b_ptr)).unwrap();
        core::set_kernel_arg(kernel, 2, core::ArgVal::mem(&c_ptr)).unwrap();
        core::set_kernel_arg(kernel, 3, core::ArgVal::scalar(&(n as i32))).unwrap();
    }

    let global_work_size = [n as usize, n as usize, 1];

    // Warm-up run for custom kernel
    unsafe {
        core::enqueue_kernel::<(), ()>(
            &cmd_queue,
            kernel,
            2,
            None,
            &global_work_size,
            None,
            None,
            None,
        )
        .unwrap();
        core::finish(&cmd_queue).unwrap();
    }

    // Benchmark custom kernel
    let iterations = 3;
    let start = Instant::now();
    for _ in 0..iterations {
        unsafe {
            core::enqueue_kernel::<(), ()>(
                &cmd_queue,
                kernel,
                2,
                None,
                &global_work_size,
                None,
                None,
                None,
            )
            .unwrap();
            core::finish(&cmd_queue).unwrap();
        }
    }
    let custom_time = start.elapsed() / iterations;

    // Unlock arrays
    a.unlock();
    b.unlock();
    c_custom.unlock();

    // Benchmark ArrayFire
    let _ = matmul(&a, &b, MatProp::NONE, MatProp::NONE);
    sync(0);

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = matmul(&a, &b, MatProp::NONE, MatProp::NONE);
    }
    sync(0);
    let af_time = start.elapsed() / iterations;

    // Calculate GFLOPS (based on custom kernel time)
    let ops = 2.0 * (n as f64).powi(3);
    let gflops = ops / custom_time.as_secs_f64() / 1e9;

    let speedup = af_time.as_secs_f64() / custom_time.as_secs_f64();

    println!(
        "{:<12} {:<15.2} {:<15.2} {:<15.2} {:<12.2}x",
        format!("{}x{}", n, n),
        custom_time.as_secs_f64() * 1000.0,
        af_time.as_secs_f64() * 1000.0,
        gflops,
        speedup
    );
}
