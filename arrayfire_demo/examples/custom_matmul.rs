use arrayfire::*;
use af_opencl_interop as afcl;
use ocl_core::{self as core};
use std::ffi::CString;
use std::time::Instant;

// Simple naive OpenCL matrix multiplication kernel
// ArrayFire uses COLUMN-MAJOR layout: A[row,col] = A[row + col*N]
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
            // Column-major: A[row,k] = A[row + k*N], B[k,col] = B[k + col*N]
            sum += A[row + k * N] * B[k + col * N];
        }
        C[row + col * N] = sum;
    }
}
"#;

fn main() {
    println!("Custom OpenCL Matrix Multiplication Kernel\n");

    // Set backend to OpenCL
    set_backend(Backend::OPENCL);
    info();
    println!();

    let n = 512;
    let dims = Dim4::new(&[n as u64, n as u64, 1, 1]);

    println!("Matrix size: {}x{}", n, n);
    println!("Creating random matrices on GPU...\n");

    // Create ArrayFire arrays (these live on GPU)
    let a = randu::<f32>(dims);
    let b = randu::<f32>(dims);
    let mut c = constant(0.0f32, dims);

    // Get OpenCL context, device, and queue from ArrayFire
    let dev_id = afcl::get_device_id();
    let ctx = afcl::get_context(false);
    let queue = afcl::get_queue(false);

    println!("✓ Got OpenCL context from ArrayFire");
    println!("  Device: {:?}", dev_id);

    // Compile the kernel
    let src = CString::new(MATMUL_KERNEL).unwrap();
    let program = unsafe {
        core::create_program_with_source(ctx, &[src])
            .expect("Failed to create program")
    };

    let device = unsafe { core::DeviceId::from_raw(dev_id) };

    unsafe {
        core::build_program(
            &program,
            Some(&[device]),
            &CString::new("").unwrap(),
            None,
            None,
        ).expect("Failed to build program");
    }

    println!("✓ Compiled custom kernel");

    let kernel = unsafe {
        core::create_kernel(&program, "matmul_naive")
            .expect("Failed to create kernel")
    };

    println!("✓ Created kernel object\n");

    // Lock arrays and get cl_mem pointers
    let a_ptr = unsafe { core::Mem::from_raw_copied_ptr(a.device_ptr() as *mut std::ffi::c_void) };
    let b_ptr = unsafe { core::Mem::from_raw_copied_ptr(b.device_ptr() as *mut std::ffi::c_void) };
    let c_ptr = unsafe { core::Mem::from_raw_copied_ptr(c.device_ptr() as *mut std::ffi::c_void) };

    // Set kernel arguments
    unsafe {
        core::set_kernel_arg(&kernel, 0, core::ArgVal::mem(&a_ptr)).unwrap();
        core::set_kernel_arg(&kernel, 1, core::ArgVal::mem(&b_ptr)).unwrap();
        core::set_kernel_arg(&kernel, 2, core::ArgVal::mem(&c_ptr)).unwrap();
        core::set_kernel_arg(&kernel, 3, core::ArgVal::scalar(&(n as i32))).unwrap();
    }

    println!("Running custom GPU kernel...");

    let cmd_queue = unsafe { core::CommandQueue::from_raw_copied_ptr(queue) };

    // Execute kernel
    let global_work_size = [n, n, 1];
    let start = Instant::now();

    unsafe {
        core::enqueue_kernel::<(), ()>(
            &cmd_queue,
            &kernel,
            2, // work_dim
            None, // global_work_offset
            &global_work_size,
            None, // local_work_size (let OpenCL decide)
            None, // wait list
            None, // event
        ).expect("Failed to enqueue kernel");

        // Wait for completion
        core::finish(&cmd_queue).unwrap();
    }

    let custom_time = start.elapsed();

    // Unlock the arrays
    a.unlock();
    b.unlock();
    c.unlock();

    println!("✓ Custom kernel completed in {:.2} ms\n", custom_time.as_secs_f64() * 1000.0);

    // Compare with ArrayFire's built-in matmul
    println!("Comparing with ArrayFire's optimized matmul...");

    let start = Instant::now();
    let c_af = matmul(&a, &b, MatProp::NONE, MatProp::NONE);
    sync(0);
    let af_time = start.elapsed();

    println!("✓ ArrayFire matmul completed in {:.2} ms", af_time.as_secs_f64() * 1000.0);

    // Verify correctness - copy both results to host and compare
    let mut custom_result = vec![0.0f32; (n * n) as usize];
    let mut af_result = vec![0.0f32; (n * n) as usize];

    c.host(&mut custom_result);
    c_af.host(&mut af_result);

    // Compute detailed error statistics
    let mut max_error = 0.0f32;
    let mut sum_sq_error = 0.0f64;

    for i in 0..(n * n) as usize {
        let error = (custom_result[i] - af_result[i]).abs();
        max_error = max_error.max(error);
        sum_sq_error += (error as f64).powi(2);
    }

    let rmse = (sum_sq_error / ((n * n) as f64)).sqrt();
    let max_val = af_result.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let relative_error = max_error / max_val;

    println!("\nResults:");
    println!("  Custom kernel: {:.2} ms", custom_time.as_secs_f64() * 1000.0);
    println!("  ArrayFire:     {:.2} ms ({:.2}x faster)",
             af_time.as_secs_f64() * 1000.0,
             custom_time.as_secs_f64() / af_time.as_secs_f64());

    println!("\nCorrectness:");
    println!("  Max absolute error: {:.2e}", max_error);
    println!("  RMSE:               {:.2e}", rmse);
    println!("  Relative error:     {:.2e}", relative_error);

    if max_error < 1e-3 {
        println!("\n✓ Results match!");
    } else {
        println!("\n⚠ Warning: Results differ significantly from ArrayFire!");
    }
}
