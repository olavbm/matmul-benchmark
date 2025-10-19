use arrayfire::*;
use af_opencl_interop as afcl;
use ocl_core::{self as core, OclPrm};
use std::ffi::CString;
use std::time::Instant;

// Simple naive OpenCL matrix multiplication kernel
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
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
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
    let (dev_id, ctx, queue) = afcl::get_device_id_context_queue();

    println!("✓ Got OpenCL context from ArrayFire");
    println!("  Device: {:?}", dev_id);

    // Compile the kernel
    let src = CString::new(MATMUL_KERNEL).unwrap();
    let program = unsafe {
        core::create_program_with_source(ctx, &[src])
            .expect("Failed to create program")
    };

    unsafe {
        core::build_program(
            &program,
            Some(&[dev_id]),
            &CString::new("").unwrap(),
            None,
            None,
        ).expect("Failed to build program");
    }

    println!("✓ Compiled custom kernel");

    let kernel = unsafe {
        core::create_kernel(&program, &CString::new("matmul_naive").unwrap())
            .expect("Failed to create kernel")
    };

    println!("✓ Created kernel object\n");

    // Lock arrays and get cl_mem pointers
    let a_ptr = a.device_ptr() as core::Mem;
    let b_ptr = b.device_ptr() as core::Mem;
    let c_ptr = c.device_ptr() as core::Mem;

    // Set kernel arguments
    unsafe {
        core::set_kernel_arg(&kernel, 0, core::ArgVal::mem(&a_ptr)).unwrap();
        core::set_kernel_arg(&kernel, 1, core::ArgVal::mem(&b_ptr)).unwrap();
        core::set_kernel_arg(&kernel, 2, core::ArgVal::mem(&c_ptr)).unwrap();
        core::set_kernel_arg(&kernel, 3, core::ArgVal::scalar(&(n as i32))).unwrap();
    }

    println!("Running custom GPU kernel...");

    // Execute kernel
    let global_work_size = [n, n];
    let start = Instant::now();

    unsafe {
        core::enqueue_kernel(
            &queue,
            &kernel,
            2, // work_dim
            None, // global_work_offset
            &global_work_size,
            None, // local_work_size (let OpenCL decide)
            None, // wait list
            None, // event
        ).expect("Failed to enqueue kernel");

        // Wait for completion
        core::finish(queue).unwrap();
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

    // Verify correctness
    let diff = sub(&c, &c_af, false);
    let max_error = max_all(&abs(&diff)).0;

    println!("\nResults:");
    println!("  Custom kernel: {:.2} ms", custom_time.as_secs_f64() * 1000.0);
    println!("  ArrayFire:     {:.2} ms ({:.2}x faster)",
             af_time.as_secs_f64() * 1000.0,
             custom_time.as_secs_f64() / af_time.as_secs_f64());
    println!("  Max error:     {:.2e}", max_error);

    if max_error < 1e-4 {
        println!("\n✓ Results match!");
    }
}
