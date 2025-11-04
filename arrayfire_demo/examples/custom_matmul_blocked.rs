use arrayfire::*;
use af_opencl_interop as afcl;
use ocl_core::{self as core};
use std::ffi::CString;
use std::time::Instant;

// Cache-blocked OpenCL matrix multiplication kernel
// Uses local memory (GPU's on-chip cache) for better performance
// ArrayFire uses COLUMN-MAJOR layout: A[row,col] = A[row + col*N]
const MATMUL_BLOCKED_KERNEL: &str = r#"
#define BLOCK_SIZE 16

__kernel void matmul_blocked(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int N)
{
    // Block index
    int bx = get_group_id(0);
    int by = get_group_id(1);

    // Thread index within block
    int tx = get_local_id(0);
    int ty = get_local_id(1);

    // Row and column of C to compute
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    // Shared memory for tiles
    __local float As[BLOCK_SIZE][BLOCK_SIZE];
    __local float Bs[BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        // Load tile of A into shared memory
        // Column-major: A[row, t*BLOCK_SIZE+tx] = A[row + (t*BLOCK_SIZE+tx)*N]
        if (row < N && (t * BLOCK_SIZE + tx) < N) {
            As[ty][tx] = A[row + (t * BLOCK_SIZE + tx) * N];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load tile of B into shared memory
        // Column-major: B[t*BLOCK_SIZE+ty, col] = B[(t*BLOCK_SIZE+ty) + col*N]
        if (col < N && (t * BLOCK_SIZE + ty) < N) {
            Bs[ty][tx] = B[(t * BLOCK_SIZE + ty) + col * N];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        // Synchronize to make sure tiles are loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Multiply tiles
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        // Synchronize before loading next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write result
    // Column-major: C[row, col] = C[row + col*N]
    if (row < N && col < N) {
        C[row + col * N] = sum;
    }
}
"#;

fn main() {
    println!("Cache-Blocked OpenCL Matrix Multiplication Kernel\n");

    set_backend(Backend::OPENCL);
    info();
    println!();

    let n = 1024;
    let dims = Dim4::new(&[n as u64, n as u64, 1, 1]);

    println!("Matrix size: {}x{}", n, n);
    println!("Block size: 16x16 (using GPU local memory)\n");

    // Create matrices
    let a = randu::<f32>(dims);
    let b = randu::<f32>(dims);
    let mut c_naive = constant(0.0f32, dims);
    let mut c_blocked = constant(0.0f32, dims);

    // Get OpenCL context
    let dev_id = afcl::get_device_id();
    let ctx = afcl::get_context(false);
    let queue = afcl::get_queue(false);

    println!("✓ Using OpenCL device: {:?}\n", dev_id);

    // Compile kernel
    let src = CString::new(MATMUL_BLOCKED_KERNEL).unwrap();
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

    let kernel = unsafe {
        core::create_kernel(&program, "matmul_blocked")
            .expect("Failed to create kernel")
    };

    println!("✓ Compiled cache-blocked kernel\n");

    // Get memory pointers
    let a_ptr = unsafe { core::Mem::from_raw_copied_ptr(a.device_ptr() as *mut std::ffi::c_void) };
    let b_ptr = unsafe { core::Mem::from_raw_copied_ptr(b.device_ptr() as *mut std::ffi::c_void) };
    let c_ptr = unsafe { core::Mem::from_raw_copied_ptr(c_blocked.device_ptr() as *mut std::ffi::c_void) };

    // Set kernel arguments
    unsafe {
        core::set_kernel_arg(&kernel, 0, core::ArgVal::mem(&a_ptr)).unwrap();
        core::set_kernel_arg(&kernel, 1, core::ArgVal::mem(&b_ptr)).unwrap();
        core::set_kernel_arg(&kernel, 2, core::ArgVal::mem(&c_ptr)).unwrap();
        core::set_kernel_arg(&kernel, 3, core::ArgVal::scalar(&(n as i32))).unwrap();
    }

    println!("Running cache-blocked GPU kernel...");

    let cmd_queue = unsafe { core::CommandQueue::from_raw_copied_ptr(queue) };

    // Execute with 16x16 work groups (local memory blocks)
    let block_size = 16;
    let global_work_size = [
        ((n + block_size - 1) / block_size) * block_size,
        ((n + block_size - 1) / block_size) * block_size,
        1
    ];
    let local_work_size = [block_size, block_size, 1];

    let start = Instant::now();

    unsafe {
        core::enqueue_kernel::<(), ()>(
            &cmd_queue,
            &kernel,
            2,
            None,
            &global_work_size,
            Some(local_work_size), // Use 16x16 work groups
            None,
            None,
        ).expect("Failed to enqueue kernel");

        core::finish(&cmd_queue).unwrap();
    }

    let blocked_time = start.elapsed();

    a.unlock();
    b.unlock();
    c_blocked.unlock();

    println!("✓ Blocked kernel completed in {:.2} ms\n", blocked_time.as_secs_f64() * 1000.0);

    // Compare with ArrayFire's matmul
    println!("Comparing with ArrayFire's optimized matmul...");
    let start = Instant::now();
    let c_af = matmul(&a, &b, MatProp::NONE, MatProp::NONE);
    sync(0);
    let af_time = start.elapsed();

    println!("✓ ArrayFire matmul completed in {:.2} ms\n", af_time.as_secs_f64() * 1000.0);

    // Verify correctness - copy both results to host and compare
    let mut custom_result = vec![0.0f32; (n * n) as usize];
    let mut af_result = vec![0.0f32; (n * n) as usize];

    c_blocked.host(&mut custom_result);
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
    let gflops = (2.0 * (n as f64).powi(3)) / (blocked_time.as_secs_f64() * 1e9);

    println!("Results:");
    println!("  Custom blocked: {:.2} ms", blocked_time.as_secs_f64() * 1000.0);
    println!("  ArrayFire:      {:.2} ms ({:.2}x faster)",
             af_time.as_secs_f64() * 1000.0,
             blocked_time.as_secs_f64() / af_time.as_secs_f64());
    println!("  GFLOPS:         {:.2}", gflops);

    println!("\nCorrectness:");
    println!("  Max absolute error: {:.2e}", max_error);
    println!("  RMSE:               {:.2e}", rmse);
    println!("  Relative error:     {:.2e}", relative_error);

    if max_error < 1e-3 {
        println!("\n✓ Results match!");
    } else {
        println!("\n⚠ Warning: Results differ significantly from ArrayFire!");
    }

    println!("\n💡 This kernel uses GPU local memory (like CPU L1 cache)");
    println!("   to reduce global memory accesses - just like your");
    println!("   cache-blocked CPU implementation!");
}
