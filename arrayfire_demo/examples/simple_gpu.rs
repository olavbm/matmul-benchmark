use arrayfire::*;

fn main() {
    println!("Simple GPU Matrix Multiplication\n");

    // Show device info
    info();

    // Use OpenCL backend (GPU)
    set_backend(Backend::OPENCL);
    println!("\n✓ Using GPU: {:?}\n", device_info());

    // Create 1024x1024 matrices
    let n = 1024;
    let dims = Dim4::new(&[n, n, 1, 1]);

    println!("Creating {}x{} random matrices...", n, n);
    let a = randu::<f32>(dims);
    let b = randu::<f32>(dims);

    println!("Performing GPU matrix multiplication...");
    let c = matmul(&a, &b, MatProp::NONE, MatProp::NONE);

    sync(0); // Wait for completion

    println!("✓ Done!");
    println!("\nResult dimensions: {:?}", c.dims());
    println!("✓ GPU matrix multiplication completed successfully!");
}
