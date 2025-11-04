use arrayfire::*;

fn main() {
    set_backend(Backend::OPENCL);

    // Create a simple 3x3 matrix with known values
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let dims = Dim4::new(&[3, 3, 1, 1]);
    let mat = Array::new(&data, dims);

    // Copy to host and print
    let mut host_data = vec![0.0f32; 9];
    mat.host(&mut host_data);

    println!("Input data vector: {:?}", data);
    println!("\nMemory layout in ArrayFire: {:?}", host_data);
    println!("\nArrayFire display:");
    af_print!("mat", mat);

    println!("\n---");
    println!("If column-major, element at [row=1, col=2] should be at index: 1 + 2*3 = 7 (value: 8.0)");
    println!("If row-major, element at [row=1, col=2] should be at index: 1*3 + 2 = 5 (value: 6.0)");
}
