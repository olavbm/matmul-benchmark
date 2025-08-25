use matmul::{Matrix, naive_matmul, generate_test_matrices};
use std::time::Instant;

fn main() {
    println!("Matrix Multiplication Benchmark");
    println!("===============================");
    
    // Test correctness
    println!("\n1. Testing correctness:");
    test_simple();
    
    // Performance comparison
    let sizes = vec![64, 128, 256];
    
    for size in sizes {
        println!("\n2. Benchmarking {}×{} matrix multiplication:", size, size);
        let test_data = generate_test_matrices(size);
        
        // Test naive implementation
        let start = Instant::now();
        let _result_naive = naive_matmul(&test_data.a, &test_data.b);
        let naive_time = start.elapsed();
        let naive_gflops = (2.0 * (size as f64).powi(3)) / naive_time.as_secs_f64() / 1e9;
        
        // Test optimized implementation (using column-major B)
        let b_col_major = test_data.b.to_col_major();
        let start = Instant::now();
        let _result_optimized = naive_matmul(&test_data.a, &b_col_major);
        let optimized_time = start.elapsed();
        let optimized_gflops = (2.0 * (size as f64).powi(3)) / optimized_time.as_secs_f64() / 1e9;
        
        println!("   Naive (row×row):       {:8.2} ms ({:6.2} GFLOP/s)", 
                 naive_time.as_secs_f64() * 1000.0, naive_gflops);
        println!("   Optimized (row×col):   {:8.2} ms ({:6.2} GFLOP/s)", 
                 optimized_time.as_secs_f64() * 1000.0, optimized_gflops);
        
        let speedup = naive_time.as_secs_f64() / optimized_time.as_secs_f64();
        println!("   Speedup: {:.2}x", speedup);
    }
    
    println!("\n3. What you're seeing:");
    println!("   - Matrix tracks memory layout (RowMajor/ColMajor)");
    println!("   - Naive: Standard row×row access pattern");  
    println!("   - Optimized: Converts B to column-major for better cache locality");
    println!("   - Performance difference shows impact of memory layout!");
    println!("\n4. Run comprehensive benchmarks:");
    println!("   cargo +nightly bench");
}

fn test_simple() {
    let a = Matrix::from_data_row_major(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = Matrix::from_data_row_major(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
    
    let naive_result = naive_matmul(&a, &b);

    // Expected: [[19,22],[43,50]]
    let expected = vec![19.0, 22.0, 43.0, 50.0];
    
    let naive_correct = (0..4).all(|i| (naive_result.get(i/2, i%2) - expected[i]).abs() < 1e-10);

    println!("  naive_matmul: {}", if naive_correct { "✓" } else { "✗" });

    // Test state tracking
    println!("  Matrix A is row-major: {}", a.is_row_major());
    let b_col = b.to_col_major();
    println!("  Matrix B converted to col-major: {}", b_col.is_col_major());
}

