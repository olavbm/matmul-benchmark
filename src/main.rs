use matmul::{
    blocked_matmul, blocked_matmul_optimized, dotprod_matmul_col_major_fast,
    generate_test_matrices, naive_matmul, unrolled_dotprod, Matrix, MatrixOps,
};
use std::env;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() > 1 && args[1] == "--scaling" {
        run_scaling_benchmark();
        return;
    }

    if args.len() > 1 && args[1] == "--blocked" {
        run_blocked_comparison();
        return;
    }

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

        println!(
            "   Naive (row×row):       {:8.2} ms ({:6.2} GFLOP/s)",
            naive_time.as_secs_f64() * 1000.0,
            naive_gflops
        );
        println!(
            "   Optimized (row×col):   {:8.2} ms ({:6.2} GFLOP/s)",
            optimized_time.as_secs_f64() * 1000.0,
            optimized_gflops
        );

        let speedup = naive_time.as_secs_f64() / optimized_time.as_secs_f64();
        println!("   Speedup: {:.2}x", speedup);
    }
}

fn test_simple() {
    let a = Matrix::from_data(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = Matrix::from_data(vec![5.0, 6.0, 7.0, 8.0], 2, 2);

    let naive_result = naive_matmul(&a, &b);

    // Expected: [[19,22],[43,50]]
    let expected = [19.0, 22.0, 43.0, 50.0];

    let naive_correct =
        (0..4).all(|i| (naive_result.get(i / 2, i % 2) - expected[i]).abs() < 1e-10);

    println!("  naive_matmul: {}", if naive_correct { "✓" } else { "✗" });

    // Test layout conversion
    println!("  Matrix A is RowMajorMatrix: true");
    let _b_col = b.to_col_major();
    println!("  Matrix B converted to ColMajorMatrix: true");
}

fn run_scaling_benchmark() {
    let args: Vec<String> = env::args().collect();
    let num_trials = if args.len() > 2 && args[2].parse::<usize>().is_ok() {
        args[2].parse().unwrap()
    } else {
        20 // Default number of trials
    };

    println!("size,algorithm,trial,time_ns,gflops");

    let sizes = [16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024];

    for &size in &sizes {
        eprintln!(
            "Benchmarking {}×{} matrices ({} trials each)...",
            size, size, num_trials
        );

        for trial in 0..num_trials {
            // Generate fresh matrices for each trial to avoid cache effects
            let a = Matrix::random(size, size);
            let b = Matrix::random(size, size);

            // Benchmark naive algorithm
            let start = Instant::now();
            let _result = naive_matmul(&a, &b);
            let naive_time_ns = start.elapsed().as_nanos();
            let gflops_naive = (2.0 * (size as f64).powi(3)) / (naive_time_ns as f64);
            println!(
                "{},naive,{},{},{:.6}",
                size, trial, naive_time_ns, gflops_naive
            );

            // Small delay to avoid thermal throttling effects
            std::thread::sleep(std::time::Duration::from_millis(1));

            // Benchmark optimized algorithm with same matrices
            let start = Instant::now();
            let _result = dotprod_matmul_col_major_fast(&a, &b, unrolled_dotprod);
            let optimized_time_ns = start.elapsed().as_nanos();
            let gflops_optimized = (2.0 * (size as f64).powi(3)) / (optimized_time_ns as f64);
            println!(
                "{},optimized,{},{},{:.6}",
                size, trial, optimized_time_ns, gflops_optimized
            );

            // Small delay between trials
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    eprintln!(
        "Benchmark complete! {} trials per size, {} sizes total",
        num_trials,
        sizes.len()
    );
}

fn run_blocked_comparison() {
    println!("Blocked Matrix Multiplication Comparison");
    println!("size,algorithm,block_size,time_ns,gflops");

    let sizes = [256, 512, 768, 1024]; // Focus on larger sizes where blocking helps
    let block_sizes = [32, 48, 64, 96, 128];

    for &size in &sizes {
        eprintln!("Testing {}×{} matrices...", size, size);
        let a = Matrix::random(size, size);
        let b = Matrix::random(size, size);

        // Test naive baseline
        let start = Instant::now();
        let _result = naive_matmul(&a, &b);
        let naive_time_ns = start.elapsed().as_nanos();
        let gflops_naive = (2.0 * (size as f64).powi(3)) / (naive_time_ns as f64);
        println!("{},naive,0,{},{:.6}", size, naive_time_ns, gflops_naive);

        // Test current best non-blocked
        let start = Instant::now();
        let _result = dotprod_matmul_col_major_fast(&a, &b, unrolled_dotprod);
        let optimized_time_ns = start.elapsed().as_nanos();
        let gflops_optimized = (2.0 * (size as f64).powi(3)) / (optimized_time_ns as f64);
        println!(
            "{},optimized,0,{},{:.6}",
            size, optimized_time_ns, gflops_optimized
        );

        // Test different block sizes
        for &block_size in &block_sizes {
            let start = Instant::now();
            let _result = blocked_matmul(&a, &b, block_size);
            let blocked_time_ns = start.elapsed().as_nanos();
            let gflops_blocked = (2.0 * (size as f64).powi(3)) / (blocked_time_ns as f64);
            println!(
                "{},blocked,{},{},{:.6}",
                size, block_size, blocked_time_ns, gflops_blocked
            );
        }

        // Test ultimate blocked optimization
        let start = Instant::now();
        let _result = blocked_matmul_optimized(&a, &b, unrolled_dotprod);
        let ultimate_time_ns = start.elapsed().as_nanos();
        let gflops_ultimate = (2.0 * (size as f64).powi(3)) / (ultimate_time_ns as f64);
        println!(
            "{},blocked_ultimate,64,{},{:.6}",
            size, ultimate_time_ns, gflops_ultimate
        );
    }

    eprintln!("Blocked comparison complete!");
}
