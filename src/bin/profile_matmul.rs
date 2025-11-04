use matmul::{blocked_matmul, naive_matmul, simd_matmul, Matrix};
use nalgebra::DMatrix;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <algorithm> <size>", args[0]);
        eprintln!("Algorithms: naive, blocked, simd, fma, nalgebra");
        std::process::exit(1);
    }

    let algorithm = &args[1];
    let size: usize = args[2].parse().expect("Size must be a number");

    println!("Profiling {} with {}×{} matrices", algorithm, size, size);

    // Run the selected algorithm multiple times to get stable perf data
    let iterations = if size <= 256 {
        100
    } else if size <= 512 {
        20
    } else {
        5
    };

    println!("Running {} iterations...", iterations);

    if algorithm == "nalgebra" {
        // Create nalgebra DMatrix directly for nalgebra implementation
        let mut rng = rand::thread_rng();
        let a = DMatrix::from_fn(size, size, |_, _| rand::Rng::gen::<f64>(&mut rng));
        let b = DMatrix::from_fn(size, size, |_, _| rand::Rng::gen::<f64>(&mut rng));

        for i in 0..iterations {
            let _result = &a * &b;

            if (i + 1) % 10 == 0 {
                println!("  Completed {} iterations", i + 1);
            }
        }
    } else {
        // Create test matrices for our implementations
        let a = Matrix::random(size, size);
        let b = Matrix::random(size, size);

        for i in 0..iterations {
            let _result = match algorithm.as_str() {
                "naive" => naive_matmul(&a, &b),
                "blocked" => blocked_matmul(&a, &b, 64),
                "simd" => simd_matmul(&a, &b),
                _ => {
                    eprintln!("Unknown algorithm: {}", algorithm);
                    std::process::exit(1);
                }
            };

            if (i + 1) % 10 == 0 {
                println!("  Completed {} iterations", i + 1);
            }
        }
    }

    println!("Done!");
}
