# Matrix Multiplication Benchmark

A Rust project demonstrating matrix multiplication with cache optimization techniques and performance benchmarking.

## 🎯 What This Demonstrates

- **Cache-aware algorithms**: Comparing row-major vs column-major memory access patterns
- **Memory layout optimization**: How data structure layout affects performance
- **Performance measurement**: Benchmarking with GFLOP/s calculations
- **Rust benchmarking**: Using built-in `#[bench]` attributes

## 🚀 Quick Start

1. **Run the interactive benchmark**:
   ```bash
   cargo run --release
   ```

2. **Run comprehensive benchmarks**:
   ```bash
   cargo +nightly bench
   ```

## 📁 Project Structure

```
src/
├── lib.rs              # Library entry point with benchmarks
├── main.rs             # Interactive benchmarking application  
├── matrix.rs           # Matrix data structure with RowMajor/ColMajor states
├── implementations.rs  # Matrix multiplication implementation
├── test_data.rs       # Test matrix generation
└── benchmark.rs       # Performance measurement helpers
```

## 🧪 Current Implementation

- **`naive_matmul`**: Standard O(n³) matrix multiplication with row-major access
- **Cache optimization**: Converts matrix B to column-major layout for better cache locality
- **State tracking**: Matrix struct tracks whether data is stored in row-major or column-major format

## 💡 How It Works

```rust
// Standard approach - may cause cache misses
let result = naive_matmul(&a, &b);

// Optimized approach - convert B to column-major first  
let b_col_major = b.to_col_major();
let result = naive_matmul(&a, &b_col_major);
```

The optimization works because:
- Matrix A is accessed row-wise (sequential)
- Matrix B is accessed column-wise (non-sequential → cache misses)
- Converting B to column-major makes B access sequential too

## 🔧 Commands

```bash
# Interactive benchmark with correctness testing
cargo run --release

# Comprehensive benchmarks (requires nightly)
cargo +nightly bench

# Run specific benchmark
cargo +nightly bench bench_naive_128x128

# Run tests
cargo test
```

## 📈 Benchmark Results

The benchmarks compare naive vs optimized implementations across different matrix sizes:
- **32×32**: ~18% improvement  
- **64×64**: Minimal difference (conversion overhead)
- **128×128**: ~15% improvement
- **256×256 and larger**: Larger improvements expected

Performance is measured in:
- **Execution time** (nanoseconds/milliseconds)
- **Throughput** (GFLOP/s - Giga Floating-Point Operations per second)

## 🔬 Learning Points

1. **Memory layout matters**: The same algorithm can perform very differently based on data layout
2. **Cache optimization trade-offs**: Conversion overhead vs. improved access patterns
3. **Benchmarking methodology**: Using `test::black_box()` to prevent compiler optimizations
4. **Matrix size scaling**: Optimizations become more effective with larger matrices