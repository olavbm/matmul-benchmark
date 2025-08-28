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

### Matrix Multiplication
- **`naive_matmul`**: Standard O(n³) matrix multiplication with row-major access
- **`dotprod_matmul`**: Abstracted matrix multiplication using pluggable dot product functions
- **`dotprod_matmul_fast`**: Zero-allocation version that reuses buffers for better performance
- **`dotprod_matmul_col_major_fast`**: Combines column-major optimization with zero allocations
- **Cache optimization**: Converts matrix B to column-major layout for better cache locality
- **State tracking**: Matrix struct tracks whether data is stored in row-major or column-major format

### Dot Product
- **`naive_dotprod`**: Basic scalar implementation of dot product
- **`unrolled_dotprod`**: Manual loop unrolling (4x) for improved performance

## 💡 How It Works

### Matrix Multiplication
```rust
// Standard approach - may cause cache misses
let result = naive_matmul(&a, &b);

// Optimized approach - convert B to column-major first  
let b_col_major = b.to_col_major();
let result = naive_matmul(&a, &b_col_major);

// Dot product abstracted approach - choose your dot product implementation
let result1 = dotprod_matmul(&a, &b, naive_dotprod);
let result2 = dotprod_matmul(&a, &b, unrolled_dotprod);

// Zero-allocation fast version (reuses buffers)
let result3 = dotprod_matmul_fast(&a, &b, naive_dotprod);
let result4 = dotprod_matmul_fast(&a, &b, unrolled_dotprod);

// Ultimate optimization: column-major + zero allocations + unrolled dot product
let result5 = dotprod_matmul_col_major_fast(&a, &b, unrolled_dotprod);
```

The optimization works because:
- Matrix A is accessed row-wise (sequential)
- Matrix B is accessed column-wise (non-sequential → cache misses)
- Converting B to column-major makes B access sequential too
- `dotprod_matmul` makes the connection explicit: each matrix element is a dot product of a row and column
- `dotprod_matmul_fast` eliminates heap allocations by reusing buffers (only 2 allocations vs 32,768 for 128×128)
- `dotprod_matmul_col_major_fast` combines all optimizations: cache-friendly access + zero allocations + pluggable dot products

### Dot Product
```rust
// Basic implementation
let result = naive_dotprod(&vec_a, &vec_b);

// Loop unrolled version for better performance
let result = unrolled_dotprod(&vec_a, &vec_b);
```

Loop unrolling benefits:
- Reduces loop overhead (fewer comparisons and jumps)
- Enables better instruction-level parallelism
- Allows the CPU to execute multiple operations simultaneously

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

### Matrix Multiplication
The benchmarks compare different approaches:
- **Original implementations**: `naive_matmul` vs cache-optimized versions
- **Dot product abstracted**: `dotprod_matmul` with `naive_dotprod` vs `unrolled_dotprod`
- **32×32**: ~18% improvement for cache optimization
- **64×64 & 128×128**: Shows direct impact of dot product optimizations on matrix multiplication

### Dot Product
Compares basic vs loop unrolled implementations:
- **256 elements**: Shows the effect of loop unrolling
- **1024 elements**: More pronounced performance difference

Performance is measured in:
- **Execution time** (nanoseconds/milliseconds)
- **Throughput** (GFLOP/s - Giga Floating-Point Operations per second)

## 🔬 Learning Points

### Matrix Multiplication
1. **Memory layout matters**: The same algorithm can perform very differently based on data layout
2. **Cache optimization trade-offs**: Conversion overhead vs. improved access patterns
3. **Matrix size scaling**: Optimizations become more effective with larger matrices
4. **Allocation overhead**: Zero-allocation versions dramatically outperform allocation-heavy approaches

### Dot Product
1. **Loop unrolling**: Reduces branch overhead and improves instruction-level parallelism
2. **CPU optimization**: Modern processors can execute multiple arithmetic operations per cycle
3. **Compiler vs manual optimization**: Understanding when manual optimization provides benefits
4. **Function abstraction**: Using function pointers/closures to make algorithms pluggable and testable

### General
1. **Benchmarking methodology**: Using `test::black_box()` to prevent compiler optimizations
2. **Performance measurement**: Importance of measuring actual performance vs. theoretical expectations

## 🚀 Next Steps: Cache Blocking

The next major optimization to implement is **cache blocking** (also called "tiling"), which should dramatically improve performance by better utilizing the CPU cache hierarchy.

### Current Performance Gap
- **BLAS**: ~241μs (highly optimized)
- **Our best**: ~2,192μs (`dotprod_matmul_col_major_fast` with `unrolled_dotprod`)
- **Gap**: ~9x slower than BLAS

### Cache Blocking Strategy

Instead of computing each result element completely in one pass, cache blocking processes the matrices in small blocks that fit in L1 cache:

```rust
// Traditional: C[i,j] = sum of A[i,k] * B[k,j] for k=0..n (all at once)
// Blocked: C[i,j] += sum of A[i,k] * B[k,j] for k in each block (partial sums)
```

### Target Cache Sizes (Intel i7-1260P)
- **L1d cache**: 37.3 KiB per core
- **Optimal block sizes**: 24×24 (4.6KB), 32×32 (8.2KB), or 48×48 (18.4KB)
- **Strategy**: Keep 2-3 blocks in L1 simultaneously (A block, B block, result block)

### Implementation Plan
1. **`blocked_matmul`**: 6-nested-loop cache-blocked algorithm
2. **Block size optimization**: Test 24×24, 32×32, 48×48 to find optimal size
3. **Hybrid approach**: Combine blocking with existing dot product optimizations
4. **Expected improvement**: Should get much closer to BLAS performance

### The Algorithm (Conceptual)
```rust
for kk in (0..n).step_by(BLOCK_SIZE) {      // Which "slice" of dot product
    for ii in (0..n).step_by(BLOCK_SIZE) {  // Which row block of result
        for jj in (0..n).step_by(BLOCK_SIZE) { // Which col block of result
            // Process BLOCK_SIZE × BLOCK_SIZE sub-matrices
            // This keeps data in L1 cache much longer
        }
    }
}
```

This approach should bridge much of the performance gap to BLAS by dramatically improving cache utilization.