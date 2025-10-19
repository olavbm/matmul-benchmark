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

## ✅ Implemented Optimizations

The project has successfully implemented:
- ✅ **Cache blocking**: 64×64 blocks optimized for L1d cache (5.2x speedup on 1024×1024)
- ✅ **SIMD vectorization**: AVX2 achieving 1.7x speedup (271ns dot product vs nalgebra's 288ns)
- ✅ **GPU exploration**: ArrayFire with 157 GFLOPS on Intel iGPU (2.63x speedup on 1024×1024)

### Current Performance (1024×1024)
- **Naive CPU**: 5,812ms
- **SIMD CPU**: 1,632ms (3.6x faster)
- **BLAS (MKL)**: 7.81ms (743x faster than naive!)
- **GPU (OpenCL)**: 13.71ms (424x faster than naive)

## 🚀 Future CPU Optimizations

While we've achieved strong performance with SIMD and cache blocking, there's still a **7x gap** between our handwritten SIMD code (1,734ms for 128×128) and BLAS (247ms). Here are the optimizations that could close that gap:

### High-Impact Optimizations

#### 1. **Multi-Threading with Rayon** ⭐ Easiest, biggest win
Currently single-threaded. Parallel processing is almost free performance:

```rust
use rayon::prelude::*;

// Parallelize outer block loop
(0..a.rows).into_par_iter().step_by(BLOCK_SIZE)
    .for_each(|i| {
        // Compute blocks in parallel
    });
```

**Expected speedup**: 6-8× on 12-core system
**Complexity**: Low (just add Rayon)

#### 2. **Fused Multiply-Add (FMA) Instructions** ⭐ Better SIMD throughput
Current SIMD uses separate multiply and add. FMA does both in one instruction:

```rust
// Current: _mm256_add_pd(sum_vec, _mm256_mul_pd(a_vec, b_vec))
// FMA:     _mm256_fmadd_pd(a_vec, b_vec, sum_vec)
```

**Expected speedup**: ~2× throughput for core computation
**Complexity**: Medium (replace AVX2 with FMA intrinsics)
**Bonus**: Better numerical accuracy

#### 3. **Multi-Level Cache Blocking** ⭐ Better large matrix performance
Current blocking only targets L1 cache (64×64). Add L2/L3 levels:

```rust
// Two-level blocking: L2 blocks (256×256) → L1 blocks (64×64)
pub fn multi_level_blocked_matmul(a: &Matrix, b: &Matrix) -> Matrix {
    const L2_BLOCK: usize = 256;  // Fits in L2 (1.5 MB)
    const L1_BLOCK: usize = 64;   // Fits in L1 (37 KB)

    // Outer loop over L2_BLOCK tiles
    // Inner loop over L1_BLOCK sub-tiles within each L2 tile
}
```

**Expected speedup**: 1.2-1.5× for matrices > 512×512
**Complexity**: Medium (hierarchical blocking logic)

#### 4. **Register Blocking / Micro-Kernel Optimization** ⭐ Maximum performance
Compute multiple output elements per inner loop to maximize register reuse:

```rust
// Compute 4×4 output block per iteration instead of 1×1
// Keeps 4 rows of A and 4 cols of B in registers
// Reduces memory traffic by 4×
```

This is what BLAS libraries (BLIS, OpenBLAS) do to approach peak FLOPS.

**Expected speedup**: 1.5-2×
**Complexity**: High (most advanced optimization)

### Advanced Optimizations

#### 5. **Memory Prefetching**
Explicitly prefetch data before it's needed:

```rust
#[cfg(target_arch = "x86_64")]
unsafe {
    use std::arch::x86_64::_mm_prefetch;
    _mm_prefetch(ptr.add(cache_line_offset) as *const i8, _MM_HINT_T0);
}
```

**Expected speedup**: 1.1-1.2×
**Complexity**: Medium (requires careful tuning)

#### 6. **Non-Temporal Stores**
For very large matrices, bypass cache when writing results:

```rust
// Use _mm256_stream_pd instead of _mm256_storeu_pd for output
// Prevents cache pollution for write-only data
```

**Expected speedup**: 1.1-1.3× for very large matrices
**Complexity**: Low (simple API change)

#### 7. **Wider SIMD (AVX-512)**
Current AVX2 processes 4×f64. AVX-512 can do 8×f64:

```rust
// AVX2: 4×f64 per instruction
// AVX-512: 8×f64 per instruction (if CPU supports it)
```

**Expected speedup**: ~1.5-2× (if hardware supports AVX-512)
**Complexity**: Medium (check CPU support, update intrinsics)

#### 8. **Algorithm Changes**

- **Strassen's Algorithm**: O(n^2.807) instead of O(n³) for very large matrices
- **Winograd Variant**: Reduces multiplications at cost of more additions
- **Panel-Panel Multiply**: How modern BLAS libraries organize computation

**Expected speedup**: Varies (Strassen good for N > 2048)
**Complexity**: High (fundamentally different algorithm)

### Roadmap to Close the Gap

Current gap analysis (128×128):
- **Our SIMD**: 1,734ms
- **BLAS**: 247ms (7× faster)

The gap comes from:
1. **No multi-threading**: 4-8× potential speedup
2. **No FMA**: ~2× potential speedup
3. **No register blocking**: 1.5-2× potential speedup
4. **No L2/L3 blocking**: 1.2-1.5× potential for large matrices

**Realistic target**: Implementing **threading + FMA** alone could close the gap to ~2× vs BLAS, which would be excellent for educational handwritten code!

### Recommended Next Step

**Start with multi-threading** - it's the easiest to implement and gives the biggest performance boost. Adding Rayon parallelism to the existing blocked SIMD code would immediately get you 6-8× faster with minimal code changes.