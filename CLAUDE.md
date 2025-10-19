# Matrix Multiplication Performance Project - Technical Documentation

## Project Overview
A comprehensive Rust matrix multiplication library with cache optimization techniques, performance benchmarking, and statistical analysis tools.

## Current Implementation Status

### ✅ Completed Features

#### Matrix Multiplication Algorithms
1. **Naive Matrix Multiplication** (`naive_matmul`)
   - Standard O(n³) implementation
   - Baseline performance reference

2. **Dot Product Optimizations**
   - `naive_dotprod`: Basic scalar implementation
   - `unrolled_dotprod`: 4x loop unrolling for better performance
   - `simd_dotprod`: AVX2 SIMD vectorization (4x f64 parallelism)
   - Pluggable dot product architecture

3. **Cache Optimizations**
   - `dotprod_matmul_col_major_fast`: Column-major conversion for cache locality
   - Zero-allocation optimizations using buffer reuse
   - State tracking (RowMajor/ColMajor) in Matrix struct

4. **🆕 Cache-Blocked Matrix Multiplication** (Major Addition)
   - `blocked_matmul`: Configurable block size implementation
   - `blocked_matmul_default`: 64×64 blocks optimized for L1d cache (37KB)
   - `blocked_matmul_optimized`: Hybrid approach with dotprod + column-major
   - Six-nested-loop algorithm for optimal cache utilization

5. **🆕 SIMD-Accelerated Matrix Multiplication** (Latest Addition)
   - `simd_matmul`: SIMD + 64×64 blocking for optimal performance
   - `simd_blocked_matmul_optimized`: SIMD + adaptive blocking
   - `simd_blocked_matmul_32/128`: SIMD with different block sizes
   - AVX2 vectorization achieving 4x dot product speedup

#### Performance Analysis Infrastructure
1. **Comprehensive Benchmarking**
   - Single-trial scaling benchmark (`--scaling`)
   - Statistical multi-trial analysis (20 trials default)
   - Block size comparison benchmark (`--blocked`)
   - Full `cargo +nightly bench` integration
   - **🆕 Organized Benchmark Categories**:
     - `vector_dotprod`: Pure dot product performance comparison
     - `mm`: Matrix multiplication algorithm benchmarks
     - `simd`: SIMD-accelerated implementation benchmarks

2. **Statistical Analysis Tools**
   - `analyze_stats.py`: Comprehensive statistical analysis
   - Performance distribution plots with error bars
   - Coefficient of variation analysis
   - Statistical significance testing (95% confidence intervals)

3. **Visualization Pipeline**
   - Distribution plots showing performance variability
   - Cache boundary effect visualization
   - Speedup analysis with confidence intervals
   - gnuplot integration for quick plotting

4. **🆕 Automated Makefile Workflow**
   - `make analyze`: Full 20-trial statistical analysis
   - `make quick`: Fast 10-trial analysis
   - `make simple`: Single-trial with gnuplot
   - `make blocked`: Block size optimization testing
   - Automatic plot generation and viewing

## Hardware-Specific Optimizations

### Target System Cache Hierarchy
- **L1d**: 448 KiB total (37.3 KiB per core)
- **L2**: 9 MiB total (1.5 MiB per core)
- **L3**: 18 MiB shared

### Cache Boundary Analysis
- **16×16-64×64**: Fits in L1d cache → Peak performance
- **96×96-384×384**: L2 cache → Good performance, optimization crossover point
- **512×512+**: L3/RAM bound → Dramatic optimization benefits

### Optimal Block Sizes Discovered
- **64×64 blocks**: Optimal for L1d cache (32KB fits in 37KB L1d)
- **32×32 blocks**: Better for very large matrices (1024×1024)
- **128×128 blocks**: Too large, causes L1d cache misses

## Performance Results

### Current Best Performance (vs Naive Baseline)
- **256×256**: **8.9ms (simd_matmul)** vs 13.4ms (previous best) - **1.7x SIMD improvement**
- **512×512**: **76.8ms (simd_matmul)** vs ~120ms (previous best) - **1.6x SIMD improvement**
- **Previous benchmarks**: 1.28 GFLOP/s (blocked, 128×128 blocks) vs 0.41 GFLOP/s (naive)

### Speedup Achievements
- **5.2x speedup** for 1024×1024 matrices
- **3.9x speedup** for 512×512 matrices
- **3.1x speedup** for 768×768 matrices

### BLAS Comparison (128×128)
- **BLAS**: 247ms (highly optimized baseline)
- **Our SIMD Implementation**: 1,734ms (7x gap, down from 10x)
- **Previous Best**: 2,047ms (8.3x gap)
- **Gap Analysis**: SIMD closed significant performance gap

### Dot Product Performance Comparison (1024 elements)
- **Our SIMD**: 271ns 
- **nalgebra**: 288ns (**We're faster!**)
- **Unrolled**: 485ns 
- **Naive**: 1,279ns

## Technical Implementation Details

### Matrix Data Structure
```rust
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub state: MatrixState, // RowMajor(Vec<f64>) | ColMajor(Vec<f64>)
}
```
- State tracking for memory layout optimization
- Conversion methods between row-major and column-major
- Buffer extraction for efficient dot product computation

### SIMD Dot Product Implementation
```rust
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
unsafe fn simd_dotprod_avx2(a: &[f64], b: &[f64]) -> f64 {
    use std::arch::x86_64::*;
    let mut sum_vec = _mm256_setzero_pd();
    
    // Process 4 f64 elements at a time with AVX2
    for i in (0..simd_len).step_by(4) {
        let a_vec = _mm256_loadu_pd(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_pd(b.as_ptr().add(i));
        let prod = _mm256_mul_pd(a_vec, b_vec);
        sum_vec = _mm256_add_pd(sum_vec, prod);
    }
    // Sum the 4 elements + handle remainder...
}
```

### Blocked Algorithm Architecture
```rust
pub fn simd_matmul(a: &Matrix, b: &Matrix) -> Matrix {
    // Combines SIMD vectorization with cache-friendly blocking
    blocked_matmul_col_major_fast(a, b, 64, simd_dotprod)
}
```

### Statistical Analysis Capabilities
- **Multi-trial benchmarking**: 20 trials default, configurable
- **Performance variability analysis**: Coefficient of variation (CV%)
- **Statistical significance**: 95% confidence intervals
- **Cache boundary detection**: Automated analysis of performance cliffs
- **Distribution visualization**: Box plots, error bars, histograms

## Benchmark Integration

### Available Benchmark Modes
1. **Categorized Benchmarks**: Clean separation for targeted performance analysis
   ```bash
   cargo +nightly bench vector_dotprod  # Pure dot product performance (11 benchmarks)
   cargo +nightly bench mm              # Matrix multiplication algorithms (12 benchmarks)  
   cargo +nightly bench simd            # SIMD-accelerated implementations
   ```

2. **Standard Benchmarks**: `cargo +nightly bench`
   - All algorithms with multiple matrix sizes
   - Block size variations (32×32, 64×64, 128×128)
   - SIMD variants with different blocking strategies

3. **Scaling Analysis**: `cargo run --release -- --scaling [trials]`
   - Statistical performance across matrix sizes 16×16 to 1024×1024
   - CSV output for analysis tools

4. **Block Size Optimization**: `cargo run --release -- --blocked`
   - Focused comparison on large matrices (256×256 to 1024×1024)
   - Block size sweep analysis

### Makefile Workflow Integration
```bash
make analyze TRIALS=20  # Full statistical analysis
make quick               # Fast 10-trial analysis  
make simple             # Single-trial + gnuplot
make blocked            # Block optimization test
make view               # Open generated plots
```

## Key Findings & Insights

### Cache Optimization Principles
1. **Small Matrix Overhead**: Optimization hurts performance for matrices ≤ 256×256
2. **Cache Boundary Effects**: Dramatic performance drops at L1d, L2, L3 boundaries
3. **Block Size Scaling**: Optimal block size decreases with matrix size
4. **Memory Layout Impact**: Column-major conversion critical for large matrices

### Performance Variability Analysis
- **Large matrices show low CV%** (0.6-1.2%): Consistent cache miss patterns
- **Small matrices show high CV%** (20-30%): Dominated by system noise
- **Statistical significance**: All performance differences validated at 95% confidence

### Algorithmic Trade-offs
- **Simple blocking often beats hybrid approaches**: Less complexity can mean better performance
- **Buffer reuse vs allocation**: Zero-allocation versions show dramatic improvements
- **Dot product abstraction**: Enables easy performance comparisons and optimizations

## GPU Acceleration Exploration

### ArrayFire Integration
- **Library**: ArrayFire 3.8 with OpenCL backend
- **Build**: Custom build from source with Intel oneAPI MKL 2025.2
- **Backend switching**: Dynamic CPU (MKL) vs OpenCL (GPU) selection

### GPU Performance Results (Intel Integrated GPU)
Matrix multiplication performance comparison:

| Matrix Size | CPU (MKL) | GPU (OpenCL) | Speedup |
|------------|-----------|--------------|---------|
| 256×256    | 0.61ms    | 2.47ms       | 0.25x   |
| 512×512    | 4.39ms    | 5.41ms       | 0.81x   |
| 1024×1024  | 36.11ms   | 13.71ms      | **2.63x** |

**Peak Performance**: 157 GFLOPS on 1024×1024 matrices

### Key GPU Insights
1. **Transfer overhead dominates small matrices**: GPU slower for N < 1024 due to PCIe/memory transfer
2. **Integrated GPU advantage**: Shares system RAM, reducing transfer overhead vs discrete GPUs
3. **Scaling characteristics**: GPU advantage increases with matrix size
4. **CPU multi-threading**: ArrayFire's CPU backend uses all cores, competitive with GPU for medium sizes

### Custom OpenCL Kernels
Implemented educational examples showing raw GPU programming:

**Naive Kernel** (`custom_matmul.rs`):
```c
__kernel void matmul_naive(__global const float* A,
                           __global const float* B,
                           __global float* C, const int N)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}
```

**Cache-Blocked Kernel** (`custom_matmul_blocked.rs`):
- Uses GPU local memory (16×16 tiles) analogous to CPU L1 cache
- Demonstrates GPU memory hierarchy optimization
- Reduces global memory accesses through shared memory blocking

### GPU vs CPU Architecture Comparison
- **CPU**: 4-16 cores, deep cache hierarchy (L1/L2/L3), complex out-of-order execution
- **GPU**: Thousands of simple cores, memory hierarchy (registers/local/global), massive parallelism
- **Trade-off**: GPU excels at throughput, CPU excels at latency and sequential tasks

### Interactive Presentation
Created `presentation.org` for Doom Emacs org-tree-slide:
- Live code execution in Org-mode
- GPU acceleration section with performance tables
- Custom navigation (C-j/C-k for slide movement)
- Progressive reveal using outline folding

## Next Development Opportunities

### Immediate Improvements
1. **Multi-level cache blocking**: L1 + L2 + L3 hierarchical blocking
2. **~~SIMD vectorization~~**: ✅ **COMPLETED** - AVX2 integration achieved competitive performance
3. **Cache line optimization**: 64-byte cache line aware blocking
4. **Memory prefetching**: Explicit prefetch instructions in SIMD loops
5. **Fused Multiply-Add (FMA)**: Use FMA instructions for better throughput

### Advanced Features
1. **Parallel processing**: Thread-level parallelism for large matrices
2. **~~GPU acceleration~~**: ✅ **EXPLORED** - ArrayFire integration with custom OpenCL kernels
3. **Mixed precision**: f16/f32/f64 optimization
4. **Specialized shapes**: Rectangular matrix optimizations

## Development Commands Reference

### Build & Test
```bash
cargo build --release          # Optimized build
cargo test blocked             # Test blocked algorithms
cargo +nightly bench          # Full benchmark suite
```

### Performance Analysis
```bash
make analyze                   # Statistical analysis with plots
make blocked                   # Block size optimization
make view                      # View generated plots
make clean                     # Clean generated files
```

### Data Generation
```bash
cargo run --release -- --scaling 50 > data.txt  # 50-trial statistical data
python3 analyze_stats.py data.txt                # Comprehensive analysis
```

### GPU Benchmarking (ArrayFire)
```bash
# Run GPU vs CPU comparison
cd arrayfire_demo
env LD_LIBRARY_PATH=/home/olav/dev/arrayfire/build/src/api/unified:/home/olav/dev/arrayfire/build/src/backend/opencl:/home/olav/dev/arrayfire/build/src/backend/cpu:/opt/intel/oneapi/compiler/2025.2/lib \
    cargo run --release --example simple_comparison

# Educational custom kernels
cargo run --release --example custom_matmul         # Naive kernel
cargo run --release --example custom_matmul_blocked # Cache-blocked kernel
```

## File Structure
```
src/
├── lib.rs              # Library entry point + benchmarks
├── main.rs             # Interactive benchmarking + CLI
├── matrix.rs           # Matrix data structure + state tracking
├── implementations.rs  # All matrix multiplication algorithms
├── dotprod.rs         # Dot product implementations
├── benchmark.rs       # Performance measurement utilities
└── test_data.rs       # Test matrix generation

analysis/
├── analyze_stats.py   # Statistical analysis tool
├── scaling.plt        # gnuplot script
└── Makefile          # Automated workflow

arrayfire_demo/        # GPU exploration with ArrayFire
├── Cargo.toml         # Dependencies: arrayfire, af-opencl-interop
├── src/main.rs        # ArrayFire demo entry point
└── examples/
    ├── simple_comparison.rs      # GPU vs CPU benchmark (working)
    ├── custom_matmul.rs          # Naive OpenCL kernel (educational)
    └── custom_matmul_blocked.rs  # Cache-blocked OpenCL kernel

presentation.org       # Interactive Org-mode slideshow
```

---

**Last Updated**: GPU acceleration exploration with ArrayFire and custom OpenCL kernels:
- **GPU exploration**: Built ArrayFire from source with Intel oneAPI MKL 2025.2
- **Performance comparison**: GPU achieves 157 GFLOPS (2.63x speedup on 1024×1024)
- **Custom kernels**: Implemented naive and cache-blocked OpenCL kernels for educational purposes
- **Architecture insights**: GPU excels for large matrices (N ≥ 1024), CPU dominates for smaller sizes due to transfer overhead
- **Interactive presentation**: Created Org-mode slideshow with live code execution in Doom Emacs
- **Complete performance spectrum**: CPU naive → CPU SIMD → CPU MKL → GPU OpenCL documented

**Previous Major Updates**:
- **SIMD implementation**: AVX2 vectorization achieving 1.7x speedup (271ns dot product vs nalgebra's 288ns)
- **Cache blocking**: 64×64 blocks optimized for L1d cache (5.2x speedup on 1024×1024)
- **Statistical analysis**: Multi-trial benchmarking with confidence intervals and visualization