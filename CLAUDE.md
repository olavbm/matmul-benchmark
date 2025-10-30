# Matrix Multiplication Performance Project

Educational Rust project exploring CPU cache optimization and SIMD acceleration for matrix multiplication.

## Quick Start

```bash
cargo run --release                              # Interactive demo
cargo run --release -- --scaling 1               # Performance across sizes
cargo +nightly bench vector_dotprod              # Dot product benchmarks
cargo +nightly bench mm                          # Matrix multiplication benchmarks
cargo +nightly bench simd                        # SIMD implementations
```

## Performance Highlights

**Best Results** (vs naive baseline):
- **1024×1024**: 5.2x speedup (9s → 1.7s)
- **512×512**: 3.9x speedup (841ms → 77ms)
- **Dot product (1024 elem)**: 271ns (beats nalgebra's 288ns!)

**Key Implementations**:
- `naive_matmul` - O(n³) baseline (~1.2 GFLOP/s)
- `blocked_matmul` - 64×64 cache blocking for L1d (37KB cache)
- `simd_matmul` - AVX2 vectorization + blocking (best performance)

## Hardware Optimizations

**Target System**: L1d=37KB, L2=1.5MB, L3=18MB
- **64×64 blocks**: Optimal for L1d (32KB fits in 37KB)
- **Cache cliffs**: Performance drops at 512×512 (L2→L3 boundary)
- **SIMD**: AVX2 processes 4×f64 in parallel

## Project Structure

```
src/
├── implementations.rs  # naive → blocked → SIMD progression
├── dotprod.rs         # naive → unrolled → SIMD dot products
├── matrix.rs          # Matrix with row/col-major state tracking
├── main.rs            # CLI with --scaling and --blocked modes
└── lib.rs             # Benchmarks (organized by category)

analysis/
├── Makefile           # make analyze, make quick, make blocked
└── analyze_stats.py   # Statistical analysis with plots

visualization/         # Interactive matrix multiplication demos
presentation.org       # Educational slideshow
```

## Key Insights

1. **Cache is everything**: Algorithm complexity < memory access patterns
2. **Small matrix overhead**: Optimization hurts perf for N ≤ 256
3. **Simple often wins**: Pure blocking beats complex hybrid approaches
4. **SIMD closed the gap**: 7x slower than BLAS (was 10x)

## GPU Exploration (arrayfire_demo/)

**Results** (Intel integrated GPU):
- 1024×1024: **2.63x speedup** (157 GFLOPS)
- Transfer overhead dominates for N < 1024
- Custom OpenCL kernels demonstrate GPU memory hierarchy

## Next Steps

**Low-hanging fruit**: FMA instructions, memory prefetching, cache line alignment
**Advanced**: Multi-level blocking (L1+L2+L3), thread parallelism, mixed precision

---

*See full module documentation in source files for implementation details.*
