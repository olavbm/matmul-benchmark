# Matrix Multiplication Benchmark

Educational Rust project exploring CPU cache optimization, SIMD acceleration, and GPU computing for matrix multiplication.

## Quick Start

```bash
cargo run --release                    # Interactive demo
cargo run --release -- --scaling 1     # Performance across sizes
cargo +nightly bench                   # Full benchmark suite
```

## Performance Highlights

**17.5x speedup** (512×512 matrices, vs naive baseline) through:
- Cache blocking (64×64 blocks for L1d cache)
- SIMD vectorization (AVX2 - 4×f64 parallel)
- Memory layout optimization (column-major conversion)
- Profiling-driven bug fixes (eliminated 49% overhead)

**Still 3× slower than nalgebra** - but our dot product beats theirs: 271ns vs 288ns ⚡

## What's Inside

**Implementations** (naive → optimized progression):
- `naive_matmul` - O(n³) baseline (~1.2 GFLOP/s)
- `blocked_matmul` - Cache-friendly 64×64 blocking
- `simd_matmul` - AVX2 + blocking (best CPU performance)

**Benchmarks** (organized by category):
```bash
cargo +nightly bench vector_dotprod    # Dot product comparison
cargo +nightly bench mm                # Matrix multiplication
cargo +nightly bench simd              # SIMD variants
```

**Analysis tools**:
- `make analyze` - Statistical analysis with plots
- `--scaling` mode - Performance across matrix sizes
- `--blocked` mode - Block size optimization

## Performance Results

**512×512 matrices** (criterion benchmarks):

| Implementation | Time/Iteration | L1 Miss Rate | Speedup |
|----------------|----------------|--------------|---------|
| Naive          | ~570ms         | 48.5%        | 1.0×    |
| Blocked        | ~150ms         | 45.8%        | 3.7×    |
| SIMD           | ~75ms          | 8.5%         | 7.7×    |
| **Optimized**  | **~32ms**      | **2.2%**     | **17.5×** |
| nalgebra       | ~10ms          | 2.8%         | 54×     |

**GPU exploration** (arrayfire_demo/): 157 GFLOPS on Intel iGPU (2.63× vs CPU)

## Key Insights

1. **Cache is everything** - Memory access patterns matter more than algorithmic complexity
2. **Performance cliffs** - Dramatic drops at L2/L3 cache boundaries (512×512)
3. **Small matrix overhead** - Optimization hurts performance for N ≤ 256
4. **SIMD impact** - Closed gap from 10× to 7× vs hand-tuned BLAS

## Project Structure

```
src/
├── implementations.rs  # naive → blocked → SIMD progression
├── dotprod.rs         # Dot product optimizations
├── matrix.rs          # Matrix with state tracking
├── main.rs            # CLI with benchmarking modes
└── lib.rs             # Organized benchmarks

analysis/              # Statistical analysis tools
visualization/         # Interactive demos
presentation.org       # Educational slideshow
```

## Learning Resources

- **Source documentation**: Run `cargo doc --open` for detailed rustdoc
- **CLAUDE.md**: Development guide and technical notes
- **presentation.org**: Interactive slideshow with live code execution

## License

MIT OR Apache-2.0
