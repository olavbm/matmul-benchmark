# ArrayFire GPU Matrix Multiplication Demo

GPU matrix multiplication experiments using ArrayFire with OpenCL backend and custom OpenCL kernels.

## Hardware

- **GPU**: Intel(R) Graphics [0x46a6] (integrated GPU)
- **Backend**: OpenCL 3.0
- **ArrayFire**: v3.9.0

## Prerequisites

ArrayFire must be installed on your system:
- **Ubuntu/Debian**: `sudo apt install arrayfire`
- **macOS**: `brew install arrayfire`
- See [ArrayFire installation guide](https://arrayfire.org/docs/installing.htm) for other platforms

## Running Examples

### 1. Simple GPU Demo
Basic GPU matrix multiplication using ArrayFire:
```bash
cargo run --release --example simple_gpu
```

### 2. Simple CPU vs GPU Comparison
Compares ArrayFire CPU and GPU backends across different matrix sizes:
```bash
cargo run --release --example simple_comparison
```

**Sample output:**
```
Matrix Size  CPU (ms)        GPU (ms)        Speedup
-------------------------------------------------------
256×256      0.23            0.42            0.55      x
512×512      7.86            1.27            6.17      x
1024×1024    10.13           10.15           1.00      x

✓ GPU achieved 211.66 GFLOPS for 1024×1024
```

### 3. Custom Naive OpenCL Kernel
Raw OpenCL kernel with naive O(n³) algorithm (512×512):
```bash
cargo run --release --example custom_matmul
```

**Performance:**
- Custom kernel: ~2.8ms
- ArrayFire optimized: ~7.3ms
- **Custom kernel is 2.6x faster than ArrayFire!**

**Correctness:**
- Max absolute error: 2.14e-4
- Relative error: 1.41e-6 (0.0001%)
- ✓ Results verified correct

### 4. Custom Cache-Blocked OpenCL Kernel
OpenCL kernel using local memory (GPU cache) with 16×16 blocking (1024×1024):
```bash
cargo run --release --example custom_matmul_blocked
```

**Performance:**
- Custom blocked kernel: ~23ms (**94 GFLOPS**)
- ArrayFire optimized: ~16ms
- Custom kernel is competitive but ArrayFire is faster at this size

**Correctness:**
- Max absolute error: 5.80e-4
- Relative error: 1.99e-6 (0.0002%)
- ✓ Results verified correct

## Main Application

Run the custom kernel vs ArrayFire comparison across multiple sizes:
```bash
cargo run --release
```

**Sample output:**
```
Size         Custom (ms)     ArrayFire (ms)  GFLOPS          Speedup
---------------------------------------------------------------------------
128x128      0.24            0.86            17.68           3.63x
256x256      0.62            0.22            54.09           0.36x
512x512      2.64            4.73            101.73          1.79x
1024x1024    18.79           8.46            114.27          0.45x
```

**Key findings:**
- Custom kernel is **3.6x faster** at 128×128 and **1.8x faster** at 512×512
- ArrayFire wins at 256×256 and 1024×1024 (better optimizations for those sizes)
- Custom kernel achieves **114 GFLOPS** at 1024×1024

## Project Structure

```
arrayfire_demo/
├── src/
│   └── main.rs              # ArrayFire scaling demo
├── examples/
│   ├── simple_gpu.rs        # Basic GPU usage
│   ├── simple_comparison.rs # CPU vs GPU comparison
│   ├── custom_matmul.rs     # Naive OpenCL kernel
│   └── custom_matmul_blocked.rs # Blocked OpenCL kernel
└── README.md                # This file
```

## Key Insights

1. **GPU Transfer Overhead**: For small matrices (< 512), CPU is faster due to data transfer costs
2. **Simple Kernels Can Win**: Naive custom kernel beats ArrayFire at 512×512 (2.6x faster)
3. **Cache Blocking Works on GPU**: Local memory optimization (like CPU L1 cache) achieves 94 GFLOPS
4. **Memory Hierarchy Universal**: Same optimization principles apply to both CPU and GPU
5. **Column-Major Layout Critical**: ArrayFire uses column-major (Fortran-style) memory layout

## Performance Summary (Intel integrated GPU)

| Implementation | Size | Time | GFLOPS | vs ArrayFire | Correctness |
|---|---|---|---|---|---|
| ArrayFire GPU | 1024×1024 | ~15ms | 153 | baseline | ✓ |
| ArrayFire GPU | 512×512 | ~7.3ms | - | baseline | ✓ |
| **Custom Naive** | **512×512** | **~2.8ms** | **~95** | **2.6x faster** | ✓ (error < 3e-4) |
| Custom Blocked | 1024×1024 | ~23ms | 94 | 1.4x slower | ✓ (error < 6e-4) |

### Correctness Verification

All custom kernels have been verified against ArrayFire's implementation:
- **Max errors**: < 6e-4 (acceptable floating-point precision)
- **Relative errors**: < 2e-6 (0.0002%)
- ✓ All implementations produce correct results

### Bug Fixed: Column-Major Layout

**Initial bug**: Custom kernels used row-major indexing (`A[row * N + col]`) but ArrayFire uses **column-major** layout.

**Fix**: Changed to column-major indexing (`A[row + col * N]`)

Before fix: 20-50 absolute error (completely wrong results)
After fix: < 6e-4 error (correct within floating-point precision)

## Related

See parent directory (`../`) for CPU-optimized matrix multiplication implementations with cache blocking and SIMD.
