#![feature(test)]

pub mod matrix;
pub mod test_data;
pub mod benchmark;
pub mod implementations;
pub mod dotprod;

pub use matrix::Matrix;
pub use test_data::*;
pub use benchmark::*;
pub use implementations::{
    naive_matmul, 
    dotprod_matmul, 
    dotprod_matmul_fast, 
    dotprod_matmul_col_major_fast,
    blocked_matmul,
    blocked_matmul_default,
    blocked_matmul_with_dotprod,
    blocked_matmul_col_major_fast,
    blocked_matmul_optimized,
    simd_matmul,
    simd_blocked_matmul_optimized,
    simd_blocked_matmul_32,
    simd_blocked_matmul_128
};
pub use dotprod::{naive_dotprod, unrolled_dotprod, simd_dotprod};

#[cfg(test)]
mod benches {
    extern crate test;
    use test::Bencher;
    use super::*;
    use nalgebra::{DMatrix, DVector};

    // Helper function to convert our Matrix to nalgebra DMatrix
    fn matrix_to_dmatrix(mat: &Matrix) -> DMatrix<f64> {
        let mut data = Vec::with_capacity(mat.rows * mat.cols);
        for i in 0..mat.rows {
            for j in 0..mat.cols {
                data.push(mat.get(i, j));
            }
        }
        DMatrix::from_row_slice(mat.rows, mat.cols, &data)
    }

    // Helper function to convert Vec<f64> to nalgebra DVector
    fn vec_to_dvector(vec: &[f64]) -> DVector<f64> {
        DVector::from_vec(vec.to_vec())
    }

    // Macro to generate matrix multiplication benchmarks
    macro_rules! bench_matmul {
        ($name:ident, $size:expr, $func:expr) => {
            #[bench]
            fn $name(b: &mut Bencher) {
                let a = Matrix::random($size, $size);
                let mat_b = Matrix::random($size, $size);
                b.iter(|| {
                    test::black_box($func(&a, &mat_b))
                });
            }
        };
    }

    // Macro to generate column-major benchmarks
    macro_rules! bench_matmul_col_major {
        ($name:ident, $size:expr, $func:expr) => {
            #[bench]
            fn $name(b: &mut Bencher) {
                let a = Matrix::random($size, $size);
                let mat_b = Matrix::random($size, $size);
                let col_major_b = mat_b.to_col_major();
                b.iter(|| {
                    test::black_box($func(&a, &col_major_b))
                });
            }
        };
    }

    // Macro to generate BLAS benchmarks
    macro_rules! bench_blas {
        ($name:ident, $size:expr) => {
            #[bench]
            fn $name(b: &mut Bencher) {
                let a = Matrix::random($size, $size);
                let mat_b = Matrix::random($size, $size);
                let na_a = matrix_to_dmatrix(&a);
                let na_b = matrix_to_dmatrix(&mat_b);
                b.iter(|| {
                    test::black_box(&na_a * &na_b)
                });
            }
        };
    }

    // Macro to generate dot product benchmarks
    macro_rules! bench_dotprod {
        ($name:ident, $size:expr, $func:expr) => {
            #[bench]
            fn $name(b: &mut Bencher) {
                let a: Vec<f64> = (0..$size).map(|i| i as f64 * 0.1).collect();
                let vec_b: Vec<f64> = (0..$size).map(|i| (i as f64 + 1.0) * 0.2).collect();
                b.iter(|| {
                    test::black_box($func(&a, &vec_b))
                });
            }
        };
    }

    // Macro to generate nalgebra dot product benchmarks
    macro_rules! bench_dotprod_nalgebra {
        ($name:ident, $size:expr) => {
            #[bench]
            fn $name(b: &mut Bencher) {
                let a: Vec<f64> = (0..$size).map(|i| i as f64 * 0.1).collect();
                let vec_b: Vec<f64> = (0..$size).map(|i| (i as f64 + 1.0) * 0.2).collect();
                let na_a = vec_to_dvector(&a);
                let na_b = vec_to_dvector(&vec_b);
                b.iter(|| {
                    test::black_box(na_a.dot(&na_b))
                });
            }
        };
    }

    // Macro to generate matrix multiplication benchmarks with custom function and dotprod
    macro_rules! bench_matmul_with_dotprod {
        ($name:ident, $size:expr, $mm_func:expr, $dotprod_func:expr) => {
            #[bench]
            fn $name(b: &mut Bencher) {
                let a = Matrix::random($size, $size);
                let mat_b = Matrix::random($size, $size);
                b.iter(|| {
                    test::black_box($mm_func(&a, &mat_b, $dotprod_func))
                });
            }
        };
    }

    // Macro to generate blocked matrix multiplication benchmarks
    macro_rules! bench_blocked {
        ($name:ident, $size:expr, $block_size:expr) => {
            #[bench]
            fn $name(b: &mut Bencher) {
                let a = Matrix::random($size, $size);
                let mat_b = Matrix::random($size, $size);
                b.iter(|| {
                    test::black_box(blocked_matmul(&a, &mat_b, $block_size))
                });
            }
        };
    }

    // Matrix multiplication benchmarks - organized by algorithm and size
    bench_matmul!(bench_naive_32x32, 32, naive_matmul);
    bench_matmul_col_major!(bench_col_major_32x32, 32, naive_matmul);
    bench_blas!(bench_blas_32x32, 32);

    bench_matmul!(bench_naive_64x64, 64, naive_matmul);
    bench_matmul_col_major!(bench_col_major_64x64, 64, naive_matmul);
    bench_blas!(bench_blas_64x64, 64);

    bench_matmul!(bench_naive_128x128, 128, naive_matmul);
    bench_matmul_col_major!(bench_col_major_128x128, 128, naive_matmul);
    bench_blas!(bench_blas_128x128, 128);

    bench_matmul!(bench_naive_256x256, 256, naive_matmul);
    bench_matmul_col_major!(bench_col_major_256x256, 256, naive_matmul);
    bench_blas!(bench_blas_256x256, 256);


    // Dot product benchmarks
    bench_dotprod!(bench_naive_dotprod_256, 256, naive_dotprod);
    bench_dotprod!(bench_unrolled_dotprod_256, 256, unrolled_dotprod);
    bench_dotprod!(bench_naive_dotprod_1024, 1024, naive_dotprod);
    bench_dotprod!(bench_unrolled_dotprod_1024, 1024, unrolled_dotprod);

    // Matrix multiplication with different dotprod implementations
    bench_matmul_with_dotprod!(bench_mm_with_naive_dotprod_64x64, 64, dotprod_matmul, naive_dotprod);
    bench_matmul_with_dotprod!(bench_mm_with_unrolled_dotprod_64x64, 64, dotprod_matmul, unrolled_dotprod);
    bench_matmul_with_dotprod!(bench_mm_with_naive_dotprod_128x128, 128, dotprod_matmul, naive_dotprod);
    bench_matmul_with_dotprod!(bench_mm_with_unrolled_dotprod_128x128, 128, dotprod_matmul, unrolled_dotprod);

    // Fast (buffer reuse) variants
    bench_matmul_with_dotprod!(bench_mm_fast_with_naive_dotprod_64x64, 64, dotprod_matmul_fast, naive_dotprod);
    bench_matmul_with_dotprod!(bench_mm_fast_with_unrolled_dotprod_64x64, 64, dotprod_matmul_fast, unrolled_dotprod);
    bench_matmul_with_dotprod!(bench_mm_fast_with_naive_dotprod_128x128, 128, dotprod_matmul_fast, naive_dotprod);
    bench_matmul_with_dotprod!(bench_mm_fast_with_unrolled_dotprod_128x128, 128, dotprod_matmul_fast, unrolled_dotprod);

    // Column-major + fast variants
    bench_matmul_with_dotprod!(bench_mm_col_major_fast_with_naive_dotprod_64x64, 64, dotprod_matmul_col_major_fast, naive_dotprod);
    bench_matmul_with_dotprod!(bench_mm_col_major_fast_with_unrolled_dotprod_64x64, 64, dotprod_matmul_col_major_fast, unrolled_dotprod);
    bench_matmul_with_dotprod!(bench_mm_col_major_fast_with_naive_dotprod_128x128, 128, dotprod_matmul_col_major_fast, naive_dotprod);
    bench_matmul_with_dotprod!(bench_mm_col_major_fast_with_unrolled_dotprod_128x128, 128, dotprod_matmul_col_major_fast, unrolled_dotprod);

    // Blocked matrix multiplication benchmarks - testing different block sizes
    bench_blocked!(bench_blocked_matmul_64x64_block32, 64, 32);
    bench_blocked!(bench_blocked_matmul_64x64_block64, 64, 64);
    bench_blocked!(bench_blocked_matmul_128x128_block32, 128, 32);
    bench_blocked!(bench_blocked_matmul_128x128_block64, 128, 64);
    bench_blocked!(bench_blocked_matmul_128x128_block128, 128, 128);
    bench_blocked!(bench_blocked_matmul_256x256_block32, 256, 32);
    bench_blocked!(bench_blocked_matmul_256x256_block64, 256, 64);
    bench_blocked!(bench_blocked_matmul_256x256_block128, 256, 128);
    bench_blocked!(bench_blocked_matmul_512x512_block32, 512, 32);
    bench_blocked!(bench_blocked_matmul_512x512_block64, 512, 64);
    bench_blocked!(bench_blocked_matmul_512x512_block128, 512, 128);

    // Blocked with optimizations (column-major + dotprod)
    bench_matmul_with_dotprod!(bench_blocked_optimized_128x128, 128, blocked_matmul_optimized, unrolled_dotprod);
    bench_matmul_with_dotprod!(bench_blocked_optimized_256x256, 256, blocked_matmul_optimized, unrolled_dotprod);
    bench_matmul_with_dotprod!(bench_blocked_optimized_512x512, 512, blocked_matmul_optimized, unrolled_dotprod);

    // SIMD benchmarks
    bench_matmul!(bench_simd_128x128, 128, simd_matmul);
    bench_matmul!(bench_simd_256x256, 256, simd_matmul);
    bench_matmul!(bench_simd_512x512, 512, simd_matmul);
    bench_matmul!(bench_simd_blocked_optimized_256x256, 256, simd_blocked_matmul_optimized);
    bench_matmul!(bench_simd_blocked_optimized_512x512, 512, simd_blocked_matmul_optimized);
    bench_matmul!(bench_simd_blocked_32_512x512, 512, simd_blocked_matmul_32);
    bench_matmul!(bench_simd_blocked_128_512x512, 512, simd_blocked_matmul_128);

    // Vector dot product benchmarks - comparing different implementations
    bench_dotprod!(bench_vector_dotprod_naive_128, 128, naive_dotprod);
    bench_dotprod!(bench_vector_dotprod_unrolled_128, 128, unrolled_dotprod);
    bench_dotprod!(bench_vector_dotprod_simd_128, 128, simd_dotprod);
    bench_dotprod_nalgebra!(bench_vector_dotprod_nalgebra_128, 128);

    bench_dotprod!(bench_vector_dotprod_naive_1024, 1024, naive_dotprod);
    bench_dotprod!(bench_vector_dotprod_unrolled_1024, 1024, unrolled_dotprod);
    bench_dotprod!(bench_vector_dotprod_simd_1024, 1024, simd_dotprod);
    bench_dotprod_nalgebra!(bench_vector_dotprod_nalgebra_1024, 1024);

    bench_dotprod!(bench_vector_dotprod_naive_4096, 4096, naive_dotprod);
    bench_dotprod!(bench_vector_dotprod_simd_4096, 4096, simd_dotprod);
    bench_dotprod_nalgebra!(bench_vector_dotprod_nalgebra_4096, 4096);
}

