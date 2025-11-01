#![feature(test)]

pub mod benchmark;
pub mod dotprod;
pub mod implementations;
pub mod matrix;
pub mod test_data;

pub use benchmark::*;
pub use dotprod::{fma_dotprod, naive_dotprod, simd_dotprod, unrolled_dotprod};
pub use implementations::{
    blocked_matmul, blocked_matmul_col_major_fast, blocked_matmul_default,
    blocked_matmul_optimized, blocked_matmul_with_dotprod, dotprod_matmul,
    dotprod_matmul_col_major_fast, dotprod_matmul_fast, naive_matmul, simd_blocked_matmul_128,
    simd_blocked_matmul_32, simd_blocked_matmul_optimized, simd_matmul,
};
pub use matrix::{ColMajorMatrix, Matrix, MatrixOps, RowMajorMatrix};
pub use test_data::*;

#[cfg(test)]
mod benches {
    extern crate test;
    use super::*;
    use nalgebra::{DMatrix, DVector};
    use test::Bencher;

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
                b.iter(|| test::black_box($func(&a, &mat_b)));
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
                b.iter(|| test::black_box($func(&a, &col_major_b)));
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
                b.iter(|| test::black_box(&na_a * &na_b));
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
                b.iter(|| test::black_box($func(&a, &vec_b)));
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
                b.iter(|| test::black_box(na_a.dot(&na_b)));
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
                b.iter(|| test::black_box($mm_func(&a, &mat_b, $dotprod_func)));
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
                b.iter(|| test::black_box(blocked_matmul(&a, &mat_b, $block_size)));
            }
        };
    }
}
