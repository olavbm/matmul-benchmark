#![feature(test)]

pub mod matrix;
pub mod test_data;
pub mod benchmark;
pub mod implementations;

pub use matrix::Matrix;
pub use test_data::*;
pub use benchmark::*;
pub use implementations::naive_matmul;

#[cfg(test)]
mod benches {
    extern crate test;
    use test::Bencher;
    use super::*;
    use nalgebra::{DMatrix};

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

    #[bench]
    fn bench_naive_32x32(b: &mut Bencher) {
        let a = Matrix::random(32, 32);
        let mat_b = Matrix::random(32, 32);
        b.iter(|| {
            test::black_box(naive_matmul(&a, &mat_b))
        });
    }

    #[bench]
    fn bench_col_major_32x32(b: &mut Bencher) {
        let a = Matrix::random(32, 32);
        let mat_b = Matrix::random(32, 32);
        let col_major_b = mat_b.to_col_major();
        b.iter(|| {
            test::black_box(naive_matmul(&a, &col_major_b))
        });
    }

    #[bench]
    fn bench_blas_32x32(b: &mut Bencher) {
        let a = Matrix::random(32, 32);
        let mat_b = Matrix::random(32, 32);
        let na_a = matrix_to_dmatrix(&a);
        let na_b = matrix_to_dmatrix(&mat_b);
        b.iter(|| {
            test::black_box(&na_a * &na_b)
        });
    }

    #[bench]
    fn bench_naive_64x64(b: &mut Bencher) {
        let a = Matrix::random(64, 64);
        let mat_b = Matrix::random(64, 64);
        b.iter(|| {
            test::black_box(naive_matmul(&a, &mat_b))
        });
    }

    #[bench]
    fn bench_col_major_64x64(b: &mut Bencher) {
        let a = Matrix::random(64, 64);
        let mat_b = Matrix::random(64, 64);
        let col_major_b = mat_b.to_col_major();
        b.iter(|| {
            test::black_box(naive_matmul(&a, &col_major_b))
        });
    }

    #[bench]
    fn bench_blas_64x64(b: &mut Bencher) {
        let a = Matrix::random(64, 64);
        let mat_b = Matrix::random(64, 64);
        let na_a = matrix_to_dmatrix(&a);
        let na_b = matrix_to_dmatrix(&mat_b);
        b.iter(|| {
            test::black_box(&na_a * &na_b)
        });
    }

    #[bench]
    fn bench_naive_128x128(b: &mut Bencher) {
        let a = Matrix::random(128, 128);
        let mat_b = Matrix::random(128, 128);
        b.iter(|| {
            test::black_box(naive_matmul(&a, &mat_b))
        });
    }

    #[bench]
    fn bench_col_major_128x128(b: &mut Bencher) {
        let a = Matrix::random(128, 128);
        let mat_b = Matrix::random(128, 128);
        let col_major_b = mat_b.to_col_major();
        b.iter(|| {
            test::black_box(naive_matmul(&a, &col_major_b))
        });
    }

    #[bench]
    fn bench_blas_128x128(b: &mut Bencher) {
        let a = Matrix::random(128, 128);
        let mat_b = Matrix::random(128, 128);
        let na_a = matrix_to_dmatrix(&a);
        let na_b = matrix_to_dmatrix(&mat_b);
        b.iter(|| {
            test::black_box(&na_a * &na_b)
        });
    }

    #[bench]
    fn bench_naive_256x256(b: &mut Bencher) {
        let a = Matrix::random(256, 256);
        let mat_b = Matrix::random(256, 256);
        b.iter(|| {
            test::black_box(naive_matmul(&a, &mat_b))
        });
    }

    #[bench]
    fn bench_col_major_256x256(b: &mut Bencher) {
        let a = Matrix::random(256, 256);
        let mat_b = Matrix::random(256, 256);
        let col_major_b = mat_b.to_col_major();
        b.iter(|| {
            test::black_box(naive_matmul(&a, &col_major_b))
        });
    }

    #[bench]
    fn bench_blas_256x256(b: &mut Bencher) {
        let a = Matrix::random(256, 256);
        let mat_b = Matrix::random(256, 256);
        let na_a = matrix_to_dmatrix(&a);
        let na_b = matrix_to_dmatrix(&mat_b);
        b.iter(|| {
            test::black_box(&na_a * &na_b)
        });
    }

    #[bench]
    fn bench_naive_1024x1024(b: &mut Bencher) {
        let a = Matrix::random(1024, 1024);
        let mat_b = Matrix::random(1024, 1024);
        b.iter(|| {
            test::black_box(naive_matmul(&a, &mat_b))
        });
    }

    #[bench]
    fn bench_col_major_1024x1024(b: &mut Bencher) {
        let a = Matrix::random(1024, 1024);
        let mat_b = Matrix::random(1024, 1024);
        let col_major_b = mat_b.to_col_major();
        b.iter(|| {
            test::black_box(naive_matmul(&a, &col_major_b))
        });
    }

    #[bench]
    fn bench_blas_1024x1024(b: &mut Bencher) {
        let a = Matrix::random(1024, 1024);
        let mat_b = Matrix::random(1024, 1024);
        let na_a = matrix_to_dmatrix(&a);
        let na_b = matrix_to_dmatrix(&mat_b);
        b.iter(|| {
            test::black_box(&na_a * &na_b)
        });
    }
}

