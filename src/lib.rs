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
    fn bench_naive_dotprod_256(b: &mut Bencher) {
        let a: Vec<f64> = (0..256).map(|i| i as f64 * 0.1).collect();
        let vec_b: Vec<f64> = (0..256).map(|i| (i as f64 + 1.0) * 0.2).collect();
        b.iter(|| {
            test::black_box(naive_dotprod(&a, &vec_b))
        });
    }

    #[bench]
    fn bench_unrolled_dotprod_256(b: &mut Bencher) {
        let a: Vec<f64> = (0..256).map(|i| i as f64 * 0.1).collect();
        let vec_b: Vec<f64> = (0..256).map(|i| (i as f64 + 1.0) * 0.2).collect();
        b.iter(|| {
            test::black_box(unrolled_dotprod(&a, &vec_b))
        });
    }

    #[bench]
    fn bench_naive_dotprod_1024(b: &mut Bencher) {
        let a: Vec<f64> = (0..1024).map(|i| i as f64 * 0.1).collect();
        let vec_b: Vec<f64> = (0..1024).map(|i| (i as f64 + 1.0) * 0.2).collect();
        b.iter(|| {
            test::black_box(naive_dotprod(&a, &vec_b))
        });
    }

    #[bench]
    fn bench_unrolled_dotprod_1024(b: &mut Bencher) {
        let a: Vec<f64> = (0..1024).map(|i| i as f64 * 0.1).collect();
        let vec_b: Vec<f64> = (0..1024).map(|i| (i as f64 + 1.0) * 0.2).collect();
        b.iter(|| {
            test::black_box(unrolled_dotprod(&a, &vec_b))
        });
    }

    #[bench]
    fn bench_mm_with_naive_dotprod_64x64(b: &mut Bencher) {
        let a = Matrix::random(64, 64);
        let mat_b = Matrix::random(64, 64);
        b.iter(|| {
            test::black_box(dotprod_matmul(&a, &mat_b, naive_dotprod))
        });
    }

    #[bench]
    fn bench_mm_with_unrolled_dotprod_64x64(b: &mut Bencher) {
        let a = Matrix::random(64, 64);
        let mat_b = Matrix::random(64, 64);
        b.iter(|| {
            test::black_box(dotprod_matmul(&a, &mat_b, unrolled_dotprod))
        });
    }

    #[bench]
    fn bench_mm_with_naive_dotprod_128x128(b: &mut Bencher) {
        let a = Matrix::random(128, 128);
        let mat_b = Matrix::random(128, 128);
        b.iter(|| {
            test::black_box(dotprod_matmul(&a, &mat_b, naive_dotprod))
        });
    }

    #[bench]
    fn bench_mm_with_unrolled_dotprod_128x128(b: &mut Bencher) {
        let a = Matrix::random(128, 128);
        let mat_b = Matrix::random(128, 128);
        b.iter(|| {
            test::black_box(dotprod_matmul(&a, &mat_b, unrolled_dotprod))
        });
    }

    #[bench]
    fn bench_mm_fast_with_naive_dotprod_128x128(b: &mut Bencher) {
        let a = Matrix::random(128, 128);
        let mat_b = Matrix::random(128, 128);
        b.iter(|| {
            test::black_box(dotprod_matmul_fast(&a, &mat_b, naive_dotprod))
        });
    }

    #[bench]
    fn bench_mm_fast_with_unrolled_dotprod_128x128(b: &mut Bencher) {
        let a = Matrix::random(128, 128);
        let mat_b = Matrix::random(128, 128);
        b.iter(|| {
            test::black_box(dotprod_matmul_fast(&a, &mat_b, unrolled_dotprod))
        });
    }

    #[bench]
    fn bench_mm_fast_with_naive_dotprod_64x64(b: &mut Bencher) {
        let a = Matrix::random(64, 64);
        let mat_b = Matrix::random(64, 64);
        b.iter(|| {
            test::black_box(dotprod_matmul_fast(&a, &mat_b, naive_dotprod))
        });
    }

    #[bench]
    fn bench_mm_fast_with_unrolled_dotprod_64x64(b: &mut Bencher) {
        let a = Matrix::random(64, 64);
        let mat_b = Matrix::random(64, 64);
        b.iter(|| {
            test::black_box(dotprod_matmul_fast(&a, &mat_b, unrolled_dotprod))
        });
    }

    #[bench]
    fn bench_mm_col_major_fast_with_unrolled_dotprod_64x64(b: &mut Bencher) {
        let a = Matrix::random(64, 64);
        let mat_b = Matrix::random(64, 64);
        b.iter(|| {
            test::black_box(dotprod_matmul_col_major_fast(&a, &mat_b, unrolled_dotprod))
        });
    }

    #[bench]
    fn bench_mm_col_major_fast_with_unrolled_dotprod_128x128(b: &mut Bencher) {
        let a = Matrix::random(128, 128);
        let mat_b = Matrix::random(128, 128);
        b.iter(|| {
            test::black_box(dotprod_matmul_col_major_fast(&a, &mat_b, unrolled_dotprod))
        });
    }

    #[bench]
    fn bench_mm_col_major_fast_with_naive_dotprod_64x64(b: &mut Bencher) {
        let a = Matrix::random(64, 64);
        let mat_b = Matrix::random(64, 64);
        b.iter(|| {
            test::black_box(dotprod_matmul_col_major_fast(&a, &mat_b, naive_dotprod))
        });
    }

    #[bench]
    fn bench_mm_col_major_fast_with_naive_dotprod_128x128(b: &mut Bencher) {
        let a = Matrix::random(128, 128);
        let mat_b = Matrix::random(128, 128);
        b.iter(|| {
            test::black_box(dotprod_matmul_col_major_fast(&a, &mat_b, naive_dotprod))
        });
    }

    // Blocked matrix multiplication benchmarks
    
    #[bench]
    fn bench_blocked_matmul_64x64_block32(b: &mut Bencher) {
        let a = Matrix::random(64, 64);
        let mat_b = Matrix::random(64, 64);
        b.iter(|| {
            test::black_box(blocked_matmul(&a, &mat_b, 32))
        });
    }

    #[bench]
    fn bench_blocked_matmul_64x64_block64(b: &mut Bencher) {
        let a = Matrix::random(64, 64);
        let mat_b = Matrix::random(64, 64);
        b.iter(|| {
            test::black_box(blocked_matmul_default(&a, &mat_b))
        });
    }

    #[bench]
    fn bench_blocked_matmul_128x128_block32(b: &mut Bencher) {
        let a = Matrix::random(128, 128);
        let mat_b = Matrix::random(128, 128);
        b.iter(|| {
            test::black_box(blocked_matmul(&a, &mat_b, 32))
        });
    }

    #[bench]
    fn bench_blocked_matmul_128x128_block64(b: &mut Bencher) {
        let a = Matrix::random(128, 128);
        let mat_b = Matrix::random(128, 128);
        b.iter(|| {
            test::black_box(blocked_matmul_default(&a, &mat_b))
        });
    }

    #[bench]
    fn bench_blocked_matmul_128x128_block128(b: &mut Bencher) {
        let a = Matrix::random(128, 128);
        let mat_b = Matrix::random(128, 128);
        b.iter(|| {
            test::black_box(blocked_matmul(&a, &mat_b, 128))
        });
    }

    #[bench]
    fn bench_blocked_matmul_256x256_block32(b: &mut Bencher) {
        let a = Matrix::random(256, 256);
        let mat_b = Matrix::random(256, 256);
        b.iter(|| {
            test::black_box(blocked_matmul(&a, &mat_b, 32))
        });
    }

    #[bench]
    fn bench_blocked_matmul_256x256_block64(b: &mut Bencher) {
        let a = Matrix::random(256, 256);
        let mat_b = Matrix::random(256, 256);
        b.iter(|| {
            test::black_box(blocked_matmul_default(&a, &mat_b))
        });
    }

    #[bench]
    fn bench_blocked_matmul_256x256_block128(b: &mut Bencher) {
        let a = Matrix::random(256, 256);
        let mat_b = Matrix::random(256, 256);
        b.iter(|| {
            test::black_box(blocked_matmul(&a, &mat_b, 128))
        });
    }

    #[bench]
    fn bench_blocked_matmul_512x512_block32(b: &mut Bencher) {
        let a = Matrix::random(512, 512);
        let mat_b = Matrix::random(512, 512);
        b.iter(|| {
            test::black_box(blocked_matmul(&a, &mat_b, 32))
        });
    }

    #[bench]
    fn bench_blocked_matmul_512x512_block64(b: &mut Bencher) {
        let a = Matrix::random(512, 512);
        let mat_b = Matrix::random(512, 512);
        b.iter(|| {
            test::black_box(blocked_matmul_default(&a, &mat_b))
        });
    }

    #[bench]
    fn bench_blocked_matmul_512x512_block128(b: &mut Bencher) {
        let a = Matrix::random(512, 512);
        let mat_b = Matrix::random(512, 512);
        b.iter(|| {
            test::black_box(blocked_matmul(&a, &mat_b, 128))
        });
    }

    // Blocked with optimizations benchmarks

    #[bench]
    fn bench_blocked_optimized_128x128(b: &mut Bencher) {
        let a = Matrix::random(128, 128);
        let mat_b = Matrix::random(128, 128);
        b.iter(|| {
            test::black_box(blocked_matmul_optimized(&a, &mat_b, unrolled_dotprod))
        });
    }

    #[bench]
    fn bench_blocked_optimized_256x256(b: &mut Bencher) {
        let a = Matrix::random(256, 256);
        let mat_b = Matrix::random(256, 256);
        b.iter(|| {
            test::black_box(blocked_matmul_optimized(&a, &mat_b, unrolled_dotprod))
        });
    }

    #[bench]
    fn bench_simd_128x128(b: &mut Bencher) {
        let a = Matrix::random(128, 128);
        let mat_b = Matrix::random(128, 128);
        b.iter(|| {
            test::black_box(simd_matmul(&a, &mat_b))
        });
    }

    #[bench]
    fn bench_simd_256x256(b: &mut Bencher) {
        let a = Matrix::random(256, 256);
        let mat_b = Matrix::random(256, 256);
        b.iter(|| {
            test::black_box(simd_matmul(&a, &mat_b))
        });
    }

    #[bench]
    fn bench_simd_512x512(b: &mut Bencher) {
        let a = Matrix::random(512, 512);
        let mat_b = Matrix::random(512, 512);
        b.iter(|| {
            test::black_box(simd_matmul(&a, &mat_b))
        });
    }

    #[bench]
    fn bench_simd_blocked_optimized_256x256(b: &mut Bencher) {
        let a = Matrix::random(256, 256);
        let mat_b = Matrix::random(256, 256);
        b.iter(|| {
            test::black_box(simd_blocked_matmul_optimized(&a, &mat_b))
        });
    }

    #[bench]
    fn bench_simd_blocked_optimized_512x512(b: &mut Bencher) {
        let a = Matrix::random(512, 512);
        let mat_b = Matrix::random(512, 512);
        b.iter(|| {
            test::black_box(simd_blocked_matmul_optimized(&a, &mat_b))
        });
    }

    #[bench]
    fn bench_simd_blocked_32_512x512(b: &mut Bencher) {
        let a = Matrix::random(512, 512);
        let mat_b = Matrix::random(512, 512);
        b.iter(|| {
            test::black_box(simd_blocked_matmul_32(&a, &mat_b))
        });
    }

    #[bench]
    fn bench_simd_blocked_128_512x512(b: &mut Bencher) {
        let a = Matrix::random(512, 512);
        let mat_b = Matrix::random(512, 512);
        b.iter(|| {
            test::black_box(simd_blocked_matmul_128(&a, &mat_b))
        });
    }

    #[bench]
    fn bench_blocked_optimized_512x512(b: &mut Bencher) {
        let a = Matrix::random(512, 512);
        let mat_b = Matrix::random(512, 512);
        b.iter(|| {
            test::black_box(blocked_matmul_optimized(&a, &mat_b, unrolled_dotprod))
        });
    }

    // Dot Product Benchmarks
    
    #[bench]
    fn bench_vector_dotprod_naive_128(b: &mut Bencher) {
        let a = (0..128).map(|i| i as f64 * 0.1).collect::<Vec<_>>();
        let vec_b = (0..128).map(|i| (i + 1) as f64 * 0.2).collect::<Vec<_>>();
        b.iter(|| {
            test::black_box(naive_dotprod(&a, &vec_b))
        });
    }

    #[bench]
    fn bench_vector_dotprod_unrolled_128(b: &mut Bencher) {
        let a = (0..128).map(|i| i as f64 * 0.1).collect::<Vec<_>>();
        let vec_b = (0..128).map(|i| (i + 1) as f64 * 0.2).collect::<Vec<_>>();
        b.iter(|| {
            test::black_box(unrolled_dotprod(&a, &vec_b))
        });
    }

    #[bench]
    fn bench_vector_dotprod_simd_128(b: &mut Bencher) {
        let a = (0..128).map(|i| i as f64 * 0.1).collect::<Vec<_>>();
        let vec_b = (0..128).map(|i| (i + 1) as f64 * 0.2).collect::<Vec<_>>();
        b.iter(|| {
            test::black_box(simd_dotprod(&a, &vec_b))
        });
    }

    #[bench]
    fn bench_vector_dotprod_nalgebra_128(b: &mut Bencher) {
        let a = (0..128).map(|i| i as f64 * 0.1).collect::<Vec<_>>();
        let vec_b = (0..128).map(|i| (i + 1) as f64 * 0.2).collect::<Vec<_>>();
        let dvec_a = vec_to_dvector(&a);
        let dvec_b = vec_to_dvector(&vec_b);
        b.iter(|| {
            test::black_box(dvec_a.dot(&dvec_b))
        });
    }

    #[bench]
    fn bench_vector_dotprod_naive_1024(b: &mut Bencher) {
        let a = (0..1024).map(|i| i as f64 * 0.1).collect::<Vec<_>>();
        let vec_b = (0..1024).map(|i| (i + 1) as f64 * 0.2).collect::<Vec<_>>();
        b.iter(|| {
            test::black_box(naive_dotprod(&a, &vec_b))
        });
    }

    #[bench]
    fn bench_vector_dotprod_unrolled_1024(b: &mut Bencher) {
        let a = (0..1024).map(|i| i as f64 * 0.1).collect::<Vec<_>>();
        let vec_b = (0..1024).map(|i| (i + 1) as f64 * 0.2).collect::<Vec<_>>();
        b.iter(|| {
            test::black_box(unrolled_dotprod(&a, &vec_b))
        });
    }

    #[bench]
    fn bench_vector_dotprod_simd_1024(b: &mut Bencher) {
        let a = (0..1024).map(|i| i as f64 * 0.1).collect::<Vec<_>>();
        let vec_b = (0..1024).map(|i| (i + 1) as f64 * 0.2).collect::<Vec<_>>();
        b.iter(|| {
            test::black_box(simd_dotprod(&a, &vec_b))
        });
    }

    #[bench]
    fn bench_vector_dotprod_nalgebra_1024(b: &mut Bencher) {
        let a = (0..1024).map(|i| i as f64 * 0.1).collect::<Vec<_>>();
        let vec_b = (0..1024).map(|i| (i + 1) as f64 * 0.2).collect::<Vec<_>>();
        let dvec_a = vec_to_dvector(&a);
        let dvec_b = vec_to_dvector(&vec_b);
        b.iter(|| {
            test::black_box(dvec_a.dot(&dvec_b))
        });
    }

    #[bench]
    fn bench_vector_dotprod_naive_4096(b: &mut Bencher) {
        let a = (0..4096).map(|i| i as f64 * 0.1).collect::<Vec<_>>();
        let vec_b = (0..4096).map(|i| (i + 1) as f64 * 0.2).collect::<Vec<_>>();
        b.iter(|| {
            test::black_box(naive_dotprod(&a, &vec_b))
        });
    }

    #[bench]
    fn bench_vector_dotprod_simd_4096(b: &mut Bencher) {
        let a = (0..4096).map(|i| i as f64 * 0.1).collect::<Vec<_>>();
        let vec_b = (0..4096).map(|i| (i + 1) as f64 * 0.2).collect::<Vec<_>>();
        b.iter(|| {
            test::black_box(simd_dotprod(&a, &vec_b))
        });
    }

    #[bench]
    fn bench_vector_dotprod_nalgebra_4096(b: &mut Bencher) {
        let a = (0..4096).map(|i| i as f64 * 0.1).collect::<Vec<_>>();
        let vec_b = (0..4096).map(|i| (i + 1) as f64 * 0.2).collect::<Vec<_>>();
        let dvec_a = vec_to_dvector(&a);
        let dvec_b = vec_to_dvector(&vec_b);
        b.iter(|| {
            test::black_box(dvec_a.dot(&dvec_b))
        });
    }
}

