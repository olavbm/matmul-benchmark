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

