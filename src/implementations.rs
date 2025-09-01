use crate::Matrix;
use crate::dotprod::simd_dotprod;

pub fn dotprod_matmul<F>(a: &Matrix, b: &Matrix, dotprod_fn: F) -> Matrix 
where 
    F: Fn(&[f64], &[f64]) -> f64
{
    assert_eq!(a.cols, b.rows, "Matrix dimensions don't match");
    
    let mut result = Matrix::new(a.rows, b.cols);
    
    for i in 0..a.rows {
        let row = a.get_row(i);
        for j in 0..b.cols {
            let col = b.get_col(j);
            let dot_result = dotprod_fn(&row, &col);
            result.set(i, j, dot_result);
        }
    }
    
    result
}

pub fn dotprod_matmul_fast<F>(a: &Matrix, b: &Matrix, dotprod_fn: F) -> Matrix 
where 
    F: Fn(&[f64], &[f64]) -> f64
{
    assert_eq!(a.cols, b.rows, "Matrix dimensions don't match");
    
    let mut result = Matrix::new(a.rows, b.cols);
    
    // Reuse buffers to avoid allocations
    let mut row_buf = vec![0.0; a.cols];
    let mut col_buf = vec![0.0; b.rows];
    
    for i in 0..a.rows {
        // Fill row buffer
        for k in 0..a.cols {
            row_buf[k] = a.get(i, k);
        }
        
        for j in 0..b.cols {
            // Fill column buffer
            for k in 0..b.rows {
                col_buf[k] = b.get(k, j);
            }
            
            let dot_result = dotprod_fn(&row_buf, &col_buf);
            result.set(i, j, dot_result);
        }
    }
    
    result
}

pub fn dotprod_matmul_col_major_fast<F>(a: &Matrix, b: &Matrix, dotprod_fn: F) -> Matrix 
where 
    F: Fn(&[f64], &[f64]) -> f64
{
    assert_eq!(a.cols, b.rows, "Matrix dimensions don't match");
    
    let mut result = Matrix::new(a.rows, b.cols);
    
    // Convert B to column-major for better cache locality
    let b_col_major = b.to_col_major();
    
    // Reuse buffers to avoid allocations
    let mut row_buf = vec![0.0; a.cols];
    let mut col_buf = vec![0.0; b.rows];
    
    for i in 0..a.rows {
        // Fill row buffer once per row
        for k in 0..a.cols {
            row_buf[k] = a.get(i, k);
        }
        
        for j in 0..b.cols {
            // Fill column buffer from column-major B (sequential access!)
            for k in 0..b.rows {
                col_buf[k] = b_col_major.get(k, j);
            }
            
            let dot_result = dotprod_fn(&row_buf, &col_buf);
            result.set(i, j, dot_result);
        }
    }
    
    result
}

pub fn naive_matmul(a: &Matrix, b: &Matrix) -> Matrix {
    assert_eq!(a.cols, b.rows, "Matrix dimensions don't match");
    
    let mut result = Matrix::new(a.rows, b.cols);
    
    for i in 0..a.rows {
        for j in 0..b.cols {
            let mut sum = 0.0;
            for k in 0..a.cols {
                sum += a.get(i, k) * b.get(k, j);
            }
            result.set(i, j, sum);
        }
    }
    
    result
}

/// Cache-blocked matrix multiplication with configurable block size
pub fn blocked_matmul(a: &Matrix, b: &Matrix, block_size: usize) -> Matrix {
    assert_eq!(a.cols, b.rows, "Matrix dimensions don't match");
    
    let mut result = Matrix::new(a.rows, b.cols);
    
    // Six-nested-loop cache-blocked algorithm
    // Outer loops: iterate over blocks
    // Inner loops: compute within blocks
    for ii in (0..a.rows).step_by(block_size) {
        for jj in (0..b.cols).step_by(block_size) {
            for kk in (0..a.cols).step_by(block_size) {
                // Define block boundaries
                let i_end = (ii + block_size).min(a.rows);
                let j_end = (jj + block_size).min(b.cols);
                let k_end = (kk + block_size).min(a.cols);
                
                // Compute the block: result[ii:i_end, jj:j_end] += a[ii:i_end, kk:k_end] * b[kk:k_end, jj:j_end]
                for i in ii..i_end {
                    for j in jj..j_end {
                        let mut sum = result.get(i, j); // Accumulate partial results
                        for k in kk..k_end {
                            sum += a.get(i, k) * b.get(k, j);
                        }
                        result.set(i, j, sum);
                    }
                }
            }
        }
    }
    
    result
}

/// Cache-blocked matrix multiplication with default block size optimized for L1d cache
pub fn blocked_matmul_default(a: &Matrix, b: &Matrix) -> Matrix {
    // 64x64 block = 32KB, fits well in 37KB L1d cache per core
    blocked_matmul(a, b, 64)
}

/// Cache-blocked matrix multiplication with dotprod integration for inner block computation
pub fn blocked_matmul_with_dotprod<F>(a: &Matrix, b: &Matrix, block_size: usize, dotprod_fn: F) -> Matrix 
where 
    F: Fn(&[f64], &[f64]) -> f64
{
    assert_eq!(a.cols, b.rows, "Matrix dimensions don't match");
    
    let mut result = Matrix::new(a.rows, b.cols);
    
    // Pre-allocate buffers for dotprod computation
    let max_block_size = block_size.min(a.cols.max(b.rows));
    let mut row_buf = vec![0.0; max_block_size];
    let mut col_buf = vec![0.0; max_block_size];
    
    // Six-nested-loop cache-blocked algorithm with dotprod optimization
    for ii in (0..a.rows).step_by(block_size) {
        for jj in (0..b.cols).step_by(block_size) {
            for kk in (0..a.cols).step_by(block_size) {
                // Define block boundaries
                let i_end = (ii + block_size).min(a.rows);
                let j_end = (jj + block_size).min(b.cols);
                let k_end = (kk + block_size).min(a.cols);
                let k_size = k_end - kk;
                
                // Compute the block using optimized dot products
                for i in ii..i_end {
                    // Fill row buffer for this row of A
                    for (idx, k) in (kk..k_end).enumerate() {
                        row_buf[idx] = a.get(i, k);
                    }
                    
                    for j in jj..j_end {
                        // Fill column buffer for this column of B
                        for (idx, k) in (kk..k_end).enumerate() {
                            col_buf[idx] = b.get(k, j);
                        }
                        
                        // Compute dot product and accumulate
                        let partial_sum = dotprod_fn(&row_buf[..k_size], &col_buf[..k_size]);
                        let current_value = result.get(i, j);
                        result.set(i, j, current_value + partial_sum);
                    }
                }
            }
        }
    }
    
    result
}

/// Ultimate optimization: Cache-blocked + column-major + dotprod + zero allocations
pub fn blocked_matmul_col_major_fast<F>(a: &Matrix, b: &Matrix, block_size: usize, dotprod_fn: F) -> Matrix 
where 
    F: Fn(&[f64], &[f64]) -> f64
{
    assert_eq!(a.cols, b.rows, "Matrix dimensions don't match");
    
    let mut result = Matrix::new(a.rows, b.cols);
    
    // Convert B to column-major for better cache locality
    let b_col_major = b.to_col_major();
    
    // Pre-allocate buffers for dotprod computation
    let max_block_size = block_size.min(a.cols.max(b.rows));
    let mut row_buf = vec![0.0; max_block_size];
    let mut col_buf = vec![0.0; max_block_size];
    
    // Six-nested-loop cache-blocked algorithm with all optimizations
    for ii in (0..a.rows).step_by(block_size) {
        for jj in (0..b.cols).step_by(block_size) {
            for kk in (0..a.cols).step_by(block_size) {
                // Define block boundaries
                let i_end = (ii + block_size).min(a.rows);
                let j_end = (jj + block_size).min(b.cols);
                let k_end = (kk + block_size).min(a.cols);
                let k_size = k_end - kk;
                
                // Compute the block using all optimizations
                for i in ii..i_end {
                    // Fill row buffer for this row of A (sequential access)
                    for (idx, k) in (kk..k_end).enumerate() {
                        row_buf[idx] = a.get(i, k);
                    }
                    
                    for j in jj..j_end {
                        // Fill column buffer from column-major B (sequential access!)
                        for (idx, k) in (kk..k_end).enumerate() {
                            col_buf[idx] = b_col_major.get(k, j);
                        }
                        
                        // Compute optimized dot product and accumulate
                        let partial_sum = dotprod_fn(&row_buf[..k_size], &col_buf[..k_size]);
                        let current_value = result.get(i, j);
                        result.set(i, j, current_value + partial_sum);
                    }
                }
            }
        }
    }
    
    result
}

/// Convenience function with optimal defaults for this hardware
pub fn blocked_matmul_optimized<F>(a: &Matrix, b: &Matrix, dotprod_fn: F) -> Matrix 
where 
    F: Fn(&[f64], &[f64]) -> f64
{
    // 64x64 block size optimized for 37KB L1d cache
    blocked_matmul_col_major_fast(a, b, 64, dotprod_fn)
}

pub fn simd_matmul(a: &Matrix, b: &Matrix) -> Matrix {
    blocked_matmul_col_major_fast(a, b, 64, simd_dotprod)
}

pub fn simd_blocked_matmul_optimized(a: &Matrix, b: &Matrix) -> Matrix {
    blocked_matmul_optimized(a, b, simd_dotprod)
}

pub fn simd_blocked_matmul_32(a: &Matrix, b: &Matrix) -> Matrix {
    blocked_matmul_col_major_fast(a, b, 32, simd_dotprod)
}

pub fn simd_blocked_matmul_128(a: &Matrix, b: &Matrix) -> Matrix {
    blocked_matmul_col_major_fast(a, b, 128, simd_dotprod)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blocked_matmul_correctness() {
        // Test: [[1,2],[3,4]] × [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = Matrix::from_data_row_major(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = Matrix::from_data_row_major(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
        
        let result = blocked_matmul(&a, &b, 1); // Block size 1 (essentially naive)
        
        assert_eq!(result.get(0, 0), 19.0);  // 1*5 + 2*7
        assert_eq!(result.get(0, 1), 22.0);  // 1*6 + 2*8
        assert_eq!(result.get(1, 0), 43.0);  // 3*5 + 4*7
        assert_eq!(result.get(1, 1), 50.0);  // 3*6 + 4*8
    }

    #[test]
    fn test_blocked_matmul_vs_naive() {
        let a = Matrix::random(8, 6);
        let b = Matrix::random(6, 8);
        
        let naive_result = naive_matmul(&a, &b);
        let blocked_result = blocked_matmul(&a, &b, 2);
        let blocked_default = blocked_matmul_default(&a, &b);
        
        // Results should be identical (within floating point precision)
        for i in 0..8 {
            for j in 0..8 {
                let diff1 = (naive_result.get(i, j) - blocked_result.get(i, j)).abs();
                let diff2 = (naive_result.get(i, j) - blocked_default.get(i, j)).abs();
                assert!(diff1 < 1e-10, "Blocked (size 2) differs from naive at ({}, {})", i, j);
                assert!(diff2 < 1e-10, "Blocked default differs from naive at ({}, {})", i, j);
            }
        }
    }

    #[test]
    fn test_blocked_with_dotprod() {
        use crate::dotprod::{naive_dotprod, unrolled_dotprod};
        
        let a = Matrix::random(6, 4);
        let b = Matrix::random(4, 6);
        
        let naive_result = naive_matmul(&a, &b);
        let blocked_naive_dp = blocked_matmul_with_dotprod(&a, &b, 3, naive_dotprod);
        let blocked_unrolled_dp = blocked_matmul_with_dotprod(&a, &b, 3, unrolled_dotprod);
        
        // All results should be identical
        for i in 0..6 {
            for j in 0..6 {
                let diff1 = (naive_result.get(i, j) - blocked_naive_dp.get(i, j)).abs();
                let diff2 = (naive_result.get(i, j) - blocked_unrolled_dp.get(i, j)).abs();
                assert!(diff1 < 1e-10, "Blocked+dotprod differs at ({}, {})", i, j);
                assert!(diff2 < 1e-10, "Blocked+unrolled differs at ({}, {})", i, j);
            }
        }
    }

    #[test]
    fn test_blocked_ultimate_optimization() {
        use crate::dotprod::unrolled_dotprod;
        
        let a = Matrix::random(5, 3);
        let b = Matrix::random(3, 5);
        
        let naive_result = naive_matmul(&a, &b);
        let blocked_ultimate = blocked_matmul_col_major_fast(&a, &b, 2, unrolled_dotprod);
        let blocked_optimized = blocked_matmul_optimized(&a, &b, unrolled_dotprod);
        
        // All results should be identical
        for i in 0..5 {
            for j in 0..5 {
                let diff1 = (naive_result.get(i, j) - blocked_ultimate.get(i, j)).abs();
                let diff2 = (naive_result.get(i, j) - blocked_optimized.get(i, j)).abs();
                assert!(diff1 < 1e-10, "Ultimate blocked differs at ({}, {})", i, j);
                assert!(diff2 < 1e-10, "Optimized blocked differs at ({}, {})", i, j);
            }
        }
    }

    #[test]
    fn test_simple_2x2_matmul() {
        // Test: [[1,2],[3,4]] × [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = Matrix::from_data_row_major(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = Matrix::from_data_row_major(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
        
        let result = naive_matmul(&a, &b);
        
        assert_eq!(result.get(0, 0), 19.0);  // 1*5 + 2*7
        assert_eq!(result.get(0, 1), 22.0);  // 1*6 + 2*8
        assert_eq!(result.get(1, 0), 43.0);  // 3*5 + 4*7
        assert_eq!(result.get(1, 1), 50.0);  // 3*6 + 4*8
    }

    #[test]
    fn test_dotprod_matmul_with_naive() {
        use crate::dotprod::naive_dotprod;
        
        // Test: [[1,2],[3,4]] × [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = Matrix::from_data_row_major(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = Matrix::from_data_row_major(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
        
        let result = dotprod_matmul(&a, &b, naive_dotprod);
        
        assert_eq!(result.get(0, 0), 19.0);  // 1*5 + 2*7
        assert_eq!(result.get(0, 1), 22.0);  // 1*6 + 2*8
        assert_eq!(result.get(1, 0), 43.0);  // 3*5 + 4*7
        assert_eq!(result.get(1, 1), 50.0);  // 3*6 + 4*8
    }

    #[test]
    fn test_dotprod_matmul_with_unrolled() {
        use crate::dotprod::unrolled_dotprod;
        
        // Same test but with unrolled dot product
        let a = Matrix::from_data_row_major(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = Matrix::from_data_row_major(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
        
        let result = dotprod_matmul(&a, &b, unrolled_dotprod);
        
        assert_eq!(result.get(0, 0), 19.0);
        assert_eq!(result.get(0, 1), 22.0);
        assert_eq!(result.get(1, 0), 43.0);
        assert_eq!(result.get(1, 1), 50.0);
    }

    #[test]
    fn test_dotprod_vs_naive_matmul() {
        use crate::dotprod::naive_dotprod;
        
        let a = Matrix::random(4, 3);
        let b = Matrix::random(3, 4);
        
        let result1 = naive_matmul(&a, &b);
        let result2 = dotprod_matmul(&a, &b, naive_dotprod);
        
        // Results should be identical
        for i in 0..result1.rows {
            for j in 0..result1.cols {
                let diff = (result1.get(i, j) - result2.get(i, j)).abs();
                assert!(diff < 1e-10, "Results differ at ({}, {}): {} vs {}", i, j, result1.get(i, j), result2.get(i, j));
            }
        }
    }

    #[test]
    fn test_dotprod_matmul_fast() {
        use crate::dotprod::naive_dotprod;
        
        // Test: [[1,2],[3,4]] × [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = Matrix::from_data_row_major(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = Matrix::from_data_row_major(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
        
        let result = dotprod_matmul_fast(&a, &b, naive_dotprod);
        
        assert_eq!(result.get(0, 0), 19.0);  // 1*5 + 2*7
        assert_eq!(result.get(0, 1), 22.0);  // 1*6 + 2*8
        assert_eq!(result.get(1, 0), 43.0);  // 3*5 + 4*7
        assert_eq!(result.get(1, 1), 50.0);  // 3*6 + 4*8
    }

    #[test]
    fn test_fast_vs_regular_dotprod_matmul() {
        use crate::dotprod::unrolled_dotprod;
        
        let a = Matrix::random(4, 3);
        let b = Matrix::random(3, 4);
        
        let result1 = dotprod_matmul(&a, &b, unrolled_dotprod);
        let result2 = dotprod_matmul_fast(&a, &b, unrolled_dotprod);
        
        // Results should be identical
        for i in 0..result1.rows {
            for j in 0..result1.cols {
                let diff = (result1.get(i, j) - result2.get(i, j)).abs();
                assert!(diff < 1e-10, "Results differ at ({}, {}): {} vs {}", i, j, result1.get(i, j), result2.get(i, j));
            }
        }
    }

    #[test]
    fn test_dotprod_matmul_col_major_fast() {
        use crate::dotprod::unrolled_dotprod;
        
        // Test: [[1,2],[3,4]] × [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = Matrix::from_data_row_major(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = Matrix::from_data_row_major(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
        
        let result = dotprod_matmul_col_major_fast(&a, &b, unrolled_dotprod);
        
        assert_eq!(result.get(0, 0), 19.0);  // 1*5 + 2*7
        assert_eq!(result.get(0, 1), 22.0);  // 1*6 + 2*8
        assert_eq!(result.get(1, 0), 43.0);  // 3*5 + 4*7
        assert_eq!(result.get(1, 1), 50.0);  // 3*6 + 4*8
    }

    #[test]
    fn test_all_dotprod_implementations_match() {
        use crate::dotprod::unrolled_dotprod;
        
        let a = Matrix::random(4, 3);
        let b = Matrix::random(3, 4);
        
        let result1 = naive_matmul(&a, &b);
        let result2 = dotprod_matmul(&a, &b, unrolled_dotprod);
        let result3 = dotprod_matmul_fast(&a, &b, unrolled_dotprod);
        let result4 = dotprod_matmul_col_major_fast(&a, &b, unrolled_dotprod);
        
        // All results should be identical
        for i in 0..result1.rows {
            for j in 0..result1.cols {
                let r1 = result1.get(i, j);
                let r2 = result2.get(i, j);
                let r3 = result3.get(i, j);
                let r4 = result4.get(i, j);
                
                assert!((r1 - r2).abs() < 1e-10, "result1 vs result2 differ at ({}, {})", i, j);
                assert!((r1 - r3).abs() < 1e-10, "result1 vs result3 differ at ({}, {})", i, j);
                assert!((r1 - r4).abs() < 1e-10, "result1 vs result4 differ at ({}, {})", i, j);
            }
        }
    }
}

