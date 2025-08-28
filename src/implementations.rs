use crate::Matrix;

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

#[cfg(test)]
mod tests {
    use super::*;

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

