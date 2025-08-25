use crate::Matrix;

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
}

