use crate::{Matrix, MatrixOps};
use rand::prelude::*;

pub struct TestData {
    pub a: Matrix,
    pub b: Matrix,
    pub expected: Option<Matrix>,
}

impl TestData {
    pub fn new(a: Matrix, b: Matrix) -> Self {
        Self { a, b, expected: None }
    }
    
    pub fn with_expected(a: Matrix, b: Matrix, expected: Matrix) -> Self {
        Self { a, b, expected: Some(expected) }
    }
}

pub fn generate_test_matrices(size: usize) -> TestData {
    let mut rng = StdRng::from_seed([42; 32]); // Fixed seed for reproducible results
    
    let a_data: Vec<f64> = (0..size * size)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    
    let b_data: Vec<f64> = (0..size * size)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    
    let a = Matrix::from_data(a_data, size, size);
    let b = Matrix::from_data(b_data, size, size);
    
    TestData::new(a, b)
}

pub fn generate_simple_test() -> TestData {
    let a_data = vec![1.0, 2.0, 3.0, 4.0];
    let b_data = vec![5.0, 6.0, 7.0, 8.0];
    let expected_data = vec![19.0, 22.0, 43.0, 50.0]; // Known result
    
    let a = Matrix::from_data(a_data, 2, 2);
    let b = Matrix::from_data(b_data, 2, 2);
    let expected = Matrix::from_data(expected_data, 2, 2);
    
    TestData::with_expected(a, b, expected)
}

// Unused utility functions - kept for potential future use

/// Generate test data multiplying matrix by identity.
/// Currently unused but useful for correctness testing.
#[allow(dead_code)]
pub fn generate_identity_test(size: usize) -> TestData {
    let mut rng = StdRng::from_seed([123; 32]);

    let a_data: Vec<f64> = (0..size * size)
        .map(|_| rng.gen_range(-10.0..10.0))
        .collect();
    let a = Matrix::from_data(a_data.clone(), size, size);

    let mut identity = Matrix::new(size, size);
    for i in 0..size {
        identity.set(i, i, 1.0);
    }

    TestData::with_expected(a.clone(), identity, a)
}

pub const EPSILON: f64 = 1e-10;

/// Convenience wrapper for `matrices_equal_with_epsilon` with default epsilon.
/// Currently unused; tests use `matrices_equal_with_epsilon` directly.
#[allow(dead_code)]
pub fn matrices_equal(a: &Matrix, b: &Matrix) -> bool {
    matrices_equal_with_epsilon(a, b, EPSILON)
}

pub fn matrices_equal_with_epsilon(a: &Matrix, b: &Matrix, epsilon: f64) -> bool {
    if a.rows() != b.rows() || a.cols() != b.cols() {
        return false;
    }

    for i in 0..a.rows() {
        for j in 0..a.cols() {
            if (a.get(i, j) - b.get(i, j)).abs() > epsilon {
                return false;
            }
        }
    }
    true
}