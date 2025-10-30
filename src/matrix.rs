use rand::prelude::*;

/// Shared operations for all matrix types
pub trait MatrixOps {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn data(&self) -> &[f64];
    fn get(&self, row: usize, col: usize) -> f64;
    fn set(&mut self, row: usize, col: usize, value: f64);

    /// Extract a row as a vector
    fn get_row(&self, row: usize) -> Vec<f64> {
        assert!(
            row < self.rows(),
            "Row index {} out of bounds for {}×{} matrix",
            row, self.rows(), self.cols()
        );
        let mut result = Vec::with_capacity(self.cols());
        for col in 0..self.cols() {
            result.push(self.get(row, col));
        }
        result
    }

    /// Extract a column as a vector
    fn get_col(&self, col: usize) -> Vec<f64> {
        assert!(
            col < self.cols(),
            "Column index {} out of bounds for {}×{} matrix",
            col, self.rows(), self.cols()
        );
        let mut result = Vec::with_capacity(self.rows());
        for row in 0..self.rows() {
            result.push(self.get(row, col));
        }
        result
    }
}

/// Row-major matrix storage: data[row * cols + col]
#[derive(Clone, Debug)]
pub struct RowMajorMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

/// Column-major matrix storage: data[col * rows + row]
#[derive(Clone, Debug)]
pub struct ColMajorMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl MatrixOps for RowMajorMatrix {
    #[inline(always)]
    fn rows(&self) -> usize {
        self.rows
    }

    #[inline(always)]
    fn cols(&self) -> usize {
        self.cols
    }

    #[inline(always)]
    fn data(&self) -> &[f64] {
        &self.data
    }

    #[inline(always)]
    fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.cols + col]
    }

    #[inline(always)]
    fn set(&mut self, row: usize, col: usize, value: f64) {
        self.data[row * self.cols + col] = value;
    }
}

impl MatrixOps for ColMajorMatrix {
    #[inline(always)]
    fn rows(&self) -> usize {
        self.rows
    }

    #[inline(always)]
    fn cols(&self) -> usize {
        self.cols
    }

    #[inline(always)]
    fn data(&self) -> &[f64] {
        &self.data
    }

    #[inline(always)]
    fn get(&self, row: usize, col: usize) -> f64 {
        self.data[col * self.rows + row]
    }

    #[inline(always)]
    fn set(&mut self, row: usize, col: usize, value: f64) {
        self.data[col * self.rows + row] = value;
    }
}

// Type alias for backwards compatibility during migration
pub type Matrix = RowMajorMatrix;

impl RowMajorMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn random(rows: usize, cols: usize) -> Self {
        let mut rng = thread_rng();
        let data: Vec<f64> = (0..rows * cols)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();

        Self { rows, cols, data }
    }

    pub fn from_data(data: Vec<f64>, rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols);
        Self { rows, cols, data }
    }

    /// Convert to column-major layout
    pub fn to_col_major(&self) -> ColMajorMatrix {
        let mut col_data = vec![0.0; self.rows * self.cols];

        for row in 0..self.rows {
            for col in 0..self.cols {
                let row_idx = row * self.cols + col;
                let col_idx = col * self.rows + row;
                col_data[col_idx] = self.data[row_idx];
            }
        }

        ColMajorMatrix {
            rows: self.rows,
            cols: self.cols,
            data: col_data,
        }
    }
}

impl ColMajorMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn random(rows: usize, cols: usize) -> Self {
        let mut rng = thread_rng();
        let data: Vec<f64> = (0..rows * cols)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();

        Self { rows, cols, data }
    }

    pub fn from_data(data: Vec<f64>, rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols);
        Self { rows, cols, data }
    }

    /// Convert to row-major layout
    pub fn to_row_major(&self) -> RowMajorMatrix {
        let mut row_data = vec![0.0; self.rows * self.cols];

        for row in 0..self.rows {
            for col in 0..self.cols {
                let row_idx = row * self.cols + col;
                let col_idx = col * self.rows + row;
                row_data[row_idx] = self.data[col_idx];
            }
        }

        RowMajorMatrix {
            rows: self.rows,
            cols: self.cols,
            data: row_data,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_row_major_creation() {
        let matrix = RowMajorMatrix::new(3, 2);
        assert_eq!(matrix.rows(), 3);
        assert_eq!(matrix.cols(), 2);

        // All elements should be zero
        for row in 0..3 {
            for col in 0..2 {
                assert_eq!(matrix.get(row, col), 0.0);
            }
        }
    }

    #[test]
    fn test_col_major_creation() {
        let matrix = ColMajorMatrix::new(3, 2);
        assert_eq!(matrix.rows(), 3);
        assert_eq!(matrix.cols(), 2);

        // All elements should be zero
        for row in 0..3 {
            for col in 0..2 {
                assert_eq!(matrix.get(row, col), 0.0);
            }
        }
    }

    #[test]
    fn test_row_major_get_set() {
        let mut matrix = RowMajorMatrix::new(2, 2);
        matrix.set(0, 1, 42.0);
        assert_eq!(matrix.get(0, 1), 42.0);
        assert_eq!(matrix.get(1, 0), 0.0);
    }

    #[test]
    fn test_col_major_get_set() {
        let mut matrix = ColMajorMatrix::new(2, 2);
        matrix.set(0, 1, 42.0);
        assert_eq!(matrix.get(0, 1), 42.0);
        assert_eq!(matrix.get(1, 0), 0.0);
    }

    #[test]
    fn test_from_data_row_major() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let matrix = RowMajorMatrix::from_data(data, 2, 2);

        assert_eq!(matrix.get(0, 0), 1.0);
        assert_eq!(matrix.get(0, 1), 2.0);
        assert_eq!(matrix.get(1, 0), 3.0);
        assert_eq!(matrix.get(1, 1), 4.0);
    }

    #[test]
    fn test_from_data_col_major() {
        let data = vec![1.0, 3.0, 2.0, 4.0]; // Column-major: [col0, col1]
        let matrix = ColMajorMatrix::from_data(data, 2, 2);

        assert_eq!(matrix.get(0, 0), 1.0);
        assert_eq!(matrix.get(0, 1), 2.0);
        assert_eq!(matrix.get(1, 0), 3.0);
        assert_eq!(matrix.get(1, 1), 4.0);
    }

    #[test]
    fn test_layout_conversion() {
        // Create row-major matrix: [[1,2,3],[4,5,6]]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let row_matrix = RowMajorMatrix::from_data(data, 2, 3);

        // Convert to column-major
        let col_matrix = row_matrix.to_col_major();

        // Should have same logical values
        for row in 0..2 {
            for col in 0..3 {
                assert_eq!(row_matrix.get(row, col), col_matrix.get(row, col));
            }
        }

        // Convert back to row-major
        let back_to_row = col_matrix.to_row_major();

        // Should match original
        for row in 0..2 {
            for col in 0..3 {
                assert_eq!(row_matrix.get(row, col), back_to_row.get(row, col));
            }
        }
    }

    #[test]
    #[should_panic]
    fn test_from_data_wrong_size() {
        let data = vec![1.0, 2.0, 3.0]; // 3 elements
        RowMajorMatrix::from_data(data, 2, 2); // But claiming 2x2 = 4 elements
    }

    #[test]
    fn test_get_row() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [[1,2,3], [4,5,6]]
        let matrix = RowMajorMatrix::from_data(data, 2, 3);

        let row0 = matrix.get_row(0);
        assert_eq!(row0, vec![1.0, 2.0, 3.0]);

        let row1 = matrix.get_row(1);
        assert_eq!(row1, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_get_col() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [[1,2,3], [4,5,6]]
        let matrix = RowMajorMatrix::from_data(data, 2, 3);

        let col0 = matrix.get_col(0);
        assert_eq!(col0, vec![1.0, 4.0]);

        let col1 = matrix.get_col(1);
        assert_eq!(col1, vec![2.0, 5.0]);

        let col2 = matrix.get_col(2);
        assert_eq!(col2, vec![3.0, 6.0]);
    }

    #[test]
    #[should_panic]
    fn test_get_row_out_of_bounds() {
        let matrix = RowMajorMatrix::new(2, 3);
        matrix.get_row(2); // Should panic
    }

    #[test]
    #[should_panic]
    fn test_get_col_out_of_bounds() {
        let matrix = RowMajorMatrix::new(2, 3);
        matrix.get_col(3); // Should panic
    }
}