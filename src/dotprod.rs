pub fn naive_dotprod(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Vector dimensions don't match");
    
    let mut sum = 0.0;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

pub fn unrolled_dotprod(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Vector dimensions don't match");
    
    let len = a.len();
    let mut sum = 0.0;
    
    let chunks = len / 4;

    for i in 0..chunks {
        let base = i * 4;
        sum += a[base] * b[base]
            + a[base + 1] * b[base + 1]
            + a[base + 2] * b[base + 2]
            + a[base + 3] * b[base + 3];
    }
    
    for i in (chunks * 4)..len {
        sum += a[i] * b[i];
    }
    
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_naive_dotprod() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = naive_dotprod(&a, &b);
        assert_eq!(result, 32.0); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    }

    #[test]
    fn test_unrolled_dotprod() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let result = unrolled_dotprod(&a, &b);
        assert_eq!(result, 70.0); // 1*2 + 2*3 + 3*4 + 4*5 + 5*6 = 2 + 6 + 12 + 20 + 30 = 70
    }

    #[test]
    fn test_both_implementations_match() {
        let a = vec![1.5, -2.0, 3.7, 0.5, -1.2, 4.1];
        let b = vec![2.1, 1.0, -0.5, 3.2, 2.7, -1.8];
        
        let naive_result = naive_dotprod(&a, &b);
        let unrolled_result = unrolled_dotprod(&a, &b);
        
        assert!((naive_result - unrolled_result).abs() < 1e-10);
    }

    #[test]
    #[should_panic]
    fn test_mismatched_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        naive_dotprod(&a, &b);
    }

}
