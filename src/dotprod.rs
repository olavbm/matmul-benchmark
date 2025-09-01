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

#[cfg(target_arch = "x86_64")]
pub fn simd_dotprod(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Vector dimensions don't match");
    
    #[cfg(target_feature = "avx2")]
    unsafe {
        simd_dotprod_avx2(a, b)
    }
    #[cfg(not(target_feature = "avx2"))]
    {
        // Fallback to unrolled version
        unrolled_dotprod(a, b)
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
unsafe fn simd_dotprod_avx2(a: &[f64], b: &[f64]) -> f64 {
    use std::arch::x86_64::*;
    
    let len = a.len();
    let simd_len = len & !3; // Round down to multiple of 4
    
    let mut sum_vec = _mm256_setzero_pd();
    
    // Process 4 f64 elements at a time with AVX2
    for i in (0..simd_len).step_by(4) {
        let a_vec = _mm256_loadu_pd(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_pd(b.as_ptr().add(i));
        let prod = _mm256_mul_pd(a_vec, b_vec);
        sum_vec = _mm256_add_pd(sum_vec, prod);
    }
    
    // Sum the 4 elements of sum_vec
    let mut sum_arr = [0.0; 4];
    _mm256_storeu_pd(sum_arr.as_mut_ptr(), sum_vec);
    let mut sum = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];
    
    // Handle remaining elements
    for i in simd_len..len {
        sum += a[i] * b[i];
    }
    
    sum
}

#[cfg(not(target_arch = "x86_64"))]
pub fn simd_dotprod(a: &[f64], b: &[f64]) -> f64 {
    // Fallback for non-x86_64 architectures
    unrolled_dotprod(a, b)
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

    #[test]
    fn test_simd_dotprod() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        
        let simd_result = simd_dotprod(&a, &b);
        let naive_result = naive_dotprod(&a, &b);
        
        assert!((simd_result - naive_result).abs() < 1e-10);
    }

}
