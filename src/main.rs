#![feature(stdarch_x86_avx512)]

use std::arch::x86_64::*;

fn main() {
    println!("Running with the following CPU features:");

    // Display detected CPU features
    #[cfg(avx512)]
    println!("  - Using AVX-512F optimized implementation ✓");

    #[cfg(avx2)]
    println!("  - Using AVX2 optimized implementation ✓");

    #[cfg(sse)]
    println!("  - Using SSE4.1 optimized implementation ✓");

    #[cfg(baseline)]
    println!("  - Using baseline implementation (no SIMD optimizations)");

    // Example operation: vector dot product
    let a = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let b = vec![
        2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
    ];

    let result = dot_product(&a, &b);
    println!("Dot product result: {}", result);
}

// Public function that uses the best available implementation (selected at compile time)
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must be the same length");

    // The compiler will select only one of these implementations
    // based on the cfg flags set by build.rs

    #[cfg(avx512)]
    return dot_product_avx512f(a, b);

    #[cfg(avx2)]
    return dot_product_avx2(a, b);

    #[cfg(sse)]
    return dot_product_sse41(a, b);

    #[cfg(baseline)]
    return dot_product_baseline(a, b);
}

// Implementation functions (only one will be compiled into the final binary)

#[cfg(avx512)]
#[inline(always)]
fn dot_product_avx512f(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        let mut sum = _mm512_setzero_ps();

        // Process 16 elements at a time using AVX-512F
        for i in (0..a.len()).step_by(16) {
            if i + 16 <= a.len() {
                let a_vec = _mm512_loadu_ps(a.as_ptr().add(i) as *const f32);
                let b_vec = _mm512_loadu_ps(b.as_ptr().add(i) as *const f32);
                // fmadd: multiply and add in one instruction
                sum = _mm512_fmadd_ps(a_vec, b_vec, sum);
            }
        }

        // Horizontal sum
        let result = _mm512_reduce_add_ps(sum);

        // Handle remaining elements
        let mut scalar_sum = result;
        for i in (a.len() - a.len() % 16)..a.len() {
            scalar_sum += a[i] * b[i];
        }

        scalar_sum
    }
    // println!("Here I am");
    // 0.0
}

#[cfg(avx2)]
#[inline(always)]
fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        let mut sum = _mm256_setzero_ps();

        // Process 8 elements at a time using AVX2
        for i in (0..a.len()).step_by(8) {
            if i + 8 <= a.len() {
                let a_vec = _mm256_loadu_ps(a.as_ptr().add(i) as *const f32);
                let b_vec = _mm256_loadu_ps(b.as_ptr().add(i) as *const f32);
                let mul = _mm256_mul_ps(a_vec, b_vec);
                sum = _mm256_add_ps(sum, mul);
            }
        }

        // Horizontal sum
        let sum_hi = _mm256_permute2f128_ps(sum, sum, 1);
        let sum1 = _mm256_add_ps(sum, sum_hi);
        let sum2 = _mm256_hadd_ps(sum1, sum1);
        let sum3 = _mm256_hadd_ps(sum2, sum2);

        // Extract the result
        let mut result_arr: [f32; 8] = [0.0; 8];
        _mm256_storeu_ps(result_arr.as_mut_ptr(), sum3);

        // Handle remaining elements
        let mut scalar_sum = result_arr[0];
        for i in (a.len() - a.len() % 8)..a.len() {
            scalar_sum += a[i] * b[i];
        }

        scalar_sum
    }
}

#[cfg(sse)]
#[inline(always)]
fn dot_product_sse41(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        let mut sum = _mm_setzero_ps();

        // Process 4 elements at a time using SSE4.1
        for i in (0..a.len()).step_by(4) {
            if i + 4 <= a.len() {
                let a_vec = _mm_loadu_ps(a.as_ptr().add(i) as *const f32);
                let b_vec = _mm_loadu_ps(b.as_ptr().add(i) as *const f32);
                // SSE4.1 has dot product instruction
                let mul = _mm_mul_ps(a_vec, b_vec);
                sum = _mm_add_ps(sum, mul);
            }
        }

        // Horizontal sum
        let sum_hi = _mm_shuffle_ps(sum, sum, 0b10_11_00_01);
        let sum_lo = _mm_add_ps(sum, sum_hi);
        let sum_hi2 = _mm_movehl_ps(sum_lo, sum_lo);
        let final_sum = _mm_add_ss(sum_lo, sum_hi2);

        // Extract the result
        let mut result = 0.0;
        _mm_store_ss(&mut result, final_sum);

        // Handle remaining elements
        let mut scalar_sum = result;
        for i in (a.len() - a.len() % 4)..a.len() {
            scalar_sum += a[i] * b[i];
        }

        scalar_sum
    }
}
#[allow(dead_code)]
#[inline(always)]
fn dot_product_baseline(a: &[f32], b: &[f32]) -> f32 {
    // Pure Rust implementation without SIMD
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

// Additional utility functions for demonstration purposes

// Function to show all available CPU features (useful for debugging)
#[allow(dead_code)]
fn print_all_cpu_features() {
    println!("Detailed CPU feature availability:");

    #[cfg(target_arch = "x86_64")]
    println!("  AVX-512F: {}", is_x86_feature_detected!("avx512f"));
    println!("  AVX2: {}", is_x86_feature_detected!("avx2"));
    println!("  SSE4.1: {}", is_x86_feature_detected!("sse4.1"));
    println!("  SSE2: {}", is_x86_feature_detected!("sse2"));
}
