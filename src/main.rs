#![feature(stdarch_x86_avx512)]

use std::arch::x86_64::*;

use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

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
    let n: usize = 100;

    let a = vec![4.0f32; n];
    let b = vec![2.0f32; n];

    // let result = fmadd(&a, &b);
    let result = add(&a, &b);

    for ((res, a), b) in result
        .as_slice()
        .chunks(16)
        .zip(a.as_slice().chunks(16))
        .zip(b.as_slice().chunks(16))
    {
        println!("{:?}", a);
        println!("{:?}", b);
        println!("{:?}", res);
        println!();
    }

    println!(
        "Dot product result: {:?} --> {}",
        result.first(),
        result.len()
    );
    println!("Dot product result:  --> {}", result.len());
}

// Public function that uses the best available implementation (selected at compile time)
pub fn fmadd(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "Vectors must be the same length");

    // The compiler will select only one of these implementations
    // based on the cfg flags set by build.rs

    #[cfg(avx512)]
    return fmadd_avx512f(a.to_vec(), b.to_vec());

    #[cfg(avx2)]
    return dot_product_avx2(a, b);

    #[cfg(sse)]
    return dot_product_sse41(a, b);

    #[cfg(baseline)]
    return dot_product_baseline(a, b);
}

// Public function that uses the best available implementation (selected at compile time)
pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "Vectors must be the same length");

    // The compiler will select only one of these implementations
    // based on the cfg flags set by build.rs
    #[cfg(avx512)]
    return add_avx512f(a.to_vec(), b.to_vec());
}

#[rustversion::nightly]
#[cfg(avx512)]
#[inline(always)]
fn fmadd_f32x16(a_chunk: Vec<f32>, b_chunk: Vec<f32>) -> Vec<f32> {
    unsafe {
        assert_eq!(
            a_chunk.len(),
            b_chunk.len(),
            "Vectors must be the same length"
        );
        assert!(
            a_chunk.len() == 16,
            "Chunk length must be == 16 for AVX-512"
        );

        let chunk_size = a_chunk.len();

        let a = _mm512_loadu_ps(a_chunk.as_ptr());
        let b = _mm512_loadu_ps(b_chunk.as_ptr());

        // Fused multiply-add: a * b + 0.0
        let c = _mm512_fmadd_ps(a, b, _mm512_setzero_ps());

        let mut result = vec![0f32; chunk_size];
        _mm512_storeu_ps(result.as_mut_ptr(), c);

        result
    }
}

#[rustversion::nightly]
#[cfg(avx512)]
#[inline(always)]
fn fmadd_f32x16_partial(a_chunk: Vec<f32>, b_chunk: Vec<f32>) -> Vec<f32> {
    unsafe {
        assert_eq!(
            a_chunk.len(),
            b_chunk.len(),
            "Vectors must be the same length"
        );
        assert!(
            a_chunk.len() <= 16,
            "Chunk length must be <= 16 for AVX-512"
        );

        let chunk_size = a_chunk.len();
        let mask: __mmask16 = (1 << chunk_size) - 1;

        let a = _mm512_maskz_loadu_ps(mask, a_chunk.as_ptr());
        let b = _mm512_maskz_loadu_ps(mask, b_chunk.as_ptr());

        // Fused multiply-add: a * b + 0.0
        let c = _mm512_maskz_fmadd_ps(mask, a, b, _mm512_setzero_ps());

        let mut result = vec![0f32; chunk_size];
        _mm512_mask_storeu_ps(result.as_mut_ptr(), mask, c);

        result
    }
}

// Implementation functions (only one will be compiled into the final binary)
#[rustversion::nightly]
#[cfg(avx512)]
#[inline(always)]
fn fmadd_avx512f(a: Vec<f32>, b: Vec<f32>) -> Vec<f32> {
    let chunk_size = 16;

    let sum: Vec<f32> = a
        .into_par_iter()
        .chunks(chunk_size)
        .zip_eq(b.into_par_iter().chunks(chunk_size))
        .map(|(a_chunk, b_chunk)| {
            if a_chunk.len() == chunk_size {
                fmadd_f32x16(a_chunk, b_chunk)
            } else {
                fmadd_f32x16_partial(a_chunk, b_chunk)
            }
        })
        .flatten()
        .collect();

    sum
}

#[rustversion::nightly]
#[cfg(avx512)]
#[inline(always)]
fn add_f32x16(a_chunk: &[f32], b_chunk: &[f32]) -> Vec<f32> {
    unsafe {
        assert_eq!(
            a_chunk.len(),
            b_chunk.len(),
            "Vectors must be the same length"
        );
        assert!(
            a_chunk.len() == 16,
            "Chunk length must be == 16 for AVX-512"
        );

        let chunk_size = a_chunk.len();

        let a = _mm512_loadu_ps(a_chunk.as_ptr());
        let b = _mm512_loadu_ps(b_chunk.as_ptr());

        // Fused multiply-add: a * b + 0.0
        let c = _mm512_add_ps(a, b);

        let mut result = vec![0f32; chunk_size];
        _mm512_storeu_ps(result.as_mut_ptr(), c);

        result
    }
}

#[rustversion::nightly]
#[cfg(avx512)]
#[inline(always)]
fn add_f32x16_partial(a_chunk: &[f32], b_chunk: &[f32]) -> Vec<f32> {
    unsafe {
        assert_eq!(
            a_chunk.len(),
            b_chunk.len(),
            "Vectors must be the same length"
        );
        assert!(
            a_chunk.len() <= 16,
            "Chunk length must be <= 16 for AVX-512"
        );

        let chunk_size = a_chunk.len();
        let mask: __mmask16 = (1 << chunk_size) - 1;

        let a = _mm512_maskz_loadu_ps(mask, a_chunk.as_ptr());
        let b = _mm512_maskz_loadu_ps(mask, b_chunk.as_ptr());

        // Fused multiply-add: a * b + 0.0
        let c = _mm512_maskz_add_ps(mask, a, b);

        let mut result = vec![0f32; chunk_size];
        _mm512_mask_storeu_ps(result.as_mut_ptr(), mask, c);

        result
    }
}

#[rustversion::nightly]
#[cfg(avx512)]
#[inline(always)]
fn add_avx512f(a: Vec<f32>, b: Vec<f32>) -> Vec<f32> {
    let chunk_size = 16;

    let sum: Vec<f32> = a
        .into_par_iter()
        .chunks(chunk_size)
        .zip_eq(b.into_par_iter().chunks(chunk_size))
        .map(|(a_chunk, b_chunk)| {
            if a_chunk.len() == chunk_size {
                add_f32x16(a_chunk.as_slice(), b_chunk.as_slice())
            } else {
                add_f32x16_partial(a_chunk.as_slice(), b_chunk.as_slice())
            }
        })
        .flatten()
        .collect();

    sum
}
// Implementation functions (only one will be compiled into the final binary)
#[rustversion::stable]
#[cfg(avx512)]
#[inline(always)]
fn dot_product_avx512f(a: Vec<f32>, b: Vec<f32>) -> Vec<f32> {
    vec![0.0; 4]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_add() {
        // Example operation: vector dot product
        let n: usize = 100;

        let a = vec![4.0f32; n];
        let b = vec![2.0f32; n];
        let c = vec![6.0f32; n];

        let result = add(&a, &b);

        assert_eq!(result, c);
    }

    // #[test]
    // fn test_vector_fused_multiply_add() {
    //     // Example operation: vector dot product
    //     let n: usize = 100;

    //     let a = vec![4.0f32; n];
    //     let b = vec![2.0f32; n];
    //     let c = vec![8.0f32; n];

    //     let result = fmadd(&a, &b);

    //     assert_eq!(result, c);
    // }
}
