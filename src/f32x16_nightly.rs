use rayon::prelude::*;
use std::arch::x86_64::*;

// const SIZE: usize = 16;

// // Define f32x16 using two f32x8
// #[derive(Copy, Clone, Debug)]
// pub struct F32x16 {
//     size: usize,

//     #[cfg(avx512)]
//     elements: __m512,
// }

// impl F32x16 {
//     #[inline]
//     pub fn splat(value: f32) -> Self {
//         {
//             Self {
//                 elements: unsafe { _mm512_set1_ps(value) },
//                 size: SIZE,
//             }
//         }
//     }

//     #[inline]
//     pub fn load(ptr: *const f32, size: usize) -> Self {
//         let msg = format!("Size must be == {}", SIZE);
//         assert!(size == SIZE, "{}", msg);

//         #[cfg(avx512)]
//         {
//             Self {
//                 elements: unsafe { _mm512_loadu_ps(ptr) },
//                 size: SIZE,
//             }
//         }
//     }

//     #[inline]
//     pub fn load_partial(ptr: *const f32, size: usize) -> Self {
//         let msg = format!("Size must be < {}", SIZE);
//         assert!(size < SIZE, "{}", msg);

//         #[cfg(avx512)]
//         {
//             let mask: __mmask16 = (1 << size) - 1;

//             Self {
//                 elements: unsafe { _mm512_maskz_loadu_ps(mask, ptr) },
//                 size,
//             }
//         }
//     }

//     #[inline]
//     pub fn to_vec(&self) -> Vec<f32> {
//         let msg = format!("Size must be <= {}", SIZE);
//         assert!(self.size <= SIZE, "{}", msg);

//         #[cfg(avx512)]
//         {
//             let mut vec = vec![0f32; self.size];

//             unsafe {
//                 if self.size == SIZE {
//                     _mm512_storeu_ps(vec.as_mut_ptr(), self.elements);
//                 } else {
//                     let mask: __mmask16 = (1 << self.size) - 1;
//                     _mm512_mask_storeu_ps(vec.as_mut_ptr(), mask, self.elements);
//                 }
//             }

//             vec
//         }
//     }

//     #[inline]
//     fn store(&self) -> [f32; 16] {
//         let msg = format!("Size must be == {}", SIZE);

//         assert!(self.size == SIZE, "{}", msg);

//         #[cfg(avx512)]
//         {
//             let mut array = [0f32; SIZE];

//             unsafe {
//                 _mm512_storeu_ps(array.as_mut_ptr(), self.elements);
//             }

//             array
//         }
//     }

//     #[inline]
//     fn store_partial(&self) -> std::vec::Vec<f32> {
//         let msg = format!("Size must be < {}", SIZE);

//         assert!(self.size < SIZE, "{}", msg);

//         #[cfg(avx512)]
//         {
//             let mut vec = vec![0f32; self.size];

//             unsafe {
//                 _mm512_storeu_ps(vec.as_mut_ptr(), self.elements);
//             }

//             vec
//         }
//     }
// }

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

#[inline(always)]
pub fn fmadd(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "Vectors must be the same length");

    #[cfg(all(avx512, rustc_channel = "nightly"))]
    return fmadd_avx512f(a.to_vec(), b.to_vec());
}
