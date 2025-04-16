use rayon::prelude::*;

#[cfg(all(avx512, rustc_channel = "nightly"))]
use crate::simd::f32x16_nightly::{F32x16, SIZE};

#[cfg(sse)]
use crate::simd::f32x4::{F32x4, SIZE};

#[cfg(neon)]
use crate::simd::f32x4::{F32x4, SIZE};

#[cfg(avx2)]
use crate::simd::f32x8::{F32x8, SIZE};

use crate::simd::utils::SimdVec;

pub trait SimdAdd<Rhs = Self> {
    type Output;

    fn simd_add(self, rhs: Rhs) -> Self::Output;
}

#[cfg(all(avx512, rustc_channel = "nightly"))]
fn add_avx512_nightly(a: &[f32], b: &[f32]) {
    let chunk_size = SIZE;

    let addition: Vec<f32> = a
        .par_chunks(chunk_size)
        .zip_eq(b.par_chunks(chunk_size))
        .map(|(a_chunk, b_chunk)| {
            let a = F32x16::new(a_chunk);
            let b = F32x16::new(b_chunk);

            let c = a + b;

            c.to_vec()
        })
        .flatten()
        .collect();

    addition
}

#[cfg(sse)]
fn add_sse(a: &[f32], b: &[f32]) -> Vec<f32> {
    let chunk_size = SIZE;

    let addition: Vec<f32> = a
        .par_chunks(chunk_size)
        .zip_eq(b.par_chunks(chunk_size))
        .map(|(a_chunk, b_chunk)| {
            let a = F32x4::new(a_chunk);
            let b = F32x4::new(b_chunk);

            let c = a + b;

            c.to_vec()
        })
        .flatten()
        .collect();

    addition
}

#[cfg(avx2)]
fn add_avx2(a: &[f32], b: &[f32]) -> Vec<f32> {
    let chunk_size = SIZE;

    let addition: Vec<f32> = a
        .par_chunks(chunk_size)
        .zip_eq(b.par_chunks(chunk_size))
        .map(|(a_chunk, b_chunk)| {
            let a = F32x8::new(a_chunk);
            let b = F32x8::new(b_chunk);

            let c = a + b;

            c.to_vec()
        })
        .flatten()
        .collect();

    addition
}

#[cfg(neon)]
fn add_neon(a: &[f32], b: &[f32]) -> Vec<f32> {
    let chunk_size = SIZE;

    let addition: Vec<f32> = a
        .par_chunks(chunk_size)
        .zip_eq(b.par_chunks(chunk_size))
        .map(|(a_chunk, b_chunk)| {
            let a = F32x4::new(a_chunk);
            let b = F32x4::new(b_chunk);

            let c = a + b;

            c.to_vec()
        })
        .flatten()
        .collect();

    addition
}

/// Core SIMD addition function (Processes chunks in parallel)
#[inline(always)]
fn add_slices(a: &[f32], b: &[f32]) -> Vec<f32> {
    #[cfg(all(avx512, rustc_channel = "nightly"))]
    let addition = add_avx512_nightly(a, b);

    #[cfg(sse)]
    let addition = add_sse(a, b);

    #[cfg(avx2)]
    let addition = add_avx2(a, b);

    #[cfg(neon)]
    let addition = add_neon(a, b);

    addition
}

impl SimdAdd for Vec<f32> {
    type Output = Vec<f32>;

    fn simd_add(self, rhs: Vec<f32>) -> Self::Output {
        let msg = format!("Operands must have the same size {}", self.len());
        assert!(self.len() == rhs.len(), "{}", msg);

        add_slices(self.as_slice(), rhs.as_slice())
    }
}

impl<'rhsl> SimdAdd<&'rhsl [f32]> for &[f32] {
    type Output = Vec<f32>;

    fn simd_add(self, rhs: &'rhsl [f32]) -> Self::Output {
        let msg = format!("Operands must have the same size {}", self.len());
        assert!(self.len() == rhs.len(), "{}", msg);

        add_slices(self, rhs)
    }
}
