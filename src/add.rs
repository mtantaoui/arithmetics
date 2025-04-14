use rayon::prelude::*;

use crate::f32x16_nightly::{F32x16, SIZE};

pub trait SimdAdd<Rhs = Self> {
    type Output;

    fn simd_add(self, rhs: Rhs) -> Self::Output;
}

/// Core SIMD addition function (Processes chunks in parallel)
#[inline(always)]
fn add_slices(a: &[f32], b: &[f32]) -> Vec<f32> {
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
