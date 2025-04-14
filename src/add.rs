use rayon::prelude::*;

#[cfg(avx512)]
use crate::f32x16_nightly::{F32x16, SIZE};

pub trait SimdAdd<Rhs = Self> {
    type Output;

    fn add(self, rhs: Rhs) -> Self::Output;
}

// TODO: Think of a way to organize mutliple implementations
// of arithmetics operations for differents architectures
#[inline(always)]
#[cfg(avx512)]
fn add(a: Vec<f32>, b: Vec<f32>) -> Vec<f32> {
    let chunk_size = SIZE;

    let sum: Vec<f32> = a
        .into_par_iter()
        .chunks(chunk_size)
        .zip_eq(b.into_par_iter().chunks(chunk_size))
        .map(|(a_chunk, b_chunk)| {
            if a_chunk.len() < 16 || b_chunk.len() < 16 {
                println!()
            }

            let a = F32x16::new(a_chunk);
            let b = F32x16::new(b_chunk);
            let c = a + b;

            c.to_vec()
        })
        .flatten()
        .collect();

    sum
}

impl SimdAdd for Vec<f32> {
    type Output = Vec<f32>;

    fn add(self, rhs: Vec<f32>) -> Self::Output {
        let msg = format!("Operands must have the same size {}", self.len());
        assert!(self.len() == rhs.len(), "{}", msg);

        add(self, rhs)
    }
}
