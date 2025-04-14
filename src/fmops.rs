use crate::f32x16_nightly::{F32x16, SIZE};
use rayon::prelude::*;

pub trait FusedMultiplyOps {
    type Output;

    /// Computes (a * b) + c in a single operation where possible
    fn fmadd(self, a: Self, b: Self) -> Self;

    /// Computes (a * b) - c in a single operation where possible
    fn fmsub(self, a: Self, b: Self) -> Self;
}

// impl FusedMultiplyOps for Vec<f32> {
//     type Output = Vec<f32>;

//     fn fmadd(self, a: Self, b: Self) -> Self {
//         let chunk_size = SIZE;

//         let fmadd: Vec<f32> = self
//             .into_par_iter()
//             .chunks(chunk_size)
//             .zip_eq(a.into_par_iter().chunks(chunk_size))
//             .zip_eq(b.into_par_iter().chunks(chunk_size))
//             .map(|((c_chunk, a_chunk), b_chunk)| {
//                 let c = F32x16::new(c_chunk);

//                 let a = F32x16::new(a_chunk);
//                 let b = F32x16::new(b_chunk);

//                 c.fmadd(a, b);

//                 c.to_vec()
//             })
//             .flatten()
//             .collect();

//         fmadd
//     }

//     fn fmsub(self, a: Self, b: Self) -> Self {
//         let chunk_size = SIZE;

//         let fmsub: Vec<f32> = self
//             .into_par_iter()
//             .chunks(chunk_size)
//             .zip_eq(a.into_par_iter().chunks(chunk_size))
//             .zip_eq(b.into_par_iter().chunks(chunk_size))
//             .map(|((c_chunk, a_chunk), b_chunk)| {
//                 let c = F32x16::new(c_chunk);

//                 let a = F32x16::new(a_chunk);
//                 let b = F32x16::new(b_chunk);

//                 c.fmsub(a, b);

//                 c.to_vec()
//             })
//             .flatten()
//             .collect();

//         fmsub
//     }
// }
