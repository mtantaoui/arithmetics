#[cfg(all(avx512, rustc_channel = "nightly"))]
pub(crate) mod f32x16_nightly;

pub mod utils;

pub mod f32x4;
