#![cfg_attr(rustc_channel = "nightly", feature(stdarch_x86_avx512))]

#[cfg(all(avx512, rustc_channel = "nightly"))]
pub mod f32x16_nightly;

pub mod simd_add;
