use std::ops::Add;

use arithmetics::add::SimdAdd;

#[cfg(avx512)]
fn avx512() {
    #[cfg(rustc_channel = "nightly")]
    {
        let n: usize = 20_000_000;

        let a = vec![4.0f32; n];
        let b = vec![2.0f32; n];

        // let _ = a.add(b);
        // let _ = a.fmadd(b);
    }

    #[cfg(rustc_channel = "stable")]
    {
        println!("AVX512 is not available for nightly mode")
    }
}

fn main() {
    println!("Running with the following CPU features:");

    // Display detected CPU features
    #[cfg(all(avx512, rustc_channel = "nightly"))]
    {
        println!("  - Using AVX-512F optimized implementation ✓");
        avx512();
    }

    #[cfg(all(avx512, rustc_channel = "stable"))]
    println!("  - AVX-512F optimized implementation is not available ✓");

    #[cfg(avx2)]
    println!("  - Using AVX2 optimized implementation ✓");

    #[cfg(sse)]
    println!("  - Using SSE4.1 optimized implementation ✓");

    #[cfg(baseline)]
    println!("  - Using baseline implementation (no SIMD optimizations)");
}
