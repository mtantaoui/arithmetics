use arithmetics::ops::add::SimdAdd;

#[cfg(all(rustc_channel = "nightly", avx512))]
fn f32x16_avx512_nightly() {
    use arithmetics::ops::add::SimdAdd;

    #[cfg(rustc_channel = "nightly")]
    {
        let n: usize = 20;

        let a = vec![4.0f32; n];
        let b = vec![2.0f32; n];

        let res = a.as_slice().simd_add(b.as_slice());

        println!("{:?}", res);
        println!("len : {:?}", res.len())
    }

    #[cfg(rustc_channel = "stable")]
    {
        println!("AVX512 is not available for stable build")
    }
}

fn test(n: usize) {
    // let a = vec![1f32; n];
    // let b = vec![1f32; n];

    let a: Vec<f32> = (1..=n).map(|i| i as f32).collect();
    let b: Vec<f32> = (1..=n).map(|i| i as f32).collect();

    #[cfg(all(avx512, rustc_channel = "nightly"))]
    let res = a.as_slice().simd_add(b.as_slice());

    #[cfg(avx2)]
    let res = a.as_slice().simd_add(b.as_slice());

    #[cfg(neon)]
    let res = a.as_slice().simd_add(b.as_slice());

    #[cfg(sse)]
    let res = a.as_slice().simd_add(b.as_slice());

    println!("{:?}", res);
    println!("len: {:?} \n", res.len())
}

fn main() {
    (1..=27).for_each(test);
    // test(1_000_000_000)
}
