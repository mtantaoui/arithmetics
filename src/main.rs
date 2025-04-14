use arithmetics::ops::add::SimdAdd;

#[cfg(all(avx512, rustc_channel = "nightly"))]
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

fn main() {
    let n: usize = 9;

    // let a = vec![4.0f32; n];
    // let b = vec![2.0f32; n];
    let a: Vec<f32> = (1..=n).map(|i| i as f32).collect();
    let b: Vec<f32> = (1..=n).map(|i| i as f32).collect();

    #[cfg(all(avx512, rustc_channel = "nightly"))]
    let res = a.as_slice().simd_add(b.as_slice());

    #[cfg(avx2)]
    let res = a.as_slice().simd_add(b.as_slice());

    println!("{:?}", res);
    println!("len : {:?}", res.len())
}
