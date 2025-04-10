use std::arch::x86_64::*;

const SIZE: usize = 16;

// Define f32x16 using two f32x8
#[derive(Copy, Clone, Debug)]
pub struct F32x16 {
    size: usize,

    #[cfg(avx512)]
    elements: __m512,
}

impl F32x16 {
    #[rustversion::nightly]
    #[inline]
    pub fn splat(value: f32) -> Self {
        #[cfg(avx512)]
        {
            Self {
                elements: unsafe { _mm512_set1_ps(value) },
                size: SIZE,
            }
        }
    }

    #[rustversion::nightly]
    #[inline]
    pub fn load(ptr: *const f32, size: usize) -> Self {
        let msg = format!("Size must be == {}", SIZE);
        assert!(size == SIZE, "{}", msg);

        #[cfg(avx512)]
        {
            Self {
                elements: unsafe { _mm512_loadu_ps(ptr) },
                size: SIZE,
            }
        }
    }

    #[rustversion::nightly]
    #[inline]
    pub fn load_partial(ptr: *const f32, size: usize) -> Self {
        let msg = format!("Size must be < {}", SIZE);
        assert!(size < SIZE, "{}", msg);

        #[cfg(avx512)]
        {
            let mask: __mmask16 = (1 << size) - 1;

            Self {
                elements: unsafe { _mm512_maskz_loadu_ps(mask, ptr) },
                size,
            }
        }
    }

    #[rustversion::nightly]
    #[inline]
    pub fn to_vec(&self) -> Vec<f32> {
        let msg = format!("Size must be <= {}", SIZE);
        assert!(self.size <= SIZE, "{}", msg);

        #[cfg(avx512)]
        {
            let mut vec = vec![0f32; self.size];

            unsafe {
                if self.size == SIZE {
                    _mm512_storeu_ps(vec.as_mut_ptr(), self.elements);
                } else {
                    let mask: __mmask16 = (1 << self.size) - 1;
                    _mm512_mask_storeu_ps(vec.as_mut_ptr(), mask, self.elements);
                }
            }

            vec
        }
    }

    #[rustversion::nightly]
    #[inline]
    fn store(&self) -> [f32; 16] {
        let msg = format!("Size must be == {}", SIZE);

        assert!(self.size == SIZE, "{}", msg);

        #[cfg(avx512)]
        {
            let mut array = [0f32; SIZE];

            unsafe {
                _mm512_storeu_ps(array.as_mut_ptr(), self.elements);
            }

            array
        }
    }

    #[rustversion::nightly]
    #[inline]
    fn store_partial(&self) -> std::vec::Vec<f32> {
        let msg = format!("Size must be < {}", SIZE);

        assert!(self.size < SIZE, "{}", msg);

        #[cfg(avx512)]
        {
            let mut vec = vec![0f32; self.size];

            unsafe {
                _mm512_storeu_ps(vec.as_mut_ptr(), self.elements);
            }

            vec
        }
    }
}
