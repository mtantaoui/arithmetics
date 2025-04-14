#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::ops::Add;

use super::utils::SimdVec;

pub const SIZE: usize = 4;

/// A SIMD vector of 4 32-bit floating point values
/// This provides a cross-platform abstraction over architecture-specific SIMD types
#[derive(Copy, Clone)]
pub struct F32x4 {
    size: usize,

    #[cfg(target_arch = "x86_64")]
    elements: __m128,

    #[cfg(target_arch = "aarch64")]
    elements: std::arch::aarch64::float32x4_t,

    #[cfg(target_arch = "arm")]
    elements: std::arch::arm::float32x4_t,

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "arm")))]
    elements: [f32; 4],
}

impl SimdVec<f32> for F32x4 {
    fn new(slice: &[f32]) -> Self {
        if slice.len() == SIZE {
            unsafe { Self::load(slice.as_ptr(), slice.len()) }
        } else if slice.len() < SIZE {
            unsafe { Self::load_partial(slice.as_ptr(), slice.len()) }
        } else {
            let msg = format!("F32x4 size must not exceed {}", SIZE);
            panic!("{}", msg);
        }
    }

    fn splat(value: f32) -> Self {
        Self {
            elements: unsafe { _mm_set1_ps(value) },
            size: SIZE,
        }
    }

    #[inline(always)]
    unsafe fn load(ptr: *const f32, size: usize) -> Self {
        let msg = format!("Size must be == {}", SIZE);
        assert!(size == SIZE, "{}", msg);

        Self {
            elements: unsafe { _mm_loadu_ps(ptr) },
            size,
        }
    }

    #[inline(always)]
    unsafe fn load_partial(ptr: *const f32, size: usize) -> Self {
        let msg = format!("Size must be < {}", SIZE);
        assert!(size < SIZE, "{}", msg);

        let elements = match size {
            1 => unsafe { _mm_set_ps(0.0, 0.0, 0.0, *ptr.add(0)) },
            2 => unsafe { _mm_set_ps(0.0, 0.0, *ptr.add(1), *ptr.add(0)) },
            3 => unsafe { _mm_set_ps(0.0, *ptr.add(2), *ptr.add(1), *ptr.add(0)) },
            _ => {
                let msg = "WTF is happening here";
                panic!("{}", msg);
            }
        };

        Self { elements, size }
    }

    #[inline(always)]
    fn to_vec(self) -> Vec<f32> {
        let msg = format!("Size must be <= {}", SIZE);
        assert!(self.size <= SIZE, "{}", msg);

        if self.size == SIZE {
            self.store()
        } else {
            self.store_partial()
        }
    }

    fn store(&self) -> Vec<f32> {
        let msg = format!("Size must be <= {}", SIZE);

        assert!(self.size <= SIZE, "{}", msg);

        let mut vec = vec![0f32; SIZE];

        unsafe {
            _mm_storeu_ps(vec.as_mut_ptr(), self.elements);
        }

        vec
    }

    fn store_partial(&self) -> Vec<f32> {
        match self.size {
            1..=3 => self.store().into_iter().take(self.size).collect(),
            _ => {
                let msg = "WTF is happening here";
                panic!("{}", msg);
            }
        }
    }

    fn simd_mask_add(&self, rhs: Self) -> Self {
        self.simd_add(rhs)
    }

    fn simd_add(&self, rhs: Self) -> Self {
        let msg = format!("Operands must have the same size {}", self.size);
        assert!(self.size == rhs.size, "{}", msg);

        unsafe {
            // Add a+b
            let elements = _mm_add_ps(self.elements, rhs.elements);

            Self {
                elements,
                size: self.size,
            }
        }
    }
}

/// Implementation of Add trait for F32x4 using custom SIMD types
impl Add for F32x4 {
    type Output = F32x4;

    fn add(self, rhs: F32x4) -> Self::Output {
        let msg = format!("Operands must have the same size {}", SIZE);

        assert!(self.size == rhs.size, "{}", msg);

        if self.size == SIZE {
            self.simd_add(rhs)
        } else if self.size < SIZE {
            self.simd_mask_add(rhs)
        } else {
            let msg = format!("F32x16 size must not exceed {}", SIZE);
            panic!("{}", msg);
        }
    }
}
