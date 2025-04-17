#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "arm")]
use std::arch::arm::*;

use std::ops::Add;

use super::utils::SimdVec;

pub const SIZE: usize = 4;

/// A SIMD vector of 4 32-bit floating point values
/// This provides a cross-platform abstraction over architecture-specific SIMD types
#[derive(Copy, Clone, Debug)]
pub struct F32x4 {
    size: usize,

    #[cfg(target_arch = "x86_64")]
    elements: __m128,

    #[cfg(target_arch = "aarch64")]
    pub elements: std::arch::aarch64::float32x4_t,

    #[cfg(target_arch = "arm")]
    elements: std::arch::arm::float32x4_t,

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "arm")))]
    elements: [f32; 4],
}

impl SimdVec<f32> for F32x4 {
    #[inline(always)]
    fn new(slice: &[f32]) -> Self {
        match slice.len().cmp(&SIZE) {
            std::cmp::Ordering::Less => unsafe { Self::load_partial(slice.as_ptr(), slice.len()) },
            std::cmp::Ordering::Equal | std::cmp::Ordering::Greater => unsafe {
                Self::load(slice.as_ptr(), SIZE)
            },
        }
    }

    #[inline(always)]
    fn splat(value: f32) -> Self {
        #[cfg(target_arch = "x86_64")]
        let splat = Self {
            elements: unsafe { _mm_set1_ps(value) },
            size: SIZE,
        };

        #[cfg(target_arch = "aarch64")]
        let splat = Self {
            elements: unsafe { vdupq_n_f32(value) },
            size: SIZE,
        };

        splat
    }

    #[inline(always)]
    unsafe fn load(ptr: *const f32, size: usize) -> Self {
        let msg = format!("Size must be == {}", SIZE);
        assert!(size == SIZE, "{}", msg);

        #[cfg(target_arch = "x86_64")]
        let loaded = Self {
            elements: unsafe { _mm_loadu_ps(ptr) },
            size,
        };

        #[cfg(target_arch = "aarch64")]
        let loaded = Self {
            elements: unsafe { vld1q_f32(ptr) },
            size,
        };

        loaded
    }

    #[inline(always)]
    unsafe fn load_partial(ptr: *const f32, size: usize) -> Self {
        let msg = format!("Size must be < {}", SIZE);
        assert!(size < SIZE, "{}", msg);

        #[cfg(target_arch = "x86_64")]
        let elements = match size {
            1 => unsafe { _mm_set_ps(0.0, 0.0, 0.0, *ptr.add(0)) },
            2 => unsafe { _mm_set_ps(0.0, 0.0, *ptr.add(1), *ptr.add(0)) },
            3 => unsafe { _mm_set_ps(0.0, *ptr.add(2), *ptr.add(1), *ptr.add(0)) },
            _ => {
                let msg = "WTF is happening here";
                panic!("{}", msg);
            }
        };

        #[cfg(target_arch = "aarch64")]
        let elements = unsafe {
            match size {
                1 => {
                    let v = vdupq_n_f32(0.0);
                    vsetq_lane_f32(*ptr.add(0), v, 0)
                }
                2 => {
                    let mut v = vdupq_n_f32(0.0);
                    v = vsetq_lane_f32(*ptr.add(0), v, 0);
                    vsetq_lane_f32(*ptr.add(1), v, 1)
                }
                3 => {
                    let mut v = vdupq_n_f32(0.0);
                    v = vsetq_lane_f32(*ptr.add(0), v, 0);
                    v = vsetq_lane_f32(*ptr.add(1), v, 1);
                    vsetq_lane_f32(*ptr.add(2), v, 2)
                }
                _ => panic!("WTF is happening here"),
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

    #[inline(always)]
    fn store(&self) -> Vec<f32> {
        let msg = format!("Size must be <= {}", SIZE);

        assert!(self.size <= SIZE, "{}", msg);

        let mut vec = vec![0f32; SIZE];

        #[cfg(target_arch = "x86_64")]
        unsafe {
            _mm_storeu_ps(vec.as_mut_ptr(), self.elements);
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            vst1q_f32(vec.as_mut_ptr(), self.elements);
        }

        vec
    }

    #[inline(always)]
    unsafe fn store_at(&self, ptr: *mut f32) {
        let msg = format!("Size must be <= {}", SIZE);

        assert!(self.size <= SIZE, "{}", msg);

        #[cfg(target_arch = "x86_64")]
        unsafe {
            println!("size {}", self.size);

            _mm_storeu_ps(ptr, self.elements);
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            vst1q_f32(ptr, self.elements);
        }
    }

    #[inline(always)]
    unsafe fn store_at_partial(&self, ptr: *mut f32) {
        let msg = format!("Size must be <= {}", SIZE);

        assert!(self.size <= SIZE, "{}", msg);

        println!("size {}", self.size);

        #[cfg(all(sse, target_arch = "x86_64"))]
        match self.size {
            3 => {
                // Store the lower 2 floats using MMX store
                _mm_storel_pi(ptr as *mut __m64, self.elements);

                // Extract the 3rd float using shuffle
                let third = _mm_extract_ps(self.elements, 2) as u32;
                *(ptr.add(2)) = core::mem::transmute(third);
            }
            2 => {
                _mm_storel_pi(ptr as *mut __m64, self.elements);
            }
            1 => {
                _mm_store_ss(ptr, self.elements);
            }
            _ => {
                let msg = "WTF is happening here";
                panic!("{}", msg);
            }
        }

        #[cfg(target_arch = "aarch64")]
        match self.size {
            3 => {
                let low = vget_low_f32(self.elements); // extract [0, 1]
                vst1_f32(ptr, low); // store [0, 1]

                let third = vgetq_lane_f32(self.elements, 2); // extract element 2
                *ptr.add(2) = third; // store element 2
            }
            2 => {
                let low = vget_low_f32(self.elements);
                vst1_f32(ptr, low);
            }
            1 => {
                let first = vgetq_lane_f32(self.elements, 0);
                *ptr = first;
            }
            _ => {
                let msg = "WTF is happening here";
                panic!("{}", msg);
            }
        }
    }

    #[inline(always)]
    fn store_partial(&self) -> Vec<f32> {
        match self.size {
            1..=3 => self.store().into_iter().take(self.size).collect(),
            _ => {
                let msg = "WTF is happening here";
                panic!("{}", msg);
            }
        }
    }

    #[inline(always)]
    fn simd_mask_add(&self, rhs: Self) -> Self {
        self.simd_add(rhs)
    }

    #[inline(always)]
    fn simd_add(&self, rhs: Self) -> Self {
        let msg = format!("Operands must have the same size {}", self.size);
        assert!(self.size == rhs.size, "{}", msg);

        unsafe {
            // Add a+b
            #[cfg(target_arch = "x86_64")]
            let elements = _mm_add_ps(self.elements, rhs.elements);

            #[cfg(target_arch = "aarch64")]
            let elements = vaddq_f32(self.elements, rhs.elements);

            Self {
                elements,
                size: self.size,
            }
        }
    }

    #[inline(always)]
    fn simd_sin(&self) {
        todo!()
    }
}

/// Implementation of Add trait for F32x4 using custom SIMD types
impl Add for F32x4 {
    type Output = F32x4;

    #[inline(always)]
    fn add(self, rhs: F32x4) -> Self::Output {
        let msg = format!("Operands must have the same size {}", SIZE);

        assert!(self.size == rhs.size, "{}", msg);

        match self.size.cmp(&SIZE) {
            std::cmp::Ordering::Less => self.simd_mask_add(rhs),
            std::cmp::Ordering::Equal => self.simd_add(rhs),
            std::cmp::Ordering::Greater => {
                let msg = format!("F32x16 size must not exceed {}", SIZE);
                panic!("{}", msg);
            }
        }
    }
}
