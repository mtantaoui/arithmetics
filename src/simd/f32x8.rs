use crate::simd::f32x4;

#[cfg(not(target_arch = "x86_64"))]
use super::{f32x4::F32x4, utils::SimdVec};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "arm")]
use std::arch::arm::*;

use std::ops::Add;

pub const SIZE: usize = 8;

#[derive(Copy, Clone, Debug)]
pub struct F32x8 {
    size: usize,

    #[cfg(target_arch = "x86_64")]
    elements: __m256,

    #[cfg(not(target_arch = "x86_64"))]
    low: F32x4,
    #[cfg(not(target_arch = "x86_64"))]
    high: F32x4,
}

impl SimdVec<f32> for F32x8 {
    fn new(slice: &[f32]) -> Self {
        match slice.len().cmp(&SIZE) {
            std::cmp::Ordering::Less => unsafe { Self::load_partial(slice.as_ptr(), slice.len()) },
            std::cmp::Ordering::Equal | std::cmp::Ordering::Greater => unsafe {
                Self::load(slice.as_ptr(), SIZE)
            },
        }
    }

    fn splat(value: f32) -> Self {
        #[cfg(target_arch = "x86_64")]
        let splat = Self {
            elements: unsafe { _mm256_set1_ps(value) },
            size: SIZE,
        };

        #[cfg(not(target_arch = "x86_64"))]
        let splat = Self {
            size: SIZE,
            low: F32x4::splat(value),
            high: F32x4::splat(value),
        };

        splat
    }

    #[inline(always)]
    unsafe fn load(ptr: *const f32, size: usize) -> Self {
        let msg = format!("Size must be == {}", SIZE);
        assert!(size == SIZE, "{}", msg);

        #[cfg(target_arch = "x86_64")]
        let loaded = Self {
            elements: unsafe { _mm256_loadu_ps(ptr) },
            size,
        };

        #[cfg(not(target_arch = "x86_64"))]
        let loaded = Self {
            size: SIZE,
            low: F32x4::load(ptr, f32x4::SIZE),
            high: F32x4::load(unsafe { ptr.add(4) }, f32x4::SIZE),
        };

        loaded
    }

    #[inline(always)]
    unsafe fn load_partial(ptr: *const f32, size: usize) -> Self {
        let msg = format!("Size must be < {}", SIZE);
        assert!(size < SIZE, "{}", msg);
        #[cfg(target_arch = "x86_64")]
        let elements = match size {
            1 => unsafe { _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, *ptr.add(0)) },
            2 => unsafe { _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, *ptr.add(1), *ptr.add(0)) },
            3 => unsafe {
                _mm256_set_ps(
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    *ptr.add(2),
                    *ptr.add(1),
                    *ptr.add(0),
                )
            },
            4 => unsafe {
                _mm256_set_ps(
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    *ptr.add(3),
                    *ptr.add(2),
                    *ptr.add(1),
                    *ptr.add(0),
                )
            },

            5 => unsafe {
                _mm256_set_ps(
                    0.0,
                    0.0,
                    0.0,
                    *ptr.add(4),
                    *ptr.add(3),
                    *ptr.add(2),
                    *ptr.add(1),
                    *ptr.add(0),
                )
            },

            6 => unsafe {
                _mm256_set_ps(
                    0.0,
                    0.0,
                    *ptr.add(5),
                    *ptr.add(4),
                    *ptr.add(3),
                    *ptr.add(2),
                    *ptr.add(1),
                    *ptr.add(0),
                )
            },

            7 => unsafe {
                _mm256_set_ps(
                    0.0,
                    *ptr.add(6),
                    *ptr.add(5),
                    *ptr.add(4),
                    *ptr.add(3),
                    *ptr.add(2),
                    *ptr.add(1),
                    *ptr.add(0),
                )
            },

            _ => {
                let msg = "WTF is happening here";
                panic!("{}", msg);
            }
        };

        #[cfg(not(target_arch = "x86_64"))]
        let (low, high): (F32x4, F32x4) = match size {
            1 => (F32x4::load_partial(ptr, 1), F32x4::splat(0.0)),
            2 => (F32x4::load_partial(ptr, 2), F32x4::splat(0.0)),
            3 => (F32x4::load_partial(ptr, 3), F32x4::splat(0.0)),
            4 => (F32x4::load(ptr, 4), F32x4::splat(0.0)),
            5 => (F32x4::load(ptr, 4), F32x4::load_partial(ptr.add(4), 1)),
            6 => (F32x4::load(ptr, 4), F32x4::load_partial(ptr.add(4), 2)),
            7 => (F32x4::load(ptr, 4), F32x4::load_partial(ptr.add(4), 3)),
            _ => {
                let msg = "WTF is happening here";
                panic!("{}", msg);
            }
        };

        #[cfg(target_arch = "x86_64")]
        let loaded = Self { elements, size };

        #[cfg(not(target_arch = "x86_64"))]
        let loaded = Self { low, high, size };

        loaded
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

        #[cfg(target_arch = "x86_64")]
        unsafe {
            let mut vec = vec![0f32; SIZE];
            _mm256_storeu_ps(vec.as_mut_ptr(), self.elements);

            vec
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            let low_vec = self.low.store();
            let high_vec = self.high.store();

            let mut vec = vec![];
            vec.extend(low_vec);
            vec.extend(high_vec);

            vec
        }
    }

    fn store_partial(&self) -> Vec<f32> {
        match self.size {
            1..=7 => self.store().into_iter().take(self.size).collect(),
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

        #[cfg(target_arch = "x86_64")]
        unsafe {
            // Add a+b
            let elements = _mm256_add_ps(self.elements, rhs.elements);

            Self {
                elements,
                size: self.size,
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            Self {
                low: self.low + rhs.low,
                high: self.high + rhs.high,
                size: self.size,
            }
        }
    }

    fn simd_sin(&self) {
        todo!()
    }

    unsafe fn store_at(&self, ptr: *mut f32) {
        let msg = format!("Size must be <= {}", SIZE);

        assert!(self.size <= SIZE, "{}", msg);

        #[cfg(target_arch = "x86_64")]
        unsafe {
            _mm256_storeu_ps(ptr, self.elements);
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            vst1q_f32(ptr, self.low.elements);
            vst1q_f32(ptr.add(f32x4::SIZE), self.high.elements);
        }
    }

    unsafe fn store_at_partial(&self, ptr: *mut f32) {
        #[cfg(target_arch = "x86_64")]
        match self.size {
            4..=7 => {
                // Store lower 4 floats with a single instruction
                _mm_storeu_ps(ptr, _mm256_castps256_ps128(self.elements));

                // Extract upper lane once
                let upper = _mm256_extractf128_ps(self.elements, 1);

                // Use a more efficient approach for remaining elements
                match self.size {
                    7 => {
                        // Store the next 3 elements with one instruction
                        _mm_storel_pi((ptr.add(4)) as *mut __m64, upper);
                        let seventh = _mm_extract_ps(upper, 2) as u32;
                        *(ptr.add(6)) = core::mem::transmute(seventh);
                    }
                    6 => {
                        // Store the next 2 elements with one instruction
                        _mm_storel_pi((ptr.add(4)) as *mut __m64, upper);
                    }
                    5 => {
                        // Store just one more element
                        _mm_store_ss(ptr.add(4), upper);
                    }
                    _ => {} // Size 4 already fully handled
                }
            }
            3 => {
                let lower = _mm256_castps256_ps128(self.elements);
                _mm_storel_pi(ptr as *mut __m64, lower);
                let third = _mm_extract_ps(lower, 2) as u32;
                *(ptr.add(2)) = core::mem::transmute(third);
            }
            2 => {
                let lower = _mm256_castps256_ps128(self.elements);
                _mm_storel_pi(ptr as *mut __m64, lower);
            }
            1 => {
                let lower = _mm256_castps256_ps128(self.elements);
                _mm_store_ss(ptr, lower);
            }
            _ => {
                let msg = "WTF is happening here";
                panic!("{}", msg);
            }
        }

        #[cfg(target_arch = "aarch64")]
        match self.size {
            5..=7 => {
                self.low.store_at(ptr);
                self.high.store_at_partial(ptr.add(4));
            }
            4 => self.low.store_at(ptr),
            1..=3 => self.low.store_at_partial(ptr),
            _ => {
                let msg = "WTF is happening here";
                panic!("{}", msg);
            }
        }
    }
}

/// Implementation of Add trait for F32x8 using custom SIMD types
impl Add for F32x8 {
    type Output = F32x8;

    fn add(self, rhs: F32x8) -> Self::Output {
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
