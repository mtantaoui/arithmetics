use std::{arch::x86_64::*, ops::Add};

use crate::fmops::FusedMultiplyOps;

pub const SIZE: usize = 16;

// Define f32x16 using two f32x8
#[derive(Copy, Clone, Debug)]
pub struct F32x16 {
    size: usize,

    elements: __m512,
}

impl F32x16 {
    #[inline(always)]
    pub fn new(slice: Vec<f32>) -> Self {
        if slice.len() == SIZE {
            Self::load(slice.as_ptr(), slice.len())
        } else if slice.len() < SIZE {
            Self::load_partial(slice.as_ptr(), slice.len())
        } else {
            let msg = format!("F32x16 size must not exceed {}", SIZE);
            panic!("{}", msg);
        }
    }

    #[inline(always)]
    pub fn splat(value: f32) -> Self {
        Self {
            elements: unsafe { _mm512_set1_ps(value) },
            size: SIZE,
        }
    }

    #[inline(always)]
    fn load(ptr: *const f32, size: usize) -> Self {
        let msg = format!("Size must be == {}", SIZE);
        assert!(size == SIZE, "{}", msg);

        Self {
            elements: unsafe { _mm512_loadu_ps(ptr) },
            size: SIZE,
        }
    }

    #[inline(always)]
    fn load_partial(ptr: *const f32, size: usize) -> Self {
        let msg = format!("Size must be < {}", SIZE);
        assert!(size < SIZE, "{}", msg);

        let mask: __mmask16 = (1 << size) - 1;

        Self {
            elements: unsafe { _mm512_maskz_loadu_ps(mask, ptr) },
            size,
        }
    }

    #[inline(always)]
    pub fn to_vec(&self) -> Vec<f32> {
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
        let msg = format!("Size must be == {}", SIZE);

        assert!(self.size == SIZE, "{}", msg);

        let mut vec = vec![0f32; SIZE];

        unsafe {
            _mm512_storeu_ps(vec.as_mut_ptr(), self.elements);
        }

        vec
    }

    #[inline(always)]
    fn store_partial(&self) -> Vec<f32> {
        let msg = format!("Size must be < {}", SIZE);

        assert!(self.size < SIZE, "{}", msg);

        let mask: __mmask16 = (1 << self.size) - 1;

        let mut vec = vec![0f32; self.size];

        unsafe {
            _mm512_mask_storeu_ps(vec.as_mut_ptr(), mask, self.elements);
        }

        vec
    }

    fn _mask_add(&self, rhs: Self) -> Self {
        let msg = format!("Operands must have the same size {}", self.size);
        assert!(self.size == rhs.size, "{}", msg);

        unsafe {
            let mask: __mmask16 = (1 << self.size) - 1;

            // Add a+b
            let elements = _mm512_maskz_add_ps(mask, self.elements, rhs.elements);

            Self {
                elements,
                size: self.size,
            }
        }
    }

    fn _add(&self, rhs: Self) -> Self {
        let msg = format!("Operands must have the same size {}", self.size);
        assert!(self.size == rhs.size, "{}", msg);

        unsafe {
            // Add a+b
            let elements = _mm512_add_ps(self.elements, rhs.elements);

            Self {
                elements,
                size: self.size,
            }
        }
    }

    fn _mask_fmadd(&self, a: Self, b: Self) -> Self {
        let msg = format!("Operands must have the same size {}", self.size);
        assert!(self.size == a.size && self.size == b.size, "{}", msg);

        unsafe {
            let mask: __mmask16 = (1 << self.size) - 1;

            // Add (a+b) + self.elements
            let elements = _mm512_maskz_fmadd_ps(mask, a.elements, b.elements, self.elements);

            Self {
                elements,
                size: self.size,
            }
        }
    }

    fn _fmadd(&self, a: Self, b: Self) -> Self {
        let msg = format!("Operands must have the same size {}", self.size);
        assert!(self.size == a.size && self.size == b.size, "{}", msg);

        unsafe {
            // Add (a+b) + self.elements
            let elements = _mm512_fmadd_ps(a.elements, b.elements, self.elements);

            Self {
                elements,
                size: self.size,
            }
        }
    }

    fn _mask_fmsub(&self, a: Self, b: Self) -> Self {
        let msg = format!("Operands must have the same size {}", self.size);
        assert!(self.size == a.size && self.size == b.size, "{}", msg);

        unsafe {
            let mask: __mmask16 = (1 << self.size) - 1;

            // Add (a+b) - self.elements
            let elements = _mm512_maskz_fmsub_ps(mask, a.elements, b.elements, self.elements);

            Self {
                elements,
                size: self.size,
            }
        }
    }

    fn _fmsub(&self, a: Self, b: Self) -> Self {
        let msg = format!("Operands must have the same size {}", self.size);
        assert!(self.size == a.size && self.size == b.size, "{}", msg);

        unsafe {
            // Add (a+b) + self.elements
            let elements = _mm512_fmsub_ps(a.elements, b.elements, self.elements);

            Self {
                elements,
                size: self.size,
            }
        }
    }
}

/// Implementation of Add trait for Vec<f32> using custom SIMD types
impl Add for F32x16 {
    type Output = F32x16;

    fn add(self, rhs: F32x16) -> Self::Output {
        let msg = format!("Operands must have the same size {}", SIZE);

        assert!(self.size == rhs.size, "{}", msg);

        if self.size == SIZE {
            self._add(rhs)
        } else if self.size < SIZE {
            self._mask_add(rhs)
        } else {
            let msg = format!("F32x16 size must not exceed {}", SIZE);
            panic!("{}", msg);
        }
    }
}

impl FusedMultiplyOps for F32x16 {
    type Output = F32x16;

    fn fused_multiply_add(self, a: Self, b: Self) -> Self {
        let msg = format!("Operands must have the same size {}", self.size);
        assert!(self.size == a.size && self.size == b.size, "{}", msg);

        if self.size == SIZE {
            self._fmadd(a, b)
        } else if self.size < SIZE {
            self._mask_fmadd(a, b)
        } else {
            let msg = format!("F32x16 size must not exceed {}", SIZE);
            panic!("{}", msg);
        }
    }

    fn fused_multiply_sub(self, a: Self, b: Self) -> Self {
        let msg = format!("Operands must have the same size {}", self.size);
        assert!(self.size == a.size && self.size == b.size, "{}", msg);

        if self.size == SIZE {
            self._fmsub(a, b)
        } else if self.size < SIZE {
            self._mask_fmsub(a, b)
        } else {
            let msg = format!("F32x16 size must not exceed {}", SIZE);
            panic!("{}", msg);
        }
    }
}
