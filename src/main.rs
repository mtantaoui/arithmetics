#![feature(simd_ffi)]
#![feature(avx512_target_feature)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m128, __m256, __m512, _mm_add_ps, _mm_loadu_ps, _mm_set1_ps, _mm_storeu_ps, _mm256_add_ps,
    _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps,
};
use std::fmt;
use std::ops::{Add, AddAssign};

use libc::{c_float, c_int};

// Foreign function interface declaration
#[link(name = "add", kind = "static")]
unsafe extern "C" {
    fn addition(a: c_int, b: c_int) -> c_int;

    fn f32x16_set(value: c_float) -> __m512;
    fn f32x16_load(mem_addr: *mut c_float) -> __m512;
    fn f32x16_add(a: __m512, b: __m512) -> __m512;
    fn f32x16_store(mem_addr: *mut c_float, a: __m512);
}

pub trait SimdVec {
    /// Create a new SIMD vector with all lanes set to the same value
    fn splat(value: f32) -> Self;

    /// Create a new SIMD vector from an array of 4 f32 values
    fn from_array(array: [f32; 4]) -> Self;

    /// Create a new SIMD vector with custom lane values
    fn new(a: f32, b: f32, c: f32, d: f32) -> Self;

    /// Load 4 f32 values from memory (unaligned OK)
    fn load(ptr: *const f32) -> Self;

    /// Store 4 f32 values to memory (unaligned OK)
    fn store(&self, ptr: *mut f32);

    /// Convert to an array of 4 f32 values
    fn to_array(&self) -> [f32; 4];

    /// Extract a single lane value from the vector
    fn extract(&self, index: usize) -> f32;

    /// Load up to 4 f32 values with a mask (for partial/remainder loads)
    fn load_partial(ptr: *const f32, count: usize) -> Self;

    /// Store up to 4 f32 values with a mask (for partial/remainder stores)
    fn store_partial(&self, ptr: *mut f32, count: usize);
}

/// A SIMD vector of 4 32-bit floating point values
/// This provides a cross-platform abstraction over architecture-specific SIMD types
#[derive(Copy, Clone)]
pub struct F32x4 {
    #[cfg(target_arch = "x86_64")]
    inner: __m128,

    #[cfg(target_arch = "aarch64")]
    inner: std::arch::aarch64::float32x4_t,

    #[cfg(target_arch = "arm")]
    inner: std::arch::arm::float32x4_t,

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "arm")))]
    inner: [f32; 4],
}

impl SimdVec for F32x4 {
    /// Create a new f32x4 with all lanes set to the same value
    #[inline]
    fn splat(value: f32) -> Self {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("sse") {
                    return Self {
                        inner: _mm_set1_ps(value),
                    };
                } else {
                    panic!(
                        "SSE not supported on this system. This operation requires SSE (Streaming SIMDs Extensions) \
                        for optimized 128-bit SIMD instructions. Please run this on a machine with SSE support."
                    );
                }
            }

            #[cfg(target_arch = "aarch64")]
            {
                if std::arch::is_aarch64_feature_detected!("neon") {
                    return Self {
                        inner: std::arch::aarch64::vdupq_n_f32(value),
                    };
                } else {
                    return Self {
                        inner: std::arch::aarch64::float32x4_t {
                            0: [value, value, value, value],
                        },
                    };
                }
            }

            #[cfg(target_arch = "arm")]
            {
                if std::arch::is_arm_feature_detected!("neon") {
                    return Self {
                        inner: std::arch::arm::vdupq_n_f32(value),
                    };
                } else {
                    return Self {
                        inner: [value, value, value, value],
                    };
                }
            }

            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "arm")))]
            {
                return Self {
                    inner: [value, value, value, value],
                };
            }
        }
    }

    /// Create a new f32x4 from an array of 4 f32 values
    #[inline]
    fn from_array(array: [f32; 4]) -> Self {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("sse") {
                    return Self {
                        inner: _mm_loadu_ps(array.as_ptr()),
                    };
                } else {
                    panic!(
                        "SSE not supported on this system. This operation requires SSE (Streaming SIMDs Extensions) \
                        for optimized 128-bit SIMD instructions. Please run this on a machine with SSE support."
                    );
                }
            }

            #[cfg(target_arch = "aarch64")]
            {
                todo!();
                return Self {
                    inner: std::arch::aarch64::vld1q_f32(array.as_ptr()),
                };
            }

            #[cfg(target_arch = "arm")]
            {
                todo!();
                return Self {
                    inner: std::arch::arm::vld1q_f32(array.as_ptr()),
                };
            }

            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "arm")))]
            {
                todo!();
                return Self { inner: array };
            }
        }
    }

    /// Create a new f32x4 with custom values
    #[inline]
    fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
        Self::from_array([a, b, c, d])
    }

    /// Load 4 f32 values from memory
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    #[inline]
    fn load(ptr: *const f32) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse") {
                return Self {
                    inner: unsafe { _mm_loadu_ps(ptr) },
                };
            } else {
                panic!(
                    "SSE not supported on this system. This operation requires SSE (Streaming SIMDs Extensions) \
                    for optimized 128-bit SIMD instructions. Please run this on a machine with SSE support."
                );
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return Self {
                    inner: std::arch::aarch64::vld1q_f32(ptr),
                };
            } else {
                let array = [*ptr, *ptr.add(1), *ptr.add(2), *ptr.add(3)];
                return Self::from_array(array);
            }
        }

        #[cfg(target_arch = "arm")]
        {
            if std::arch::is_arm_feature_detected!("neon") {
                return Self {
                    inner: std::arch::arm::vld1q_f32(ptr),
                };
            } else {
                let array = [*ptr, *ptr.add(1), *ptr.add(2), *ptr.add(3)];
                return Self::from_array(array);
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "arm")))]
        {
            let array = [*ptr, *ptr.add(1), *ptr.add(2), *ptr.add(3)];
            return Self::from_array(array);
        }
    }

    /// Store 4 f32 values to memory
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    #[inline]
    fn store(&self, ptr: *mut f32) {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("sse") {
                unsafe { _mm_storeu_ps(ptr, self.inner) };
            } else {
                panic!(
                    "SSE not supported on this system. This operation requires SSE (Streaming SIMDs Extensions) \
                    for optimized 128-bit SIMD instructions. Please run this on a machine with SSE support."
                );
            }
            return;
        }

        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                std::arch::aarch64::vst1q_f32(ptr, self.inner);
            } else {
                let array = self.to_array();
                *ptr = array[0];
                *ptr.add(1) = array[1];
                *ptr.add(2) = array[2];
                *ptr.add(3) = array[3];
            }
            return;
        }

        #[cfg(target_arch = "arm")]
        {
            if std::arch::is_arm_feature_detected!("neon") {
                std::arch::arm::vst1q_f32(ptr, self.inner);
            } else {
                let array = self.to_array();
                *ptr = array[0];
                *ptr.add(1) = array[1];
                *ptr.add(2) = array[2];
                *ptr.add(3) = array[3];
            }
            return;
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "arm")))]
        {
            *ptr = self.inner[0];
            *ptr.add(1) = self.inner[1];
            *ptr.add(2) = self.inner[2];
            *ptr.add(3) = self.inner[3];
        }
    }

    /// Convert to an array of 4 f32 values
    #[inline]
    fn to_array(&self) -> [f32; 4] {
        let mut result = [0.0f32; 4];

        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("sse") {
                unsafe { _mm_storeu_ps(result.as_mut_ptr(), self.inner) };
            } else {
                panic!(
                    "SSE not supported on this system. This operation requires SSE (Streaming SIMDs Extensions) \
                        for optimized 128-bit SIMD instructions. Please run this on a machine with SSE support."
                );
            }
            return result;
        }

        #[cfg(target_arch = "aarch64")]
        {
            std::arch::aarch64::vst1q_f32(result.as_mut_ptr(), self.inner);
            return result;
        }

        #[cfg(target_arch = "arm")]
        {
            std::arch::arm::vst1q_f32(result.as_mut_ptr(), self.inner);
            return result;
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "arm")))]
        {
            return self.inner;
        }
    }

    /// Extract a single lane from the vector
    #[inline]
    fn extract(&self, index: usize) -> f32 {
        assert!(index < 4, "Index out of bounds");
        self.to_array()[index]
    }

    /// Load 4 f32 values with a mask (for handling remainders)
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    #[inline]
    fn load_partial(ptr: *const f32, count: usize) -> Self {
        assert!(count <= 4, "Count must be <= 4");

        let mut array = [0.0f32; 4];
        for i in 0..count {
            array[i] = unsafe { *ptr.add(i) };
        }

        Self::from_array(array)
    }

    /// Store 4 f32 values with a mask (for handling remainders)
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    #[inline]
    fn store_partial(&self, ptr: *mut f32, count: usize) {
        assert!(count <= 4, "Count must be <= 4");

        let array = self.to_array();
        for i in 0..count {
            unsafe { *ptr.add(i) = array[i] };
        }
    }
}

// Implementation of Add for f32x4
impl Add for F32x4 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                if std::arch::is_x86_feature_detected!("sse") {
                    return Self {
                        inner: _mm_add_ps(self.inner, rhs.inner),
                    };
                } else {
                    let a = self.to_array();
                    let b = rhs.to_array();
                    return Self::new(a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]);
                }
            }

            #[cfg(target_arch = "aarch64")]
            {
                if std::arch::is_aarch64_feature_detected!("neon") {
                    return Self {
                        inner: std::arch::aarch64::float32x4_add_f32(self.inner, rhs.inner),
                    };
                } else {
                    let a = self.to_array();
                    let b = rhs.to_array();
                    return Self::new(a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]);
                }
            }

            #[cfg(target_arch = "arm")]
            {
                if std::arch::is_arm_feature_detected!("neon") {
                    return Self {
                        inner: std::arch::arm::vaddq_f32(self.inner, rhs.inner),
                    };
                } else {
                    let a = self.to_array();
                    let b = rhs.to_array();
                    return Self::new(a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]);
                }
            }

            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "arm")))]
            {
                return Self {
                    inner: [
                        self.inner[0] + rhs.inner[0],
                        self.inner[1] + rhs.inner[1],
                        self.inner[2] + rhs.inner[2],
                        self.inner[3] + rhs.inner[3],
                    ],
                };
            }
        }
    }
}

// Implementation of AddAssign for f32x4
impl AddAssign for F32x4 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

// Debug implementation for f32x4
impl fmt::Debug for F32x4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let array = self.to_array();
        write!(
            f,
            "f32x4({:?}, {:?}, {:?}, {:?})",
            array[0], array[1], array[2], array[3]
        )
    }
}

// Define higher-level SIMD vectors based on f32x4
#[derive(Copy, Clone, Debug)]
pub struct F32x8 {
    #[cfg(target_arch = "x86_64")]
    inner: __m256,

    #[cfg(not(target_arch = "x86_64"))]
    low: F32x4,
    #[cfg(not(target_arch = "x86_64"))]
    high: F32x4,
}

impl F32x8 {
    /// Create a new f32x8 with all lanes set to the same value
    #[inline]
    fn splat(value: f32) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if is_x86_feature_detected!("avx") {
                return Self {
                    inner: _mm256_set1_ps(value),
                };
            } else {
                panic!(
                    "AVX not supported on this system. This operation requires AVX (Advanced Vector Extensions) \
                        for optimized 256-bit SIMD instructions. Please run this on a machine with AVX support."
                );
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self {
                low: F32x4::splat(value),
                high: F32x4::splat(value),
            }
        }
    }

    /// Load 8 f32 values from memory
    #[inline]
    fn load(ptr: *const f32) -> Self {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if is_x86_feature_detected!("avx") {
                return Self {
                    inner: _mm256_loadu_ps(ptr),
                };
            } else {
                panic!(
                    "AVX not supported on this system. This operation requires AVX (Advanced Vector Extensions) \
                    for optimized 256-bit SIMD instructions. Please run this on a machine with AVX support."
                );
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            Self {
                low: F32x4::load(ptr),
                high: F32x4::load(unsafe { ptr.add(4) }),
            }
        }
    }

    /// Load 8 f32 values with a mask (for handling remainders)
    #[inline]
    fn load_partial(ptr: *const f32, count: usize) -> Self {
        assert!(count <= 8, "Count must be <= 8");

        let mut array = [0.0f32; 8];

        for i in 0..count {
            array[i] = unsafe { *ptr.add(i) };
        }

        Self::from_array(array)
    }

    /// Create a new f32x8 from an array of 4 f32 values
    #[inline]
    fn from_array(array: [f32; 8]) -> Self {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx") {
                    return Self {
                        inner: _mm256_loadu_ps(array.as_ptr()),
                    };
                } else {
                    panic!(
                        "AVX not supported on this system. This operation requires AVX (Advanced Vector Extensions) \
                        for optimized 256-bit SIMD instructions. Please run this on a machine with AVX support."
                    );
                }
            }

            #[cfg(target_arch = "aarch64")]
            {
                todo!()
            }

            #[cfg(target_arch = "arm")]
            {
                todo!()
            }

            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "arm")))]
            {
                todo!()
            }
        }
    }

    /// Convert to an array of 8 f32 values
    #[inline]
    fn to_array(&self) -> [f32; 8] {
        unsafe {
            let mut result = [0.0f32; 8];

            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx") {
                    _mm256_storeu_ps(result.as_mut_ptr(), self.inner);
                    return result;
                } else {
                    panic!(
                        "AVX not supported on this system. This operation requires AVX (Advanced Vector Extensions) \
                        for optimized 256-bit SIMD instructions. Please run this on a machine with AVX support."
                    );
                }
            }

            #[cfg(target_arch = "aarch64")]
            {
                todo!()
            }

            #[cfg(target_arch = "arm")]
            {
                todo!()
            }

            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "arm")))]
            {
                todo!()
            }
        }
    }

    /// Store 4 f32 values to memory
    #[inline]
    fn store(&self, ptr: *mut f32) {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("sse") {
                unsafe { _mm256_storeu_ps(ptr, self.inner) };
            } else {
                let mut array = [0.0f32; 4];
                unsafe { _mm256_storeu_ps(array.as_mut_ptr(), self.inner) };
                unsafe {
                    *ptr = array[0];
                    *ptr.add(1) = array[1];
                    *ptr.add(2) = array[2];
                    *ptr.add(3) = array[3];
                };
            }
            return;
        }

        #[cfg(target_arch = "aarch64")]
        {
            todo!()
        }

        #[cfg(target_arch = "arm")]
        {
            todo!()
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "arm")))]
        {
            todo!()
        }
    }

    /// Store 8 f32 values with a mask (for handling remainders)
    #[inline]
    fn store_partial(&self, ptr: *mut f32, count: usize) {
        assert!(count <= 8, "Count must be <= 8");

        let array = self.to_array();
        for i in 0..count {
            unsafe { *ptr.add(i) = array[i] };
        }
    }

    /// Create a new f32x8 with custom values
    #[inline]
    fn new(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, g: f32, h: f32) -> Self {
        Self::from_array([a, b, c, d, e, f, g, h])
    }
}

// Implementation of Add for f32x8
impl Add for F32x8 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if std::arch::is_x86_feature_detected!("sse") {
                return Self {
                    inner: _mm256_add_ps(self.inner, rhs.inner),
                };
            } else {
                let a = self.to_array();
                let b = rhs.to_array();
                return Self::new(
                    a[0] + b[0],
                    a[1] + b[1],
                    a[2] + b[2],
                    a[3] + b[3],
                    a[4] + b[4],
                    a[5] + b[5],
                    a[6] + b[6],
                    a[7] + b[7],
                );
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            todo!()
        }

        #[cfg(target_arch = "arm")]
        {
            todo!()
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "arm")))]
        {
            todo!()
        }
    }
}

// Implementation of AddAssign for f32x8
impl AddAssign for F32x8 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

// Define f32x16 using two f32x8
#[derive(Copy, Clone, Debug)]
pub struct F32x16 {
    low: F32x8,
    high: F32x8,
}

impl F32x16 {
    #[inline]
    pub fn splat(value: f32) -> Self {
        {
            #[cfg(target_arch = "x86_64")]
            {
                Self {
                    low: F32x8::splat(value),
                    high: F32x8::splat(value),
                }
            }

            #[cfg(target_arch = "aarch64")]
            {
                if std::arch::is_aarch64_feature_detected!("neon") {
                    return Self {
                        inner: std::arch::aarch64::vdupq_n_f32(value),
                    };
                } else {
                    return Self {
                        inner: std::arch::aarch64::float32x4_t {
                            0: [value, value, value, value],
                        },
                    };
                }
            }

            #[cfg(target_arch = "arm")]
            {
                if std::arch::is_arm_feature_detected!("neon") {
                    return Self {
                        inner: std::arch::arm::vdupq_n_f32(value),
                    };
                } else {
                    return Self {
                        inner: [value, value, value, value],
                    };
                }
            }

            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "arm")))]
            {
                return Self {
                    inner: [value, value, value, value],
                };
            }
        }
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    #[inline]
    pub fn load(ptr: *const f32) -> Self {
        Self {
            low: F32x8::load(ptr),
            high: F32x8::load(unsafe { ptr.add(8) }),
        }
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    #[inline]
    pub fn store(&self, ptr: *mut f32) {
        self.low.store(ptr);
        self.high.store(unsafe { ptr.add(8) });
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    #[inline]
    pub fn load_partial(ptr: *const f32, count: usize) -> Self {
        assert!(count <= 16, "Count must be <= 16");

        if count <= 8 {
            Self {
                low: F32x8::load_partial(ptr, count),
                high: F32x8::splat(0.0),
            }
        } else {
            Self {
                low: F32x8::load(ptr),
                high: F32x8::load_partial(unsafe { ptr.add(8) }, count - 8),
            }
        }
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    #[inline]
    pub fn store_partial(&self, ptr: *mut f32, count: usize) {
        assert!(count <= 16, "Count must be <= 16");

        if count <= 8 {
            self.low.store_partial(ptr, count);
        } else {
            self.low.store(ptr);
            self.high.store_partial(unsafe { ptr.add(8) }, count - 8);
        }
    }

    #[inline]
    pub fn new(low: F32x8, high: F32x8) -> Self {
        Self { low, high }
    }
}

// Implementation of Add for f32x16
impl Add for F32x16 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            low: self.low + rhs.low,
            high: self.high + rhs.high,
        }
    }
}

// Implementation of AddAssign for f32x16
impl AddAssign for F32x16 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

pub trait SimdAdd<Rhs = Self> {
    type Output;

    fn add(self, rhs: Rhs) -> Self::Output;
}

/// Implementation of Add trait for Vec<f32> using our custom SIMD types
impl SimdAdd for Vec<f32> {
    type Output = Vec<f32>;

    fn add(self, rhs: Vec<f32>) -> Self::Output {
        // Use the smaller length to avoid out of bounds access
        let min_len = self.len().min(rhs.len());
        let mut result = self.clone();

        // Ensure result has at least the size of the longer vector
        if rhs.len() > result.len() {
            result.resize(rhs.len(), 0.0);
        }

        unsafe {
            // Try to use AVX-512 equivalent (f32x16)
            if cfg!(target_arch = "x86_64") && is_x86_feature_detected!("avx512f") {
                // Process 16 elements at a time
                let chunk_size = 16;
                let full_chunks = min_len / chunk_size;
                let remainder = min_len % chunk_size;

                for i in 0..full_chunks {
                    let offset = i * chunk_size;
                    let a = F32x16::load(self.as_ptr().add(offset));
                    let b = F32x16::load(rhs.as_ptr().add(offset));
                    let sum = a + b;
                    sum.store(result.as_mut_ptr().add(offset));
                }

                if remainder > 0 {
                    let offset = full_chunks * chunk_size;
                    let a = F32x16::load_partial(self.as_ptr().add(offset), remainder);
                    let b = F32x16::load_partial(rhs.as_ptr().add(offset), remainder);
                    let sum = a + b;
                    sum.store_partial(result.as_mut_ptr().add(offset), remainder);
                }
            }
            // Try to use AVX equivalent (f32x8)
            else if cfg!(target_arch = "x86_64") && is_x86_feature_detected!("avx") {
                // Process 8 elements at a time
                let chunk_size = 8;
                let full_chunks = min_len / chunk_size;
                let remainder = min_len % chunk_size;

                for i in 0..full_chunks {
                    let offset = i * chunk_size;
                    let a = F32x8::load(self.as_ptr().add(offset));
                    let b = F32x8::load(rhs.as_ptr().add(offset));
                    let sum = a + b;
                    sum.store(result.as_mut_ptr().add(offset));
                }

                if remainder > 0 {
                    let offset = full_chunks * chunk_size;
                    let a = F32x8::load_partial(self.as_ptr().add(offset), remainder);
                    let b = F32x8::load_partial(rhs.as_ptr().add(offset), remainder);
                    let sum = a + b;
                    sum.store_partial(result.as_mut_ptr().add(offset), remainder);
                }
            }
            // Use f32x4 for SSE or NEON
            else {
                // Process 4 elements at a time
                let chunk_size = 4;
                let full_chunks = min_len / chunk_size;
                let remainder = min_len % chunk_size;

                for i in 0..full_chunks {
                    let offset = i * chunk_size;
                    let a = F32x4::load(self.as_ptr().add(offset));
                    let b = F32x4::load(rhs.as_ptr().add(offset));
                    let sum = a + b;
                    sum.store(result.as_mut_ptr().add(offset));
                }

                if remainder > 0 {
                    let offset = full_chunks * chunk_size;
                    let a = F32x4::load_partial(self.as_ptr().add(offset), remainder);
                    let b = F32x4::load_partial(rhs.as_ptr().add(offset), remainder);
                    let sum = a + b;
                    sum.store_partial(result.as_mut_ptr().add(offset), remainder);
                }
            }
        }

        result
    }
}

fn align_to_64_bytes(data: &mut [f32]) -> *mut f32 {
    let addr = data.as_mut_ptr() as usize;
    let misalignment = addr % 64;
    if misalignment == 0 {
        data.as_mut_ptr()
    } else {
        // Ensure there's enough space after alignment
        assert!(data.len() >= 16 + (64 - misalignment) / 4);
        let aligned_addr = addr + (64 - misalignment);
        aligned_addr as *mut f32
    }
}

#[repr(align(64))]
struct AlignedF32x16 {
    data: [f32; 16],
}

/// Utility function to demonstrate the usage
fn main() {
    // // Call the C function from Rust
    // let result = unsafe { addition(5, 7) };
    // println!("Result from C: {}", result);

    let mut simd_vec1 = AlignedF32x16 { data: [12.0; 16] };
    let mut simd_vec2 = AlignedF32x16 { data: [5.0; 16] };

    let mut simd_res = AlignedF32x16 { data: [0.0; 16] };

    let v1 = unsafe { f32x16_load(simd_vec1.data.as_mut_ptr()) };
    let v2 = unsafe { f32x16_load(simd_vec2.data.as_mut_ptr()) };

    let v3 = unsafe { f32x16_add(v1, v2) };
    unsafe { f32x16_store(simd_res.data.as_mut_ptr(), v3) };

    println!("simd_res = {:?}", simd_res.data);

    // Example with a standard aligned vector
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];

    let c = a.add(b);
    println!("Standard result: {:?}", c);

    // Example with edge case (5 elements)
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![10.0, 20.0, 30.0, 40.0, 50.0];

    let z = x.add(y);
    println!("Edge case (5 elements) result: {:?}", z);

    if is_x86_feature_detected!("avx512f") {
        println!("Yes")
    } else {
        println!("No")
    }

    // Demonstration of f32x4
    let v1 = F32x4::new(1.0, 2.0, 3.0, 4.0);
    let v2 = F32x4::new(10.0, 20.0, 30.0, 40.0);
    let sum = v1 + v2;
    println!("f32x4 addition result: {:?}", sum);
}

/// Example usage
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32x4_basic() {
        let a = F32x4::new(1.0, 2.0, 3.0, 4.0);
        let b = F32x4::new(5.0, 6.0, 7.0, 8.0);
        let c = a + b;

        let result = c.to_array();
        assert_eq!(result, [6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_vector_addition() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];

        let expected = vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0];
        let result = a.add(b);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_edge_case_vector_lengths() {
        // Test with length 5 (not a multiple of vector widths)
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0, 20.0, 30.0, 40.0, 50.0];

        let expected = vec![11.0, 22.0, 33.0, 44.0, 55.0];
        let result = a.add(b);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_edge_case_vector_lengths_2() {
        // Test with length 5 (not a multiple of vector widths)
        let a = vec![1.0; 17];
        let b = vec![2.0; 17];

        let expected = vec![3.0; 17];
        let result = a.add(b);

        assert_eq!(result, expected);
    }
}
