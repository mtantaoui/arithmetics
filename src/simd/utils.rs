pub trait SimdVec<T> {
    fn new(slice: &[T]) -> Self;

    fn splat(value: T) -> Self;

    /// .
    ///
    /// # Safety
    ///
    /// .
    unsafe fn load(ptr: *const T, size: usize) -> Self;

    /// .
    ///
    /// # Safety
    ///
    /// .
    unsafe fn load_partial(ptr: *const T, size: usize) -> Self;

    fn to_vec(self) -> Vec<T>;

    fn store(&self) -> Vec<T>;

    fn store_partial(&self) -> Vec<T>;

    fn simd_mask_add(&self, rhs: Self) -> Self;

    fn simd_add(&self, rhs: Self) -> Self;
}
