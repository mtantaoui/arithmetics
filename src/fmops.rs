pub trait FusedMultiplyOps {
    type Output;

    /// Computes (a * b) + c in a single operation where possible
    fn fused_multiply_add(self, a: Self, b: Self) -> Self;

    /// Computes (a * b) - c in a single operation where possible
    fn fused_multiply_sub(self, a: Self, b: Self) -> Self;
}

impl FusedMultiplyOps for Vec<f32> {
    type Output = Vec<f32>;

    fn fused_multiply_add(self, a: Self, b: Self) -> Self {
        todo!()
    }

    fn fused_multiply_sub(self, a: Self, b: Self) -> Self {
        todo!()
    }
}
