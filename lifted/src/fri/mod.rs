use alloc::vec::Vec;

pub mod deep;
pub mod fold;
pub mod prover;
pub mod verifier;

/// Evaluations for a group of matrices: `[matrix][col]`.
#[derive(Clone, Debug)]
pub struct MatrixGroupEvals<T>(Vec<Vec<T>>);

impl<T> MatrixGroupEvals<T> {
    pub const fn new(evals: Vec<Vec<T>>) -> Self {
        Self(evals)
    }

    pub fn iter(&self) -> impl Iterator<Item = &[T]> {
        self.0.iter().map(|v| v.as_slice())
    }

    pub fn flatten(&self) -> impl Iterator<Item = &T> {
        self.0.iter().flatten()
    }
}
