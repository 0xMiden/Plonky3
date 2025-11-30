use alloc::vec::Vec;

use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field};

/// Parameters for the lifted FRI instance.
///
/// Notes:
/// - `mmcs` is used for commit-phase codewords (2-wide vectors at shrinking heights).
/// - `padding_width` should match the LMCS hasher padding width (rows are padded to multiples of this).
///   We surface it here so the DEEP combination can compute β_j = β^{2 * U_j} where U_j counts padded
///   lanes across prior matrices.
#[derive(Clone, Debug)]
pub struct LiftedFriParams<M> {
    pub log_blowup: usize,
    pub log_final_poly_len: usize,
    pub num_queries: usize,
    pub proof_of_work_bits: usize,
    pub padding_width: usize,
    pub mmcs: M,
}

impl<M> LiftedFriParams<M> {
    pub const fn blowup(&self) -> usize {
        1 << self.log_blowup
    }

    pub const fn final_poly_len(&self) -> usize {
        1 << self.log_final_poly_len
    }

    /// Conjectured soundness (ethSTARK style): log_blowup * num_queries + pow_bits.
    pub const fn conjectured_soundness_bits(&self) -> usize {
        self.log_blowup * self.num_queries + self.proof_of_work_bits
    }
}
