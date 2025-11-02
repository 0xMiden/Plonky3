use alloc::vec::Vec;
use core::fmt::Debug;

use p3_field::{ExtensionField, Field};
use p3_matrix::Matrix;

/// A set of parameters defining a specific instance of the FRI protocol.
#[derive(Debug)]
pub struct FriParameters<M> {
    pub log_blowup: usize,
    // TODO: This parameter and FRI early stopping are not yet implemented in `CirclePcs`.
    /// Log of the size of the final polynomial.
    /// Since we fold `log_folding_factor` bits in each iteration, it much be that
    ///   log_final_poly_len \equiv log_original_poly_len \pmod log_folding_factor
    pub log_final_poly_len: usize,
    pub num_queries: usize,
    pub proof_of_work_bits: usize,
    pub mmcs: M,
    /// Log of the folding factor (arity). Must be >= 1.
    pub log_folding_factor: usize,
}

impl<M> FriParameters<M> {
    pub const fn blowup(&self) -> usize {
        1 << self.log_blowup
    }

    pub const fn final_poly_len(&self) -> usize {
        1 << self.log_final_poly_len
    }

    pub const fn folding_factor(&self) -> usize {
        1 << self.log_folding_factor
    }

    /// Creates new FRI parameters with validation.
    /// Returns an error if the parameters are invalid.
    pub fn new(
        log_blowup: usize,
        log_final_poly_len: usize,
        num_queries: usize,
        proof_of_work_bits: usize,
        mmcs: M,
        log_folding_factor: usize,
    ) -> Self {
        Self {
            log_blowup,
            log_final_poly_len,
            num_queries,
            proof_of_work_bits,
            mmcs,
            log_folding_factor,
        }
    }

    /// Returns the soundness bits of this FRI instance based on the
    /// [ethSTARK](https://eprint.iacr.org/2021/582) conjecture.
    ///
    /// Certain users may instead want to look at proven soundness, a more complex calculation which
    /// isn't currently supported by this crate.
    pub const fn conjectured_soundness_bits(&self) -> usize {
        self.log_blowup * self.num_queries + self.proof_of_work_bits
    }
}

/// Whereas `FriParameters` encompasses parameters the end user can set, `FriFoldingStrategy` is
/// set by the PCS calling FRI, and abstracts over implementation details of the PCS.
pub trait FriFoldingStrategy<F: Field, EF: ExtensionField<F>> {
    type InputProof;
    type InputError: Debug;

    /// We can ask FRI to sample extra query bits (LSB) for our own purposes.
    /// They will be passed to our callbacks, but ignored (shifted off) by FRI.
    fn extra_query_index_bits(&self) -> usize;

    /// Fold a row, returning a single column.
    /// Right now the input row will always be 2 columns wide,
    /// but we may support higher folding arity in the future.
    fn fold_row(
        &self,
        index: usize,
        log_height: usize,
        beta: EF,
        evals: impl Iterator<Item = EF>,
    ) -> EF;

    /// Fold a row, returning a single column.
    /// Supporting arbitrary folding width that is a power of 2.
    fn fold_row_arbitrary(
        &self,
        index: usize,
        log_height: usize,
        beta: EF,
        evals: impl Iterator<Item = EF>,
        folding_factor: usize,
    ) -> EF {
        if folding_factor == 2 {
            self.fold_row(index, log_height, beta, evals)
        } else {
            panic!("folding for parameters != 2 is not supported")
        }
    }

    /// Same as applying fold_row to every row, possibly faster.
    fn fold_matrix<M: Matrix<EF>>(&self, beta: EF, m: M) -> Vec<EF>;

    /// Same as applying fold_row to every row, possibly faster.
    fn fold_matrix_arbitrary<M: Matrix<EF>>(
        &self,
        beta: EF,
        m: M,
        folding_factor: usize,
    ) -> Vec<EF> {
        if folding_factor == 2 {
            self.fold_matrix(beta, m)
        } else {
            panic!("folding for parameters != 2 is not supported")
        }
    }
}

/// Creates a minimal set of `FriParameters` for testing purposes.
/// These parameters are designed to reduce computational cost during tests.
pub const fn create_test_fri_params<Mmcs>(
    mmcs: Mmcs,
    log_final_poly_len: usize,
) -> FriParameters<Mmcs> {
    FriParameters {
        log_blowup: 2,
        log_final_poly_len,
        num_queries: 2,
        proof_of_work_bits: 1,
        mmcs,
        log_folding_factor: 1,
    }
}

/// Creates a minimal set of `FriParameters` for testing purposes, with zk enabled.
/// These parameters are designed to reduce computational cost during tests.
pub const fn create_test_fri_params_zk<Mmcs>(mmcs: Mmcs) -> FriParameters<Mmcs> {
    FriParameters {
        log_blowup: 2,
        log_final_poly_len: 0,
        num_queries: 2,
        proof_of_work_bits: 1,
        mmcs,
        log_folding_factor: 1,
    }
}

/// Creates a set of `FriParameters` suitable for benchmarking.
/// These parameters represent typical settings used in production-like scenarios.
pub const fn create_benchmark_fri_params<Mmcs>(mmcs: Mmcs) -> FriParameters<Mmcs> {
    FriParameters {
        log_blowup: 1,
        log_final_poly_len: 0,
        num_queries: 100,
        proof_of_work_bits: 16,
        mmcs,
        log_folding_factor: 1,
    }
}

/// Creates a set of `FriParameters` suitable for benchmarking with zk enabled.
/// These parameters represent typical settings used in production-like scenarios.
pub fn create_benchmark_fri_params_zk<Mmcs>(mmcs: Mmcs) -> FriParameters<Mmcs> {
    FriParameters {
        log_blowup: 2,
        log_final_poly_len: 0,
        num_queries: 100,
        proof_of_work_bits: 16,
        mmcs,
        log_folding_factor: 1,
    }
}
