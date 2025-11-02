use alloc::vec::Vec;
use core::fmt::Debug;

use p3_field::{ExtensionField, Field};
use p3_matrix::Matrix;

/// A set of parameters defining a specific instance of the FRI protocol.
#[derive(Debug)]
pub struct FriParameters<M> {
    pub log_blowup: usize,
    // TODO: This parameter and FRI early stopping are not yet implemented in `CirclePcs`.
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

    /// Checks if the FRI parameters are valid.
    ///
    /// - `log_folding_factor` must be at least 1 (folding factor >= 2)
    /// - `log_final_poly_len` should be a multiple of `log_folding_factor`
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.log_folding_factor == 0 {
            return Err(
                "log_folding_factor must be at least 1 (folding factor must be at least 2)",
            );
        }

        // Check if log_final_poly_len is compatible with log_folding_factor
        // When folding by 2^k, we reduce the degree by k bits each round
        // So log_final_poly_len should be reachable from the initial degree
        if self.log_final_poly_len % self.log_folding_factor != 0 {
            return Err(
                "log_final_poly_len should be a multiple of log_folding_factor for optimal alignment",
            );
        }

        Ok(())
    }

    /// Creates new FRI parameters with validation.
    /// Returns an error if the parameters are invalid.
    ///
    /// This is the preferred method to create a FriParameter.
    pub fn new(
        log_blowup: usize,
        log_final_poly_len: usize,
        num_queries: usize,
        proof_of_work_bits: usize,
        mmcs: M,
        log_folding_factor: usize,
    ) -> Result<Self, &'static str> {
        let params = Self {
            log_blowup,
            log_final_poly_len,
            num_queries,
            proof_of_work_bits,
            mmcs,
            log_folding_factor,
        };
        params.validate()?;
        Ok(params)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_compatible_params() {
        // Compatible: log_final_poly_len is a multiple of log_folding_factor
        let params = FriParameters {
            log_blowup: 1,
            log_final_poly_len: 4,
            num_queries: 10,
            proof_of_work_bits: 8,
            mmcs: (),
            log_folding_factor: 2,
        };
        assert!(params.validate().is_ok());

        let params = FriParameters {
            log_blowup: 1,
            log_final_poly_len: 6,
            num_queries: 10,
            proof_of_work_bits: 8,
            mmcs: (),
            log_folding_factor: 3,
        };
        assert!(params.validate().is_ok());

        let params = FriParameters {
            log_blowup: 1,
            log_final_poly_len: 0,
            num_queries: 10,
            proof_of_work_bits: 8,
            mmcs: (),
            log_folding_factor: 2,
        };
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_validate_incompatible_params() {
        // Incompatible: log_final_poly_len is not a multiple of log_folding_factor
        let params = FriParameters {
            log_blowup: 1,
            log_final_poly_len: 3,
            num_queries: 10,
            proof_of_work_bits: 8,
            mmcs: (),
            log_folding_factor: 2,
        };
        assert!(params.validate().is_err());

        let params = FriParameters {
            log_blowup: 1,
            log_final_poly_len: 5,
            num_queries: 10,
            proof_of_work_bits: 8,
            mmcs: (),
            log_folding_factor: 3,
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_validate_zero_folding_factor() {
        // Invalid: log_folding_factor must be at least 1
        let params = FriParameters {
            log_blowup: 1,
            log_final_poly_len: 0,
            num_queries: 10,
            proof_of_work_bits: 8,
            mmcs: (),
            log_folding_factor: 0,
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_new_constructor_validates() {
        // Valid parameters
        let result = FriParameters::new(1, 4, 10, 8, (), 2);
        assert!(result.is_ok());

        // Invalid parameters
        let result = FriParameters::new(1, 3, 10, 8, (), 2);
        assert!(result.is_err());

        // Zero folding factor
        let result = FriParameters::new(1, 0, 10, 8, (), 0);
        assert!(result.is_err());
    }
}
