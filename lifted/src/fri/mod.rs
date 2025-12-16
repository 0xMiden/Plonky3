//! # FRI Protocol Implementation
//!
//! Fast Reed-Solomon Interactive Oracle Proof for low-degree testing.
//! Proves that a committed polynomial has degree below a target bound.

pub mod fold;
pub mod prover;
pub mod verifier;

pub use prover::CommitPhaseData;
use thiserror::Error;
pub use verifier::CommitPhaseProof;

/// FRI protocol parameters.
///
/// Controls the trade-off between proof size, prover time, and verifier time.
pub struct FriParams {
    /// Log₂ of the blowup factor (LDE domain size / polynomial degree).
    ///
    /// Higher values increase soundness but also proof size and prover time.
    /// Typical values: 2-4 (blowup factors of 4-16).
    pub log_blowup: usize,

    /// Log₂ of the folding factor per round.
    ///
    /// - `1`: Arity-2 folding (halves degree per round)
    /// - `2`: Arity-4 folding (quarters degree per round)
    pub log_folding_factor: usize,

    /// Log₂ of the final polynomial degree.
    ///
    /// Folding stops when degree reaches `2^log_final_degree`.
    /// Final polynomial is sent in clear (coefficients, not evaluations).
    pub log_final_degree: usize,

    /// Number of query repetitions for soundness amplification.
    ///
    /// Each query provides ~`log_blowup` bits of security.
    /// Total security ≈ `num_queries * log_blowup` bits.
    pub num_queries: usize,
}

impl FriParams {
    /// Compute the number of folding rounds for a given initial evaluation domain size.
    ///
    /// Each round reduces the domain by `2^log_folding_factor`. We fold until the domain
    /// size reaches `2^(log_final_degree + log_blowup)`, at which point the polynomial
    /// degree is at most `2^log_final_degree`.
    ///
    /// Uses `div_ceil` to round up, ensuring we always reach the target degree even if
    /// the domain size doesn't divide evenly by the folding factor.
    #[inline]
    pub const fn num_rounds(&self, log_domain_size: usize) -> usize {
        // Final domain size = final_degree × blowup = 2^(log_final_degree + log_blowup)
        let log_max_final_size = self.log_final_degree + self.log_blowup;
        // Number of times we need to divide by 2^log_folding_factor
        log_domain_size
            .saturating_sub(log_max_final_size)
            .div_ceil(self.log_folding_factor)
    }

    /// Compute the final polynomial degree after folding.
    ///
    /// After `num_rounds` folding rounds, the domain shrinks from `2^log_domain_size`
    /// to `2^(log_domain_size - num_rounds × log_folding_factor)`. The polynomial
    /// degree is then `domain_size / blowup`.
    ///
    /// Due to `div_ceil` in `num_rounds`, the actual final degree may be smaller than
    /// `2^log_final_degree` when the folding doesn't divide evenly.
    #[inline]
    pub const fn final_poly_degree(&self, log_domain_size: usize) -> usize {
        let num_rounds = self.num_rounds(log_domain_size);
        // log of final domain size after folding
        let log_final_size = log_domain_size - num_rounds * self.log_folding_factor;
        // degree = domain_size / blowup = 2^(log_final_size - log_blowup)
        1 << log_final_size.saturating_sub(self.log_blowup)
    }
}

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during FRI verification.
#[derive(Debug, Error)]
pub enum FriError<MmcsError> {
    /// Merkle verification failed.
    #[error("Merkle verification failed at round {1}: {0:?}")]
    MmcsError(MmcsError, usize),
    /// Proof structure doesn't match expected format.
    ///
    /// This includes wrong number of commitments, openings, betas, or final polynomial length.
    #[error("invalid proof structure")]
    InvalidProofStructure,
    /// Evaluation mismatch during folding.
    #[error("evaluation mismatch at row {row_index}, position {position}")]
    EvaluationMismatch { row_index: usize, position: usize },
    /// Final polynomial evaluation doesn't match folded value.
    #[error("final polynomial mismatch")]
    FinalPolyMismatch,
}

#[cfg(test)]
mod tests;
