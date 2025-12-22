//! # DEEP Quotient for Lifted FRI
//!
//! DEEP converts evaluation claims into a low-degree test. Given committed polynomials
//! `{fᵢ}` and claimed evaluations `fᵢ(zⱼ) = vᵢⱼ`, the quotient
//!
//! ```text
//! Q(X) = Σⱼ βʲ · Σᵢ αⁱ · (vᵢⱼ - fᵢ(X)) / (zⱼ - X)
//! ```
//!
//! is low-degree iff all claims are correct. A false claim creates a pole, detectable by FRI.
//!
//! ## Design Choices
//!
//! **Uniform opening points.** All columns share the same opening points `{zⱼ}`. This enables
//! factoring out `f_reduced(X) = Σᵢ αⁱ·fᵢ(X)`, so the verifier computes one inner product
//! per query rather than one per column per point.
//!
//! **Two challenges.** Separating α (columns) from β (points) improves soundness. With a
//! single challenge, a cheating prover must avoid collisions among k·m terms; with two,
//! only k+m terms matter. This costs one extra field element in the transcript.
//!
//! **Lifting.** Polynomials of degree d on domain D embed into a larger domain D* via
//! `f(X) ↦ f(Xʳ)` where r = |D*|/|D|. In bit-reversed order, this means each evaluation
//! repeats r times consecutively—implemented by virtual upsampling without data movement.
//!
//! **Verifier's view of lifting.** From the verifier's perspective, all polynomials
//! appear to be evaluated at the same point z on the same domain. The prover computes
//! `fᵢ(zʳ)` for degree-d polynomials, but this equals `fᵢ'(z)` where `fᵢ'(X) = fᵢ(Xʳ)`
//! is the lifted polynomial. This uniformity enables the `f_reduced` factorization.

pub mod interpolate;
pub mod prover;
pub mod verifier;

use alloc::vec::Vec;

use p3_challenger::FieldChallenger;
use p3_commit::{BatchOpening, Mmcs};
use p3_field::{ExtensionField, Field};

use crate::utils::alignment_padding;

/// Challenges for DEEP quotient batching.
///
/// - `alpha` (α): Batches polynomial columns into `f_reduced = Σᵢ αⁱ·fᵢ`
/// - `beta` (β): Batches opening points into `Q = Σⱼ βʲ·Qⱼ`
///
/// Constructed via [`DeepChallenges::sample`], which observes the evaluations
/// before sampling to enforce correct Fiat-Shamir transcript ordering.
#[derive(Clone, Copy, Debug)]
pub struct DeepChallenges<EF> {
    /// Column batching challenge α
    pub alpha: EF,
    /// Point batching challenge β
    pub beta: EF,
}

impl<EF: Field> DeepChallenges<EF> {
    /// Observe evaluations and sample DEEP challenges from the transcript.
    ///
    /// This enforces the correct Fiat-Shamir order: observe data, then sample.
    /// Each matrix's columns are observed with alignment padding to match
    /// the coefficient derivation in the DEEP quotient.
    pub fn sample<F, Challenger>(
        evals: &[Vec<MatrixGroupEvals<EF>>],
        challenger: &mut Challenger,
        alignment: usize,
    ) -> Self
    where
        F: Field,
        EF: ExtensionField<F>,
        Challenger: FieldChallenger<F>,
    {
        // Observe evaluations with alignment padding
        for point_evals in evals {
            for group_evals in point_evals {
                for matrix_evals in group_evals.iter_matrices() {
                    for val in matrix_evals {
                        challenger.observe_algebra_element(*val);
                    }
                    // Pad to alignment with zeros (must match coefficient alignment)
                    for _ in 0..alignment_padding(matrix_evals.len(), alignment) {
                        challenger.observe_algebra_element(EF::ZERO);
                    }
                }
            }
        }

        Self {
            alpha: challenger.sample_algebra_element(),
            beta: challenger.sample_algebra_element(),
        }
    }
}

/// Query proof containing Merkle openings for DEEP quotient verification.
///
/// Holds the batch openings from the input commitment that the verifier
/// needs to reconstruct `f_reduced(X)` at the queried point.
pub struct DeepQuery<F: Field, Commit: Mmcs<F>> {
    openings: Vec<BatchOpening<F, Commit>>,
}

/// Evaluations of polynomial columns at an out-of-domain point, organized by matrix.
///
/// Structure: `evals[matrix_idx][column_idx]` holds `f_{matrix,col}(z)`.
///
/// The grouping by matrix preserves the structure needed for batched reduction,
/// where matrices are processed in height order and each matrix's columns are
/// reduced with consecutive challenge powers.
#[derive(Clone, Debug)]
pub struct MatrixGroupEvals<T>(pub(crate) Vec<Vec<T>>);

impl<T> MatrixGroupEvals<T> {
    /// Create a new `MatrixGroupEvals` from nested vectors.
    ///
    /// Structure: `evals[matrix_idx][column_idx]` for each matrix in a commitment group.
    pub const fn new(evals: Vec<Vec<T>>) -> Self {
        Self(evals)
    }

    /// Returns the number of matrices in this group.
    pub const fn num_matrices(&self) -> usize {
        self.0.len()
    }

    /// Iterate over matrices, yielding the column evaluations for each.
    pub fn iter_matrices(&self) -> impl Iterator<Item = &[T]> {
        self.0.iter().map(|v| v.as_slice())
    }

    /// Iterate over all column evaluations across all matrices.
    ///
    /// Yields evaluations in order: all columns of matrix 0, then matrix 1, etc.
    pub fn iter_evals(&self) -> impl Iterator<Item = &T> {
        self.0.iter().flatten()
    }

    /// Transform each evaluation using the provided closure.
    ///
    /// Preserves the matrix/column structure while mapping `T -> U`.
    pub fn map<U, F: FnMut(&T) -> U>(&self, mut f: F) -> MatrixGroupEvals<U> {
        MatrixGroupEvals::new(
            self.0
                .iter()
                .map(|matrix| matrix.iter().map(&mut f).collect())
                .collect(),
        )
    }
}

/// A claimed evaluation at a single point, with evaluations grouped by commitment.
///
/// Used by the verifier to check prover claims. Structure:
/// `evals[commit_idx][matrix_idx][col_idx]` = claimed value at `point`.
#[derive(Clone, Debug)]
pub struct OpeningClaim<EF> {
    /// The out-of-domain evaluation point `z`.
    pub point: EF,
    /// Claimed evaluations `f_i(z)` grouped by commitment, then matrix, then column.
    pub evals: Vec<MatrixGroupEvals<EF>>,
}

#[cfg(test)]
mod tests;
