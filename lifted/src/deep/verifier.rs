use alloc::vec::Vec;
use core::iter::zip;
use core::marker::PhantomData;

use p3_challenger::FieldChallenger;
use p3_commit::{BatchOpeningRef, Mmcs};
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::Dimensions;
use p3_util::{log2_strict_usize, reverse_bits_len};
use thiserror::Error;

use super::{DeepQuery, OpeningClaim};

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during DEEP verifier construction.
#[derive(Debug, Error)]
pub enum DeepError {
    /// No openings provided.
    #[error("no openings provided")]
    EmptyOpenings,
    /// Number of evaluation groups doesn't match number of commitments.
    #[error(
        "evaluation group count mismatch at opening {opening}: expected {expected}, got {actual}"
    )]
    EvalGroupCountMismatch {
        opening: usize,
        expected: usize,
        actual: usize,
    },
    /// Number of matrices in evaluation group doesn't match commitment dimensions.
    #[error(
        "matrix count mismatch at opening {opening}, group {group}: expected {expected}, got {actual}"
    )]
    MatrixCountMismatch {
        opening: usize,
        group: usize,
        expected: usize,
        actual: usize,
    },
    /// Number of columns in matrix evaluation doesn't match committed width.
    #[error(
        "column count mismatch at opening {opening}, group {group}, matrix {matrix}: expected {expected}, got {actual}"
    )]
    ColumnCountMismatch {
        opening: usize,
        group: usize,
        matrix: usize,
        expected: usize,
        actual: usize,
    },
}

/// Verifier's view of the DEEP quotient as a point-query oracle.
///
/// Stores commitments and the prover's reduced claims `(zⱼ, f_reduced(zⱼ))`.
/// At query time, verifies Merkle openings and reconstructs `Q(X)` at that point:
///
/// ```text
/// Q(X) = Σⱼ βʲ · (f_reduced(zⱼ) - f_reduced(X)) / (zⱼ - X)
/// ```
///
/// where `f_reduced = Σᵢ αⁱ · fᵢ` batches all polynomial columns.
///
/// From the verifier's perspective, all opened columns appear to have the same height—
/// lifting is transparent. The prover evaluates `fᵢ(zʳ)` for degree-d polynomials
/// (where r is the lift factor), but the verifier sees this as `fᵢ'(z)` where
/// `fᵢ'(X) = fᵢ(Xʳ)` is the lifted polynomial on the full domain.
///
/// An alternative implementation could open rows padded with zeros to the alignment
/// width, allowing the hasher to process fixed-size chunks. This implementation
/// uses alignment > 1 to support such padding virtually (without materializing zeros).
pub struct DeepOracle<F: TwoAdicField, EF: ExtensionField<F>, Commit: Mmcs<F>> {
    /// Commitments with their associated matrix dimensions.
    commitments: Vec<(Commit::Commitment, Vec<Dimensions>)>,

    /// Reduced openings: pairs of `(zⱼ, f_reduced(zⱼ))` from the prover's claims.
    reduced_openings: Vec<(EF, EF)>,

    /// Challenge `α` for batching columns into `f_reduced`.
    challenge_columns: EF,
    /// Challenge `β` for batching opening points.
    challenge_points: EF,

    /// Alignment width for Horner reduction (typically hasher's rate).
    /// When alignment > 1, coefficient indices are padded to multiples of this value,
    /// equivalent to virtually appending zeros to each row before hashing.
    alignment: usize,

    _marker: PhantomData<F>,
}

impl<F: TwoAdicField, EF: ExtensionField<F>, Commit: Mmcs<F>> DeepOracle<F, EF, Commit> {
    /// Construct from claimed openings and commitments.
    ///
    /// # Arguments
    /// - `openings`: Claimed evaluations at each opening point
    /// - `commitments`: Pairs `(commitment, dims)` for Merkle verification
    /// - `challenger`: Fiat-Shamir challenger for sampling challenges
    /// - `alignment`: Width for coefficient alignment (see struct doc)
    ///
    /// We reduce each opening's evaluations to `f_reduced(zⱼ) = Σᵢ αⁱ · fᵢ(zⱼʳ)` eagerly.
    /// This optimization is possible because all columns share the same opening points—
    /// at query time, we only compute one Horner reduction per query, not per-column.
    ///
    /// # Errors
    ///
    /// Returns `DeepError` if the proof structure is invalid.
    pub fn new<Challenger: FieldChallenger<F>>(
        openings: &[OpeningClaim<EF>],
        commitments: Vec<(Commit::Commitment, Vec<Dimensions>)>,
        challenger: &mut Challenger,
        alignment: usize,
    ) -> Result<Self, DeepError> {
        if openings.is_empty() {
            return Err(DeepError::EmptyOpenings);
        }
        let num_commits = commitments.len();

        // Validate structure: evals_groups[commit][matrix][col] matches dims
        for (opening_idx, claim) in openings.iter().enumerate() {
            if claim.evals.len() != num_commits {
                return Err(DeepError::EvalGroupCountMismatch {
                    opening: opening_idx,
                    expected: num_commits,
                    actual: claim.evals.len(),
                });
            }
            for (group_idx, (evals, (_, dims))) in zip(&claim.evals, &commitments).enumerate() {
                if evals.0.len() != dims.len() {
                    return Err(DeepError::MatrixCountMismatch {
                        opening: opening_idx,
                        group: group_idx,
                        expected: dims.len(),
                        actual: evals.0.len(),
                    });
                }
                for (matrix_idx, (matrix_evals, matrix_dims)) in zip(&evals.0, dims).enumerate() {
                    if matrix_evals.len() != matrix_dims.width {
                        return Err(DeepError::ColumnCountMismatch {
                            opening: opening_idx,
                            group: group_idx,
                            matrix: matrix_idx,
                            expected: matrix_dims.width,
                            actual: matrix_evals.len(),
                        });
                    }
                }
            }
        }

        let challenge_columns: EF = challenger.sample_algebra_element();
        let challenge_points: EF = challenger.sample_algebra_element();

        // Reduce each opening's evaluations via Horner: (z_j, f_reduced(z_j))
        let reduced_openings: Vec<(EF, EF)> = openings
            .iter()
            .map(|claim| {
                let slices = claim.evals.iter().flat_map(|g| g.iter());
                let reduced_eval = reduce_with_powers(slices, challenge_columns, alignment);
                (claim.point, reduced_eval)
            })
            .collect();

        Ok(Self {
            commitments,
            reduced_openings,
            challenge_columns,
            challenge_points,
            alignment,
            _marker: PhantomData,
        })
    }

    /// Verify Merkle openings and compute `Q(X)` at the queried domain point.
    ///
    /// Reduces opened row values via Horner to get `f_reduced(X)`, then computes
    /// `Σⱼ βʲ · (f_reduced(zⱼ) - f_reduced(X)) / (zⱼ - X)`.
    pub fn query(
        &self,
        c: &Commit,
        index: usize,
        proof: &DeepQuery<F, Commit>,
    ) -> Result<EF, Commit::Error> {
        for ((commit, dims), opening) in zip(&self.commitments, &proof.openings) {
            c.verify_batch(commit, dims, index, opening.into())?;
        }

        let rows_iter = proof
            .openings
            .iter()
            .flat_map(|opening| BatchOpeningRef::from(opening).opened_values)
            .map(Vec::as_slice);

        // Reconstruct the domain point X from the query index.
        // The LDE domain is the coset gK in bit-reversed order where:
        //   g = F::GENERATOR (coset shift, avoids subgroup)
        //   K = <ω> with ω = primitive 2^log_n root of unity
        // In bit-reversed order: X = g · ω^{bit_rev(index)}
        let row_point = {
            let max_height = self.commitments.last().unwrap().1.last().unwrap().height;
            let log_max_height = log2_strict_usize(max_height);
            let generator = F::two_adic_generator(log_max_height);
            let shift = F::GENERATOR;
            let index_bit_rev = reverse_bits_len(index, log_max_height);
            shift * generator.exp_u64(index_bit_rev as u64)
        };

        let reduced_row = reduce_with_powers(rows_iter, self.challenge_columns, self.alignment);

        let eval = zip(&self.reduced_openings, self.challenge_points.powers())
            .map(|((point, reduced_eval), coeff_point)| {
                coeff_point * (*reduced_eval - reduced_row) / (*point - row_point)
            })
            .sum();
        Ok(eval)
    }
}

/// Horner reduction: computes `Σᵢ αⁿ⁻¹⁻ⁱ · vᵢ` via left-to-right accumulation.
///
/// For each value v, computes `acc = α·acc + v`. The reversed coefficient order
/// (from [`super::prover::derive_coeffs_from_challenge`]) makes this produce the
/// same result as explicit `Σᵢ coeffs[i] · vals[i]`.
///
/// # Alignment
///
/// After each slice, multiplies by `α^gap` where `gap` pads the slice length to
/// the next multiple of `alignment`. This is equivalent to:
/// - Padding each row with zeros to the alignment width
/// - Including those zeros in the Horner accumulation
///
/// An alternative implementation could materialize zero-padded rows; this approach
/// achieves the same result without allocating the padding.
pub(crate) fn reduce_with_powers<'a, F, EF>(
    slices: impl IntoIterator<Item = &'a [F]>,
    challenge: EF,
    alignment: usize,
) -> EF
where
    F: Field + 'a,
    EF: ExtensionField<F>,
{
    slices.into_iter().fold(EF::ZERO, |acc, slice| {
        // Horner's method on this slice: acc = α·acc + v for each v
        let acc = slice.iter().fold(acc, |a, &val| a * challenge + val);
        // Skip alignment gap: equivalent to processing implicit zeros
        let gap = slice.len().next_multiple_of(alignment) - slice.len();
        acc * challenge.exp_u64(gap as u64)
    })
}
