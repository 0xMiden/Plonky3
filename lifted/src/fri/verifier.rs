use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_commit::{BatchOpening, BatchOpeningRef, Mmcs};
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::Dimensions;
use p3_util::{log2_strict_usize, reverse_bits_len};

use super::MatrixGroupEvals;
use crate::fri::deep::{eval_deep, reduce_with_powers};

/// Virtual polynomial for FRI verification.
///
/// Represents a batched polynomial constructed from multiple committed matrices,
/// evaluated at multiple opening points. The polynomial is "virtual" because it
/// is never explicitly constructed; instead, evaluations are computed on-the-fly
/// during FRI verification.
pub struct VirtualPoly<F: TwoAdicField, EF: ExtensionField<F>, Commit: Mmcs<F>> {
    /// Commitments with their associated matrix dimensions.
    /// Each entry is `(commitment, dims)` where `dims[i]` gives the dimensions
    /// of the i-th matrix in that commitment batch.
    commitments: Vec<(Commit::Commitment, Vec<Dimensions>)>,

    /// Reduced openings: pairs of `(point, reduced_eval)` where `reduced_eval`
    /// is the batched evaluation of all polynomials at `point` using Horner's method.
    reduced_openings: Vec<(EF, EF)>,

    /// Batching challenge used to combine multiple polynomials into one.
    challenge: EF,

    /// Coefficient alignment width for Horner reduction.
    /// Must match the padding used during commitment.
    padding: usize,

    _marker: PhantomData<F>,
}

impl<F: TwoAdicField, EF: ExtensionField<F>, Commit: Mmcs<F>> VirtualPoly<F, EF, Commit> {
    /// Create a new VirtualPoly from openings and commit/dimensions pairs.
    ///
    /// # Arguments
    /// - `openings`: List of `(point, evals_groups)` where evals_groups is `[commit][matrix][col]`
    /// - `commitments`: List of `(commitment, dims)` pairs where dims is `[matrix]`
    /// - `challenge`: Batching challenge for Horner reduction
    /// - `padding`: Coefficient alignment for Horner reduction
    pub fn new(
        openings: &[(EF, Vec<MatrixGroupEvals<EF>>)],
        commitments: Vec<(Commit::Commitment, Vec<Dimensions>)>,
        challenge: EF,
        padding: usize,
    ) -> Self {
        // Must have at least one opening point to construct a meaningful DEEP quotient.
        assert!(!openings.is_empty(), "must have at least one opening");

        let num_commits = commitments.len();

        // Validate that the structure of evals_groups matches commitments:
        // - Each opening must have evaluations for all commitments
        // - Each commitment's evals must match the number of matrices
        // - Each matrix's evals must match the matrix width (number of columns)
        for (_, evals_groups) in openings {
            assert_eq!(
                evals_groups.len(),
                num_commits,
                "evals must have same number of commits"
            );
            for (evals, (_, dims)) in evals_groups.iter().zip(&commitments) {
                assert_eq!(
                    evals.0.len(),
                    dims.len(),
                    "evals must have same number of matrices as dims"
                );
                for (matrix_evals, matrix_dims) in evals.0.iter().zip(dims) {
                    assert_eq!(
                        matrix_evals.len(),
                        matrix_dims.width,
                        "evals must have same number of columns as dims.width"
                    );
                }
            }
        }

        // For each opening point z_j, compute the reduced evaluation:
        //   reduced_eval_j = Σ_i challenge^{n-1-i} * f_i(z_j)
        // where f_i ranges over all polynomial columns across all matrices.
        // This uses Horner's method with padding alignment to match the
        // coefficient structure used during commitment.
        let reduced_openings: Vec<(EF, EF)> = openings
            .iter()
            .map(|(point, evals_groups)| {
                // Flatten [commit][matrix][col] -> iterator of &[EF] slices
                let slices = evals_groups.iter().flat_map(|g| g.iter());
                // Compute batched evaluation using Horner's method
                let reduced_eval = reduce_with_powers(slices, challenge, padding);
                (*point, reduced_eval)
            })
            .collect();

        Self {
            commitments,
            reduced_openings,
            challenge,
            padding,
            _marker: PhantomData,
        }
    }

    /// Evaluate the virtual polynomial at a domain point specified by `index`.
    ///
    /// This verifies the Merkle openings and computes the DEEP quotient:
    /// ```text
    /// Q(X) = Σ_j (reduced_eval_j - reduced_row) / (z_j - X)
    /// ```
    /// where `reduced_row` is the batched evaluation of opened values at row `index`.
    ///
    /// # Arguments
    /// - `c`: The MMCS (Merkle) commitment scheme for verification
    /// - `index`: Row index in the committed matrices (bit-reversed order)
    /// - `openings`: Merkle opening proofs for each commitment at `index`
    ///
    /// # Returns
    /// The DEEP quotient evaluation at the domain point, or an error if verification fails.
    pub fn eval(
        &self,
        c: &Commit,
        index: usize,
        openings: &[BatchOpening<F, Commit>],
    ) -> Result<EF, Commit::Error> {
        // Step 1: Verify each Merkle opening against its commitment.
        // This ensures the opened values are consistent with the committed matrices.
        for ((commit, dims), opening) in self.commitments.iter().zip(openings) {
            c.verify_batch(commit, dims, index, opening.into())?;
        }

        // Step 2: Extract opened row values from all openings.
        // Flatten [commit][matrix][col] -> iterator of row slices.
        // Each slice contains one row from a matrix (values at `index` for all columns).
        let rows_iter = openings
            .iter()
            .flat_map(|opening| BatchOpeningRef::from(opening).opened_values)
            .map(Vec::as_slice);

        // Step 3: Compute the domain point X corresponding to `index`.
        // The domain is the coset gK in bit-reversed order, where:
        // - g = F::GENERATOR (coset shift)
        // - K = <ω> is the subgroup of order max_height
        // - X = g * ω^{bit_rev(index)}
        let row_point = {
            // Get the largest matrix height to determine the domain size
            let max_height = self.commitments.last().unwrap().1.last().unwrap().height;
            let log_max_height = log2_strict_usize(max_height);

            // ω = primitive root of unity for the subgroup K
            let generator = F::two_adic_generator(log_max_height);

            // g = coset shift (multiplicative generator of F*)
            let shift = F::GENERATOR;

            // Convert index to bit-reversed form for the coset ordering
            let index_bit_rev = reverse_bits_len(index, log_max_height);

            // X = g * ω^{bit_rev(index)}
            shift * generator.exp_u64(index_bit_rev as u64)
        };

        // Step 4: Compute the DEEP quotient at row_point.
        // This evaluates: Q(X) = Σ_j (reduced_eval_j - reduced_row) / (z_j - X)
        // where reduced_row = Σ_i challenge^{n-1-i} * row_i is computed inside eval_deep.
        let eval = eval_deep(
            &self.reduced_openings,
            rows_iter,
            row_point,
            self.challenge,
            self.padding,
        );
        Ok(eval)
    }
}
