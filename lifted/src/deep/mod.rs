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

pub use interpolate::SinglePointQuotient;
use p3_commit::{BatchOpening, Mmcs};
use p3_field::{ExtensionField, Field, TwoAdicField};
pub use verifier::DeepError;

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

/// Prover's quotient data at a single opening point.
///
/// Bundles the precomputed quotient `1/(z - X)` with the polynomial evaluations
/// at that point. Used to construct the DEEP quotient polynomial.
pub struct QuotientOpening<'a, F: TwoAdicField, EF: ExtensionField<F>> {
    /// Precomputed quotient data for point `z`.
    pub quotient: &'a SinglePointQuotient<F, EF>,
    /// Evaluations `f_i(z^r)` grouped by commitment, then matrix, then column.
    /// The lift factor `r` varies per matrix based on its height.
    pub evals: Vec<MatrixGroupEvals<EF>>,
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::{BabyBear as F, Poseidon2BabyBear};
    use p3_challenger::{CanObserve, DuplexChallenger};
    use p3_commit::Mmcs;
    use p3_field::Field;
    use p3_field::extension::BinomialExtensionField;
    use p3_matrix::Matrix;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::distr::StandardUniform;
    use rand::prelude::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::prover::DeepPoly;
    use super::verifier::DeepOracle;
    use super::{OpeningClaim, QuotientOpening, SinglePointQuotient};
    use crate::merkle_tree::{Lifting, MerkleTreeLmcs};
    use crate::utils::bit_reversed_coset_points;

    type EF = BinomialExtensionField<F, 4>;
    type P = <F as Field>::Packing;

    const WIDTH: usize = 16;
    const RATE: usize = 8;
    const DIGEST: usize = 8;

    type Perm = Poseidon2BabyBear<WIDTH>;
    type Sponge = PaddingFreeSponge<Perm, WIDTH, RATE, DIGEST>;
    type Compressor = TruncatedPermutation<Perm, 2, DIGEST, WIDTH>;
    type Lmcs = MerkleTreeLmcs<P, P, Sponge, Compressor, WIDTH, DIGEST>;
    type Challenger = DuplexChallenger<F, Perm, WIDTH, RATE>;

    fn test_components() -> (Perm, Lmcs) {
        let mut rng = SmallRng::seed_from_u64(123);
        let perm = Perm::new_from_rng_128(&mut rng);
        let sponge = PaddingFreeSponge::<_, WIDTH, RATE, DIGEST>::new(perm.clone());
        let compress = TruncatedPermutation::<_, 2, DIGEST, WIDTH>::new(perm.clone());
        let lmcs = MerkleTreeLmcs::new(sponge, compress, Lifting::Upsample);
        (perm, lmcs)
    }

    /// End-to-end: prover's `DeepPoly.open()` must match verifier's `DeepOracle.query()`.
    #[test]
    fn deep_quotient_end_to_end() {
        let rng = &mut SmallRng::seed_from_u64(42);
        let (perm, lmcs) = test_components();

        // Parameters
        let log_blowup: usize = 2;
        let log_n = 10;
        let n = 1 << log_n;
        let alignment = RATE; // Use sponge rate for coefficient alignment

        // Two random opening points
        let z1: EF = rng.sample(StandardUniform);
        let z2: EF = rng.sample(StandardUniform);

        // Coset points in bit-reversed order
        let coset_points = bit_reversed_coset_points::<F>(log_n);

        // Create matrices of varying heights (ascending order required)
        // specs: (log_scaling, width) where height = n >> log_scaling
        let specs: Vec<(usize, usize)> = vec![(2, 2), (1, 3), (0, 4)]; // heights: n/4, n/2, n
        let matrices: Vec<RowMajorMatrix<F>> = specs
            .iter()
            .map(|&(log_scaling, width)| {
                let height = n >> log_scaling;
                RowMajorMatrix::rand(rng, height, width)
            })
            .collect();

        // Step 1: Commit matrices via LMCS
        let (commitment, prover_data) = lmcs.commit(matrices.clone());
        let dims: Vec<_> = matrices.iter().map(|m| m.dimensions()).collect();

        // Step 2: Compute evaluations at both opening points
        let q1 = SinglePointQuotient::<F, EF>::new(z1, &coset_points);
        let q2 = SinglePointQuotient::<F, EF>::new(z2, &coset_points);

        let matrices_ref: Vec<&RowMajorMatrix<F>> = matrices.iter().collect();
        let matrices_groups = [matrices_ref];
        let evals1 = q1.batch_eval_lifted(&matrices_groups, &coset_points, log_blowup);
        let evals2 = q2.batch_eval_lifted(&matrices_groups, &coset_points, log_blowup);

        // Step 3: Prover constructs DeepPoly with challenger
        // The challenger samples alpha (column batching) and beta (point batching) internally
        let mut prover_challenger = Challenger::new(perm.clone());
        prover_challenger.observe(commitment);

        let openings_for_prover: Vec<QuotientOpening<'_, F, EF>> = vec![
            QuotientOpening {
                quotient: &q1,
                evals: evals1.clone(),
            },
            QuotientOpening {
                quotient: &q2,
                evals: evals2.clone(),
            },
        ];
        let deep_poly = DeepPoly::new(
            &lmcs,
            &openings_for_prover,
            vec![&prover_data],
            &mut prover_challenger,
            alignment,
        );

        // Step 4: Verifier constructs DeepOracle with separate challenger (same initial state)
        // Verifier's challenger must start in the same state as prover's was before DeepPoly::new
        let mut verifier_challenger = Challenger::new(perm);
        verifier_challenger.observe(commitment);

        let openings_for_verifier: Vec<OpeningClaim<EF>> = vec![
            OpeningClaim {
                point: z1,
                evals: evals1,
            },
            OpeningClaim {
                point: z2,
                evals: evals2,
            },
        ];
        let deep_oracle = DeepOracle::new(
            &openings_for_verifier,
            vec![(commitment, dims)],
            &mut verifier_challenger,
            alignment,
        )
        .expect("DeepOracle construction should succeed");

        // Step 5: Verify at random query indices
        let sample_indices = [0, 1, n / 4, n / 2, n - 1];
        for &index in &sample_indices {
            // Prover opens at index
            let deep_query = deep_poly.open(&lmcs, index);

            // Verifier evaluates at index (also verifies Merkle proofs)
            let verifier_eval = deep_oracle
                .query(&lmcs, index, &deep_query)
                .expect("Merkle verification should pass");

            let prover_eval = deep_poly.evals()[index];
            assert_eq!(
                prover_eval, verifier_eval,
                "Prover and verifier disagree at index {index}"
            );
        }
    }
}
