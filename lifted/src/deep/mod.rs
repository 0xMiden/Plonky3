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

use p3_field::TwoAdicField;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_util::reverse_slice_index_bits;

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

/// Coset points `gK` in bit-reversed order.
///
/// Bit-reversal gives two properties essential for lifting:
/// - **Adjacent negation**: `gK[2i+1] = -gK[2i]`, so both square to the same value
/// - **Prefix nesting**: `gK[0..n/r]` equals the r-th power coset `(gK)ʳ`
///
/// Together these enable iterative weight folding in barycentric evaluation.
pub fn bit_reversed_coset_points<F: TwoAdicField>(log_n: usize) -> Vec<F> {
    let coset = TwoAdicMultiplicativeCoset::new(F::GENERATOR, log_n).unwrap();
    let mut pts: Vec<F> = coset.iter().collect();
    reverse_slice_index_bits(&mut pts);
    pts
}
#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::{BabyBear as F, Poseidon2BabyBear};
    use p3_commit::Mmcs;
    use p3_field::Field;
    use p3_field::extension::BinomialExtensionField;
    use p3_matrix::Matrix;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::distr::StandardUniform;
    use rand::prelude::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::interpolate::SinglePointQuotient;
    use super::prover::DeepPoly;
    use super::verifier::DeepOracle;
    use super::{MatrixGroupEvals, bit_reversed_coset_points};
    use crate::merkle_tree::{Lifting, MerkleTreeLmcs};

    type EF = BinomialExtensionField<F, 4>;
    type P = <F as Field>::Packing;

    const WIDTH: usize = 16;
    const RATE: usize = 8;
    const DIGEST: usize = 8;

    type Sponge = PaddingFreeSponge<Poseidon2BabyBear<WIDTH>, WIDTH, RATE, DIGEST>;
    type Compressor = TruncatedPermutation<Poseidon2BabyBear<WIDTH>, 2, DIGEST, WIDTH>;
    type Lmcs = MerkleTreeLmcs<P, P, Sponge, Compressor, WIDTH, DIGEST>;

    fn make_lmcs() -> Lmcs {
        let mut rng = SmallRng::seed_from_u64(123);
        let perm = Poseidon2BabyBear::<WIDTH>::new_from_rng_128(&mut rng);
        let sponge = PaddingFreeSponge::<_, WIDTH, RATE, DIGEST>::new(perm.clone());
        let compress = TruncatedPermutation::<_, 2, DIGEST, WIDTH>::new(perm);
        MerkleTreeLmcs::new(sponge, compress, Lifting::Upsample)
    }

    /// End-to-end: prover's `DeepPoly.open()` must match verifier's `DeepOracle.eval()`.
    #[test]
    fn deep_quotient_end_to_end() {
        let rng = &mut SmallRng::seed_from_u64(42);
        let lmcs = make_lmcs();

        // Parameters
        let log_blowup: usize = 2;
        let log_n = 10;
        let n = 1 << log_n;
        let alignment = RATE; // Use sponge rate for coefficient alignment

        // Two random opening points and challenges
        let z1: EF = rng.sample(StandardUniform);
        let z2: EF = rng.sample(StandardUniform);
        let alpha: EF = rng.sample(StandardUniform); // Column batching challenge
        let beta: EF = rng.sample(StandardUniform); // Point batching challenge

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

        // Step 3: Prover constructs DeepPoly
        #[allow(clippy::type_complexity)]
        let openings_for_prover: Vec<(&SinglePointQuotient<F, EF>, Vec<MatrixGroupEvals<EF>>)> =
            vec![(&q1, evals1.clone()), (&q2, evals2.clone())];
        let deep_poly = DeepPoly::new(
            &lmcs,
            &openings_for_prover,
            vec![&prover_data],
            beta,  // challenge_points
            alpha, // challenge_columns
            alignment,
        );

        // Step 4: Verifier constructs DeepOracle from claimed openings
        let openings_for_verifier: Vec<(EF, Vec<MatrixGroupEvals<EF>>)> =
            vec![(z1, evals1), (z2, evals2)];
        let deep_oracle = DeepOracle::new(
            &openings_for_verifier,
            vec![(commitment, dims)],
            alpha, // challenge_columns
            beta,  // challenge_points
            alignment,
        );

        // Step 5: Verify at random query indices
        let sample_indices = [0, 1, n / 4, n / 2, n - 1];
        for &index in &sample_indices {
            // Prover opens at index
            let (prover_eval, batch_openings) = deep_poly.open(&lmcs, index);

            // Verifier evaluates at index (also verifies Merkle proofs)
            let verifier_eval = deep_oracle
                .eval(&lmcs, index, &batch_openings)
                .expect("Merkle verification should pass");

            assert_eq!(
                prover_eval, verifier_eval,
                "Prover and verifier disagree at index {index}"
            );
        }
    }
}
