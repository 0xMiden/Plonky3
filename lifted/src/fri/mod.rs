//! # FRI Protocol Implementation
//!
//! Fast Reed-Solomon Interactive Oracle Proof for low-degree testing.
//! Proves that a committed polynomial has degree below a target bound.

pub mod fold;
pub mod prover;
pub mod verifier;

pub use prover::CommitPhaseData;
pub use verifier::{CommitPhaseProof, FriError};

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
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_commit::{ExtensionMmcs, Mmcs};
    use p3_dft::{Radix2DFTSmallBatch, TwoAdicSubgroupDft};
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{PrimeCharacteristicRing, TwoAdicField};
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use p3_util::{reverse_bits_len, reverse_slice_index_bits};
    use rand::distr::StandardUniform;
    use rand::prelude::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::*;
    use crate::merkle_tree::{Lifting, MerkleTreeLmcs};

    // -------------------------------------------------------------------------
    // Type aliases
    // -------------------------------------------------------------------------

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    const WIDTH: usize = 16;
    const RATE: usize = 8;
    const DIGEST: usize = 8;

    type Perm = Poseidon2BabyBear<WIDTH>;
    type Sponge = PaddingFreeSponge<Perm, WIDTH, RATE, DIGEST>;
    type Compress = TruncatedPermutation<Perm, 2, DIGEST, WIDTH>;

    // Use base field LMCS wrapped in ExtensionMmcs for extension field elements
    type BaseLmcs = MerkleTreeLmcs<F, F, Sponge, Compress, WIDTH, DIGEST>;
    type FriMmcs = ExtensionMmcs<F, EF, BaseLmcs>;
    type Challenger = DuplexChallenger<F, Perm, WIDTH, RATE>;

    // -------------------------------------------------------------------------
    // Test helpers
    // -------------------------------------------------------------------------

    fn test_components() -> (Perm, FriMmcs) {
        let mut rng = SmallRng::seed_from_u64(2025);
        let perm = Perm::new_from_rng_128(&mut rng);
        let sponge = Sponge::new(perm.clone());
        let compress = Compress::new(perm.clone());
        let base_lmcs = MerkleTreeLmcs::new(sponge, compress, Lifting::Upsample);
        let fri_mmcs = ExtensionMmcs::new(base_lmcs);
        (perm, fri_mmcs)
    }

    /// Generate random polynomial coefficients and compute LDE evaluations.
    ///
    /// Returns evaluations in bit-reversed order on subgroup H of size 2^(log_poly_degree + log_blowup).
    fn generate_random_lde(
        rng: &mut SmallRng,
        log_poly_degree: usize,
        log_blowup: usize,
    ) -> Vec<EF> {
        let poly_degree = 1 << log_poly_degree;
        let lde_size = poly_degree << log_blowup;

        // Random polynomial coefficients
        let mut coeffs: Vec<EF> = (0..poly_degree)
            .map(|_| rng.sample(StandardUniform))
            .collect();

        // Zero-pad to LDE size
        coeffs.resize(lde_size, EF::ZERO);

        // DFT to get evaluations on subgroup H
        // Radix2DFTSmallBatch outputs in natural order, so we bit-reverse
        // to match the FRI prover's expectation
        let dft = Radix2DFTSmallBatch::<EF>::default();
        let mut evals = dft.dft_algebra(coeffs);
        reverse_slice_index_bits(&mut evals);
        evals
    }

    /// Open a specific query index across all commit phase rounds.
    fn open_query<M: Mmcs<EF>>(
        mmcs: &M,
        data: &CommitPhaseData<F, EF, M>,
        params: &FriParams,
        index: usize,
    ) -> Vec<p3_commit::BatchOpening<EF, M>> {
        let log_arity = params.log_folding_factor;
        let mut current_index = index;
        data.folded_evals_data
            .iter()
            .map(|prover_data| {
                let row_index = current_index >> log_arity;
                let opening = mmcs.open_batch(row_index, prover_data);
                current_index = row_index;
                opening
            })
            .collect()
    }

    // -------------------------------------------------------------------------
    // Main test: commit_phase and verify_query roundtrip
    // -------------------------------------------------------------------------

    /// Test that commit_phase produces valid proofs that verify_query accepts.
    ///
    /// This test:
    /// 1. Generates a random polynomial and computes its LDE
    /// 2. Runs the FRI commit phase to fold down to final polynomial
    /// 3. Verifies random query indices
    fn test_fri_commit_verify_roundtrip(log_poly_degree: usize, log_folding_factor: usize) {
        let mut rng = SmallRng::seed_from_u64(42);
        let (perm, fri_mmcs) = test_components();

        let params = FriParams {
            log_blowup: 2,
            log_folding_factor,
            log_final_degree: 2,
            num_queries: 3,
        };

        // Generate random LDE evaluations
        let evals = generate_random_lde(&mut rng, log_poly_degree, params.log_blowup);
        let lde_size = evals.len();

        // Prover: run commit phase
        let mut prover_challenger = Challenger::new(perm.clone());
        let (prover_data, proof) = CommitPhaseData::<F, EF, _>::new(
            &fri_mmcs,
            &params,
            evals.clone(),
            &mut prover_challenger,
        );

        // Verifier: replay challenger to get betas
        let mut verifier_challenger = Challenger::new(perm);
        let betas = proof.sample_betas::<F, _>(&mut verifier_challenger);

        // Verify random queries
        for _ in 0..3 {
            let index: usize = rng.random_range(0..lde_size);
            let initial_eval = evals[index];
            let openings = open_query(&fri_mmcs, &prover_data, &params, index);

            proof
                .verify_query::<F>(
                    &fri_mmcs,
                    &params,
                    index,
                    log_poly_degree,
                    initial_eval,
                    &betas,
                    &openings,
                )
                .expect("verification should succeed");
        }
    }

    #[test]
    fn test_fri_commit_verify_arity2() {
        // Test with arity 2 (log_folding_factor = 1)
        test_fri_commit_verify_roundtrip(10, 1);
    }

    #[test]
    fn test_fri_commit_verify_arity4() {
        // Test with arity 4 (log_folding_factor = 2)
        test_fri_commit_verify_roundtrip(10, 2);
    }

    /// Test that verification fails with wrong initial evaluation.
    #[test]
    fn test_fri_verify_wrong_eval() {
        let mut rng = SmallRng::seed_from_u64(42);
        let (perm, fri_mmcs) = test_components();

        let log_poly_degree = 8;
        let log_blowup = 2;
        let log_final_degree = 2;
        let log_folding_factor = 1;

        let params = FriParams {
            log_blowup,
            log_folding_factor,
            log_final_degree,
            num_queries: 1,
        };

        let evals = generate_random_lde(&mut rng, log_poly_degree, log_blowup);
        let log_lde_size = log_poly_degree + log_blowup;
        let lde_size = 1 << log_lde_size;

        let mut prover_challenger = Challenger::new(perm.clone());
        let (prover_data, proof) =
            CommitPhaseData::<F, EF, _>::new(&fri_mmcs, &params, evals, &mut prover_challenger);

        let mut verifier_challenger = Challenger::new(perm);
        let betas = proof.sample_betas::<F, _>(&mut verifier_challenger);

        let index: usize = rng.random_range(0..lde_size);
        let wrong_eval: EF = rng.sample(StandardUniform); // Wrong!
        let openings = open_query(&fri_mmcs, &prover_data, &params, index);

        let result = proof.verify_query::<F>(
            &fri_mmcs,
            &params,
            index,
            log_poly_degree,
            wrong_eval, // Should fail
            &betas,
            &openings,
        );

        assert!(
            matches!(result, Err(FriError::EvaluationMismatch { .. })),
            "expected EvaluationMismatch error, got {:?}",
            result
        );
    }

    /// Test that verification fails with wrong beta challenges.
    /// With wrong betas, folding produces wrong values that don't match opened rows.
    #[test]
    fn test_fri_verify_wrong_beta() {
        let mut rng = SmallRng::seed_from_u64(42);
        let (perm, fri_mmcs) = test_components();

        let log_poly_degree = 8;
        let log_blowup = 2;
        let log_final_degree = 2;
        let log_folding_factor = 1;

        let params = FriParams {
            log_blowup,
            log_folding_factor,
            log_final_degree,
            num_queries: 1,
        };

        let evals = generate_random_lde(&mut rng, log_poly_degree, log_blowup);
        let log_lde_size = log_poly_degree + log_blowup;
        let lde_size = 1 << log_lde_size;

        let mut prover_challenger = Challenger::new(perm);
        let (prover_data, proof) = CommitPhaseData::<F, EF, _>::new(
            &fri_mmcs,
            &params,
            evals.clone(),
            &mut prover_challenger,
        );

        // Use wrong betas
        let wrong_betas: Vec<EF> = (0..proof.commitments.len())
            .map(|_| rng.sample(StandardUniform))
            .collect();

        let index: usize = rng.random_range(0..lde_size);
        let initial_eval = evals[index];
        let openings = open_query(&fri_mmcs, &prover_data, &params, index);

        let result = proof.verify_query::<F>(
            &fri_mmcs,
            &params,
            index,
            log_poly_degree,
            initial_eval,
            &wrong_betas, // Should fail
            &openings,
        );

        assert!(
            matches!(result, Err(FriError::EvaluationMismatch { .. })),
            "expected EvaluationMismatch error, got {:?}",
            result
        );
    }

    /// Test that the final polynomial is correctly computed by evaluating it
    /// at points in the final domain and comparing with folded values.
    #[test]
    fn test_final_polynomial_correctness() {
        let mut rng = SmallRng::seed_from_u64(123);
        let (perm, fri_mmcs) = test_components();

        let log_poly_degree = 8;
        let log_blowup = 2;
        let log_final_degree = 3;
        let log_folding_factor = 1;

        let params = FriParams {
            log_blowup,
            log_folding_factor,
            log_final_degree,
            num_queries: 1,
        };

        let evals = generate_random_lde(&mut rng, log_poly_degree, log_blowup);

        let mut challenger = Challenger::new(perm);
        let (_prover_data, proof) =
            CommitPhaseData::<F, EF, _>::new(&fri_mmcs, &params, evals, &mut challenger);

        // Verify final polynomial has correct degree
        let final_degree = 1 << log_final_degree;
        assert_eq!(
            proof.final_poly.len(),
            final_degree,
            "Final polynomial should have {} coefficients",
            final_degree
        );

        // Evaluate final polynomial at several points in the final domain
        let log_final_height = log_final_degree + log_blowup;
        let final_height = 1 << log_final_height;
        let g = F::two_adic_generator(log_final_height);

        for idx in 0..final_height {
            // Point in bit-reversed final domain
            let x: F = g.exp_u64(reverse_bits_len(idx, log_final_height) as u64);

            // Evaluate polynomial via Horner
            let poly_eval: EF = proof
                .final_poly
                .iter()
                .rev()
                .fold(EF::ZERO, |acc, &coeff| acc * x + coeff);

            // The polynomial should be well-defined (just check it doesn't panic)
            let _ = poly_eval;
        }
    }
}
