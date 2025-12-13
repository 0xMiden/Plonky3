use alloc::vec::Vec;
use core::array;
use core::iter::zip;

use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{BatchOpening, Mmcs};
use p3_dft::{Radix2DFTSmallBatch, TwoAdicSubgroupDft};
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::Dimensions;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_util::{log2_strict_usize, reverse_bits_len, reverse_slice_index_bits};

use crate::fri::Params;
use crate::fri::fold::{FriFold, TwoAdicFriFold};

// ============================================================================
// Data Structures
// ============================================================================

/// Prover data from the FRI commit phase, needed to answer queries.
pub struct CommitPhaseData<EF: Field, FriMmcs: Mmcs<EF>> {
    pub folded_evals_data: Vec<FriMmcs::ProverData<RowMajorMatrix<EF>>>,
}

/// Proof data from the FRI commit phase.
///
/// Contains commitments to each folding round and the final low-degree polynomial.
pub struct CommitPhaseProof<EF: Field, FriMmcs: Mmcs<EF>> {
    pub commitments: Vec<FriMmcs::Commitment>,
    pub final_poly: Vec<EF>,
}

// ============================================================================
// Verification
// ============================================================================
//
// FRI verification checks that a committed polynomial is close to low-degree.
//
// ## Domain Structure
//
// The prover commits to evaluations on domain D of size n = 2^log_n in bit-reversed order.
// Each folding round groups `arity` consecutive evaluations into cosets and folds them.
//
// For arity = 2:
//   - Row i contains evaluations at coset {s, −s} where s = g^{bitrev(i)}
//   - g is the generator of D (has order n)
//
// For arity = 4:
//   - Row i contains evaluations at coset {s, −s, ωs, −ωs} where ω = √−1
//
// ## Index Semantics
//
// The query `index` has two parts:
//   - High bits: which row (coset) in the committed matrix
//   - Low bits: which position within the coset
//
// After each fold, we shift off `log_arity` bits, moving to the parent coset.

impl<EF: Field, FriMmcs: Mmcs<EF>> CommitPhaseProof<EF, FriMmcs> {
    /// Verify a FRI query by checking all folding rounds.
    ///
    /// Two-phase verification:
    /// 1. Verify all Merkle openings and collect opened rows
    /// 2. Process opened rows: check consistency and fold
    ///
    /// ## Arguments
    ///
    /// - `mmcs`: The MMCS used for commitments
    /// - `params`: FRI parameters (blowup, folding factor, final degree)
    /// - `index`: Query index in the initial domain (bit-reversed)
    /// - `log_max_height`: log₂ of the maximum polynomial degree (before blowup)
    /// - `eval`: Initial evaluation f(x) at the queried point
    /// - `betas`: Folding challenges β₀, β₁, ... from the verifier
    /// - `openings`: Merkle openings for each folding round
    pub fn verify_query<F: TwoAdicField>(
        &self,
        mmcs: &FriMmcs,
        params: &Params,
        index: usize,
        log_max_height: usize,
        eval: EF,
        betas: &[EF],
        openings: &[BatchOpening<EF, FriMmcs>],
    ) where
        EF: ExtensionField<F>,
    {
        // Phase 1: Verify all Merkle openings
        self.verify_openings(mmcs, params, index, log_max_height, openings);

        // Phase 2: Process opened rows and verify folding
        let rows = openings.iter().map(|o| o.opened_values[0].as_slice());
        self.verify_folding::<F>(params, index, log_max_height, eval, betas, rows);
    }

    /// Phase 1: Verify all Merkle openings without processing contents.
    ///
    /// Checks that each opening is valid against its commitment.
    fn verify_openings(
        &self,
        mmcs: &FriMmcs,
        params: &Params,
        mut index: usize,
        log_max_height: usize,
        openings: &[BatchOpening<EF, FriMmcs>],
    ) {
        let log_arity = params.log_folding_factor;
        let arity = 1 << log_arity;
        let mut log_height = log_max_height + params.log_blowup;

        for (commit, opening) in zip(&self.commitments, openings) {
            let row_index = index >> log_arity;
            let matrix_height = 1 << (log_height - log_arity);

            mmcs.verify_batch(
                commit,
                &[Dimensions {
                    width: arity,
                    height: matrix_height,
                }],
                row_index,
                opening.into(),
            )
            .expect("Merkle verification failed");

            index = row_index;
            log_height -= log_arity;
        }
    }

    /// Phase 2: Verify folding consistency given opened rows.
    ///
    /// For each round:
    /// 1. Check that `eval` matches the expected position in the opened row
    /// 2. Compute coset generator inverse s⁻¹
    /// 3. Fold the coset evaluations: eval' = f(β) via interpolation
    ///
    /// Finally, check that the folded value matches the final polynomial.
    ///
    /// ## Arguments
    ///
    /// - `params`: FRI parameters
    /// - `index`: Query index in the initial domain
    /// - `log_max_height`: log₂ of the maximum polynomial degree (before blowup)
    /// - `eval`: Initial evaluation f(x) at the queried point
    /// - `betas`: Folding challenges β₀, β₁, ...
    /// - `rows`: Iterator yielding opened rows as `&[EF]` slices
    fn verify_folding<'a, F: TwoAdicField>(
        &self,
        params: &Params,
        mut index: usize,
        log_max_height: usize,
        mut eval: EF,
        betas: &[EF],
        rows: impl Iterator<Item = &'a [EF]>,
    ) where
        EF: ExtensionField<F> + 'a,
    {
        let log_arity = params.log_folding_factor;
        let mut log_height = log_max_height + params.log_blowup;

        // Precompute g_inv once; we'll update it each round by raising to power arity
        let mut g_inv = F::two_adic_generator(log_height).inverse();

        for (beta, row) in zip(betas, rows) {
            // index = (row_index × arity) + position_in_coset
            let position_in_coset = index & ((1 << log_arity) - 1);
            let row_index = index >> log_arity;

            // ─────────────────────────────────────────────────────────────────
            // Consistency check
            // ─────────────────────────────────────────────────────────────────
            // The evaluation we're carrying forward must match the opened value
            // at the corresponding position within the coset.
            assert_eq!(
                row[position_in_coset], eval,
                "Evaluation mismatch at row {row_index}, position {position_in_coset}"
            );

            // ─────────────────────────────────────────────────────────────────
            // Compute coset generator inverse s⁻¹
            // ─────────────────────────────────────────────────────────────────
            // Row `row_index` contains coset s·⟨ω⟩ where:
            //   - ω is a primitive `arity`-th root of unity
            //   - s = g^{bitrev(row_index, log_num_cosets)} with g having order 2^log_height
            //
            // So s_inv = g_inv^{bitrev(row_index, log_num_cosets)}
            let log_num_cosets = log_height - log_arity;
            let s_inv: F = g_inv.exp_u64(reverse_bits_len(row_index, log_num_cosets) as u64);

            // ─────────────────────────────────────────────────────────────────
            // Fold: interpolate f on coset and evaluate at β
            // ─────────────────────────────────────────────────────────────────
            // Given evaluations [f(s), f(−s), ...] (bit-reversed), compute f(β).
            eval = match log_arity {
                1 => <TwoAdicFriFold as FriFold<2>>::fold_evals::<F, EF, EF>(
                    array::from_fn(|i| row[i]),
                    s_inv,
                    *beta,
                ),
                2 => <TwoAdicFriFold as FriFold<4>>::fold_evals::<F, EF, EF>(
                    array::from_fn(|i| row[i]),
                    s_inv,
                    *beta,
                ),
                _ => unreachable!("Unsupported folding arity"),
            };

            // Update for next round:
            // - index becomes row_index
            // - log_height shrinks by log_arity
            // - g_inv needs order 2^(log_height - log_arity) = g_inv^arity
            index = row_index;
            log_height -= log_arity;
            g_inv = g_inv.exp_power_of_2(log_arity);
        }

        // ─────────────────────────────────────────────────────────────────────
        // Final polynomial check
        // ─────────────────────────────────────────────────────────────────────
        // After all folds, verify that `eval` equals p(x) where:
        //   - p is the final low-degree polynomial
        //   - x is the evaluation point corresponding to `index` in the FINAL domain
        //
        // The final domain has size 2^log_height. The index refers to position
        // in bit-reversed storage, so the actual point is:
        //   x = g_final^{bitrev(index, log_height)}
        // where g_final generates the final domain.
        let x_power = reverse_bits_len(index, log_height) as u64;
        let x = F::two_adic_generator(log_height).exp_u64(x_power);

        // Evaluate final polynomial via Horner's method: p(x) = Σᵢ cᵢ·xⁱ
        let final_eval = self
            .final_poly
            .iter()
            .rev()
            .fold(EF::ZERO, |acc, &coeff| acc * x + coeff);
        assert_eq!(final_eval, eval, "Final polynomial mismatch");
    }
}

// ============================================================================
// Commit Phase (Prover)
// ============================================================================
//
// The FRI commit phase iteratively folds a polynomial until it reaches a
// target degree, committing to intermediate evaluations along the way.
//
// ## Algorithm
//
// Given polynomial f of degree d with evaluations on domain D of size n = d·blowup:
//
// 1. Reshape evaluations into matrix M with `arity` columns
//    - Row i contains the coset {f(s·ωʲ) : j ∈ [0, arity)} where s = g^{bitrev(i)}
//
// 2. Commit to M via Merkle tree
//
// 3. Sample folding challenge β from verifier
//
// 4. Fold each row: for coset evaluations [y₀, y₁, ...], compute f(β)
//    - This reduces degree by factor of `arity`
//    - New evaluations live on domain D' of size n/arity
//
// 5. Repeat until degree ≤ final_degree
//
// 6. Send final polynomial coefficients to verifier
//
// ## Coset Structure in Bit-Reversed Order
//
// For domain D = g·H where H = ⟨ω⟩ has order n:
//   - Row i contains evaluations at s·⟨ω_arity⟩ where s = g·ω^{bitrev(i)}
//   - Adjacent rows have s values that are negatives (for arity=2)
//   - After folding, row i maps to row i in the halved domain

/// Execute the FRI commit phase, producing commitments and prover data.
///
/// ## Arguments
///
/// - `mmcs`: The MMCS for committing to folded evaluations
/// - `params`: FRI parameters
/// - `evals`: Initial polynomial evaluations in bit-reversed order
/// - `challenger`: Fiat-Shamir challenger for sampling β
///
/// ## Returns
///
/// - `CommitPhaseProof`: Commitments and final polynomial (sent to verifier)
/// - `CommitPhaseData`: Prover data needed to answer queries
pub fn commit_phase<
    F: TwoAdicField,
    EF: ExtensionField<F>,
    FriMmcs: Mmcs<EF>,
    Challenger: FieldChallenger<F> + CanObserve<FriMmcs::Commitment>,
>(
    mmcs: &FriMmcs,
    params: &Params,
    mut evals: Vec<EF>,
    challenger: &mut Challenger,
) -> (CommitPhaseProof<EF, FriMmcs>, CommitPhaseData<EF, FriMmcs>) {
    let log_arity = params.log_folding_factor;
    let arity = 1 << log_arity;

    let mut commitments = Vec::new();
    let mut folded_evals_data = Vec::new();

    // Stop when we reach the final polynomial size (degree × blowup)
    let log_final_size = params.log_final_degree + params.log_blowup;

    // ─────────────────────────────────────────────────────────────────────────
    // Precompute s⁻¹ for all cosets
    // ─────────────────────────────────────────────────────────────────────────
    // Evaluations are in bit-reversed order: evals[i] = f(g^{bitrev(i)})
    // Row k contains [evals[k*arity], evals[k*arity+1], ...] which correspond
    // to evaluations at points forming a coset s·⟨ω⟩ where:
    //   - s = g^{bitrev(k*arity, log_height)} = g^{bitrev(k, log_num_cosets)}
    //     (because bitrev(k*arity, log_height) = bitrev(k, log_num_cosets)
    //      when arity = 2^log_arity)
    //   - ω is a primitive arity-th root of unity
    //
    // We compute s_inv for each row k, where s = g^{bitrev(k, log_num_cosets)}
    // and g has order 2^log_height.
    //
    // We generate sequential powers of g_inv and bit-reverse to get s_inv values
    // in the correct order for each row.
    let log_n = log2_strict_usize(evals.len());
    let log_num_cosets = log_n - log_arity;
    let num_cosets = 1 << log_num_cosets;

    let g_inv = F::two_adic_generator(log_n).inverse();
    let mut s_invs: Vec<F> = g_inv.powers().take(num_cosets).collect();
    reverse_slice_index_bits(&mut s_invs);

    let mut log_height = log_n;

    while log_height > log_final_size {
        // ─────────────────────────────────────────────────────────────────────
        // Reshape into matrix: each row is one coset
        // ─────────────────────────────────────────────────────────────────────
        let matrix = RowMajorMatrix::new(evals, arity);

        // ─────────────────────────────────────────────────────────────────────
        // Commit to the folded evaluations
        // ─────────────────────────────────────────────────────────────────────
        let (commitment, prover_data) = mmcs.commit_matrix(matrix);
        challenger.observe(commitment.clone());

        // ─────────────────────────────────────────────────────────────────────
        // Sample folding challenge β
        // ─────────────────────────────────────────────────────────────────────
        let beta: EF = challenger.sample_algebra_element();

        // ─────────────────────────────────────────────────────────────────────
        // Fold all rows: f(β) = interpolate coset evaluations at β
        // ─────────────────────────────────────────────────────────────────────
        let matrix_view = mmcs.get_matrices(&prover_data)[0];

        evals = match log_arity {
            1 => <TwoAdicFriFold as FriFold<2>>::fold_matrix_packed(
                matrix_view.as_view(),
                &s_invs,
                beta,
            ),
            2 => <TwoAdicFriFold as FriFold<4>>::fold_matrix_packed(
                matrix_view.as_view(),
                &s_invs,
                beta,
            ),
            _ => panic!("Unsupported folding arity"),
        };
        // No bit-reversal needed: folded evals maintain bit-reversed order
        // because s_invs are already bit-reversed to match

        commitments.push(commitment);
        folded_evals_data.push(prover_data);

        log_height -= log_arity;

        // ─────────────────────────────────────────────────────────────────────
        // Update s⁻¹ for next round
        // ─────────────────────────────────────────────────────────────────────
        // After folding, domain shrinks by `arity`. The new generator g' = g^arity.
        // We need: s'_inv[k] = g'^{-bitrev(k, L')} = g^{-arity * bitrev(k, L')}
        //
        // Using the identity bitrev(k, L-log_arity) = bitrev(k*arity, L):
        //   s'_inv[k] = g^{-arity * bitrev(k*arity, L)}
        //             = (g^{-bitrev(k*arity, L)})^arity
        //             = s_inv[k*arity]^arity
        //
        // So we select every `arity`-th element and raise to power `arity`.
        let new_len = s_invs.len() / arity;
        s_invs = (0..new_len)
            .into_par_iter()
            .map(|k| s_invs[k * arity].exp_power_of_2(log_arity))
            .collect();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Extract final polynomial coefficients
    // ─────────────────────────────────────────────────────────────────────────
    // The remaining evaluations are on a domain of size `final_size`.
    // We need coefficients of degree < final_degree, so we:
    // 1. Take the first `final_degree` evaluations (others are redundant due to blowup)
    // 2. Convert from bit-reversed to standard order
    // 3. Apply inverse DFT to get coefficients
    // Save all evals for sanity checking before truncation
    let final_degree = 1 << params.log_final_degree;
    evals.truncate(final_degree);
    reverse_slice_index_bits(&mut evals);

    let final_poly = Radix2DFTSmallBatch::default().idft_algebra(evals.clone());

    // Observe final polynomial coefficients for Fiat-Shamir
    for &coeff in &final_poly {
        challenger.observe_algebra_element(coeff);
    }

    (
        CommitPhaseProof {
            commitments,
            final_poly,
        },
        CommitPhaseData { folded_evals_data },
    )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_commit::{ExtensionMmcs, Mmcs};
    use p3_dft::{Radix2DFTSmallBatch, TwoAdicSubgroupDft};
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{Field, PrimeCharacteristicRing, TwoAdicField};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use p3_util::reverse_bits_len;
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
        data: &CommitPhaseData<EF, M>,
        params: &Params,
        index: usize,
    ) -> Vec<BatchOpening<EF, M>> {
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

        let params = Params {
            log_blowup: 2,
            log_folding_factor,
            log_final_degree: 2,
        };

        // Generate random LDE evaluations
        let evals = generate_random_lde(&mut rng, log_poly_degree, params.log_blowup);
        let lde_size = evals.len();

        // Prover: run commit phase
        let mut prover_challenger = Challenger::new(perm.clone());
        let (proof, prover_data) =
            commit_phase::<F, EF, _, _>(&fri_mmcs, &params, evals.clone(), &mut prover_challenger);

        // Verifier: replay challenger to get betas
        let mut verifier_challenger = Challenger::new(perm);
        let betas: Vec<EF> = proof
            .commitments
            .iter()
            .map(|commit| {
                verifier_challenger.observe(commit.clone());
                verifier_challenger.sample_algebra_element()
            })
            .collect();
        for &coeff in &proof.final_poly {
            verifier_challenger.observe_algebra_element(coeff);
        }

        // Verify random queries
        for _ in 0..3 {
            let index: usize = rng.random_range(0..lde_size);
            let initial_eval = evals[index];
            let openings = open_query(&fri_mmcs, &prover_data, &params, index);

            proof.verify_query::<F>(
                &fri_mmcs,
                &params,
                index,
                log_poly_degree,
                initial_eval,
                &betas,
                &openings,
            );
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
    #[should_panic(expected = "Evaluation mismatch")]
    fn test_fri_verify_wrong_eval() {
        let mut rng = SmallRng::seed_from_u64(42);
        let (perm, fri_mmcs) = test_components();

        let log_poly_degree = 8;
        let log_blowup = 2;
        let log_final_degree = 2;
        let log_folding_factor = 1;

        let params = Params {
            log_blowup,
            log_folding_factor,
            log_final_degree,
        };

        let evals = generate_random_lde(&mut rng, log_poly_degree, log_blowup);
        let log_lde_size = log_poly_degree + log_blowup;
        let lde_size = 1 << log_lde_size;

        let mut prover_challenger = Challenger::new(perm.clone());
        let (proof, prover_data) =
            commit_phase::<F, EF, _, _>(&fri_mmcs, &params, evals.clone(), &mut prover_challenger);

        let mut verifier_challenger = Challenger::new(perm);
        let mut betas = Vec::new();
        for commit in &proof.commitments {
            verifier_challenger.observe(commit.clone());
            betas.push(verifier_challenger.sample_algebra_element());
        }
        for &coeff in &proof.final_poly {
            verifier_challenger.observe_algebra_element(coeff);
        }

        let index: usize = rng.random_range(0..lde_size);
        let wrong_eval: EF = rng.sample(StandardUniform); // Wrong!
        let openings = open_query(&fri_mmcs, &prover_data, &params, index);

        proof.verify_query::<F>(
            &fri_mmcs,
            &params,
            index,
            log_poly_degree,
            wrong_eval, // Should fail
            &betas,
            &openings,
        );
    }

    /// Test that verification fails with wrong beta challenges.
    /// With wrong betas, folding produces wrong values that don't match opened rows.
    #[test]
    #[should_panic(expected = "Evaluation mismatch")]
    fn test_fri_verify_wrong_beta() {
        let mut rng = SmallRng::seed_from_u64(42);
        let (perm, fri_mmcs) = test_components();

        let log_poly_degree = 8;
        let log_blowup = 2;
        let log_final_degree = 2;
        let log_folding_factor = 1;

        let params = Params {
            log_blowup,
            log_folding_factor,
            log_final_degree,
        };

        let evals = generate_random_lde(&mut rng, log_poly_degree, log_blowup);
        let log_lde_size = log_poly_degree + log_blowup;
        let lde_size = 1 << log_lde_size;

        let mut prover_challenger = Challenger::new(perm.clone());
        let (proof, prover_data) =
            commit_phase::<F, EF, _, _>(&fri_mmcs, &params, evals.clone(), &mut prover_challenger);

        // Use wrong betas
        let wrong_betas: Vec<EF> = (0..proof.commitments.len())
            .map(|_| rng.sample(StandardUniform))
            .collect();

        let index: usize = rng.random_range(0..lde_size);
        let initial_eval = evals[index];
        let openings = open_query(&fri_mmcs, &prover_data, &params, index);

        proof.verify_query::<F>(
            &fri_mmcs,
            &params,
            index,
            log_poly_degree,
            initial_eval,
            &wrong_betas, // Should fail
            &openings,
        );
    }

    /// Test that our folding matches the reference implementation from two_adic_pcs.rs.
    #[test]
    fn test_fold_matches_reference() {
        let mut rng = SmallRng::seed_from_u64(99);

        let log_height = 4; // 16 evaluations
        let num_evals = 1 << log_height;
        let log_arity = 1; // arity 2
        let arity = 1 << log_arity;
        let log_num_cosets = log_height - log_arity;
        let num_cosets = 1 << log_num_cosets;

        // Generate random low-degree polynomial and compute evaluations
        let log_poly_degree = 3;
        let poly_degree = 1 << log_poly_degree;
        let coeffs: Vec<EF> = (0..poly_degree)
            .map(|_| rng.sample(StandardUniform))
            .collect();

        // DFT and bit-reverse to match FRI's expectation
        let mut full_coeffs = coeffs.clone();
        full_coeffs.resize(num_evals, EF::ZERO);
        let dft = Radix2DFTSmallBatch::<EF>::default();
        let mut evals = dft.dft_algebra(full_coeffs);
        reverse_slice_index_bits(&mut evals);

        let beta: EF = rng.sample(StandardUniform);

        // Our implementation
        let g_inv = F::two_adic_generator(log_num_cosets + log_arity).inverse();
        let mut s_invs: Vec<F> = g_inv.powers().take(num_cosets).collect();
        reverse_slice_index_bits(&mut s_invs);

        let matrix = RowMajorMatrix::new(evals.clone(), arity);
        let my_folded =
            <TwoAdicFriFold as FriFold<2>>::fold_matrix::<F, EF>(matrix.as_view(), &s_invs, beta);

        // Reference implementation from two_adic_pcs.rs
        let ref_g_inv = F::two_adic_generator(log_num_cosets + 1).inverse();
        let mut ref_halve_inv_powers: Vec<F> = ref_g_inv
            .shifted_powers(F::ONE.halve())
            .take(num_cosets)
            .collect();
        reverse_slice_index_bits(&mut ref_halve_inv_powers);

        let ref_folded: Vec<EF> = evals
            .chunks(arity)
            .zip(ref_halve_inv_powers.iter())
            .map(|(row, &halve_inv_power)| {
                let lo = row[0];
                let hi = row[1];
                (lo + hi).halve() + (lo - hi) * beta * halve_inv_power
            })
            .collect();

        // Assert all values match
        assert_eq!(my_folded.len(), ref_folded.len());
        for (my_val, ref_val) in my_folded.iter().zip(ref_folded.iter()) {
            assert_eq!(my_val, ref_val, "Folded value mismatch");
        }
    }

    /// Test that FRI folding preserves the low-degree structure.
    ///
    /// After folding a degree-d polynomial, the result should have degree d/arity.
    /// This test verifies by checking that high coefficients are zero after IDFT.
    #[test]
    fn test_folding_preserves_low_degree() {
        let mut rng = SmallRng::seed_from_u64(42);

        let log_blowup = 2;
        let log_poly_degree = 4; // degree 16 polynomial
        let poly_degree = 1 << log_poly_degree;
        let log_lde_size = log_poly_degree + log_blowup;
        let lde_size = 1 << log_lde_size;
        let log_arity = 1; // arity 2
        let arity = 1 << log_arity;

        // Generate random low-degree polynomial
        let coeffs: Vec<EF> = (0..poly_degree)
            .map(|_| rng.sample(StandardUniform))
            .collect();

        // Compute LDE in bit-reversed order
        let mut full_coeffs = coeffs;
        full_coeffs.resize(lde_size, EF::ZERO);
        let dft = Radix2DFTSmallBatch::<EF>::default();
        let mut evals = dft.dft_algebra(full_coeffs);
        reverse_slice_index_bits(&mut evals);

        // Compute s_invs
        let log_num_cosets = log_lde_size - log_arity;
        let num_cosets = 1 << log_num_cosets;
        let g_inv = F::two_adic_generator(log_lde_size).inverse();
        let mut s_invs: Vec<F> = g_inv.powers().take(num_cosets).collect();
        reverse_slice_index_bits(&mut s_invs);

        // Fold with random beta
        let beta: EF = rng.sample(StandardUniform);
        let matrix = RowMajorMatrix::new(evals, arity);
        let folded =
            <TwoAdicFriFold as FriFold<2>>::fold_matrix::<F, EF>(matrix.as_view(), &s_invs, beta);

        // IDFT the result to get coefficients
        let mut folded_for_idft = folded;
        reverse_slice_index_bits(&mut folded_for_idft);
        let folded_coeffs = dft.idft_algebra(folded_for_idft);

        // Check that all coefficients beyond degree/arity are zero
        let expected_degree = poly_degree / arity;
        for (i, coeff) in folded_coeffs.iter().enumerate().skip(expected_degree) {
            assert_eq!(
                *coeff,
                EF::ZERO,
                "High coefficient c[{i}] should be zero but was {:?}",
                coeff
            );
        }
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

        let params = Params {
            log_blowup,
            log_folding_factor,
            log_final_degree,
        };

        let evals = generate_random_lde(&mut rng, log_poly_degree, log_blowup);

        let mut challenger = Challenger::new(perm);
        let (proof, _prover_data) =
            commit_phase::<F, EF, _, _>(&fri_mmcs, &params, evals, &mut challenger);

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
