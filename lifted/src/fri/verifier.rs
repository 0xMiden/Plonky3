use alloc::vec::Vec;
use core::iter::zip;
use core::{array, fmt};

use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{BatchOpening, Mmcs};
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::Dimensions;
use p3_util::reverse_bits_len;

use crate::fri::fold::{FriFold, FriFold2, FriFold4};
use crate::fri::{FriError, FriParams};

// ============================================================================
// Proof Data Structure
// ============================================================================

/// Proof data from the FRI commit phase.
///
/// Contains commitments to each folding round and the final low-degree polynomial.
pub struct CommitPhaseProof<EF: Field, FriMmcs: Mmcs<EF>> {
    /// Merkle commitments to folded evaluations at each round.
    pub(crate) commitments: Vec<FriMmcs::Commitment>,
    /// Coefficients of the final low-degree polynomial (sent in clear).
    pub(crate) final_poly: Vec<EF>,
}

// ============================================================================
// Verification
// ============================================================================
//
// FRI verification checks that a committed polynomial is close to low-degree.
//
// ## Domain Structure
//
// The prover commits to evaluations on domain D of size 2^log_domain_size in bit-reversed order.
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
    /// - `log_max_degree`: log₂ of the maximum polynomial degree (before blowup)
    /// - `eval`: Initial evaluation f(x) at the queried point
    /// - `betas`: Folding challenges β₀, β₁, ... from the verifier
    /// - `openings`: Merkle openings for each folding round
    ///
    /// ## Errors
    ///
    /// Returns `FriError` if any verification check fails.
    #[allow(clippy::too_many_arguments)]
    pub fn verify_query<F: TwoAdicField>(
        &self,
        mmcs: &FriMmcs,
        params: &FriParams,
        index: usize,
        log_max_degree: usize,
        eval: EF,
        betas: &[EF],
        openings: &[BatchOpening<EF, FriMmcs>],
    ) -> Result<(), FriError<FriMmcs::Error>>
    where
        EF: ExtensionField<F>,
        FriMmcs::Error: fmt::Debug,
    {
        let log_domain_size = log_max_degree + params.log_blowup;

        // Verify the proof has the expected structure
        let expected_num_rounds = params.num_rounds(log_domain_size);
        let expected_final_poly_len = params.final_poly_degree(log_domain_size);

        if self.commitments.len() != expected_num_rounds
            || betas.len() != expected_num_rounds
            || openings.len() != expected_num_rounds
            || self.final_poly.len() != expected_final_poly_len
        {
            return Err(FriError::InvalidProofStructure);
        }

        // Phase 1: Verify all Merkle openings
        self.verify_openings(mmcs, params, index, log_max_degree, openings)?;

        // Phase 2: Process opened rows and verify folding
        let rows = openings.iter().map(|o| o.opened_values[0].as_slice());
        self.verify_folding::<F, FriMmcs::Error>(params, index, log_max_degree, eval, betas, rows)?;

        Ok(())
    }

    /// Phase 1: Verify all Merkle openings without processing contents.
    ///
    /// Checks that each opening is valid against its commitment.
    fn verify_openings(
        &self,
        mmcs: &FriMmcs,
        params: &FriParams,
        mut index: usize,
        log_max_degree: usize,
        openings: &[BatchOpening<EF, FriMmcs>],
    ) -> Result<(), FriError<FriMmcs::Error>>
    where
        FriMmcs::Error: fmt::Debug,
    {
        let log_arity = params.log_folding_factor;
        let arity = 1 << log_arity;
        let mut num_rows = 1 << (log_max_degree + params.log_blowup).saturating_sub(log_arity);

        for (round, (commit, opening)) in zip(&self.commitments, openings).enumerate() {
            let row_index = index >> log_arity;

            mmcs.verify_batch(
                commit,
                &[Dimensions {
                    width: arity,
                    height: num_rows,
                }],
                row_index,
                opening.into(),
            )
            .map_err(|e| FriError::MmcsError(e, round))?;

            index = row_index;
            num_rows >>= log_arity;
        }
        Ok(())
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
    /// - `log_max_degree`: log₂ of the maximum polynomial degree (before blowup)
    /// - `eval`: Initial evaluation f(x) at the queried point
    /// - `betas`: Folding challenges β₀, β₁, ...
    /// - `rows`: Iterator yielding opened rows as `&[EF]` slices
    fn verify_folding<'a, F: TwoAdicField, MmcsError: fmt::Debug>(
        &self,
        params: &FriParams,
        mut index: usize,
        log_max_degree: usize,
        mut eval: EF,
        betas: &[EF],
        rows: impl Iterator<Item = &'a [EF]>,
    ) -> Result<(), FriError<MmcsError>>
    where
        EF: ExtensionField<F> + 'a,
    {
        let log_arity = params.log_folding_factor;
        let mut log_domain_size = log_max_degree + params.log_blowup;

        // Precompute g_inv once; we'll update it each round by raising to power arity
        let mut g_inv = F::two_adic_generator(log_domain_size).inverse();

        for (beta, row) in zip(betas, rows) {
            // index = (row_index × arity) + position_in_coset
            let position_in_coset = index & ((1 << log_arity) - 1);
            let row_index = index >> log_arity;

            // ─────────────────────────────────────────────────────────────────
            // Consistency check
            // ─────────────────────────────────────────────────────────────────
            // The evaluation we're carrying forward must match the opened value
            // at the corresponding position within the coset.
            if row[position_in_coset] != eval {
                return Err(FriError::EvaluationMismatch {
                    row_index,
                    position: position_in_coset,
                });
            }

            // ─────────────────────────────────────────────────────────────────
            // Compute coset generator inverse s⁻¹
            // ─────────────────────────────────────────────────────────────────
            // Row `row_index` contains coset s·⟨ω⟩ where:
            //   - ω is a primitive `arity`-th root of unity
            //   - s = g^{bitrev(row_index, log_num_rows)} with g having order 2^log_domain_size
            //
            // So s_inv = g_inv^{bitrev(row_index, log_num_rows)}
            let s_inv = {
                let log_num_rows = log_domain_size - log_arity;
                g_inv.exp_u64(reverse_bits_len(row_index, log_num_rows) as u64)
            };

            // ─────────────────────────────────────────────────────────────────
            // Fold: interpolate f on coset and evaluate at β
            // ─────────────────────────────────────────────────────────────────
            // Given evaluations [f(s), f(−s), ...] (bit-reversed), compute f(β).
            eval = match log_arity {
                1 => FriFold2::fold_evals::<F, EF, EF>(array::from_fn(|i| row[i]), s_inv, *beta),
                2 => FriFold4::fold_evals::<F, EF, EF>(array::from_fn(|i| row[i]), s_inv, *beta),
                _ => unreachable!("Unsupported folding arity"),
            };

            // Update for next round:
            // - index becomes row_index
            // - log_domain_size shrinks by log_arity
            // - g_inv needs order 2^(log_domain_size - log_arity) = g_inv^arity
            index = row_index;
            log_domain_size -= log_arity;
            g_inv = g_inv.exp_power_of_2(log_arity);
        }

        // ─────────────────────────────────────────────────────────────────────
        // Final polynomial check
        // ─────────────────────────────────────────────────────────────────────
        // After all folds, verify that `eval` equals p(x) where:
        //   - p is the final low-degree polynomial
        //   - x is the evaluation point corresponding to `index` in the FINAL domain
        //
        // The final domain has size 2^log_domain_size. The index refers to position
        // in bit-reversed storage, so the actual point is:
        //   x = g_final^{bitrev(index, log_domain_size)}
        // where g_final generates the final domain.
        let x_power = reverse_bits_len(index, log_domain_size) as u64;
        let x = F::two_adic_generator(log_domain_size).exp_u64(x_power);

        // Evaluate final polynomial via Horner's method: p(x) = Σᵢ cᵢ·xⁱ
        let final_eval = self
            .final_poly
            .iter()
            .rev()
            .fold(EF::ZERO, |acc, &coeff| acc * x + coeff);

        if final_eval != eval {
            return Err(FriError::FinalPolyMismatch);
        }

        Ok(())
    }

    /// Absorb the proof into a challenger and return the folding challenges.
    ///
    /// This method observes each commitment and samples a beta challenge after each,
    /// then observes the final polynomial coefficients. The returned betas are needed
    /// for query verification.
    ///
    /// ## Arguments
    ///
    /// - `challenger`: The Fiat-Shamir challenger (must be in the same state as the prover's
    ///   challenger after the initial commitment phase)
    ///
    /// ## Returns
    ///
    /// A vector of folding challenges β₀, β₁, ... (one per commitment)
    pub fn sample_betas<F, Challenger>(&self, challenger: &mut Challenger) -> Vec<EF>
    where
        F: Field,
        EF: ExtensionField<F>,
        Challenger: FieldChallenger<F> + CanObserve<FriMmcs::Commitment>,
    {
        let betas: Vec<EF> = self
            .commitments
            .iter()
            .map(|commit| {
                challenger.observe(commit.clone());
                challenger.sample_algebra_element()
            })
            .collect();

        // Observe final polynomial coefficients
        for &coeff in &self.final_poly {
            challenger.observe_algebra_element(coeff);
        }

        betas
    }
}
