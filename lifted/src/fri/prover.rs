use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{BatchOpening, Mmcs};
use p3_dft::{Radix2DFTSmallBatch, TwoAdicSubgroupDft};
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_util::{log2_strict_usize, reverse_slice_index_bits};

use crate::fri::FriParams;
use crate::fri::fold::{FriFold, FriFold2, FriFold4};
use crate::fri::verifier::CommitPhaseProof;

// ============================================================================
// Prover Data Structure
// ============================================================================

/// Prover data from the FRI commit phase, needed to answer queries.
pub struct CommitPhaseData<F: TwoAdicField, EF: ExtensionField<F>, FriMmcs: Mmcs<EF>> {
    /// Prover data for each folding round, used to open Merkle paths at query indices.
    pub folded_evals_data: Vec<FriMmcs::ProverData<RowMajorMatrix<EF>>>,
    _marker: PhantomData<F>,
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

impl<F: TwoAdicField, EF: ExtensionField<F>, FriMmcs: Mmcs<EF>> CommitPhaseData<F, EF, FriMmcs> {
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
    pub fn new<Challenger: FieldChallenger<F> + CanObserve<FriMmcs::Commitment>>(
        mmcs: &FriMmcs,
        params: &FriParams,
        mut evals: Vec<EF>,
        challenger: &mut Challenger,
    ) -> (Self, CommitPhaseProof<EF, FriMmcs>)
    where
        EF: ExtensionField<F>,
    {
        let log_arity = params.log_folding_factor;
        let arity = 1 << log_arity;

        let mut commitments = Vec::new();
        let mut folded_evals_data = Vec::new();

        let mut domain_size = evals.len();
        let log_domain_size = log2_strict_usize(domain_size); // needed for two_adic_generator
        let final_poly_degree = params.final_poly_degree(log_domain_size);
        let final_domain_size = final_poly_degree << params.log_blowup;

        // ─────────────────────────────────────────────────────────────────────────
        // Precompute s⁻¹ for all cosets
        // ─────────────────────────────────────────────────────────────────────────
        // Evaluations are in bit-reversed order: evals[i] = f(g^{bitrev(i)})
        // Row k contains [evals[k*arity], evals[k*arity+1], ...] which correspond
        // to evaluations at points forming a coset s·⟨ω⟩ where:
        //   - s = g^{bitrev(k*arity, log_domain_size)} = g^{bitrev(k, log_num_rows)}
        //     (because bitrev(k*arity, log_domain_size) = bitrev(k, log_num_rows)
        //      when arity = 2^log_arity)
        //   - ω is a primitive arity-th root of unity
        //
        // We compute s_inv for each row k, where s = g^{bitrev(k, log_num_rows)}
        // and g has order 2^log_domain_size.
        //
        // We generate sequential powers of g_inv and bit-reverse to get s_inv values
        // in the correct order for each row.
        let mut num_rows = domain_size >> log_arity;

        let g_inv = F::two_adic_generator(log_domain_size).inverse();
        let mut s_invs: Vec<F> = g_inv.powers().take(num_rows).collect();
        reverse_slice_index_bits(&mut s_invs);

        while domain_size > final_domain_size {
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
                1 => FriFold2::fold_matrix_packed(matrix_view.as_view(), &s_invs, beta),
                2 => FriFold4::fold_matrix_packed(matrix_view.as_view(), &s_invs, beta),
                _ => panic!("Unsupported folding arity"),
            };
            // No bit-reversal needed: folded evals maintain bit-reversed order
            // because s_invs are already bit-reversed to match

            commitments.push(commitment);
            folded_evals_data.push(prover_data);

            domain_size /= arity;

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
            // After domain_size update, new num_rows = domain_size / arity.
            num_rows = domain_size >> log_arity;
            s_invs = (0..num_rows)
                .into_par_iter()
                .map(|k| s_invs[k * arity].exp_power_of_2(log_arity))
                .collect();
        }

        // ─────────────────────────────────────────────────────────────────────────
        // Extract final polynomial coefficients
        // ─────────────────────────────────────────────────────────────────────────
        // The remaining evaluations are on a domain of size `final_domain_size`.
        // The polynomial degree is `final_poly_degree = final_domain_size / blowup`.
        // We need coefficients of degree < final_poly_degree, so we:
        // 1. Take the first `final_poly_degree` evaluations (others are redundant due to blowup)
        // 2. Convert from bit-reversed to standard order
        // 3. Apply inverse DFT to get coefficients
        evals.truncate(final_poly_degree);
        reverse_slice_index_bits(&mut evals);

        let final_poly = Radix2DFTSmallBatch::default().idft_algebra(evals);

        // Observe final polynomial coefficients for Fiat-Shamir
        for &coeff in &final_poly {
            challenger.observe_algebra_element(coeff);
        }

        (
            Self {
                folded_evals_data,
                _marker: PhantomData,
            },
            CommitPhaseProof {
                commitments,
                final_poly,
            },
        )
    }

    /// Open a specific query index across all FRI commit phase rounds.
    ///
    /// Returns Merkle openings for each folding round at the given query index.
    /// The index is progressively reduced by shifting off `log_arity` bits per round.
    pub fn open_query(
        &self,
        mmcs: &FriMmcs,
        params: &FriParams,
        index: usize,
    ) -> Vec<BatchOpening<EF, FriMmcs>> {
        let log_arity = params.log_folding_factor;
        let mut current_index = index;
        self.folded_evals_data
            .iter()
            .map(|prover_data| {
                let row_index = current_index >> log_arity;
                let opening = mmcs.open_batch(row_index, prover_data);
                current_index = row_index;
                opening
            })
            .collect()
    }
}
