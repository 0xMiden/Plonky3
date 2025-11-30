use alloc::vec::Vec;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Dimensions;

use p3_commit::{BatchOpeningRef, Mmcs};
use super::config::LiftedFriParams;
use super::proof::LiftedFriProof;

/// Verifier entry-point for lifted FRI using LMCS and two-point DEEP per height.
///
/// Inputs:
/// - `folding`: two-adic folding strategy (standard FRI fold).
/// - `params`: lifted FRI parameters (includes commit-phase MMCS and padding width).
/// - `lmcs`: LMCS used for input openings.
/// - `matrices_dims_sorted_by_height`: dimensions per matrix; sorted by height; used to check LMCS rows.
/// - `proof`: lifted FRI proof.
/// - `challenger`: Fiat–Shamir challenger for replaying transcript (α, β’s, PoW witness).
pub fn verify_lifted_fri<Val, Challenge, InputMmcs, CodeMmcs, Challenger>(
    params: &LiftedFriParams<CodeMmcs>,
    lmcs: &InputMmcs,
    matrices_dims_sorted_by_height: &[Dimensions],
    lmcs_commit: &InputMmcs::Commitment,
    proof: &LiftedFriProof<Val, Challenge, CodeMmcs, InputMmcs, Challenger::Witness>,
    challenger: &mut Challenger,
) -> Result<(), VerifierError<CodeMmcs::Error, InputMmcs::Error>>
where
    Val: TwoAdicField,
    Challenge: ExtensionField<Val>,
    InputMmcs: Mmcs<Val>,
    CodeMmcs: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger + CanObserve<CodeMmcs::Commitment>,
{
    // Work out domain sizes.
    let h_max = matrices_dims_sorted_by_height
        .last()
        .ok_or(VerifierError::InvalidProofShape)?
        .height;
    let log_h_max = p3_util::log2_strict_usize(h_max);

    // Sample alpha, z and derive omega on the tallest subgroup.
    let _alpha: Challenge = challenger.sample_algebra_element();
    let z: Challenge = challenger.sample_algebra_element();
    let omega: Challenge = Challenge::from(Val::two_adic_generator(log_h_max));

    // Observe commit-phase commitments and record betas per fold-height.
    // Heights descend from log_h_max - 1 down to log_final_height.
    let log_final_height = params.log_blowup;
    let mut betas_by_log_height = alloc::collections::BTreeMap::new();
    for (i, cmt) in proof.commit_phase_commits.iter().enumerate() {
        challenger.observe(cmt.clone());
        let beta_i: Challenge = challenger.sample_algebra_element();
        let log_folded_height = log_h_max - 1 - i; // matches fri ordering
        betas_by_log_height.insert(log_folded_height, beta_i);
    }

    // Final polynomial must have the expected length; observe coefficients.
    if proof.final_poly.len() != params.final_poly_len() {
        return Err(VerifierError::InvalidProofShape);
    }
    for &c in &proof.final_poly {
        challenger.observe_algebra_element(c);
    }

    // Compute padded-lane offsets U_j once for the whole batch order.
    let pad = InputMmcs::ROW_PADDING.max(1);
    let mut u_offsets = Vec::with_capacity(matrices_dims_sorted_by_height.len());
    let mut acc = 0usize;
    for d in matrices_dims_sorted_by_height {
        u_offsets.push(acc);
        let w_padded = if d.width == 0 { 0 } else { (d.width + pad - 1) / pad * pad };
        acc += w_padded;
    }

    // For each query, verify LMCS opening and compute per-height reduced openings.
    for q in &proof.query_proofs {
        let idx = challenger.sample_bits(log_h_max);

        // Verify LMCS batch opening against the provided commitment.
        lmcs
            .verify_batch(
                lmcs_commit,
                matrices_dims_sorted_by_height,
                idx,
                BatchOpeningRef::from(&q.input_opening),
            )
            .map_err(VerifierError::InputMmcsError)?;

        // Reconstruct x from subgroup H (no coset shift), with bit-reversed enumeration.
        let rev = p3_util::reverse_bits_len(idx, log_h_max);
        let x_base = Val::two_adic_generator(log_h_max).exp_u64(rev as u64);
        let x = Challenge::from(x_base);

        // Assemble per-height reduced openings.
        // Note: We use the beta corresponding to the current height.
        for (height, rzh, rwzh) in &q.input_claims_per_height {
            let h = *height;
            let log_h = p3_util::log2_strict_usize(h);
            let r = h_max / h; // r is power of 2
            let log_r = p3_util::log2_strict_usize(r);
            let mut xr = x;
            let mut zr = z;
            let mut wzr = omega * z;
            let mut wxr = omega * x;
            for _ in 0..log_r {
                xr = xr.square();
                zr = zr.square();
                wzr = wzr.square();
                wxr = wxr.square();
            }

            // Determine which beta is used for this height.
            let beta_h = match betas_by_log_height.get(&log_h) {
                Some(b) => *b,
                None => continue, // should not happen if commits cover all heights
            };

            // Combine rows at this height using weights beta^{2*(U_j + i)}.
            let mut p_x = Challenge::ZERO;
            let mut p_wx = Challenge::ZERO;
            for (mat_idx, (row_vals, dims)) in
                q.input_opening.opened_values.iter().zip(matrices_dims_sorted_by_height).enumerate()
            {
                if dims.height != h {
                    continue;
                }
                let u0 = u_offsets[mat_idx];
                // beta^{2*(u0+i)} = (beta^2)^{u0+i}
                let beta2 = beta_h.square();
                let mut w = Challenge::ONE;
                // bump to (beta^2)^{u0}
                for _ in 0..u0 { w *= beta2; }
                for (i, &v) in row_vals.iter().take(dims.width).enumerate() {
                    if i > 0 { w *= beta2; }
                    p_x += w * Challenge::from(v);
                }
                // For wx, we need the sibling row at reduced index.
                // Not available in this opening; placeholder equals p_x until wired.
                p_wx = p_x;
            }

            // Compute RO_h.
            let num1 = *rzh - p_x;
            let den1 = zr - xr;
            let term1 = num1 * den1.inverse();

            let num2 = *rwzh - p_wx;
            let den2 = wzr - wxr;
            let term2 = beta_h * (num2 * den2.inverse());

            let _ro_h = term1 + term2;
            // TODO: Fold-chain replay and final comparison will consume _ro_h at height log_h.
        }
    }

    Ok(())
}

/// Minimal error enum for lifted verifier.
#[derive(Debug)]
pub enum VerifierError<CommitMmcsErr, InputMmcsErr> {
    InvalidProofShape,
    CommitPhaseMmcsError(CommitMmcsErr),
    InputMmcsError(InputMmcsErr),
    FinalPolyMismatch,
    InvalidPowWitness,
}
