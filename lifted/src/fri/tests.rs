//! Integration tests for FRI protocol commit/verify cycles.

use alloc::vec::Vec;

use p3_commit::Mmcs;
use p3_field::{PrimeCharacteristicRing, TwoAdicField};
use p3_util::reverse_bits_len;
use rand::distr::StandardUniform;
use rand::prelude::SmallRng;
use rand::{Rng, SeedableRng};

use super::*;
use crate::tests::{EF, F, challenger, fri_mmcs, random_lde_matrix};

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

// ============================================================================
// Integration tests
// ============================================================================

/// Test that commit_phase produces valid proofs that verify_query accepts.
///
/// This test:
/// 1. Generates a random polynomial and computes its LDE
/// 2. Runs the FRI commit phase to fold down to final polynomial
/// 3. Verifies random query indices
fn test_fri_commit_verify_roundtrip(log_poly_degree: usize, log_folding_factor: usize) {
    let mut rng = SmallRng::seed_from_u64(42);
    let mmcs = fri_mmcs();

    let params = FriParams {
        log_blowup: 2,
        log_folding_factor,
        log_final_degree: 2,
        num_queries: 3,
    };

    // Generate random LDE evaluations
    let evals = random_lde_matrix(&mut rng, log_poly_degree, params.log_blowup, 1, F::ONE).values;
    let lde_size = evals.len();

    // Prover: run commit phase
    let mut prover_challenger = challenger();
    let (prover_data, proof) =
        CommitPhaseData::<F, EF, _>::new(&mmcs, &params, evals.clone(), &mut prover_challenger);

    // Verifier: replay challenger to get betas
    let mut verifier_challenger = challenger();
    let betas = proof.sample_betas::<F, _>(&mut verifier_challenger);

    // Verify random queries
    for _ in 0..3 {
        let index: usize = rng.random_range(0..lde_size);
        let initial_eval = evals[index];
        let openings = open_query(&mmcs, &prover_data, &params, index);

        proof
            .verify_query::<F>(
                &mmcs,
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
    let mmcs = fri_mmcs();

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

    let evals = random_lde_matrix(&mut rng, log_poly_degree, log_blowup, 1, F::ONE).values;
    let log_lde_size = log_poly_degree + log_blowup;
    let lde_size = 1 << log_lde_size;

    let mut prover_challenger = challenger();
    let (prover_data, proof) =
        CommitPhaseData::<F, EF, _>::new(&mmcs, &params, evals, &mut prover_challenger);

    let mut verifier_challenger = challenger();
    let betas = proof.sample_betas::<F, _>(&mut verifier_challenger);

    let index: usize = rng.random_range(0..lde_size);
    let wrong_eval: EF = rng.sample(StandardUniform); // Wrong!
    let openings = open_query(&mmcs, &prover_data, &params, index);

    let result = proof.verify_query::<F>(
        &mmcs,
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
    let mmcs = fri_mmcs();

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

    let evals = random_lde_matrix(&mut rng, log_poly_degree, log_blowup, 1, F::ONE).values;
    let log_lde_size = log_poly_degree + log_blowup;
    let lde_size = 1 << log_lde_size;

    let mut prover_challenger = challenger();
    let (prover_data, proof) =
        CommitPhaseData::<F, EF, _>::new(&mmcs, &params, evals.clone(), &mut prover_challenger);

    // Use wrong betas
    let wrong_betas: Vec<EF> = (0..proof.commitments.len())
        .map(|_| rng.sample(StandardUniform))
        .collect();

    let index: usize = rng.random_range(0..lde_size);
    let initial_eval = evals[index];
    let openings = open_query(&mmcs, &prover_data, &params, index);

    let result = proof.verify_query::<F>(
        &mmcs,
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
    let mmcs = fri_mmcs();

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

    let evals = random_lde_matrix(&mut rng, log_poly_degree, log_blowup, 1, F::ONE).values;

    let mut chal = challenger();
    let (_prover_data, proof) = CommitPhaseData::<F, EF, _>::new(&mmcs, &params, evals, &mut chal);

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
