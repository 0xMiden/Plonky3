//! End-to-end tests for DEEP quotient prover/verifier agreement.

use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::CanObserve;
use p3_commit::Mmcs;
use p3_field::FieldArray;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use rand::distr::StandardUniform;
use rand::prelude::SmallRng;
use rand::{Rng, SeedableRng};

use super::interpolate::PointQuotients;
use super::prover::DeepPoly;
use super::verifier::DeepOracle;
use super::{MatrixGroupEvals, OpeningClaim};
use crate::tests::{EF, F, RATE, base_lmcs, challenger};
use crate::utils::bit_reversed_coset_points;

// ============================================================================
// End-to-end test
// ============================================================================

/// End-to-end: prover's `DeepPoly.open()` must match verifier's `DeepOracle.query()`.
#[test]
fn deep_quotient_end_to_end() {
    let rng = &mut SmallRng::seed_from_u64(42);
    let lmcs = base_lmcs();

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

    // Step 2: Compute batched evaluations at both opening points
    let quotient = PointQuotients::<F, EF, 2>::new(FieldArray([z1, z2]), &coset_points);

    let matrices_ref: Vec<&RowMajorMatrix<F>> = matrices.iter().collect();
    let matrices_groups = [matrices_ref];
    let batched_evals = quotient.batch_eval_lifted(&matrices_groups, &coset_points, log_blowup);

    // Transpose batched evals to per-point format for verifier
    let evals1: Vec<MatrixGroupEvals<EF>> =
        batched_evals.iter().map(|g| g.map(|arr| arr[0])).collect();
    let evals2: Vec<MatrixGroupEvals<EF>> =
        batched_evals.iter().map(|g| g.map(|arr| arr[1])).collect();

    // Step 3: Prover constructs DeepPoly with challenger
    // The challenger samples alpha (column batching) and beta (point batching) internally
    let mut prover_challenger = challenger();
    prover_challenger.observe(commitment);

    let deep_poly = DeepPoly::new(
        &lmcs,
        &quotient,
        &batched_evals,
        vec![&prover_data],
        &mut prover_challenger,
        alignment,
    );

    // Step 4: Verifier constructs DeepOracle with separate challenger (same initial state)
    // Verifier's challenger must start in the same state as prover's was before DeepPoly::new
    let mut verifier_challenger = challenger();
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
