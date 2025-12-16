//! End-to-end tests for the complete PCS (DEEP + FRI).

use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::CanObserve;
use p3_commit::Mmcs;
use p3_field::Field;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use rand::distr::StandardUniform;
use rand::prelude::SmallRng;
use rand::{Rng, SeedableRng};

use super::*;
use crate::fri::FriParams;
use crate::tests::{EF, F, RATE, base_lmcs, challenger, fri_mmcs, random_lde_matrix};

// ============================================================================
// End-to-end test
// ============================================================================

#[test]
fn test_pcs_open_verify_roundtrip() {
    let rng = &mut SmallRng::seed_from_u64(42);
    let lmcs = base_lmcs();
    let mmcs = fri_mmcs();

    let config = PcsConfig {
        fri: FriParams {
            log_blowup: 2,
            log_folding_factor: 1,
            log_final_degree: 2,
            num_queries: 5,
        },
        alignment: RATE,
    };

    // Create a matrix of LDE evaluations.
    // Each column is a random polynomial of degree < 2^log_poly_degree,
    // evaluated on the coset gK (g = F::GENERATOR, K = subgroup of order 2^log_n).
    //
    // The DEEP quotient Q(X) computed from these polynomials will have degree
    // at most 2^log_poly_degree - 1, satisfying FRI's low-degree requirement.
    let log_poly_degree = 6; // polynomial degree = 64
    let num_columns = 3;
    let matrix = random_lde_matrix(
        rng,
        log_poly_degree,
        config.fri.log_blowup,
        num_columns,
        F::GENERATOR,
    );
    let matrices: Vec<RowMajorMatrix<F>> = vec![matrix];

    // Commit matrices
    let (commitment, prover_data) = lmcs.commit(matrices.clone());
    let dims: Vec<_> = matrices.iter().map(|m| m.dimensions()).collect();

    // Evaluation points
    let z1: EF = rng.sample(StandardUniform);
    let z2: EF = rng.sample(StandardUniform);
    let eval_points = [z1, z2];

    // Prover
    let mut prover_challenger = challenger();
    prover_challenger.observe(commitment);

    let proof = open::<F, EF, _, _, _, _>(
        &lmcs,
        vec![&prover_data],
        &eval_points,
        &mut prover_challenger,
        &config,
        &mmcs,
    );

    // Verifier
    let mut verifier_challenger = challenger();
    verifier_challenger.observe(commitment);

    let result = verify::<F, EF, _, _, _>(
        &lmcs,
        &[(commitment, dims)],
        &eval_points,
        &proof,
        &mut verifier_challenger,
        &config,
        &mmcs,
    );

    assert!(result.is_ok(), "Verification should succeed");
    let verified_evals = result.unwrap();
    assert_eq!(verified_evals.len(), 2, "Should have 2 evaluation points");
}
