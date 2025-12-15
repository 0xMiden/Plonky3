//! Polynomial Commitment Scheme combining DEEP quotient and FRI.
//!
//! This module provides high-level `open` and `verify` functions that orchestrate
//! the DEEP quotient construction and FRI protocol into a complete PCS.
//!
//! # Overview
//!
//! The PCS operates in two phases:
//!
//! 1. **Opening (Prover)**: Given committed matrices and evaluation points,
//!    computes polynomial evaluations, constructs a DEEP quotient, and generates
//!    a FRI proof of low-degree.
//!
//! 2. **Verification (Verifier)**: Given commitments, evaluation points, and a proof,
//!    verifies the DEEP quotient and FRI queries to confirm the claimed evaluations.

mod proof;

use alloc::vec::Vec;

use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::{Dimensions, Matrix};
use p3_util::log2_strict_usize;

pub use self::proof::{PcsError, Proof, QueryProof};
use crate::deep::MatrixGroupEvals;
use crate::deep::interpolate::SinglePointQuotient;
use crate::deep::prover::DeepPoly;
use crate::deep::verifier::DeepOracle;
use crate::fri::{CommitPhaseData, Params};
use crate::utils::bit_reversed_coset_points;

/// Open committed matrices at multiple evaluation points.
///
/// # Type Parameters
/// - `F`: Base field (must be two-adic for FRI)
/// - `EF`: Extension field for challenges and evaluations
/// - `InputMmcs`: MMCS used to commit the input matrices
/// - `FriMmcs`: MMCS used for FRI round commitments (typically `ExtensionMmcs<F, EF, _>`)
/// - `M`: Matrix type for input matrices
/// - `Challenger`: Fiat-Shamir challenger
/// - `NUM_POINTS`: Number of evaluation points (const generic)
///
/// # Arguments
/// - `input_mmcs`: The MMCS instance used for initial commitment
/// - `prover_data`: Prover data from the commitment phase (one per committed group)
/// - `eval_points`: Array of out-of-domain evaluation points
/// - `challenger`: Mutable reference to the Fiat-Shamir challenger
/// - `params`: FRI/PCS parameters (includes num_queries)
/// - `fri_mmcs`: MMCS instance for FRI round commitments
/// - `alignment`: Column alignment for batching (typically hasher's rate)
///
/// # Returns
/// A `Proof` containing evaluations and all opening proofs
#[allow(clippy::too_many_arguments)]
pub fn open<F, EF, InputMmcs, FriMmcs, M, Challenger, const NUM_POINTS: usize>(
    input_mmcs: &InputMmcs,
    prover_data: Vec<&InputMmcs::ProverData<M>>,
    eval_points: &[EF; NUM_POINTS],
    challenger: &mut Challenger,
    params: &Params,
    fri_mmcs: &FriMmcs,
    alignment: usize,
) -> Proof<F, EF, InputMmcs, FriMmcs>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    InputMmcs: Mmcs<F>,
    FriMmcs: Mmcs<EF>,
    M: Matrix<F>,
    Challenger: FieldChallenger<F> + CanObserve<FriMmcs::Commitment>,
{
    // ─────────────────────────────────────────────────────────────────────────
    // 1. Extract matrix structure from prover data
    // ─────────────────────────────────────────────────────────────────────────
    let matrices_groups: Vec<Vec<&M>> = prover_data
        .iter()
        .map(|pd| input_mmcs.get_matrices(*pd))
        .collect();

    // Determine LDE domain size from tallest matrix
    let max_height = matrices_groups
        .iter()
        .flat_map(|g| g.iter().map(|m| m.height()))
        .max()
        .expect("at least one matrix required");
    let log_n = log2_strict_usize(max_height);
    let coset_points = bit_reversed_coset_points::<F>(log_n);

    // ─────────────────────────────────────────────────────────────────────────
    // 2. Compute evaluations at each opening point
    // ─────────────────────────────────────────────────────────────────────────
    #[allow(clippy::type_complexity)]
    let mut quotients_and_evals: Vec<(SinglePointQuotient<F, EF>, Vec<MatrixGroupEvals<EF>>)> =
        Vec::with_capacity(NUM_POINTS);

    for &z in eval_points {
        let quotient = SinglePointQuotient::<F, EF>::new(z, &coset_points);
        let evals = quotient.batch_eval_lifted(&matrices_groups, &coset_points, params.log_blowup);
        quotients_and_evals.push((quotient, evals));
    }

    // Extract evals for the proof: evals[point_idx][commit_idx] = MatrixGroupEvals
    let evals: Vec<Vec<MatrixGroupEvals<EF>>> = quotients_and_evals
        .iter()
        .map(|(_, evals)| evals.clone())
        .collect();

    // Observe evaluations in challenger
    for point_evals in &evals {
        for group_evals in point_evals {
            for val in group_evals.flatten() {
                challenger.observe_algebra_element(*val);
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 3. Construct DEEP quotient
    // ─────────────────────────────────────────────────────────────────────────
    #[allow(clippy::type_complexity)]
    let openings_for_deep: Vec<(&SinglePointQuotient<F, EF>, Vec<MatrixGroupEvals<EF>>)> =
        quotients_and_evals
            .iter()
            .map(|(q, e)| (q, e.clone()))
            .collect();

    let deep_poly = DeepPoly::new(
        input_mmcs,
        &openings_for_deep,
        prover_data,
        challenger,
        alignment,
    );

    // ─────────────────────────────────────────────────────────────────────────
    // 4. FRI commit phase
    // ─────────────────────────────────────────────────────────────────────────
    // The deep_poly contains evaluations on the LDE domain (size max_height).
    // FRI will prove that this polynomial is low-degree.
    let deep_evals = deep_poly.evals().to_vec();

    let (fri_commit_data, fri_commit_proof) =
        CommitPhaseData::<F, EF, _>::new(fri_mmcs, params, deep_evals, challenger);

    // ─────────────────────────────────────────────────────────────────────────
    // 5. Sample query indices
    // ─────────────────────────────────────────────────────────────────────────
    let query_indices: Vec<usize> = (0..params.num_queries)
        .map(|_| challenger.sample_bits(log_n))
        .collect();

    // ─────────────────────────────────────────────────────────────────────────
    // 6. Generate query proofs
    // ─────────────────────────────────────────────────────────────────────────
    let query_proofs: Vec<QueryProof<F, EF, InputMmcs, FriMmcs>> = query_indices
        .iter()
        .map(|&index| {
            // Open DeepPoly at this index
            let deep_query = deep_poly.open(input_mmcs, index);

            // Open FRI rounds at this index
            let fri_round_openings = fri_commit_data.open_query(fri_mmcs, params, index);

            QueryProof::new(deep_query, fri_round_openings)
        })
        .collect();

    // ─────────────────────────────────────────────────────────────────────────
    // 7. Assemble and return proof
    // ─────────────────────────────────────────────────────────────────────────
    Proof {
        evals,
        fri_commit_proof,
        query_proofs,
    }
}

/// Verify polynomial evaluation claims against a commitment.
///
/// # Type Parameters
/// Same as `open`
///
/// # Arguments
/// - `input_mmcs`: The MMCS instance used for initial commitment
/// - `commitments`: The commitments to verify against (with dimensions)
/// - `eval_points`: Array of out-of-domain evaluation points
/// - `proof`: The proof to verify
/// - `challenger`: Mutable reference to the Fiat-Shamir challenger
/// - `params`: FRI/PCS parameters
/// - `fri_mmcs`: MMCS instance for FRI round commitments
/// - `alignment`: Column alignment (must match prover's)
///
/// # Returns
/// `Ok(evals)` where `evals[point_idx][commit_idx]` contains the verified evaluations,
/// or `Err` if verification fails.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn verify<F, EF, InputMmcs, FriMmcs, Challenger, const NUM_POINTS: usize>(
    input_mmcs: &InputMmcs,
    commitments: &[(InputMmcs::Commitment, Vec<Dimensions>)],
    eval_points: &[EF; NUM_POINTS],
    proof: &Proof<F, EF, InputMmcs, FriMmcs>,
    challenger: &mut Challenger,
    params: &Params,
    fri_mmcs: &FriMmcs,
    alignment: usize,
) -> Result<Vec<Vec<MatrixGroupEvals<EF>>>, PcsError<InputMmcs::Error, FriMmcs::Error>>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    InputMmcs: Mmcs<F>,
    FriMmcs: Mmcs<EF>,
    Challenger: FieldChallenger<F> + CanObserve<FriMmcs::Commitment>,
    InputMmcs::Error: core::fmt::Debug,
    FriMmcs::Error: core::fmt::Debug,
{
    // ─────────────────────────────────────────────────────────────────────────
    // 1. Validate proof structure
    // ─────────────────────────────────────────────────────────────────────────
    if proof.query_proofs.len() != params.num_queries {
        return Err(PcsError::WrongNumQueries {
            expected: params.num_queries,
            actual: proof.query_proofs.len(),
        });
    }

    // Extract dimensions for computing domain
    let max_height = commitments
        .iter()
        .flat_map(|(_, dims)| dims.iter().map(|d| d.height))
        .max()
        .expect("at least one matrix required");
    let log_n = log2_strict_usize(max_height);
    let log_max_degree = log_n - params.log_blowup;

    // ─────────────────────────────────────────────────────────────────────────
    // 2. Observe claimed evaluations
    // ─────────────────────────────────────────────────────────────────────────
    for point_evals in &proof.evals {
        for group_evals in point_evals {
            for val in group_evals.flatten() {
                challenger.observe_algebra_element(*val);
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 3. Construct verifier's DEEP oracle
    // ─────────────────────────────────────────────────────────────────────────
    // Build openings for oracle: pair each eval_point with its evaluations
    let openings_for_oracle: Vec<(EF, Vec<MatrixGroupEvals<EF>>)> = eval_points
        .iter()
        .zip(proof.evals.iter())
        .map(|(&z, evals)| (z, evals.clone()))
        .collect();

    let deep_oracle = DeepOracle::new(
        &openings_for_oracle,
        commitments.to_vec(),
        challenger,
        alignment,
    );

    // ─────────────────────────────────────────────────────────────────────────
    // 4. Process FRI commit proof to get betas
    // ─────────────────────────────────────────────────────────────────────────
    let betas = proof.fri_commit_proof.sample_betas::<F, _>(challenger);

    // ─────────────────────────────────────────────────────────────────────────
    // 5. Sample query indices (must match prover)
    // ─────────────────────────────────────────────────────────────────────────
    let query_indices: Vec<usize> = (0..params.num_queries)
        .map(|_| challenger.sample_bits(log_n))
        .collect();

    // ─────────────────────────────────────────────────────────────────────────
    // 6. Verify each query
    // ─────────────────────────────────────────────────────────────────────────
    for (index, query_proof) in query_indices.into_iter().zip(proof.query_proofs.iter()) {
        // 7a. Verify input matrix openings and compute expected DeepPoly value
        let deep_eval = deep_oracle
            .query(input_mmcs, index, &query_proof.input_openings)
            .map_err(PcsError::InputMmcsError)?;

        // 7c. Verify FRI rounds
        proof.fri_commit_proof.verify_query::<F>(
            fri_mmcs,
            params,
            index,
            log_max_degree,
            deep_eval,
            &betas,
            &query_proof.fri_round_openings,
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 7. Return verified evaluations
    // ─────────────────────────────────────────────────────────────────────────
    Ok(proof.evals.clone())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_commit::{ExtensionMmcs, Mmcs};
    use p3_dft::{Radix2DFTSmallBatch, TwoAdicSubgroupDft};
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_matrix::Matrix;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use p3_util::reverse_slice_index_bits;
    use rand::distr::StandardUniform;
    use rand::prelude::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::*;
    use crate::fri::Params;
    use crate::merkle_tree::{Lifting, MerkleTreeLmcs};

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type P = <F as Field>::Packing;

    const WIDTH: usize = 16;
    const RATE: usize = 8;
    const DIGEST: usize = 8;

    type Perm = Poseidon2BabyBear<WIDTH>;
    type Sponge = PaddingFreeSponge<Perm, WIDTH, RATE, DIGEST>;
    type Compress = TruncatedPermutation<Perm, 2, DIGEST, WIDTH>;

    type BaseLmcs = MerkleTreeLmcs<P, P, Sponge, Compress, WIDTH, DIGEST>;
    type FriMmcs = ExtensionMmcs<F, EF, BaseLmcs>;
    type Challenger = DuplexChallenger<F, Perm, WIDTH, RATE>;

    fn test_components() -> (Perm, BaseLmcs, FriMmcs) {
        let mut rng = SmallRng::seed_from_u64(2025);
        let perm = Perm::new_from_rng_128(&mut rng);
        let sponge = Sponge::new(perm.clone());
        let compress = Compress::new(perm.clone());
        let base_lmcs = MerkleTreeLmcs::new(sponge, compress, Lifting::Upsample);
        let fri_mmcs = ExtensionMmcs::new(base_lmcs.clone());
        (perm, base_lmcs, fri_mmcs)
    }

    /// Generate a matrix of LDE evaluations for random low-degree polynomials.
    ///
    /// Each column is a polynomial of degree `poly_degree`, evaluated on the coset gK
    /// in bit-reversed order, where g = F::GENERATOR and K is a subgroup of order `lde_size`.
    ///
    /// The coset evaluation is computed by scaling coefficients: for f(X) = Σ c_j X^j,
    /// the coset evaluations f(gX) = Σ (c_j g^j) X^j are obtained by DFT of scaled coefficients.
    fn generate_lde_matrix(
        rng: &mut SmallRng,
        log_poly_degree: usize,
        log_blowup: usize,
        num_columns: usize,
    ) -> RowMajorMatrix<F> {
        let poly_degree = 1 << log_poly_degree;
        let lde_size = poly_degree << log_blowup;
        let dft = Radix2DFTSmallBatch::<F>::default();
        let g = F::GENERATOR;

        // Generate LDE for each column
        let mut all_evals: Vec<Vec<F>> = Vec::with_capacity(num_columns);
        for _ in 0..num_columns {
            // Random polynomial coefficients
            let coeffs: Vec<F> = (0..poly_degree)
                .map(|_| rng.sample(StandardUniform))
                .collect();

            // Scale coefficients by g^j for coset evaluation: f(gX) = Σ (c_j g^j) X^j
            let mut scaled_coeffs: Vec<F> = coeffs
                .into_iter()
                .zip(g.powers())
                .map(|(c, g_pow)| c * g_pow)
                .collect();

            // Zero-pad to LDE size
            scaled_coeffs.resize(lde_size, F::ZERO);

            // DFT to get evaluations on subgroup K (which become coset gK evaluations)
            let mut evals = dft.dft_algebra(scaled_coeffs);
            reverse_slice_index_bits(&mut evals);
            all_evals.push(evals);
        }

        // Transpose to row-major: rows are evaluation points, columns are polynomials
        let mut values = Vec::with_capacity(lde_size * num_columns);
        for row_idx in 0..lde_size {
            for col in &all_evals {
                values.push(col[row_idx]);
            }
        }

        RowMajorMatrix::new(values, num_columns)
    }

    #[test]
    fn test_pcs_open_verify_roundtrip() {
        let rng = &mut SmallRng::seed_from_u64(42);
        let (perm, base_lmcs, fri_mmcs) = test_components();

        let log_blowup = 2;
        let params = Params {
            log_blowup,
            log_folding_factor: 1,
            log_final_degree: 2,
            num_queries: 5,
        };
        let alignment = RATE;

        // Create a matrix of LDE evaluations.
        // Each column is a random polynomial of degree < 2^log_poly_degree,
        // evaluated on the coset gK (g = F::GENERATOR, K = subgroup of order 2^log_n).
        //
        // The DEEP quotient Q(X) computed from these polynomials will have degree
        // at most 2^log_poly_degree - 1, satisfying FRI's low-degree requirement.
        let log_poly_degree = 6; // polynomial degree = 64
        let num_columns = 3;
        let matrix = generate_lde_matrix(rng, log_poly_degree, log_blowup, num_columns);
        let matrices: Vec<RowMajorMatrix<F>> = vec![matrix];

        // Commit matrices
        let (commitment, prover_data) = base_lmcs.commit(matrices.clone());
        let dims: Vec<Dimensions> = matrices.iter().map(|m| m.dimensions()).collect();

        // Evaluation points
        let z1: EF = rng.sample(StandardUniform);
        let z2: EF = rng.sample(StandardUniform);
        let eval_points = [z1, z2];

        // Prover
        let mut prover_challenger = Challenger::new(perm.clone());
        prover_challenger.observe(commitment);

        let proof = open::<F, EF, _, _, _, _, 2>(
            &base_lmcs,
            vec![&prover_data],
            &eval_points,
            &mut prover_challenger,
            &params,
            &fri_mmcs,
            alignment,
        );

        // Verifier
        let mut verifier_challenger = Challenger::new(perm);
        verifier_challenger.observe(commitment);

        let result = verify::<F, EF, _, _, _, 2>(
            &base_lmcs,
            &[(commitment, dims)],
            &eval_points,
            &proof,
            &mut verifier_challenger,
            &params,
            &fri_mmcs,
            alignment,
        );

        assert!(result.is_ok(), "Verification should succeed");
        let verified_evals = result.unwrap();
        assert_eq!(verified_evals.len(), 2, "Should have 2 evaluation points");
    }
}
