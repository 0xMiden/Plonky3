use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::{CanObserve, CanSampleBits, DuplexChallenger, FieldChallenger};
use p3_commit::{ExtensionMmcs, Pcs};
use p3_dft::Radix2Dit;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;

type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs =
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 8>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type MyPcs = TwoAdicFriPcs<BabyBear, Radix2Dit<BabyBear>, ValMmcs, ChallengeMmcs>;

/// Returns a permutation and a FRI-pcs instance.
fn get_ldt_for_testing<R: Rng>(
    rng: &mut R,
    log_final_poly_len: usize,
    log_folding_factor: usize,
) -> (Perm, MyPcs) {
    let perm = Perm::new_from_rng_128(rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let input_mmcs = ValMmcs::new(hash.clone(), compress.clone());
    let fri_mmcs = ChallengeMmcs::new(ValMmcs::new(hash, compress));
    let fri_params = FriParameters {
        log_blowup: 1,
        log_final_poly_len,
        num_queries: 10,
        proof_of_work_bits: 8,
        mmcs: fri_mmcs,
        log_folding_factor,
    };
    let dft = Radix2Dit::default();
    let pcs = MyPcs::new(dft, input_mmcs, fri_params);
    (perm, pcs)
}

/// Check that the loop of `pcs.commit`, `pcs.open`, and `pcs.verify` work correctly.
///
/// We create a random polynomial of size `1 << log_size` for each size in `polynomial_log_sizes`.
/// We then commit to these polynomials using a `log_blowup` of `1`.
///
/// We open each polynomial at the same point `zeta` and run FRI to verify the openings, stopping
/// FRI at `log_final_poly_len`.
fn do_test_fri_ldt<R: Rng>(
    rng: &mut R,
    log_final_poly_len: usize,
    polynomial_log_sizes: &[u8],
    log_folding_factor: usize,
) {
    let (perm, pcs) = get_ldt_for_testing(rng, log_final_poly_len, log_folding_factor);

    // Convert the polynomial_log_sizes into field elements so they can be observed.
    let val_sizes: Vec<Val> = polynomial_log_sizes
        .iter()
        .map(|&i| Val::from_u8(i))
        .collect();

    // --- Prover World ---
    let (commitment, opened_values, opening_proof, mut p_challenger) = {
        // Initialize the challenger and observe the `polynomial_log_sizes`.
        let mut challenger = Challenger::new(perm.clone());
        challenger.observe_slice(&val_sizes);

        // Generate random evaluation matrices for each polynomial degree.
        let evaluations: Vec<(TwoAdicMultiplicativeCoset<Val>, RowMajorMatrix<Val>)> =
            polynomial_log_sizes
                .iter()
                .map(|deg_bits| {
                    let deg = 1 << deg_bits;
                    (
                        // Get the TwoAdicSubgroup of this degree.
                        <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, deg),
                        // Generate a random matrix of evaluations.
                        RowMajorMatrix::<Val>::rand_nonzero(rng, deg, 16),
                    )
                })
                .collect();

        let num_evaluations = evaluations.len();

        // Commit to all the evaluation matrices.
        let (commitment, prover_data) =
            <TwoAdicFriPcs<BabyBear, Radix2Dit<BabyBear>, ValMmcs, ChallengeMmcs> as Pcs<
                Challenge,
                Challenger,
            >>::commit(&pcs, evaluations);

        // Observe the commitment.
        challenger.observe(commitment);

        // Sample the challenge point zeta which all polynomials
        // will be opened at.
        let zeta: Challenge = challenger.sample_algebra_element();

        // Prepare the data into the form expected by `pcs.open`.
        let open_data = vec![(&prover_data, vec![vec![zeta]; num_evaluations])]; // open every chunk at zeta

        // Open all polynomials at zeta and produce the opening proof.
        let (opened_values, opening_proof) = pcs.open(open_data, &mut challenger);

        // Return the commitment, opened values, opening proof and challenger.
        // The first three of these are always passed to the verifier. The
        // last is to double check that the prover and verifiers challengers
        // agree at appropriate points.
        (commitment, opened_values, opening_proof, challenger)
    };

    // --- Verifier World ---
    let mut v_challenger = {
        // Initialize the verifier's challenger with the same permutation.
        // Observe the `polynomial_log_sizes` and `commitment` in the same order
        // as the prover.
        let mut challenger = Challenger::new(perm);
        challenger.observe_slice(&val_sizes);
        challenger.observe(commitment);

        // Sample the opening point.
        let zeta = challenger.sample_algebra_element();

        // Construct the expected initial polynomial domains.
        // Right now it doesn't matter what these are so long as the size
        // is correct.
        let domains = polynomial_log_sizes.iter().map(|&size| {
            <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, 1 << size)
        });

        // Prepare the data into the form expected by `pcs.verify`.
        // Note that commitment and opened_values are always sent by
        // the prover.
        let commitments_with_opening_points = vec![(
            commitment,
            domains
                .into_iter()
                .zip(opened_values.into_iter().flatten().flatten())
                .map(|(domain, value)| (domain, vec![(zeta, value)]))
                .collect(),
        )];

        // Verify the opening proof.
        let verification = pcs.verify(
            commitments_with_opening_points,
            &opening_proof,
            &mut challenger,
        );
        assert!(verification.is_ok());
        challenger
    };

    // Check that the prover and verifier challengers agree.
    assert_eq!(
        p_challenger.sample_bits(8),
        v_challenger.sample_bits(8),
        "prover and verifier transcript have same state after FRI"
    );
}

/// Test that the FRI commit, open and verify process work correctly
/// for a range of `final_poly_degree` values and folding factors.
#[test]
fn test_fri_ldt() {
    // Test with different folding factors: 2, 4, and 8
    // Each configuration tests different polynomial sizes with different remainders
    for log_folding_factor in [1, 2, 3] {
        // Test multiple configurations: aligned and non-aligned cases
        let test_configs: Vec<(Vec<u8>, Vec<usize>)> = match log_folding_factor {
            1 => vec![
                // Folding factor 2: all sizes work
                (vec![5, 8, 10, 7, 5, 5, 7], (0..=4).collect()),
            ],
            2 => vec![
                // Folding factor 4: aligned case (all even)
                (vec![4, 6, 8, 10, 6, 4], vec![0, 2, 4]),
                // Non-aligned case: all odd (remainder 1)
                (vec![5, 7, 9, 11, 7, 5], vec![1, 3, 5]),
            ],
            3 => vec![
                // Folding factor 8: aligned case (all multiples of 3)
                (vec![6, 9, 12, 6, 9], vec![0, 3, 6]),
                // Non-aligned case: remainder 1 when divided by 3
                (vec![7, 10, 13, 7, 10], vec![1, 4, 7]),
                // Non-aligned case: remainder 2 when divided by 3
                (vec![8, 11, 14, 8, 11], vec![2, 5, 8]),
            ],
            _ => unreachable!(),
        };

        for (polynomial_log_sizes, log_final_poly_len_candidates) in test_configs {
            // Test different log_final_poly_len values
            // Key constraint: (log_poly_size - log_final_poly_len) must be divisible by log_folding_factor
            // for all polynomials. This ensures we can fold an integer number of times.
            //
            // Also need to ensure: min_polynomial_size > log_final_poly_len + log_blowup
            // With log_blowup = 1, we need min_polynomial_size > log_final_poly_len + 1
            let min_poly_size = *polynomial_log_sizes.iter().min().unwrap() as usize;
            let log_blowup = 1;
            let max_final_len = min_poly_size.saturating_sub(log_blowup + 1);

            // Filter test cases to only include those within the valid range
            let test_cases: Vec<usize> = log_final_poly_len_candidates
                .into_iter()
                .filter(|&x| x <= max_final_len)
                .collect();

            for &log_final_poly_len in &test_cases {
                let mut rng =
                    SmallRng::seed_from_u64((log_final_poly_len + log_folding_factor * 10) as u64);
                do_test_fri_ldt(
                    &mut rng,
                    log_final_poly_len,
                    &polynomial_log_sizes,
                    log_folding_factor,
                );
            }
        }
    }
}

/// This test is expected to panic because there is a polynomial degree which
/// the prover commits too which is less than `final_poly_degree`.
/// Tests with folding factor 2 (log_folding_factor = 1).
#[test]
#[should_panic]
fn test_fri_ldt_should_panic() {
    // Chosen to ensure there are both multiple polynomials
    // of the same size and that the array is not ordered.
    let polynomial_log_sizes = [5, 8, 10, 7, 5, 5, 7];
    let mut rng = SmallRng::seed_from_u64(5);
    do_test_fri_ldt(&mut rng, 5, &polynomial_log_sizes, 1);
}
