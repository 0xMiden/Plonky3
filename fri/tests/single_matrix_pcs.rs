use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::{CanObserve, DuplexChallenger, FieldChallenger};
use p3_commit::{ExtensionMmcs, Pcs};
use p3_dft::Radix2DitParallel;
use p3_field::Field;
use p3_field::extension::BinomialExtensionField;
use p3_fri::{FriParameters, SingleMatrixPcs};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

fn seeded_rng() -> impl Rng {
    SmallRng::seed_from_u64(42)
}

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;

type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;

type ValMmcs =
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 8>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;

type Dft = Radix2DitParallel<Val>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type MyPcs = SingleMatrixPcs<Val, Dft, ValMmcs, ChallengeMmcs>;

fn get_pcs(log_blowup: usize) -> (MyPcs, Challenger) {
    let mut rng = seeded_rng();
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());

    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    let fri_params = FriParameters {
        log_blowup,
        log_final_poly_len: 0,
        num_queries: 10,
        proof_of_work_bits: 8,
        mmcs: challenge_mmcs,
        log_folding_factor: 1, // Default folding factor of 2
    };

    let pcs = MyPcs::new(Dft::default(), val_mmcs, fri_params);
    (pcs, Challenger::new(perm))
}

#[test]
fn test_single_matrix_commit_and_open() {
    let mut rng = seeded_rng();
    let (pcs, challenger) = get_pcs(1);

    // Create a single matrix
    let log_degree = 4;
    let degree = 1 << log_degree;
    let width = 8;
    let matrix = RowMajorMatrix::<Val>::rand(&mut rng, degree, width);

    // Get the natural domain for this degree
    let domain = <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, degree);

    // Commit to the single matrix
    let (commitment, prover_data) =
        <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, vec![(domain, matrix.clone())]);

    // Generate a random challenge point to open at
    let mut p_challenger = challenger.clone();
    p_challenger.observe(commitment.clone());
    let zeta: Challenge = p_challenger.sample_algebra_element();

    // Open at the challenge point
    let points = vec![vec![zeta]];
    let (openings, proof) = <MyPcs as Pcs<Challenge, Challenger>>::open(
        &pcs,
        vec![(&prover_data, points.clone())],
        &mut p_challenger,
    );

    // Verify the opening
    let mut v_challenger = challenger.clone();
    v_challenger.observe(commitment.clone());
    let verifier_zeta: Challenge = v_challenger.sample_algebra_element();
    assert_eq!(verifier_zeta, zeta);

    let claims = vec![(domain, vec![(zeta, openings[0][0][0].clone())])];
    <MyPcs as Pcs<Challenge, Challenger>>::verify(
        &pcs,
        vec![(commitment, claims)],
        &proof,
        &mut v_challenger,
    )
    .unwrap();
}

#[test]
fn test_single_matrix_multiple_points() {
    let mut rng = seeded_rng();
    let (pcs, challenger) = get_pcs(2);

    // Create a single matrix with larger degree
    let log_degree = 5;
    let degree = 1 << log_degree;
    let width = 10;
    let matrix = RowMajorMatrix::<Val>::rand(&mut rng, degree, width);

    let domain = <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, degree);

    // Commit
    let (commitment, prover_data) =
        <MyPcs as Pcs<Challenge, Challenger>>::commit(&pcs, vec![(domain, matrix.clone())]);

    // Generate multiple challenge points
    let mut p_challenger = challenger.clone();
    p_challenger.observe(commitment.clone());
    let zeta1: Challenge = p_challenger.sample_algebra_element();
    let zeta2: Challenge = p_challenger.sample_algebra_element();

    // Open at multiple points
    let points = vec![vec![zeta1, zeta2]];
    let (openings, proof) = <MyPcs as Pcs<Challenge, Challenger>>::open(
        &pcs,
        vec![(&prover_data, points.clone())],
        &mut p_challenger,
    );

    // Verify
    let mut v_challenger = challenger.clone();
    v_challenger.observe(commitment.clone());
    let v_zeta1: Challenge = v_challenger.sample_algebra_element();
    let v_zeta2: Challenge = v_challenger.sample_algebra_element();
    assert_eq!(v_zeta1, zeta1);
    assert_eq!(v_zeta2, zeta2);

    let claims = vec![(
        domain,
        vec![
            (zeta1, openings[0][0][0].clone()),
            (zeta2, openings[0][0][1].clone()),
        ],
    )];
    <MyPcs as Pcs<Challenge, Challenger>>::verify(
        &pcs,
        vec![(commitment, claims)],
        &proof,
        &mut v_challenger,
    )
    .unwrap();
}

#[test]
#[should_panic(expected = "SingleMatrixPcs only supports committing to exactly one matrix")]
fn test_single_matrix_rejects_multiple_matrices() {
    let mut rng = seeded_rng();
    let (pcs, _) = get_pcs(1);

    let log_degree = 3;
    let degree = 1 << log_degree;
    let width = 4;
    let matrix1 = RowMajorMatrix::<Val>::rand(&mut rng, degree, width);
    let matrix2 = RowMajorMatrix::<Val>::rand(&mut rng, degree, width);

    let domain = <MyPcs as Pcs<Challenge, Challenger>>::natural_domain_for_degree(&pcs, degree);

    // This should panic because we're trying to commit to two matrices
    let _ = <MyPcs as Pcs<Challenge, Challenger>>::commit(
        &pcs,
        vec![(domain, matrix1), (domain, matrix2)],
    );
}
