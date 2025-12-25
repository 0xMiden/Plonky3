//! Rescue-like hash AIR with periodic round constants.
//!
//! This test arithmetizes a hash chain of length NUM_HASHES to demonstrate
//! periodic column integration. The round constants are provided as periodic
//! columns with period NUM_ROUNDS, allowing them to repeat across multiple
//! hash invocations without duplicating data in the trace.

use core::marker::PhantomData;
use core::ops::Mul;

use p3_air::{Air, AirBuilder, BaseAir, PeriodicAirBuilder};
use p3_baby_bear::BabyBear;
use p3_challenger::{HashChallenger, SerializingChallenger32};
use p3_circle::{CirclePcs, CirclePeriodicEvaluator};
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::exponentiation::{exp_1717986917, exp_1725656503};
use p3_field::extension::BinomialExtensionField;
use p3_field::{InjectiveMonomial, PrimeCharacteristicRing, PrimeField64};
use p3_fri::TwoAdicPeriodicEvaluator;
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_keccak::Keccak256Hash;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_mersenne_31::Mersenne31;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use p3_uni_stark::{StarkConfig, prove_with_periodic, verify_with_periodic};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Standard FRI parameters for both tests
const LOG_BLOWUP: usize = 3;
const NUM_QUERIES: usize = 28;

/// Number of rounds per hash invocation. The periodic columns have this period.
const NUM_ROUNDS: usize = 8;

/// Length of the hash chain being proved.
const NUM_HASHES: usize = 8;

/// Trace size = NUM_ROUNDS × NUM_HASHES (one row per round).
const TRACE_SIZE: usize = NUM_ROUNDS * NUM_HASHES;

/// Circulant matrix coefficients for width 24.
/// This is supposed to be MDS (or near-MDS for M31) but most likely it is not.
const MDS_COEFFS_24: [u64; 24] = [
    7, 1, 3, 8, 4, 6, 2, 9, 5, 1, 3, 7, 8, 2, 4, 9, 5, 1, 6, 3, 7, 2, 8, 4,
];

/// Apply circulant MDS matrix to state.
fn apply_mds<F: PrimeField64, const WIDTH: usize>(
    state: [F; WIDTH],
    coeffs: &[u64; WIDTH],
) -> [F; WIDTH] {
    core::array::from_fn(|i| {
        (0..WIDTH)
            .map(|j| state[(i + j) % WIDTH] * F::from_u64(coeffs[j]))
            .sum()
    })
}

/// Rescue-like AIR with periodic round constants.
///
/// This AIR proves that a Rescue-like hash was computed correctly.
/// Each row represents the state after applying the round function.
/// Round constants (ark1, ark2) are provided as periodic columns with period = NUM_ROUNDS.
///
/// The constraint structure avoids computing the inverse S-box:
///   MDS((MDS(h) + ark1)^α) + ark2 = h'^α
///
/// Both sides have degree α, so the constraint degree equals the S-box degree.
///
/// Generic over:
/// - F: the field
/// - WIDTH: state width
/// - ALPHA: S-box degree (x^ALPHA)
#[derive(Clone)]
struct RescueLikeAir<F, const WIDTH: usize, const ALPHA: u64> {
    /// First round constants (ark1): ark1_periodic[i][round] = constant for element i at round
    ark1_periodic: Vec<Vec<F>>,
    /// Second round constants (ark2): ark2_periodic[i][round] = constant for element i at round
    ark2_periodic: Vec<Vec<F>>,
    /// MDS coefficients
    mds_coeffs: [u64; WIDTH],
}

impl<F, const WIDTH: usize, const ALPHA: u64> RescueLikeAir<F, WIDTH, ALPHA>
where
    F: PrimeField64,
{
    fn new(mds_coeffs: [u64; WIDTH]) -> Self {
        // Generate deterministic round constants (two sets: ark1, ark2)
        let mut rng = SmallRng::seed_from_u64(0x524553435545); // "RESCUE" in hex

        // Generate ark1 constants
        let ark1: Vec<F> = (0..NUM_ROUNDS * WIDTH)
            .map(|_| F::from_u64(rng.random::<u64>() % (1 << 30)))
            .collect();

        // Generate ark2 constants
        let ark2: Vec<F> = (0..NUM_ROUNDS * WIDTH)
            .map(|_| F::from_u64(rng.random::<u64>() % (1 << 30)))
            .collect();

        // Organize as periodic columns: ark_periodic[element][round]
        let ark1_periodic: Vec<Vec<F>> = (0..WIDTH)
            .map(|elem_idx| {
                (0..NUM_ROUNDS)
                    .map(|round| ark1[round * WIDTH + elem_idx])
                    .collect()
            })
            .collect();

        let ark2_periodic: Vec<Vec<F>> = (0..WIDTH)
            .map(|elem_idx| {
                (0..NUM_ROUNDS)
                    .map(|round| ark2[round * WIDTH + elem_idx])
                    .collect()
            })
            .collect();

        Self {
            ark1_periodic,
            ark2_periodic,
            mds_coeffs,
        }
    }

    /// Get ark1 constants for a specific round.
    fn get_ark1(&self, round: usize) -> [F; WIDTH] {
        core::array::from_fn(|i| self.ark1_periodic[i][round])
    }

    /// Get ark2 constants for a specific round.
    fn get_ark2(&self, round: usize) -> [F; WIDTH] {
        core::array::from_fn(|i| self.ark2_periodic[i][round])
    }

    /// Compute one round of Rescue-like hash.
    ///
    /// Full Rescue round: h' = MDS((MDS(h) + ark1)^α + ark2)^(1/α)
    ///
    /// We compute this as:
    ///   temp = MDS(h) + ark1        (linear)
    ///   temp = temp^α               (forward S-box)
    ///   temp = MDS(temp) + ark2     (linear)
    ///   h' = temp^(1/α)             (inverse S-box)
    fn compute_round(&self, state: &mut [F; WIDTH], round: usize)
    where
        F: InjectiveMonomial<ALPHA>,
    {
        let ark1 = self.get_ark1(round);
        let ark2 = self.get_ark2(round);

        // MDS
        *state = apply_mds(*state, &self.mds_coeffs);

        // Add ark1
        for i in 0..WIDTH {
            state[i] += ark1[i];
        }

        // Forward S-box: x^α
        for s in state.iter_mut() {
            *s = s.injective_exp_n();
        }

        // MDS
        *state = apply_mds(*state, &self.mds_coeffs);

        // Add ark2
        for i in 0..WIDTH {
            state[i] += ark2[i];
        }

        // Inverse S-box: x^(1/α)
        // Uses optimized addition chains for the inverse exponents:
        // - Mersenne31 (α=5): x^1717986917 since 5 * 1717986917 ≡ 1 (mod p-1)
        // - BabyBear (α=7): x^1725656503 since 7 * 1725656503 ≡ 1 (mod p-1)
        for s in state.iter_mut() {
            *s = match ALPHA {
                5 => exp_1717986917(*s),
                7 => exp_1725656503(*s),
                _ => panic!("Unsupported ALPHA for inverse S-box: {}", ALPHA),
            };
        }
    }

    /// Compute the full hash.
    fn hash(&self, input: [F; WIDTH]) -> [F; WIDTH]
    where
        F: InjectiveMonomial<ALPHA>,
    {
        let mut state = input;
        for round in 0..NUM_ROUNDS {
            self.compute_round(&mut state, round);
        }
        state
    }

    /// Generate trace for proving knowledge of preimage.
    /// Each row i contains the state BEFORE round (i % NUM_ROUNDS).
    /// The trace has TRACE_SIZE rows, with the hash computation repeating.
    fn generate_trace(&self, preimage: [F; WIDTH]) -> RowMajorMatrix<F>
    where
        F: InjectiveMonomial<ALPHA>,
    {
        let mut values = Vec::with_capacity(TRACE_SIZE * WIDTH);
        let mut state = preimage;

        for row in 0..TRACE_SIZE {
            // Store state before this round
            values.extend_from_slice(&state);
            // Compute round (wrapping around when hash completes)
            let round = row % NUM_ROUNDS;
            self.compute_round(&mut state, round);
        }

        RowMajorMatrix::new(values, WIDTH)
    }
}

impl<F, const WIDTH: usize, const ALPHA: u64> BaseAir<F> for RescueLikeAir<F, WIDTH, ALPHA>
where
    F: PrimeCharacteristicRing + Sync + Copy,
{
    fn width(&self) -> usize {
        WIDTH
    }

    fn periodic_table(&self) -> Vec<Vec<F>> {
        // Interleave ark1 and ark2: [ark1[0], ark1[1], ..., ark2[0], ark2[1], ...]
        let mut table = self.ark1_periodic.clone();
        table.extend(self.ark2_periodic.clone());
        table
    }
}

/// Compute x^5 explicitly (for Mersenne31, α=5)
fn exp5<E: Clone + Mul<Output = E>>(x: E) -> E {
    let x2 = x.clone() * x.clone();
    let x4 = x2.clone() * x2;
    x4 * x
}

/// Compute x^7 explicitly (for BabyBear/Goldilocks, α=7)
fn exp7<E: Clone + Mul<Output = E>>(x: E) -> E {
    let x2 = x.clone() * x.clone();
    let x4 = x2.clone() * x2.clone();
    let x6 = x4.clone() * x2;
    x6 * x
}

impl<AB, const WIDTH: usize, const ALPHA: u64> Air<AB> for RescueLikeAir<AB::F, WIDTH, ALPHA>
where
    AB: AirBuilder + PeriodicAirBuilder,
    AB::F: PrimeCharacteristicRing + PrimeField64 + Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).expect("matrix should have a local row");
        let next = main.row_slice(1).expect("matrix should have a next row");

        // Get state from local row (h)
        let h: [AB::Expr; WIDTH] = core::array::from_fn(|i| local[i].clone().into());

        // Get next state (h')
        let h_next: [AB::Expr; WIDTH] = core::array::from_fn(|i| next[i].clone().into());

        // Get periodic round constants (ark1 is first WIDTH columns, ark2 is next WIDTH)
        let periodic = builder.periodic_values();
        let ark1: [AB::Expr; WIDTH] = core::array::from_fn(|i| periodic[i].clone().into());
        let ark2: [AB::Expr; WIDTH] = core::array::from_fn(|i| periodic[WIDTH + i].clone().into());

        // Convert MDS coefficients to field elements
        let mds_field: [AB::F; WIDTH] =
            core::array::from_fn(|i| AB::F::from_u64(self.mds_coeffs[i]));

        // Helper function to apply MDS to an expression array
        let apply_mds_expr = |input: &[AB::Expr; WIDTH]| -> [AB::Expr; WIDTH] {
            core::array::from_fn(|i| {
                (0..WIDTH)
                    .map(|j| {
                        let coeff: AB::Expr = mds_field[j].into();
                        input[(i + j) % WIDTH].clone() * coeff
                    })
                    .sum()
            })
        };

        // Helper function to apply S-box to an expression array
        let apply_sbox = |input: &[AB::Expr; WIDTH]| -> [AB::Expr; WIDTH] {
            core::array::from_fn(|i| match ALPHA {
                5 => exp5(input[i].clone()),
                7 => exp7(input[i].clone()),
                _ => panic!("Unsupported ALPHA: {}", ALPHA),
            })
        };

        // Rescue-like constraint: MDS((MDS(h) + ark1)^α) + ark2 = h'^α
        //
        // Left side (forward path from h):
        //   step1 = MDS(h)
        //   step2 = step1 + ark1
        //   step3 = step2^α           (forward S-box)
        //   step4 = MDS(step3) + ark2
        //
        // Right side (from h'):
        //   step5 = h'^α
        //
        // Constraint: step4 == step5

        // Left side computation
        let step1 = apply_mds_expr(&h);
        let step2: [AB::Expr; WIDTH] = core::array::from_fn(|i| step1[i].clone() + ark1[i].clone());
        let step3 = apply_sbox(&step2);
        let step4_mds = apply_mds_expr(&step3);
        let step4: [AB::Expr; WIDTH] =
            core::array::from_fn(|i| step4_mds[i].clone() + ark2[i].clone());

        // Right side computation
        let step5 = apply_sbox(&h_next);

        // Constraint: step4[i] == step5[i] (on transition rows)
        for i in 0..WIDTH {
            builder
                .when_transition()
                .assert_eq(step4[i].clone(), step5[i].clone());
        }
    }
}

/// Test proving knowledge of preimage using two-adic FRI with BabyBear.
/// State width 24, digest size 8, S-box x^7.
#[test]
fn test_rescue_preimage_two_adic_babybear() {
    const WIDTH: usize = 24;
    const ALPHA: u64 = 7;

    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;
    type ByteHash = Keccak256Hash;
    type FieldHash = SerializingHasher<ByteHash>;
    type MyCompress = CompressionFunctionFromHasher<ByteHash, 2, 32>;
    type ValMmcs = MerkleTreeMmcs<Val, u8, FieldHash, MyCompress, 32>;
    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    type Dft = Radix2DitParallel<Val>;
    type Challenger = SerializingChallenger32<Val, HashChallenger<u8, ByteHash, 32>>;
    type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
    type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;

    // Create the Rescue-like AIR
    let air = RescueLikeAir::<Val, WIDTH, ALPHA>::new(MDS_COEFFS_24);

    // Generate a random preimage (the secret)
    let mut rng = SmallRng::seed_from_u64(42);
    let preimage: [Val; WIDTH] =
        core::array::from_fn(|_| Val::from_u64(rng.random::<u64>() % (1 << 30)));

    // Compute the hash (the public output)
    let hash_output = air.hash(preimage);
    println!("BabyBear Rescue Preimage (first 3): {:?}", &preimage[..3]);
    println!(
        "BabyBear Rescue Hash output (first 3): {:?}",
        &hash_output[..3]
    );

    // Generate the trace
    let trace = air.generate_trace(preimage);
    println!(
        "BabyBear Rescue Trace: {} rows x {} cols",
        trace.height(),
        trace.width()
    );

    // Set up PCS with Keccak256
    let byte_hash = ByteHash {};
    let field_hash = FieldHash::new(byte_hash);
    let compress = MyCompress::new(byte_hash);
    let val_mmcs = ValMmcs::new(field_hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();

    let fri_params = FriParameters {
        log_blowup: LOG_BLOWUP,
        log_final_poly_len: 0,
        num_queries: NUM_QUERIES,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 0,
        mmcs: challenge_mmcs,
    };
    let pcs = Pcs::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::from_hasher(vec![], byte_hash);
    let config = MyConfig::new(pcs, challenger);

    // Prove and verify
    let proof =
        prove_with_periodic::<_, _, TwoAdicPeriodicEvaluator<Dft>>(&config, &air, trace, &[]);

    verify_with_periodic::<_, _, TwoAdicPeriodicEvaluator<Dft>>(&config, &air, &proof, &[])
        .expect("verification failed");
    println!("BabyBear Rescue verification succeeded!");
}

/// Test proving knowledge of preimage using Circle STARKs with Mersenne31.
/// State width 24, digest size 8, S-box x^5.
#[test]
fn test_rescue_preimage_circle_m31() {
    const WIDTH: usize = 24;
    const ALPHA: u64 = 5;

    type Val = Mersenne31;
    type Challenge = BinomialExtensionField<Val, 3>; // M31 only supports degree-3 extension
    type ByteHash = Keccak256Hash;
    type FieldHash = SerializingHasher<ByteHash>;
    type MyCompress = CompressionFunctionFromHasher<ByteHash, 2, 32>;
    type ValMmcs = MerkleTreeMmcs<Val, u8, FieldHash, MyCompress, 32>;
    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    type Challenger = SerializingChallenger32<Val, HashChallenger<u8, ByteHash, 32>>;
    type Pcs = CirclePcs<Val, ValMmcs, ChallengeMmcs>;
    type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;

    // Create the Rescue-like AIR
    let air = RescueLikeAir::<Val, WIDTH, ALPHA>::new(MDS_COEFFS_24);

    // Generate a random preimage
    let mut rng = SmallRng::seed_from_u64(42);
    let preimage: [Val; WIDTH] =
        core::array::from_fn(|_| Val::from_u64(rng.random::<u64>() % (1 << 30)));

    // Compute the hash
    let hash_output = air.hash(preimage);
    println!("Circle M31 Rescue Preimage (first 3): {:?}", &preimage[..3]);
    println!(
        "Circle M31 Rescue Hash output (first 3): {:?}",
        &hash_output[..3]
    );

    // Generate the trace
    let trace = air.generate_trace(preimage);
    println!(
        "Circle M31 Rescue Trace: {} rows x {} cols",
        trace.height(),
        trace.width()
    );

    // Set up Circle PCS with Keccak256
    let byte_hash = ByteHash {};
    let field_hash = FieldHash::new(byte_hash);
    let compress = MyCompress::new(byte_hash);
    let val_mmcs = ValMmcs::new(field_hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    let fri_params = FriParameters {
        log_blowup: LOG_BLOWUP,
        log_final_poly_len: 0,
        num_queries: NUM_QUERIES,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 0,
        mmcs: challenge_mmcs,
    };

    let pcs = Pcs {
        mmcs: val_mmcs,
        fri_params,
        _phantom: PhantomData,
    };
    let challenger = Challenger::from_hasher(vec![], byte_hash);
    let config = MyConfig::new(pcs, challenger);

    // Prove and verify
    let proof = prove_with_periodic::<_, _, CirclePeriodicEvaluator>(&config, &air, trace, &[]);

    verify_with_periodic::<_, _, CirclePeriodicEvaluator>(&config, &air, &proof, &[])
        .expect("verification failed");
    println!("Circle M31 Rescue verification succeeded!");
}
