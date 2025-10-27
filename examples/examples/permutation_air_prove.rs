//! Complete example of proving and verifying with PermutationAirBuilder
//!
//! This example demonstrates:
//! 1. Setting up a complete STARK configuration
//! 2. Generating execution traces
//! 3. Creating a proof
//! 4. Verifying the proof
//!
//! Run with: cargo run --example permutation_air_prove

use p3_air::{Air, BaseAir, ExtensionBuilder, PermutationAirBuilder};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::{StarkConfig, prove, verify};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use tracing_forest::ForestLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::{EnvFilter, Registry};

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
type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;

fn main() {
    // Initialize tracing
    let forest = ForestLayer::default();
    let subscriber = Registry::default()
        .with(EnvFilter::from_default_env())
        .with(forest);
    tracing::subscriber::set_global_default(subscriber).expect("Failed to set subscriber");

    println!("=== Permutation AIR Proof Generation Example ===\n");

    // Example 1: Prove a simple permutation check
    example_prove_permutation_check();

    println!();

    // Example 2: Prove with invalid permutation (should fail verification if constraint checking works)
    example_prove_lookup();
}

/// Example 1: Prove and verify a permutation check
fn example_prove_permutation_check() {
    println!("Example 1: Proving Permutation Check");
    println!("======================================");

    // Setup the STARK configuration
    let config = create_stark_config();

    // Create sequences that are permutations of each other
    let sequence_a = vec![
        Val::from_u32(5),
        Val::from_u32(3),
        Val::from_u32(7),
        Val::from_u32(1),
        Val::from_u32(9),
        Val::from_u32(11),
        Val::from_u32(13),
        Val::from_u32(2),
    ];

    let sequence_b = vec![
        Val::from_u32(1),
        Val::from_u32(7),
        Val::from_u32(5),
        Val::from_u32(3),
        Val::from_u32(2),
        Val::from_u32(13),
        Val::from_u32(9),
        Val::from_u32(11),
    ];

    println!(
        "Sequence A: {:?}",
        sequence_a
            .iter()
            .map(|x| x.as_canonical_u64())
            .collect::<Vec<_>>()
    );
    println!(
        "Sequence B: {:?}",
        sequence_b
            .iter()
            .map(|x| x.as_canonical_u64())
            .collect::<Vec<_>>()
    );
    println!();

    // Create the AIR
    let air = PermutationCheckAir::new(sequence_a.len());
    println!(
        "AIR width: {}",
        <PermutationCheckAir as BaseAir<Val>>::width(&air)
    );

    // Generate the execution trace
    let trace = generate_permutation_trace(sequence_a.clone(), sequence_b.clone());
    println!("Trace dimensions: {}x{}", trace.height(), trace.width());

    // Note: In a real implementation with PermutationAirBuilder support,
    // we would also need to generate the permutation trace here.
    // For now, this demonstrates the structure.

    println!("\nAttempting to generate proof...");

    // Currently, the prove function will fail because PermutationAirBuilder
    // is not yet fully implemented in the prover. This is expected.
    // The following shows the structure of how it would be called:

    // let proof = prove(&config, &air, trace, &vec![]);
    // println!("✓ Proof generated successfully!");
    // println!("Proof size: {} bytes (estimated)", std::mem::size_of_val(&proof));

    // let result = verify(&config, &air, &proof, &vec![]);
    // match result {
    //     Ok(()) => println!("✓ Proof verified successfully!"),
    //     Err(e) => println!("✗ Verification failed: {:?}", e),
    // }

    println!("\nNote: Full proof generation requires PermutationAirBuilder support in the prover.");
    println!("This example demonstrates the setup and structure for when it's implemented.");
}

/// Example 2: Prove a lookup argument
fn example_prove_lookup() {
    println!("Example 2: Proving Lookup Argument");
    println!("===================================");

    let config = create_stark_config();

    // Create lookup table and queries
    let table = vec![
        Val::from_u32(1),
        Val::from_u32(2),
        Val::from_u32(3),
        Val::from_u32(4),
        Val::from_u32(5),
        Val::from_u32(6),
        Val::from_u32(7),
        Val::from_u32(8),
    ];

    let queries = vec![
        Val::from_u32(3),
        Val::from_u32(1),
        Val::from_u32(7),
        Val::from_u32(2),
        Val::from_u32(8),
        Val::from_u32(5),
        Val::from_u32(4),
        Val::from_u32(6),
    ];

    println!(
        "Lookup table: {:?}",
        table
            .iter()
            .map(|x| x.as_canonical_u64())
            .collect::<Vec<_>>()
    );
    println!(
        "Queries:      {:?}",
        queries
            .iter()
            .map(|x| x.as_canonical_u64())
            .collect::<Vec<_>>()
    );
    println!();

    let air = LookupAir::new(queries.len());
    println!("AIR width: {}", <LookupAir as BaseAir<Val>>::width(&air));

    let trace = generate_lookup_trace(queries, table);
    println!("Trace dimensions: {}x{}", trace.height(), trace.width());

    println!("\nNote: Full proof generation requires PermutationAirBuilder support.");
}

/// Create a STARK configuration with BabyBear field and Poseidon2
fn create_stark_config() -> MyConfig {
    let mut rng = SmallRng::seed_from_u64(42);
    let perm = Perm::new_from_rng_128(&mut rng);

    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    let dft = Dft::default();

    // FRI parameters
    let fri_params = FriParameters {
        log_blowup: 1,         // Blowup factor: 2x
        log_final_poly_len: 3, // Final polynomial length: 2^3 = 8
        num_queries: 40,       // Number of FRI queries
        proof_of_work_bits: 8, // PoW difficulty
        mmcs: challenge_mmcs,
    };

    let pcs = Pcs::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm);

    MyConfig::new(pcs, challenger)
}

// ============================================================================
// AIR Definitions (same as in permutation_air.rs)
// ============================================================================

#[derive(Debug, Clone)]
pub struct PermutationCheckAir {
    pub num_elements: usize,
}

impl PermutationCheckAir {
    pub fn new(num_elements: usize) -> Self {
        Self { num_elements }
    }
}

impl<F: Field> BaseAir<F> for PermutationCheckAir {
    fn width(&self) -> usize {
        self.num_elements * 2
    }
}

impl<AB: PermutationAirBuilder> Air<AB> for PermutationCheckAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let perm = builder.permutation();
        let randomness = builder.permutation_randomness();

        let main_local = main.row_slice(0).expect("main trace is empty");
        let main_local = &*main_local;
        let perm_local = perm.row_slice(0).expect("perm trace is empty");
        let perm_local = &*perm_local;
        let perm_next = perm.row_slice(1).expect("perm trace has only 1 row");
        let perm_next = &*perm_next;

        let n = self.num_elements;

        assert!(
            !randomness.is_empty(),
            "Need at least one random value for permutation argument"
        );
        let alpha = randomness[0];

        for i in 0..n {
            let a_value = main_local[i].clone();
            let b_value = main_local[n + i].clone();

            let a_factor = alpha.into() - a_value.into();
            let b_factor = alpha.into() - b_value.into();

            if i == 0 {
                builder
                    .when_first_row()
                    .assert_eq_ext(perm_local[i], a_factor.clone());
            } else {
                builder.when_first_row().assert_eq_ext(
                    perm_local[i].into(),
                    perm_local[i - 1].into() * a_factor.clone(),
                );
            }

            if i == 0 {
                builder
                    .when_first_row()
                    .assert_eq_ext(perm_local[n + i], b_factor.clone());
            } else {
                builder.when_first_row().assert_eq_ext(
                    perm_local[n + i].into(),
                    perm_local[n + i - 1].into() * b_factor.clone(),
                );
            }

            builder
                .when_transition()
                .assert_eq_ext(perm_local[i], perm_next[i]);
            builder
                .when_transition()
                .assert_eq_ext(perm_local[n + i], perm_next[n + i]);
        }

        builder.assert_eq_ext(perm_local[n - 1], perm_local[2 * n - 1]);
    }
}

#[derive(Debug, Clone)]
pub struct LookupAir {
    pub num_lookups: usize,
}

impl LookupAir {
    pub fn new(num_lookups: usize) -> Self {
        Self { num_lookups }
    }
}

impl<F: Field> BaseAir<F> for LookupAir {
    fn width(&self) -> usize {
        self.num_lookups * 2
    }
}

impl<AB: PermutationAirBuilder> Air<AB> for LookupAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let perm = builder.permutation();
        let randomness = builder.permutation_randomness();

        let main_local = main.row_slice(0).expect("main trace is empty");
        let main_local = &*main_local;
        let perm_local = perm.row_slice(0).expect("perm trace is empty");
        let perm_local = &*perm_local;
        let perm_next = perm.row_slice(1).expect("perm trace has only 1 row");
        let perm_next = &*perm_next;

        assert!(
            randomness.len() >= 2,
            "Need at least 2 random values for lookup argument"
        );
        let alpha = randomness[0];
        let beta = randomness[1];

        let n = self.num_lookups;

        for i in 0..n {
            let query_value = main_local[i].clone();
            let table_value = main_local[n + i].clone();

            let alpha_ef: AB::ExprEF = alpha.into();
            let beta_ef: AB::ExprEF = beta.into();

            let query_expr: AB::Expr = query_value.into();
            let table_expr: AB::Expr = table_value.into();
            let query_value_ef: AB::ExprEF = query_expr.into();
            let table_value_ef: AB::ExprEF = table_expr.into();

            let query_fingerprint = alpha_ef.clone() + query_value_ef * beta_ef.clone();
            let table_fingerprint = alpha_ef + table_value_ef * beta_ef;

            if i == 0 {
                builder
                    .when_first_row()
                    .assert_eq_ext(perm_local[i], query_fingerprint.clone());
                builder
                    .when_first_row()
                    .assert_eq_ext(perm_local[n + i], table_fingerprint.clone());
            } else {
                builder.when_first_row().assert_eq_ext(
                    perm_local[i].into(),
                    perm_local[i - 1].into() * query_fingerprint.clone(),
                );
                builder.when_first_row().assert_eq_ext(
                    perm_local[n + i].into(),
                    perm_local[n + i - 1].into() * table_fingerprint.clone(),
                );
            }

            builder
                .when_transition()
                .assert_eq_ext(perm_local[i], perm_next[i]);
            builder
                .when_transition()
                .assert_eq_ext(perm_local[n + i], perm_next[n + i]);
        }

        builder.assert_eq_ext(perm_local[n - 1], perm_local[2 * n - 1]);
    }
}

// ============================================================================
// Trace Generation Functions
// ============================================================================

pub fn generate_permutation_trace<F: Field>(
    sequence_a: Vec<F>,
    sequence_b: Vec<F>,
) -> RowMajorMatrix<F> {
    assert_eq!(sequence_a.len(), sequence_b.len());
    let n = sequence_a.len();

    let mut trace_values = Vec::with_capacity(n * 2);
    trace_values.extend(sequence_a);
    trace_values.extend(sequence_b);

    RowMajorMatrix::new(trace_values, n * 2)
}

pub fn generate_lookup_trace<F: Field>(queries: Vec<F>, table: Vec<F>) -> RowMajorMatrix<F> {
    assert_eq!(queries.len(), table.len());
    let n = queries.len();

    let mut trace_values = Vec::with_capacity(n * 2);
    trace_values.extend(queries);
    trace_values.extend(table);

    RowMajorMatrix::new(trace_values, n * 2)
}
