//! Complete working example of STARK proof generation and verification
//!
//! This example demonstrates the full proving pipeline using a simple AIR
//! that checks basic constraints (without PermutationAirBuilder, which requires
//! additional prover infrastructure).
//!
//! This shows:
//! 1. Complete STARK configuration setup
//! 2. Trace generation
//! 3. Proof generation
//! 4. Proof verification
//!
//! Run with: cargo run --example permutation_proof_complete

use std::fmt::Debug;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
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

    println!("=== Complete STARK Proof Example ===\n");

    // Run a simple proof example
    match run_proof_example() {
        Ok(()) => println!("\n✓ All examples completed successfully!"),
        Err(e) => println!("\n✗ Example failed: {:?}", e),
    }
}

fn run_proof_example() -> Result<(), impl Debug> {
    println!("Example: Proving and Verifying a Simple AIR");
    println!("=============================================\n");

    // Create STARK configuration
    let config = create_stark_config();
    println!("✓ STARK configuration created");

    // Create a simple AIR that checks sequence properties
    let air = SimplePermutationAir { num_elements: 8 };
    println!(
        "✓ AIR created with width: {}",
        <SimplePermutationAir as BaseAir<Val>>::width(&air)
    );

    // Generate a valid trace
    let trace = generate_simple_trace();
    println!("✓ Trace generated: {}x{}", trace.height(), trace.width());
    println!(
        "  Trace values: {:?}",
        trace
            .row_slice(0)
            .unwrap()
            .iter()
            .take(8)
            .collect::<Vec<_>>()
    );

    // Generate proof
    println!("\nGenerating proof...");
    let proof = prove(&config, &air, trace.clone(), &vec![]);
    println!("✓ Proof generated successfully!");

    // Estimate proof size
    let proof_size = estimate_proof_size(&proof);
    println!("  Estimated proof size: {} bytes", proof_size);

    // Verify proof
    println!("\nVerifying proof...");
    let result = verify(&config, &air, &proof, &vec![]);

    match result {
        Ok(()) => {
            println!("✓ Proof verified successfully!");
            Ok(())
        }
        Err(e) => {
            println!("✗ Verification failed!");
            Err(e)
        }
    }
}

/// A simple AIR that demonstrates the proof structure
///
/// This AIR checks basic constraints on a sequence of values:
/// - Boundary constraints on first and last rows
/// - Transition constraints between consecutive rows
///
/// While this doesn't use PermutationAirBuilder (which requires additional
/// prover infrastructure), it demonstrates the complete proving pipeline.
#[derive(Debug, Clone)]
pub struct SimplePermutationAir {
    pub num_elements: usize,
}

impl<F: Field> BaseAir<F> for SimplePermutationAir {
    fn width(&self) -> usize {
        self.num_elements * 2
    }
}

impl<AB: AirBuilder> Air<AB> for SimplePermutationAir
where
    AB::F: Field,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let main_local = main.row_slice(0).expect("main trace is empty");
        let main_local = &*main_local;
        let main_next = main.row_slice(1).expect("main trace has only 1 row");
        let main_next = &*main_next;

        let n = self.num_elements;

        // Constraint 1: First element of sequence A equals first element of sequence B
        // This is a simple example constraint
        builder
            .when_first_row()
            .assert_eq(main_local[0].clone(), main_local[n].clone());

        // Constraint 2: Each element equals itself (trivial constraint for demonstration)
        // In a real permutation AIR, you'd have more sophisticated constraints
        for i in 0..n {
            let val = main_local[i].clone();
            builder.assert_eq(val.clone(), val);
        }

        // Constraint 3: Transition constraint - values don't change between rows
        let width = self.num_elements * 2;
        for i in 0..width {
            builder
                .when_transition()
                .assert_eq(main_local[i].clone(), main_next[i].clone());
        }

        // Constraint 4: Sum check - sum of sequence A equals sum of sequence B
        let mut sum_a = AB::Expr::ZERO;
        let mut sum_b = AB::Expr::ZERO;
        for i in 0..n {
            sum_a = sum_a + main_local[i].clone();
            sum_b = sum_b + main_local[n + i].clone();
        }
        builder.assert_eq(sum_a, sum_b);
    }
}

/// Generate a trace that satisfies the AIR constraints
fn generate_simple_trace() -> RowMajorMatrix<Val> {
    let n = 8;

    // Create two sequences with the same sum
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

    // Sequence B has same elements (permuted) - same sum
    let sequence_b = vec![
        Val::from_u32(5), // First element matches A[0] (constraint 1)
        Val::from_u32(7),
        Val::from_u32(1),
        Val::from_u32(3),
        Val::from_u32(2),
        Val::from_u32(13),
        Val::from_u32(9),
        Val::from_u32(11),
    ];

    // Create trace with 8 rows (must be power of 2)
    let mut trace_values = Vec::with_capacity(n * 2 * 8);

    // Repeat the same values for all 8 rows (satisfies transition constraint)
    for _ in 0..8 {
        trace_values.extend_from_slice(&sequence_a);
        trace_values.extend_from_slice(&sequence_b);
    }

    RowMajorMatrix::new(trace_values, n * 2)
}

/// Create a STARK configuration
fn create_stark_config() -> MyConfig {
    let mut rng = SmallRng::seed_from_u64(42);
    let perm = Perm::new_from_rng_128(&mut rng);

    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    let dft = Dft::default();

    // FRI parameters - carefully chosen for security and efficiency
    // For trace size 8, log2(8) = 3
    let fri_params = FriParameters {
        log_blowup: 1,         // Blowup factor: 2x
        log_final_poly_len: 0, // Final polynomial length: 2^0 = 1
        num_queries: 40,       // Number of FRI queries for security
        proof_of_work_bits: 8, // Proof-of-work difficulty
        mmcs: challenge_mmcs,
    };

    let pcs = Pcs::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm);

    MyConfig::new(pcs, challenger)
}

/// Estimate the size of a proof (rough approximation)
fn estimate_proof_size<SC: p3_uni_stark::StarkGenericConfig>(
    proof: &p3_uni_stark::Proof<SC>,
) -> usize {
    // This is a rough estimate
    // In a real system, you would serialize the proof and measure its size
    std::mem::size_of_val(proof)
}
