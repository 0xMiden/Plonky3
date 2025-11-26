//! Fibonacci example using the miden-prover framework.
//!
//! This example demonstrates how to:
//! 1. Define a Fibonacci AIR using MidenAir trait
//! 2. Generate a trace for the Fibonacci sequence
//! 3. Generate and verify a STARK proof
//!
//! The Fibonacci AIR has 2 columns and enforces:
//! - Initial conditions: a[0] = 0, b[0] = 1
//! - Transition constraints: a' = b, b' = a + b

use miden_air::{ExtensionField, Matrix, MidenAir, MidenAirBuilder, PrimeCharacteristicRing};
use miden_prover::{StarkConfig, prove, verify};
use p3_baby_bear::BabyBear;
use p3_challenger::{HashChallenger, SerializingChallenger32};
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeField64};
use p3_fri::{TwoAdicFriPcs, create_benchmark_fri_params};
use p3_keccak::Keccak256Hash;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};

/// Fibonacci AIR with 2 columns (a, b)
/// Constraints:
/// - Boundary: a[0] = 0, b[0] = 1
/// - Transition: a' = b, b' = a + b
#[derive(Clone)]
#[allow(dead_code)]
struct FibonacciAir {
    num_rows: usize,
}

impl FibonacciAir {
    fn new(num_rows: usize) -> Self {
        Self { num_rows }
    }
}

impl<F: Field, EF: ExtensionField<F>> MidenAir<F, EF> for FibonacciAir {
    fn width(&self) -> usize {
        2 // Two columns: a and b
    }

    fn eval<AB: MidenAirBuilder<F = F>>(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let next = main.row_slice(1);

        let local = local.unwrap();
        let next = next.unwrap();

        let a = local[0].clone();
        let b = local[1].clone();
        let a_next = next[0].clone();
        let b_next = next[1].clone();

        // Boundary constraints (only enforced on first row)
        builder.when_first_row().assert_zero(a.clone());
        builder.when_first_row().assert_eq(b.clone(), AB::Expr::ONE);

        // Transition constraints (enforced on all rows except last)
        builder.when_transition().assert_eq(a_next, b.clone());
        builder
            .when_transition()
            .assert_eq(b_next, a.clone() + b.clone());
    }
}

/// Generate a trace for the Fibonacci sequence
fn generate_fibonacci_trace<F: Field>(num_rows: usize) -> RowMajorMatrix<F> {
    let mut values = Vec::with_capacity(num_rows * 2);

    let mut a = F::ZERO;
    let mut b = F::ONE;

    for _ in 0..num_rows {
        values.push(a);
        values.push(b);

        let next_a = b;
        let next_b = a + b;
        a = next_a;
        b = next_b;
    }

    RowMajorMatrix::new(values, 2)
}

fn main() {
    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;

    // Configure logging
    println!("Fibonacci STARK Proof Example");
    println!("==============================\n");

    // Number of Fibonacci steps to prove
    let num_rows = 1 << 8; // 256 rows
    println!("Generating Fibonacci trace with {} rows...", num_rows);

    // Generate the trace
    let trace = generate_fibonacci_trace::<Val>(num_rows);
    println!("Trace generated successfully!");
    println!("First few Fibonacci numbers in trace:");
    for i in 0..core::cmp::min(10, num_rows) {
        let a = trace.get(i, 0).unwrap();
        let b = trace.get(i, 1).unwrap();
        println!(
            "  Row {}: a = {}, b = {}",
            i,
            a.as_canonical_u64(),
            b.as_canonical_u64()
        );
    }

    // Setup cryptographic primitives
    type ByteHash = Keccak256Hash;
    type FieldHash = SerializingHasher<ByteHash>;
    let byte_hash = ByteHash {};
    let field_hash = FieldHash::new(ByteHash {});

    type MyCompress = CompressionFunctionFromHasher<ByteHash, 2, 32>;
    let compress = MyCompress::new(byte_hash);

    type ValMmcs = MerkleTreeMmcs<Val, u8, FieldHash, MyCompress, 32>;
    let val_mmcs = ValMmcs::new(field_hash, compress);

    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    type Dft = Radix2DitParallel<Val>;
    let dft = Dft::default();

    type Challenger = SerializingChallenger32<Val, HashChallenger<u8, ByteHash, 32>>;
    let challenger = Challenger::from_hasher(vec![], byte_hash);

    let fri_params = create_benchmark_fri_params(challenge_mmcs);

    type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
    let pcs = Pcs::new(dft, val_mmcs, fri_params);

    type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;
    let config = MyConfig::new(pcs, challenger);

    // Create the AIR
    let air = FibonacciAir::new(num_rows);

    // Generate proof
    println!("\nGenerating STARK proof...");
    let proof = prove(&config, &air, &trace, &[]);
    println!("Proof generated successfully!");

    // Verify proof
    println!("\nVerifying proof...");
    assert!(verify(&config, &air, &proof, &[]).is_ok());
    println!("Verification successful!");
}
