// // This example demonstrates how to use the miden-air MidenAir trait to create
// // a simple Fibonacci STARK proof.
// //
// // The MidenAir trait provides a simplified interface compared to implementing
// // the individual p3-air traits (BaseAir, Air, etc.) separately.

// use miden_air::{
//     Air, AirBuilder, BaseAir, BaseAirWithAuxTrace, BaseAirWithPublicValues, ExtensionField, Field,
//     Matrix, MidenAir, MidenAirBuilder, PrimeCharacteristicRing, RowMajorMatrix,
// };
// use p3_baby_bear::BabyBear;
// use p3_challenger::SerializingChallenger32;
// use p3_commit::ExtensionMmcs;
// use p3_dft::Radix2DitParallel;
// use p3_field::PrimeField64;
// use p3_field::extension::BinomialExtensionField;
// use p3_fri::{TwoAdicFriPcs, create_benchmark_fri_params};
// use p3_keccak::{Keccak256Hash, KeccakF};
// use p3_matrix::dense::RowMajorMatrix as RowMatrix;
// use p3_merkle_tree::MerkleTreeMmcs;
// use p3_symmetric::{CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher};
// use p3_uni_stark::{StarkConfig, prove, verify};

// /// Fibonacci AIR (Algebraic Intermediate Representation)
// ///
// /// This AIR proves computation of the Fibonacci sequence: f(n+2) = f(n+1) + f(n)
// ///
// /// The execution trace has 2 columns:
// /// - Column 0: f(n)
// /// - Column 1: f(n+1)
// ///
// /// Constraints:
// /// 1. Boundary constraint: First row must have f(0) = 0, f(1) = 1
// /// 2. Transition constraint: For each row, verify f(n+2) = f(n+1) + f(n)
// ///    This is expressed as: next_row[1] = current_row[1] + current_row[0]
// #[derive(Debug, Clone)]
// pub struct FibonacciAir {
//     /// Number of Fibonacci steps to compute
//     pub num_steps: usize,
// }

// impl FibonacciAir {
//     pub fn new(num_steps: usize) -> Self {
//         assert!(
//             num_steps.is_power_of_two(),
//             "num_steps must be a power of 2"
//         );
//         Self { num_steps }
//     }

//     /// Generate the execution trace for the Fibonacci sequence
//     ///
//     /// Returns a matrix where:
//     /// - Row i, Column 0 contains f(i)
//     /// - Row i, Column 1 contains f(i+1)
//     pub fn generate_trace<F: Field>(&self) -> RowMajorMatrix<F> {
//         let mut trace = RowMatrix::new(
//             vec![F::ZERO; self.num_steps * 2],
//             2, // width: 2 columns
//         );

//         // Initialize first row: f(0) = 0, f(1) = 1
//         trace.row_mut(0)[0] = F::ZERO;
//         trace.row_mut(0)[1] = F::ONE;

//         // Compute subsequent rows using the Fibonacci recurrence
//         for i in 1..self.num_steps {
//             // Get the previous values before mutating
//             let (f_n, f_n_plus_1) = {
//                 let prev_row = trace.row_slice(i - 1).unwrap();
//                 (prev_row[0], prev_row[1])
//             };
//             let f_n_plus_2 = f_n + f_n_plus_1;

//             trace.row_mut(i)[0] = f_n_plus_1;
//             trace.row_mut(i)[1] = f_n_plus_2;
//         }

//         trace
//     }
// }

// // Implement MidenAir trait - this is the main trait that defines the AIR constraints
// impl<F: Field, EF: ExtensionField<F>> MidenAir<F, EF> for FibonacciAir {
//     fn width(&self) -> usize {
//         2 // Two columns: f(n) and f(n+1)
//     }

//     fn eval<AB: MidenAirBuilder<F = F, EF = EF>>(&self, builder: &mut AB) {
//         // Get access to the main trace columns
//         let main = builder.main();
//         let local = main.row_slice(0).unwrap();
//         let next = main.row_slice(1).unwrap();

//         // Column indices
//         let f_n = local[0].clone();
//         let f_n_plus_1 = local[1].clone();
//         let f_n_plus_2 = next[1].clone();

//         // Boundary constraints: enforce f(0) = 0 and f(1) = 1 on the first row
//         builder.when_first_row().assert_zero(f_n.clone());
//         builder.when_first_row().assert_one(f_n_plus_1.clone());

//         // Transition constraints: enforce f(n+2) = f(n+1) + f(n) on all rows except the last
//         // This is checked by verifying: next[1] - (local[1] + local[0]) = 0
//         builder
//             .when_transition()
//             .assert_eq(f_n_plus_2, f_n_plus_1 + f_n);
//     }
// }

// // We need to manually implement the p3-air traits to bridge to the concrete builder types.
// // While MidenAir provides a convenient interface, the STARK prover/verifier still need
// // the standard p3-air traits implemented for the concrete builder types.

// impl<F: Field> BaseAir<F> for FibonacciAir {
//     fn width(&self) -> usize {
//         2
//     }
// }

// impl<F: Field> BaseAirWithPublicValues<F> for FibonacciAir {
//     fn num_public_values(&self) -> usize {
//         0
//     }
// }

// impl<F: Field, EF: ExtensionField<F>> BaseAirWithAuxTrace<F, EF> for FibonacciAir {}

// // Implement Air trait for all AirBuilder types by delegating to MidenAir::eval.
// // This is required because the STARK prover uses various builder types
// // (SymbolicAirBuilder, ProverConstraintFolder, VerifierConstraintFolder, etc.)
// impl<AB: AirBuilder> Air<AB> for FibonacciAir
// where
//     AB::F: Field,
//     AB::Expr: From<AB::F>,
// {
//     fn eval(&self, builder: &mut AB) {
//         let main = builder.main();
//         let local = main.row_slice(0).unwrap();
//         let next = main.row_slice(1).unwrap();

//         let f_n = local[0].clone();
//         let f_n_plus_1 = local[1].clone();
//         let f_n_plus_2 = next[1].clone();

//         // Boundary constraints
//         let is_first_row = builder.is_first_row();
//         builder.assert_zero(is_first_row.clone() * f_n.clone());
//         builder.assert_zero(is_first_row * (f_n_plus_1.clone() - AB::Expr::ONE));

//         // Transition constraint
//         let is_transition = builder.is_transition();
//         builder.assert_zero(is_transition * (f_n_plus_2 - f_n_plus_1 - f_n));
//     }
// }

// fn main() {
//     println!("=== Fibonacci STARK Proof Example using MidenAir ===\n");
//     println!("This example demonstrates the MidenAir trait, which provides");
//     println!("a simplified interface for defining AIR constraints.\n");

//     // Configuration
//     type F = BabyBear;
//     type EF = BinomialExtensionField<F, 4>;
//     const NUM_STEPS: usize = 1 << 8; // 256 steps

//     println!("Computing Fibonacci sequence with {} steps\n", NUM_STEPS);

//     // Create the Fibonacci AIR
//     let fib_air = FibonacciAir::new(NUM_STEPS);

//     // Generate the execution trace
//     println!("[1/4] Generating execution trace...");
//     let trace = fib_air.generate_trace::<F>();

//     // Print some values from the trace to verify correctness
//     println!("\n  First few Fibonacci numbers:");
//     for i in 0..10 {
//         let row = trace.row_slice(i).unwrap();
//         println!("    f({}) = {}", i, row[0].as_canonical_u64());
//     }
//     println!("    ...");
//     let last_idx = NUM_STEPS - 1;
//     let last_row = trace.row_slice(last_idx).unwrap();
//     println!(
//         "    f({}) = {} (mod {})",
//         last_idx,
//         last_row[0].as_canonical_u64(),
//         F::ORDER_U64
//     );

//     // Set up the STARK proving system
//     println!("\n[2/4] Setting up STARK configuration...");

//     // Create a Keccak-based Merkle tree for polynomial commitments
//     let keccak_perm = KeccakF {};
//     let u64_hash = PaddingFreeSponge::<_, 25, 17, 4>::new(keccak_perm);
//     let field_hash = SerializingHasher::new(u64_hash);
//     let compress = CompressionFunctionFromHasher::<_, 2, 4>::new(u64_hash);

//     // Use the correct vector length for Keccak (matches p3_keccak::VECTOR_LEN)
//     const KECCAK_VECTOR_LEN: usize = 8;
//     let val_mmcs = MerkleTreeMmcs::<[F; KECCAK_VECTOR_LEN], [u64; KECCAK_VECTOR_LEN], _, _, 4>::new(
//         field_hash, compress,
//     );

//     // Create FRI PCS (Polynomial Commitment Scheme)
//     let challenge_mmcs = ExtensionMmcs::<F, EF, _>::new(val_mmcs.clone());
//     let fri_params = create_benchmark_fri_params(challenge_mmcs);
//     let dft = Radix2DitParallel::default();
//     let pcs = TwoAdicFriPcs::new(dft, val_mmcs, fri_params);

//     // Create challenger for Fiat-Shamir
//     let challenger = SerializingChallenger32::from_hasher(vec![], Keccak256Hash {});

//     // Create the STARK configuration
//     let config = StarkConfig::new(pcs, challenger);

//     // Generate the proof
//     println!("[3/4] Generating STARK proof...");
//     let proof = prove(&config, &fib_air, &trace, &[]);
//     println!("  Proof generated successfully!");

//     // Verify the proof
//     println!("[4/4] Verifying proof...");
//     let result = verify(&config, &fib_air, &proof, &[]);

//     match result {
//         Ok(_) => {
//             println!("\n✓ Proof verified successfully!\n");
//             println!("Summary:");
//             println!("  - Used MidenAir trait to define constraints with a clean API");
//             println!(
//                 "  - Proved correct computation of {} Fibonacci numbers",
//                 NUM_STEPS
//             );
//             println!("  - Used STARK (Scalable Transparent ARgument of Knowledge)");
//             println!("  - Verification is fast and transparent (no trusted setup)\n");
//         }
//         Err(e) => {
//             println!("\n✗ Proof verification failed: {:?}\n", e);
//             std::process::exit(1);
//         }
//     }
// }

fn main() {}
