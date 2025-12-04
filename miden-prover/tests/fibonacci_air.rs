use core::borrow::Borrow;

use miden_air::{MidenAir, MidenAirBuilder};
use miden_prover::{StarkConfig, prove, verify};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing, PrimeField64};
use p3_fri::{TwoAdicFriPcs, create_test_fri_params};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::SeedableRng;
use rand::rngs::SmallRng;

/// Fibonacci AIR with 2 columns (a, b)
/// Constraints:
/// - Boundary: a[0] = pis[0], b[0] = pis[1]
/// - Transition: a' = b, b' = a + b
/// - Final: b[n-1] = pis[2]
pub struct FibonacciAir;

impl FibonacciAir {
    pub const fn new() -> Self {
        Self
    }
}

impl Default for FibonacciAir {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Field, EF: ExtensionField<F>> MidenAir<F, EF> for FibonacciAir {
    fn width(&self) -> usize {
        2 // Two columns: a and b
    }

    fn eval<AB: MidenAirBuilder<F = F>>(&self, builder: &mut AB) {
        let main = builder.main();
        let pis = builder.public_values();
        let (local, next) = (
            main.row_slice(0).expect("Matrix is empty?"),
            main.row_slice(1).expect("Matrix only has 1 row?"),
        );
        let local: &FibRow<AB::Var> = (*local).borrow();
        let next: &FibRow<AB::Var> = (*next).borrow();

        let a_init = pis[0];
        let b_init = pis[1];
        let b_final = pis[2];

        // Boundary constraints (only enforced on first row)
        let mut when_first_row = builder.when_first_row();
        when_first_row.assert_eq(local.a.clone(), a_init);
        when_first_row.assert_eq(local.b.clone(), b_init);

        // Transition constraints (enforced on all rows except last)
        let mut when_transition = builder.when_transition();
        when_transition.assert_eq(next.a.clone(), local.b.clone());
        when_transition.assert_eq(next.b.clone(), local.a.clone() + local.b.clone());

        // Final constraint
        builder.when_last_row().assert_eq(local.b.clone(), b_final);
    }
}

/// Generate a trace for the Fibonacci sequence
pub fn generate_fibonacci_trace<F: PrimeField64>(a: u64, b: u64, n: usize) -> RowMajorMatrix<F> {
    assert!(n.is_power_of_two());

    let mut trace = RowMajorMatrix::new(F::zero_vec(n * 2), 2);

    let (prefix, rows, suffix) = unsafe { trace.values.align_to_mut::<FibRow<F>>() };
    assert!(prefix.is_empty(), "Alignment should match");
    assert!(suffix.is_empty(), "Alignment should match");
    assert_eq!(rows.len(), n);

    rows[0] = FibRow::new(F::from_u64(a), F::from_u64(b));

    for i in 1..n {
        rows[i].a = rows[i - 1].b;
        rows[i].b = rows[i - 1].a + rows[i - 1].b;
    }

    trace
}

// A row in the Fibonacci trace
pub struct FibRow<F> {
    pub a: F,
    pub b: F,
}

impl<F> FibRow<F> {
    const fn new(a: F, b: F) -> Self {
        Self { a, b }
    }
}

impl<F> Borrow<FibRow<F>> for [F] {
    fn borrow(&self) -> &FibRow<F> {
        debug_assert_eq!(self.len(), 2);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<FibRow<F>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

type Val = Goldilocks;
type Perm = Poseidon2Goldilocks<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs =
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 8>;
type Challenge = BinomialExtensionField<Val, 2>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type Dft = Radix2DitParallel<Val>;
type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;

/// n-th Fibonacci number starting from (a, b) expected to be x
fn test_fibonacci_impl(a: u64, b: u64, n: usize, x: u64, log_final_poly_len: usize) {
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let trace = generate_fibonacci_trace::<Val>(a, b, n);
    let fri_params = create_test_fri_params(challenge_mmcs, log_final_poly_len);
    let pcs = Pcs::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm);

    let config = MyConfig::new(pcs, challenger);
    let pis = vec![
        Goldilocks::from_u64(a),
        Goldilocks::from_u64(b),
        Goldilocks::from_u64(x),
    ];

    let air = FibonacciAir::new();

    let proof = prove(&config, &air, &trace, &pis);
    verify(&config, &air, &proof, &pis).expect("verification failed");
}

#[test]
fn test_one_row_trace() {
    // Starting with (0, 1), after 1 step we have b = 1
    // Need to set log_final_poly_len to ensure log_min_height > params.log_final_poly_len + params.log_blowup
    test_fibonacci_impl(0, 1, 1, 1, 0);
}

#[test]
fn test_fibonacci_classic() {
    // Starting with (0, 1), after 8 steps: 0, 1, 1, 2, 3, 5, 8, 13, 21
    // b values: 1, 1, 2, 3, 5, 8, 13, 21
    test_fibonacci_impl(0, 1, 1 << 3, 21, 2);
}

#[test]
fn test_fibonacci_larger() {
    // Test with more rows (32 rows = 31 transitions)
    test_fibonacci_impl(0, 1, 1 << 5, 2178309, 3);
}

#[test]
fn test_fibonacci_custom_start() {
    // Starting with (2, 3): 8 rows = 7 transitions
    // Row 0: (2,3), Row 7: (55,89)
    test_fibonacci_impl(2, 3, 1 << 3, 89, 2);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "Constraint failed")]
fn test_incorrect_fibonacci_value() {
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let fri_params = create_test_fri_params(challenge_mmcs, 1);
    let trace = generate_fibonacci_trace::<Val>(0, 1, 1 << 3);
    let pcs = Pcs::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm);

    let config = MyConfig::new(pcs, challenger);
    let pis = vec![
        Goldilocks::ZERO,
        Goldilocks::ONE,
        Goldilocks::from_u64(999), // incorrect result
    ];

    let air = FibonacciAir::new();
    let proof = prove(&config, &air, &trace, &pis);
    verify(&config, &air, &proof, &pis).expect("verification failed");
}
