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

/// Fibonacci AIR with 3 columns (a, b, selector) and 1 periodic column
///
/// The periodic column alternates between 0 and 1 on each row: [0, 1, 0, 1, ...]
/// The selector column tracks whether we're on an "even" row (based on the periodic column)
///
/// Constraints:
/// - Boundary: a[0] = pis[0], b[0] = pis[1]
/// - Transition: a' = b, b' = a + b
/// - Selector constraint: selector = periodic[i]  (enforces that selector matches the periodic column)
/// - Final: b[n-1] = pis[2]
pub struct FibonacciPeriodicAir;

impl FibonacciPeriodicAir {
    pub fn new() -> Self {
        Self
    }
}

impl<F: Field, EF: ExtensionField<F>> MidenAir<F, EF> for FibonacciPeriodicAir {
    fn width(&self) -> usize {
        3 // Three columns: a, b, selector
    }

    fn periodic_table(&self) -> Vec<Vec<F>> {
        // One periodic column that alternates between 0 and 1
        vec![vec![F::ZERO, F::ONE]]
    }

    fn eval<AB: MidenAirBuilder<F = F>>(&self, builder: &mut AB) {
        let main = builder.main();
        let pis = builder.public_values();

        // Copy periodic value before using builder mutably
        let periodic_selector = builder.periodic_evals()[0];

        let (local, next) = (
            main.row_slice(0).expect("Matrix is empty?"),
            main.row_slice(1).expect("Matrix only has 1 row?"),
        );
        let local: &FibPeriodicRow<AB::Var> = (*local).borrow();
        let next: &FibPeriodicRow<AB::Var> = (*next).borrow();

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

        // Periodic column constraint:
        // Note: periodic values are in the extension field (ExprEF), so we need to convert
        // Convert selector (Var) to Expr, then to ExprEF

        // 1. selector must equal the periodic value
        let selector_expr: AB::Expr = local.selector.clone().into();
        let selector_ef: AB::ExprEF = selector_expr.into();
        builder.assert_eq_ext(selector_ef.clone(), periodic_selector);

        // 2. verify the alternating pattern
        // If current selector is 0, next must be 1, and vice versa
        let next_selector_expr: AB::Expr = next.selector.clone().into();
        let next_selector_ef: AB::ExprEF = next_selector_expr.into();
        let current_selector_ef = selector_ef.clone();
        builder
            .when_transition()
            .assert_eq_ext(next_selector_ef, AB::ExprEF::ONE - current_selector_ef);

        // Final constraint
        builder.when_last_row().assert_eq(local.b.clone(), b_final);
    }
}

/// Generate a trace for the Fibonacci sequence with periodic selector
pub fn generate_fibonacci_periodic_trace<F: PrimeField64>(
    a: u64,
    b: u64,
    n: usize,
) -> RowMajorMatrix<F> {
    assert!(n.is_power_of_two());

    let mut trace = RowMajorMatrix::new(F::zero_vec(n * 3), 3);

    let (prefix, rows, suffix) = unsafe { trace.values.align_to_mut::<FibPeriodicRow<F>>() };
    assert!(prefix.is_empty(), "Alignment should match");
    assert!(suffix.is_empty(), "Alignment should match");
    assert_eq!(rows.len(), n);

    rows[0] = FibPeriodicRow::new(F::from_u64(a), F::from_u64(b), F::ZERO);

    for i in 1..n {
        rows[i].a = rows[i - 1].b;
        rows[i].b = rows[i - 1].a + rows[i - 1].b;
        // Selector alternates between 0 and 1 (matches periodic column)
        rows[i].selector = if i % 2 == 0 { F::ZERO } else { F::ONE };
    }

    trace
}

// A row in the Fibonacci trace with periodic selector
#[repr(C)]
pub struct FibPeriodicRow<F> {
    pub a: F,
    pub b: F,
    pub selector: F,
}

impl<F> FibPeriodicRow<F> {
    const fn new(a: F, b: F, selector: F) -> Self {
        Self { a, b, selector }
    }
}

impl<F> Borrow<FibPeriodicRow<F>> for [F] {
    fn borrow(&self) -> &FibPeriodicRow<F> {
        debug_assert_eq!(self.len(), 3);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<FibPeriodicRow<F>>() };
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

/// Test Fibonacci with periodic column
fn test_fibonacci_periodic_impl(a: u64, b: u64, n: usize, x: u64, log_final_poly_len: usize) {
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let trace = generate_fibonacci_periodic_trace::<Val>(a, b, n);
    let fri_params = create_test_fri_params(challenge_mmcs, log_final_poly_len);
    let pcs = Pcs::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm);

    let config = MyConfig::new(pcs, challenger);
    let pis = vec![
        Goldilocks::from_u64(a),
        Goldilocks::from_u64(b),
        Goldilocks::from_u64(x),
    ];

    let air = FibonacciPeriodicAir::new();

    let proof = prove(&config, &air, &trace, &pis);
    verify(&config, &air, &proof, &pis).expect("verification failed");
}

#[test]
fn test_fibonacci_periodic_small() {
    // Starting with (0, 1), after 8 steps: 0, 1, 1, 2, 3, 5, 8, 13, 21
    // b values: 1, 1, 2, 3, 5, 8, 13, 21
    test_fibonacci_periodic_impl(0, 1, 1 << 3, 21, 2);
}

#[test]
fn test_fibonacci_periodic_medium() {
    // Test with 16 rows
    test_fibonacci_periodic_impl(0, 1, 1 << 4, 987, 3);
}

#[test]
fn test_fibonacci_periodic_large() {
    // Test with 32 rows
    test_fibonacci_periodic_impl(0, 1, 1 << 5, 2178309, 3);
}

#[test]
fn test_fibonacci_periodic_custom_start() {
    // Starting with (2, 3): 8 rows = 7 transitions
    // Row 0: (2,3), Row 7: (55,89)
    test_fibonacci_periodic_impl(2, 3, 1 << 3, 89, 2);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "Extension field constraint failed")]
fn test_fibonacci_periodic_wrong_selector() {
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();

    // Generate trace with incorrect selector values
    let n = 1 << 3;
    let mut trace = RowMajorMatrix::new(Goldilocks::zero_vec(n * 3), 3);
    let (prefix, rows, suffix) =
        unsafe { trace.values.align_to_mut::<FibPeriodicRow<Goldilocks>>() };
    assert!(prefix.is_empty(), "Alignment should match");
    assert!(suffix.is_empty(), "Alignment should match");

    rows[0] = FibPeriodicRow::new(Goldilocks::ZERO, Goldilocks::ONE, Goldilocks::ZERO);
    for i in 1..n {
        rows[i].a = rows[i - 1].b;
        rows[i].b = rows[i - 1].a + rows[i - 1].b;
        // WRONG: set all selectors to 0 instead of alternating
        rows[i].selector = Goldilocks::ZERO;
    }

    let fri_params = create_test_fri_params(challenge_mmcs, 2);
    let pcs = Pcs::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm);
    let config = MyConfig::new(pcs, challenger);
    let pis = vec![Goldilocks::ZERO, Goldilocks::ONE, Goldilocks::from_u32(21)];

    let air = FibonacciPeriodicAir::new();

    // This should fail because selector[1] = 0, but periodic[1] = 1
    let proof = prove(&config, &air, &trace, &pis);
    verify(&config, &air, &proof, &pis).expect("verification failed");
}
