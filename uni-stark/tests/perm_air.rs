use core::borrow::Borrow;

use p3_air::{
    Air, AirBuilder, AirBuilderWithPublicValues, BaseAir, ExtensionBuilder, PermutationAirBuilder,
};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
use p3_fri::{TwoAdicFriPcs, create_test_fri_params};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::{StarkConfig, StarkGenericConfig, prove, verify};
use rand::SeedableRng;
use rand::rngs::SmallRng;

/// An Air that enforce Fibonacci sequence and permutations.
pub struct FibPermAir {}

impl<F> BaseAir<F> for FibPermAir {
    fn width(&self) -> usize {
        3
    }
}

impl<AB: AirBuilderWithPublicValues + PermutationAirBuilder> Air<AB> for FibPermAir
where
    AB::F: Field,
{
    fn eval(&self, builder: &mut AB) {
        // | m1 | m2 | m3 | a1      | a2      | a3 |
        // | 0  | 1  | 8  | 1/(r-1) | 1/(r-8) | .. |
        // | 1  | 1  | 5  | 1/(r-1) | 1/(r-5) | .. |
        // | 1  | 2  | 3  | 1/(r-2) | 1/(r-3) | .. |
        // | 2  | 3  | 2  | 1/(r-3) | 1/(r-2) | .. |
        // | 3  | 5  | 1  | 1/(r-5) | 1/(r-1) | .. |
        // | 5  | 8  | 1  | 1/(r-8) | 1/(r-1) | .. |

        let main = builder.main();
        let _ = builder.permutation();

        let pis = builder.public_values();
        let (local, next) = (
            main.row_slice(0).expect("Matrix is empty?"),
            main.row_slice(1).expect("Matrix only has 1 row?"),
        );
        let local: &MainTraceRow<AB::Var> = (*local).borrow();
        let next: &MainTraceRow<AB::Var> = (*next).borrow();

        // main constraints
        {
            let a = pis[0];
            let b = pis[1];
            let x = pis[2];

            let mut when_first_row = builder.when_first_row();

            when_first_row.assert_eq(local.m1.clone(), a);
            when_first_row.assert_eq(local.m2.clone(), b);

            let mut when_transition = builder.when_transition();

            // a' <- b
            when_transition.assert_eq(local.m2.clone(), next.m1.clone());

            // b' <- a + b
            when_transition.assert_eq(local.m1.clone() + local.m2.clone(), next.m2.clone());

            builder.when_last_row().assert_eq(local.m2.clone(), x);
        }
        // aux constraints
        {
            let xi = local.m2.clone();
            let yi = local.m3.clone();

            let aux = builder.permutation();
            let aux_local = aux.row_slice(0).expect("Matrix is empty?");
            let aux_next = aux.row_slice(1).expect("Matrix only has 1 row?");

            let t_i: AB::ExprEF = aux_local[0].clone().into();
            let w_i: AB::ExprEF = aux_local[1].clone().into();
            let s_i: AB::ExprEF = aux_local[2].clone().into();
            let t_next: AB::ExprEF = aux_next[0].clone().into();
            let w_next: AB::ExprEF = aux_next[1].clone().into();
            let s_next: AB::ExprEF = aux_next[2].clone().into();

            let r_expr = builder.permutation_randomness()[0].into();

            // t * (r - x_i) == 1
            {
                let xi_ext: AB::ExprEF = AB::Expr::from(xi.clone()).into();
                builder.assert_one_ext(t_i.clone() * (r_expr.clone() - xi_ext));
            }
            // w * (r - y_i) == 1
            {
                let yi_ext: AB::ExprEF = AB::Expr::from(yi.clone()).into();
                builder.assert_one_ext(w_i.clone() * (r_expr - yi_ext));
            }

            // running sums
            {
                builder
                    .when_first_row()
                    .assert_eq_ext(s_i.clone(), t_i.clone() - w_i.clone());
                builder
                    .when_transition()
                    .assert_eq_ext(s_next, s_i.clone() + t_next - w_next);
                builder.when_last_row().assert_zero_ext(s_i);
            }
        }
    }
}

pub fn generate_trace_rows<F: PrimeField64>(a: u64, b: u64, n: usize) -> RowMajorMatrix<F> {
    assert!(n.is_power_of_two());

    let mut trace = RowMajorMatrix::new(F::zero_vec(n * 3), 3);

    let (prefix, rows, suffix) = unsafe { trace.values.align_to_mut::<MainTraceRow<F>>() };
    assert!(prefix.is_empty(), "Alignment should match");
    assert!(suffix.is_empty(), "Alignment should match");
    assert_eq!(rows.len(), n);

    rows[0] = MainTraceRow::new(F::from_u64(a), F::from_u64(b), F::ZERO);

    for i in 1..n {
        rows[i].m1 = rows[i - 1].m2;
        rows[i].m2 = rows[i - 1].m1 + rows[i - 1].m2;
    }

    for i in 0..n {
        rows[i].m3 = rows[n - i - 1].m2
    }

    trace
}

// A row in Main trace.
// The first two columns are used for Fibonacci computation.
// The last column is a permutation of the second column.
pub struct MainTraceRow<F> {
    pub m1: F,
    pub m2: F,
    pub m3: F,
}

impl<F> MainTraceRow<F> {
    const fn new(m1: F, m2: F, m3: F) -> Self {
        Self { m1, m2, m3 }
    }
}

impl<F> Borrow<MainTraceRow<F>> for [F] {
    fn borrow(&self) -> &MainTraceRow<F> {
        debug_assert_eq!(self.len(), 3);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<MainTraceRow<F>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

type Val = BabyBear;
type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs =
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 8>;
type Challenge = BinomialExtensionField<Val, 4>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type Dft = Radix2DitParallel<Val>;
type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;

/// n-th Fibonacci number expected to be x
fn test_public_value_impl(n: usize, x: u64, log_final_poly_len: usize) {
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let trace = generate_trace_rows::<Val>(0, 1, n);
    let fri_params = create_test_fri_params(challenge_mmcs, log_final_poly_len);
    let pcs = Pcs::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm);

    let config = MyConfig::new(pcs, challenger).with_aux_builder(
        1,
        12,
        |main: &RowMajorMatrix<Val>, challenges: &[Challenge]| {
            p3_uni_stark::generate_logup_trace::<Challenge, _>(main, &challenges[0])
        },
    );
    let pis = vec![BabyBear::ZERO, BabyBear::ONE, BabyBear::from_u64(x)];

    let proof = prove(&config, &FibPermAir {}, trace, &pis);
    verify(&config, &FibPermAir {}, &proof, &pis).expect("verification failed");
}

#[test]
fn test_one_row_trace() {
    // Need to set log_final_poly_len to ensure log_min_height > params.log_final_poly_len + params.log_blowup
    test_public_value_impl(1, 1, 0);
}

#[test]
fn test_public_value() {
    test_public_value_impl(1 << 3, 21, 2);
}

#[test]
fn test_public_value_deg5() {
    test_public_value_impl_deg5(1 << 3, 21, 2);
}

#[test]
fn test_public_value_deg8() {
    test_public_value_impl_deg8(1 << 3, 21, 2);
}

// Degree-5 extension variant
fn test_public_value_impl_deg5(n: usize, x: u64, log_final_poly_len: usize) {
    use p3_commit::ExtensionMmcs;
    use p3_field::extension::BinomialExtensionField;
    use p3_fri::TwoAdicFriPcs;
    use p3_uni_stark::StarkConfig;

    type Challenge5 = BinomialExtensionField<Val, 5>;
    type ChallengeMmcs5 = ExtensionMmcs<Val, Challenge5, ValMmcs>;
    type Pcs5 = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs5>;
    type MyConfig5 = StarkConfig<Pcs5, Challenge5, Challenger>;

    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs5::new(val_mmcs.clone());
    let dft = Dft::default();
    let trace = generate_trace_rows::<Val>(0, 1, n);
    let fri_params = create_test_fri_params(challenge_mmcs, log_final_poly_len);
    let pcs = Pcs5::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm);

    let aux_width = 3 * 5; // three EF columns flattened
    let config = MyConfig5::new(pcs, challenger).with_aux_builder(
        1,
        aux_width,
        |main: &RowMajorMatrix<Val>, challenges: &[Challenge5]| {
            p3_uni_stark::generate_logup_trace::<Challenge5, _>(main, &challenges[0])
        },
    );
    let pis = vec![BabyBear::ZERO, BabyBear::ONE, BabyBear::from_u64(x)];

    let proof = prove(&config, &FibPermAir {}, trace.clone(), &pis);
    verify(&config, &FibPermAir {}, &proof, &pis).expect("verification failed");
}

// Degree-8 extension variant
fn test_public_value_impl_deg8(n: usize, x: u64, log_final_poly_len: usize) {
    use p3_commit::ExtensionMmcs;
    use p3_field::extension::BinomialExtensionField;
    use p3_fri::TwoAdicFriPcs;
    use p3_uni_stark::StarkConfig;

    type Challenge8 = BinomialExtensionField<Val, 8>;
    type ChallengeMmcs8 = ExtensionMmcs<Val, Challenge8, ValMmcs>;
    type Pcs8 = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs8>;
    type MyConfig8 = StarkConfig<Pcs8, Challenge8, Challenger>;

    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs8::new(val_mmcs.clone());
    let dft = Dft::default();
    let trace = generate_trace_rows::<Val>(0, 1, n);
    let fri_params = create_test_fri_params(challenge_mmcs, log_final_poly_len);
    let pcs = Pcs8::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm);

    let aux_width = 3 * 8; // three EF columns flattened
    let config = MyConfig8::new(pcs, challenger).with_aux_builder(
        1,
        aux_width,
        |main: &RowMajorMatrix<Val>, challenges: &[Challenge8]| {
            p3_uni_stark::generate_logup_trace::<Challenge8, _>(main, &challenges[0])
        },
    );
    let pis = vec![BabyBear::ZERO, BabyBear::ONE, BabyBear::from_u64(x)];

    let proof = prove(&config, &FibPermAir {}, trace.clone(), &pis);
    verify(&config, &FibPermAir {}, &proof, &pis).expect("verification failed");
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "assertion `left == right` failed: constraints had nonzero value")]
fn test_incorrect_public_value() {
    use p3_uni_stark::StarkGenericConfig;

    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let fri_params = create_test_fri_params(challenge_mmcs, 1);
    let trace = generate_trace_rows::<Val>(0, 1, 1 << 3);
    let pcs = Pcs::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm);
    let config = MyConfig::new(pcs, challenger).with_aux_builder(
        1,
        12,
        |main: &RowMajorMatrix<Val>, challenges: &[Challenge]| {
            p3_uni_stark::generate_logup_trace::<Challenge, _>(main, &challenges[0])
        },
    );
    let pis = vec![
        BabyBear::ZERO,
        BabyBear::ONE,
        BabyBear::from_u32(123_123), // incorrect result
    ];
    prove(&config, &FibPermAir {}, trace, &pis);
}
