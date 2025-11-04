use core::borrow::Borrow;

use p3_air::{
    Air, AirBuilder, AirBuilderWithLogUp, AirBuilderWithPublicValues, BaseAir, MultiPhaseBaseAir,
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
use p3_uni_stark::{StarkConfig, prove, prove_single_matrix_pcs, verify, verify_single_matrix_pcs};
use rand::SeedableRng;
use rand::rngs::SmallRng;

/// For testing the public values feature
pub struct FibonacciAir {}

impl<F> BaseAir<F> for FibonacciAir {
    fn width(&self) -> usize {
        3
    }
}

impl<F> MultiPhaseBaseAir<F> for FibonacciAir {
    fn aux_width_in_base_field(&self) -> usize {
        12
    }

    fn num_randomness_in_base_field(&self) -> usize {
        4
    }
}

impl<AB: AirBuilderWithPublicValues + AirBuilderWithLogUp> Air<AB> for FibonacciAir
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
        let aux = builder.logup_permutation();

        let pis = builder.public_values();

        // main constraints
        let a = pis[0];
        let b = pis[1];
        let x = pis[2];

        let (local, next) = (
            main.row_slice(0).expect("Matrix is empty?"),
            main.row_slice(1).expect("Matrix only has 1 row?"),
        );
        let local: &MainTraceRow<AB::Var> = (*local).borrow();
        let next: &MainTraceRow<AB::Var> = (*next).borrow();

        let mut when_first_row = builder.when_first_row();

        when_first_row.assert_eq(local.m1.clone(), a);
        when_first_row.assert_eq(local.m2.clone(), b);

        let mut when_transition = builder.when_transition();

        // a' <- b
        when_transition.assert_eq(local.m2.clone(), next.m1.clone());

        // b' <- a + b
        when_transition.assert_eq(local.m1.clone() + local.m2.clone(), next.m2.clone());

        builder.when_last_row().assert_eq(local.m2.clone(), x);

        // aux constraints
        {
            let xi = local.m2.clone().into();
            let yi = local.m3.clone().into();
            let (aux_local, aux_next) = (
                aux.row_slice(0).expect("Matrix is empty?"),
                aux.row_slice(1).expect("Matrix only has 1 row?"),
            );

            let randomnesses = builder.logup_permutation_randomness();
            let r_width =
                <FibonacciAir as MultiPhaseBaseAir<AB::F>>::num_randomness_in_base_field(self);
            assert_eq!(r_width, 4);
            let w = AB::F::from_i8(11); // 11 is the constant term w for BabyBear Ext4

            // t * (r - x_i) == 1
            {
                let r_min_xi = ext_field_sub::<AB>(
                    &randomnesses,
                    &[xi.clone(), AB::Expr::ZERO, AB::Expr::ZERO, AB::Expr::ZERO],
                );

                let t_mul_r_min_xi = ext_field_mul::<AB>(
                    &aux_local[..r_width]
                        .iter()
                        .map(|x| x.clone().into())
                        .collect::<Vec<_>>(),
                    &r_min_xi,
                    &w,
                );

                builder.assert_one(t_mul_r_min_xi[0].clone());
                builder.assert_zero(t_mul_r_min_xi[1].clone());
                builder.assert_zero(t_mul_r_min_xi[2].clone());
                builder.assert_zero(t_mul_r_min_xi[3].clone());
            }
            // w * (r - y_i) == 1
            {
                let r_min_yi = ext_field_sub::<AB>(
                    &randomnesses,
                    &[yi.clone(), AB::Expr::ZERO, AB::Expr::ZERO, AB::Expr::ZERO],
                );

                let w_mul_r_min_yi = ext_field_mul::<AB>(
                    &aux_local[r_width..2 * r_width]
                        .iter()
                        .map(|x| x.clone().into())
                        .collect::<Vec<_>>(),
                    &r_min_yi,
                    &w,
                );

                builder.assert_one(w_mul_r_min_yi[0].clone());
                builder.assert_zero(w_mul_r_min_yi[1].clone());
                builder.assert_zero(w_mul_r_min_yi[2].clone());
                builder.assert_zero(w_mul_r_min_yi[3].clone());
            }

            // running sums
            for i in 0..r_width {
                let ti = aux_local[i].clone().into();
                let wi = aux_local[r_width + i].clone().into();
                let next_ti = aux_next[i].clone().into();
                let next_wi = aux_next[r_width + i].clone().into();
                let running_sum = aux_local[2 * r_width + i].clone().into();
                let next_running_sum = aux_next[2 * r_width + i].clone().into();

                // first row running_sum = ti - wi
                builder
                    .when_first_row()
                    .assert_eq(running_sum.clone(), ti - wi);
                // next_running_sum = running_sum + ti - wi
                builder
                    .when_transition()
                    .assert_eq(next_running_sum, running_sum.clone() + next_ti - next_wi);
                // last row running sum is zero
                builder.when_last_row().assert_zero(running_sum);
            }
        }
    }
}

// Multiplication in BinomialExtensionField<F, W>
// Example: BabyBearExt4 is defined by X^4 = W, where W = 11
// Hardcoded for degree 4 extension field.
// TODO: maybe this already exits somewhere?
fn ext_field_mul<AB: AirBuilder>(a: &[AB::Expr], b: &[AB::Expr], w: &AB::F) -> Vec<AB::Expr>
where
    AB::F: Field,
{
    assert_eq!(a.len(), 4, "Expected degree 4 extension field element");
    assert_eq!(b.len(), 4, "Expected degree 4 extension field element");

    let mut res = vec![
        AB::Expr::ZERO,
        AB::Expr::ZERO,
        AB::Expr::ZERO,
        AB::Expr::ZERO,
    ];

    // Expanding the multiplication:
    // res[0] = a0*b0 + W*(a1*b3 + a2*b2 + a3*b1)
    // res[1] = a0*b1 + a1*b0 + W*(a2*b3 + a3*b2)
    // res[2] = a0*b2 + a1*b1 + a2*b0 + W*a3*b3
    // res[3] = a0*b3 + a1*b2 + a2*b1 + a3*b0
    for i in 0..4 {
        for j in 0..4 {
            let prod = a[i].clone() * b[j].clone();
            if i + j < 4 {
                res[i + j] = res[i + j].clone() + prod;
            } else {
                // i + j >= 4, multiply by W since X^(i+j) = X^(i+j-4) * W
                res[i + j - 4] = res[i + j - 4].clone() + prod * w.clone();
            }
        }
    }

    res
}

fn ext_field_sub<AB: AirBuilder>(a: &[AB::Expr], b: &[AB::Expr]) -> Vec<AB::Expr> {
    assert_eq!(a.len(), 4, "Expected degree 4 extension field element");
    assert_eq!(b.len(), 4, "Expected degree 4 extension field element");
    a.iter()
        .zip(b.iter())
        .map(|(a, b)| a.clone() - b.clone())
        .collect()
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

    let config = MyConfig::new(pcs, challenger);
    let pis = vec![BabyBear::ZERO, BabyBear::ONE, BabyBear::from_u64(x)];

    let proof = prove(&config, &FibonacciAir {}, trace.clone(), &pis);
    verify(&config, &FibonacciAir {}, &proof, &pis).expect("verification failed");

    let proof = prove_single_matrix_pcs(&config, &FibonacciAir {}, trace, &pis);
    verify_single_matrix_pcs(&config, &FibonacciAir {}, &proof, &pis).expect("verification failed");
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

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "assertion `left == right` failed: constraints had nonzero value")]
fn test_incorrect_public_value() {
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
    let config = MyConfig::new(pcs, challenger);
    let pis = vec![
        BabyBear::ZERO,
        BabyBear::ONE,
        BabyBear::from_u32(123_123), // incorrect result
    ];
    prove(&config, &FibonacciAir {}, trace, &pis);
}
