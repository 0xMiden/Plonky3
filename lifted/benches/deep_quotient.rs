use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::Mmcs;
use p3_field::extension::BinomialExtensionField;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_lifted::deep::prover::DeepPoly;
use p3_lifted::deep::{QuotientOpening, SinglePointQuotient};
use p3_lifted::merkle_tree::{Lifting, MerkleTreeLmcs};
use p3_lifted::utils::bit_reversed_coset_points;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_symmetric::{CryptographicPermutation, PaddingFreeSponge, TruncatedPermutation};
use rand::distr::StandardUniform;
use rand::prelude::Distribution;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

const WIDTH: usize = 16;
const RATE: usize = 8;
const DIGEST: usize = 8;

/// Generate benchmark matrix groups from relative specs.
///
/// Each group represents a separate commitment with matrices of varying heights.
/// Specs are (offset_from_max, width) where log_degree = log_d_max - offset.
fn generate_matrix_groups<F: Field>(
    relative_specs: &[Vec<(usize, usize)>],
    log_d_max: usize,
    log_blowup: usize,
) -> Vec<Vec<RowMajorMatrix<F>>>
where
    StandardUniform: Distribution<F>,
{
    let rng = &mut SmallRng::seed_from_u64(42);
    let n = (1 << log_d_max) << log_blowup;

    relative_specs
        .iter()
        .map(|group_specs| {
            let mut matrices: Vec<RowMajorMatrix<F>> = group_specs
                .iter()
                .map(|&(offset, width)| {
                    let log_d = log_d_max - offset;
                    let log_scaling = log_d_max - log_d;
                    let height = n >> log_scaling;
                    RowMajorMatrix::rand(rng, height, width)
                })
                .collect();
            // Sort by ascending height (required by DEEP quotient)
            matrices.sort_by_key(|m| m.height());
            matrices
        })
        .collect()
}

/// Run benchmarks for BabyBear field.
fn bench_babybear(c: &mut Criterion) {
    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type P = <F as Field>::Packing;
    type Perm = Poseidon2BabyBear<WIDTH>;
    type Sponge = PaddingFreeSponge<Perm, WIDTH, RATE, DIGEST>;
    type Compressor = TruncatedPermutation<Perm, 2, DIGEST, WIDTH>;
    type Lmcs = MerkleTreeLmcs<P, P, Sponge, Compressor, WIDTH, DIGEST>;

    let perm = Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(123));
    let sponge = PaddingFreeSponge::<_, WIDTH, RATE, DIGEST>::new(perm.clone());
    let compress = TruncatedPermutation::<_, 2, DIGEST, WIDTH>::new(perm.clone());
    let lmcs: Lmcs = MerkleTreeLmcs::new(sponge, compress, Lifting::Upsample);

    run_benchmarks::<F, EF, Perm, Lmcs>(c, "babybear", lmcs, perm);
}

/// Run benchmarks for Goldilocks field.
fn bench_goldilocks(c: &mut Criterion) {
    type F = Goldilocks;
    type EF = BinomialExtensionField<F, 2>;
    type P = <F as Field>::Packing;
    type Perm = Poseidon2Goldilocks<WIDTH>;
    type Sponge = PaddingFreeSponge<Perm, WIDTH, RATE, DIGEST>;
    type Compressor = TruncatedPermutation<Perm, 2, DIGEST, WIDTH>;
    type Lmcs = MerkleTreeLmcs<P, P, Sponge, Compressor, WIDTH, DIGEST>;

    let perm = Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(123));
    let sponge = PaddingFreeSponge::<_, WIDTH, RATE, DIGEST>::new(perm.clone());
    let compress = TruncatedPermutation::<_, 2, DIGEST, WIDTH>::new(perm.clone());
    let lmcs: Lmcs = MerkleTreeLmcs::new(sponge, compress, Lifting::Upsample);

    run_benchmarks::<F, EF, Perm, Lmcs>(c, "goldilocks", lmcs, perm);
}

/// Run the actual benchmarks for a given field configuration.
#[allow(clippy::needless_pass_by_value)]
fn run_benchmarks<F, EF, Perm, Lmcs>(c: &mut Criterion, field_name: &str, lmcs: Lmcs, perm: Perm)
where
    F: TwoAdicField + p3_field::PrimeField64,
    EF: ExtensionField<F> + TwoAdicField,
    Perm: CryptographicPermutation<[F; WIDTH]> + Clone,
    Lmcs: Mmcs<F>,
    StandardUniform: Distribution<F> + Distribution<EF>,
{
    // Configurable parameters
    let log_d_max: usize = 20;
    let log_blowup: usize = 3;
    let alignment: usize = RATE;

    // Relative specs: (offset_from_max, width) where log_degree = log_d_max - offset
    // Each inner vec is a separate commitment group
    let relative_specs: Vec<Vec<(usize, usize)>> = vec![
        vec![(4, 10), (2, 100), (0, 50)],
        vec![(4, 8), (2, 20), (0, 20)],
        vec![(0, 16)],
    ];

    let mut group = c.benchmark_group("deep_quotient");
    group.sample_size(10);

    // Setup data (this is slow, do it once)
    println!(
        "Setting up benchmark data for {field_name} (log_d_max={log_d_max}, log_blowup={log_blowup})..."
    );
    let matrix_groups = generate_matrix_groups::<F>(&relative_specs, log_d_max, log_blowup);

    // Commit each group separately
    let committed: Vec<_> = matrix_groups
        .iter()
        .map(|matrices| lmcs.commit(matrices.clone()))
        .collect();
    let prover_data: Vec<_> = committed.iter().map(|(_, pd)| pd).collect();

    let log_n = log_d_max + log_blowup;
    let coset_points = bit_reversed_coset_points::<F>(log_n);
    let n = coset_points.len();
    let total_matrices: usize = matrix_groups.iter().map(|g| g.len()).sum();
    println!(
        "Data ready for {field_name}: n={n}, {} groups, {} total matrices",
        matrix_groups.len(),
        total_matrices
    );

    // Benchmark 1: batch_eval_lifted (includes SinglePointQuotient::new)
    let matrices_groups: Vec<Vec<&RowMajorMatrix<F>>> =
        matrix_groups.iter().map(|g| g.iter().collect()).collect();
    group.bench_function(format!("{field_name}/batch_eval_lifted"), |b| {
        let mut rng = SmallRng::seed_from_u64(456);
        b.iter(|| {
            let z: EF = rng.sample(StandardUniform);
            let quotient = SinglePointQuotient::<F, EF>::new(z, &coset_points);
            black_box(quotient.batch_eval_lifted(&matrices_groups, &coset_points, log_blowup));
        });
    });

    // Benchmark 2: DeepPoly::new (pre-compute quotients and evals, not timed)
    let mut rng = SmallRng::seed_from_u64(123);
    let z1: EF = rng.sample(StandardUniform);
    let z2: EF = rng.sample(StandardUniform);

    let q1 = SinglePointQuotient::<F, EF>::new(z1, &coset_points);
    let q2 = SinglePointQuotient::<F, EF>::new(z2, &coset_points);
    let evals1 = q1.batch_eval_lifted(&matrices_groups, &coset_points, log_blowup);
    let evals2 = q2.batch_eval_lifted(&matrices_groups, &coset_points, log_blowup);

    // Create a base challenger state (outside the benchmark loop)
    let base_challenger = DuplexChallenger::<F, Perm, WIDTH, RATE>::new(perm);

    group.bench_function(format!("{field_name}/deep_poly_new"), |b| {
        b.iter(|| {
            // Clone challenger for each iteration to ensure consistent state
            let mut challenger = base_challenger.clone();

            let openings: Vec<QuotientOpening<'_, F, EF>> = vec![
                QuotientOpening {
                    quotient: &q1,
                    evals: evals1.clone(),
                },
                QuotientOpening {
                    quotient: &q2,
                    evals: evals2.clone(),
                },
            ];
            black_box(DeepPoly::new(
                &lmcs,
                &openings,
                prover_data.clone(),
                &mut challenger,
                alignment,
            ));
        });
    });

    group.finish();
}

fn bench_deep_quotient(c: &mut Criterion) {
    bench_babybear(c);
    bench_goldilocks(c);
}

criterion_group!(benches, bench_deep_quotient);
criterion_main!(benches);
