use std::hint::black_box;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::time::Duration;
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::uniform::{build_leaves_cyclic, build_leaves_upsampled};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use p3_matrix::bitrev::BitReversibleMatrix;
use p3_matrix::Matrix;
use p3_util::reverse_slice_index_bits;

type F = BabyBear;
type Packed = <F as Field>::Packing;

const WIDTH: usize = 16;
const RATE: usize = 8;
const DIGEST: usize = 8;

type Sponge = PaddingFreeSponge<Poseidon2BabyBear<WIDTH>, WIDTH, RATE, DIGEST>;

fn poseidon_components() -> (Sponge, TruncatedPermutation<Poseidon2BabyBear<WIDTH>, 2, DIGEST, WIDTH>) {
    let mut rng = SmallRng::seed_from_u64(1);
    let permutation = Poseidon2BabyBear::<WIDTH>::new_from_rng_128(&mut rng);
    let sponge = PaddingFreeSponge::<_, WIDTH, RATE, DIGEST>::new(permutation.clone());
    let compressor = TruncatedPermutation::<_, 2, DIGEST, WIDTH>::new(permutation);
    (sponge, compressor)
}

fn bench_lifted(c: &mut Criterion) {
    let (sponge, _compressor) = poseidon_components();

    // Mirror matrix sizes used elsewhere: large height and non-multiple-of-rate width.
    const ROWS_LARGE: usize = 1 << 15; // 32768
    const WIDTH_COLS: usize = 40;

    let mut rng = SmallRng::seed_from_u64(123);

    // Scenarios: single, two, and three matrices with descending power-of-two heights.
    let scenarios: Vec<(&str, Vec<RowMajorMatrix<F>>)> = vec![
        (
            "1x(2^15 x 135)",
            vec![RowMajorMatrix::<F>::rand(&mut rng, ROWS_LARGE, WIDTH_COLS)],
        ),
        (
            "(2^14,2^15) x 135",
            vec![
                RowMajorMatrix::<F>::rand(&mut rng, ROWS_LARGE / 2, WIDTH_COLS),
                RowMajorMatrix::<F>::rand(&mut rng, ROWS_LARGE, WIDTH_COLS),
            ],
        ),
        (
            "(2^13,2^14,2^15) x 135",
            vec![
                RowMajorMatrix::<F>::rand(&mut rng, 2, WIDTH_COLS),
                RowMajorMatrix::<F>::rand(&mut rng, 8, WIDTH_COLS),
                RowMajorMatrix::<F>::rand(&mut rng, 128, WIDTH_COLS),
                RowMajorMatrix::<F>::rand(&mut rng, ROWS_LARGE / 4, WIDTH_COLS),
                RowMajorMatrix::<F>::rand(&mut rng, ROWS_LARGE / 2, WIDTH_COLS),
                RowMajorMatrix::<F>::rand(&mut rng, ROWS_LARGE, WIDTH_COLS),
            ],
        ),
    ];

    let mut group = c.benchmark_group("UniformLifting");
    // Group inherits timing/sampling from the configured Criterion instance.

    for (label, matrices) in scenarios {
        let dims: Vec<_> = matrices.iter().map(|m| m.dimensions()).collect();

        group.bench_with_input(BenchmarkId::new(format!("cyclic/{label}"), format!("{dims:?}")), &matrices, |b, mats| {
            b.iter(|| {
                let out = build_leaves_cyclic::<Packed, _, _, WIDTH, RATE, DIGEST>(
                    black_box(&mats[..]),
                    black_box(&sponge),
                );
                black_box(out);
            });
        });

        group.bench_with_input(BenchmarkId::new(format!("upsampled/{label}"), format!("{dims:?}")), &matrices, |b, mats| {
            b.iter(|| {
                let out = build_leaves_upsampled::<Packed, _, _, WIDTH, RATE, DIGEST>(
                    black_box(&mats[..]),
                    black_box(&sponge),
                );
                black_box(out);
            });
        });

        let mats_bitrev: Vec<_> = matrices.iter().map(|m| m.as_view().bit_reverse_rows()).collect();
        group.bench_with_input(BenchmarkId::new(format!("bitrev/upsampled/{label}"), format!("{dims:?}")), &matrices, |b, mats| {
            b.iter(|| {
                let mut out = build_leaves_cyclic::<Packed, _, _, WIDTH, RATE, DIGEST>(
                    black_box(&mats_bitrev[..]),
                    black_box(&sponge),
                );
                reverse_slice_index_bits(&mut out);
                black_box(out);
            });
        });
    }

    group.finish();
}

fn setup_criterion() -> Criterion {
    // Configure globally: disable plots, ensure enough time for large inputs, and respect
    // Criterion's minimum sample size without per-group overrides.
    Criterion::default()
        .without_plots()
        .sample_size(10)
        .measurement_time(Duration::from_secs(15))
        .warm_up_time(Duration::from_secs(3))
}

criterion_group!{
    name = benches;
    config = setup_criterion();
    targets = bench_lifted
}
criterion_main!(benches);
