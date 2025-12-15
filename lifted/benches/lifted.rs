use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_commit::Mmcs;
use p3_field::Field;
use p3_lifted::merkle_tree::{
    Lifting, MerkleTreeLmcs, build_leaf_states_cyclic, build_leaf_states_upsampled,
};
use p3_matrix::Matrix;
use p3_matrix::bitrev::BitReversibleMatrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_util::reverse_slice_index_bits;
use rand::SeedableRng;
use rand::rngs::SmallRng;

type F = BabyBear;
type Packed = <F as Field>::Packing;

// Use a wider Poseidon2 permutation (WIDTH=24) and explicit rate.
const WIDTH: usize = 24;
const RATE: usize = 16;
const DIGEST: usize = 8;

type Sponge = PaddingFreeSponge<Poseidon2BabyBear<WIDTH>, WIDTH, RATE, DIGEST>;

fn poseidon_components() -> (
    Sponge,
    TruncatedPermutation<Poseidon2BabyBear<WIDTH>, 2, DIGEST, WIDTH>,
) {
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

    let mut group = c.benchmark_group("UniformLifting_W24_R16");
    // Group inherits timing/sampling from the configured Criterion instance.

    for (label, matrices) in scenarios {
        let dims: Vec<_> = matrices.iter().map(|m| m.dimensions()).collect();

        let padded_bytes = |ds: &[p3_matrix::Dimensions]| -> u64 {
            let bytes: usize = ds
                .iter()
                .map(|d| d.height * d.width.next_multiple_of(RATE))
                .sum::<usize>()
                * core::mem::size_of::<F>();
            bytes as u64
        };

        group.throughput(Throughput::Bytes(padded_bytes(&dims)));
        group.bench_with_input(
            BenchmarkId::new(format!("cyclic/{label}"), format!("{dims:?}")),
            &matrices,
            |b, mats| {
                b.iter(|| {
                    let out = build_leaf_states_cyclic::<Packed, Packed, _, _, WIDTH, DIGEST>(
                        black_box(&mats[..]),
                        black_box(&sponge),
                    );
                    black_box(out);
                });
            },
        );

        group.throughput(Throughput::Bytes(padded_bytes(&dims)));
        group.bench_with_input(
            BenchmarkId::new(format!("upsampled/{label}"), format!("{dims:?}")),
            &matrices,
            |b, mats| {
                b.iter(|| {
                    let out = build_leaf_states_upsampled::<Packed, Packed, _, _, WIDTH, DIGEST>(
                        black_box(&mats[..]),
                        black_box(&sponge),
                    );
                    black_box(out);
                });
            },
        );

        let mats_bitrev: Vec<_> = matrices
            .iter()
            .map(|m| m.as_view().bit_reverse_rows())
            .collect();
        group.throughput(Throughput::Bytes(padded_bytes(&dims)));
        group.bench_with_input(
            BenchmarkId::new(format!("bitrev/upsampled/{label}"), format!("{dims:?}")),
            &matrices,
            |b, _mats| {
                b.iter(|| {
                    let mut out =
                        build_leaf_states_upsampled::<Packed, Packed, _, _, WIDTH, DIGEST>(
                            black_box(&mats_bitrev[..]),
                            black_box(&sponge),
                        );
                    reverse_slice_index_bits(&mut out);
                    black_box(out);
                });
            },
        );
    }

    group.finish();
}

// Flat (non-macro) benches equivalent to the previous single macro instantiation.
fn lmcs_components() -> (
    PaddingFreeSponge<Poseidon2BabyBear<WIDTH>, WIDTH, RATE, DIGEST>,
    TruncatedPermutation<Poseidon2BabyBear<WIDTH>, 2, DIGEST, WIDTH>,
) {
    let mut rng = SmallRng::seed_from_u64(42);
    let perm = <Poseidon2BabyBear<WIDTH>>::new_from_rng_128(&mut rng);
    let sponge = PaddingFreeSponge::<_, WIDTH, RATE, DIGEST>::new(perm.clone());
    let compressor = TruncatedPermutation::<_, 2, DIGEST, WIDTH>::new(perm);
    (sponge, compressor)
}

fn bench_lmcs_commit(c: &mut Criterion) {
    type FField = BabyBear;
    type Packed = <FField as Field>::Packing;
    let (sponge, compressor) = lmcs_components();

    const ROWS_LARGE: usize = 1 << 15;
    const WIDTH_COLS: usize = 40;

    let mut rng = SmallRng::seed_from_u64(77);
    let scenarios: Vec<(&str, Vec<RowMajorMatrix<FField>>)> = vec![
        (
            "1x(2^15 x 40)",
            vec![RowMajorMatrix::<FField>::rand(
                &mut rng, ROWS_LARGE, WIDTH_COLS,
            )],
        ),
        (
            "(2^14,2^15) x 40",
            vec![
                RowMajorMatrix::<FField>::rand(&mut rng, ROWS_LARGE / 2, WIDTH_COLS),
                RowMajorMatrix::<FField>::rand(&mut rng, ROWS_LARGE, WIDTH_COLS),
            ],
        ),
    ];

    let mut group = c.benchmark_group("LMCS_Commit_bb_p2_w24_r16");
    for (label, matrices) in scenarios {
        let dims: Vec<_> = matrices.iter().map(|m| m.dimensions()).collect();
        let bytes: usize = dims
            .iter()
            .map(|d| d.height * d.width.next_multiple_of(RATE))
            .sum::<usize>()
            * core::mem::size_of::<FField>();
        group.throughput(Throughput::Bytes(bytes as u64));

        for lifting in [Lifting::Upsample, Lifting::Cyclic] {
            let lmcs = MerkleTreeLmcs::<Packed, Packed, _, _, WIDTH, DIGEST>::new(
                sponge.clone(),
                compressor.clone(),
                lifting,
            );
            group.bench_with_input(
                BenchmarkId::new(format!("{:?}/{label}", lifting), format!("{dims:?}")),
                &matrices,
                |b, mats| {
                    b.iter(|| {
                        let _ = lmcs.commit(mats.clone());
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_lmcs_verify(c: &mut Criterion) {
    type FField = BabyBear;
    type Packed = <FField as Field>::Packing;
    let (sponge, compressor) = lmcs_components();

    const ROWS: usize = 1 << 15;
    const COLS: usize = 40;

    let mut rng = SmallRng::seed_from_u64(1234);
    let small = RowMajorMatrix::<FField>::rand(&mut rng, ROWS / 2, COLS);
    let large = RowMajorMatrix::<FField>::rand(&mut rng, ROWS, COLS);
    let dims = vec![small.dimensions(), large.dimensions()];
    let matrices = vec![small, large];

    let mut group = c.benchmark_group("LMCS_Verify_bb_p2_w24_r16");
    let bytes: usize = dims
        .iter()
        .map(|d| d.height * d.width.next_multiple_of(RATE))
        .sum::<usize>()
        * core::mem::size_of::<FField>();
    group.throughput(Throughput::Bytes(bytes as u64));

    for lifting in [Lifting::Upsample, Lifting::Cyclic] {
        let lmcs = MerkleTreeLmcs::<Packed, Packed, _, _, WIDTH, DIGEST>::new(
            sponge.clone(),
            compressor.clone(),
            lifting,
        );
        let (commit, tree) = lmcs.commit(matrices.clone());
        let final_h = dims.last().unwrap().height;
        let mut indices: Vec<usize> = (0..16).map(|i| i * (final_h / 16)).collect();
        if indices.is_empty() {
            indices.push(0);
        }
        let openings: Vec<_> = indices
            .iter()
            .map(|&idx| lmcs.open_batch(idx, &tree))
            .collect();
        let dims_local = dims.clone();

        group.bench_function(format!("{:?}", lifting), |b| {
            let mut k = 0usize;
            b.iter(|| {
                let idx = indices[k % indices.len()];
                let opening_ref = (&openings[k % openings.len()]).into();
                lmcs.verify_batch(&commit, &dims_local, idx, opening_ref)
                    .unwrap();
                k = k.wrapping_add(1);
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

criterion_group! {
    name = benches;
    config = setup_criterion();
    targets = bench_lifted, bench_lmcs_commit, bench_lmcs_verify
}
criterion_main!(benches);
