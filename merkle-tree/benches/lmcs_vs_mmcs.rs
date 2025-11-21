use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_commit::Mmcs;
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_merkle_tree::{Lifting, MerkleTreeLmcs, MerkleTreeMmcs};
use p3_symmetric::{PaddingFreeSponge, StatefulSponge, TruncatedPermutation};
use rand::rngs::SmallRng;
use rand::SeedableRng;

type F = BabyBear;
type P = <F as Field>::Packing;

const WIDTH: usize = 24;
const RATE: usize = 16;
const DIGEST: usize = 8;

type LMcsSponge = StatefulSponge<Poseidon2BabyBear<WIDTH>, WIDTH, DIGEST, RATE>;
type MmcsHash = PaddingFreeSponge<Poseidon2BabyBear<WIDTH>, WIDTH, RATE, DIGEST>;
type Compress = TruncatedPermutation<Poseidon2BabyBear<WIDTH>, 2, DIGEST, WIDTH>;

fn components() -> (LMcsSponge, MmcsHash, Compress) {
    let mut rng = SmallRng::seed_from_u64(2024);
    let perm = Poseidon2BabyBear::<WIDTH>::new_from_rng_128(&mut rng);
    let lmcs_sponge = StatefulSponge::<_, WIDTH, DIGEST, RATE> { p: perm.clone() };
    let mmcs_hasher = PaddingFreeSponge::<_, WIDTH, RATE, DIGEST>::new(perm.clone());
    let compressor = TruncatedPermutation::<_, 2, DIGEST, WIDTH>::new(perm);
    (lmcs_sponge, mmcs_hasher, compressor)
}

fn padded_bytes(dims: &[p3_matrix::Dimensions]) -> u64 {
    let bytes: usize = dims
        .iter()
        .map(|d| d.height * d.width.next_multiple_of(RATE))
        .sum::<usize>()
        * core::mem::size_of::<F>();
    bytes as u64
}

fn make_scenarios() -> Vec<(&'static str, Vec<RowMajorMatrix<F>>)> {
    const ROWS_LARGE: usize = 1 << 15; // 32768
    const WIDTH_COLS: usize = 40;
    let mut rng = SmallRng::seed_from_u64(7);

    vec![
        (
            "1x(2^15 x 40)",
            vec![RowMajorMatrix::<F>::rand(&mut rng, ROWS_LARGE, WIDTH_COLS)],
        ),
        (
            "(2^14,2^15) x 40",
            vec![
                RowMajorMatrix::<F>::rand(&mut rng, ROWS_LARGE / 2, WIDTH_COLS),
                RowMajorMatrix::<F>::rand(&mut rng, ROWS_LARGE, WIDTH_COLS),
            ],
        ),
        (
            "(2^13,2^14,2^15) x 40",
            vec![
                RowMajorMatrix::<F>::rand(&mut rng, ROWS_LARGE / 8, WIDTH_COLS),
                RowMajorMatrix::<F>::rand(&mut rng, ROWS_LARGE / 4, WIDTH_COLS),
                RowMajorMatrix::<F>::rand(&mut rng, ROWS_LARGE / 2, WIDTH_COLS),
                RowMajorMatrix::<F>::rand(&mut rng, ROWS_LARGE, WIDTH_COLS),
            ],
        ),
    ]
}

fn bench_commit(c: &mut Criterion) {
    let (lmcs_sponge, mmcs_hasher, compressor) = components();
    let mut group = c.benchmark_group("LMCS_vs_MMCS_Commit_W24_R16");

    for (label, matrices) in make_scenarios() {
        let dims: Vec<_> = matrices.iter().map(|m| m.dimensions()).collect();
        group.throughput(Throughput::Bytes(padded_bytes(&dims)));

        // LMCS upsample
        let lmcs_u = MerkleTreeLmcs::<P, P, _, _, WIDTH, DIGEST>::new(
            lmcs_sponge.clone(),
            compressor.clone(),
            Lifting::Upsample,
        );
        group.bench_with_input(
            BenchmarkId::new(format!("LMCS/upsample/{label}"), format!("{dims:?}")),
            &matrices,
            |b, mats| b.iter(|| black_box(lmcs_u.commit(mats.clone()))),
        );

        // LMCS cyclic
        let lmcs_c = MerkleTreeLmcs::<P, P, _, _, WIDTH, DIGEST>::new(
            lmcs_sponge.clone(),
            compressor.clone(),
            Lifting::Cyclic,
        );
        group.bench_with_input(
            BenchmarkId::new(format!("LMCS/cyclic/{label}"), format!("{dims:?}")),
            &matrices,
            |b, mats| b.iter(|| black_box(lmcs_c.commit(mats.clone()))),
        );

        // MMCS
        let mmcs = MerkleTreeMmcs::<P, P, _, _, DIGEST>::new(mmcs_hasher.clone(), compressor.clone());
        group.bench_with_input(
            BenchmarkId::new(format!("MMCS/{label}"), format!("{dims:?}")),
            &matrices,
            |b, mats| b.iter(|| black_box(mmcs.commit(mats.clone()))),
        );
    }

    group.finish();
}

fn bench_verify(c: &mut Criterion) {
    let (lmcs_sponge, mmcs_hasher, compressor) = components();
    let mut group = c.benchmark_group("LMCS_vs_MMCS_Verify_W24_R16");

    for (label, matrices) in make_scenarios() {
        let dims: Vec<_> = matrices.iter().map(|m| m.dimensions()).collect();
        group.throughput(Throughput::Bytes(padded_bytes(&dims)));

        // LMCS upsample
        {
            let lmcs = MerkleTreeLmcs::<P, P, _, _, WIDTH, DIGEST>::new(
                lmcs_sponge.clone(),
                compressor.clone(),
                Lifting::Upsample,
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
            group.bench_function(format!("LMCS/upsample/{label}"), |b| {
                let mut k = 0usize;
                b.iter(|| {
                    let idx = indices[k % indices.len()];
                    let opening_ref = (&openings[k % openings.len()]).into();
                    lmcs.verify_batch(&commit, &dims, idx, opening_ref).unwrap();
                    k = k.wrapping_add(1);
                });
            });
        }

        // LMCS cyclic
        {
            let lmcs = MerkleTreeLmcs::<P, P, _, _, WIDTH, DIGEST>::new(
                lmcs_sponge.clone(),
                compressor.clone(),
                Lifting::Cyclic,
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
            group.bench_function(format!("LMCS/cyclic/{label}"), |b| {
                let mut k = 0usize;
                b.iter(|| {
                    let idx = indices[k % indices.len()];
                    let opening_ref = (&openings[k % openings.len()]).into();
                    lmcs.verify_batch(&commit, &dims, idx, opening_ref).unwrap();
                    k = k.wrapping_add(1);
                });
            });
        }

        // MMCS
        {
            let mmcs = MerkleTreeMmcs::<P, P, _, _, DIGEST>::new(
                mmcs_hasher.clone(),
                compressor.clone(),
            );
            let (commit, tree) = mmcs.commit(matrices.clone());
            let final_h = dims
                .iter()
                .map(|d| d.height)
                .max()
                .expect("non-empty");
            let mut indices: Vec<usize> = (0..16).map(|i| i * (final_h / 16)).collect();
            if indices.is_empty() {
                indices.push(0);
            }
            let openings: Vec<_> = indices
                .iter()
                .map(|&idx| mmcs.open_batch(idx, &tree))
                .collect();
            let dims_local = dims.clone();
            group.bench_function(format!("MMCS/{label}"), |b| {
                let mut k = 0usize;
                b.iter(|| {
                    let idx = indices[k % indices.len()];
                    let opening_ref = (&openings[k % openings.len()]).into();
                    mmcs.verify_batch(&commit, &dims_local, idx, opening_ref)
                        .unwrap();
                    k = k.wrapping_add(1);
                });
            });
        }
    }

    group.finish();
}

fn setup_criterion() -> Criterion {
    Criterion::default()
        .without_plots()
        .sample_size(10)
        .measurement_time(Duration::from_secs(15))
        .warm_up_time(Duration::from_secs(3))
}

criterion_group! {
    name = benches;
    config = setup_criterion();
    targets = bench_commit, bench_verify
}
criterion_main!(benches);

