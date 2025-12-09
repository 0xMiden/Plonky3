use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear, default_babybear_poseidon2_24};
use p3_field::Field;
use p3_keccak::KeccakF;
use p3_lifted::build_leaf_states_upsampled;
use p3_matrix::dense::RowMajorMatrix;
use p3_sha256::Sha256;
use p3_symmetric::{ChainingHasher, PaddingFreeSponge};
use rand::SeedableRng;
use rand::rngs::SmallRng;

type F = BabyBear;
type P = <F as Field>::Packing;

// Poseidon2 over BabyBear (field-native sponge)
const P2_WIDTH: usize = 24;
const P2_RATE: usize = 16;
const P2_DIGEST: usize = 8;
type P2Sponge = PaddingFreeSponge<Poseidon2BabyBear<P2_WIDTH>, P2_WIDTH, P2_RATE, P2_DIGEST>;

// Deterministic Poseidon2 sponge factory (BabyBear, width 24)
#[inline]
fn p2_sponge() -> P2Sponge {
    P2Sponge::new(default_babybear_poseidon2_24())
}

// Keccak-f sponge over u64 lanes, digests are 4 u64 words
const K_DIGEST: usize = 4;
const K_WIDTH: usize = K_DIGEST;
type KInner = PaddingFreeSponge<KeccakF, 25, 17, K_DIGEST>;
type KHash = ChainingHasher<KInner>;

// Constant Keccak-f based chaining hasher
static K_INNER: KInner = PaddingFreeSponge::new(KeccakF);
static K_HASH: KHash = ChainingHasher::new(K_INNER);

// SHA256 over bytes, digests are 32 bytes
const S_DIGEST: usize = 32;
const S_WIDTH: usize = S_DIGEST;
type SHash = ChainingHasher<Sha256>;

// Constant SHA-256 based chaining hasher
static S_HASH: SHash = ChainingHasher::new(Sha256);

fn rand_matrices(rng: &mut SmallRng, scenarios: &[(usize, usize)]) -> Vec<RowMajorMatrix<F>> {
    scenarios
        .iter()
        .map(|&(h, w)| RowMajorMatrix::<F>::rand(rng, h, w))
        .collect()
}

fn benchmark_lmcs_hashes(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);

    // Scenario sets: varying heights and widths
    let scenarios: Vec<(&str, Vec<(usize, usize)>)> = vec![
        // ("single_large", vec![(1 << 15, 40)]),
        // ("two_descending", vec![(1 << 14, 40), (1 << 15, 40)]),
        (
            "mixed_multi",
            vec![
                (2, 40),
                (8, 40),
                (128, 40),
                (1 << 13, 40),
                (1 << 14, 40),
                (1 << 15, 40),
            ],
        ),
    ];

    // Poseidon2 path (deterministic constants)
    {
        let group_name = "lmcs_upsampled_poseidon2";
        let mut group = c.benchmark_group(group_name);
        let sponge = p2_sponge();

        for (label, dims) in &scenarios {
            let mats = rand_matrices(&mut rng, dims);
            let bytes: u64 = dims
                .iter()
                .map(|d| d.0 * d.1.next_multiple_of(P2_RATE))
                .sum::<usize>()
                .saturating_mul(core::mem::size_of::<F>()) as u64;
            group.throughput(Throughput::Bytes(bytes));
            group.bench_with_input(BenchmarkId::new("upsampled", *label), &mats, |b, mats| {
                b.iter(|| {
                    let out = build_leaf_states_upsampled::<P, P, _, _, P2_WIDTH, P2_DIGEST>(
                        black_box(mats),
                        black_box(&sponge),
                    )
                    .unwrap();
                    black_box(out)
                });
            });
        }
        group.finish();
    }

    // SHA256 path (deterministic)
    {
        let group_name = "lmcs_upsampled_sha256";
        let mut group = c.benchmark_group(group_name);
        let hash = &S_HASH;

        for (label, dims) in &scenarios {
            let mats = rand_matrices(&mut rng, dims);
            // PADDING_WIDTH = 1 element for chaining adapters
            let bytes: u64 = dims
                .iter()
                .map(|d| d.0 * d.1)
                .sum::<usize>()
                .saturating_mul(core::mem::size_of::<F>()) as u64;
            group.throughput(Throughput::Bytes(bytes));
            group.bench_with_input(BenchmarkId::new("upsampled", *label), &mats, |b, mats| {
                b.iter(|| {
                    let out = build_leaf_states_upsampled::<F, u8, _, _, S_WIDTH, S_DIGEST>(
                        black_box(mats),
                        black_box(hash),
                    )
                    .unwrap();
                    black_box(out)
                });
            });
        }
        group.finish();
    }

    // Keccak-f path (deterministic)
    {
        let group_name = "lmcs_upsampled_keccakf";
        let mut group = c.benchmark_group(group_name);
        let hash = &K_HASH;

        for (label, dims) in &scenarios {
            let mats = rand_matrices(&mut rng, dims);
            // PADDING_WIDTH = 1 element for chaining adapters
            let bytes: u64 = dims
                .iter()
                .map(|d| d.0 * d.1)
                .sum::<usize>()
                .saturating_mul(core::mem::size_of::<F>()) as u64;
            group.throughput(Throughput::Bytes(bytes));
            group.bench_with_input(BenchmarkId::new("upsampled", *label), &mats, |b, mats| {
                b.iter(|| {
                    let out = build_leaf_states_upsampled::<
                        [F; p3_keccak::VECTOR_LEN],
                        [u64; p3_keccak::VECTOR_LEN],
                        _,
                        _,
                        K_WIDTH,
                        K_DIGEST,
                    >(black_box(mats), black_box(hash))
                    .unwrap();
                    black_box(out)
                });
            });
        }
        group.finish();
    }
}

fn setup_criterion() -> Criterion {
    Criterion::default()
        .without_plots()
        .sample_size(10)
        .measurement_time(Duration::from_secs(15))
        .warm_up_time(Duration::from_secs(3))
}

criterion_group! { name = benches; config = setup_criterion(); targets = benchmark_lmcs_hashes }
criterion_main!(benches);
