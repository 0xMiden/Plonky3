//! Benchmark for Merkle tree commit operations.
//!
//! Compares different hash functions (Poseidon2, Keccak) and packing modes
//! (scalar vs SIMD) across various input sizes.
//!
//! Run with:
//! ```bash
//! # Single-threaded
//! RUSTFLAGS="-Ctarget-cpu=native" cargo bench --bench merkle_commit
//!
//! # Parallel
//! RUSTFLAGS="-Ctarget-cpu=native" cargo bench --bench merkle_commit --features parallel
//! ```

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_commit::Mmcs;
use p3_field::Field;
use p3_keccak::KeccakF;
use p3_lifted::merkle_tree::{Lifting, MerkleTreeLmcs};
use p3_matrix::dense::RowMajorMatrix;
use p3_symmetric::{ChainingHasher, PaddingFreeSponge, TruncatedPermutation};
use rand::SeedableRng;
use rand::rngs::SmallRng;

#[path = "bench_utils.rs"]
mod bench_utils;
use bench_utils::{
    LOG_HEIGHTS, PARALLEL_STR, RELATIVE_SPECS, generate_matrices_from_specs, total_elements,
};

// =============================================================================
// Type definitions
// =============================================================================

type F = BabyBear;
type Packing = <F as Field>::Packing;

// Poseidon2 configuration (field-native)
const P2_WIDTH: usize = 24;
const P2_RATE: usize = 16;
const P2_DIGEST: usize = 8;

type P2Perm = Poseidon2BabyBear<P2_WIDTH>;
type P2Sponge = PaddingFreeSponge<P2Perm, P2_WIDTH, P2_RATE, P2_DIGEST>;
type P2Compress = TruncatedPermutation<P2Perm, 2, P2_DIGEST, P2_WIDTH>;

// Keccak configuration (byte-based)
const K_DIGEST: usize = 4;
const K_WIDTH: usize = K_DIGEST;
const K_VEC: usize = p3_keccak::VECTOR_LEN;

type KInner = PaddingFreeSponge<KeccakF, 25, 17, K_DIGEST>;
type KHash = ChainingHasher<KInner>;
type KCompress = TruncatedPermutation<KeccakF, 2, K_DIGEST, 25>;

// =============================================================================
// Component factories
// =============================================================================

fn poseidon2_components() -> (P2Sponge, P2Compress) {
    let mut rng = SmallRng::seed_from_u64(2025);
    let perm = P2Perm::new_from_rng_128(&mut rng);
    let sponge = P2Sponge::new(perm.clone());
    let compress = P2Compress::new(perm);
    (sponge, compress)
}

fn keccak_components() -> (KHash, KCompress) {
    static K_INNER: KInner = PaddingFreeSponge::new(KeccakF);
    let hash = ChainingHasher::new(K_INNER);
    let compress = TruncatedPermutation::new(KeccakF);
    (hash, compress)
}

// =============================================================================
// Benchmark implementations
// =============================================================================

fn bench_merkle_commit(c: &mut Criterion) {
    for &log_max_height in LOG_HEIGHTS {
        let n_leaves = 1usize << log_max_height;
        let group_name = format!("MerkleCommit/{}/{}", PARALLEL_STR, n_leaves);
        let mut group = c.benchmark_group(&group_name);

        group.sample_size(10);
        group.measurement_time(Duration::from_secs(12));
        group.warm_up_time(Duration::from_secs(3));

        // Generate matrices once per size
        let matrix_groups: Vec<Vec<RowMajorMatrix<F>>> =
            generate_matrices_from_specs(RELATIVE_SPECS, log_max_height);
        let total_elems = total_elements(&matrix_groups);
        group.throughput(Throughput::Elements(total_elems));

        // -------------------------------------------------------------------------
        // Poseidon2 benchmarks
        // -------------------------------------------------------------------------

        // Poseidon2 - scalar
        {
            let (sponge, compress) = poseidon2_components();
            let lmcs = MerkleTreeLmcs::<F, F, _, _, P2_WIDTH, P2_DIGEST>::new(
                sponge,
                compress,
                Lifting::Upsample,
            );

            let id = BenchmarkId::from_parameter("poseidon2/scalar");
            group.bench_with_input(id, &matrix_groups, |b, groups| {
                b.iter(|| {
                    for matrices in groups {
                        black_box(lmcs.commit(matrices.clone()));
                    }
                });
            });
        }

        // Poseidon2 - packed
        {
            let (sponge, compress) = poseidon2_components();
            let lmcs = MerkleTreeLmcs::<Packing, Packing, _, _, P2_WIDTH, P2_DIGEST>::new(
                sponge,
                compress,
                Lifting::Upsample,
            );

            let id = BenchmarkId::from_parameter("poseidon2/packed");
            group.bench_with_input(id, &matrix_groups, |b, groups| {
                b.iter(|| {
                    for matrices in groups {
                        black_box(lmcs.commit(matrices.clone()));
                    }
                });
            });
        }

        // -------------------------------------------------------------------------
        // Keccak benchmarks
        // -------------------------------------------------------------------------

        // Keccak - scalar
        {
            let (hash, compress) = keccak_components();
            let lmcs = MerkleTreeLmcs::<F, u64, _, _, K_WIDTH, K_DIGEST>::new(
                hash,
                compress,
                Lifting::Upsample,
            );

            let id = BenchmarkId::from_parameter("keccak/scalar");
            group.bench_with_input(id, &matrix_groups, |b, groups| {
                b.iter(|| {
                    for matrices in groups {
                        black_box(lmcs.commit(matrices.clone()));
                    }
                });
            });
        }

        // Keccak - packed
        {
            let (hash, compress) = keccak_components();
            let lmcs = MerkleTreeLmcs::<[F; K_VEC], [u64; K_VEC], _, _, K_WIDTH, K_DIGEST>::new(
                hash,
                compress,
                Lifting::Upsample,
            );

            let id = BenchmarkId::from_parameter("keccak/packed");
            group.bench_with_input(id, &matrix_groups, |b, groups| {
                b.iter(|| {
                    for matrices in groups {
                        black_box(lmcs.commit(matrices.clone()));
                    }
                });
            });
        }

        group.finish();
    }
}

fn setup_criterion() -> Criterion {
    Criterion::default()
}

criterion_group! {
    name = benches;
    config = setup_criterion();
    targets = bench_merkle_commit
}
criterion_main!(benches);
