//! Merkle tree commit benchmarks for LMCS.
//!
//! Benchmarks LMCS commit operations with different packing modes (scalar vs packed).
//! Hash function is selected via feature flags.
//!
//! Run with:
//! ```bash
//! # BabyBear + Poseidon2
//! RUSTFLAGS="-Ctarget-cpu=native" cargo bench --bench merkle_commit \
//!     --features bench-babybear,bench-poseidon2
//!
//! # Goldilocks + Keccak
//! RUSTFLAGS="-Ctarget-cpu=native" cargo bench --bench merkle_commit \
//!     --features bench-goldilocks,bench-keccak
//!
//! # With parallelism
//! RUSTFLAGS="-Ctarget-cpu=native" cargo bench --bench merkle_commit \
//!     --features bench-babybear,bench-poseidon2,parallel
//! ```

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_commit::{ExtensionMmcs, Mmcs};
use p3_matrix::dense::RowMajorMatrix;
use rand::SeedableRng;
use rand::rngs::SmallRng;

#[path = "bench_utils.rs"]
mod bench_utils;
use bench_utils::{
    EF, F, FIELD_NAME, LOG_HEIGHTS, PARALLEL_STR, RELATIVE_SPECS, generate_matrices_from_specs,
    hash, total_elements,
};

fn bench_merkle_commit(c: &mut Criterion) {
    for &log_max_height in LOG_HEIGHTS {
        let n_leaves = 1usize << log_max_height;
        let group_name = format!(
            "MerkleCommit/{}/{}/{}/{}",
            n_leaves,
            FIELD_NAME,
            hash::HASH_NAME,
            PARALLEL_STR
        );
        let mut group = c.benchmark_group(&group_name);

        group.sample_size(10);
        group.measurement_time(Duration::from_secs(12));
        group.warm_up_time(Duration::from_secs(3));

        // Generate matrices using canonical specs
        let matrix_groups = generate_matrices_from_specs(RELATIVE_SPECS, log_max_height);
        let total_elems = total_elements(&matrix_groups);
        group.throughput(Throughput::Elements(total_elems));

        // ---------------------------------------------------------------------
        // Scalar LMCS
        // ---------------------------------------------------------------------
        {
            let (sponge, compress) = hash::lmcs_components();
            let lmcs = hash::ScalarLmcs::new(sponge, compress);

            let id = BenchmarkId::from_parameter("scalar");
            group.bench_with_input(id, &matrix_groups, |b, groups| {
                b.iter(|| {
                    for matrices in groups {
                        black_box(lmcs.commit(matrices.clone()));
                    }
                });
            });
        }

        // ---------------------------------------------------------------------
        // Packed LMCS
        // ---------------------------------------------------------------------
        {
            let (sponge, compress) = hash::lmcs_components();
            let lmcs = hash::PackedLmcs::new(sponge, compress);

            let id = BenchmarkId::from_parameter("packed");
            group.bench_with_input(id, &matrix_groups, |b, groups| {
                b.iter(|| {
                    for matrices in groups {
                        black_box(lmcs.commit(matrices.clone()));
                    }
                });
            });
        }

        // ---------------------------------------------------------------------
        // ExtensionMmcs with width-2 matrix (simulates FRI arity-2 commit)
        // ---------------------------------------------------------------------
        {
            let (sponge, compress) = hash::lmcs_components();
            let lmcs = hash::PackedLmcs::new(sponge, compress);
            let ext_mmcs = ExtensionMmcs::<F, EF, _>::new(lmcs);

            let rng = &mut SmallRng::seed_from_u64(bench_utils::BENCH_SEED);
            let ext_matrix = RowMajorMatrix::<EF>::rand(rng, n_leaves, 2);

            let id = BenchmarkId::from_parameter("ext_mmcs/arity2");
            group.bench_with_input(id, &ext_matrix, |b, matrix| {
                b.iter(|| black_box(ext_mmcs.commit_matrix(matrix.clone())));
            });
        }

        // ---------------------------------------------------------------------
        // ExtensionMmcs with width-4 matrix (simulates FRI arity-4 commit)
        // ---------------------------------------------------------------------
        {
            let (sponge, compress) = hash::lmcs_components();
            let lmcs = hash::PackedLmcs::new(sponge, compress);
            let ext_mmcs = ExtensionMmcs::<F, EF, _>::new(lmcs);

            let rng = &mut SmallRng::seed_from_u64(bench_utils::BENCH_SEED);
            let ext_matrix = RowMajorMatrix::<EF>::rand(rng, n_leaves, 4);

            let id = BenchmarkId::from_parameter("ext_mmcs/arity4");
            group.bench_with_input(id, &ext_matrix, |b, matrix| {
                b.iter(|| black_box(ext_mmcs.commit_matrix(matrix.clone())));
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
