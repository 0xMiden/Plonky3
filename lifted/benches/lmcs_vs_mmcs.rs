//! LMCS vs MMCS comparison benchmarks.
//!
//! Compares the lifted LMCS implementation against the workspace MerkleTreeMmcs
//! using identical hash configurations.
//!
//! Note: This benchmark requires Poseidon2 hash (not Keccak) because the workspace
//! MerkleTreeMmcs with ChainingHasher (keccak) doesn't implement CryptographicHasher
//! for field elements directly.
//!
//! Run with:
//! ```bash
//! RUSTFLAGS="-Ctarget-cpu=native" cargo bench --bench lmcs_vs_mmcs \
//!     --features bench-babybear,bench-poseidon2
//!
//! # With parallelism
//! RUSTFLAGS="-Ctarget-cpu=native" cargo bench --bench lmcs_vs_mmcs \
//!     --features bench-babybear,bench-poseidon2,parallel
//! ```

#[cfg(not(feature = "bench-poseidon2"))]
compile_error!(
    "LMCS vs MMCS benchmark requires bench-poseidon2 feature (keccak MMCS doesn't support field elements)"
);

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_commit::Mmcs;

#[path = "bench_utils.rs"]
mod bench_utils;
use bench_utils::{
    FIELD_NAME, LOG_HEIGHTS, PARALLEL_STR, RELATIVE_SPECS, generate_matrices_from_specs, hash,
    total_elements,
};

fn bench_lmcs_vs_mmcs(c: &mut Criterion) {
    for &log_max_height in LOG_HEIGHTS {
        let n_leaves = 1usize << log_max_height;
        let group_name = format!(
            "LMCS_vs_MMCS/{}/{}/{}/{}",
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

        // Packed mode for Poseidon2
        // LMCS uses StatefulSponge, MMCS uses PaddingFreeSponge
        let (lmcs_sponge, lmcs_compress) = hash::lmcs_components();
        let lmcs = hash::PackedLmcs::new(lmcs_sponge, lmcs_compress);

        let (mmcs_sponge, mmcs_compress) = hash::mmcs_components();
        let mmcs = hash::PackedMmcs::new(mmcs_sponge, mmcs_compress);

        let id_lmcs = BenchmarkId::from_parameter("lmcs");
        group.bench_with_input(id_lmcs, &matrix_groups, |b, groups| {
            b.iter(|| {
                for matrices in groups {
                    black_box(lmcs.commit(matrices.clone()));
                }
            });
        });

        let id_mmcs = BenchmarkId::from_parameter("mmcs");
        group.bench_with_input(id_mmcs, &matrix_groups, |b, groups| {
            b.iter(|| {
                for matrices in groups {
                    black_box(mmcs.commit(matrices.clone()));
                }
            });
        });

        group.finish();
    }
}

fn setup_criterion() -> Criterion {
    Criterion::default()
}

criterion_group! {
    name = benches;
    config = setup_criterion();
    targets = bench_lmcs_vs_mmcs
}
criterion_main!(benches);
