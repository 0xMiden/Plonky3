//! DEEP quotient benchmarks.
//!
//! Benchmarks:
//! 1. `batch_eval` - Barycentric evaluation only (PointQuotients<2>)
//! 2. `N1` - PointQuotients<1> + DeepPoly::new (1 point)
//! 3. `N2` - PointQuotients<2> + DeepPoly::new (2 points)
//!
//! Note: This benchmark requires Poseidon2 hash (not Keccak) because it uses
//! a DuplexChallenger which needs a permutation-based hash.
//!
//! Run with:
//! ```bash
//! RUSTFLAGS="-Ctarget-cpu=native" cargo bench --bench deep_quotient \
//!     --features bench-babybear,bench-poseidon2
//!
//! # With parallelism
//! RUSTFLAGS="-Ctarget-cpu=native" cargo bench --bench deep_quotient \
//!     --features bench-babybear,bench-poseidon2,parallel
//! ```

#[cfg(not(feature = "bench-poseidon2"))]
compile_error!("DEEP quotient benchmark requires bench-poseidon2 feature (uses DuplexChallenger)");

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_challenger::DuplexChallenger;
use p3_commit::Mmcs;
use p3_field::FieldArray;
use p3_lifted::deep::prover::DeepPoly;
use p3_lifted::merkle_tree::MerkleTreeLmcs;
use p3_lifted::utils::bit_reversed_coset_points;
use p3_symmetric::CryptographicPermutation;
use rand::distr::StandardUniform;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

#[path = "bench_utils.rs"]
mod bench_utils;
use bench_utils::{
    EF, F, FIELD_NAME, LOG_HEIGHTS, P, PARALLEL_STR, RELATIVE_SPECS, generate_matrices_from_specs,
    hash, total_elements,
};
use p3_lifted::deep::interpolate::PointQuotients;

/// Log blowup factor for LDE.
const LOG_BLOWUP: usize = 3;

fn bench_deep_quotient(c: &mut Criterion) {
    let (sponge, compress) = hash::components();

    for &log_max_height in LOG_HEIGHTS {
        let n_leaves = 1usize << log_max_height;
        let group_name = format!("DEEP_Quotient/{}/{}/{}", n_leaves, FIELD_NAME, PARALLEL_STR);
        let mut group = c.benchmark_group(&group_name);

        group.sample_size(10);
        group.measurement_time(Duration::from_secs(12));
        group.warm_up_time(Duration::from_secs(3));

        // Generate matrices using canonical specs
        let matrix_groups = generate_matrices_from_specs(RELATIVE_SPECS, log_max_height);
        let total_elems = total_elements(&matrix_groups);
        group.throughput(Throughput::Elements(total_elems));

        // Setup LMCS and commit
        let lmcs = MerkleTreeLmcs::<P, P, _, _, { hash::WIDTH }, { hash::DIGEST }>::new(
            sponge.clone(),
            compress.clone(),
        );

        let committed: Vec<_> = matrix_groups
            .iter()
            .map(|matrices| lmcs.commit(matrices.clone()))
            .collect();
        let prover_data: Vec<_> = committed.iter().map(|(_, pd)| pd).collect();

        // Precompute coset points (LDE domain matches max matrix height)
        let coset_points = bit_reversed_coset_points::<F>(log_max_height);

        let matrices_refs: Vec<Vec<_>> = matrix_groups.iter().map(|g| g.iter().collect()).collect();

        // ---------------------------------------------------------------------
        // Benchmark 1: batch_eval_lifted only (PointQuotients<2>)
        // ---------------------------------------------------------------------
        group.bench_function(BenchmarkId::from_parameter("batch_eval"), |b| {
            let mut rng = SmallRng::seed_from_u64(789);
            b.iter(|| {
                let z1: EF = rng.sample(StandardUniform);
                let z2: EF = rng.sample(StandardUniform);
                let quotient = PointQuotients::<F, EF, 2>::new(FieldArray([z1, z2]), &coset_points);
                black_box(quotient.batch_eval_lifted(&matrices_refs, &coset_points, LOG_BLOWUP))
            });
        });

        // ---------------------------------------------------------------------
        // Benchmark 2: Single point with PointQuotients<1> + DeepPoly::new
        // ---------------------------------------------------------------------
        let perm = create_perm();
        let base_challenger =
            DuplexChallenger::<F, hash::Perm, { hash::WIDTH }, { hash::RATE }>::new(perm);
        group.bench_function(BenchmarkId::from_parameter("N1"), |b| {
            let mut rng = SmallRng::seed_from_u64(789);
            b.iter(|| {
                let z: EF = rng.sample(StandardUniform);

                let quotient = PointQuotients::<F, EF, 1>::new(FieldArray([z]), &coset_points);
                let evals = quotient.batch_eval_lifted(&matrices_refs, &coset_points, LOG_BLOWUP);

                let mut challenger = base_challenger.clone();
                black_box(DeepPoly::new(
                    &lmcs,
                    &quotient,
                    &evals,
                    prover_data.clone(),
                    &mut challenger,
                    hash::RATE,
                ))
            });
        });

        // ---------------------------------------------------------------------
        // Benchmark 3: Two points with PointQuotients<2> + DeepPoly::new
        // ---------------------------------------------------------------------
        group.bench_function(BenchmarkId::from_parameter("N2"), |b| {
            let mut rng = SmallRng::seed_from_u64(789);
            b.iter(|| {
                let z1: EF = rng.sample(StandardUniform);
                let z2: EF = rng.sample(StandardUniform);

                let quotient = PointQuotients::<F, EF, 2>::new(FieldArray([z1, z2]), &coset_points);
                let evals = quotient.batch_eval_lifted(&matrices_refs, &coset_points, LOG_BLOWUP);

                let mut challenger = base_challenger.clone();
                black_box(DeepPoly::new(
                    &lmcs,
                    &quotient,
                    &evals,
                    prover_data.clone(),
                    &mut challenger,
                    hash::RATE,
                ))
            });
        });

        group.finish();
    }
}

// Helper to create permutation for challenger
#[cfg(feature = "bench-poseidon2")]
fn create_perm() -> hash::Perm
where
    hash::Perm: CryptographicPermutation<[F; hash::WIDTH]>,
{
    let mut rng = SmallRng::seed_from_u64(bench_utils::BENCH_SEED);
    hash::Perm::new_from_rng_128(&mut rng)
}

#[cfg(feature = "bench-keccak")]
fn create_perm() -> hash::Perm {
    // Keccak doesn't need initialization
    let (sponge, _) = hash::components();
    sponge
}

fn setup_criterion() -> Criterion {
    Criterion::default()
}

criterion_group! {
    name = benches;
    config = setup_criterion();
    targets = bench_deep_quotient
}
criterion_main!(benches);
