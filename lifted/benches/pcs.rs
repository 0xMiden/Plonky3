//! PCS comparison benchmarks: Lifted PCS vs Workspace TwoAdicFriPcs.
//!
//! Compares the complete open operation for both PCS implementations
//! using identical polynomial data and FRI parameters.
//!
//! Note: This benchmark only supports Poseidon2 hash (not Keccak) because
//! the workspace TwoAdicFriPcs requires a permutation-based challenger.
//!
//! Run with:
//! ```bash
//! RUSTFLAGS="-Ctarget-cpu=native" cargo bench --bench pcs \
//!     --features bench-babybear,bench-poseidon2
//!
//! # With parallelism
//! RUSTFLAGS="-Ctarget-cpu=native" cargo bench --bench pcs \
//!     --features bench-babybear,bench-poseidon2,parallel
//! ```

#[cfg(not(feature = "bench-poseidon2"))]
compile_error!(
    "PCS benchmark requires bench-poseidon2 feature (workspace FRI needs permutation-based challenger)"
);

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_challenger::{CanObserve, DuplexChallenger, FieldChallenger};
use p3_commit::{ExtensionMmcs, Mmcs, Pcs};
use p3_dft::Radix2DitParallel;
use p3_field::Field;
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_lifted::fri::FriParams;
use p3_lifted::merkle_tree::Lifting;
use p3_lifted::pcs::{self, PcsConfig};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use rand::SeedableRng;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;

#[path = "bench_utils.rs"]
mod bench_utils;
use bench_utils::{EF, F, FIELD_NAME, LOG_HEIGHTS, PARALLEL_STR, hash};

/// Log blowup factor for FRI.
const LOG_BLOWUP: usize = 2;

/// Number of FRI queries.
const NUM_QUERIES: usize = 30;

/// Log degree of final polynomial.
const LOG_FINAL_DEGREE: usize = 8;

/// Proof of work bits (workspace FRI only).
const POW_BITS: usize = 8;

// =============================================================================
// Type aliases for workspace PCS
// =============================================================================

type ValMmcs = hash::PackedMmcs;
type ChallengeMmcs = ExtensionMmcs<F, EF, ValMmcs>;
type Dft = Radix2DitParallel<F>;
type Challenger = DuplexChallenger<F, hash::Perm, { hash::WIDTH }, { hash::RATE }>;
type WorkspacePcs = TwoAdicFriPcs<F, Dft, ValMmcs, ChallengeMmcs>;

// =============================================================================
// Type aliases for lifted PCS
// =============================================================================

type LiftedLmcs = hash::PackedLmcs;
type LiftedFriMmcs = ExtensionMmcs<F, EF, LiftedLmcs>;

// =============================================================================
// Benchmark implementation
// =============================================================================

fn bench_pcs(c: &mut Criterion) {
    let (sponge, compress) = hash::components();
    let perm = create_perm();

    for &log_max_degree in LOG_HEIGHTS {
        // For PCS, we benchmark polynomial degree, not LDE size
        // LDE size = degree << log_blowup
        let degree = 1usize << log_max_degree;
        let lde_size = degree << LOG_BLOWUP;

        let group_name = format!(
            "PCS_Open/{}/{}/{}/{}",
            lde_size,
            FIELD_NAME,
            hash::HASH_NAME,
            PARALLEL_STR
        );
        let mut group = c.benchmark_group(&group_name);

        group.sample_size(10);
        group.measurement_time(Duration::from_secs(15));
        group.warm_up_time(Duration::from_secs(3));
        group.throughput(Throughput::Elements(lde_size as u64));

        // Generate polynomial data
        let rng = &mut SmallRng::seed_from_u64(bench_utils::BENCH_SEED);
        let width = 50; // Similar to trace width
        let poly_evals = RowMajorMatrix::<F>::rand(rng, degree, width);

        // ---------------------------------------------------------------------
        // Workspace TwoAdicFriPcs
        // ---------------------------------------------------------------------
        {
            let val_mmcs = ValMmcs::new(sponge.clone(), compress.clone());
            let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

            let fri_params = FriParameters {
                log_blowup: LOG_BLOWUP,
                log_final_poly_len: LOG_FINAL_DEGREE,
                num_queries: NUM_QUERIES,
                proof_of_work_bits: POW_BITS,
                mmcs: challenge_mmcs,
            };

            let workspace_pcs = WorkspacePcs::new(Dft::default(), val_mmcs, fri_params);
            let domain = <WorkspacePcs as Pcs<EF, Challenger>>::natural_domain_for_degree(
                &workspace_pcs,
                degree,
            );

            // Commit (includes LDE computation)
            let (commitment, prover_data) = <WorkspacePcs as Pcs<EF, Challenger>>::commit(
                &workspace_pcs,
                [(domain, poly_evals.clone())].into_iter(),
            );

            let base_challenger = Challenger::new(perm.clone());

            let id = BenchmarkId::from_parameter("workspace");
            group.bench_function(id, |b| {
                b.iter(|| {
                    let mut challenger = base_challenger.clone();
                    challenger.observe(commitment);
                    let z: EF = challenger.sample_algebra_element();

                    // Open at a single point
                    let data_and_points = vec![(&prover_data, vec![vec![z]])];
                    let (_openings, proof) = <WorkspacePcs as Pcs<EF, Challenger>>::open(
                        &workspace_pcs,
                        black_box(data_and_points),
                        &mut challenger,
                    );
                    black_box(proof)
                });
            });
        }

        // ---------------------------------------------------------------------
        // Lifted PCS (log_folding_factor = 1)
        // ---------------------------------------------------------------------
        {
            let lmcs = LiftedLmcs::new(sponge.clone(), compress.clone(), Lifting::Upsample);
            let fri_mmcs = LiftedFriMmcs::new(lmcs.clone());

            let config = PcsConfig {
                fri: FriParams {
                    log_blowup: LOG_BLOWUP,
                    log_folding_factor: 1,
                    log_final_degree: LOG_FINAL_DEGREE,
                    num_queries: NUM_QUERIES,
                },
                alignment: hash::RATE,
            };

            // For lifted PCS, we need to provide already-LDE'd data
            // Compute LDE manually to match workspace
            let dft = Dft::default();
            let shift = F::GENERATOR;
            let lde_evals = p3_dft::TwoAdicSubgroupDft::coset_lde_batch(
                &dft,
                poly_evals.clone(),
                LOG_BLOWUP,
                shift,
            );
            let lde_bitrev = p3_matrix::bitrev::BitReversibleMatrix::bit_reverse_rows(lde_evals)
                .to_row_major_matrix();

            // Commit
            let (commitment, prover_data) = lmcs.commit(vec![lde_bitrev.clone()]);

            let base_challenger = Challenger::new(perm.clone());

            let id = BenchmarkId::from_parameter("lifted/arity2");
            group.bench_function(id, |b| {
                b.iter(|| {
                    let mut challenger = base_challenger.clone();
                    challenger.observe(commitment);
                    let z: EF = challenger.sample_algebra_element();

                    let proof = pcs::open::<F, EF, _, _, _, _>(
                        &lmcs,
                        vec![&prover_data],
                        &[z],
                        &mut challenger,
                        &config,
                        &fri_mmcs,
                    );
                    black_box(proof)
                });
            });

            // -----------------------------------------------------------------
            // Lifted PCS (log_folding_factor = 2)
            // -----------------------------------------------------------------
            let config_arity4 = PcsConfig {
                fri: FriParams {
                    log_blowup: LOG_BLOWUP,
                    log_folding_factor: 2,
                    log_final_degree: LOG_FINAL_DEGREE,
                    num_queries: NUM_QUERIES,
                },
                alignment: hash::RATE,
            };

            let id = BenchmarkId::from_parameter("lifted/arity4");
            group.bench_function(id, |b| {
                b.iter(|| {
                    let mut challenger = base_challenger.clone();
                    challenger.observe(commitment);
                    let z: EF = challenger.sample_algebra_element();

                    let proof = pcs::open::<F, EF, _, _, _, _>(
                        &lmcs,
                        vec![&prover_data],
                        &[z],
                        &mut challenger,
                        &config_arity4,
                        &fri_mmcs,
                    );
                    black_box(proof)
                });
            });
        }

        group.finish();
    }
}

fn create_perm() -> hash::Perm
where
    StandardUniform: Distribution<F>,
{
    let mut rng = SmallRng::seed_from_u64(bench_utils::BENCH_SEED);
    hash::Perm::new_from_rng_128(&mut rng)
}

fn setup_criterion() -> Criterion {
    Criterion::default()
}

criterion_group! {
    name = benches;
    config = setup_criterion();
    targets = bench_pcs
}
criterion_main!(benches);
