//! PCS comparison benchmarks: Lifted PCS vs Workspace TwoAdicFriPcs.
//!
//! Compares the complete open operation for both PCS implementations
//! using multiple trace groups with different heights (simulating real STARK scenarios).
//!
//! Setup uses `RELATIVE_SPECS` from bench_utils which defines 3 groups:
//! - Group 0: Main trace columns at various heights
//! - Group 1: Auxiliary/permutation columns
//! - Group 2: Quotient polynomial chunks
//!
//! Opening points:
//! - Lifted PCS: all groups opened at two points (z1, z2)
//! - Workspace PCS: groups 0-1 at [z1, z2], group 2 at [z1] only
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
use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::Field;
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_lifted::fri::FriParams;
use p3_lifted::pcs::{self, PcsConfig};
use p3_matrix::Matrix;
use p3_matrix::bitrev::BitReversibleMatrix;
use rand::SeedableRng;
use rand::distr::{Distribution, StandardUniform};

#[path = "bench_utils.rs"]
mod bench_utils;
use bench_utils::{
    EF, F, FIELD_NAME, LOG_HEIGHTS, PARALLEL_STR, RELATIVE_SPECS, generate_matrices_from_specs,
    hash, total_elements,
};

/// Log blowup factor for FRI.
const LOG_BLOWUP: usize = 2;

/// Number of FRI queries.
const NUM_QUERIES: usize = 30;

/// Log degree of final polynomial.
const LOG_FINAL_DEGREE: usize = 8;

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
    let dft = Dft::default();
    let shift = F::GENERATOR;

    for &log_max_height in LOG_HEIGHTS {
        let max_lde_size = 1usize << log_max_height;

        let group_name = format!(
            "PCS_Open/{}/{}/{}/{}",
            max_lde_size,
            FIELD_NAME,
            hash::HASH_NAME,
            PARALLEL_STR
        );
        let mut group = c.benchmark_group(&group_name);

        group.sample_size(10);
        group.measurement_time(Duration::from_secs(30));
        group.warm_up_time(Duration::from_secs(3));

        // Generate matrix groups from RELATIVE_SPECS
        // Each group contains matrices of varying heights (already sorted)
        let matrix_groups = generate_matrices_from_specs(RELATIVE_SPECS, log_max_height);
        let total_elems = total_elements(&matrix_groups);
        group.throughput(Throughput::Elements(total_elems));

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
                proof_of_work_bits: 0,
                mmcs: challenge_mmcs,
            };

            let workspace_pcs = WorkspacePcs::new(Dft::default(), val_mmcs, fri_params);

            // Commit each group separately (workspace PCS commits one group at a time)
            // Each group's matrices are committed together with their domains
            let commits_and_data: Vec<_> = matrix_groups
                .iter()
                .map(|matrices| {
                    let domains_and_evals = matrices
                        .iter()
                        .map(|m| {
                            let domain =
                                <WorkspacePcs as Pcs<EF, Challenger>>::natural_domain_for_degree(
                                    &workspace_pcs,
                                    m.height(),
                                );
                            (domain, m.clone())
                        });
                    <WorkspacePcs as Pcs<EF, Challenger>>::commit(
                        &workspace_pcs,
                        domains_and_evals.into_iter(),
                    )
                })
                .collect();

            let base_challenger = Challenger::new(perm.clone());

            let id = BenchmarkId::from_parameter("workspace");
            group.bench_function(id, |b| {
                b.iter(|| {
                    let mut challenger = base_challenger.clone();
                    for (commitment, _) in &commits_and_data {
                        challenger.observe(*commitment);
                    }
                    let z1: EF = challenger.sample_algebra_element();
                    let z2: EF = challenger.sample_algebra_element();

                    // Open: groups 0-1 at [z1,z2], group 2 at [z1] only
                    let data_and_points: Vec<_> = commits_and_data
                        .iter()
                        .enumerate()
                        .map(|(i, (_, prover_data))| {
                            let num_matrices = matrix_groups[i].len();
                            let points = if i < 2 {
                                // Groups 0 and 1: open at both points
                                vec![vec![z1, z2]; num_matrices]
                            } else {
                                // Group 2: open at z1 only
                                vec![vec![z1]; num_matrices]
                            };
                            (prover_data, points)
                        })
                        .collect();

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
        // Lifted PCS
        // ---------------------------------------------------------------------
        {
            let lmcs = LiftedLmcs::new(sponge.clone(), compress.clone());
            let fri_mmcs = LiftedFriMmcs::new(lmcs.clone());

            // Compute LDEs and bit-reverse for each group
            let lde_groups = matrix_groups
                .iter()
                .map(|matrices| {
                    matrices
                        .iter()
                        .map(|m| {
                            let lde = dft.coset_lde_batch(m.clone(), LOG_BLOWUP, shift);
                            lde.bit_reverse_rows().to_row_major_matrix()
                        })
                        .collect()
                });

            // Commit each group
            let commits_and_data: Vec<_> = lde_groups
                .into_iter()
                .map(|group| lmcs.commit(group))
                .collect();

            let base_challenger = Challenger::new(perm.clone());

            // -----------------------------------------------------------------
            // Lifted PCS (log_folding_factor = 1)
            // -----------------------------------------------------------------
            let config = PcsConfig {
                fri: FriParams {
                    log_blowup: LOG_BLOWUP,
                    log_folding_factor: 1,
                    log_final_degree: LOG_FINAL_DEGREE,
                    num_queries: NUM_QUERIES,
                },
                alignment: hash::RATE,
            };

            let id = BenchmarkId::from_parameter("lifted/arity2");
            group.bench_function(id, |b| {
                b.iter(|| {
                    let mut challenger = base_challenger.clone();
                    for (commitment, _) in &commits_and_data {
                        challenger.observe(*commitment);
                    }
                    let z1: EF = challenger.sample_algebra_element();
                    let z2: EF = challenger.sample_algebra_element();

                    // Open all groups at both points
                    let prover_data_refs: Vec<_> =
                        commits_and_data.iter().map(|(_, data)| data).collect();
                    let proof = pcs::open::<F, EF, _, _, _, _>(
                        &lmcs,
                        prover_data_refs,
                        &[z1, z2],
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
                    for (commitment, _) in &commits_and_data {
                        challenger.observe(*commitment);
                    }
                    let z1: EF = challenger.sample_algebra_element();
                    let z2: EF = challenger.sample_algebra_element();

                    // Open all groups at both points
                    let prover_data_refs: Vec<_> =
                        commits_and_data.iter().map(|(_, data)| data).collect();
                    let proof = pcs::open::<F, EF, _, _, _, _>(
                        &lmcs,
                        prover_data_refs,
                        &[z1, z2],
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
    let mut rng = rand::rngs::SmallRng::seed_from_u64(bench_utils::BENCH_SEED);
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
