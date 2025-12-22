//! FRI folding benchmarks for lifted implementation.
//!
//! Benchmarks FRI fold operations at different arities (2, 4) and packing modes
//! (scalar vs packed/SIMD).
//!
//! Run with:
//! ```bash
//! RUSTFLAGS="-Ctarget-cpu=native" cargo bench --bench fri_fold \
//!     --features bench-babybear,bench-poseidon2
//!
//! # With parallelism
//! RUSTFLAGS="-Ctarget-cpu=native" cargo bench --bench fri_fold \
//!     --features bench-babybear,bench-poseidon2,parallel
//! ```

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_lifted::fri::fold::{FriFold, FriFold2, FriFold4};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

#[path = "bench_utils.rs"]
mod bench_utils;
use bench_utils::{EF, F, FIELD_NAME, LOG_HEIGHTS, PARALLEL_STR};

/// Target number of rows after all folding rounds.
const TARGET: usize = 8;

// =============================================================================
// Lifted FRI fold benchmarks
// =============================================================================

fn bench_lifted_fold<FF: FriFold<ARITY>, const ARITY: usize>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    n_elems: usize,
    packed: bool,
) where
    StandardUniform: Distribution<F> + Distribution<EF>,
{
    let rng = &mut SmallRng::seed_from_u64(bench_utils::BENCH_SEED);

    let n_rows = n_elems / ARITY;
    let s_invs: Vec<F> = rng.sample_iter(StandardUniform).take(n_rows).collect();

    let values: Vec<EF> = rng.sample_iter(StandardUniform).take(n_elems).collect();
    let input = RowMajorMatrix::new(values, ARITY);

    let packed_str = if packed { "packed" } else { "scalar" };
    let id = BenchmarkId::from_parameter(format!("lifted/{}/{}", ARITY, packed_str));

    group.bench_with_input(id, &n_elems, |b, &_n| {
        b.iter(|| {
            let mut current = input.clone();

            while current.height() > TARGET {
                let rows = current.height();
                let beta: EF = rng.sample(StandardUniform);
                let evals = if packed {
                    FF::fold_matrix_packed::<F, EF>(
                        black_box(current.as_view()),
                        black_box(&s_invs[..rows]),
                        black_box(beta),
                    )
                } else {
                    FF::fold_matrix_scalar::<F, EF>(
                        black_box(current.as_view()),
                        black_box(&s_invs[..rows]),
                        black_box(beta),
                    )
                };
                current = RowMajorMatrix::new(evals, ARITY);
            }
            black_box(current)
        });
    });
}

// =============================================================================
// Main benchmark function
// =============================================================================

fn bench_fri_fold(c: &mut Criterion) {
    for &log_height in LOG_HEIGHTS {
        let n_elems = 1usize << log_height;
        let group_name = format!("FRI_Fold/{}/{}/{}", n_elems, FIELD_NAME, PARALLEL_STR);
        let mut group = c.benchmark_group(&group_name);

        group.sample_size(10);
        group.measurement_time(Duration::from_secs(12));
        group.warm_up_time(Duration::from_secs(3));
        group.throughput(Throughput::Elements(n_elems as u64));

        // Lifted arity-2: scalar and packed
        bench_lifted_fold::<FriFold2, 2>(&mut group, n_elems, false);
        bench_lifted_fold::<FriFold2, 2>(&mut group, n_elems, true);

        // Lifted arity-4: scalar and packed
        bench_lifted_fold::<FriFold4, 4>(&mut group, n_elems, false);
        bench_lifted_fold::<FriFold4, 4>(&mut group, n_elems, true);

        group.finish();
    }
}

fn setup_criterion() -> Criterion {
    Criterion::default()
}

criterion_group! {
    name = benches;
    config = setup_criterion();
    targets = bench_fri_fold
}
criterion_main!(benches);
