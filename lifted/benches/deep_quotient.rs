use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use p3_field::extension::BinomialExtensionField;
use p3_goldilocks::Goldilocks;
use p3_lifted::fri::deep::Precomputation;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use rand::distr::StandardUniform;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

type F = Goldilocks;
type EF = BinomialExtensionField<F, 2>;

/// Benchmark data for the DEEP quotient computation.
struct BenchData {
    matrices_groups: Vec<Vec<RowMajorMatrix<F>>>,
    log_blowup: usize,
}

impl BenchData {
    fn new(group_specs: &[Vec<(usize, usize)>], log_blowup: usize) -> Self {
        let rng = &mut SmallRng::seed_from_u64(42);

        // Find max degree across all groups
        let log_d_max = group_specs
            .iter()
            .flat_map(|g| g.iter().map(|(log_d, _)| *log_d))
            .max()
            .unwrap();
        let d_max = 1 << log_d_max;
        let n = d_max << log_blowup;

        let mut matrices_groups = Vec::new();

        for specs in group_specs {
            let mut group_matrices = Vec::new();

            for &(log_d, width) in specs {
                let log_scaling = log_d_max - log_d;
                let height = n >> log_scaling;

                group_matrices.push(RowMajorMatrix::rand(rng, height, width));
            }

            matrices_groups.push(group_matrices);
        }

        Self {
            matrices_groups,
            log_blowup,
        }
    }
}

fn bench_deep_quotient(c: &mut Criterion) {
    // Configurable max log degree
    let log_d_max: usize = 20;
    let log_blowup: usize = 3;
    let padding: usize = 8;

    // Relative specs: (offset_from_max, width) where log_degree = log_d_max - offset
    let relative_specs: Vec<Vec<(usize, usize)>> = vec![
        vec![(4, 10), (2, 100), (0, 50)],
        vec![(4, 8), (2, 20), (0, 20)],
        vec![(0, 16)],
    ];

    // Convert relative specs to absolute log_degree
    let group_specs: Vec<Vec<(usize, usize)>> = relative_specs
        .iter()
        .map(|group| {
            group
                .iter()
                .map(|&(offset, width)| (log_d_max - offset, width))
                .collect()
        })
        .collect();

    let mut group = c.benchmark_group("deep_quotient");
    group.sample_size(10);

    // Setup data (this is slow, do it once)
    println!("Setting up benchmark data (log_d_max={log_d_max}, log_blowup={log_blowup})...");
    let data = BenchData::new(&group_specs, log_blowup);
    let n = data
        .matrices_groups
        .iter()
        .flat_map(|g| g.iter().map(|m| m.height()))
        .max()
        .unwrap();
    println!("Data ready: n={}, {} groups", n, data.matrices_groups.len());

    // Phase 1: Benchmark precomputation construction (includes matrix evaluation)
    group.bench_function("precomputation", |b| {
        let mut rng = SmallRng::seed_from_u64(456);
        b.iter(|| {
            let z: EF = rng.sample(StandardUniform);
            black_box(Precomputation::<F, EF>::new(
                z,
                &data.matrices_groups,
                data.log_blowup,
            ))
        })
    });

    // Phase 2: Benchmark DEEP reduction (with pre-built precomputations)
    let mut rng = SmallRng::seed_from_u64(123);
    let z1: EF = rng.sample(StandardUniform);
    let z2: EF = rng.sample(StandardUniform);
    let challenge: EF = rng.sample(StandardUniform);

    let precomputations = vec![
        Precomputation::<F, EF>::new(z1, &data.matrices_groups, data.log_blowup),
        Precomputation::<F, EF>::new(z2, &data.matrices_groups, data.log_blowup),
    ];

    group.bench_function("reduction", |b| {
        b.iter(|| {
            black_box(Precomputation::compute_deep_quotient(
                &precomputations,
                &data.matrices_groups,
                challenge,
                padding,
            ))
        })
    });

    // Full pipeline for comparison
    group.bench_function("full", |b| {
        let mut rng = SmallRng::seed_from_u64(789);
        b.iter(|| {
            let pre1 = Precomputation::<F, EF>::new(z1, &data.matrices_groups, data.log_blowup);
            let pre2 = Precomputation::<F, EF>::new(z2, &data.matrices_groups, data.log_blowup);

            black_box(Precomputation::compute_deep_quotient(
                &[pre1, pre2],
                &data.matrices_groups,
                challenge,
                padding,
            ))
        })
    });

    group.finish();
}

criterion_group!(benches, bench_deep_quotient);
criterion_main!(benches);
