use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use p3_field::PrimeCharacteristicRing;
use p3_field::extension::BinomialExtensionField;
use p3_goldilocks::Goldilocks;
use p3_lifted::fri::deep::{Precomputation, reduce_matrices};
use p3_matrix::dense::RowMajorMatrix;
use rand::distr::StandardUniform;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

type F = Goldilocks;
type EF = BinomialExtensionField<F, 2>;

/// Benchmark data for the DEEP quotient computation.
struct BenchData {
    matrices_groups: Vec<Vec<RowMajorMatrix<F>>>,
    coeffs_groups: Vec<Vec<Vec<EF>>>,
    d_max: usize,
    log_blowup: usize,
    n: usize,
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
        let mut coeffs_groups = Vec::new();

        for specs in group_specs {
            let mut group_matrices = Vec::new();
            let mut group_coeffs = Vec::new();

            for &(log_d, width) in specs {
                let log_scaling = log_d_max - log_d;
                let height = n >> log_scaling;

                group_matrices.push(RowMajorMatrix::rand(rng, height, width));
                group_coeffs.push(rng.sample_iter(StandardUniform).take(width).collect());
            }

            matrices_groups.push(group_matrices);
            coeffs_groups.push(group_coeffs);
        }

        Self {
            matrices_groups,
            coeffs_groups,
            d_max,
            log_blowup,
            n,
        }
    }

    fn run_deep_quotient(&self) -> Vec<EF> {
        let mut rng = SmallRng::seed_from_u64(123);
        let z1: EF = rng.sample(StandardUniform);
        let z2: EF = rng.sample(StandardUniform);

        let pre1 = Precomputation::<F, EF>::new(z1, self.d_max, self.log_blowup);
        let pre2 = Precomputation::<F, EF>::new(z2, self.d_max, self.log_blowup);

        let evals_groups_1: Vec<Vec<Vec<EF>>> = self
            .matrices_groups
            .iter()
            .map(|group| group.iter().map(|m| pre1.eval_matrix(m)).collect())
            .collect();
        let evals_groups_2: Vec<Vec<Vec<EF>>> = self
            .matrices_groups
            .iter()
            .map(|group| group.iter().map(|m| pre2.eval_matrix(m)).collect())
            .collect();

        let neg_reduced = reduce_matrices(&self.matrices_groups, &self.coeffs_groups, self.n);

        let mut acc = EF::zero_vec(self.n);
        pre1.accumulate_deep_quotient(&mut acc, &neg_reduced, &evals_groups_1, &self.coeffs_groups);
        pre2.accumulate_deep_quotient(&mut acc, &neg_reduced, &evals_groups_2, &self.coeffs_groups);

        acc
    }
}

fn bench_deep_quotient(c: &mut Criterion) {
    // Configurable max log degree
    let log_d_max: usize = 20;
    let log_blowup: usize = 3;

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
    println!(
        "Data ready: d_max={}, n={}, {} groups",
        data.d_max,
        data.n,
        data.matrices_groups.len()
    );

    group.bench_function("full", |b| b.iter(|| black_box(data.run_deep_quotient())));

    group.finish();
}

criterion_group!(benches, bench_deep_quotient);
criterion_main!(benches);
