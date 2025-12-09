use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_field::{ExtensionField, PrimeCharacteristicRing, TwoAdicField};
use p3_goldilocks::Goldilocks;
use p3_lifted::fri::fold::{FriFold, TwoAdicFriFold};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Target number of rows after all folding rounds.
const TARGET: usize = 8;

fn bench_fold_matrix_impl<F, EF, const ARITY: usize>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    label: &str,
    n_rows: usize,
) where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    TwoAdicFriFold: FriFold<F, ARITY>,
    StandardUniform: Distribution<F> + Distribution<EF>,
{
    let rng = &mut SmallRng::seed_from_u64(2025);

    let s_invs: Vec<F> = rng.sample_iter(StandardUniform).take(n_rows).collect();
    let beta: EF = rng.sample(StandardUniform);

    let values: Vec<EF> = rng.sample_iter(StandardUniform).take(n_rows).collect();
    let input = RowMajorMatrix::new(values, ARITY);

    group.throughput(Throughput::Elements(n_rows as u64));
    group.bench_with_input(BenchmarkId::new(label, n_rows), &n_rows, |b, &_n| {
        b.iter(|| {
            let mut current = input.clone();

            while current.values.len() > TARGET {
                let rows = current.height();
                current = TwoAdicFriFold::fold_matrix(
                    black_box(current.as_view()),
                    black_box(&s_invs[..rows]),
                    black_box(beta),
                );
            }
            black_box(current)
        });
    });
}

fn bench_fold_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("FRI_FoldMatrix");

    group.sample_size(10);
    group.measurement_time(Duration::from_secs(12));
    group.warm_up_time(Duration::from_secs(3));

    for &n_rows in &[1 << 10, 1 << 14, 1 << 17] {
        bench_fold_matrix_impl::<BabyBear, BinomialExtensionField<BabyBear, 4>, 2>(
            &mut group,
            "babybear-d4/arity2",
            n_rows,
        );
        bench_fold_matrix_impl::<BabyBear, BinomialExtensionField<BabyBear, 4>, 4>(
            &mut group,
            "babybear-d4/arity4",
            n_rows,
        );
        bench_fold_matrix_impl::<Goldilocks, BinomialExtensionField<Goldilocks, 2>, 2>(
            &mut group,
            "goldilocks-d2/arity2",
            n_rows,
        );
        bench_fold_matrix_impl::<Goldilocks, BinomialExtensionField<Goldilocks, 2>, 4>(
            &mut group,
            "goldilocks-d2/arity4",
            n_rows,
        );
    }

    group.finish();
}

fn setup_criterion() -> Criterion {
    Criterion::default().without_plots()
}

criterion_group! {
    name = benches;
    config = setup_criterion();
    targets = bench_fold_matrix
}
criterion_main!(benches);
