use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_field::{ExtensionField, TwoAdicField};
use p3_goldilocks::Goldilocks;
use p3_lifted::fri::fold::{FriFold, TwoAdicFriFold};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Target number of rows after all folding rounds.
const TARGET: usize = 8;

const PARALLEL_STR: &str = if cfg!(feature = "parallel") {
    "parallel"
} else {
    "single"
};

fn bench_fold_impl<F, EF, const ARITY: usize>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    field_name: &str,
    packed: bool,
    n_elems: usize,
) where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    TwoAdicFriFold: FriFold<ARITY>,
    StandardUniform: Distribution<F> + Distribution<EF>,
{
    let rng = &mut SmallRng::seed_from_u64(2025);

    let n_rows = n_elems / ARITY;
    let s_invs: Vec<F> = rng.sample_iter(StandardUniform).take(n_rows).collect();

    let values: Vec<EF> = rng.sample_iter(StandardUniform).take(n_elems).collect();
    let input = RowMajorMatrix::new(values, ARITY);

    let packed_str = if packed { "packed" } else { "scalar" };

    // Use BenchmarkId::from_parameter for better comparison charts
    let id = BenchmarkId::from_parameter(format!("{}/{}/{}", field_name, ARITY, packed_str));

    group.bench_with_input(id, &n_elems, |b, &_n| {
        b.iter(|| {
            let mut current = input.clone();

            while current.height() > TARGET {
                let rows = current.height();
                let beta: EF = rng.sample(StandardUniform);
                current = if packed {
                    TwoAdicFriFold::fold_matrix_packed::<F, EF>(
                        black_box(current.as_view()),
                        black_box(&s_invs[..rows]),
                        black_box(beta),
                    )
                } else {
                    TwoAdicFriFold::fold_matrix::<F, EF>(
                        black_box(current.as_view()),
                        black_box(&s_invs[..rows]),
                        black_box(beta),
                    )
                };
            }
            black_box(current)
        });
    });
}

fn bench_fold_matrix(c: &mut Criterion) {
    for &n_elems in &[1 << 16, 1 << 18, 1 << 20] {
        let group_name = format!("FRI_Fold/{}/{}", PARALLEL_STR, n_elems);
        let mut group = c.benchmark_group(&group_name);

        group.sample_size(10);
        group.measurement_time(Duration::from_secs(12));
        group.warm_up_time(Duration::from_secs(3));
        group.throughput(Throughput::Elements(n_elems as u64));

        // BabyBear with degree-4 extension
        for &arity in &[2, 4] {
            for packed in [false, true] {
                if arity == 2 {
                    bench_fold_impl::<BabyBear, BinomialExtensionField<BabyBear, 4>, 2>(
                        &mut group, "babybear", packed, n_elems,
                    );
                } else {
                    bench_fold_impl::<BabyBear, BinomialExtensionField<BabyBear, 4>, 4>(
                        &mut group, "babybear", packed, n_elems,
                    );
                }
            }
        }
        // Goldilocks with degree-2 extension
        for &arity in &[2, 4] {
            for packed in [false, true] {
                if arity == 2 {
                    bench_fold_impl::<Goldilocks, BinomialExtensionField<Goldilocks, 2>, 2>(
                        &mut group, "goldilocks", packed, n_elems,
                    );
                } else {
                    bench_fold_impl::<Goldilocks, BinomialExtensionField<Goldilocks, 2>, 4>(
                        &mut group, "goldilocks", packed, n_elems,
                    );
                }
            }
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
    targets = bench_fold_matrix
}
criterion_main!(benches);
