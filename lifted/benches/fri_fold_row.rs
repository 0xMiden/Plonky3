use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_field::{ExtensionField, PrimeCharacteristicRing, TwoAdicField};
use p3_goldilocks::Goldilocks;
use p3_lifted::fri::fold4::fold_evals;
use rand::distr::StandardUniform;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

fn bench_fold_row_impl<F, EF>(c: &mut Criterion, label: &str)
where
    F: TwoAdicField,
    EF: ExtensionField<F> + PrimeCharacteristicRing,
    StandardUniform: rand::distr::Distribution<F> + rand::distr::Distribution<EF>,
{
    // Benchmark arity-4 FRI folding implemented via inverse FFT in `ifft::fold_evals`.
    // We pre-generate random inputs and then measure only the folding work in the hot loop.

    // Dataset sizes (number of rows folded per iteration).
    let sizes: [usize; 3] = [1 << 10, 1 << 14, 1 << 17];
    let mut group = c.benchmark_group(format!("FRI_FoldRow_IFFT_arity4/{label}"));

    // Keep sample/warmup times modest to fit typical CI while allowing large inputs.
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(12));
    group.warm_up_time(Duration::from_secs(3));

    for &n_rows in &sizes {
        // Precompute random beta once per round to reflect real FRI usage.
        let mut rng = SmallRng::seed_from_u64(2025);
        let beta: EF = rng.sample(StandardUniform);

        // Pre-generate inputs: per-row s_inv and 4 evals in bit-reversed order.
        // We deliberately avoid extra structure; fold_evals does not require evals to come
        // from a consistent polynomial for benchmarking its arithmetic cost.
        let s_invs: Vec<F> = (0..n_rows).map(|_| rng.sample(StandardUniform)).collect();
        let evals: Vec<[EF; 4]> = (0..n_rows)
            .map(|_| {
                [
                    rng.sample(StandardUniform),
                    rng.sample(StandardUniform),
                    rng.sample(StandardUniform),
                    rng.sample(StandardUniform),
                ]
            })
            .collect();

        group.throughput(Throughput::Elements(n_rows as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n_rows), &n_rows, |b, &_n| {
            b.iter(|| {
                for i in 0..n_rows {
                    let out = fold_evals::<F, EF>(
                        black_box(evals[i]),
                        black_box(s_invs[i]),
                        black_box(beta),
                    );
                    let _ = black_box(out);
                }
            });
        });
    }

    group.finish();
}

fn bench_goldilocks(c: &mut Criterion) {
    bench_fold_row_impl::<Goldilocks, BinomialExtensionField<Goldilocks, 2>>(c, "goldilocks-d2");
}

fn bench_babybear(c: &mut Criterion) {
    bench_fold_row_impl::<BabyBear, BinomialExtensionField<BabyBear, 4>>(c, "babybear-d4");
}

fn setup_criterion() -> Criterion {
    Criterion::default().without_plots()
}

criterion_group! {
    name = benches;
    config = setup_criterion();
    targets = bench_babybear, bench_goldilocks
}
criterion_main!(benches);
