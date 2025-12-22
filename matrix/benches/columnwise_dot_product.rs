//! Benchmarks for `columnwise_dot_product` and `columnwise_dot_product_batched`.
//!
//! Measures:
//! - Overhead of `batched<1>` vs unbatched (should be minimal)
//! - Benefit of `batched<2>` vs 2× unbatched calls

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_field::FieldArray;
use p3_field::extension::BinomialExtensionField;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use rand::SeedableRng;
use rand::rngs::SmallRng;

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;

const LOG_ROWS: &[usize] = &[16, 18, 20];
const WIDTHS: &[usize] = &[128, 512, 4096];

fn columnwise_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("columnwise_dot_product");
    group.sample_size(20);

    for &log_rows in LOG_ROWS {
        for &width in WIDTHS {
            let mut rng = SmallRng::seed_from_u64(0);
            let rows = 1 << log_rows;
            let m = RowMajorMatrix::<F>::rand_nonzero(&mut rng, rows, width);
            let v1: Vec<EF> = RowMajorMatrix::<EF>::rand_nonzero(&mut rng, rows, 1).values;
            let v2: Vec<EF> = RowMajorMatrix::<EF>::rand_nonzero(&mut rng, rows, 1).values;
            let vs1: Vec<FieldArray<EF, 1>> = v1.iter().map(|&x| FieldArray([x])).collect();
            let vs2: Vec<FieldArray<EF, 2>> = v1
                .iter()
                .zip(&v2)
                .map(|(&a, &b)| FieldArray([a, b]))
                .collect();

            let param = format!("2^{log_rows}x{width}");

            // Measure batched<1> overhead
            group.bench_with_input(BenchmarkId::new("unbatched", &param), &(), |b, _| {
                b.iter(|| m.columnwise_dot_product(&v1));
            });
            group.bench_with_input(BenchmarkId::new("batched<1>", &param), &(), |b, _| {
                b.iter(|| m.columnwise_dot_product_batched::<EF, 1>(&vs1));
            });

            // Measure batched<2> benefit
            group.bench_with_input(BenchmarkId::new("unbatched_x2", &param), &(), |b, _| {
                b.iter(|| (m.columnwise_dot_product(&v1), m.columnwise_dot_product(&v2)));
            });
            group.bench_with_input(BenchmarkId::new("batched<2>", &param), &(), |b, _| {
                b.iter(|| m.columnwise_dot_product_batched::<EF, 2>(&vs2));
            });
        }
    }

    group.finish();
}

criterion_group!(benches, columnwise_dot_product);
criterion_main!(benches);
