//! Shared utilities for lifted crate benchmarks.
//!
//! Include in benchmark files with:
//! ```ignore
//! #[path = "bench_utils.rs"]
//! mod bench_utils;
//! ```

use p3_field::Field;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use rand::SeedableRng;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;

/// Standard relative specs for benchmark matrix groups.
///
/// Each inner vec is a separate commitment group.
/// Tuple format: `(offset_from_max, width)` where `log_height = log_max_height - offset`.
///
/// This gives realistic matrix configurations similar to STARK traces:
/// - Group 0: Main trace columns at various heights
/// - Group 1: Auxiliary/permutation columns
/// - Group 2: Quotient polynomial chunks
pub const RELATIVE_SPECS: &[&[(usize, usize)]] = &[
    &[(4, 10), (2, 100), (0, 50)],
    &[(4, 8), (2, 20), (0, 20)],
    &[(0, 16)],
];

/// Standard log heights for benchmarking: 2^16, 2^18, 2^20 leaves.
pub const LOG_HEIGHTS: &[usize] = &[16, 18, 20];

/// Parallelism mode string for benchmark grouping.
pub const PARALLEL_STR: &str = if cfg!(feature = "parallel") {
    "parallel"
} else {
    "single"
};

/// Generate benchmark matrices from relative specs.
///
/// Creates matrices with heights relative to `max_height = 1 << log_max_height`.
/// Each spec `(offset, width)` creates a matrix with:
/// - height = `max_height >> offset`
/// - width = `width`
///
/// Matrices in each group are sorted by ascending height.
pub fn generate_matrices_from_specs<F: Field>(
    specs: &[&[(usize, usize)]],
    log_max_height: usize,
) -> Vec<Vec<RowMajorMatrix<F>>>
where
    StandardUniform: Distribution<F>,
{
    let rng = &mut SmallRng::seed_from_u64(42);
    let max_height = 1usize << log_max_height;

    specs
        .iter()
        .map(|group_specs| {
            let mut matrices: Vec<RowMajorMatrix<F>> = group_specs
                .iter()
                .map(|&(offset, width)| {
                    let height = max_height >> offset;
                    RowMajorMatrix::rand(rng, height, width)
                })
                .collect();
            // Sort by ascending height (required by LMCS)
            matrices.sort_by_key(|m| m.height());
            matrices
        })
        .collect()
}

/// Generate a single flat list of matrices from relative specs.
///
/// Useful when you don't need separate commitment groups.
#[allow(dead_code)]
pub fn generate_flat_matrices<F: Field>(
    specs: &[(usize, usize)],
    log_max_height: usize,
) -> Vec<RowMajorMatrix<F>>
where
    StandardUniform: Distribution<F>,
{
    let rng = &mut SmallRng::seed_from_u64(42);
    let max_height = 1usize << log_max_height;

    let mut matrices: Vec<RowMajorMatrix<F>> = specs
        .iter()
        .map(|&(offset, width)| {
            let height = max_height >> offset;
            RowMajorMatrix::rand(rng, height, width)
        })
        .collect();
    matrices.sort_by_key(|m| m.height());
    matrices
}

/// Calculate total elements across all matrices.
pub fn total_elements<F: Clone + Send + Sync>(matrix_groups: &[Vec<RowMajorMatrix<F>>]) -> u64 {
    matrix_groups
        .iter()
        .flat_map(|g| g.iter())
        .map(|m| {
            let dims = m.dimensions();
            (dims.height * dims.width) as u64
        })
        .sum()
}

/// Calculate total elements for a flat matrix list.
#[allow(dead_code)]
pub fn total_elements_flat<F: Clone + Send + Sync>(matrices: &[RowMajorMatrix<F>]) -> u64 {
    matrices
        .iter()
        .map(|m| {
            let dims = m.dimensions();
            (dims.height * dims.width) as u64
        })
        .sum()
}
