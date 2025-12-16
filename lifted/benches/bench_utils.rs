//! Shared utilities and fixtures for lifted crate benchmarks.
//!
//! This module provides field-agnostic benchmark fixtures via feature flags.
//!
//! ## Feature Flags
//!
//! **Field selection** (mutually exclusive, exactly one required):
//! - `bench-babybear`: Use BabyBear field with degree-4 extension
//! - `bench-goldilocks`: Use Goldilocks field with degree-2 extension
//!
//! **Hash selection** (mutually exclusive, exactly one required):
//! - `bench-poseidon2`: Use Poseidon2 hash
//! - `bench-keccak`: Use Keccak hash
//!
//! ## Usage
//!
//! Include in benchmark files with:
//! ```ignore
//! #[path = "bench_utils.rs"]
//! mod bench_utils;
//! ```
//!
//! Run benchmarks with:
//! ```bash
//! RUSTFLAGS="-Ctarget-cpu=native" cargo bench --bench <name> \
//!     --features bench-babybear,bench-poseidon2
//! ```

#![allow(dead_code)]

use p3_field::Field;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use rand::SeedableRng;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;

// =============================================================================
// Compile-time feature validation
// =============================================================================

#[cfg(all(feature = "bench-babybear", feature = "bench-goldilocks"))]
compile_error!("Features `bench-babybear` and `bench-goldilocks` are mutually exclusive");

#[cfg(not(any(feature = "bench-babybear", feature = "bench-goldilocks")))]
compile_error!("One of `bench-babybear` or `bench-goldilocks` must be enabled");

#[cfg(all(feature = "bench-poseidon2", feature = "bench-keccak"))]
compile_error!("Features `bench-poseidon2` and `bench-keccak` are mutually exclusive");

#[cfg(not(any(feature = "bench-poseidon2", feature = "bench-keccak")))]
compile_error!("One of `bench-poseidon2` or `bench-keccak` must be enabled");

// =============================================================================
// Field type aliases (selected via feature flag)
// =============================================================================

#[cfg(feature = "bench-babybear")]
pub use p3_baby_bear::BabyBear as F;
#[cfg(feature = "bench-goldilocks")]
pub use p3_goldilocks::Goldilocks as F;

/// Extension field type.
#[cfg(feature = "bench-babybear")]
pub type EF = p3_field::extension::BinomialExtensionField<F, 4>;

#[cfg(feature = "bench-goldilocks")]
pub type EF = p3_field::extension::BinomialExtensionField<F, 2>;

/// Packed base field for SIMD operations.
pub type P = <F as Field>::Packing;

/// Field name for benchmark labels.
#[cfg(feature = "bench-babybear")]
pub const FIELD_NAME: &str = "babybear";

#[cfg(feature = "bench-goldilocks")]
pub const FIELD_NAME: &str = "goldilocks";

// =============================================================================
// Hash configuration (selected via feature flag)
// =============================================================================

// --- Poseidon2 for BabyBear: width=24, rate=16, digest=8 ---
#[cfg(all(feature = "bench-babybear", feature = "bench-poseidon2"))]
pub mod hash {
    use p3_baby_bear::Poseidon2BabyBear;
    use p3_lifted::merkle_tree::MerkleTreeLmcs;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::{F, P};

    pub const WIDTH: usize = 24;
    pub const RATE: usize = 16;
    pub const DIGEST: usize = 8;
    pub const HASH_NAME: &str = "poseidon2";

    pub type Perm = Poseidon2BabyBear<WIDTH>;
    pub type Sponge = PaddingFreeSponge<Perm, WIDTH, RATE, DIGEST>;
    pub type Compress = TruncatedPermutation<Perm, 2, DIGEST, WIDTH>;

    // Scalar LMCS
    pub type ScalarLmcs = MerkleTreeLmcs<F, F, Sponge, Compress, WIDTH, DIGEST>;
    pub type ScalarMmcs = MerkleTreeMmcs<F, F, Sponge, Compress, DIGEST>;

    // Packed LMCS
    pub type PackedLmcs = MerkleTreeLmcs<P, P, Sponge, Compress, WIDTH, DIGEST>;
    pub type PackedMmcs = MerkleTreeMmcs<P, P, Sponge, Compress, DIGEST>;

    pub fn components() -> (Sponge, Compress) {
        let mut rng = SmallRng::seed_from_u64(2025);
        let perm = Perm::new_from_rng_128(&mut rng);
        let sponge = Sponge::new(perm.clone());
        let compress = Compress::new(perm);
        (sponge, compress)
    }
}

// --- Poseidon2 for Goldilocks: width=12, rate=8, digest=4
#[cfg(all(feature = "bench-goldilocks", feature = "bench-poseidon2"))]
pub mod hash {
    use p3_goldilocks::Poseidon2Goldilocks;
    use p3_lifted::merkle_tree::MerkleTreeLmcs;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::{F, P};

    pub const WIDTH: usize = 12;
    pub const RATE: usize = 8;
    pub const DIGEST: usize = 4;
    pub const HASH_NAME: &str = "poseidon2";

    pub type Perm = Poseidon2Goldilocks<WIDTH>;
    pub type Sponge = PaddingFreeSponge<Perm, WIDTH, RATE, DIGEST>;
    pub type Compress = TruncatedPermutation<Perm, 2, DIGEST, WIDTH>;

    // Scalar LMCS
    pub type ScalarLmcs = MerkleTreeLmcs<F, F, Sponge, Compress, WIDTH, DIGEST>;
    pub type ScalarMmcs = MerkleTreeMmcs<F, F, Sponge, Compress, DIGEST>;

    // Packed LMCS
    pub type PackedLmcs = MerkleTreeLmcs<P, P, Sponge, Compress, WIDTH, DIGEST>;
    pub type PackedMmcs = MerkleTreeMmcs<P, P, Sponge, Compress, DIGEST>;

    pub fn components() -> (Sponge, Compress) {
        let mut rng = SmallRng::seed_from_u64(2025);
        let perm = Perm::new_from_rng_128(&mut rng);
        let sponge = Sponge::new(perm.clone());
        let compress = Compress::new(perm);
        (sponge, compress)
    }
}

// --- Keccak for BabyBear ---
#[cfg(all(feature = "bench-babybear", feature = "bench-keccak"))]
pub mod hash {
    use p3_keccak::KeccakF;
    use p3_lifted::merkle_tree::MerkleTreeLmcs;
    use p3_symmetric::{ChainingHasher, PaddingFreeSponge, TruncatedPermutation};

    use super::F;

    pub const WIDTH: usize = 4;
    pub const RATE: usize = 16; // Alignment for row hashing
    pub const DIGEST: usize = 4;
    pub const HASH_NAME: &str = "keccak";

    // Keccak uses u64 digests
    type KInner = PaddingFreeSponge<KeccakF, 25, 17, DIGEST>;
    pub type Sponge = ChainingHasher<KInner>;
    pub type Compress = TruncatedPermutation<KeccakF, 2, DIGEST, 25>;

    // Scalar LMCS (F -> u64 digest)
    pub type ScalarLmcs = MerkleTreeLmcs<F, u64, Sponge, Compress, WIDTH, DIGEST>;
    // Note: MerkleTreeMmcs with keccak doesn't work with field elements directly
    // because ChainingHasher doesn't implement CryptographicHasher<F, [u64; N]>

    // Packed LMCS (vectorized keccak)
    pub const K_VEC: usize = p3_keccak::VECTOR_LEN;
    pub type PackedLmcs = MerkleTreeLmcs<[F; K_VEC], [u64; K_VEC], Sponge, Compress, WIDTH, DIGEST>;

    static K_INNER: KInner = PaddingFreeSponge::new(KeccakF);

    pub fn components() -> (Sponge, Compress) {
        let sponge = ChainingHasher::new(K_INNER);
        let compress = TruncatedPermutation::new(KeccakF);
        (sponge, compress)
    }
}

// --- Keccak for Goldilocks ---
#[cfg(all(feature = "bench-goldilocks", feature = "bench-keccak"))]
pub mod hash {
    use p3_keccak::KeccakF;
    use p3_lifted::merkle_tree::MerkleTreeLmcs;
    use p3_symmetric::{ChainingHasher, PaddingFreeSponge, TruncatedPermutation};

    use super::F;

    pub const WIDTH: usize = 4;
    pub const RATE: usize = 8; // Alignment for row hashing
    pub const DIGEST: usize = 4;
    pub const HASH_NAME: &str = "keccak";

    // Keccak uses u64 digests
    type KInner = PaddingFreeSponge<KeccakF, 25, 17, DIGEST>;
    pub type Sponge = ChainingHasher<KInner>;
    pub type Compress = TruncatedPermutation<KeccakF, 2, DIGEST, 25>;

    // Scalar LMCS (F -> u64 digest)
    pub type ScalarLmcs = MerkleTreeLmcs<F, u64, Sponge, Compress, WIDTH, DIGEST>;
    // Note: MerkleTreeMmcs with keccak doesn't work with field elements directly
    // because ChainingHasher doesn't implement CryptographicHasher<F, [u64; N]>

    // Packed LMCS (vectorized keccak)
    pub const K_VEC: usize = p3_keccak::VECTOR_LEN;
    pub type PackedLmcs = MerkleTreeLmcs<[F; K_VEC], [u64; K_VEC], Sponge, Compress, WIDTH, DIGEST>;

    static K_INNER: KInner = PaddingFreeSponge::new(KeccakF);

    pub fn components() -> (Sponge, Compress) {
        let sponge = ChainingHasher::new(K_INNER);
        let compress = TruncatedPermutation::new(KeccakF);
        (sponge, compress)
    }
}

// =============================================================================
// Benchmark constants
// =============================================================================

/// Standard relative specs for benchmark matrix groups.
///
/// Each inner slice is a separate commitment group.
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

/// Standard seed for reproducible benchmarks.
pub const BENCH_SEED: u64 = 2025;

// =============================================================================
// Matrix generation utilities
// =============================================================================

/// Generate benchmark matrices from relative specs.
///
/// Creates matrices with heights relative to `max_height = 1 << log_max_height`.
/// Each spec `(offset, width)` creates a matrix with:
/// - height = `max_height >> offset`
/// - width = `width`
///
/// Matrices in each group are sorted by ascending height.
pub fn generate_matrices_from_specs(
    specs: &[&[(usize, usize)]],
    log_max_height: usize,
) -> Vec<Vec<RowMajorMatrix<F>>>
where
    StandardUniform: Distribution<F>,
{
    let rng = &mut SmallRng::seed_from_u64(BENCH_SEED);
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

/// Generate a single flat matrix for FRI fold benchmarks.
pub fn generate_flat_matrix(log_height: usize, width: usize) -> RowMajorMatrix<F>
where
    StandardUniform: Distribution<F>,
{
    let rng = &mut SmallRng::seed_from_u64(BENCH_SEED);
    RowMajorMatrix::rand(rng, 1 << log_height, width)
}

/// Calculate total elements across all matrices.
pub fn total_elements(matrix_groups: &[Vec<RowMajorMatrix<F>>]) -> u64 {
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
pub fn total_elements_flat(matrices: &[RowMajorMatrix<F>]) -> u64 {
    matrices
        .iter()
        .map(|m| {
            let dims = m.dimensions();
            (dims.height * dims.width) as u64
        })
        .sum()
}

// =============================================================================
// Benchmark ID formatting
// =============================================================================

/// Create a benchmark ID with the standard ordering: [size][hash][parallel][extra...]
pub fn bench_id(log_size: usize, extra: &str) -> String {
    format!(
        "{}/{}/{}/{}",
        1usize << log_size,
        hash::HASH_NAME,
        PARALLEL_STR,
        extra
    )
}

/// Create a benchmark ID with packing info.
pub fn bench_id_packed(log_size: usize, packed: bool) -> String {
    let packing = if packed { "packed" } else { "scalar" };
    bench_id(log_size, packing)
}
