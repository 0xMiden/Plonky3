use alloc::vec::Vec;
use core::iter::zip;
use core::marker::PhantomData;

use p3_commit::{BatchOpening, BatchOpeningRef, Mmcs};
use p3_field::PackedValue;
use p3_matrix::{Dimensions, Matrix};
use p3_symmetric::{Hash, PseudoCompressionFunction, StatefulHasher};
use p3_util::log2_strict_usize;
use serde::{Deserialize, Serialize};
use thiserror::Error;

mod hiding_lmcs;
mod lifted_tree;
#[cfg(test)]
mod test_helpers;

pub use hiding_lmcs::MerkleTreeHidingLmcs;
pub use lifted_tree::{LiftedMerkleTree, build_leaf_states_cyclic, build_leaf_states_upsampled};

/// Lifting method used to align matrices of different heights to a common height.
///
/// Consider matrices `M_0, …, M_{t-1}` with heights `h_0 ≤ h_1 ≤ … ≤ h_{t-1}` (each a power of
/// two), and let `H = h_{t-1}`. For each matrix `M_i`, lifting defines a row-index mapping
/// `f_i: {0,…,H-1} → {0,…,h_i-1}` and thereby a virtual height-`H` matrix `M_i^↑` of the same
/// width, whose `r`-th row is `row_{f_i(r)}(M_i)`. In other words, lifting “extends” each matrix
/// vertically to height `H` without changing its width. The LMCS leaf at position `r` then uses,
/// in order, the row `r` from each lifted matrix `M_i^↑` as input to the sponge (with per-matrix
/// zero padding to a multiple of the hasher's padding width for absorption).
///
/// Two canonical choices for `f_i` are supported:
/// - Upsample (nearest-neighbor): each original row is repeated contiguously
///   `s_i = H / h_i` times; with `s_i = 2^k`, we have `f_i(r) = r >> k = floor(r / s_i)`.
/// - Cyclic: the entire `h_i`-row matrix is repeated periodically until height `H`;
///   equivalently `f_i(r) = r mod h_i = r & (h_i - 1)`.
///
/// Example (h_i = 4, H = 8):
/// - Original rows of `M_i`: `[r0, r1, r2, r3]`.
/// - Upsample (s_i = 2): `M_i^↑` rows by index `r = 0..7` are
///   `[r0, r0, r1, r1, r2, r2, r3, r3]` (blocks of length 2).
/// - Cyclic: `M_i^↑` rows are `[r0, r1, r2, r3, r0, r1, r2, r3]` (period 4).
///
/// Summary view:
/// - Upsample lifting: virtually extend by repeating each row `s_i` times (blocks of identical
///   rows), width unchanged.
/// - Cyclic lifting: virtually extend by tiling the `h_i` original rows `s_i` times, width
///   unchanged.
///
/// The implementation realizes these semantics incrementally by maintaining one sponge state per
/// final row `r ∈ [0, H)`. As taller matrices are processed, states are duplicated contiguously
/// (upsample) or tiled in cycles (cyclic) so that each state continues to absorb
/// `row_{f_i(r)}(M_i)` from the current matrix’s virtual extension.
///
/// Power-of-two requirement: All matrix heights and the final height `H` must be powers of two.
/// The implementation relies on this (via bit-shifts and masks) and rejects non-powers-of-two.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Lifting {
    /// Nearest-neighbor upsampling. For a matrix of height `h` lifted to `H`, each original row is
    /// duplicated contiguously `s = H / h` times, and the lifted index map is
    /// `r ↦ floor(r / s)` (with `s` a power of two). This produces blocks of identical rows.
    Upsample,
    /// Cyclic repetition. For a matrix of height `h` lifted to `H`, the lifted index map is
    /// `r ↦ r mod h`, i.e. rows repeat with period `h` across the final height.
    Cyclic,
}

impl Lifting {
    /// Map a final leaf row index `index ∈ [0, max_height)` to the corresponding row index of a
    /// particular matrix of height `height`, according to this lifting.
    ///
    /// Preconditions:
    /// - `height` and `max_height` are powers of two with `height ≤ max_height`.
    /// - `index < max_height`.
    ///
    /// Semantics:
    /// - Upsample: returns `floor(index / (max_height / height))`.
    /// - Cyclic: returns `index mod height`.
    ///
    /// # Panics
    /// Panics in debug builds if preconditions are violated.
    pub fn map_index(&self, index: usize, height: usize, max_height: usize) -> usize {
        debug_assert!(index < max_height, "index must be < max_height");
        debug_assert!(height.is_power_of_two(), "height must be power of two");
        debug_assert!(
            max_height.is_power_of_two(),
            "max_height must be power of two"
        );
        debug_assert!(height <= max_height, "height must be <= max_height ");

        match self {
            Self::Upsample => {
                let log_scaling_factor = log2_strict_usize(max_height / height);
                index >> log_scaling_factor
            }
            Self::Cyclic => index & (height - 1),
        }
    }
}

/// Lifted MMCS built on top of [`LiftedMerkleTree`].
///
/// Matrices of different heights are aligned to the tallest height via a chosen [`Lifting`]
/// (either upsampled/nearest-neighbor or cyclic repetition). Conceptually, each matrix is
/// virtually extended vertically to height `H` (width unchanged) and the leaf `r` absorbs the
/// `r`-th row from each extended matrix. See [`Lifting`] for the precise definition of the
/// row-index mapping.
///
/// Equivalent single-matrix view: the scheme is equivalent to lifting every matrix to height `H`,
/// padding each horizontally with zeros to a multiple of the hasher's padding width, and concatenating them side-by-
/// side into one matrix. The Merkle tree and verification behavior are identical to committing to
/// and opening that single concatenated matrix.
#[derive(Copy, Clone, Debug)]
pub struct MerkleTreeLmcs<PF, PD, H, C, const WIDTH: usize, const DIGEST_ELEMS: usize> {
    sponge: H,
    compress: C,
    lifting: Lifting,
    _phantom: PhantomData<(PF, PD)>,
}

impl<PF, PD, H, C, const WIDTH: usize, const DIGEST_ELEMS: usize>
    MerkleTreeLmcs<PF, PD, H, C, WIDTH, DIGEST_ELEMS>
{
    /// Create a new lifted Merkle tree commitment scheme.
    ///
    /// # Arguments
    ///
    /// - `sponge`: Stateful sponge for hashing matrix rows into leaf digests.
    /// - `compress`: 2-to-1 compression function for building internal tree nodes.
    /// - `lifting`: Strategy for aligning matrices of different heights (see [`Lifting`]).
    pub const fn new(sponge: H, compress: C, lifting: Lifting) -> Self {
        Self {
            sponge,
            compress,
            lifting,
            _phantom: PhantomData,
        }
    }

    /// Recompute the Merkle root from opened rows and an authentication path.
    ///
    /// Used internally during verification to reconstruct the root hash from:
    /// - `rows`: the opened matrix rows at the given index
    /// - `index`: the leaf index that was opened
    /// - `dimensions`: the dimensions of each committed matrix
    /// - `proof`: the Merkle authentication path (sibling digests)
    /// - `salt`: optional salt row that was absorbed after the matrix rows
    ///
    /// This absorbs the rows (and optional salt) into a fresh sponge state, squeezes to
    /// get the leaf digest, then follows the authentication path up to the root.
    ///
    /// # Errors
    /// Returns `LmcsError` if any validation fails (wrong dimensions, out of bounds index, etc.).
    fn compute_root(
        &self,
        rows: &[Vec<PF::Value>],
        index: usize,
        dimensions: &[Dimensions],
        proof: &[[PD::Value; DIGEST_ELEMS]],
        salt: Option<&[PF::Value]>,
    ) -> Result<Hash<PF::Value, PD::Value, DIGEST_ELEMS>, LmcsError>
    where
        PF: PackedValue + Default,
        PD: PackedValue + Default,
        H: StatefulHasher<PF::Value, [PD::Value; WIDTH], [PD::Value; DIGEST_ELEMS]>,
        C: PseudoCompressionFunction<[PD::Value; DIGEST_ELEMS], 2>,
    {
        // Verify that the number of opened rows matches the number of matrix dimensions
        if dimensions.len() != rows.len() {
            return Err(LmcsError::WrongBatchSize);
        }

        let final_height = dimensions.last().unwrap().height;
        // Verify that the leaf index is within the tree bounds
        if index >= final_height {
            return Err(LmcsError::IndexOutOfBounds {
                max_height: final_height,
                index,
            });
        }

        let expected_proof_len = log2_strict_usize(final_height);
        // Verify that the authentication path has the correct length for the tree height
        if proof.len() != expected_proof_len {
            return Err(LmcsError::WrongProofLen {
                expected: expected_proof_len,
                actual: proof.len(),
            });
        }

        let mut state = [PD::Value::default(); WIDTH];
        for (idx, (row, dimension)) in zip(rows, dimensions).enumerate() {
            let expected_width = dimension.width;
            if row.len() != expected_width {
                return Err(LmcsError::WrongWidth {
                    matrix: idx,
                    expected: expected_width,
                    actual: row.len(),
                });
            }
            self.sponge.absorb_into(&mut state, row.iter().copied());
        }

        if let Some(salt) = salt {
            self.sponge.absorb_into(&mut state, salt.iter().copied());
        }

        let mut digest = self.sponge.squeeze(&state);

        let mut current_index = index;
        for sibling in proof {
            let (left, right) = if current_index & 1 == 0 {
                (digest, *sibling)
            } else {
                (*sibling, digest)
            };
            digest = self.compress.compress([left, right]);
            current_index >>= 1;
        }

        Ok(digest.into())
    }
}

impl<PF, PD, H, C, const WIDTH: usize, const DIGEST_ELEMS: usize> Mmcs<PF::Value>
    for MerkleTreeLmcs<PF, PD, H, C, WIDTH, DIGEST_ELEMS>
where
    PF: PackedValue + Default,
    PD: PackedValue + Default,
    H: StatefulHasher<PF, [PD; WIDTH], [PD; DIGEST_ELEMS]>
        + StatefulHasher<PF::Value, [PD::Value; WIDTH], [PD::Value; DIGEST_ELEMS]>
        + Sync,
    C: PseudoCompressionFunction<[PD::Value; DIGEST_ELEMS], 2>
        + PseudoCompressionFunction<[PD; DIGEST_ELEMS], 2>
        + Sync,
    [PD::Value; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
{
    type ProverData<M> = LiftedMerkleTree<PF::Value, PD::Value, M, DIGEST_ELEMS>;
    type Commitment = Hash<PF::Value, PD::Value, DIGEST_ELEMS>;
    type Proof = Vec<[PD::Value; DIGEST_ELEMS]>;
    type Error = LmcsError;

    fn commit<M: Matrix<PF::Value>>(
        &self,
        inputs: Vec<M>,
    ) -> (Self::Commitment, Self::ProverData<M>) {
        let tree = LiftedMerkleTree::new_with_optional_salt::<PF, PD, H, C, WIDTH>(
            &self.sponge,
            &self.compress,
            self.lifting,
            inputs,
            None,
        )
        .expect("tree construction failed");
        let root = tree.root();

        (root, tree)
    }

    fn open_batch<M: Matrix<PF::Value>>(
        &self,
        index: usize,
        tree: &Self::ProverData<M>,
    ) -> BatchOpening<PF::Value, Self> {
        let final_height = tree.height();
        assert!(
            index < final_height,
            "index {index} out of range {final_height}"
        );

        let opened_rows = tree.rows(index);

        let proof = tree.authentication_path(index);

        BatchOpening::new(opened_rows, proof)
    }

    fn get_matrices<'a, M: Matrix<PF::Value>>(&self, tree: &'a Self::ProverData<M>) -> Vec<&'a M> {
        // Return references to the originally committed matrices in original order.
        tree.leaves.iter().collect()
    }

    fn verify_batch(
        &self,
        commit: &Self::Commitment,
        dimensions: &[Dimensions],
        index: usize,
        batch_opening: BatchOpeningRef<'_, PF::Value, Self>,
    ) -> Result<(), Self::Error> {
        let (opened_values, opening_proof) = batch_opening.unpack();

        let expected_root =
            self.compute_root(opened_values, index, dimensions, opening_proof, None)?;

        if &expected_root == commit {
            Ok(())
        } else {
            Err(LmcsError::RootMismatch)
        }
    }
}

/// Errors that can arise while building or verifying lifted Merkle commitments.
#[derive(Debug, Error)]
pub enum LmcsError {
    /// Number of opened rows doesn't match number of committed matrices.
    #[error("wrong batch size: number of opened rows doesn't match committed matrices")]
    WrongBatchSize,
    /// Opened row width doesn't match the committed matrix width.
    #[error("wrong width at matrix {matrix}: expected {expected}, got {actual}")]
    WrongWidth {
        /// Index of the matrix with mismatched width.
        matrix: usize,
        /// Expected width from commitment dimensions.
        expected: usize,
        /// Actual width of the opened row.
        actual: usize,
    },
    /// Salt row length doesn't match expected width.
    #[error("wrong salt width: expected {expected}, got {actual}")]
    WrongSalt {
        /// Expected salt width.
        expected: usize,
        /// Actual salt width provided.
        actual: usize,
    },
    /// Matrix height doesn't match expected height.
    #[error("wrong height: expected {expected}, got {actual}")]
    WrongHeight {
        /// Expected height.
        expected: usize,
        /// Actual height.
        actual: usize,
    },
    /// Authentication path length doesn't match tree height.
    #[error("wrong proof length: expected {expected}, got {actual}")]
    WrongProofLen {
        /// Expected proof length (log₂ of tree height).
        expected: usize,
        /// Actual proof length provided.
        actual: usize,
    },
    /// Query index exceeds tree height.
    #[error("index {index} out of bounds (max height: {max_height})")]
    IndexOutOfBounds {
        /// Maximum valid index (tree height).
        max_height: usize,
        /// Requested index.
        index: usize,
    },
    /// Recomputed root doesn't match the commitment.
    #[error("root mismatch: recomputed root doesn't match commitment")]
    RootMismatch,
    /// No matrices provided for commitment.
    #[error("empty batch: no matrices provided for commitment")]
    EmptyBatch,
    /// Matrix height is not a power of two (required for lifting).
    #[error("non-power-of-two height at matrix {matrix}: height {height}")]
    NonPowerOfTwoHeight {
        /// Index of the invalid matrix.
        matrix: usize,
        /// The non-power-of-two height.
        height: usize,
    },
    /// Matrix height doesn't divide the final tree height.
    #[error(
        "height not divisor at matrix {matrix}: height {height} doesn't divide final height {final_height}"
    )]
    HeightNotDivisor {
        /// Index of the invalid matrix.
        matrix: usize,
        /// Height of the matrix.
        height: usize,
        /// Final tree height (must be divisible by matrix height).
        final_height: usize,
    },
    /// Matrix has zero height.
    #[error("zero height matrix at index {matrix}")]
    ZeroHeightMatrix {
        /// Index of the zero-height matrix.
        matrix: usize,
    },
    /// Matrices are not sorted by non-decreasing height.
    #[error("unsorted by height: matrices must be sorted by non-decreasing height")]
    UnsortedByHeight,
    /// Salt matrix height doesn't match tree height.
    #[error("salt height mismatch: expected {expected}, got {actual}")]
    SaltHeightMismatch {
        /// Expected salt height (equal to tree height).
        expected: usize,
        /// Actual salt matrix height.
        actual: usize,
    },
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::Poseidon2BabyBear;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::rngs::SmallRng;
    use rand::{RngCore, SeedableRng};

    use super::test_helpers::{DIGEST, F, P, RATE, WIDTH};
    use super::*;

    fn components() -> (
        PaddingFreeSponge<Poseidon2BabyBear<WIDTH>, WIDTH, RATE, DIGEST>,
        TruncatedPermutation<Poseidon2BabyBear<WIDTH>, 2, DIGEST, WIDTH>,
    ) {
        let mut rng = SmallRng::seed_from_u64(123);
        let perm = Poseidon2BabyBear::<WIDTH>::new_from_rng_128(&mut rng);
        let sponge = PaddingFreeSponge::<_, WIDTH, RATE, DIGEST>::new(perm.clone());
        let compress = TruncatedPermutation::<_, 2, DIGEST, WIDTH>::new(perm);
        (sponge, compress)
    }

    fn rand_matrix(h: usize, w: usize, rng: &mut SmallRng) -> RowMajorMatrix<F> {
        let vals = (0..h * w).map(|_| F::new(rng.next_u32())).collect();
        RowMajorMatrix::new(vals, w)
    }

    #[test]
    fn commit_open_verify_roundtrip() {
        let (sponge, compress) = components();
        let lmcs =
            MerkleTreeLmcs::<P, P, _, _, WIDTH, DIGEST>::new(sponge, compress, Lifting::Upsample);

        let mut rng = SmallRng::seed_from_u64(9);
        let matrices = vec![
            rand_matrix(2, 3, &mut rng),
            rand_matrix(4, 5, &mut rng),
            rand_matrix(8, 7, &mut rng),
        ];
        let dims: Vec<Dimensions> = matrices
            .iter()
            .map(|m: &RowMajorMatrix<F>| m.dimensions())
            .collect();

        let (commitment, tree) = lmcs.commit(matrices);
        let final_height = dims.last().unwrap().height;
        let index = final_height - 1; // valid index within range

        let opening = lmcs.open_batch(index, &tree);
        let opening_ref: BatchOpeningRef<'_, F, _> = (&opening).into();
        assert!(
            lmcs.verify_batch(&commitment, &dims, index, opening_ref)
                .is_ok()
        );
    }
}
