use alloc::vec::Vec;
use core::iter::zip;
use core::marker::PhantomData;

use p3_commit::{BatchOpening, BatchOpeningRef, Mmcs};
use p3_field::PackedValue;
use p3_matrix::{Dimensions, Matrix};
use p3_symmetric::{Hash, PseudoCompressionFunction, StatefulHasher};
use p3_util::log2_strict_usize;
use serde::{Deserialize, Serialize};

mod merkle_tree;
#[cfg(test)]
mod test_helpers;
mod utils;

pub use merkle_tree::{LiftedMerkleTree, build_leaves_cyclic, build_leaves_upsampled};

use crate::lmcs::utils::validate_heights;

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
    pub fn map_index(&self, index: usize, height: usize, max_height: usize) -> usize {
        assert!(index < max_height);
        assert!(height.is_power_of_two());
        assert!(max_height.is_power_of_two());
        assert!(height <= max_height);

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
#[derive(Clone, Debug)]
pub struct MerkleTreeLmcs<PF, PD, H, C, const WIDTH: usize, const DIGEST_ELEMS: usize> {
    sponge: H,
    compress: C,
    lifting: Lifting,
    _phantom: PhantomData<(PF, PD)>,
}

impl<PF, PD, H, C, const WIDTH: usize, const DIGEST_ELEMS: usize>
    MerkleTreeLmcs<PF, PD, H, C, WIDTH, DIGEST_ELEMS>
{
    pub const fn new(sponge: H, compress: C, lifting: Lifting) -> Self {
        Self {
            sponge,
            compress,
            lifting,
            _phantom: PhantomData,
        }
    }
}

impl<PF, PD, H, C, const WIDTH: usize, const DIGEST_ELEMS: usize> Mmcs<PF::Value>
    for MerkleTreeLmcs<PF, PD, H, C, WIDTH, DIGEST_ELEMS>
where
    PF: PackedValue + Default,
    PF::Value: Copy + Default + Send + Sync + Clone,
    PD: PackedValue + Default,
    PD::Value: Copy + Default + Send + Sync + Clone,
    H: StatefulHasher<PF, [PD; WIDTH], [PD; DIGEST_ELEMS]>
        + StatefulHasher<PF::Value, [PD::Value; WIDTH], [PD::Value; DIGEST_ELEMS]>
        + Sync
        + Clone,
    C: PseudoCompressionFunction<[PD::Value; DIGEST_ELEMS], 2>
        + PseudoCompressionFunction<[PD; DIGEST_ELEMS], 2>
        + Sync
        + Clone,
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
        let tree = LiftedMerkleTree::new::<PF, PD, H, C, WIDTH>(
            &self.sponge,
            &self.compress,
            self.lifting,
            inputs,
        )
        .unwrap();
        let root = tree.root();

        (root, tree)
    }

    fn open_batch<M: Matrix<PF::Value>>(
        &self,
        index: usize,
        tree: &Self::ProverData<M>,
    ) -> BatchOpening<PF::Value, Self> {
        let final_height = tree.leaves.last().unwrap().height();
        assert!(
            index < final_height,
            "index {index} out of range {final_height}"
        );

        // Map to per-matrix indices.
        let mut opened_rows = tree.rows(index);
        // Pad each row to a multiple of the hasher's padding width with zeros.
        let pad = <H as StatefulHasher<PF::Value, _, _>>::PADDING_WIDTH;
        if pad > 1 {
            for row in &mut opened_rows {
                let target = row.len().next_multiple_of(pad);
                if target > row.len() {
                    row.resize(target, PF::Value::default());
                }
            }
        }

        let mut proof = Vec::with_capacity(tree.digest_layers.len().saturating_sub(1));
        let mut layer_index = index;
        for layer in &tree.digest_layers {
            if layer.len() == 1 {
                break;
            }
            let sibling = layer[layer_index ^ 1];
            proof.push(sibling);
            layer_index >>= 1;
        }

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

        validate_heights(dimensions.iter().map(|d| d.height))?;

        if dimensions.len() != opened_values.len() {
            return Err(LmcsError::WrongBatchSize);
        }

        let final_height = dimensions.last().unwrap().height;
        if index >= final_height {
            return Err(LmcsError::IndexOutOfBounds {
                max_height: final_height,
                index,
            });
        }

        let expected_proof_len = log2_strict_usize(final_height);
        if opening_proof.len() != expected_proof_len {
            return Err(LmcsError::WrongHeight {
                expected: expected_proof_len,
                actual: opening_proof.len(),
            });
        }

        let pad = <H as StatefulHasher<PF::Value, _, _>>::PADDING_WIDTH;
        for (idx, (opened_row, dimension)) in zip(opened_values, dimensions).enumerate() {
            let expected_width = if pad > 1 {
                dimension.width.next_multiple_of(pad)
            } else {
                dimension.width
            };
            if opened_row.len() != expected_width {
                return Err(LmcsError::WrongWidth {
                    matrix: idx,
                    expected: expected_width,
                    actual: opened_row.len(),
                });
            }
        }

        let sponge_state =
            opened_values
                .iter()
                .fold([PD::Value::default(); WIDTH], |mut state, opened_row| {
                    self.sponge
                        .absorb_into(&mut state, opened_row.iter().copied());
                    state
                });

        let mut digest = self.sponge.squeeze(&sponge_state);
        let mut current_index = index;
        for sibling in opening_proof {
            let (left, right) = if current_index & 1 == 0 {
                (digest, *sibling)
            } else {
                (*sibling, digest)
            };
            digest = self.compress.compress([left, right]);
            current_index >>= 1;
        }

        let expected_root: Hash<_, _, DIGEST_ELEMS> = digest.into();
        if &expected_root == commit {
            Ok(())
        } else {
            Err(LmcsError::RootMismatch)
        }
    }
}

/// Errors that can arise while verifying LMCS openings.
#[derive(Debug)]
pub enum LmcsError {
    WrongBatchSize,
    WrongWidth {
        matrix: usize,
        expected: usize,
        actual: usize,
    },
    WrongHeight {
        expected: usize,
        actual: usize,
    },
    IndexOutOfBounds {
        max_height: usize,
        index: usize,
    },
    RootMismatch,
    EmptyBatch,
    NonPowerOfTwoHeight {
        matrix: usize,
        height: usize,
    },
    HeightNotDivisor {
        matrix: usize,
        height: usize,
        final_height: usize,
    },
    ZeroHeightMatrix {
        matrix: usize,
    },
    UnsortedByHeight,
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
