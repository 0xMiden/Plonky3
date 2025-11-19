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

pub use merkle_tree::{LiftedMerkleTree, Lifting, build_leaves_cyclic, build_leaves_upsampled};

use crate::lmcs::utils::validate_heights;

/// Lifted MMCS built on top of [`LiftedMerkleTree`].
///
/// Matrices of different heights are aligned to the tallest height via a chosen [`Lifting`]
/// (either upsampled/nearest-neighbor or cyclic repetition). Conceptually, each matrix is
/// virtually extended vertically to height `H` (width unchanged) and the leaf `r` absorbs the
/// `r`-th row from each extended matrix. See [`Lifting`] for the precise definition of the
/// row-index mapping.
///
/// Equivalent single-matrix view: the scheme is equivalent to lifting every matrix to height `H`,
/// padding each horizontally with zeros to a multiple of `RATE`, and concatenating them side-by-
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
        let opened_rows = tree.rows(index);

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

        for (idx, (opened_row, dimension)) in zip(opened_values, dimensions).enumerate() {
            if opened_row.len() != dimension.width {
                return Err(LmcsError::WrongWidth {
                    matrix: idx,
                    expected: dimension.width,
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
    use p3_symmetric::{StatefulSponge, TruncatedPermutation};
    use rand::rngs::SmallRng;
    use rand::{RngCore, SeedableRng};

    use super::test_helpers::{DIGEST, F, P, RATE, WIDTH};
    use super::*;

    fn components() -> (
        StatefulSponge<Poseidon2BabyBear<WIDTH>, WIDTH, DIGEST, RATE>,
        TruncatedPermutation<Poseidon2BabyBear<WIDTH>, 2, DIGEST, WIDTH>,
    ) {
        let mut rng = SmallRng::seed_from_u64(123);
        let perm = Poseidon2BabyBear::<WIDTH>::new_from_rng_128(&mut rng);
        let sponge = StatefulSponge::<_, WIDTH, DIGEST, RATE> { p: perm.clone() };
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
        let lmcs = MerkleTreeLmcs::<P, P, _, _, WIDTH, DIGEST>::new(
            sponge.clone(),
            compress.clone(),
            Lifting::Upsample,
        );

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
