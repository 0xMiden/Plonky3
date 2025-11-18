use alloc::vec::Vec;
use core::iter::zip;
use core::marker::PhantomData;

use p3_commit::{BatchOpening, BatchOpeningRef, Mmcs};
use p3_field::PackedValue;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::lifted::UpsampledLiftIndexMap;
use p3_matrix::{Dimensions, Matrix};
use p3_symmetric::{Hash, PseudoCompressionFunction, StatefulSponge};
use p3_util::log2_strict_usize;
use serde::{Deserialize, Serialize};

use crate::uniform::{LiftDimensions, UniformMerkleTree};

/// Lifted MMCS built on top of [`UniformMerkleTree`].
///
/// The tree treats every matrix as if it were expanded to the tallest height by duplicating
/// rows, allowing the commitment to follow a uniform Merkle structure.
#[derive(Clone, Debug)]
pub struct MerkleTreeLmcs<P, H, C, const WIDTH: usize, const RATE: usize, const DIGEST_ELEMS: usize>
{
    sponge: H,
    compress: C,
    _phantom: PhantomData<P>,
}

impl<P, H, C, const WIDTH: usize, const RATE: usize, const DIGEST_ELEMS: usize>
    MerkleTreeLmcs<P, H, C, WIDTH, RATE, DIGEST_ELEMS>
{
    pub const fn new(sponge: H, compress: C) -> Self {
        Self {
            sponge,
            compress,
            _phantom: PhantomData,
        }
    }
}

impl<P, H, C, const WIDTH: usize, const RATE: usize, const DIGEST_ELEMS: usize> Mmcs<P::Value>
    for MerkleTreeLmcs<P, H, C, WIDTH, RATE, DIGEST_ELEMS>
where
    P: PackedValue + Default,
    P::Value: Copy + Default + Send + Sync + Clone,
    H: StatefulSponge<P, WIDTH, RATE> + StatefulSponge<P::Value, WIDTH, RATE> + Sync + Clone,
    C: PseudoCompressionFunction<[P::Value; DIGEST_ELEMS], 2>
        + PseudoCompressionFunction<[P; DIGEST_ELEMS], 2>
        + Sync
        + Clone,
    [P::Value; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
{
    type ProverData<M> = UniformMerkleTree<P::Value, M, DIGEST_ELEMS>;
    type Commitment = Hash<P::Value, P::Value, DIGEST_ELEMS>;
    type Proof = Vec<[P::Value; DIGEST_ELEMS]>;
    type Error = LmcsError;

    fn commit<M: Matrix<P::Value>>(
        &self,
        inputs: Vec<M>,
    ) -> (Self::Commitment, Self::ProverData<M>) {
        let tree =
            UniformMerkleTree::new::<P, H, C, WIDTH, RATE>(&self.sponge, &self.compress, inputs);
        let root = tree.root();

        (root, tree)
    }

    fn open_batch<M: Matrix<P::Value>>(
        &self,
        index: usize,
        tree: &Self::ProverData<M>,
    ) -> BatchOpening<P::Value, Self> {
        let lift = tree
            .lift_dims
            .as_ref()
            .expect("missing lift dimensions; tree must be constructed from matrices");
        let final_height = lift.largest_height();
        assert!(
            index < final_height,
            "index {index} out of range {final_height}"
        );

        // Map to per-matrix indices and pad to RATE-aligned widths.
        let mapped = lift.map_idxs_upsampled(index);
        let padded_widths = lift.padded_widths::<RATE>();
        let opened_rows: Vec<Vec<P::Value>> = tree
            .leaves
            .iter()
            .zip(mapped)
            .zip(padded_widths)
            .map(|((m, mapped_idx), padded_w)| {
                let mut row: Vec<_> = m
                    .row(mapped_idx)
                    .expect("invalid index")
                    .into_iter()
                    .collect();
                row.resize(padded_w, P::Value::default());
                row
            })
            .collect();

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

    fn get_matrices<'a, M: Matrix<P::Value>>(&self, tree: &'a Self::ProverData<M>) -> Vec<&'a M> {
        // Return references to the originally committed matrices in original order.
        tree.leaves.iter().collect()
    }

    fn verify_batch(
        &self,
        commit: &Self::Commitment,
        dimensions: &[Dimensions],
        index: usize,
        batch_opening: BatchOpeningRef<'_, P::Value, Self>,
    ) -> Result<(), Self::Error> {
        let (opened_values, opening_proof) = batch_opening.unpack();

        if dimensions.len() != opened_values.len() {
            return Err(LmcsError::WrongBatchSize);
        }

        let lift = LiftDimensions::new(dimensions.to_vec())?;
        let final_height = lift.largest_height();
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

        for (idx, (opened_row, padded_width)) in
            zip(opened_values, lift.padded_widths::<RATE>()).enumerate()
        {
            if padded_width != opened_row.len() {
                return Err(LmcsError::WrongWidth {
                    matrix: idx,
                    expected: padded_width,
                    actual: opened_row.len(),
                });
            }
        }

        let sponge_state = opened_values.iter().fold([P::Value::default(); WIDTH], |mut state, opened_row| {
            self.sponge.absorb(&mut state, opened_row.iter().copied());
            state
        });

        let mut digest = self.sponge.squeeze::<DIGEST_ELEMS>(&sponge_state);
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
    FinalHeightNotPowerOfTwo {
        height: usize,
    },
    /// Dimensions are not sorted by non-decreasing height.
    UnsortedByHeight,
}
