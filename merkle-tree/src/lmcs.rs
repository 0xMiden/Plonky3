use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_commit::{BatchOpening, BatchOpeningRef, Mmcs};
use p3_field::PackedValue;
use p3_matrix::{Dimensions, Matrix};
use p3_symmetric::{Hash, PseudoCompressionFunction, StatefulSponge};
use p3_util::log2_strict_usize;
use serde::{Deserialize, Serialize};

use crate::uniform::UniformMerkleTree;

/// Prover-side data for a lifted mixed-matrix commitment.
pub struct MerkleTreeLmcsProverData<F, M, const DIGEST_ELEMS: usize> {
    pub(crate) tree: UniformMerkleTree<F, M, DIGEST_ELEMS>,
    pub(crate) sorted_to_original: Vec<usize>,
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
}

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
    type ProverData<M> = MerkleTreeLmcsProverData<P::Value, M, DIGEST_ELEMS>;
    type Commitment = Hash<P::Value, P::Value, DIGEST_ELEMS>;
    type Proof = Vec<[P::Value; DIGEST_ELEMS]>;
    type Error = LmcsError;

    fn commit<M: Matrix<P::Value>>(
        &self,
        inputs: Vec<M>,
    ) -> (Self::Commitment, Self::ProverData<M>) {
        assert!(
            !inputs.is_empty(),
            "LMCS commit requires at least one matrix"
        );

        let mut enumerated: Vec<(usize, M)> = inputs.into_iter().enumerate().collect();
        enumerated.sort_by(|(idx_a, mat_a), (idx_b, mat_b)| {
            mat_a.height().cmp(&mat_b.height()).then(idx_a.cmp(idx_b))
        });

        let mut heights = Vec::with_capacity(enumerated.len());
        for (i, (_, matrix)) in enumerated.iter().enumerate() {
            let height = matrix.height();
            assert!(height > 0, "matrix {i} has zero height");
            assert!(
                height.is_power_of_two(),
                "matrix {i} height {height} must be a power of two"
            );
            heights.push(height);
        }

        let sorted_to_original: Vec<usize> = enumerated
            .iter()
            .map(|(original_idx, _)| *original_idx)
            .collect();
        let sorted_matrices: Vec<M> = enumerated.into_iter().map(|(_, matrix)| matrix).collect();

        let tree = UniformMerkleTree::new::<P, H, C, WIDTH, RATE>(
            &self.sponge,
            &self.compress,
            sorted_matrices,
        );
        let root = tree.root();

        (
            root,
            MerkleTreeLmcsProverData {
                tree,
                sorted_to_original,
            },
        )
    }

    fn open_batch<M: Matrix<P::Value>>(
        &self,
        index: usize,
        prover_data: &Self::ProverData<M>,
    ) -> BatchOpening<P::Value, Self> {
        let tree = &prover_data.tree;
        let num_matrices = tree.leaves.len();
        assert!(num_matrices > 0, "no matrices committed");

        let final_height = tree.leaves.last().expect("at least one matrix").height();
        assert!(
            index < final_height,
            "index {index} out of range {final_height}"
        );

        let mut opened_rows: Vec<Vec<P::Value>> = vec![Vec::new(); num_matrices];

        for (sorted_idx, matrix) in tree.leaves.iter().enumerate() {
            let height = matrix.height();
            let factor = final_height / height;
            let row_idx = index / factor;
            let row = matrix
                .row(row_idx)
                .unwrap_or_else(|| panic!("row {row_idx} missing in matrix {sorted_idx}"));
            let values: Vec<P::Value> = row.into_iter().collect();
            let original_idx = prover_data.sorted_to_original[sorted_idx];
            opened_rows[original_idx] = values;
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

    fn get_matrices<'a, M: Matrix<P::Value>>(
        &self,
        prover_data: &'a Self::ProverData<M>,
    ) -> Vec<&'a M> {
        let num = prover_data.tree.leaves.len();
        let mut ordered: Vec<Option<&M>> = vec![None; num];
        for (sorted_idx, matrix) in prover_data.tree.leaves.iter().enumerate() {
            let original_idx = prover_data.sorted_to_original[sorted_idx];
            ordered[original_idx] = Some(matrix);
        }
        ordered
            .into_iter()
            .map(|opt| opt.expect("matrix missing in Lmcs prover data"))
            .collect()
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
        if dimensions.is_empty() {
            return Err(LmcsError::EmptyBatch);
        }

        let mut order: Vec<usize> = (0..dimensions.len()).collect();
        order.sort_by_key(|&i| (dimensions[i].height, i));

        let final_height = dimensions[order.last().copied().unwrap()].height;
        if final_height == 0 {
            return Err(LmcsError::ZeroHeightMatrix {
                matrix: order.last().copied().unwrap(),
            });
        }
        if !final_height.is_power_of_two() {
            return Err(LmcsError::FinalHeightNotPowerOfTwo {
                height: final_height,
            });
        }
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

        let mut state = [P::Value::default(); WIDTH];

        for &idx in &order {
            let dims = dimensions[idx];
            if dims.height == 0 {
                return Err(LmcsError::ZeroHeightMatrix { matrix: idx });
            }
            if !dims.height.is_power_of_two() {
                return Err(LmcsError::NonPowerOfTwoHeight {
                    matrix: idx,
                    height: dims.height,
                });
            }
            if final_height % dims.height != 0 {
                return Err(LmcsError::HeightNotDivisor {
                    matrix: idx,
                    height: dims.height,
                    final_height,
                });
            }

            let values = &opened_values[idx];
            if dims.width != values.len() {
                return Err(LmcsError::WrongWidth {
                    matrix: idx,
                    expected: dims.width,
                    actual: values.len(),
                });
            }

            // Consume the provided row in the same order used during commitment.
            self.sponge.absorb(&mut state, values.iter().copied());
        }

        let mut digest = self.sponge.squeeze::<DIGEST_ELEMS>(&state);
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
