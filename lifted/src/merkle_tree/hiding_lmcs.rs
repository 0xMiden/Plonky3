use alloc::vec::Vec;
use core::cell::RefCell;

use p3_commit::{BatchOpening, BatchOpeningRef, Mmcs};
use p3_field::PackedValue;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::{Dimensions, Matrix};
use p3_symmetric::{Hash, PseudoCompressionFunction, StatefulHasher};
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use super::{LiftedMerkleTree, LmcsError};

/// Hiding LMCS wrapper around `MerkleTreeLmcs`.
///
/// Provides zero-knowledge hiding by appending a random salt row to each leaf. The salt
/// is absorbed into the per-leaf sponge state prior to squeezing during commitment, and
/// is included in the proof for verification. When `salt_elems` is 0, this degenerates
/// to non-hiding behavior (though you should use `MerkleTreeLmcs` directly in that case).
#[derive(Clone, Debug)]
pub struct MerkleTreeHidingLmcs<PF, PD, H, C, R, const WIDTH: usize, const DIGEST_ELEMS: usize> {
    /// Base (non-hiding) LMCS instance.
    pub(crate) inner: super::MerkleTreeLmcs<PF, PD, H, C, WIDTH, DIGEST_ELEMS>,
    salt_elems: usize,
    rng: RefCell<R>,
}

impl<PF, PD, H, C, R, const WIDTH: usize, const DIGEST_ELEMS: usize>
    MerkleTreeHidingLmcs<PF, PD, H, C, R, WIDTH, DIGEST_ELEMS>
{
    /// Construct a hiding LMCS wrapper.
    ///
    /// - `inner`: the base (non-hiding) LMCS instance to wrap.
    /// - `salt_elems`: number of random field elements to append to each leaf as salt.
    ///   Must be greater than 0 for hiding properties.
    /// - `rng`: random number generator used to generate salt during commitment.
    ///
    /// # Panics
    /// Panics if `salt_elems` is 0, as this would provide no hiding.
    pub fn new(
        inner: super::MerkleTreeLmcs<PF, PD, H, C, WIDTH, DIGEST_ELEMS>,
        salt_elems: usize,
        rng: R,
    ) -> Self {
        assert!(salt_elems > 0, "salt_elems must be greater than 0",);
        Self {
            inner,
            salt_elems,
            rng: RefCell::new(rng),
        }
    }
}

impl<PF, PD, H, C, R, const WIDTH: usize, const DIGEST_ELEMS: usize> Mmcs<PF::Value>
    for MerkleTreeHidingLmcs<PF, PD, H, C, R, WIDTH, DIGEST_ELEMS>
where
    PF: PackedValue + Default,
    PD: PackedValue + Default,
    PF::Value: Serialize + DeserializeOwned,
    H: StatefulHasher<PF, [PD; WIDTH], [PD; DIGEST_ELEMS]>
        + StatefulHasher<PF::Value, [PD::Value; WIDTH], [PD::Value; DIGEST_ELEMS]>
        + Sync,
    C: PseudoCompressionFunction<[PD::Value; DIGEST_ELEMS], 2>
        + PseudoCompressionFunction<[PD; DIGEST_ELEMS], 2>
        + Sync,
    R: Rng + Clone,
    PD::Value: Eq,
    [PD::Value; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    StandardUniform: Distribution<PF::Value>,
{
    const ROW_PADDING: usize = <H as StatefulHasher<
        PF::Value,
        [PD::Value; WIDTH],
        [PD::Value; DIGEST_ELEMS],
    >>::PADDING_WIDTH;
    type ProverData<M> = LiftedMerkleTree<PF::Value, PD::Value, M, DIGEST_ELEMS>;
    type Commitment = Hash<PF::Value, PD::Value, DIGEST_ELEMS>;
    /// Proof consists of the Merkle authentication path and the salt row.
    ///
    /// The tuple contains:
    /// - Merkle sibling digests (one per tree layer, leaf to root)
    /// - Salt row (the random field elements absorbed at this leaf)
    type Proof = (Vec<[PD::Value; DIGEST_ELEMS]>, Vec<PF::Value>);
    type Error = LmcsError;

    fn commit<M: Matrix<PF::Value>>(
        &self,
        inputs: Vec<M>,
    ) -> (Self::Commitment, Self::ProverData<M>) {
        let tree_height = inputs
            .last()
            .expect("inputs must be non-empty for commitment")
            .height();
        let salt = Some(RowMajorMatrix::<PF::Value>::rand(
            &mut *self.rng.borrow_mut(),
            tree_height,
            self.salt_elems,
        ));

        let tree = LiftedMerkleTree::new_with_optional_salt::<PF, PD, H, C, WIDTH>(
            &self.inner.sponge,
            &self.inner.compress,
            self.inner.lifting,
            inputs,
            salt,
        )
        .expect("tree construction failed: inputs must have valid power-of-two heights");
        let root = tree.root();
        (root, tree)
    }

    fn open_batch<M: Matrix<PF::Value>>(
        &self,
        index: usize,
        tree: &Self::ProverData<M>,
    ) -> BatchOpening<PF::Value, Self> {
        let tree_height = tree.height();
        assert!(index < tree_height, "index out of range");

        let BatchOpening {
            opened_values,
            opening_proof,
        } = self.inner.open_batch(index, tree);

        // Safe: tree was constructed with salt in commit()
        let salt = tree
            .salt(index)
            .expect("salt must exist for hiding LMCS tree");

        let proof = (opening_proof, salt);

        BatchOpening::new(opened_values, proof)
    }

    fn get_matrices<'a, M: Matrix<PF::Value>>(&self, tree: &'a Self::ProverData<M>) -> Vec<&'a M> {
        self.inner.get_matrices(tree)
    }

    fn verify_batch(
        &self,
        commit: &Self::Commitment,
        dimensions: &[Dimensions],
        index: usize,
        batch_opening: BatchOpeningRef<'_, PF::Value, Self>,
    ) -> Result<(), Self::Error> {
        let (opened_values, (opening_proof, salt)) = batch_opening.unpack();

        // Validate salt row
        if salt.len() != self.salt_elems {
            return Err(LmcsError::WrongSalt {
                expected: self.salt_elems,
                actual: salt.len(),
            });
        }

        let expected_root =
            self.inner
                .compute_root(opened_values, index, dimensions, opening_proof, Some(salt))?;

        if &expected_root == commit {
            Ok(())
        } else {
            Err(LmcsError::RootMismatch)
        }
    }
}
