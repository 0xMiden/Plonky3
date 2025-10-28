//! A Single Matrix Commitment Scheme (MCS) - a simplified version of MMCS for committing to a single matrix.
//!
//! This module provides a commitment scheme specifically designed for single matrices,
//! removing the complexity of handling multiple matrices with different heights.
//! It's built on top of a binary Merkle tree structure.

use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Debug;
use core::marker::PhantomData;

use p3_field::PackedValue;
use p3_matrix::{Dimensions, Matrix};
use p3_symmetric::{CryptographicHasher, Hash, PseudoCompressionFunction};
use p3_util::log2_strict_usize;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

/// A Single Matrix Commitment Scheme trait.
///
/// This is a simplified version of MMCS that supports committing to a single matrix
/// and opening individual rows. Unlike MMCS, there's no batch semantics or height
/// adjustment - row indices directly correspond to matrix rows.
pub trait Mcs<T: Send + Sync + Clone>: Clone {
    type ProverData<M>;
    type Commitment: Clone + Serialize + DeserializeOwned;
    type Proof: Clone + Serialize + DeserializeOwned;
    type Error: Debug;

    /// Commits to a single matrix.
    ///
    /// # Parameters
    /// - `matrix`: The matrix to commit to.
    ///
    /// # Returns
    /// A tuple `(commitment, prover_data)` where:
    /// - `commitment` is a compact representation sent to the verifier (typically a Merkle root).
    /// - `prover_data` is auxiliary data used by the prover to open the commitment.
    fn commit<M: Matrix<T>>(&self, matrix: M) -> (Self::Commitment, Self::ProverData<M>);

    /// Opens a specific row from the committed matrix.
    ///
    /// # Parameters
    /// - `index`: The row index to open (0-indexed).
    /// - `prover_data`: Prover data returned from `commit`.
    ///
    /// # Returns
    /// An `Opening` containing the opened row values and the proof.
    fn open<M: Matrix<T>>(
        &self,
        index: usize,
        prover_data: &Self::ProverData<M>,
    ) -> Opening<T, Self>;

    /// Returns a reference to the committed matrix.
    ///
    /// # Parameters
    /// - `prover_data`: The prover data returned by `commit`.
    ///
    /// # Returns
    /// A reference to the committed matrix.
    fn get_matrix<'a, M: Matrix<T>>(&self, prover_data: &'a Self::ProverData<M>) -> &'a M;

    /// Returns the height (number of rows) of the committed matrix.
    fn get_height<M: Matrix<T>>(&self, prover_data: &Self::ProverData<M>) -> usize {
        self.get_matrix(prover_data).height()
    }

    /// Verifies an opening at a specific row index against the original commitment.
    ///
    /// # Parameters
    /// - `commit`: The original commitment.
    /// - `dimensions`: Dimensions of the committed matrix.
    /// - `index`: The row index that was opened.
    /// - `opening`: A reference to the values and proof to verify.
    ///
    /// # Returns
    /// `Ok(())` if the opening is valid; otherwise returns a verification error.
    fn verify(
        &self,
        commit: &Self::Commitment,
        dimensions: Dimensions,
        index: usize,
        opening: OpeningRef<T, Self>,
    ) -> Result<(), Self::Error>;
}

/// An opening proof for a single row.
///
/// Contains the opened values and a Merkle proof.
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(serialize = "T: Serialize"))]
#[serde(bound(deserialize = "T: DeserializeOwned"))]
pub struct Opening<T: Send + Sync + Clone, InputMcs: Mcs<T>> {
    /// The opened row values.
    pub opened_values: Vec<T>,
    /// The proof showing the values are valid.
    pub opening_proof: InputMcs::Proof,
}

impl<T: Send + Sync + Clone, InputMcs: Mcs<T>> Opening<T, InputMcs> {
    /// Creates a new opening proof.
    #[inline]
    pub fn new(opened_values: Vec<T>, opening_proof: InputMcs::Proof) -> Self {
        Self {
            opened_values,
            opening_proof,
        }
    }

    /// Unpacks the opening proof into its components.
    #[inline]
    pub fn unpack(self) -> (Vec<T>, InputMcs::Proof) {
        (self.opened_values, self.opening_proof)
    }
}

/// A reference to an opening proof.
///
/// Used by the verifier.
#[derive(Copy, Clone)]
pub struct OpeningRef<'a, T: Send + Sync + Clone, InputMcs: Mcs<T>> {
    /// Reference to the opened row values.
    pub opened_values: &'a [T],
    /// Reference to the proof.
    pub opening_proof: &'a InputMcs::Proof,
}

impl<'a, T: Send + Sync + Clone, InputMcs: Mcs<T>> OpeningRef<'a, T, InputMcs> {
    /// Creates a new opening reference.
    #[inline]
    pub fn new(opened_values: &'a [T], opening_proof: &'a InputMcs::Proof) -> Self {
        Self {
            opened_values,
            opening_proof,
        }
    }

    /// Unpacks the opening reference into its components.
    #[inline]
    pub fn unpack(&self) -> (&'a [T], &'a InputMcs::Proof) {
        (self.opened_values, self.opening_proof)
    }
}

impl<'a, T: Send + Sync + Clone, InputMcs: Mcs<T>> From<&'a Opening<T, InputMcs>>
    for OpeningRef<'a, T, InputMcs>
{
    #[inline]
    fn from(opening: &'a Opening<T, InputMcs>) -> Self {
        Self::new(&opening.opened_values, &opening.opening_proof)
    }
}

/// A binary Merkle tree for a single matrix.
///
/// This is a simplified version of the multi-matrix MerkleTree that only handles one matrix.
#[derive(Debug, Serialize, Deserialize)]
pub struct SingleMerkleTree<F, W, M, const DIGEST_ELEMS: usize> {
    /// The committed matrix.
    pub(crate) matrix: M,

    /// All intermediate digest layers, index 0 being the first layer above
    /// the leaves and the last layer containing exactly one root digest.
    #[serde(
        bound(serialize = "[W; DIGEST_ELEMS]: Serialize"),
        bound(deserialize = "[W; DIGEST_ELEMS]: Deserialize<'de>")
    )]
    pub(crate) digest_layers: Vec<Vec<[W; DIGEST_ELEMS]>>,

    _phantom: PhantomData<F>,
}

impl<F: Clone + Send + Sync, W: Clone + Copy + Default, M: Matrix<F>, const DIGEST_ELEMS: usize>
    SingleMerkleTree<F, W, M, DIGEST_ELEMS>
{
    /// Build a Merkle tree from a single matrix.
    ///
    /// # Parameters
    /// - `h`: Cryptographic hash function for hashing matrix rows.
    /// - `c`: Pseudo-compression function for internal nodes.
    /// - `matrix`: The matrix to commit to.
    ///
    /// # Panics
    /// - If the matrix has zero rows.
    /// - If packing widths of `P` and `PW` differ.
    pub fn new<P, PW, H, C>(h: &H, c: &C, matrix: M) -> Self
    where
        P: PackedValue<Value = F>,
        PW: PackedValue<Value = W>,
        H: CryptographicHasher<F, [W; DIGEST_ELEMS]>
            + CryptographicHasher<P, [PW; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[W; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[PW; DIGEST_ELEMS], 2>
            + Sync,
    {
        assert!(matrix.height() > 0, "Matrix must have at least one row");
        assert_eq!(P::WIDTH, PW::WIDTH, "Packing widths must match");

        // Hash all rows to create the first digest layer
        let first_layer = Self::hash_rows::<P, PW, H>(h, &matrix);

        let mut digest_layers = vec![first_layer];

        // Build the tree by repeatedly compressing pairs of digests
        loop {
            let prev_layer = digest_layers.last().unwrap();
            if prev_layer.len() == 1 {
                break;
            }

            let next_layer = Self::compress_layer::<PW, C>(c, prev_layer);
            digest_layers.push(next_layer);
        }

        Self {
            matrix,
            digest_layers,
            _phantom: PhantomData,
        }
    }

    /// Hash all rows of the matrix to create the leaf layer.
    fn hash_rows<P, PW, H>(h: &H, matrix: &M) -> Vec<[PW::Value; DIGEST_ELEMS]>
    where
        P: PackedValue<Value = F>,
        PW: PackedValue<Value = W>,
        H: CryptographicHasher<F, [W; DIGEST_ELEMS]>
            + CryptographicHasher<P, [PW; DIGEST_ELEMS]>
            + Sync,
    {
        let height = matrix.height();
        let height_padded = height.next_power_of_two();
        let default_digest = [PW::Value::default(); DIGEST_ELEMS];

        let mut digests = vec![default_digest; height_padded];

        // Hash each row
        for i in 0..height {
            digests[i] = h.hash_iter(matrix.row(i).unwrap());
        }

        digests
    }

    /// Compress one layer of digests into the next layer.
    fn compress_layer<PW, C>(
        c: &C,
        prev_layer: &[[PW::Value; DIGEST_ELEMS]],
    ) -> Vec<[PW::Value; DIGEST_ELEMS]>
    where
        PW: PackedValue<Value = W>,
        C: PseudoCompressionFunction<[W; DIGEST_ELEMS], 2>,
    {
        let next_len = (prev_layer.len() / 2).max(1);
        let mut next_layer = Vec::with_capacity(next_len);

        for i in 0..next_len {
            let left = prev_layer[2 * i];
            let right = if 2 * i + 1 < prev_layer.len() {
                prev_layer[2 * i + 1]
            } else {
                [PW::Value::default(); DIGEST_ELEMS]
            };
            next_layer.push(c.compress([left, right]));
        }

        next_layer
    }

    /// Return the root digest of the tree.
    #[must_use]
    pub fn root(&self) -> Hash<F, W, DIGEST_ELEMS>
    where
        W: Copy,
    {
        self.digest_layers.last().unwrap()[0].into()
    }
}

/// A Merkle tree-based commitment scheme for a single matrix.
#[derive(Copy, Clone, Debug)]
pub struct MerkleTreeMcs<P, PW, H, C, const DIGEST_ELEMS: usize> {
    /// The hash function used to hash individual matrix rows.
    hash: H,

    /// The compression function used to hash internal tree nodes.
    compress: C,

    _phantom: PhantomData<(P, PW)>,
}

/// Errors that may arise during commitment, opening, or verification.
#[derive(Debug)]
pub enum MerkleTreeError {
    /// The opened row has the wrong width.
    WrongWidth,

    /// The proof has the wrong length.
    WrongHeight {
        expected_height: usize,
        num_siblings: usize,
    },

    /// The computed root doesn't match the commitment.
    RootMismatch,

    /// Index out of bounds.
    IndexOutOfBounds,
}

impl<P, PW, H, C, const DIGEST_ELEMS: usize> MerkleTreeMcs<P, PW, H, C, DIGEST_ELEMS> {
    /// Create a new `MerkleTreeMcs` with the given hash and compression functions.
    pub const fn new(hash: H, compress: C) -> Self {
        Self {
            hash,
            compress,
            _phantom: PhantomData,
        }
    }
}

impl<P, PW, H, C, const DIGEST_ELEMS: usize> Mcs<P::Value>
    for MerkleTreeMcs<P, PW, H, C, DIGEST_ELEMS>
where
    P: PackedValue,
    PW: PackedValue,
    H: CryptographicHasher<P::Value, [PW::Value; DIGEST_ELEMS]>
        + CryptographicHasher<P, [PW; DIGEST_ELEMS]>
        + Sync,
    C: PseudoCompressionFunction<[PW::Value; DIGEST_ELEMS], 2>
        + PseudoCompressionFunction<[PW; DIGEST_ELEMS], 2>
        + Sync,
    PW::Value: Eq,
    [PW::Value; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
{
    type ProverData<M> = SingleMerkleTree<P::Value, PW::Value, M, DIGEST_ELEMS>;
    type Commitment = Hash<P::Value, PW::Value, DIGEST_ELEMS>;
    type Proof = Vec<[PW::Value; DIGEST_ELEMS]>;
    type Error = MerkleTreeError;

    fn commit<M: Matrix<P::Value>>(
        &self,
        matrix: M,
    ) -> (Self::Commitment, Self::ProverData<M>) {
        let tree = SingleMerkleTree::new::<P, PW, H, C>(&self.hash, &self.compress, matrix);
        let root = tree.root();
        (root, tree)
    }

    fn open<M: Matrix<P::Value>>(
        &self,
        index: usize,
        prover_data: &SingleMerkleTree<P::Value, PW::Value, M, DIGEST_ELEMS>,
    ) -> Opening<P::Value, Self> {
        let height = prover_data.matrix.height();
        let tree_height = log2_strict_usize(height.next_power_of_two());

        // Get the row values
        let opened_values = prover_data
            .matrix
            .row(index)
            .unwrap()
            .into_iter()
            .collect();

        // Collect sibling digests along the path from leaf to root
        let proof = (0..tree_height)
            .map(|i| prover_data.digest_layers[i][(index >> i) ^ 1])
            .collect();

        Opening::new(opened_values, proof)
    }

    fn get_matrix<'a, M: Matrix<P::Value>>(
        &self,
        prover_data: &'a Self::ProverData<M>,
    ) -> &'a M {
        &prover_data.matrix
    }

    fn verify(
        &self,
        commit: &Self::Commitment,
        dimensions: Dimensions,
        mut index: usize,
        opening: OpeningRef<P::Value, Self>,
    ) -> Result<(), Self::Error> {
        let (opened_values, opening_proof) = opening.unpack();

        let height = dimensions.height;
        let tree_height = log2_strict_usize(height.next_power_of_two());

        // Check proof length
        if opening_proof.len() != tree_height {
            return Err(MerkleTreeError::WrongHeight {
                expected_height: tree_height,
                num_siblings: opening_proof.len(),
            });
        }

        // Check index bounds
        if index >= height {
            return Err(MerkleTreeError::IndexOutOfBounds);
        }

        // Hash the opened row to get the leaf digest
        let mut root = self.hash.hash_slice(opened_values);

        // Traverse up the tree, combining with siblings
        for &sibling in opening_proof {
            let (left, right) = if index & 1 == 0 {
                (root, sibling)
            } else {
                (sibling, root)
            };

            root = self.compress.compress([left, right]);
            index >>= 1;
        }

        // Check if computed root matches commitment
        if commit == &root {
            Ok(())
        } else {
            Err(MerkleTreeError::RootMismatch)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    type F = BabyBear;
    type Perm = Poseidon2BabyBear<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type MyMcs =
        MerkleTreeMcs<<F as Field>::Packing, <F as Field>::Packing, MyHash, MyCompress, 8>;

    #[test]
    fn test_commit_and_verify_single_row() {
        let mut rng = SmallRng::seed_from_u64(42);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mcs = MyMcs::new(hash, compress);

        // Create a 1x4 matrix
        let mat = RowMajorMatrix::new(vec![F::from_u8(1), F::from_u8(2), F::from_u8(3), F::from_u8(4)], 4);
        let dims = mat.dimensions();

        let (commit, prover_data) = mcs.commit(mat);
        let opening = mcs.open(0, &prover_data);

        assert!(mcs.verify(&commit, dims, 0, (&opening).into()).is_ok());
    }

    #[test]
    fn test_commit_and_verify_multiple_rows() {
        let mut rng = SmallRng::seed_from_u64(42);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mcs = MyMcs::new(hash, compress);

        // Create a 4x2 matrix
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 4, 2);
        let dims = mat.dimensions();

        let (commit, prover_data) = mcs.commit(mat);

        // Test opening each row
        for i in 0..4 {
            let opening = mcs.open(i, &prover_data);
            assert!(mcs.verify(&commit, dims, i, (&opening).into()).is_ok());
        }
    }

    #[test]
    fn test_verify_wrong_values_fails() {
        let mut rng = SmallRng::seed_from_u64(42);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mcs = MyMcs::new(hash, compress);

        let mat = RowMajorMatrix::<F>::rand(&mut rng, 4, 2);
        let dims = mat.dimensions();

        let (commit, prover_data) = mcs.commit(mat);
        let mut opening = mcs.open(0, &prover_data);

        // Tamper with the opened values
        opening.opened_values[0] += F::from_u8(1);

        assert!(mcs.verify(&commit, dims, 0, (&opening).into()).is_err());
    }

    #[test]
    fn test_verify_wrong_proof_fails() {
        let mut rng = SmallRng::seed_from_u64(42);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mcs = MyMcs::new(hash, compress);

        let mat = RowMajorMatrix::<F>::rand(&mut rng, 8, 2);
        let dims = mat.dimensions();

        let (commit, prover_data) = mcs.commit(mat);
        let mut opening = mcs.open(0, &prover_data);

        // Tamper with the proof
        if !opening.opening_proof.is_empty() {
            opening.opening_proof[0][0] += F::from_u8(1);
        }

        assert!(mcs.verify(&commit, dims, 0, (&opening).into()).is_err());
    }
}
