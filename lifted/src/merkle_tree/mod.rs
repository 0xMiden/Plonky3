//! Lifted Merkle tree commitment scheme for matrices with power-of-two heights.
//!
//! This module provides a Merkle tree commitment scheme optimized for matrices that store
//! polynomial evaluations in **bit-reversed order** over multiplicative cosets.
//!
//! # Mathematical Foundation
//!
//! Consider a polynomial `f(X)` of degree less than `d`, and let `g` be the coset generator and
//! `K` a subgroup of order `n ≥ d` with primitive root `ω`. The coset evaluations
//! `{f(g·ω^j) : j ∈ [0, n)}` can be stored in two orderings:
//!
//! - **Canonical order**: `[f(g·ω^0), f(g·ω^1), ..., f(g·ω^{n-1})]`
//! - **Bit-reversed order**: `[f(g·ω^{bitrev(0)}), f(g·ω^{bitrev(1)}), ..., f(g·ω^{bitrev(n-1)})]`
//!
//! where `bitrev(i)` is the bit-reversal of index `i` within `log2(n)` bits.
//!
//! # Lifting by Upsampling
//!
//! When we have matrices with different heights `n_0 ≤ n_1 ≤ ... ≤ n_{t-1}` (each a power of two),
//! we "lift" smaller matrices to the maximum height `N = n_{t-1}` using **nearest-neighbor
//! upsampling**: each row is repeated contiguously `r = N/n` times.
//!
//! For a matrix of height `n` lifted to `N`, the index map is: `i ↦ floor(i / r) = i >> log2(r)`
//!
//! **Example** (`n=4`, `N=8`):
//! - Original rows: `[row0, row1, row2, row3]`
//! - Upsampled: `[row0, row0, row1, row1, row2, row2, row3, row3]` (blocks of 2)
//!
//! # Why Upsampling for Bit-Reversed Data
//!
//! Given bit-reversed evaluations of `f(X)` over a coset `gK` where `|K| = n`, upsampling to
//! height `N = n · r` (where `r = 2^k`) produces the bit-reversed evaluations of `f'(X) = f(X^r)`
//! over the coset `gK'` where `|K'| = N`.
//!
//! Mathematically, if the input contains `f(g·(ω_n)^{bitrev_n(j)})` at index `j`, then after
//! upsampling, each index `i` in `[0, N)` maps to the original index `j = i >> k`, giving:
//!
//! ```text
//! upsampled[i] = f(g·(ω_n)^{bitrev_n(i >> k)}) = f'(g·(ω_N)^{bitrev_N(i)})
//! ```
//!
//! where `f'(X) = f(X^r)`. This is exactly the bit-reversed evaluation of `f'` over `gK'`.
//!
//! # Opening Semantics
//!
//! When opening at index `i`, we retrieve the value at position `i` in the bit-reversed list.
//! For the lifted polynomial `f'(X) = f(X^r)`, this gives `f'(g·(ω_N)^{bitrev_N(i)})`.
//!
//! Equivalently, this is `f'(g·ξ^i)` where `ξ = (ω_N)^{bitrev_N(i)}` is the `i`-th element
//! when iterating over `K'` in the order induced by bit-reversed indices.
//!
//! # Equivalence to Cyclic Lifting
//!
//! Upsampling bit-reversed data is equivalent to cyclically repeating canonically-ordered data:
//!
//! ```text
//! Upsample(BitReverse(data)) = BitReverse(Cyclic(data))
//! ```
//!
//! where cyclic repetition tiles the original `n` rows periodically: `[row0, row1, ..., row_{n-1}, row0, ...]`.
//!
//! This equivalence follows from the bit-reversal identity: for `r = N/n = 2^k`,
//! `bitrev_N(i) mod n = bitrev_n(i >> k)`.

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

pub use hiding_lmcs::MerkleTreeHidingLmcs;
pub use lifted_tree::{LiftedMerkleTree, build_leaf_states_upsampled};

/// Lifted MMCS built on top of [`LiftedMerkleTree`].
///
/// Matrices of different heights are aligned to the tallest height via nearest-neighbor
/// upsampling. For a matrix of height `n` lifted to `N`, each original row is duplicated
/// contiguously `r = N / n` times, and the lifted index map is `i ↦ floor(i / r)` (with `r` a
/// power of two). This produces blocks of identical rows.
///
/// Conceptually, each matrix is virtually extended vertically to height `N` (width unchanged)
/// and the leaf at index `i` absorbs the `i`-th row from each extended matrix.
///
/// Equivalent single-matrix view: the scheme is equivalent to lifting every matrix to height `N`,
/// padding each horizontally with zeros to a multiple of the hasher's padding width, and
/// concatenating them side-by-side into one matrix. The Merkle tree and verification behavior are
/// identical to committing to and opening that single concatenated matrix.
#[derive(Copy, Clone, Debug)]
pub struct MerkleTreeLmcs<PF, PD, H, C, const WIDTH: usize, const DIGEST_ELEMS: usize> {
    pub(crate) sponge: H,
    pub(crate) compress: C,
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
    pub const fn new(sponge: H, compress: C) -> Self {
        Self {
            sponge,
            compress,
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
            return Err(LmcsError::IndexOutOfBounds);
        }

        let expected_proof_len = log2_strict_usize(final_height);
        // Verify that the authentication path has the correct length for the tree height
        if proof.len() != expected_proof_len {
            return Err(LmcsError::WrongProofLen);
        }

        let mut state = [PD::Value::default(); WIDTH];
        for (idx, (row, dimension)) in zip(rows, dimensions).enumerate() {
            if row.len() != dimension.width {
                return Err(LmcsError::WrongWidth { matrix: idx });
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
            inputs,
            None,
        );
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

/// Errors that can occur during Merkle verification.
#[derive(Debug, Error)]
pub enum LmcsError {
    /// Number of opened rows doesn't match number of committed matrices.
    #[error("wrong batch size")]
    WrongBatchSize,
    /// Opened row width doesn't match the committed matrix width.
    #[error("wrong width at matrix {matrix}")]
    WrongWidth { matrix: usize },
    /// Salt row length doesn't match expected width.
    #[error("wrong salt width")]
    WrongSalt,
    /// Authentication path length doesn't match tree height.
    #[error("wrong proof length")]
    WrongProofLen,
    /// Query index exceeds tree height.
    #[error("index out of bounds")]
    IndexOutOfBounds,
    /// Recomputed root doesn't match the commitment.
    #[error("root mismatch")]
    RootMismatch,
}

#[cfg(test)]
mod tests;
