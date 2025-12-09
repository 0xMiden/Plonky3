use alloc::vec;
use alloc::vec::Vec;
use core::array;

use p3_field::PackedValue;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_symmetric::{Hash, PseudoCompressionFunction, StatefulHasher};
use serde::{Deserialize, Serialize};

use super::utils::{pack_arrays, unpack_array_into, validate_heights};
use crate::{Lifting, LmcsError};

/// A uniform binary Merkle tree whose leaves are constructed from matrices with power-of-two heights.
///
/// * `F` – scalar field element type used in both matrices and digests.
/// * `M` – matrix type. Must implement [`Matrix<F>`].
/// * `DIGEST_ELEMS` – number of `F` elements in one digest.
///
/// Unlike the standard `MerkleTree`, this uniform variant requires:
/// - **All matrix heights must be powers of two**
/// - **Matrices must be sorted by height** (shortest to tallest)
/// - Uses incremental hashing via [`StatefulSponge`] instead of one-shot hashing
///
/// The per-leaf row composition follows the chosen [`Lifting`]: each matrix `M_i` is virtually
/// extended to height `H` (width unchanged) via the index map described in [`Lifting`]. For leaf
/// row `r`, the sponge absorbs the `r`-th row from each lifted matrix in sequence (with
/// per-matrix zero padding to a multiple of the hasher's padding width for
/// absorption).
///
/// Equivalent single-matrix view: this commitment is equivalent to first forming a single
/// height-`H` matrix by (a) lifting every input matrix to height `H` (per [`Lifting`]), (b)
/// padding each lifted matrix horizontally with zero columns so each width is a multiple of the
/// hasher's padding width, and (c) concatenating the results side-by-side. The leaf digest at
/// row `r` is then the
/// sponge of that single concatenated matrix’s row `r`. From the verifier’s perspective, the two
/// constructions are indistinguishable: verification absorbs the same padded row segments in the
/// same order and checks the same Merkle path.
///
/// Since [`StatefulSponge`] operates on a single field type, this tree uses the same type `F`
/// for both matrix elements and digest words, unlike `MerkleTree` which can hash `F → W`.
///
/// Use [`root`](Self::root) to fetch the final digest once the tree is built.
///
/// This generally shouldn't be used directly. If you're using a Merkle tree as an MMCS,
/// see the MMCS wrapper types.
#[derive(Debug, Serialize, Deserialize)]
pub struct LiftedMerkleTree<F, D, M, const DIGEST_ELEMS: usize> {
    /// All leaf matrices in insertion order.
    ///
    /// Matrices must be sorted by height (shortest to tallest) and all heights must be
    /// powers of two. Each matrix's rows are absorbed into sponge states that are
    /// maintained and upsampled across matrices of increasing height.
    ///
    /// This vector is retained for inspection or re-opening of the tree; it is not used
    /// after construction time.
    pub(crate) leaves: Vec<M>,

    /// All intermediate digest layers, index 0 being the leaf digest layer
    /// and the last layer containing exactly one root digest.
    ///
    /// Every inner vector holds contiguous digests. Higher layers are built by
    /// compressing pairs from the previous layer.
    ///
    /// Serialization requires that `[F; DIGEST_ELEMS]` implements `Serialize` and
    /// `Deserialize`. This is automatically satisfied when `F` is a fixed-size type.
    #[serde(
        bound(serialize = "[D; DIGEST_ELEMS]: Serialize"),
        bound(deserialize = "[D; DIGEST_ELEMS]: Deserialize<'de>")
    )]
    pub(crate) digest_layers: Vec<Vec<[D; DIGEST_ELEMS]>>,

    pub(crate) lifting: Lifting,

    pub(crate) salt: Option<RowMajorMatrix<F>>,
}

impl<F, D, M, const DIGEST_ELEMS: usize> LiftedMerkleTree<F, D, M, DIGEST_ELEMS>
where
    F: Clone + Send + Sync,
    M: Matrix<F>,
{
    /// Build a uniform tree from matrices with power-of-two heights, with optional salt for hiding.
    ///
    /// - `h`: stateful sponge used for incremental hashing of matrix rows.
    /// - `c`: 2-to-1 compression function used on digests.
    /// - `lifting`: method used to align matrices of different heights to a common height.
    /// - `leaves`: matrices to commit. Must be non-empty, sorted by height (shortest to tallest),
    ///   and all heights must be powers of two.
    /// - `salt`: optional salt matrix absorbed into each leaf state prior to squeezing. When provided,
    ///   must have height equal to the final number of leaves. The width determines the number of
    ///   salt elements per leaf row.
    ///
    /// Matrices are processed from shortest to tallest. For each matrix, per-leaf sponge states are
    /// maintained and lifted to the final height across matrices. Once all matrices have been
    /// absorbed (including optional salt), this constructor squeezes the final leaf digests and
    /// builds the upper Merkle layers.
    ///
    /// For a public hiding variant that automatically generates random salt, see
    /// [`MerkleTreeHidingLmcs`](super::MerkleTreeHidingLmcs).
    ///
    /// # Panics
    /// - If `leaves` is empty.
    /// - If matrices are not sorted by non-decreasing height.
    /// - If any matrix height is not a power of two.
    /// - If `salt` is provided but its height doesn't equal the final leaf count.
    pub(crate) fn new_with_optional_salt<PF, PD, H, C, const WIDTH: usize>(
        h: &H,
        c: &C,
        lifting: Lifting,
        leaves: Vec<M>,
        salt: Option<RowMajorMatrix<PF::Value>>,
    ) -> Result<Self, LmcsError>
    where
        PF: PackedValue<Value = F>,
        PD: PackedValue<Value = D>,
        H: StatefulHasher<F, [D; WIDTH], [D; DIGEST_ELEMS]>
            + StatefulHasher<PF, [PD; WIDTH], [PD; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[D; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[PD; DIGEST_ELEMS], 2>
            + Sync,
    {
        if leaves.is_empty() {
            return Err(LmcsError::EmptyBatch);
        }

        // Build leaf states from matrices using the sponge
        let mut leaf_states: Vec<[PD::Value; WIDTH]> = match lifting {
            Lifting::Upsample => {
                build_leaf_states_upsampled::<PF, PD, M, H, WIDTH, DIGEST_ELEMS>(&leaves, h)?
            }
            Lifting::Cyclic => {
                build_leaf_states_cyclic::<PF, PD, M, H, WIDTH, DIGEST_ELEMS>(&leaves, h)?
            }
        };

        // Optionally absorb salt rows into the states prior to squeezing.
        if let Some(salt_matrix) = salt.as_ref() {
            let tree_height = leaf_states.len();
            if salt_matrix.height() != tree_height {
                return Err(LmcsError::SaltHeightMismatch {
                    expected: tree_height,
                    actual: salt_matrix.height(),
                });
            }
            // Fold the salt matrix rows into the states.
            absorb_matrix::<PF, PD, _, H, WIDTH, DIGEST_ELEMS>(&mut leaf_states, salt_matrix, h);
        }

        // Squeeze the final digests from the states
        let leaf_digests: Vec<[PD::Value; DIGEST_ELEMS]> =
            leaf_states.iter().map(|state| h.squeeze(state)).collect();

        // Build digest layers by repeatedly compressing until we reach the root
        let mut digest_layers = vec![leaf_digests];

        loop {
            let prev_layer = digest_layers.last().unwrap();
            if prev_layer.len() == 1 {
                break;
            }

            let next_layer = compress_uniform::<PD, C, DIGEST_ELEMS>(prev_layer, c);
            digest_layers.push(next_layer);
        }

        Ok(Self {
            leaves,
            digest_layers,
            lifting,
            salt,
        })
    }

    /// Return the root digest of the tree.
    #[must_use]
    pub fn root(&self) -> Hash<F, D, DIGEST_ELEMS>
    where
        D: Copy,
    {
        self.digest_layers.last().unwrap()[0].into()
    }

    /// Return the height of the tree (number of leaves).
    #[must_use]
    pub fn height(&self) -> usize {
        self.leaves.last().unwrap().height()
    }

    pub fn rows(&self, index: usize) -> Vec<Vec<F>> {
        let max_height = self.height();

        self.leaves
            .iter()
            .map(|m| {
                let row_index = self.lifting.map_index(index, m.height(), max_height);
                m.row_slice(row_index)
                    .expect("row_index must be valid after lifting.map_index")
                    .to_vec()
            })
            .collect()
    }

    /// Extract the rows for the given leaf index with padding applied.
    ///
    /// Like [`Self::rows`], but pads each row to a multiple of `padding_multiple` with
    /// default field elements (zeros). This is useful for preparing rows for absorption
    /// into a sponge that requires a specific padding width.
    ///
    /// - `index`: the leaf row index to extract across all committed matrices.
    /// - `padding_multiple`: each row will be padded to the next multiple of this value.
    pub fn rows_padded(&self, index: usize, padding_multiple: usize) -> Vec<Vec<F>>
    where
        F: Default,
    {
        let mut rows = self.rows(index);
        for row in rows.iter_mut() {
            let padded_width = row.len().next_multiple_of(padding_multiple);
            row.resize(padded_width, F::default());
        }
        rows
    }

    /// Extract the Merkle authentication path (sibling digests) for the given leaf index.
    ///
    /// Returns a vector of sibling digests, one per tree layer, ordered from leaf layer upward.
    /// Does not include the root digest (since the path terminates there).
    ///
    /// - `index`: the leaf index for which to extract the authentication path.
    pub fn authentication_path(&self, index: usize) -> Vec<[D; DIGEST_ELEMS]>
    where
        D: Copy,
    {
        let mut layers = Vec::with_capacity(self.digest_layers.len().saturating_sub(1));
        let mut layer_index = index;
        for layer in &self.digest_layers {
            if layer.len() == 1 {
                break;
            }
            let sibling = layer[layer_index ^ 1];
            layers.push(sibling);
            layer_index >>= 1;
        }
        layers
    }

    /// Extract the salt row for the given leaf index, if salt was used during commitment.
    ///
    /// Returns `None` if this tree was constructed without salt (non-hiding variant).
    /// Returns `Some(salt_row)` containing the random field elements absorbed at the specified leaf.
    ///
    /// - `index`: the leaf index for which to extract the salt row.
    pub fn salt(&self, index: usize) -> Option<Vec<F>> {
        self.salt.as_ref().map(|salt| {
            salt.row_slice(index)
                .expect("index must be valid for salt matrix")
                .to_vec()
        })
    }
}

/// Build leaf states using the upsampled view; see [`Lifting::Upsample`] for semantics.
///
/// Returns the sponge states after absorbing all matrix rows but **before squeezing**.
/// Callers must squeeze the states to obtain final leaf digests.
///
/// Conceptually, each matrix is virtually extended to height `H` by repeating each row
/// `L = H / h` times (width unchanged), and the leaf `r` absorbs the `r`-th row from each
/// extended matrix in order. Each absorbed row is virtually padded with zeros to a multiple of the
/// hasher's padding width for absorption; see [`LiftedMerkleTree`] docs for the equivalent
/// single-matrix view.
///
/// # Preconditions
/// - `matrices` is non-empty and sorted by non-decreasing power-of-two heights.
/// - `P::WIDTH` is a power of two.
///
/// Panics in debug builds if preconditions are violated.
pub fn build_leaf_states_upsampled<PF, PD, M, H, const WIDTH: usize, const DIGEST_ELEMS: usize>(
    matrices: &[M],
    sponge: &H,
) -> Result<Vec<[PD::Value; WIDTH]>, LmcsError>
where
    PF: PackedValue,
    PD: PackedValue,
    M: Matrix<PF::Value>,
    H: StatefulHasher<PF::Value, [PD::Value; WIDTH], [PD::Value; DIGEST_ELEMS]>
        + StatefulHasher<PF, [PD; WIDTH], [PD; DIGEST_ELEMS]>
        + Sync,
{
    const { assert!(PF::WIDTH.is_power_of_two()) };
    const { assert!(PD::WIDTH.is_power_of_two()) };
    let final_height = validate_heights(matrices.iter().map(|d| d.dimensions().height))?;

    // Memory buffers:
    // - states: Per-leaf scalar states (one per final row), maintained across matrices.
    // - scratch_states: Temporary buffer used when duplicating states during upsampling.
    let default_state = [PD::Value::default(); WIDTH];
    let mut states = vec![default_state; final_height];
    let mut scratch_states = vec![default_state; final_height];

    let mut active_height = matrices.first().unwrap().height();

    for matrix in matrices {
        let height = matrix.height();

        // Upsample states when height increases (applies to both scalar and packed paths).
        // Duplicate each existing state to fill the expanded height.
        // E.g., [s0, s1] with scaling_factor=2 → [s0, s0, s1, s1]
        if height > active_height {
            let scaling_factor = height / active_height;

            // Copy `states` into `scratch_states`, repeating each entry `scaling_factor` times
            // so we keep the accumulated sponge states aligned with the taller matrix.
            scratch_states[..height]
                .par_chunks_mut(scaling_factor)
                .zip(states[..active_height].par_iter())
                .for_each(|(chunk, state)| chunk.fill(*state));

            // Copy upsampled states back to canonical buffer
            states[..height].copy_from_slice(&scratch_states[..height]);
        }

        // Absorb the rows of the matrix into the extended state vector
        absorb_matrix::<PF, PD, _, _, _, _>(&mut states[..height], matrix, sponge);

        active_height = height;
    }

    Ok(states)
}

/// Build leaf states using the cyclic view; see [`Lifting::Cyclic`] for semantics.
///
/// Returns the sponge states after absorbing all matrix rows but **before squeezing**.
/// Callers must squeeze the states to obtain final leaf digests.
///
/// Conceptually, each matrix is virtually extended to height `H` by repeating its `h` rows
/// `L = H / h` times in a cycle (width unchanged), and the leaf `r` absorbs the `r`-th row from
/// each extended matrix in order. Each absorbed row is virtually padded with zeros to a multiple
/// of the hasher's padding width for absorption; see [`LiftedMerkleTree`] docs for the
/// equivalent single-matrix view.
///
/// # Preconditions
/// - `matrices` is non-empty and sorted by non-decreasing power-of-two heights.
/// - `P::WIDTH` is a power of two.
///
/// Panics in debug builds if preconditions are violated.
#[allow(dead_code)]
pub fn build_leaf_states_cyclic<PF, PD, M, H, const WIDTH: usize, const DIGEST_ELEMS: usize>(
    matrices: &[M],
    sponge: &H,
) -> Result<Vec<[PD::Value; WIDTH]>, LmcsError>
where
    PF: PackedValue,
    PD: PackedValue,
    M: Matrix<PF::Value>,
    H: StatefulHasher<PF::Value, [PD::Value; WIDTH], [PD::Value; DIGEST_ELEMS]>
        + StatefulHasher<PF, [PD; WIDTH], [PD; DIGEST_ELEMS]>
        + Sync,
{
    const { assert!(PF::WIDTH.is_power_of_two()) };
    const { assert!(PD::WIDTH.is_power_of_two()) };
    let final_height = validate_heights(matrices.iter().map(|d| d.dimensions().height))?;

    let default_state = [PD::Value::default(); WIDTH];
    let mut states = vec![default_state; final_height];
    let mut active_height = matrices.first().unwrap().height();

    // Process matrices in ascending height, cycling each shorter matrix over the final leaf range.
    for matrix in matrices {
        let height = matrix.height();

        // Extend the state vector cyclically to reach the height of the next matrix.
        if height > active_height {
            let (first_chunk, remaining) = states.split_at_mut(active_height);
            remaining
                .par_chunks_exact_mut(active_height)
                .for_each(|states_chunk| states_chunk.copy_from_slice(first_chunk));
        }

        // Absorb the rows of the matrix into the extended state vector
        absorb_matrix::<PF, PD, _, _, WIDTH, DIGEST_ELEMS>(&mut states[..height], matrix, sponge);

        active_height = height;
    }

    Ok(states)
}

/// Incorporate one matrix’s row-wise contribution into the running per-leaf states.
///
/// Semantics: given `states` of length `h = matrix.height()`, for each row index `r ∈ [0, h)`
/// update `states[r]` by absorbing the matrix row `r` into that state. In the overall tree
/// construction, callers ensure that `states` is the correct lifted view for the current matrix
/// (either the “nearest-neighbor” duplication or the “modulo” duplication across the final
/// height). This helper performs exactly one absorption round for that matrix and returns with the
/// states mutated; it does not change the lifting shape or squeeze digests.
///
/// The implementation may use scalar or SIMD packing internally depending on the matrix height.
fn absorb_matrix<PF, PD, M, H, const WIDTH: usize, const DIGEST_ELEMS: usize>(
    states: &mut [[PD::Value; WIDTH]],
    matrix: &M,
    sponge: &H,
) where
    PF: PackedValue,
    PD: PackedValue,
    M: Matrix<PF::Value>,
    H: StatefulHasher<PF::Value, [PD::Value; WIDTH], [PD::Value; DIGEST_ELEMS]>
        + StatefulHasher<PF, [PD; WIDTH], [PD; DIGEST_ELEMS]>
        + Sync,
{
    let height = matrix.height();
    assert_eq!(height, states.len());

    if height < PF::WIDTH {
        // Scalar path: walk every final leaf state and absorb the wrapped row for this matrix.
        states
            .iter_mut()
            .zip(matrix.rows())
            .for_each(|(state, row)| {
                sponge.absorb_into(state, row);
            });
    } else {
        // SIMD path: gather → absorb wrapped packed row → scatter per chunk.
        states
            .par_chunks_mut(PF::WIDTH)
            .enumerate()
            .for_each(|(packed_idx, states_chunk)| {
                let mut packed_state: [PD; WIDTH] = pack_arrays(states_chunk);
                let row_idx = packed_idx * PF::WIDTH;
                let row = matrix.vertically_packed_row(row_idx);
                sponge.absorb_into(&mut packed_state, row);
                unpack_array_into(&packed_state, states_chunk);
            });
    }
}

/// Compress a layer of digests in a uniform Merkle tree.
///
/// Takes a layer of digests and compresses pairs into a new layer with half as many elements.
/// The layer length must be a power of two.
///
/// When the result would be smaller than the packing width, uses a pure scalar path.
/// Otherwise uses SIMD parallelization. Since both the result length and packing width are
/// powers of two, the result is always a multiple of the packing width in the SIMD path,
/// requiring no scalar fallback for remainders.
fn compress_uniform<
    P: PackedValue,
    C: PseudoCompressionFunction<[P::Value; DIGEST_ELEMS], 2>
        + PseudoCompressionFunction<[P; DIGEST_ELEMS], 2>
        + Sync,
    const DIGEST_ELEMS: usize,
>(
    prev_layer: &[[P::Value; DIGEST_ELEMS]],
    c: &C,
) -> Vec<[P::Value; DIGEST_ELEMS]> {
    assert!(
        prev_layer.len().is_power_of_two(),
        "previous layer length must be a power of 2"
    );

    let next_len = prev_layer.len() / 2;
    let default_digest = [P::Value::default(); DIGEST_ELEMS];
    let mut next_digests = vec![default_digest; next_len];

    // Use scalar path when output is too small for packing
    if next_len < P::WIDTH {
        for (i, next_digest) in next_digests.iter_mut().enumerate() {
            *next_digest = c.compress([prev_layer[2 * i], prev_layer[2 * i + 1]]);
        }
    } else {
        // Packed path: since next_len and P::WIDTH are both powers of 2,
        // next_len is a multiple of P::WIDTH, so no remainder handling needed.
        next_digests
            .par_chunks_exact_mut(P::WIDTH)
            .enumerate()
            .for_each(|(packed_chunk_idx, digests_chunk)| {
                let chunk_idx = packed_chunk_idx * P::WIDTH;
                let left: [P; DIGEST_ELEMS] =
                    array::from_fn(|j| P::from_fn(|k| prev_layer[2 * (chunk_idx + k)][j]));
                let right: [P; DIGEST_ELEMS] =
                    array::from_fn(|j| P::from_fn(|k| prev_layer[2 * (chunk_idx + k) + 1][j]));
                let packed_digest = c.compress([left, right]);
                unpack_array_into(&packed_digest, digests_chunk);
            });
    }
    next_digests
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_matrix::Matrix;
    use p3_matrix::bitrev::BitReversibleMatrix;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_symmetric::StatefulHasher;
    use p3_util::reverse_slice_index_bits;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use crate::merkle_tree::Lifting;
    use crate::merkle_tree::test_helpers::{
        DIGEST, F, P, RATE, Sponge, build_leaves_single, components, concatenate_matrices,
        lift_matrix, matrix_scenarios, rand_matrix,
    };

    fn build_leaves_cyclic(matrices: &[RowMajorMatrix<F>], sponge: &Sponge) -> Vec<[F; DIGEST]> {
        let mut states =
            super::build_leaf_states_cyclic::<P, P, _, _, _, _>(matrices, sponge).unwrap();
        states.iter_mut().map(|s| sponge.squeeze(s)).collect()
    }

    fn build_leaves_upsampled(matrices: &[RowMajorMatrix<F>], sponge: &Sponge) -> Vec<[F; DIGEST]> {
        let mut states =
            super::build_leaf_states_upsampled::<P, P, _, _, _, _>(matrices, sponge).unwrap();
        states.iter_mut().map(|s| sponge.squeeze(s)).collect()
    }

    #[test]
    fn cyclic_upsampled_equivalence() {
        let (sponge, _compressor) = components();
        let mut rng = SmallRng::seed_from_u64(42);

        for scenario in matrix_scenarios() {
            let matrices: Vec<RowMajorMatrix<F>> = scenario
                .into_iter()
                .map(|(h, w)| rand_matrix(h, w, &mut rng))
                .collect();
            let matrices_bitreversed: Vec<_> = matrices
                .iter()
                .map(|m: &RowMajorMatrix<F>| m.as_view().bit_reverse_rows().to_row_major_matrix())
                .collect();

            let max_height = matrices.last().unwrap().height();

            // Cyclic path equivalence vs explicit cyclic lifting and single-concat baseline
            {
                let leaves = build_leaves_cyclic(&matrices, &sponge);

                let matrices_cyclic: Vec<_> = matrices
                    .iter()
                    .map(|m: &RowMajorMatrix<F>| lift_matrix(m, Lifting::Cyclic, max_height))
                    .collect();
                let leaves_lifted = build_leaves_cyclic(&matrices_cyclic, &sponge);
                assert_eq!(leaves, leaves_lifted);

                let matrix_single = concatenate_matrices::<RATE>(&matrices_cyclic);
                let leaves_single = build_leaves_single(&matrix_single, &sponge);
                assert_eq!(leaves, leaves_single);

                let mut leaves_bitreversed = build_leaves_upsampled(&matrices_bitreversed, &sponge);
                reverse_slice_index_bits(&mut leaves_bitreversed);
                assert_eq!(leaves, leaves_bitreversed);
            }

            // Upsampled path equivalence vs explicit upsampled lifting and single-concat baseline
            {
                let leaves = build_leaves_upsampled(&matrices, &sponge);

                let matrices_upsampled: Vec<_> = matrices
                    .iter()
                    .map(|m: &RowMajorMatrix<F>| lift_matrix(m, Lifting::Upsample, max_height))
                    .collect();
                let leaves_lifted = build_leaves_upsampled(&matrices_upsampled, &sponge);
                assert_eq!(leaves, leaves_lifted);

                let matrix_single = concatenate_matrices::<RATE>(&matrices_upsampled);
                let leaves_single = build_leaves_single(&matrix_single, &sponge);
                assert_eq!(leaves, leaves_single);

                let mut leaves_bitreversed = build_leaves_cyclic(&matrices_bitreversed, &sponge);
                reverse_slice_index_bits(&mut leaves_bitreversed);
                assert_eq!(leaves, leaves_bitreversed);
            }
        }
    }
}
