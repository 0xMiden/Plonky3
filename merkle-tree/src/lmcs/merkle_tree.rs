use alloc::vec;
use alloc::vec::Vec;
use core::array;
use core::marker::PhantomData;

use p3_field::PackedValue;
use p3_matrix::Matrix;
// use p3_matrix::lifted::LiftableMatrix;
use p3_maybe_rayon::prelude::{
    IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelSliceMut,
};
use p3_symmetric::{Hash, PseudoCompressionFunction, StatefulSponge};
use p3_util::log2_strict_usize;
use serde::{Deserialize, Serialize};

use super::utils::{pack_arrays, pad_rows, unpack_array_into, validate_heights};
use crate::LmcsError;

/// Lifting method used to align matrices of different heights to a common height.
///
/// Consider matrices `M_0, …, M_{t-1}` with heights `h_0 ≤ h_1 ≤ … ≤ h_{t-1}` (each a power of
/// two), and let `H = h_{t-1}`. For each matrix `M_i`, lifting defines a row-index mapping
/// `f_i: {0,…,H-1} → {0,…,h_i-1}` and thereby a virtual height-`H` matrix `M_i^↑` of the same
/// width, whose `r`-th row is `row_{f_i(r)}(M_i)`. In other words, lifting “extends” each matrix
/// vertically to height `H` without changing its width. The LMCS leaf at position `r` then uses,
/// in order, the row `r` from each lifted matrix `M_i^↑` as input to the sponge (with per-matrix
/// zero padding to a multiple of `RATE` for absorption).
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
/// per-matrix zero padding to a multiple of `RATE` for absorption).
///
/// Equivalent single-matrix view: this commitment is equivalent to first forming a single
/// height-`H` matrix by (a) lifting every input matrix to height `H` (per [`Lifting`]), (b)
/// padding each lifted matrix horizontally with zero columns so each width is a multiple of
/// `RATE`, and (c) concatenating the results side-by-side. The leaf digest at row `r` is then the
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
pub struct LiftedMerkleTree<F, M, const DIGEST_ELEMS: usize> {
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
        bound(serialize = "[F; DIGEST_ELEMS]: Serialize"),
        bound(deserialize = "[F; DIGEST_ELEMS]: Deserialize<'de>")
    )]
    pub(crate) digest_layers: Vec<Vec<[F; DIGEST_ELEMS]>>,

    pub(crate) lifting: Lifting,

    /// Zero-sized marker for type safety.
    _phantom: PhantomData<M>,
}

impl<F: Clone + Send + Sync + Copy + Default, M: Matrix<F>, const DIGEST_ELEMS: usize>
    LiftedMerkleTree<F, M, DIGEST_ELEMS>
{
    /// Build a uniform tree from **one or more matrices** with power-of-two heights.
    ///
    /// * `h` – stateful sponge used for incremental hashing of matrix rows.
    /// * `c` – 2-to-1 compression function used on digests.
    /// * `leaves` – matrices to commit to. Must be non-empty, sorted by height (shortest
    ///   to tallest), and all heights must be powers of two.
    ///
    /// Matrices are processed from shortest to tallest. For each matrix, sponge states
    /// are maintained per final leaf row, upsampling (duplicating) as matrices grow.
    /// This ensures uniform state evolution across all leaves.
    ///
    /// After leaf digests are built, they are repeatedly compressed pair-wise with `c`
    /// until a single root remains.
    ///
    /// # Panics
    /// * If `leaves` is empty.
    /// * If matrices are not sorted by height.
    /// * If any matrix height is not a power of two.
    pub fn new<
        P: PackedValue<Value = F> + Default,
        H: StatefulSponge<P, WIDTH, RATE> + StatefulSponge<F, WIDTH, RATE> + Sync,
        C: Sync
            + PseudoCompressionFunction<[F; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[P; DIGEST_ELEMS], 2>,
        const WIDTH: usize,
        const RATE: usize,
    >(
        h: &H,
        c: &C,
        lifting: Lifting,
        leaves: Vec<M>,
    ) -> Result<Self, LmcsError> {
        assert!(!leaves.is_empty(), "No matrices given");

        // Build leaf digests from matrices using the sponge
        let leaf_digests = match lifting {
            Lifting::Upsample => {
                build_leaves_upsampled::<P, M, H, WIDTH, RATE, DIGEST_ELEMS>(&leaves, h)
            }
            Lifting::Cyclic => {
                build_leaves_cyclic::<P, M, H, WIDTH, RATE, DIGEST_ELEMS>(&leaves, h)
            }
        }?;

        // Build digest layers by repeatedly compressing until we reach the root
        let mut digest_layers = vec![leaf_digests];

        loop {
            let prev_layer = digest_layers.last().unwrap();
            if prev_layer.len() == 1 {
                break;
            }

            let next_layer = compress_uniform::<P, C, DIGEST_ELEMS>(prev_layer, c);
            digest_layers.push(next_layer);
        }

        Ok(Self {
            leaves,
            digest_layers,
            lifting,
            _phantom: PhantomData,
        })
    }

    /// Return the root digest of the tree.
    #[must_use]
    pub fn root(&self) -> Hash<F, F, DIGEST_ELEMS> {
        self.digest_layers.last().unwrap()[0].into()
    }

    pub fn rows(&self, index: usize, padding_multiple: usize) -> Vec<Vec<F>> {
        let max_height = self.leaves.last().unwrap().height();
        let mut rows = Vec::with_capacity(self.leaves.len());

        for m in &self.leaves {
            let row_index = self.lifting.map_index(index, m.height(), max_height);
            let row = m.row_slice(row_index).unwrap().to_vec();

            rows.push(row);
        }
        pad_rows(rows, padding_multiple)
    }
}

/// Build leaf digests using the upsampled view; see [`Lifting::Upsample`] for semantics.
/// Conceptually, each matrix is virtually extended to height `H` by repeating each row
/// `L = H / h` times (width unchanged), and the leaf `r` absorbs the `r`-th row from each
/// extended matrix in order. Each absorbed row is virtually padded with zeros to a multiple of
/// `RATE` for absorption; see [`LiftedMerkleTree`] docs for the equivalent single-matrix view.
///
/// # Preconditions
/// - `matrices` is non-empty and sorted by non-decreasing power-of-two heights.
/// - `P::WIDTH` is a power of two.
///
/// Panics in debug builds if preconditions are violated.
pub fn build_leaves_upsampled<
    P: PackedValue + Default,
    M: Matrix<P::Value>,
    H: StatefulSponge<P, WIDTH, RATE> + StatefulSponge<P::Value, WIDTH, RATE> + Sync,
    const WIDTH: usize,
    const RATE: usize,
    const DIGEST_ELEMS: usize,
>(
    matrices: &[M],
    sponge: &H,
) -> Result<Vec<[P::Value; DIGEST_ELEMS]>, LmcsError> {
    const {
        assert!(P::WIDTH.is_power_of_two());
    };
    validate_heights(matrices.iter().map(|d| d.dimensions().height))?;

    let final_height = matrices.last().unwrap().height();

    let scalar_default = [P::Value::default(); WIDTH];

    // Memory buffers:
    // - states: Per-leaf scalar states (one per final row), maintained across matrices.
    // - scratch_states: Temporary buffer used when duplicating states during upsampling.
    let mut states = vec![scalar_default; final_height];
    let mut scratch_states = vec![scalar_default; final_height];

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
        absorb_matrix::<P, _, _, _, _>(&mut states[..height], matrix, sponge);

        active_height = height;
    }

    Ok(states
        .par_iter_mut()
        .map(|state| sponge.squeeze::<DIGEST_ELEMS>(state))
        .collect())
}

/// Build leaf digests using the cyclic view; see [`Lifting::Cyclic`] for semantics.
/// Conceptually, each matrix is virtually extended to height `H` by repeating its `h` rows
/// `L = H / h` times in a cycle (width unchanged), and the leaf `r` absorbs the `r`-th row from
/// each extended matrix in order. Each absorbed row is virtually padded with zeros to a multiple
/// of `RATE` for absorption; see [`LiftedMerkleTree`] docs for the equivalent single-matrix view.
///
/// # Preconditions
/// - `matrices` is non-empty and sorted by non-decreasing power-of-two heights.
/// - `P::WIDTH` is a power of two.
///
/// Panics in debug builds if preconditions are violated.
#[allow(dead_code)]
pub fn build_leaves_cyclic<
    P: PackedValue + Default,
    M: Matrix<P::Value>,
    H: StatefulSponge<P, WIDTH, RATE> + StatefulSponge<P::Value, WIDTH, RATE> + Sync,
    const WIDTH: usize,
    const RATE: usize,
    const DIGEST_ELEMS: usize,
>(
    matrices: &[M],
    sponge: &H,
) -> Result<Vec<[P::Value; DIGEST_ELEMS]>, LmcsError> {
    const { assert!(P::WIDTH.is_power_of_two()) };
    validate_heights(matrices.iter().map(|d| d.dimensions().height))?;

    let final_height = matrices.last().unwrap().height();

    let default_state = [P::Value::default(); WIDTH];
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
        absorb_matrix::<P, _, _, _, _>(&mut states[..height], matrix, sponge);

        active_height = height;
    }

    Ok(states
        .into_iter()
        .map(|state| sponge.squeeze::<DIGEST_ELEMS>(&state))
        .collect())
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
fn absorb_matrix<
    P: PackedValue + Default,
    M: Matrix<P::Value>,
    H: StatefulSponge<P, WIDTH, RATE> + StatefulSponge<P::Value, WIDTH, RATE> + Sync,
    const WIDTH: usize,
    const RATE: usize,
>(
    states: &mut [[P::Value; WIDTH]],
    matrix: &M,
    sponge: &H,
) {
    let height = matrix.height();
    assert_eq!(height, states.len());

    if height < P::WIDTH {
        // Scalar path: walk every final leaf state and absorb the wrapped row for this matrix.
        states
            .iter_mut()
            .zip(matrix.rows())
            .for_each(|(state, row)| {
                sponge.absorb(state, row);
            });
    } else {
        // SIMD path: gather → absorb wrapped packed row → scatter per chunk.
        states
            .par_chunks_mut(P::WIDTH)
            .enumerate()
            .for_each(|(packed_idx, states_chunk)| {
                let mut packed_state: [P; WIDTH] = pack_arrays(states_chunk);
                let row_idx = packed_idx * P::WIDTH;
                let row = matrix.vertically_packed_row(row_idx);
                sponge.absorb(&mut packed_state, row);
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
    C: Sync
        + PseudoCompressionFunction<[P::Value; DIGEST_ELEMS], 2>
        + PseudoCompressionFunction<[P; DIGEST_ELEMS], 2>,
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
    use p3_util::reverse_slice_index_bits;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use crate::lmcs::merkle_tree::Lifting;
    use crate::lmcs::test_helpers::{
        DIGEST, F, P, RATE, Sponge, build_leaves_single, components, concatenate_matrices,
        lift_matrix, matrix_scenarios, rand_matrix,
    };

    fn build_leaves_cyclic(matrices: &[RowMajorMatrix<F>], sponge: &Sponge) -> Vec<[F; DIGEST]> {
        super::build_leaves_cyclic::<P, _, _, _, _, _>(matrices, sponge).unwrap()
    }

    fn build_leaves_upsampled(matrices: &[RowMajorMatrix<F>], sponge: &Sponge) -> Vec<[F; DIGEST]> {
        super::build_leaves_upsampled::<P, _, _, _, _, _>(matrices, sponge).unwrap()
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
