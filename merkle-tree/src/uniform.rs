use alloc::vec;
use alloc::vec::Vec;
use core::array;
use core::marker::PhantomData;

use p3_field::PackedValue;
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_symmetric::{Hash, PseudoCompressionFunction, StatefulSponge};
use serde::{Deserialize, Serialize};

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
/// The tree construction uses state upsampling: as matrices grow in height, sponge states
/// duplicate to match the new height before absorbing additional rows. This ensures
/// uniform tree structure where all leaves at the same level have consistent hash state history.
///
/// Since [`StatefulSponge`] operates on a single field type, this tree uses the same type `F`
/// for both matrix elements and digest words, unlike `MerkleTree` which can hash `F → W`.
///
/// Use [`root`](Self::root) to fetch the final digest once the tree is built.
///
/// This generally shouldn't be used directly. If you're using a Merkle tree as an MMCS,
/// see the MMCS wrapper types.
#[derive(Debug, Serialize, Deserialize)]
pub struct UniformMerkleTree<F, M, const DIGEST_ELEMS: usize> {
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

    /// Zero-sized marker for type safety.
    _phantom: PhantomData<M>,
}

impl<F: Clone + Send + Sync + Copy + Default, M: Matrix<F>, const DIGEST_ELEMS: usize>
    UniformMerkleTree<F, M, DIGEST_ELEMS>
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
        leaves: Vec<M>,
    ) -> Self {
        assert!(!leaves.is_empty(), "No matrices given");

        // Build leaf digests from matrices using the sponge
        let leaf_digests = build_leaves_upsampled::<P, M, H, WIDTH, RATE, DIGEST_ELEMS>(&leaves, h);

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

        Self {
            leaves,
            digest_layers,
            _phantom: PhantomData,
        }
    }

    /// Return the root digest of the tree.
    #[must_use]
    pub fn root(&self) -> Hash<F, F, DIGEST_ELEMS> {
        self.digest_layers.last().unwrap()[0].into()
    }
}

/// Build leaf digests using an upsampled (nearest-neighbor) view of each matrix.
///
/// Conceptually, fix `H` to be the tallest input height. For a matrix with height `h`, define its
/// upsampled lifting to `H` by duplicating each original row into `H / h` consecutive rows. For
/// every final row index `r ∈ [0, H)`, form a single long row by horizontally concatenating the
/// rows at index `r` from all upsampled matrices (padding each matrix's width up to a multiple of
/// `RATE` with zeros). The digest for leaf `r` is the `squeeze` of a sponge that has `absorb`ed
/// exactly that long row.
///
/// This function produces the same result as the above “single concatenated matrix” definition,
/// but does so incrementally: it maintains one sponge state per final row and, as heights grow,
/// duplicates those states so that the per-row histories remain aligned with the upsampled view.
///
/// # Preconditions:
/// - `matrices` is non-empty and sorted by non-decreasing power-of-two heights.
/// - `P::WIDTH` is a power of two.
///
/// Panics in debug builds if preconditions are violated.
fn build_leaves_upsampled<
    P: PackedValue + Default,
    M: Matrix<P::Value>,
    H: StatefulSponge<P, WIDTH, RATE> + StatefulSponge<P::Value, WIDTH, RATE> + Sync,
    const WIDTH: usize,
    const RATE: usize,
    const DIGEST_ELEMS: usize,
>(
    matrices: &[M],
    sponge: &H,
) -> Vec<[P::Value; DIGEST_ELEMS]> {
    const {
        assert!(P::WIDTH.is_power_of_two());
    };
    let final_height = validate_matrix_heights(matrices);

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

    states
        .par_iter_mut()
        .map(|state| sponge.squeeze::<DIGEST_ELEMS>(state))
        .collect()
}

/// Build leaf digests using a cyclic (modulo) view of each matrix.
///
/// Let `H` be the tallest input height. For a matrix of height `h`, define its cyclic lifting to
/// `H` by mapping row index `r` to the original row `(r mod h)`. For every final row `r ∈ [0, H)`,
/// form one long row by horizontally concatenating the rows at index `r mod h_i` from all
/// matrices `i` (padding each matrix's width to a multiple of `RATE` with zeros). The digest for
/// leaf `r` is the `squeeze` of a sponge that has `absorb`ed exactly that long row.
///
/// This function realizes that semantics incrementally by keeping one running sponge state per
/// final row and, whenever the working height increases, duplicating the accumulated states so
/// that each new row range inherits the same history as its modulo counterpart.
///
/// # Preconditions:
/// - `matrices` is non-empty and sorted by non-decreasing power-of-two heights.
/// - `P::WIDTH` is a power of two.
///
/// Panics in debug builds if preconditions are violated.
#[cfg_attr(not(test), allow(dead_code))]
fn build_leaves_cyclic<
    P: PackedValue + Default,
    M: Matrix<P::Value>,
    H: StatefulSponge<P, WIDTH, RATE> + StatefulSponge<P::Value, WIDTH, RATE> + Sync,
    const WIDTH: usize,
    const RATE: usize,
    const DIGEST_ELEMS: usize,
>(
    matrices: &[M],
    sponge: &H,
) -> Vec<[P::Value; DIGEST_ELEMS]> {
    const { assert!(P::WIDTH.is_power_of_two()) };

    let final_height = validate_matrix_heights(matrices);
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

    states
        .into_iter()
        .map(|state| sponge.squeeze::<DIGEST_ELEMS>(&state))
        .collect()
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

/// Unpack a SIMD-packed array into multiple scalar arrays (one per SIMD lane).
///
/// Transposes packed SIMD layout into scalar layout. Each SIMD lane's values across all
/// array elements are extracted into a separate scalar array.
///
/// # Example Layout
/// Input: `packed_array = [[a0, a1, a2, a3], [b0, b1, b2, b3]]` (2 packed elements, width=4)
/// Output: `scalar_arrays[0] = [a0, b0]`, `scalar_arrays[1] = [a1, b1]`, etc.
fn unpack_array_into<P: PackedValue, const N: usize>(
    packed_array: &[P; N],
    scalar_arrays: &mut [[P::Value; N]],
) {
    for (lane, scalar_array) in scalar_arrays.iter_mut().take(P::WIDTH).enumerate() {
        *scalar_array = array::from_fn(|col| packed_array[col].as_slice()[lane]);
    }
}

/// Pack multiple scalar arrays (one per SIMD lane) into a SIMD-packed array.
///
/// Transposes scalar layout into packed SIMD layout. Values at the same position across
/// all scalar arrays are combined into a single SIMD-packed element.
///
/// # Example Layout
/// Input: `scalar[0] = [a0, b0]`, `scalar[1] = [a1, b1]`, ... (4 scalar arrays)
/// Output: `packed = [[a0, a1, a2, a3], [b0, b1, b2, b3]]` (2 packed elements, width=4)
fn pack_arrays<P: PackedValue, const N: usize>(scalar: &[[P::Value; N]]) -> [P; N] {
    array::from_fn(|col| P::from_fn(|lane| scalar[lane][col]))
}

fn validate_matrix_heights<P: PackedValue, M: Matrix<P>>(matrices: &[M]) -> usize {
    assert!(!matrices.is_empty(), "matrices cannot be empty");

    let mut prev_height = 1;
    for (idx, matrix) in matrices.iter().enumerate() {
        let height = matrix.height();
        assert!(
            height.is_power_of_two(),
            "matrix {idx} height {height} must be a power of two"
        );
        assert!(
            prev_height <= height,
            "matrices must be sorted by non-decreasing height"
        );
        prev_height = height;
    }
    prev_height
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_matrix::bitrev::BitReversibleMatrix;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_matrix::lifted::LiftableMatrix;
    use p3_symmetric::{PaddingFreeSponge, StatefulSponge, TruncatedPermutation};
    use p3_util::reverse_slice_index_bits;
    use rand::rngs::SmallRng;
    use rand::{RngCore, SeedableRng};

    use super::*;

    type F = BabyBear;
    type Packed = <F as Field>::Packing;

    const WIDTH: usize = 16;
    const RATE: usize = 8;
    const DIGEST: usize = 8;
    const PACK_WIDTH: usize = <Packed as PackedValue>::WIDTH;

    type Sponge = PaddingFreeSponge<Poseidon2BabyBear<WIDTH>, WIDTH, RATE, DIGEST>;

    /// Constructs deterministic Poseidon2 sponge and compression components for reproducible tests.
    fn poseidon_components() -> (
        Sponge,
        TruncatedPermutation<Poseidon2BabyBear<WIDTH>, 2, DIGEST, WIDTH>,
    ) {
        let mut rng = SmallRng::seed_from_u64(1);
        let permutation = Poseidon2BabyBear::<WIDTH>::new_from_rng_128(&mut rng);
        let sponge = PaddingFreeSponge::<_, WIDTH, RATE, DIGEST>::new(permutation.clone());
        let compressor = TruncatedPermutation::<_, 2, DIGEST, WIDTH>::new(permutation);
        (sponge, compressor)
    }

    /// Verifies that cyclic and upsampled leaf builders stay in sync across many dimensional scenarios.
    #[test]
    fn test_cyclic_upsampled_equivalence() {
        let (sponge, _) = poseidon_components();
        let mut rng = SmallRng::seed_from_u64(42);

        // Iterate through deterministic height/width configurations that cover scalar vs SIMD paths,
        // padding behaviour, and mixed-height batches.
        for scenario in matrix_scenarios() {
            let matrices: Vec<_> = scenario
                .into_iter()
                .map(|(height, width)| random_matrix(height, width, &mut rng))
                .collect();

            let max_height = matrices.last().unwrap().height();

            // Verify cyclic lifting
            {
                let leaves = build_leaves_cyclic::<Packed, _, _, _, _, _>(&matrices, &sponge);

                // Compare against matrices explicitly lifted cyclically to the final height.
                let matrices_cyclic: Vec<_> = matrices
                    .iter()
                    .map(|m| m.as_view().lift_cyclic(max_height).to_row_major_matrix())
                    .collect();
                let leaves_lifted =
                    build_leaves_cyclic::<Packed, _, _, _, _, _>(&matrices_cyclic, &sponge);
                assert_eq!(leaves, leaves_lifted);

                // Validate against a single concatenated baseline that pads widths to the rate.
                let matrix_single = concatenate_matrices(&matrices_cyclic);
                let leaves_single = build_leaves_single(&matrix_single, &sponge);
                assert_eq!(leaves, leaves_single);

                // Ensure equivalence after bit-reversing rows and using upsampled lifting.
                let matrices_bitreversed: Vec<_> = matrices
                    .iter()
                    .map(|m| m.as_view().bit_reverse_rows().to_row_major_matrix())
                    .collect();
                let mut leaves_bitreversed =
                    build_leaves_upsampled::<Packed, _, _, _, _, _>(&matrices_bitreversed, &sponge);
                reverse_slice_index_bits(&mut leaves_bitreversed);
                assert_eq!(leaves, leaves_bitreversed);
            };

            {
                let leaves =
                    build_leaves_upsampled::<Packed, _, _, _, _, DIGEST>(&matrices, &sponge);

                // Compare against matrices explicitly upsampled (row duplication) to the final height.
                let matrices_upsampled: Vec<_> = matrices
                    .iter()
                    .map(|m| m.as_view().lift_upsampled(max_height).to_row_major_matrix())
                    .collect();
                let leaves_lifted =
                    build_leaves_upsampled::<Packed, _, _, _, _, _>(&matrices_upsampled, &sponge);
                assert_eq!(leaves, leaves_lifted);

                // Validate against a single concatenated baseline that pads widths to the rate.
                let matrix_single = concatenate_matrices(&matrices_upsampled);
                let leaves_single = build_leaves_single(&matrix_single, &sponge);
                assert_eq!(leaves, leaves_single);

                // Ensure equivalence after bit-reversing rows and using cyclic lifting.
                let matrices_bitreversed: Vec<_> = matrices
                    .iter()
                    .map(|m| m.as_view().bit_reverse_rows().to_row_major_matrix())
                    .collect();
                let mut leaves_bitreversed =
                    build_leaves_cyclic::<Packed, _, _, _, _, _>(&matrices_bitreversed, &sponge);
                reverse_slice_index_bits(&mut leaves_bitreversed);
                assert_eq!(leaves, leaves_bitreversed);
            };
        }
    }

    /// Builds a `height × width` matrix filled with pseudo-random BabyBear values from `rng`.
    fn random_matrix(height: usize, width: usize, rng: &mut SmallRng) -> RowMajorMatrix<F> {
        let data = (0..height * width)
            .map(|_| F::new(rng.next_u32()))
            .collect();
        RowMajorMatrix::new(data, width)
    }

    /// Horizontally concatenates matrices after padding each row width to the rate and lifting to max height.
    fn concatenate_matrices(matrices: &[RowMajorMatrix<F>]) -> RowMajorMatrix<F> {
        let max_height = matrices.last().unwrap().height();
        let width: usize = matrices
            .iter()
            .map(|m| m.width().next_multiple_of(RATE))
            .sum();

        let concatenated_data: Vec<_> = (0..max_height)
            .flat_map(|idx| {
                matrices.iter().flat_map(move |m| {
                    let mut row = m.row_slice(idx).unwrap().to_vec();
                    let padded_width = row.len().next_multiple_of(RATE);
                    row.resize(padded_width, F::ZERO);
                    row
                })
            })
            .collect();
        RowMajorMatrix::new(concatenated_data, width)
    }

    /// Scalar baseline: absorb each row independently and squeeze a digest for comparison.
    fn build_leaves_single(
        matrix: &RowMajorMatrix<F>,
        sponge: &PaddingFreeSponge<Poseidon2BabyBear<WIDTH>, WIDTH, RATE, DIGEST>,
    ) -> Vec<[F; DIGEST]> {
        matrix
            .rows()
            .map(|row| {
                let mut state = [F::ZERO; WIDTH];
                sponge.absorb(&mut state, row);
                sponge.squeeze(&state)
            })
            .collect()
    }

    /// Enumerates deterministic height/width scenarios covering padding and SIMD corner cases.
    fn matrix_scenarios() -> Vec<Vec<(usize, usize)>> {
        vec![
            vec![(1, 1)],
            vec![(1, RATE - 1)],
            vec![(2, 3), (4, 5), (8, RATE)],
            vec![(1, 5), (1, 3), (2, 7), (4, 1), (8, RATE + 1)],
            vec![
                (PACK_WIDTH / 2, RATE - 1),
                (PACK_WIDTH, RATE),
                (PACK_WIDTH * 2, RATE + 3),
            ],
            vec![(PACK_WIDTH, RATE + 5), (PACK_WIDTH * 2, 25)],
            vec![
                (1, RATE * 2),
                (PACK_WIDTH / 2, RATE * 2 - 1),
                (PACK_WIDTH, RATE * 2),
                (PACK_WIDTH * 2, RATE * 3 - 2),
            ],
            vec![(4, RATE - 1), (4, RATE), (8, RATE + 3), (8, RATE * 2)],
            vec![(PACK_WIDTH * 2, RATE - 1)],
        ]
    }
}
