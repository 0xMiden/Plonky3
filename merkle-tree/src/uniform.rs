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

/// Build the leaf digests that would appear at the base of a uniform Merkle tree.
///
/// Matrices are processed from shortest to tallest. For each matrix we rebuild a scratch slice of
/// scalar states sized to that matrix, pack it into SIMD lanes, absorb the matrix-provided inputs
/// into the sponge state, then write the updated lanes back into the canonical per-leaf state
/// buffer.
///
/// # Preconditions
/// - **Matrices must be pre-sorted by height** (shortest to tallest). Verified at runtime.
/// - Every matrix height and `P::WIDTH` must be powers of two.
///
/// # Algorithm
///
/// Maintains one sponge state per final leaf row. As matrices grow from height h → 2h,
/// each state duplicates (upsamples) before absorbing the new matrix's rows.
///
/// **Scalar path** (height < P::WIDTH): Process rows individually to avoid reading
/// beyond matrix bounds with `vertically_packed_row` (which uses modulo wrapping).
///
/// **Packed path** (height >= P::WIDTH):
/// 1. Upsample: duplicate states to match new height
/// 2. Pack: transpose scalar states into SIMD layout
/// 3. Absorb: process matrix rows with packed operations
/// 4. Unpack: transpose SIMD states back to scalar layout
///
/// # Panics
/// Panics if `matrices` is empty or if `P::WIDTH` is not a power of two.
/// Debug builds also verify all matrix heights are non-zero and powers of two.
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
    assert!(!matrices.is_empty(), "matrices cannot be empty");
    assert!(P::WIDTH.is_power_of_two());

    // Sanity-check matrix heights once so the main loop can assume clean invariants.
    let mut prev_height = 1;
    for matrix in matrices {
        let height = matrix.height();
        assert!(
            height.is_power_of_two(),
            "matrix heights must be a power of two",
        );
        assert!(
            prev_height <= height,
            "matrices must be sorted by height and non-empty"
        );
        prev_height = height;
    }

    let initial_height = matrices[0].height();

    let final_height = matrices.last().unwrap().height();
    let final_height_packed = final_height.div_ceil(P::WIDTH);

    let scalar_default = [P::Value::default(); WIDTH];
    let packed_default = [P::default(); WIDTH];

    // Memory buffers:
    // - states: Per-leaf scalar states (one per final row), maintained across matrices
    // - scratch_states: Temporary buffer for upsampling during matrix growth
    // - packed_states: Transposed SIMD layout for packed absorption
    let mut states = vec![scalar_default; final_height];
    let mut scratch_states = vec![scalar_default; final_height];
    let mut packed_states = vec![packed_default; final_height_packed];

    let mut active_height = initial_height;

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

        // For small matrices whose height is smaller than the packing width,
        // fall back to the scalar case for absorbing rows into the state.
        if height < P::WIDTH {
            states[..height]
                .iter_mut()
                .zip(matrix.rows())
                .for_each(|(state, row)| {
                    sponge.absorb(state, row);
                });
            continue;
        }

        // Pack the scalar states into SIMD-friendly buffers so we can absorb packed rows.
        let packed_height = height.div_ceil(P::WIDTH);
        packed_states[..packed_height]
            .par_iter_mut()
            .enumerate()
            .for_each(|(packed_idx, packed_state)| {
                let base_row_idx = packed_idx * P::WIDTH;
                let states_chunk = &states[base_row_idx..base_row_idx + P::WIDTH];
                pack_arrays_into(states_chunk, packed_state);
            });

        // Absorb matrix packed matrix rows
        packed_states[..packed_height]
            .par_iter_mut()
            .enumerate()
            .for_each(|(packed_idx, packed_state)| {
                let idx = packed_idx * P::WIDTH;
                let row = matrix.vertically_packed_row(idx);
                sponge.absorb(packed_state, row);
            });

        // Scatter packed scalar states so the next matrix sees scalar layout.
        states[..height]
            .par_chunks_mut(P::WIDTH)
            .zip(packed_states[..packed_height].par_iter())
            .for_each(|(states_chunk, packed_chunk)| {
                unpack_array_into(packed_chunk, states_chunk);
            });

        active_height = height;
    }

    states
        .par_iter_mut()
        .map(|state| sponge.squeeze::<DIGEST_ELEMS>(state))
        .collect()
}

/// Build the leaf digests that would appear at the base of a uniform Merkle tree.
///
/// Every matrix is virtually lifted to the tallest height by cycling through its rows until the
/// target height is reached. For each final row index `r`, we absorb the row `r % h` from each
/// matrix of height `h` into a shared sponge state, keeping the matrices logically aligned.
///
/// # Preconditions
/// - Matrices must be pre-sorted by height (shortest to tallest). Verified at runtime.
/// - Every matrix height must be a non-zero power of two.
///
/// # Panics
/// Panics if `matrices` is empty or if the ordering/height invariants are violated.
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
    assert!(!matrices.is_empty(), "matrices cannot be empty");
    assert!(P::WIDTH.is_power_of_two());

    let mut prev_height = 0usize;
    for (idx, matrix) in matrices.iter().enumerate() {
        let height = matrix.height();
        assert!(height > 0, "matrix {idx} has zero height");
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

    let final_height = matrices.last().unwrap().height();
    assert!(
        final_height.is_power_of_two(),
        "final height must be a power of two"
    );
    let final_height_packed = final_height.div_ceil(P::WIDTH);

    let scalar_default = [P::Value::default(); WIDTH];
    let packed_default = [P::default(); WIDTH];

    let mut states = vec![scalar_default; final_height];
    let mut packed_states = vec![packed_default; final_height_packed];

    // Process matrices in ascending height, cycling each shorter matrix over the final leaf range.
    for matrix in matrices {
        let height = matrix.height();
        let height_mask = height - 1;

        if height < P::WIDTH {
            // Scalar path: walk every final leaf state and absorb the corresponding wrapped row.
            states.iter_mut().enumerate().for_each(|(row_idx, state)| {
                let wrapped_row = row_idx & height_mask;
                let row = matrix.row(wrapped_row).unwrap();
                sponge.absorb(state, row);
            });
            continue;
        }

        // Pack scalar states for SIMD absorption when we have full-width chunks.
        packed_states
            .par_iter_mut()
            .enumerate()
            .for_each(|(packed_idx, packed_state)| {
                let base_row_idx = packed_idx * P::WIDTH;
                let states_chunk = &states[base_row_idx..base_row_idx + P::WIDTH];
                pack_arrays_into(states_chunk, packed_state);
            });

        // Absorb rows in a vertically packed form; row indices wrap via mask.
        packed_states
            .par_iter_mut()
            .enumerate()
            .for_each(|(packed_idx, packed_state)| {
                let base_row_idx = packed_idx * P::WIDTH;
                let start = base_row_idx & height_mask;
                let row = matrix.vertically_packed_row(start);
                sponge.absorb(packed_state, row);
            });

        // Scatter SIMD lanes back into scalar layout for continued processing.
        states
            .par_chunks_mut(P::WIDTH)
            .zip(packed_states.par_iter())
            .for_each(|(states_chunk, packed_chunk)| {
                unpack_array_into(packed_chunk, states_chunk);
            });
    }

    states
        .into_iter()
        .map(|state| sponge.squeeze::<DIGEST_ELEMS>(&state))
        .collect()
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
fn pack_arrays_into<P: PackedValue, const N: usize>(scalar: &[[P::Value; N]], packed: &mut [P; N]) {
    for (col, val) in packed.iter_mut().enumerate() {
        *val = P::from_fn(|lane| scalar[lane][col]);
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_matrix::bitrev::BitReversibleMatrix;
    use p3_matrix::dense::RowMajorMatrix;
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

    #[derive(Clone, Copy, Debug)]
    enum BuildMode {
        Upsampled,
        Cyclic,
    }

    impl BuildMode {
        fn name(self) -> &'static str {
            match self {
                BuildMode::Upsampled => "upsampled",
                BuildMode::Cyclic => "cyclic",
            }
        }

        fn build(
            self,
            matrices: &[RowMajorMatrix<F>],
            sponge: &PaddingFreeSponge<Poseidon2BabyBear<WIDTH>, WIDTH, RATE, DIGEST>,
        ) -> Vec<[F; DIGEST]> {
            match self {
                BuildMode::Upsampled => {
                    build_leaves_upsampled::<Packed, _, _, WIDTH, RATE, DIGEST>(matrices, sponge)
                }
                BuildMode::Cyclic => {
                    build_leaves_cyclic::<Packed, _, _, WIDTH, RATE, DIGEST>(matrices, sponge)
                }
            }
        }

        fn reference(
            self,
            matrices: Vec<RowMajorMatrix<F>>,
            sponge: &PaddingFreeSponge<Poseidon2BabyBear<WIDTH>, WIDTH, RATE, DIGEST>,
        ) -> Vec<[F; DIGEST]> {
            match self {
                BuildMode::Upsampled => reference_leaves_upsampled(matrices, sponge),
                BuildMode::Cyclic => reference_leaves_cyclic(matrices, sponge),
            }
        }
    }

    fn poseidon_components() -> (
        PaddingFreeSponge<Poseidon2BabyBear<WIDTH>, WIDTH, RATE, DIGEST>,
        TruncatedPermutation<Poseidon2BabyBear<WIDTH>, 2, DIGEST, WIDTH>,
    ) {
        let mut rng = SmallRng::seed_from_u64(1);
        let permutation = Poseidon2BabyBear::<WIDTH>::new_from_rng_128(&mut rng);
        let sponge = PaddingFreeSponge::<_, WIDTH, RATE, DIGEST>::new(permutation.clone());
        let compressor = TruncatedPermutation::<_, 2, DIGEST, WIDTH>::new(permutation);
        (sponge, compressor)
    }

    fn field_matrix(rows: usize, cols: usize, offset: u32) -> RowMajorMatrix<F> {
        let data = (0..rows * cols)
            .map(|i| F::new(offset + i as u32))
            .collect::<Vec<_>>();
        RowMajorMatrix::new(data, cols)
    }

    fn reference_leaves_upsampled(
        mut matrices: Vec<RowMajorMatrix<F>>,
        sponge: &PaddingFreeSponge<Poseidon2BabyBear<WIDTH>, WIDTH, RATE, DIGEST>,
    ) -> Vec<[F; DIGEST]> {
        matrices.sort_by_key(|m| m.height());
        assert!(!matrices.is_empty());

        let final_height = matrices.last().unwrap().height();
        let mut states = vec![[F::ZERO; WIDTH]; final_height];
        let mut scratch = states.clone();
        let mut active_height = matrices.first().unwrap().height();

        for matrix in matrices.iter() {
            let height = matrix.height();

            if height > active_height {
                let growth = height / active_height;
                for (row, scratch_state) in scratch.iter_mut().enumerate().take(height) {
                    *scratch_state = states[row / growth];
                }
                states[..height].copy_from_slice(&scratch[..height]);
            }

            for (row, state) in states.iter_mut().enumerate().take(height) {
                let row_iter = matrix.row(row).expect("row exists").into_iter();
                sponge.absorb(state, row_iter);
            }

            active_height = height;
        }

        states
            .iter_mut()
            .map(|state| sponge.squeeze::<DIGEST>(state))
            .collect()
    }

    fn reference_leaves_cyclic(
        mut matrices: Vec<RowMajorMatrix<F>>,
        sponge: &PaddingFreeSponge<Poseidon2BabyBear<WIDTH>, WIDTH, RATE, DIGEST>,
    ) -> Vec<[F; DIGEST]> {
        matrices.sort_by_key(|m| m.height());
        assert!(!matrices.is_empty());

        let final_height = matrices.last().unwrap().height();
        let mut leaves = Vec::with_capacity(final_height);

        for row_idx in 0..final_height {
            let mut state = [F::ZERO; WIDTH];
            for (matrix_idx, matrix) in matrices.iter().enumerate() {
                let height = matrix.height();
                let wrapped_row = row_idx % height;
                let row = matrix
                    .row(wrapped_row)
                    .unwrap_or_else(|| panic!("row {wrapped_row} missing in matrix {matrix_idx}"));
                sponge.absorb(&mut state, row.into_iter());
            }
            leaves.push(sponge.squeeze::<DIGEST>(&mut state));
        }

        leaves
    }

    fn upsample_matrix(matrix: &RowMajorMatrix<F>, target_height: usize) -> RowMajorMatrix<F> {
        assert!(target_height.is_power_of_two());
        assert!(target_height >= matrix.height());
        assert_eq!(target_height % matrix.height(), 0);
        assert!(matrix.height().is_power_of_two());

        let width = matrix.width();
        let scaling = target_height / matrix.height();
        let mut data = Vec::with_capacity(target_height * width);

        for row_idx in 0..target_height {
            let src = matrix
                .row(row_idx / scaling)
                .expect("source row exists")
                .into_iter();
            data.extend(src);
        }

        RowMajorMatrix::new(data, width)
    }

    fn lift_matrix(matrix: &RowMajorMatrix<F>, target_height: usize) -> RowMajorMatrix<F> {
        assert!(target_height.is_power_of_two());
        assert!(target_height >= matrix.height());
        assert_eq!(target_height % matrix.height(), 0);
        assert!(matrix.height().is_power_of_two());

        let width = matrix.width();
        let mut data = Vec::with_capacity(target_height * width);

        for row_idx in 0..target_height {
            let src = matrix
                .row(row_idx % matrix.height())
                .expect("source row exists")
                .into_iter();
            data.extend(src);
        }

        RowMajorMatrix::new(data, width)
    }

    fn build_reference_matrix(
        mode: BuildMode,
        matrices: &[RowMajorMatrix<F>],
        max_height: usize,
    ) -> RowMajorMatrix<F> {
        let mut total_width = 0;

        for matrix in matrices {
            total_width += matrix.width().next_multiple_of(RATE);
        }

        let mut result_data = vec![F::ZERO; max_height * total_width];
        let mut col_offset = 0;

        for matrix in matrices {
            let height = matrix.height();
            let width = matrix.width();
            let padded_width = width.next_multiple_of(RATE);
            let scaling = max_height / height;

            for dst_row in 0..max_height {
                let src_row = match mode {
                    BuildMode::Upsampled => dst_row / scaling,
                    BuildMode::Cyclic => dst_row % height,
                };
                for col in 0..width {
                    let dst_idx = dst_row * total_width + col_offset + col;
                    result_data[dst_idx] = matrix.get(src_row, col).expect("row and col in bounds");
                }
            }

            col_offset += padded_width;
        }

        RowMajorMatrix::new(result_data, total_width)
    }

    fn reference_leaves_from_single_matrix(
        matrix: &RowMajorMatrix<F>,
        sponge: &PaddingFreeSponge<Poseidon2BabyBear<WIDTH>, WIDTH, RATE, DIGEST>,
    ) -> Vec<[F; DIGEST]> {
        let mut leaves = Vec::with_capacity(matrix.height());
        for row_idx in 0..matrix.height() {
            let mut state = [F::ZERO; WIDTH];
            let row = matrix.row(row_idx).expect("row exists");
            sponge.absorb(&mut state, row.into_iter());
            leaves.push(sponge.squeeze::<DIGEST>(&mut state));
        }
        leaves
    }

    #[test]
    fn leaves_match_reference() {
        let (sponge, _) = poseidon_components();

        for mode in [BuildMode::Upsampled, BuildMode::Cyclic] {
            let mut matrices = vec![field_matrix(2, 3, 10), field_matrix(4, 5, 1_000)];
            matrices.sort_by_key(|m| m.height());

            let actual = mode.build(&matrices, &sponge);
            let expected = mode.reference(matrices.clone(), &sponge);

            assert_eq!(
                actual,
                expected,
                "{} leaves should match reference construction",
                mode.name()
            );
        }
    }

    #[test]
    fn small_heights_regression() {
        let (sponge, _) = poseidon_components();

        let mut heights = Vec::new();
        let mut h = 1;
        while h < PACK_WIDTH && h <= 64 {
            heights.push(h);
            h *= 2;
        }

        if heights.is_empty() {
            heights.push(1);
        }

        let mut matrices: Vec<_> = heights
            .iter()
            .enumerate()
            .map(|(i, &height)| field_matrix(height, 3, i as u32 * 200))
            .collect();
        matrices.sort_by_key(|m| m.height());

        for mode in [BuildMode::Upsampled, BuildMode::Cyclic] {
            let actual = mode.build(&matrices, &sponge);
            let expected = mode.reference(matrices.clone(), &sponge);

            assert_eq!(
                actual,
                expected,
                "small heights should match reference ({})",
                mode.name()
            );
        }
    }

    #[test]
    fn random_matrices_match_reference() {
        let (sponge, _) = poseidon_components();
        let mut rng = SmallRng::seed_from_u64(42);

        let test_cases = [
            vec![1, 2, 4, 8],
            vec![2, 4, 8, 16],
            vec![1, 1, 2, 4, 8],
            vec![4, 8, 8, 16],
            vec![1, 2, 4, 8, 16, 32],
        ];

        for (test_idx, heights) in test_cases.iter().enumerate() {
            let max_height = *heights.iter().max().unwrap();
            if max_height > 64 {
                continue;
            }

            let mut matrices: Vec<RowMajorMatrix<F>> = heights
                .iter()
                .enumerate()
                .map(|(i, &h)| {
                    let width = (i % 5) + 1;
                    let data: Vec<F> = (0..h * width).map(|_| F::new(rng.next_u32())).collect();
                    RowMajorMatrix::new(data, width)
                })
                .collect();
            matrices.sort_by_key(|m| m.height());

            for mode in [BuildMode::Upsampled, BuildMode::Cyclic] {
                let actual = mode.build(&matrices, &sponge);
                let expected = mode.reference(matrices.clone(), &sponge);

                assert_eq!(
                    actual,
                    expected,
                    "test case {} (mode {}) should match reference",
                    test_idx,
                    mode.name()
                );
                assert_eq!(
                    actual.len(),
                    max_height,
                    "test case {} (mode {}) should produce max_height leaves",
                    test_idx,
                    mode.name()
                );
            }
        }
    }

    #[test]
    fn bit_reverse_equivalence_between_modes() {
        let (sponge, _) = poseidon_components();
        let mut rng = SmallRng::seed_from_u64(404);

        let scenarios = [
            vec![1, 2, 4, 8],
            vec![2, 4, 8, 16],
            vec![1, 1, 2, 4, 8],
            vec![4, 8, 8, 16],
        ];

        for (case_idx, heights) in scenarios.iter().enumerate() {
            let mut matrices: Vec<RowMajorMatrix<F>> = heights
                .iter()
                .enumerate()
                .map(|(i, &height)| {
                    let width = (i % 5) + 1;
                    let data = (0..height * width)
                        .map(|_| F::new(rng.next_u32()))
                        .collect::<Vec<_>>();
                    RowMajorMatrix::new(data, width)
                })
                .collect();
            matrices.sort_by_key(|m| m.height());

            let upsampled = BuildMode::Upsampled.build(&matrices, &sponge);
            let cyclic = BuildMode::Cyclic.build(&matrices, &sponge);

            let mut bitrev_matrices: Vec<RowMajorMatrix<F>> = matrices
                .iter()
                .map(|m| m.clone().bit_reverse_rows().to_row_major_matrix())
                .collect();
            bitrev_matrices.sort_by_key(|m| m.height());

            let upsampled_from_bitrev = BuildMode::Upsampled.build(&bitrev_matrices, &sponge);
            let cyclic_from_bitrev = BuildMode::Cyclic.build(&bitrev_matrices, &sponge);

            let mut upsampled_bitrev = upsampled.clone();
            reverse_slice_index_bits(&mut upsampled_bitrev);
            assert_eq!(
                upsampled_bitrev, cyclic_from_bitrev,
                "scenario {case_idx} (upsampled → cyclic via bit-reversal) mismatch"
            );

            let mut cyclic_bitrev = cyclic.clone();
            reverse_slice_index_bits(&mut cyclic_bitrev);
            assert_eq!(
                cyclic_bitrev, upsampled_from_bitrev,
                "scenario {case_idx} (cyclic → upsampled via bit-reversal) mismatch"
            );
        }
    }

    #[test]
    fn bit_reverse_equivalence_single_matrix() {
        let mut rng = SmallRng::seed_from_u64(2024);
        let base_height = 2;
        let width = 5;
        let scaling = 8;
        let target_height = base_height * scaling;

        let data = (0..base_height * width)
            .map(|_| F::new(rng.next_u32()))
            .collect::<Vec<_>>();
        let base_matrix = RowMajorMatrix::new(data, width);

        let upsampled = upsample_matrix(&base_matrix, target_height);
        let lifted = lift_matrix(&base_matrix, target_height);

        let upsampled_br = upsampled.clone().bit_reverse_rows().to_row_major_matrix();
        let lifted_br = lifted.clone().bit_reverse_rows().to_row_major_matrix();

        assert_eq!(
            upsampled_br, lifted,
            "bit-reversed upsampling should match cyclic lifting"
        );
        assert_eq!(
            lifted_br, upsampled,
            "bit-reversed lifting should match upsampling"
        );
    }

    #[test]
    fn reference_matrix_comparison() {
        let (sponge, _) = poseidon_components();
        let mut rng = SmallRng::seed_from_u64(1337);

        let scenarios = [vec![1, 2, 4, 8], vec![2, 2, 4, 8, 8], vec![1, 4, 16]];

        for (case_idx, heights) in scenarios.iter().enumerate() {
            let max_height = *heights.iter().max().unwrap();

            let mut matrices: Vec<RowMajorMatrix<F>> = heights
                .iter()
                .enumerate()
                .map(|(col_idx, &height)| {
                    let width = (col_idx % 4) + 1;
                    let data: Vec<F> = (0..height * width)
                        .map(|_| F::new(rng.next_u32()))
                        .collect();
                    RowMajorMatrix::new(data, width)
                })
                .collect();
            matrices.sort_by_key(|m| m.height());

            for mode in [BuildMode::Upsampled, BuildMode::Cyclic] {
                let leaves = mode.build(&matrices, &sponge);
                let reference_matrix = build_reference_matrix(mode, &matrices, max_height);
                let expected = reference_leaves_from_single_matrix(&reference_matrix, &sponge);

                assert_eq!(
                    leaves,
                    expected,
                    "scenario {case_idx} (mode {}) should match reference matrix",
                    mode.name()
                );
                assert_eq!(
                    leaves.len(),
                    max_height,
                    "scenario {case_idx} (mode {}) should produce max_height leaves",
                    mode.name()
                );
            }
        }
    }

    #[test]
    fn digest_layers_match_truncated_poseidon() {
        let (sponge, compressor) = poseidon_components();

        let matrix = field_matrix(PACK_WIDTH * 2, 4, 10_000);
        let matrices = vec![matrix];

        let leaves =
            build_leaves_upsampled::<Packed, _, _, WIDTH, RATE, DIGEST>(&matrices, &sponge);
        let reference = reference_leaves_upsampled(matrices, &sponge);
        assert_eq!(leaves, reference);

        let mut naive_layers = vec![reference.clone()];
        let mut current = reference;
        while current.len() > 1 {
            let mut next = Vec::with_capacity(current.len() / 2);
            for pair in current.chunks_exact(2) {
                next.push(compressor.compress([pair[0], pair[1]]));
            }
            naive_layers.push(next.clone());
            current = next;
        }

        let mut actual_layers = vec![leaves];
        loop {
            let prev_layer = actual_layers.last().unwrap();
            if prev_layer.len() == 1 {
                break;
            }
            let next_layer = compress_uniform::<Packed, _, DIGEST>(prev_layer, &compressor);
            actual_layers.push(next_layer);
        }

        assert_eq!(actual_layers, naive_layers);
    }
}
