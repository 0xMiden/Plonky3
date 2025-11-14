use alloc::vec;
use alloc::vec::Vec;
use core::array;
use core::iter::zip;

use p3_field::PackedValue;
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_symmetric::{PseudoCompressionFunction, StatefulSponge};

/// Build all digest layers of a uniform Merkle tree from matrices.
///
/// Constructs a complete Merkle tree by first building leaf digests from the input matrices,
/// then repeatedly compressing pairs of digests until reaching a single root digest.
///
/// # Preconditions
/// - `matrices` must be sorted by height (shortest to tallest)
/// - All matrix heights must be powers of two
/// - `P::WIDTH` must be a power of two
///
/// # Parameters
/// - `matrices`: Input matrices to hash into leaf digests, pre-sorted by height
/// - `h`: Stateful sponge for incremental hashing of matrix rows
/// - `c`: Compression function for pair-wise digest compression
///
/// # Returns
/// Vector of digest layers, where:
/// - `result[0]` contains all leaf digests (one per row of the tallest matrix)
/// - `result[i+1]` contains half as many digests as `result[i]`
/// - `result[last]` contains a single root digest
///
/// # Example
/// ```ignore
/// let layers = build_matrix_digest_layers::<Packed, _, _, _, WIDTH, RATE, DIGEST>(
///     &[small_matrix, large_matrix],
///     &sponge,
///     &compressor
/// );
/// let root = layers.last().unwrap()[0];
/// ```
pub fn build_matrix_digest_layers<
    P,
    M,
    C,
    H,
    const WIDTH: usize,
    const RATE: usize,
    const DIGEST_ELEMS: usize,
>(
    matrices: &[M],
    h: &H,
    c: &C,
) -> Vec<Vec<[P::Value; DIGEST_ELEMS]>>
where
    P: PackedValue + Default,
    M: Matrix<P::Value>,
    H: StatefulSponge<P, WIDTH, RATE> + StatefulSponge<P::Value, WIDTH, RATE> + Sync,
    C: PseudoCompressionFunction<[P::Value; DIGEST_ELEMS], 2>
        + PseudoCompressionFunction<[P; DIGEST_ELEMS], 2>
        + Sync,
{
    let leaves = build_uniform_leaves::<P, M, H, WIDTH, RATE, DIGEST_ELEMS>(matrices, h);
    build_digest_layers::<P, C, DIGEST_ELEMS>(leaves, c)
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
fn build_uniform_leaves<P, M, H, const WIDTH: usize, const RATE: usize, const DIGEST_ELEMS: usize>(
    matrices: &[M],
    sponge: &H,
) -> Vec<[P::Value; DIGEST_ELEMS]>
where
    P: PackedValue + Default,
    M: Matrix<P::Value>,
    H: StatefulSponge<P, WIDTH, RATE> + StatefulSponge<P::Value, WIDTH, RATE> + Sync,
{
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

    // Process matrices from shortest to tallest, expanding the canonical states as we go.
    for matrix in matrices {
        let height = matrix.height();

        // Use scalar path when height < packing width to avoid vertically_packed_row
        // reading beyond matrix bounds (it uses modulo wrapping).
        if height < P::WIDTH {
            for (state, row) in zip(states.iter_mut(), matrix.rows()) {
                sponge.absorb(state, row);
            }
        } else {
            let scaling_factor = height / active_height;

            // Upsample states: duplicate each state to fill new rows.
            // E.g., [s0, s1] with scaling_factor=2 → [s0, s0, s1, s1]
            // Copy `states` into `scratch_states`, repeating each entry `scaling_factor` times
            scratch_states[..height]
                .par_chunks_mut(scaling_factor)
                .zip(states[..active_height].par_iter())
                .for_each(|(chunk, state)| chunk.fill(*state));

            // Pack the replicated scalar states into SIMD-friendly buffers.
            let packed_height = height.div_ceil(P::WIDTH);
            packed_states[..packed_height]
                .par_iter_mut()
                .enumerate()
                .for_each(|(packed_idx, packed_state)| {
                    let base_row_idx = packed_idx * P::WIDTH;
                    let states_chunk = &scratch_states[base_row_idx..base_row_idx + P::WIDTH];
                    pack_arrays_into(states_chunk, packed_state);
                });

            // Absorb the packed rows from the matrix into the sponge states.
            packed_states[..packed_height]
                .par_iter_mut()
                .enumerate()
                .for_each(|(packed_idx, packed_state)| {
                    let idx = packed_idx * P::WIDTH;
                    let row = matrix.vertically_packed_row(idx);
                    sponge.absorb(packed_state, row);
                });

            // Scatter the updated SIMD states back into the canonical scalar layout.
            states[..height]
                .par_chunks_mut(P::WIDTH)
                .zip(packed_states[..packed_height].par_iter())
                .for_each(|(states_chunk, packed_chunk)| {
                    unpack_array_into(packed_chunk, states_chunk);
                });
        }

        active_height = height;
    }

    states
        .par_iter_mut()
        .map(|state| sponge.squeeze::<DIGEST_ELEMS>(state))
        .collect()
}

/// Build all digest layers from leaf digests up to the root.
///
/// Repeatedly compresses each layer until a single root digest remains. Returns all layers
/// including the input leaf layer and the final root layer.
///
/// # Panics
/// Panics if `leaf_digests` is empty.
fn build_digest_layers<P, C, const DIGEST_ELEMS: usize>(
    leaf_digests: Vec<[P::Value; DIGEST_ELEMS]>,
    c: &C,
) -> Vec<Vec<[P::Value; DIGEST_ELEMS]>>
where
    P: PackedValue,
    C: PseudoCompressionFunction<[P::Value; DIGEST_ELEMS], 2>
        + PseudoCompressionFunction<[P; DIGEST_ELEMS], 2>
        + Sync,
{
    assert!(!leaf_digests.is_empty(), "leaf_digests cannot be empty");

    let mut digest_layers = vec![leaf_digests];

    loop {
        let prev_layer = digest_layers.last().unwrap();
        if prev_layer.len() == 1 {
            break;
        }

        let next_layer = compress_uniform::<P, C, DIGEST_ELEMS>(prev_layer, c);
        digest_layers.push(next_layer);
    }

    digest_layers
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
fn compress_uniform<P, C, const DIGEST_ELEMS: usize>(
    prev_layer: &[[P::Value; DIGEST_ELEMS]],
    c: &C,
) -> Vec<[P::Value; DIGEST_ELEMS]>
where
    P: PackedValue,
    C: PseudoCompressionFunction<[P::Value; DIGEST_ELEMS], 2>
        + PseudoCompressionFunction<[P; DIGEST_ELEMS], 2>
        + Sync,
{
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
    use p3_matrix::Matrix;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_symmetric::{PaddingFreeSponge, StatefulSponge, TruncatedPermutation};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    type F = BabyBear;
    type Packed = <F as Field>::Packing;

    const WIDTH: usize = 16;
    const RATE: usize = 8;
    const DIGEST: usize = 8;
    const PACK_WIDTH: usize = <Packed as PackedValue>::WIDTH;

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

    fn reference_uniform_leaves(
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

            if height < PACK_WIDTH {
                // Scalar path: directly copy states to scratch
                scratch[..height].copy_from_slice(&states[..height]);
            } else {
                // Packed path: upsample states
                let growth = height / active_height;
                for (row, scratch_state) in scratch.iter_mut().enumerate().take(height) {
                    *scratch_state = states[row / growth];
                }
            }

            // Absorb matrix rows into scratch states
            for (row, scratch_state) in scratch.iter_mut().enumerate().take(height) {
                let row_iter = matrix.row(row).expect("row exists").into_iter();
                sponge.absorb(scratch_state, row_iter);
            }

            // Copy updated scratch back to canonical states
            states[..height].copy_from_slice(&scratch[..height]);
            active_height = height;
        }

        states
            .iter_mut()
            .map(|state| sponge.squeeze::<DIGEST>(state))
            .collect()
    }

    #[test]
    fn uniform_leaves_match_reference() {
        let (sponge, _) = poseidon_components();

        let small_height = if PACK_WIDTH > 1 { PACK_WIDTH / 2 } else { 1 };
        let large_height = PACK_WIDTH * 2;
        assert!(small_height.is_power_of_two() && small_height > 0);
        assert!(large_height.is_power_of_two());

        let small = field_matrix(small_height, 3, 1);
        let large = field_matrix(large_height, 5, 1_000);

        let matrices = vec![small, large];
        let leaves = build_uniform_leaves::<Packed, _, _, WIDTH, RATE, DIGEST>(&matrices, &sponge);

        let expected = reference_uniform_leaves(matrices, &sponge);
        assert_eq!(leaves, expected);
    }

    #[test]
    fn digest_layers_match_truncated_poseidon() {
        let (sponge, compressor) = poseidon_components();

        let matrix = field_matrix(PACK_WIDTH * 2, 4, 10_000);
        let matrices = vec![matrix];

        let leaves = build_uniform_leaves::<Packed, _, _, WIDTH, RATE, DIGEST>(&matrices, &sponge);
        let reference = reference_uniform_leaves(matrices, &sponge);
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

        let actual_layers = build_digest_layers::<Packed, _, DIGEST>(leaves, &compressor);
        assert_eq!(actual_layers, naive_layers);
    }
}
