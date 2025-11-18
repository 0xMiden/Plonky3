use core::array;

use p3_field::PackedValue;

/// Unpack a SIMD-packed array into multiple scalar arrays (one per SIMD lane).
///
/// Transposes packed SIMD layout into scalar layout. Each SIMD lane's values across all
/// array elements are extracted into a separate scalar array.
///
/// # Example Layout
/// Input: `packed_array = [[a0, a1, a2, a3], [b0, b1, b2, b3]]` (2 packed elements, width=4)
/// Output: `scalar_arrays[0] = [a0, b0]`, `scalar_arrays[1] = [a1, b1]`, etc.
pub(crate) fn unpack_array_into<P: PackedValue, const N: usize>(
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
pub(crate) fn pack_arrays<P: PackedValue, const N: usize>(scalar: &[[P::Value; N]]) -> [P; N] {
    array::from_fn(|col| P::from_fn(|lane| scalar[lane][col]))
}
