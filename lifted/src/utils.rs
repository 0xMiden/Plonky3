use alloc::vec::Vec;

use p3_field::TwoAdicField;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_util::reverse_slice_index_bits;

use crate::LmcsError;

/// Validate a sequence of matrix heights for LMCS.
///
/// Requirements enforced:
/// - Non-empty sequence (at least one matrix).
/// - Every height is a power of two and non-zero.
/// - Heights are in non-decreasing order (sorted by height), so the last height is the maximum
///   `H` used by lifting.
///
/// Returns `Ok(max_height)` with the maximum height if all checks pass; otherwise returns a
/// specific [`LmcsError`]:
/// - [`LmcsError::ZeroHeightMatrix`]
/// - [`LmcsError::NonPowerOfTwoHeight`]
/// - [`LmcsError::UnsortedByHeight`]
/// - [`LmcsError::EmptyBatch`]
///
/// The `matrix` index in the errors refers to the position within the provided iterator.
pub fn validate_heights(heights: impl IntoIterator<Item = usize>) -> Result<usize, LmcsError> {
    let mut active_height = 0;

    for (matrix, height) in heights.into_iter().enumerate() {
        if height == 0 {
            return Err(LmcsError::ZeroHeightMatrix { matrix });
        }

        if !height.is_power_of_two() {
            return Err(LmcsError::NonPowerOfTwoHeight { matrix, height });
        }

        if height < active_height {
            return Err(LmcsError::UnsortedByHeight);
        }
        active_height = height;
    }

    if active_height == 0 {
        return Err(LmcsError::EmptyBatch);
    }
    Ok(active_height)
}

/// Coset points `gK` in bit-reversed order.
///
/// Bit-reversal gives two properties essential for lifting:
/// - **Adjacent negation**: `gK[2i+1] = -gK[2i]`, so both square to the same value
/// - **Prefix nesting**: `gK[0..n/r]` equals the r-th power coset `(gK)ʳ`
///
/// Together these enable iterative weight folding in barycentric evaluation.
pub fn bit_reversed_coset_points<F: TwoAdicField>(log_n: usize) -> Vec<F> {
    let coset = TwoAdicMultiplicativeCoset::new(F::GENERATOR, log_n).unwrap();
    let mut pts: Vec<F> = coset.iter().collect();
    reverse_slice_index_bits(&mut pts);
    pts
}
