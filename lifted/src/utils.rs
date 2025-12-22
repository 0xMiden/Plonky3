use alloc::vec::Vec;

use p3_field::TwoAdicField;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_util::reverse_slice_index_bits;

/// Compute padding needed to align `len` to `alignment`.
///
/// Returns the number of zeros to add so that `len + padding` is a multiple of `alignment`.
#[inline]
pub const fn alignment_padding(len: usize, alignment: usize) -> usize {
    len.next_multiple_of(alignment) - len
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
