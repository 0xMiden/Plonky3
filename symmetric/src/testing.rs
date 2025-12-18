//! Mock implementations for testing symmetric cryptographic primitives.
//!
//! This module provides [`MockPermutation`] and [`MockHasher`] for testing
//! adapter behavior without real cryptographic primitives.

use crate::CryptographicHasher;
use crate::permutation::{CryptographicPermutation, Permutation};

// =============================================================================
// Binary Mock Implementations (u8, u32, u64)
// =============================================================================

/// A mock permutation for binary types that sums all elements (with wrapping) and fills the state.
///
/// This deterministic, easily traceable behavior makes it ideal for testing
/// adapter logic without the complexity of a real cryptographic permutation.
///
/// # Example
/// ```
/// use p3_symmetric::testing::MockPermutation;
/// use p3_symmetric::Permutation;
///
/// let perm = MockPermutation;
/// let mut state = [1u64, 2, 3, 4];
/// perm.permute_mut(&mut state);
/// assert_eq!(state, [10, 10, 10, 10]); // 1+2+3+4 = 10
/// ```
#[derive(Clone, Debug, Default)]
pub struct MockPermutation;

/// A mock hasher for binary types that sums all input and fills output with the sum.
///
/// Supports u8, u32, u64 in both scalar and parallel (any M) modes.
/// This deterministic behavior makes it ideal for testing adapter logic.
///
/// # Example
/// ```
/// use p3_symmetric::testing::MockHasher;
/// use p3_symmetric::CryptographicHasher;
///
/// let hasher = MockHasher;
/// let output: [u8; 4] = hasher.hash_iter([1u8, 2, 3, 4]);
/// assert_eq!(output, [10, 10, 10, 10]); // 1+2+3+4 = 10
/// ```
#[derive(Clone, Debug, Default)]
pub struct MockHasher;

macro_rules! impl_mock_binary_permutation {
    ($($t:ty),*) => {$(
        impl<const WIDTH: usize> Permutation<[$t; WIDTH]> for MockPermutation {
            fn permute_mut(&self, input: &mut [$t; WIDTH]) {
                let sum: $t = input.iter().copied().fold(0, |a, b| a.wrapping_add(b));
                *input = [sum; WIDTH];
            }
        }
        impl<const WIDTH: usize> CryptographicPermutation<[$t; WIDTH]> for MockPermutation {}
    )*};
}

impl_mock_binary_permutation!(u8, u32, u64, i32);

macro_rules! impl_mock_binary_hasher {
    ($($t:ty),*) => {$(
        // Scalar implementation
        impl<const N: usize> CryptographicHasher<$t, [$t; N]> for MockHasher {
            fn hash_iter<I: IntoIterator<Item = $t>>(&self, iter: I) -> [$t; N] {
                let sum: $t = iter.into_iter().fold(0 as $t, |a, b| a.wrapping_add(b));
                [sum; N]
            }
        }

        // Parallel implementation for arrays of any size M
        impl<const N: usize, const M: usize> CryptographicHasher<[$t; M], [[$t; M]; N]>
            for MockHasher
        {
            fn hash_iter<I: IntoIterator<Item = [$t; M]>>(&self, iter: I) -> [[$t; M]; N] {
                let sum = iter.into_iter().fold([0 as $t; M], |acc, x| {
                    core::array::from_fn(|i| acc[i].wrapping_add(x[i]))
                });
                [sum; N]
            }
        }
    )*};
}

impl_mock_binary_hasher!(u8, u32, u64);
