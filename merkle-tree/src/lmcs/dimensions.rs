use alloc::vec::Vec;

use p3_matrix::Dimensions;

use super::LmcsError;

/// Dimensions helper for lifting-related computations over a batch of matrices.
#[derive(Clone, Debug)]
pub struct LiftDimensions {
    dims: Vec<Dimensions>,
}

impl LiftDimensions {
    /// Construct and validate lifting dimensions.
    ///
    /// Returns an `LmcsError` when:
    /// - The list is empty (`EmptyBatch`)
    /// - Any height is zero (`ZeroHeightMatrix`)
    /// - Any height is not a power of two (`NonPowerOfTwoHeight`)
    /// - The final height is not a power of two (`FinalHeightNotPowerOfTwo`)
    /// - A height does not divide the final height (`HeightNotDivisor`)
    pub fn new(dims: Vec<Dimensions>) -> Result<Self, LmcsError> {
        if dims.is_empty() {
            return Err(LmcsError::EmptyBatch);
        }

        // Check non-decreasing heights order.
        let heights_sorted = dims
            .iter()
            .zip(dims.iter().skip(1))
            .all(|(a, b)| a.height <= b.height);
        if !heights_sorted {
            return Err(LmcsError::UnsortedByHeight);
        }

        // Compute the tallest height (input need not be pre-sorted before this step).
        let mut max_height = 0usize;
        for (i, d) in dims.iter().copied().enumerate() {
            let h = d.height;
            if h == 0 {
                return Err(LmcsError::ZeroHeightMatrix { matrix: i });
            }
            if !h.is_power_of_two() {
                return Err(LmcsError::NonPowerOfTwoHeight {
                    matrix: i,
                    height: h,
                });
            }
            max_height = max_height.max(h);
        }

        if !max_height.is_power_of_two() {
            return Err(LmcsError::FinalHeightNotPowerOfTwo { height: max_height });
        }

        for (i, d) in dims.iter().copied().enumerate() {
            let h = d.height;
            if !max_height.is_multiple_of(h) {
                return Err(LmcsError::HeightNotDivisor {
                    matrix: i,
                    height: h,
                    final_height: max_height,
                });
            }
        }

        Ok(Self { dims })
    }

    #[inline]
    pub fn smallest_height(&self) -> usize {
        self.dims.first().unwrap().height
    }

    #[inline]
    pub fn largest_height(&self) -> usize {
        self.dims.last().unwrap().height
    }

    #[inline]
    pub fn dimensions(&self) -> &[Dimensions] {
        &self.dims
    }

    /// Iterate over all matrix heights.
    pub fn heights(&self) -> impl Iterator<Item = usize> + '_ {
        self.dims.iter().map(|d| d.height)
    }

    /// Map a global row index to the corresponding row index for matrix `matrix_idx`
    /// under upsampled lifting semantics.
    #[inline]
    pub fn map_idx_upsampled_for(&self, matrix_idx: usize, index: usize) -> usize {
        let max_h = self.largest_height();
        let h = self.dims[matrix_idx].height;
        let scaling = max_h / h;
        let log_s = p3_util::log2_strict_usize(scaling);
        index >> log_s
    }

    /// Map a global row index to every matrix's local row index under upsampled lifting.
    pub fn map_idxs_upsampled(&self, index: usize) -> impl Iterator<Item = usize> + '_ {
        let max_h = self.largest_height();
        self.dims.iter().map(move |d| {
            let scaling = max_h / d.height;
            let log_s = p3_util::log2_strict_usize(scaling);
            index >> log_s
        })
    }

    /// Iterator of widths padded up to the given RATE.
    pub fn padded_widths<const RATE: usize>(&self) -> impl Iterator<Item = usize> + '_ {
        self.dims.iter().map(|d| d.width.next_multiple_of(RATE))
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;

    #[test]
    fn rejects_empty_batch() {
        assert!(matches!(
            LiftDimensions::new(vec![]),
            Err(LmcsError::EmptyBatch)
        ));
    }

    #[test]
    fn rejects_unsorted_heights() {
        let dims = vec![
            Dimensions {
                width: 1,
                height: 4,
            },
            Dimensions {
                width: 1,
                height: 2,
            },
        ];
        assert!(matches!(
            LiftDimensions::new(dims),
            Err(LmcsError::UnsortedByHeight)
        ));
    }

    #[test]
    fn rejects_zero_height_and_non_power_of_two() {
        let dims_zero = vec![Dimensions {
            width: 1,
            height: 0,
        }];
        assert!(matches!(
            LiftDimensions::new(dims_zero),
            Err(LmcsError::ZeroHeightMatrix { matrix: 0 })
        ));

        let dims_np2 = vec![Dimensions {
            width: 1,
            height: 3,
        }];
        assert!(matches!(
            LiftDimensions::new(dims_np2),
            Err(LmcsError::NonPowerOfTwoHeight {
                matrix: 0,
                height: 3
            })
        ));
    }

    #[test]
    fn rejects_height_not_divisor() {
        // With current invariants (all heights power-of-two), HeightNotDivisor cannot occur.
        // Keep a sanity check of the positive path computations instead.
        let dims_ok = vec![
            Dimensions {
                width: 5,
                height: 2,
            },
            Dimensions {
                width: 7,
                height: 4,
            },
            Dimensions {
                width: 9,
                height: 8,
            },
        ];
        let lift = LiftDimensions::new(dims_ok).unwrap();
        assert_eq!(lift.smallest_height(), 2);
        assert_eq!(lift.largest_height(), 8);
        let idx_map: Vec<_> = lift.map_idxs_upsampled(7).collect();
        // scaling factors: 4, 2, 1 → indices: 7>>2=1, 7>>1=3, 7>>0=7
        assert_eq!(idx_map, vec![1, 3, 7]);
        let padded: Vec<_> = lift.padded_widths::<8>().collect();
        assert_eq!(padded, vec![8, 8, 16]);
    }
}
