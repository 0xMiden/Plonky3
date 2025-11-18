use p3_util::log2_strict_usize;

use crate::Matrix;
use crate::row_index_mapped::{RowIndexMap, RowIndexMappedView};

/// Row-index mapping that lifts an inner matrix to a larger height by repeating rows cyclically.
#[derive(Clone, Debug)]
pub struct CyclicLiftIndexMap {
    log_inner_height: usize,
    log_scaling: usize,
    mask: usize,
}

impl CyclicLiftIndexMap {
    pub fn new(inner_height: usize, target_height: usize) -> Self {
        assert!(inner_height > 0, "inner matrix height must be positive");

        let log_inner_height = log2_strict_usize(inner_height);

        assert!(
            target_height >= inner_height,
            "target height must be at least the inner height"
        );
        assert!(
            target_height.is_multiple_of(inner_height),
            "target height must be a multiple of inner height"
        );
        let scaling = target_height / inner_height;
        let log_scaling = log2_strict_usize(scaling);

        let mask = (1 << log_inner_height) - 1;

        Self {
            log_inner_height,
            log_scaling,
            mask,
        }
    }

    /// Construct a lifted view over the given matrix.
    pub fn new_view<T: Clone + Send + Sync, Inner: Matrix<T>>(
        inner: Inner,
        target_height: usize,
    ) -> LiftedMatrixView<Inner> {
        RowIndexMappedView {
            index_map: Self::new(inner.height(), target_height),
            inner,
        }
    }
}

impl RowIndexMap for CyclicLiftIndexMap {
    fn height(&self) -> usize {
        1 << (self.log_inner_height + self.log_scaling)
    }

    fn map_row_index(&self, r: usize) -> usize {
        r & self.mask
    }
}

/// Row-index mapping that lifts an inner matrix by duplicating rows contiguously.
#[derive(Clone, Debug)]
pub struct UpsampledLiftIndexMap {
    log_inner_height: usize,
    log_scaling: usize,
}

impl UpsampledLiftIndexMap {
    pub fn new(inner_height: usize, target_height: usize) -> Self {
        assert!(inner_height > 0, "inner matrix height must be positive");

        let log_inner_height = log2_strict_usize(inner_height);

        assert!(
            target_height >= inner_height,
            "target height must be at least the inner height"
        );
        assert!(
            target_height.is_multiple_of(inner_height),
            "target height must be a multiple of inner height"
        );
        let scaling = target_height / inner_height;
        let log_scaling = log2_strict_usize(scaling);

        Self {
            log_inner_height,
            log_scaling,
        }
    }

    pub fn new_view<T: Clone + Send + Sync, Inner: Matrix<T>>(
        inner: Inner,
        target_height: usize,
    ) -> UpsampledLiftedMatrixView<Inner> {
        RowIndexMappedView {
            index_map: Self::new(inner.height(), target_height),
            inner,
        }
    }
}

impl RowIndexMap for UpsampledLiftIndexMap {
    fn height(&self) -> usize {
        1 << (self.log_inner_height + self.log_scaling)
    }

    fn map_row_index(&self, r: usize) -> usize {
        r >> self.log_scaling
    }
}

/// A matrix view that repeats rows of its inner matrix up to a target height.
///
/// Rows are selected cyclically; the `r`-th visible row corresponds to row `r % inner_height`
/// of the underlying matrix. The inner height and scaling factor must both be powers of two.
pub type LiftedMatrixView<Inner> = RowIndexMappedView<CyclicLiftIndexMap, Inner>;
/// A matrix view that duplicates rows contiguously to reach a target height.
pub type UpsampledLiftedMatrixView<Inner> = RowIndexMappedView<UpsampledLiftIndexMap, Inner>;

/// Extension trait for matrices that can create lifted views.
pub trait LiftableMatrix<T: Clone + Send + Sync>: Matrix<T> + Sized {
    /// Returns a view that repeats the matrix rows until reaching `target_height`.
    ///
    /// # Panics
    /// Panics if the matrix height is zero, if the matrix height or scaling factor is not a power
    /// of two, or if `target_height` is not a multiple of the matrix height.
    fn lift_cyclic(self, target_height: usize) -> LiftedMatrixView<Self>;

    /// Returns a view that up-samples the matrix rows until reaching `target_height`.
    ///
    /// Rows are duplicated in contiguous blocks rather than wrapping.
    ///
    /// # Panics
    /// Same conditions as [`LiftableMatrix::lift_cyclic`].
    fn lift_upsampled(self, target_height: usize) -> UpsampledLiftedMatrixView<Self>;
}

impl<T, M> LiftableMatrix<T> for M
where
    T: Clone + Send + Sync,
    M: Matrix<T>,
{
    fn lift_cyclic(self, target_height: usize) -> LiftedMatrixView<Self> {
        CyclicLiftIndexMap::new_view(self, target_height)
    }

    fn lift_upsampled(self, target_height: usize) -> UpsampledLiftedMatrixView<Self> {
        UpsampledLiftIndexMap::new_view(self, target_height)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;
    use crate::dense::RowMajorMatrix;

    #[test]
    fn cyclic_lift_index_map_height_and_scaling() {
        let map = CyclicLiftIndexMap::new(4, 16);
        assert_eq!(map.height(), 16);
        assert_eq!(map.map_row_index(0), 0);
        assert_eq!(map.map_row_index(5), 1);
        assert_eq!(map.map_row_index(14), 2);
    }

    #[test]
    fn upsampled_lift_index_map_height_and_mapping() {
        let map = UpsampledLiftIndexMap::new(4, 16);
        assert_eq!(map.height(), 16);
        let expected = [0, 0, 0, 0, 1, 1, 1, 1];
        for (idx, expected_row) in expected.iter().enumerate() {
            assert_eq!(map.map_row_index(idx), *expected_row);
        }
    }

    #[test]
    fn lift_cyclic_view_repeats_rows() {
        let matrix = RowMajorMatrix::new((0u32..8).collect::<Vec<_>>(), 2);
        let view = matrix.clone().lift_cyclic(16);
        assert_eq!(view.height(), 16);
        assert_eq!(view.width(), matrix.width());

        for row in 0..view.height() {
            let expected_row: Vec<_> = matrix
                .row(row % matrix.height())
                .unwrap()
                .into_iter()
                .collect();
            let actual_row: Vec<_> = view.row(row).unwrap().into_iter().collect();
            assert_eq!(actual_row, expected_row, "row {}", row);
        }
    }

    #[test]
    fn lift_cyclic_view_from_ref() {
        let matrix = RowMajorMatrix::new((0u32..8).collect::<Vec<_>>(), 2);

        // Use the extension trait on a reference
        let view_ref = (&matrix).lift_cyclic(16);
        assert_eq!(view_ref.height(), 16);
        assert_eq!(view_ref.width(), matrix.width());

        for row in 0..view_ref.height() {
            let expected_row: Vec<_> = matrix
                .row(row % matrix.height())
                .unwrap()
                .into_iter()
                .collect();
            let actual_row: Vec<_> = view_ref.row(row).unwrap().into_iter().collect();
            assert_eq!(actual_row, expected_row, "row {}", row);
        }

        // Use the explicit constructor directly with a reference
        let explicit = super::CyclicLiftIndexMap::new_view(&matrix, 16);
        assert_eq!(explicit.height(), 16);
        assert_eq!(explicit.width(), matrix.width());
    }

    #[test]
    fn lift_upsampled_view_duplicates_rows() {
        let matrix = RowMajorMatrix::new((0u32..8).collect::<Vec<_>>(), 2);
        let view = matrix.clone().lift_upsampled(16);
        let scaling = 16 / matrix.height();

        assert_eq!(view.height(), 16);
        assert_eq!(view.width(), matrix.width());

        for row in 0..view.height() {
            let base_row = row / scaling;
            let expected_row: Vec<_> = matrix.row(base_row).unwrap().into_iter().collect();
            let actual_row: Vec<_> = view.row(row).unwrap().into_iter().collect();
            assert_eq!(actual_row, expected_row, "row {}", row);
        }
    }

    #[test]
    fn lift_upsampled_view_from_ref() {
        let matrix = RowMajorMatrix::new((0u32..8).collect::<Vec<_>>(), 2);
        let scaling = 16 / matrix.height();

        // Use the extension trait on a reference
        let view_ref = (&matrix).lift_upsampled(16);
        assert_eq!(view_ref.height(), 16);
        assert_eq!(view_ref.width(), matrix.width());
        for row in 0..view_ref.height() {
            let base_row = row / scaling;
            let expected_row: Vec<_> = matrix.row(base_row).unwrap().into_iter().collect();
            let actual_row: Vec<_> = view_ref.row(row).unwrap().into_iter().collect();
            assert_eq!(actual_row, expected_row, "row {}", row);
        }

        // Use the explicit constructor directly with a reference
        let explicit = super::UpsampledLiftIndexMap::new_view(&matrix, 16);
        assert_eq!(explicit.height(), 16);
        assert_eq!(explicit.width(), matrix.width());
    }

    #[test]
    fn to_row_major_matrix_cyclic() {
        let matrix = RowMajorMatrix::new((0i32..8).collect::<Vec<_>>(), 2);
        let map = CyclicLiftIndexMap::new(matrix.height(), 8);
        let lifted = map.to_row_major_matrix(matrix.clone());

        let mut expected = Vec::new();
        for row in 0..8 {
            expected.extend(matrix.row(row % matrix.height()).unwrap().into_iter());
        }
        assert_eq!(lifted.values, expected);
    }

    #[test]
    fn to_row_major_matrix_upsampled() {
        let matrix = RowMajorMatrix::new((0i32..8).collect::<Vec<_>>(), 2);
        let target_height = 8;
        let map = UpsampledLiftIndexMap::new(matrix.height(), target_height);
        let lifted = map.to_row_major_matrix(matrix.clone());

        let scaling = target_height / matrix.height();
        let mut expected = Vec::new();
        for row in 0..target_height {
            let base_row = row / scaling;
            expected.extend(matrix.row(base_row).unwrap().into_iter());
        }
        assert_eq!(lifted.values, expected);
    }

    #[test]
    fn liftable_trait_round_trip_cyclic() {
        let matrix = RowMajorMatrix::new((0usize..16).collect::<Vec<_>>(), 4);
        let lifted = matrix.clone().lift_cyclic(16);
        assert_eq!(lifted.height(), 16);
        assert_eq!(lifted.width(), matrix.width());

        let collected: Vec<_> = lifted.rows().map(|row| row.collect::<Vec<_>>()).collect();
        let expected: Vec<_> = (0..16)
            .map(|row| {
                matrix
                    .row(row % matrix.height())
                    .unwrap()
                    .into_iter()
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(collected, expected);
    }

    #[test]
    fn liftable_trait_round_trip_upsampled() {
        let matrix = RowMajorMatrix::new((0usize..8).collect::<Vec<_>>(), 2);
        let lifted = matrix.clone().lift_upsampled(8);
        assert_eq!(lifted.height(), 8);
        assert_eq!(lifted.width(), matrix.width());

        let collected: Vec<_> = lifted.rows().map(|row| row.collect::<Vec<_>>()).collect();
        let scaling = 8 / matrix.height();
        let expected: Vec<_> = (0..8)
            .map(|row| {
                matrix
                    .row(row / scaling)
                    .unwrap()
                    .into_iter()
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(collected, expected);
    }
}
