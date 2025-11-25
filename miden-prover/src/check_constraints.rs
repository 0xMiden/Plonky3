use miden_air::{MidenAir, MidenAirBuilder};
// use p3_air::{
//     AirBuilder, AirBuilderWithPublicValues, ExtensionBuilder, PairBuilder, PermutationAirBuilder,
// };
use p3_field::{BasedVectorSpace, ExtensionField, Field};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::stack::ViewPair;
use tracing::instrument;

/// Runs constraint checks using a given AIR definition and trace matrix.
///
/// Iterates over every row in `main`, providing both the current and next row
/// (with wraparound) to the AIR logic. Also injects public values into the builder
/// for first/last row assertions.
///
/// # Arguments
/// - `air`: The AIR logic to run
/// - `main`: The trace matrix (rows of witness values)
/// - `aux`: The aux trace matrix (if 2 phase proving)
/// - `aux_randomness`: The randomness values that are used to generate `aux` trace
/// - `public_values`: Public values provided to the builder
#[instrument(name = "check constraints", skip_all)]
pub(crate) fn check_constraints<F, EF, A>(
    air: &A,
    main: &RowMajorMatrix<F>,
    aux_trace: &Option<RowMajorMatrix<F>>,
    aux_randomness: &[EF],
    public_values: &Vec<F>,
) where
    F: Field,
    EF: ExtensionField<F> + BasedVectorSpace<F>,
    A: MidenAir<F, EF>,
{
    let height = main.height();
    let preprocessed = air.preprocessed_trace();

    (0..height).for_each(|row_index| {
        let row_index_next = (row_index + 1) % height;

        // row_index < height so we can used unchecked indexing.
        let local = unsafe { main.row_slice_unchecked(row_index) };
        // row_index_next < height so we can used unchecked indexing.
        let next = unsafe { main.row_slice_unchecked(row_index_next) };
        let main = ViewPair::new(
            RowMajorMatrixView::new_row(&*local),
            RowMajorMatrixView::new_row(&*next),
        );

        // Keep these Vecs in the outer scope so their backing memory lives
        // long enough for the `RowMajorMatrixView` references stored in `aux`.
        let aux_local_ext;
        let aux_next_ext;

        #[allow(clippy::option_if_let_else)]
        let aux = if let Some(aux_matrix) = aux_trace.as_ref() {
            let aux_local = unsafe { aux_matrix.row_slice_unchecked(row_index) };
            aux_local_ext = row_to_ext::<F, EF>(&aux_local);

            let aux_next = unsafe { aux_matrix.row_slice_unchecked(row_index_next) };
            aux_next_ext = row_to_ext::<F, EF>(&aux_next);

            p3_matrix::stack::VerticalPair::new(
                RowMajorMatrixView::new_row(&aux_local_ext),
                RowMajorMatrixView::new_row(&aux_next_ext),
            )
        } else {
            // Create an empty ViewPair with zero width
            let empty: &[EF] = &[];
            p3_matrix::stack::VerticalPair::new(
                RowMajorMatrixView::new_row(empty),
                RowMajorMatrixView::new_row(empty),
            )
        };

        let preprocessed_pair = preprocessed.as_ref().map(|preprocessed_matrix| {
            let preprocessed_local = preprocessed_matrix
                .values
                .chunks(preprocessed_matrix.width)
                .nth(row_index)
                .unwrap();
            let preprocessed_next = preprocessed_matrix
                .values
                .chunks(preprocessed_matrix.width)
                .nth(row_index_next)
                .unwrap();
            ViewPair::new(
                RowMajorMatrixView::new_row(preprocessed_local),
                RowMajorMatrixView::new_row(preprocessed_next),
            )
        });

        let mut builder = DebugConstraintBuilder {
            row_index,
            main,
            aux,
            aux_randomness,
            preprocessed: preprocessed_pair,
            public_values,
            is_first_row: F::from_bool(row_index == 0),
            is_last_row: F::from_bool(row_index == height - 1),
            is_transition: F::from_bool(row_index != height - 1),
        };

        air.eval(&mut builder);
    });
}

/// Helper: convert a flattened base-field row (slice of `F`) into a Vec<EF>
fn row_to_ext<F, EF>(row: &[F]) -> Vec<EF>
where
    F: Field,
    EF: ExtensionField<F> + p3_field::BasedVectorSpace<F>,
{
    row.chunks(EF::DIMENSION)
        .map(|chunk| EF::from_basis_coefficients_slice(chunk).unwrap())
        .collect()
}

/// A builder that runs constraint assertions during testing.
///
/// Used in conjunction with [`check_constraints`] to simulate
/// an execution trace and verify that the AIR logic enforces all constraints.
#[derive(Debug)]
pub struct DebugConstraintBuilder<'a, F: Field, EF: ExtensionField<F>> {
    /// The index of the row currently being evaluated.
    row_index: usize,
    /// A view of the current and next row as a vertical pair.
    main: ViewPair<'a, F>,
    /// A view of the current and next aux row as a vertical pair.
    aux: ViewPair<'a, EF>,
    /// randomness that is used to compute aux trace
    aux_randomness: &'a [EF],
    /// A view of the preprocessed current and next row as a vertical pair (if present).
    preprocessed: Option<ViewPair<'a, F>>,
    /// The public values provided for constraint validation (e.g. inputs or outputs).
    public_values: &'a [F],
    /// A flag indicating whether this is the first row.
    is_first_row: F,
    /// A flag indicating whether this is the last row.
    is_last_row: F,
    /// A flag indicating whether this is a transition row (not the last row).
    is_transition: F,
}

impl<'a, F, EF> MidenAirBuilder for DebugConstraintBuilder<'a, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    type F = F;
    type Expr = F;
    type Var = F;
    type M = ViewPair<'a, F>;
    type PublicVar = F;
    type EF = EF;
    type ExprEF = EF;
    type VarEF = EF;
    type MP = ViewPair<'a, EF>;
    type RandomVar = EF;

    fn main(&self) -> Self::M {
        self.main
    }

    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row
    }

    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row
    }

    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            self.is_transition
        } else {
            panic!("DebugConstraintBuilder only supports transition window of size 2");
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let value = x.into();
        assert!(
            value == F::ZERO,
            "Constraint failed at row {}: expected zero, got {:?}",
            self.row_index,
            value
        );
    }

    fn public_values(&self) -> &[Self::PublicVar] {
        self.public_values
    }

    fn preprocessed(&self) -> Self::M {
        self.preprocessed.unwrap_or_else(|| {
            // Return an empty ViewPair if there are no preprocessed columns
            let empty: &[F] = &[];
            ViewPair::new(
                RowMajorMatrixView::new_row(empty),
                RowMajorMatrixView::new_row(empty),
            )
        })
    }

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        let value = x.into();
        assert!(
            value == EF::ZERO,
            "Extension field constraint failed at row {}: expected zero, got {:?}",
            self.row_index,
            value
        );
    }

    fn permutation(&self) -> Self::MP {
        self.aux
    }

    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        self.aux_randomness
    }
}

#[cfg(test)]
mod tests {

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;

    use super::*;

    /// A test AIR that enforces a simple linear transition logic:
    /// - Each cell in the next row must equal the current cell plus 1 (i.e., `next = current + 1`)
    /// - On the last row, the current row must match the provided public values.
    ///
    /// This is useful for validating constraint evaluation, transition logic,
    /// and row condition flags (first/last/transition).
    #[derive(Debug)]
    struct RowLogicAir;

    impl<F, EF> MidenAir<F, EF> for RowLogicAir
    where
        F: Field,
        EF: ExtensionField<F>,
    {
        fn width(&self) -> usize {
            2
        }

        fn eval<AB: MidenAirBuilder<F = F>>(&self, builder: &mut AB) {
            let main = builder.main();

            for col in 0..2 {
                let a = main.get(0, col).unwrap();
                let b = main.get(1, col).unwrap();

                // New logic: enforce row[i+1] = row[i] + 1, only on transitions
                builder.when_transition().assert_eq(b, a + F::ONE);
            }

            // Add public value equality on last row for extra coverage
            let public_values = builder.public_values();
            let pv0 = public_values[0];
            let pv1 = public_values[1];

            let mut when_last = builder.when_last_row();
            when_last.assert_eq(main.get(0, 0).unwrap(), pv0);
            when_last.assert_eq(main.get(0, 1).unwrap(), pv1);
        }
    }

    #[test]
    fn test_incremental_rows_with_last_row_check() {
        // Each row = previous + 1, with 4 rows total, 2 columns.
        // Last row must match public values [4, 4]
        let air = RowLogicAir;
        let values = vec![
            BabyBear::ONE,
            BabyBear::ONE, // Row 0
            BabyBear::new(2),
            BabyBear::new(2), // Row 1
            BabyBear::new(3),
            BabyBear::new(3), // Row 2
            BabyBear::new(4),
            BabyBear::new(4), // Row 3 (last)
        ];
        let main = RowMajorMatrix::new(values, 2);
        check_constraints::<_, BinomialExtensionField<BabyBear, 4>, _>(
            &air,
            &main,
            &None,
            &[],
            &vec![BabyBear::new(4); 2],
        );
    }

    #[test]
    #[should_panic]
    fn test_incorrect_increment_logic() {
        // Row 2 does not equal row 1 + 1 → should fail on transition from row 1 to 2.
        let air = RowLogicAir;
        let values = vec![
            BabyBear::ONE,
            BabyBear::ONE, // Row 0
            BabyBear::new(2),
            BabyBear::new(2), // Row 1
            BabyBear::new(5),
            BabyBear::new(5), // Row 2 (wrong)
            BabyBear::new(6),
            BabyBear::new(6), // Row 3
        ];
        let main = RowMajorMatrix::new(values, 2);
        check_constraints::<_, BinomialExtensionField<BabyBear, 4>, _>(
            &air,
            &main,
            &None,
            &[],
            &vec![BabyBear::new(6); 2],
        );
    }

    #[test]
    #[should_panic]
    fn test_wrong_last_row_public_value() {
        // The transition logic is fine, but public value check fails at the last row.
        let air = RowLogicAir;
        let values = vec![
            BabyBear::ONE,
            BabyBear::ONE, // Row 0
            BabyBear::new(2),
            BabyBear::new(2), // Row 1
            BabyBear::new(3),
            BabyBear::new(3), // Row 2
            BabyBear::new(4),
            BabyBear::new(4), // Row 3
        ];
        let main = RowMajorMatrix::new(values, 2);
        // Wrong public value on column 1
        check_constraints::<_, BinomialExtensionField<BabyBear, 4>, _>(
            &air,
            &main,
            &None,
            &[],
            &vec![BabyBear::new(4), BabyBear::new(5)],
        );
    }

    #[test]
    fn test_single_row_wraparound_logic() {
        // A single-row matrix still performs a wraparound check with itself.
        // row[0] == row[0] + 1 ⇒ fails unless handled properly by transition logic.
        // Here: is_transition == false ⇒ so no assertions are enforced.
        let air = RowLogicAir;
        let values = vec![
            BabyBear::new(99),
            BabyBear::new(77), // Row 0
        ];
        let main = RowMajorMatrix::new(values, 2);
        check_constraints::<_, BinomialExtensionField<BabyBear, 4>, _>(
            &air,
            &main,
            &None,
            &[],
            &vec![BabyBear::new(99), BabyBear::new(77)],
        );
    }
}
