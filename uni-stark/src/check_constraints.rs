use p3_air::{
    AirBuilder, AirBuilderWithPublicValues, ExtensionBuilder, PairBuilder, PermutationAirBuilder,
};
use p3_field::{ExtensionField, Field};
use p3_matrix::stack::ViewPair;
#[cfg(debug_assertions)]
use p3_matrix::{Matrix, dense::RowMajorMatrix, dense::RowMajorMatrixView};
#[cfg(debug_assertions)]
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
#[cfg(debug_assertions)]
pub(crate) fn check_constraints<F, EF, A>(
    air: &A,
    main: &RowMajorMatrix<F>,
    aux_trace: &Option<RowMajorMatrix<F>>,
    aux_randomness: &[EF],
    public_values: &alloc::vec::Vec<F>,
) where
    F: Field,
    EF: ExtensionField<F> + p3_field::BasedVectorSpace<F>,
    A: for<'a> p3_air::Air<DebugConstraintBuilder<'a, F, EF>>,
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

        let preprocessed_pair = if let Some(preprocessed_matrix) = preprocessed.as_ref() {
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
            Some(ViewPair::new(
                RowMajorMatrixView::new_row(preprocessed_local),
                RowMajorMatrixView::new_row(preprocessed_next),
            ))
        } else {
            None
        };

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

// Helper: convert a flattened base-field row (slice of `F`) into a Vec<EF>
#[cfg(debug_assertions)]
fn row_to_ext<F, EF>(row: &[F]) -> alloc::vec::Vec<EF>
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

impl<'a, F, EF> AirBuilder for DebugConstraintBuilder<'a, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    type F = F;
    type Expr = F;
    type Var = F;
    type M = ViewPair<'a, F>;

    fn main(&self) -> Self::M {
        self.main
    }

    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row
    }

    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row
    }

    /// # Panics
    /// This function panics if `size` is not `2`.
    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            self.is_transition
        } else {
            panic!("only supports a window size of 2")
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        assert_eq!(
            x.into(),
            F::ZERO,
            "constraints had nonzero value on row {}",
            self.row_index
        );
    }

    fn assert_eq<I1: Into<Self::Expr>, I2: Into<Self::Expr>>(&mut self, x: I1, y: I2) {
        let x = x.into();
        let y = y.into();
        assert_eq!(
            x, y,
            "values didn't match on row {}: {} != {}",
            self.row_index, x, y
        );
    }
}

impl<F: Field, EF: ExtensionField<F>> AirBuilderWithPublicValues
    for DebugConstraintBuilder<'_, F, EF>
{
    type PublicVar = Self::F;

    fn public_values(&self) -> &[Self::F] {
        self.public_values
    }
}

impl<F: Field, EF: ExtensionField<F>> ExtensionBuilder for DebugConstraintBuilder<'_, F, EF> {
    type EF = EF;
    type ExprEF = EF;
    type VarEF = EF;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        let val: EF = x.into();
        for limb in val.as_basis_coefficients_slice() {
            self.assert_zero(*limb);
        }
    }
}

impl<'a, F: Field, EF: ExtensionField<F>> PermutationAirBuilder
    for DebugConstraintBuilder<'a, F, EF>
{
    type MP = ViewPair<'a, EF>;
    type RandomVar = EF;

    fn permutation(&self) -> Self::MP {
        self.aux
    }

    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        self.aux_randomness
    }
}

impl<'a, F: Field, EF: ExtensionField<F>> PairBuilder for DebugConstraintBuilder<'a, F, EF> {
    fn preprocessed(&self) -> Self::M {
        self.preprocessed
            .expect("DebugConstraintBuilder requires preprocessed columns when used as PairBuilder")
    }
}

#[cfg(test)]
#[cfg(debug_assertions)]
mod tests {
    use alloc::vec;

    use p3_air::{BaseAir, BaseAirWithPublicValues};
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

    impl<F: Field> BaseAir<F> for RowLogicAir {
        fn width(&self) -> usize {
            2
        }
    }

    impl<F: Field> BaseAirWithPublicValues<F> for RowLogicAir {}

    impl<F, EF> p3_air::Air<DebugConstraintBuilder<'_, F, EF>> for RowLogicAir
    where
        F: Field,
        EF: ExtensionField<F>,
    {
        fn eval(&self, builder: &mut DebugConstraintBuilder<'_, F, EF>) {
            let main = builder.main();
            let aux_pair = builder.aux;

            for col in 0..W {
                let a = main.top.get(0, col).unwrap();
                let b = main.bottom.get(0, col).unwrap();

                // New logic: enforce row[i+1] = row[i] + 1, only on transitions
                builder.when_transition().assert_eq(b, a + F::ONE);
            }

            // New logic: enforce row[i+1] = row[i] + 1, only on transitions
            builder.when_transition().assert_eq(b, a + F::ONE);

            // ======================
            // aux trace
            // ======================
            // Note: For now this is hard coded with LogUp
            // To show that {x_i} and {y_i} are permutations of each other
            // We compute
            // |    aux1           |    aux2           |   aux3                          |
            // | t_i = 1/(r - x_i) | w_i = 1/(r - y_i) | aux3[i] = aux3[i-1] + t_i - w_i |
            //
            // - r is the input randomness
            // - in practice x_i and y_i should be copied from corresponding main trace (with selectors)
            //
            // ZZ note:
            // This is practically LogUp with univariate. This requires 3 extension columns = 12 base columns.
            // It is better than checking \prod(r-xi) == \prod(r-yi) which requires 4 extension columns (the last two store the running product)

            // aux row computation is correct
            let xi = main.top.get(0, 0).unwrap();
            let yi = main.top.get(0, 1).unwrap();

            // let aux_ef = DebugEfView::<F, EF>::new(aux_pair);
            let r = builder.aux_randomness[0];

            // current row EF elements
            let t_i = aux_pair.get(0, 0).unwrap();
            let w_i = aux_pair.get(0, 1).unwrap();
            let s_i = aux_pair.get(0, 2).unwrap();
            // next row EF elements
            let t_next = aux_pair.get(1, 0).unwrap();
            let w_next = aux_pair.get(1, 1).unwrap();
            let s_next = aux_pair.get(1, 2).unwrap();

            // t * (r - x_i) == 1  and  w * (r - y_i) == 1
            builder.assert_eq_ext(t_i * (r - EF::from(xi)), EF::ONE);
            builder.assert_eq_ext(w_i * (r - EF::from(yi)), EF::ONE);

            // transition is correct: s' = s + t' - w'
            builder
                .when_transition()
                .assert_eq_ext(s_next, s_i + t_next - w_next);

            // a3[last] = Σ(t - w) == 0 if multisets match
            builder.when_last_row().assert_zero_ext(s_i);

            // ======================
            // public input
            // ======================
            // Add public value equality on last row for extra coverage
            let public_values = builder.public_values;
            let mut when_last = builder.when(builder.is_last_row);
            for (i, &pv) in public_values.iter().enumerate().take(W) {
                when_last.assert_eq(main.top.get(0, i).unwrap(), pv);
            }
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
