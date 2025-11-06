use alloc::vec::Vec;

use p3_air::{AirBuilder, AirBuilderWithPublicValues, ExtensionBuilder, PermutationAirBuilder};
use p3_field::{BasedVectorSpace, ExtensionField, Field};
use p3_matrix::stack::ViewPair;

#[cfg(debug_assertions)]
use p3_air::Air;
#[cfg(debug_assertions)]
use p3_matrix::Matrix;
#[cfg(debug_assertions)]
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
#[cfg(debug_assertions)]
use p3_matrix::stack::VerticalPair;
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
    public_values: &Vec<F>,
) where
    F: Field,
    EF: ExtensionField<F> + BasedVectorSpace<F>,
    A: for<'a> Air<DebugConstraintBuilder<'a, F, EF>>,
{
    let height = main.height();

    (0..height).for_each(|row_index| {
        let row_index_next = (row_index + 1) % height;

        // row_index < height so we can used unchecked indexing.
        let local = unsafe { main.row_slice_unchecked(row_index) };
        // row_index_next < height so we can used unchecked indexing.
        let next = unsafe { main.row_slice_unchecked(row_index_next) };
        let main = VerticalPair::new(
            RowMajorMatrixView::new_row(&*local),
            RowMajorMatrixView::new_row(&*next),
        );

        // Keep these Vecs in the outer scope so their backing memory lives
        // long enough for the `RowMajorMatrixView` references stored in `aux`.
        let (aux_local_ext, aux_next_ext);

        let aux = if let Some(aux_matrix) = aux_trace.as_ref() {
            let aux_local = unsafe { aux_matrix.row_slice_unchecked(row_index) };
            aux_local_ext = row_to_ext::<F, EF>(&*aux_local);

            let aux_next = unsafe { aux_matrix.row_slice_unchecked(row_index_next) };
            aux_next_ext = row_to_ext::<F, EF>(&*aux_next);

            Some(VerticalPair::new(
                RowMajorMatrixView::new_row(aux_local_ext.as_ref()),
                RowMajorMatrixView::new_row(aux_next_ext.as_ref()),
            ))
        } else {
            None
        };

        let mut builder = DebugConstraintBuilder {
            row_index,
            main,
            aux,
            aux_randomness,
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
fn row_to_ext<F, EF>(row: &[F]) -> Vec<EF>
where
    F: Field,
    EF: ExtensionField<F> + BasedVectorSpace<F>,
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
    aux: Option<ViewPair<'a, EF>>,
    /// randomness that is used to compute aux trace
    aux_randomness: &'a [EF],
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
        self.aux.expect("permutation called but aux trace is None - AIR should check num_randomness > 0 before using permutation columns")
    }

    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        self.aux_randomness
    }
}

#[cfg(test)]
#[cfg(debug_assertions)]
mod tests {
    use alloc::vec;

    use p3_air::{BaseAir, BaseAirWithPublicValues};
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{BasedVectorSpace, ExtensionField};

    use super::*;

    /// A test AIR that enforces a simple linear transition logic:
    /// - Each cell in the next row must equal the current cell plus 1 (i.e., `next = current + 1`)
    /// - On the last row, the current row must match the provided public values.
    ///
    /// This is useful for validating constraint evaluation, transition logic,
    /// and row condition flags (first/last/transition).
    #[derive(Debug)]
    struct RowLogicAir<const W: usize>;

    impl<F: Field, const W: usize> BaseAir<F> for RowLogicAir<W> {
        fn width(&self) -> usize {
            W
        }
    }

    impl<F: Field, const W: usize> BaseAirWithPublicValues<F> for RowLogicAir<W> {}

    impl<F, EF, const W: usize> Air<DebugConstraintBuilder<'_, F, EF>> for RowLogicAir<W>
    where
        F: Field,
        EF: ExtensionField<F>,
    {
        fn eval(&self, builder: &mut DebugConstraintBuilder<'_, F, EF>) {
            let main = builder.main();
            let aux = builder.aux;

            // ======================
            // main trace
            // ======================
            // | main1             | main2            |
            // | row[i]            | perm(main1)[i]   |
            // | row[i+1]=row[i]+1 | perm(main1)[i+1] |

            let a = main.top.get(0, 0).unwrap();
            let b = main.bottom.get(0, 0).unwrap();

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
            // - Potentially even less: the extension fields for aux1 and aux2 are identical. So we should be able to save another 3 base columns.
            // - It is better than checking \prod(r-xi) == \prod(r-yi) which requires 4 extension columns (the last two store the running product)

            // aux row computation is correct
            let xi = main.top.get(0, 0).unwrap();
            let yi = main.top.get(0, 1).unwrap();

            let aux_pair = aux.expect("test expects aux trace");
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

    // Generate a main trace.
    // The first column is incremental
    // The second column is the rev of the first column
    fn gen_main(main_col: &Vec<BabyBear>) -> RowMajorMatrix<BabyBear> {
        let main_rev = permute(main_col);
        let main_values = main_col
            .iter()
            .zip(main_rev.iter())
            .flat_map(|(a, b)| vec![a, b])
            .cloned()
            .collect();
        RowMajorMatrix::new(main_values, 2)
    }

    // Generate the aux trace for logup arguments (flattened for storage).
    fn gen_aux(
        main_col: &Vec<BabyBear>,
        aux_randomness: &BinomialExtensionField<BabyBear, 4>,
    ) -> RowMajorMatrix<BabyBear> {
        use p3_matrix::dense::DenseMatrix;
        // Build a DenseMatrix main trace with width 2
        let main_rev = permute(main_col);
        let main_values = main_col
            .iter()
            .zip(main_rev.iter())
            .flat_map(|(a, b)| vec![*a, *b])
            .collect();
        let main = DenseMatrix::new(main_values, 2);
        // Use the library generator and return the flattened aux
        super::super::generate_logup_trace::<BinomialExtensionField<BabyBear, 4>, _>(
            &main,
            aux_randomness,
        )
    }

    #[test]
    fn test_permuted_incremental_rows_with_last_row_check() {
        let len = 100;

        // Each row = previous + 1, with 4 rows total, 2 columns.
        // Last row must match public values [4, 1]
        // randomness = 5 + 10x + 15x^2 + 20x^3
        // | m1 | m2 | a1      | a2      | a3 |
        // | 1  | 4  | 1/(r-1) | 1/(r-4) | .. |
        // | 2  | 3  | 1/(r-2) | 1/(r-3) | .. |
        // | 3  | 2  | 1/(r-3) | 1/(r-2) | .. |
        // | 4  | 1  | 1/(r-4) | 1/(r-1) | .. |
        let air = RowLogicAir::<2>;

        let main_col = (1..=len).map(|i| BabyBear::new(i)).collect();
        let main = gen_main(&main_col);

        let aux_randomness = BinomialExtensionField::<BabyBear, 4>::from_basis_coefficients_slice(
            [
                BabyBear::new(1005),
                BabyBear::new(10010),
                BabyBear::new(10015),
                BabyBear::new(10020),
            ]
            .as_ref(),
        )
        .unwrap();

        let aux = gen_aux(&main_col, &aux_randomness);

        check_constraints::<BabyBear, BinomialExtensionField<BabyBear, 4>, _>(
            &air,
            &main,
            &Some(aux),
            &aux_randomness.as_basis_coefficients_slice(),
            &vec![BabyBear::new(len), BabyBear::new(1)],
        );
    }

    #[test]
    #[should_panic]
    fn test_incorrect_increment_logic() {
        let len = 100;

        // Each row = previous + 1, with 4 rows total, 2 columns.
        // Last row must match public values [4, 1]
        // randomness = 5 + 10x + 15x^2 + 20x^3
        // | m1 | m2 | a1      | a2      | a3 |
        // | 1  | 4  | 1/(r-1) | 1/(r-4) | .. |
        // | 2  | 0  | 1/(r-2) | 1/(r-3) | .. |
        // | 0  | 2  | 1/(r-3) | 1/(r-2) | .. | <- wrong value
        // | 4  | 1  | 1/(r-4) | 1/(r-1) | .. |
        let air = RowLogicAir::<2>;

        let mut main_col: Vec<BabyBear> = (1..=len).map(|i| BabyBear::new(i)).collect();
        main_col[2] = BabyBear::new(0);
        let main = gen_main(&main_col);

        let aux_randomness = BinomialExtensionField::<BabyBear, 4>::from_basis_coefficients_slice(
            [
                BabyBear::new(5),
                BabyBear::new(10),
                BabyBear::new(15),
                BabyBear::new(20),
            ]
            .as_ref(),
        )
        .unwrap();
        let aux = gen_aux(&main_col, &aux_randomness);
        let aux_randomness_bases = aux_randomness.as_basis_coefficients_slice();
        check_constraints::<BabyBear, BinomialExtensionField<BabyBear, 4>, _>(
            &air,
            &main,
            &Some(aux),
            &aux_randomness_bases,
            &vec![BabyBear::new(len), BabyBear::new(1)],
        );
    }

    #[test]
    #[should_panic]
    fn test_wrong_last_row_public_value() {
        let len = 100;

        // Each row = previous + 1, with 4 rows total, 2 columns.
        // Last row must match public values [4, 1]
        // randomness = 5 + 10x + 15x^2 + 20x^3
        // | m1 | m2 | a1      | a2      | a3 |
        // | 1  | 4  | 1/(r-1) | 1/(r-4) | .. |
        // | 2  | 3  | 1/(r-2) | 1/(r-3) | .. |
        // | 3  | 2  | 1/(r-3) | 1/(r-2) | .. |
        // | 4  | 1  | 1/(r-4) | 1/(r-1) | .. | <- wrong value
        let air = RowLogicAir::<2>;

        let main_col = (1..=len).map(|i| BabyBear::new(i)).collect();
        let main = gen_main(&main_col);

        let aux_randomness = BinomialExtensionField::<BabyBear, 4>::from_basis_coefficients_slice(
            [
                BabyBear::new(5),
                BabyBear::new(10),
                BabyBear::new(15),
                BabyBear::new(20),
            ]
            .as_ref(),
        )
        .unwrap();
        let aux = gen_aux(&main_col, &aux_randomness);
        let aux_randomness_bases = aux_randomness.as_basis_coefficients_slice();
        check_constraints::<BabyBear, BinomialExtensionField<BabyBear, 4>, _>(
            &air,
            &main,
            &Some(aux),
            aux_randomness_bases,
            &vec![BabyBear::new(len), BabyBear::new(len)],
        );
    }

    #[test]
    #[should_panic]
    fn test_wrong_permutation_value() {
        let len = 100;

        // Each row = previous + 1, with 4 rows total, 2 columns.
        // Last row must match public values [4, 1]
        // randomness = 5 + 10x + 15x^2 + 20x^3
        // | m1 | m2 | a1      | a2      | a3 |
        // | 1  | 4  | 0       | 1/(r-4) | .. |  <- wrong value
        // | 2  | 3  | 1/(r-2) | 1/(r-3) | .. |
        // | 3  | 2  | 1/(r-3) | 1/(r-2) | .. |
        // | 4  | 1  | 1/(r-4) | 1/(r-1) | .. |
        let air = RowLogicAir::<2>;

        let main_col = (1..=len).map(|i| BabyBear::new(i)).collect();
        let main = gen_main(&main_col);

        let aux_randomness = BinomialExtensionField::<BabyBear, 4>::from_basis_coefficients_slice(
            [
                BabyBear::new(5),
                BabyBear::new(10),
                BabyBear::new(15),
                BabyBear::new(20),
            ]
            .as_ref(),
        )
        .unwrap();
        let mut aux = gen_aux(&main_col, &aux_randomness);
        aux.values[0] = BabyBear::new(0).into();
        let aux_randomness_bases = aux_randomness.as_basis_coefficients_slice();

        check_constraints::<BabyBear, BinomialExtensionField<BabyBear, 4>, _>(
            &air,
            &main,
            &Some(aux),
            aux_randomness_bases,
            &vec![BabyBear::new(len), BabyBear::new(len)],
        );
    }

    #[test]
    fn test_single_row_wraparound_logic() {
        // A single-row matrix still performs a wraparound check with itself.
        // row[0] == row[0] + 1 ⇒ fails unless handled properly by transition logic.
        // Here: is_transition == false ⇒ so no assertions are enforced.
        let len = 1;

        // Each row = previous + 1, with 4 rows total, 2 columns.
        // Last row must match public values [4, 1]
        // randomness = 5 + 10x + 15x^2 + 20x^3
        // | m1 | m2 | a1      | a2      | a3 |
        // | 1  | 1  | 1/(r-1) | 1/(r-1) | 0 |
        let air = RowLogicAir::<2>;

        let main_col: Vec<BabyBear> = (1..=len).map(|i| BabyBear::new(i)).collect();
        let main = gen_main(&main_col);

        let aux_randomness = BinomialExtensionField::<BabyBear, 4>::from_basis_coefficients_slice(
            [
                BabyBear::new(1005),
                BabyBear::new(10010),
                BabyBear::new(10015),
                BabyBear::new(10020),
            ]
            .as_ref(),
        )
        .unwrap();
        let aux = gen_aux(&main_col, &aux_randomness);

        check_constraints::<BabyBear, BinomialExtensionField<BabyBear, 4>, _>(
            &air,
            &main,
            &Some(aux),
            &aux_randomness.as_basis_coefficients_slice(),
            &vec![BabyBear::new(len), BabyBear::new(1)],
        );
    }

    // A very simple permutation
    fn permute<F: Field>(x: &Vec<F>) -> Vec<F> {
        x.iter().rev().cloned().collect::<Vec<F>>()
    }
}
