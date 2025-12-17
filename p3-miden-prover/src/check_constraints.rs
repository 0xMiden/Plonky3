use alloc::vec::Vec;

use p3_miden_air::{MidenAir, MidenAirBuilder};
use p3_field::{BasedVectorSpace, ExtensionField, Field};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::stack::ViewPair;
use tracing::instrument;

use crate::util::prover_row_to_ext;

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
            aux_local_ext = prover_row_to_ext::<F, EF>(&aux_local);

            let aux_next = unsafe { aux_matrix.row_slice_unchecked(row_index_next) };
            aux_next_ext = prover_row_to_ext::<F, EF>(&aux_next);

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

        // Compute periodic values for the current row
        let periodic_table = air.periodic_table();
        let periodic_values: Vec<EF> = periodic_table
            .iter()
            .map(|col| {
                if col.is_empty() {
                    EF::ZERO
                } else {
                    // Use modulo to get the repeating value
                    // Convert from base field to extension field using From trait
                    EF::from(col[row_index % col.len()])
                }
            })
            .collect();

        let mut builder = DebugConstraintBuilder {
            row_index,
            main,
            aux,
            aux_randomness,
            preprocessed: preprocessed_pair,
            public_values,
            periodic_values,
            is_first_row: F::from_bool(row_index == 0),
            is_last_row: F::from_bool(row_index == height - 1),
            is_transition: F::from_bool(row_index != height - 1),
        };

        air.eval(&mut builder);
    });
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
    /// Periodic column values (computed for the current row)
    periodic_values: Vec<EF>,
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
    type PeriodicVal = EF;

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

    fn aux_bus_boundary_values(&self) -> &[Self::VarEF] {
        unimplemented!()
    }

    fn periodic_evals(&self) -> &[Self::PeriodicVal] {
        &self.periodic_values
    }
}

#[cfg(test)]
mod tests {

    use alloc::vec;

    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_goldilocks::Goldilocks;

    use super::*;

    /// A test AIR that enforces a simple linear transition logic:
    /// - Each cell in the next row must equal the current cell plus 1 (i.e., `next = current + 1`)
    /// - On the last row, the current row must match the provided public values.
    ///
    /// This is useful for validating constraint evaluation, transition logic,
    /// and row condition flags (first/last/transition).
    #[derive(Debug)]
    struct RowLogicAir {
        with_aux: bool,
    }

    impl<F, EF> MidenAir<F, EF> for RowLogicAir
    where
        F: Field,
        EF: ExtensionField<F>,
    {
        fn width(&self) -> usize {
            2
        }

        fn aux_width(&self) -> usize {
            if self.with_aux { 3 } else { 0 }
        }

        fn num_randomness(&self) -> usize {
            if self.with_aux { 1 } else { 0 }
        }

        fn eval<AB: MidenAirBuilder<F = F>>(&self, builder: &mut AB) {
            let main = builder.main();
            let aux = builder.permutation();

            // ======================
            // main trace
            // ======================
            // | main1             | main2            |
            // | row[i]            | perm(main1)[i]   |
            // | row[i+1]=row[i]+1 | perm(main1)[i+1] |

            let a = main.get(0, 0).unwrap();
            let b = main.get(1, 0).unwrap();

            // New logic: enforce row[i+1] = row[i] + 1, only on transitions
            builder.when_transition().assert_eq(b, a + F::ONE);

            // ======================
            // aux trace
            // ======================
            if <Self as MidenAir<F, EF>>::num_randomness(self) != 0 {
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
                let xi = main.get(0, 0).unwrap();
                let yi = main.get(0, 1).unwrap();

                let r = builder.permutation_randomness()[0];

                // current row EF elements
                let t_i = aux.get(0, 0).unwrap();
                let w_i = aux.get(0, 1).unwrap();
                let s_i = aux.get(0, 2).unwrap();
                // next row EF elements
                let t_next = aux.get(1, 0).unwrap();
                let w_next = aux.get(1, 1).unwrap();
                let s_next = aux.get(1, 2).unwrap();

                // t * (r - x_i) == 1  and  w * (r - y_i) == 1
                // Convert xi and yi to ExprEF by going through the Into trait
                let r_expr = r.into();
                let xi_ef: AB::ExprEF = AB::ExprEF::from(xi.into());
                let yi_ef: AB::ExprEF = AB::ExprEF::from(yi.into());
                builder.assert_eq_ext(t_i.into() * (r_expr.clone() - xi_ef), AB::ExprEF::ONE);
                builder.assert_eq_ext(w_i.into() * (r_expr - yi_ef), AB::ExprEF::ONE);

                // transition is correct: s' = s + t' - w'
                builder
                    .when_transition()
                    .assert_eq_ext(s_next.into(), s_i.into() + t_next.into() - w_next.into());

                // a3[last] = Σ(t - w) == 0 if multisets match
                builder.when_last_row().assert_zero_ext(s_i);
            }

            // ======================
            // public input
            // ======================
            // Add public value equality on last row for extra coverage
            let public_values = builder.public_values();
            let pv0 = public_values[0];
            let pv1 = public_values[1];

            let mut when_last = builder.when_last_row();
            when_last.assert_eq(main.get(0, 0).unwrap(), pv0);
            when_last.assert_eq(main.get(0, 1).unwrap(), pv1);
        }
    }

    // A very simple permutation
    fn permute<F: Field>(x: &[F]) -> Vec<F> {
        x.iter().rev().cloned().collect::<Vec<F>>()
    }

    // Generate a main trace.
    // The first column is incremental
    // The second column is the rev of the first column
    fn gen_main(main_col: &[Goldilocks]) -> RowMajorMatrix<Goldilocks> {
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
        main_col: &[Goldilocks],
        aux_randomness: &BinomialExtensionField<Goldilocks, 2>,
    ) -> RowMajorMatrix<Goldilocks> {
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
        super::super::generate_logup_trace::<BinomialExtensionField<Goldilocks, 2>, _>(
            &main,
            aux_randomness,
        )
    }

    #[test]
    fn test_incremental_rows_with_last_row_check() {
        // Each row = previous + 1, with 4 rows total, 2 columns.
        // Last row must match public values [4, 4]
        let air = RowLogicAir { with_aux: false };
        let values = vec![
            Goldilocks::ONE,    // Row 0
            Goldilocks::from_u64(2), // Row 1
            Goldilocks::from_u64(3), // Row 2
            Goldilocks::from_u64(4), // Row 3 (last)
        ];
        let main = gen_main(&values);
        check_constraints::<_, BinomialExtensionField<Goldilocks, 2>, _>(
            &air,
            &main,
            &None,
            &[],
            &vec![Goldilocks::from_u64(4), Goldilocks::from_u64(1)],
        );
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
        let air = RowLogicAir { with_aux: true };
        let main_col: Vec<_> = (1..=len).map(Goldilocks::from_u64).collect();
        let main = gen_main(&main_col);

        let aux_randomness =
            BinomialExtensionField::<Goldilocks, 2>::from_basis_coefficients_slice(
                [Goldilocks::from_u64(1005), Goldilocks::from_u64(10010)].as_ref(),
            )
            .unwrap();

        let aux = gen_aux(&main_col, &aux_randomness);

        check_constraints::<Goldilocks, BinomialExtensionField<Goldilocks, 2>, _>(
            &air,
            &main,
            &Some(aux),
            aux_randomness.as_basis_coefficients_slice(),
            &vec![Goldilocks::from_u64(len), Goldilocks::from_u64(1)],
        );
    }

    #[test]
    #[should_panic]
    fn test_incorrect_increment_logic() {
        // Row 2 does not equal row 1 + 1 → should fail on transition from row 1 to 2.
        let air = RowLogicAir { with_aux: false };
        let values = vec![
            Goldilocks::ONE,
            Goldilocks::ONE, // Row 0
            Goldilocks::from_u64(2),
            Goldilocks::from_u64(2), // Row 1
            Goldilocks::from_u64(5),
            Goldilocks::from_u64(5), // Row 2 (wrong)
            Goldilocks::from_u64(6),
            Goldilocks::from_u64(6), // Row 3
        ];
        let main = RowMajorMatrix::new(values, 2);
        check_constraints::<_, BinomialExtensionField<Goldilocks, 2>, _>(
            &air,
            &main,
            &None,
            &[],
            &vec![Goldilocks::from_u64(6); 2],
        );
    }

    #[test]
    #[should_panic]
    fn test_wrong_last_row_public_value() {
        // The transition logic is fine, but public value check fails at the last row.
        let air = RowLogicAir { with_aux: false };
        let values = vec![
            Goldilocks::ONE,
            Goldilocks::ONE, // Row 0
            Goldilocks::from_u64(2),
            Goldilocks::from_u64(2), // Row 1
            Goldilocks::from_u64(3),
            Goldilocks::from_u64(3), // Row 2
            Goldilocks::from_u64(4),
            Goldilocks::from_u64(4), // Row 3
        ];
        let main = RowMajorMatrix::new(values, 2);
        // Wrong public value on column 1
        check_constraints::<_, BinomialExtensionField<Goldilocks, 2>, _>(
            &air,
            &main,
            &None,
            &[],
            &vec![Goldilocks::from_u64(4), Goldilocks::from_u64(5)],
        );
    }

    #[test]
    fn test_single_row_wraparound_logic() {
        // A single-row matrix still performs a wraparound check with itself.
        // row[0] == row[0] + 1 ⇒ fails unless handled properly by transition logic.
        // Here: is_transition == false ⇒ so no assertions are enforced.
        let air = RowLogicAir { with_aux: false };
        let values = vec![
            Goldilocks::from_u64(99),
            Goldilocks::from_u64(77), // Row 0
        ];
        let main = RowMajorMatrix::new(values, 2);
        check_constraints::<_, BinomialExtensionField<Goldilocks, 2>, _>(
            &air,
            &main,
            &None,
            &[],
            &vec![Goldilocks::from_u64(99), Goldilocks::from_u64(77)],
        );
    }
}
