use alloc::vec::Vec;

use p3_air::{
    Air, AirBuilder, AirBuilderWithPublicValues, ExtensionBuilder, PermutationAirBuilder,
};
use p3_field::{ExtensionField, Field};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::stack::{VerticalPair, ViewPair};
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
/// - `public_values`: Public values provided to the builder
#[instrument(name = "check constraints", skip_all)]
pub(crate) fn check_constraints<EF, F, A>(
    air: &A,
    main: &RowMajorMatrix<F>,
    aux: &RowMajorMatrix<EF>,
    aux_randomness: &[EF],
    public_values: &Vec<F>,
) where
    EF: ExtensionField<F>,
    F: Field,
    A: for<'a> Air<DebugConstraintBuilder<'a, EF, F>>,
{
    ark_std::println!("check constraints");
    let height = main.height();
    ark_std::println!("height: {}", height);

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

        let aux_local = unsafe { aux.row_slice_unchecked(row_index) };
        let aux_next = unsafe { aux.row_slice_unchecked(row_index_next) };
        let aux = VerticalPair::new(
            RowMajorMatrixView::new_row(&*aux_local),
            RowMajorMatrixView::new_row(&*aux_next),
        );

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

/// A builder that runs constraint assertions during testing.
///
/// Used in conjunction with [`check_constraints`] to simulate
/// an execution trace and verify that the AIR logic enforces all constraints.
#[derive(Debug)]
pub struct DebugConstraintBuilder<'a, EF, F>
where
    EF: ExtensionField<F>,
    F: Field,
{
    /// The index of the row currently being evaluated.
    row_index: usize,
    /// A view of the current and next row as a vertical pair.
    main: ViewPair<'a, F>,
    /// A view of the current and next row as a vertical pair.
    /// Note that although the aux field elements are extension field elements,
    /// they are stored with their base field representations.
    aux: ViewPair<'a, EF>,
    /// randomness that is used to compute aux
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

impl<'a, EF, F> AirBuilder for DebugConstraintBuilder<'a, EF, F>
where
    EF: ExtensionField<F>,
    F: Field,
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

impl<EF, F: Field> AirBuilderWithPublicValues for DebugConstraintBuilder<'_, EF, F>
where
    EF: ExtensionField<F>,
{
    type PublicVar = Self::F;

    fn public_values(&self) -> &[Self::F] {
        self.public_values
    }
}

impl<EF, F> ExtensionBuilder for DebugConstraintBuilder<'_, EF, F>
where
    EF: ExtensionField<F>,
    F: Field,
{
    type EF = EF;

    type ExprEF = EF;

    type VarEF = EF;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        x.into()
            .as_base()
            .iter()
            .for_each(|&xi| self.assert_zero(xi));
    }
}

impl<'a, EF, F> PermutationAirBuilder for DebugConstraintBuilder<'a, EF, F>
where
    EF: ExtensionField<F>,
    F: Field,
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

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_air::{BaseAir, BaseAirWithPublicValues, MultiPhaseBaseAir};
    use p3_baby_bear::BabyBear;
    use p3_field::BasedVectorSpace;
    use p3_field::extension::BinomialExtensionField;

    use super::*;
    use crate::{gen_logup_col, permute};

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

    impl<F: Field, const W: usize> MultiPhaseBaseAir<F> for RowLogicAir<W> {
        fn aux_width(&self) -> usize {
            // 3 extension field elements per row
            12
        }

        fn num_randomness(&self) -> usize {
            1
        }
    }

    impl<EF, F, const W: usize> Air<DebugConstraintBuilder<'_, EF, F>> for RowLogicAir<W>
    where
        EF: ExtensionField<F>,
        F: Field,
    {
        fn eval(&self, builder: &mut DebugConstraintBuilder<'_, EF, F>) {

            ark_std::println!("debug constraint builder check");
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

            let r = builder.aux_randomness[0];
            let xi = main.top.get(0, 0).unwrap();
            let yi = main.top.get(0, 1).unwrap();
            let ti = aux.top.get(0, 0).unwrap();
            let wi = aux.top.get(0, 1).unwrap();
            let a3_top = aux.top.get(0, 2).unwrap();
            let a3_bot = aux.bottom.get(0, 2).unwrap();

            ark_std::println!("randomness: {:?}", r);
            builder.assert_eq_ext::<EF, EF>(EF::ONE, ti * (r - EF::from(xi)));
            builder.assert_eq_ext::<EF, EF>(EF::ONE, wi * (r - EF::from(yi)));

            builder
                .when_transition()
                .assert_eq_ext::<EF, EF>(a3_bot, a3_top + ti - wi);
            // a3[last] = \sum ti - \sim w_i
            // it is 0 is {ti} is a permutation of {wi}
            builder.when_last_row().assert_zero_ext::<EF>(a3_bot);

            // ======================
            // public input
            // ======================
            // Add public value equality on last row for extra coverage
            let public_values = builder.public_values;
            let mut when_last = builder.when(builder.is_last_row);
            for (i, &pv) in public_values.iter().enumerate().take(W) {
                ark_std::println!("{}: {:?}", i, pv);
                ark_std::println!("{}: {:?}", i, main.top.get(0, i));
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

    // Generate the aux trace for logup arguments.
    fn gen_aux(
        main_col: &Vec<BabyBear>,
        aux_randomness: &BinomialExtensionField<BabyBear, 4>,
    ) -> RowMajorMatrix<BinomialExtensionField<BabyBear, 4>> {
        let perm_main_col = permute(main_col);
        let len = main_col.len();

        let aux1 = gen_logup_col(&main_col, aux_randomness);
        let aux2 = gen_logup_col(&perm_main_col, aux_randomness);
        let mut aux3 = vec![aux1[0] - aux2[0]];
        for i in 1..len {
            aux3.push(aux3[i - 1] + aux1[i] - aux2[i])
        }
        let aux_values = aux1
            .iter()
            .zip(aux2.iter().zip(aux3.iter()))
            .flat_map(|(a1, (a2, a3))| vec![a1, a2, a3])
            .cloned()
            .collect();

        RowMajorMatrix::new(aux_values, 3)
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
                BabyBear::new(5),
                BabyBear::new(10),
                BabyBear::new(15),
                BabyBear::new(20),
            ]
            .as_ref(),
        )
        .unwrap();
        let aux = gen_aux(&main_col, &aux_randomness);

        check_constraints(
            &air,
            &main,
            &aux,
            &[aux_randomness],
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

        check_constraints(
            &air,
            &main,
            &aux,
            &[aux_randomness],
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

        check_constraints(
            &air,
            &main,
            &aux,
            &[aux_randomness],
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

        check_constraints(
            &air,
            &main,
            &aux,
            &[aux_randomness],
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
                BabyBear::new(5),
                BabyBear::new(10),
                BabyBear::new(15),
                BabyBear::new(20),
            ]
            .as_ref(),
        )
        .unwrap();
        let aux = gen_aux(&main_col, &aux_randomness);

        check_constraints(
            &air,
            &main,
            &aux,
            &[aux_randomness],
            &vec![BabyBear::new(len), BabyBear::new(1)],
        );
    }
}
