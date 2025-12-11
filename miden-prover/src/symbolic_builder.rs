use alloc::vec;
use alloc::vec::Vec;

use miden_air::{MidenAir, MidenAirBuilder, RowMajorMatrix};
use p3_field::{ExtensionField, Field};
use p3_util::log2_ceil_usize;
use tracing::instrument;

use crate::{Entry, SymbolicExpression, SymbolicVariable};

#[instrument(name = "infer log of constraint degree", skip_all)]
pub fn get_log_quotient_degree<F, EF, A>(
    air: &A,
    preprocessed_width: usize,
    num_public_values: usize,
    is_zk: usize,
    aux_width: usize,
    num_randomness: usize,
) -> usize
where
    F: Field,
    EF: ExtensionField<F>,
    A: MidenAir<F, EF>,
{
    assert!(is_zk <= 1, "is_zk must be either 0 or 1");
    // We pad to at least degree 2, since a quotient argument doesn't make sense with smaller degrees.
    let constraint_degree = (get_max_constraint_degree::<F, EF, A>(
        air,
        preprocessed_width,
        num_public_values,
        aux_width,
        num_randomness,
    ) + is_zk)
        .max(2);

    // The quotient's actual degree is approximately (max_constraint_degree - 1) n,
    // where subtracting 1 comes from division by the vanishing polynomial.
    // But we pad it to a power of two so that we can efficiently decompose the quotient.
    log2_ceil_usize(constraint_degree - 1)
}

#[instrument(name = "infer constraint degree", skip_all, level = "debug")]
pub fn get_max_constraint_degree<F, EF, A>(
    air: &A,
    preprocessed_width: usize,
    num_public_values: usize,
    aux_width: usize,
    num_randomness: usize,
) -> usize
where
    F: Field,
    EF: ExtensionField<F>,
    A: MidenAir<F, EF>,
{
    get_symbolic_constraints::<F, EF, A>(
        air,
        preprocessed_width,
        num_public_values,
        aux_width,
        num_randomness,
    )
    .iter()
    .map(|c| c.degree_multiple())
    .max()
    .unwrap_or(0)
}

#[instrument(name = "evaluate constraints symbolically", skip_all, level = "debug")]
pub fn get_symbolic_constraints<F, EF, A>(
    air: &A,
    preprocessed_width: usize,
    num_public_values: usize,
    aux_width: usize,
    num_randomness: usize,
) -> Vec<SymbolicExpression<F>>
where
    F: Field,
    EF: ExtensionField<F>,
    A: MidenAir<F, EF>,
{
    let num_periodic_values = air.periodic_table().len();
    let mut builder = SymbolicAirBuilder::<F>::new_with_periodic(
        preprocessed_width,
        air.width(),
        aux_width,
        num_randomness,
        num_public_values,
        num_periodic_values,
    );
    air.eval(&mut builder);
    builder.constraints()
}

/// An `AirBuilder` for evaluating constraints symbolically, and recording them for later use.
#[derive(Debug)]
pub struct SymbolicAirBuilder<F: Field> {
    preprocessed: RowMajorMatrix<SymbolicVariable<F>>,
    main: RowMajorMatrix<SymbolicVariable<F>>,
    aux: RowMajorMatrix<SymbolicVariable<F>>,
    aux_randomness: Vec<SymbolicVariable<F>>,
    public_values: Vec<SymbolicVariable<F>>,
    periodic_values: Vec<SymbolicVariable<F>>,
    constraints: Vec<SymbolicExpression<F>>,
}

impl<F: Field> SymbolicAirBuilder<F> {
    pub fn new(
        preprocessed_width: usize,
        width: usize,
        aux_width: usize,
        num_randomness: usize,
        num_public_values: usize,
    ) -> Self {
        Self::new_with_periodic(
            preprocessed_width,
            width,
            aux_width,
            num_randomness,
            num_public_values,
            0,
        )
    }

    pub fn new_with_periodic(
        preprocessed_width: usize,
        width: usize,
        aux_width: usize,
        num_randomness: usize,
        num_public_values: usize,
        num_periodic_values: usize,
    ) -> Self {
        let prep_values = [0, 1]
            .into_iter()
            .flat_map(|offset| {
                (0..preprocessed_width)
                    .map(move |index| SymbolicVariable::new(Entry::Preprocessed { offset }, index))
            })
            .collect();
        let main_values = [0, 1]
            .into_iter()
            .flat_map(|offset| {
                (0..width).map(move |index| SymbolicVariable::new(Entry::Main { offset }, index))
            })
            .collect();
        let aux_values = [0, 1] // Aux trace also use consecutive rows for LogUp based permutation check
            .into_iter()
            .flat_map(|offset| {
                (0..aux_width).map(move |index| SymbolicVariable::new(Entry::Aux { offset }, index))
            })
            .collect();
        let aux = RowMajorMatrix::new(aux_values, aux_width);
        let randomness = Self::sample_randomness(num_randomness);
        let public_values = (0..num_public_values)
            .map(move |index| SymbolicVariable::new(Entry::Public, index))
            .collect();
        let periodic_values = (0..num_periodic_values)
            .map(move |index| SymbolicVariable::new(Entry::Periodic, index))
            .collect();
        Self {
            preprocessed: RowMajorMatrix::new(prep_values, preprocessed_width),
            main: RowMajorMatrix::new(main_values, width),
            aux,
            aux_randomness: randomness,
            public_values,
            periodic_values,
            constraints: vec![],
        }
    }

    pub fn constraints(self) -> Vec<SymbolicExpression<F>> {
        self.constraints
    }

    pub(crate) fn sample_randomness(num_randomness: usize) -> Vec<SymbolicVariable<F>> {
        (0..num_randomness)
            .map(|index| SymbolicVariable::new(Entry::Challenge, index))
            .collect()
    }
}

impl<F: Field> MidenAirBuilder for SymbolicAirBuilder<F> {
    type F = F;
    type Expr = SymbolicExpression<F>;
    type Var = SymbolicVariable<F>;
    type M = RowMajorMatrix<Self::Var>;
    type EF = F;
    type ExprEF = SymbolicExpression<F>;
    type VarEF = SymbolicVariable<F>;
    type PeriodicVal = SymbolicVariable<F>;

    type PublicVar = SymbolicVariable<F>;

    type MP = RowMajorMatrix<SymbolicVariable<F>>;
    type RandomVar = SymbolicVariable<F>;

    fn main(&self) -> Self::M {
        self.main.clone()
    }

    fn is_first_row(&self) -> Self::Expr {
        SymbolicExpression::IsFirstRow
    }

    fn is_last_row(&self) -> Self::Expr {
        SymbolicExpression::IsLastRow
    }

    /// # Panics
    /// This function panics if `size` is not `2`.
    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            SymbolicExpression::IsTransition
        } else {
            panic!("uni-stark only supports a window size of 2")
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.constraints.push(x.into());
    }

    fn permutation(&self) -> Self::MP {
        self.aux.clone()
    }

    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        &self.aux_randomness
    }

    fn aux_bus_boundary_values(&self) -> &[Self::VarEF] {
        unimplemented!()
    }

    fn public_values(&self) -> &[Self::PublicVar] {
        &self.public_values
    }

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        self.constraints.push(x.into());
    }

    fn preprocessed(&self) -> Self::M {
        self.preprocessed.clone()
    }

    fn periodic_evals(&self) -> &[<Self as MidenAirBuilder>::PeriodicVal] {
        &self.periodic_values
    }
}

#[cfg(test)]
mod tests {
    use miden_air::MidenAir;
    use p3_goldilocks::Goldilocks;
    use p3_matrix::Matrix;

    use super::*;

    #[derive(Debug)]
    struct MockAir {
        // Store (entry_type, index) pairs instead of SymbolicVariables
        constraint_specs: Vec<(Entry, usize)>,
        width: usize,
    }

    impl MidenAir<Goldilocks, Goldilocks> for MockAir {
        fn width(&self) -> usize {
            self.width
        }

        fn eval<AB: MidenAirBuilder<F = Goldilocks>>(&self, builder: &mut AB) {
            let main = builder.main();

            for (entry, index) in &self.constraint_specs {
                match entry {
                    Entry::Main { offset } => {
                        builder.assert_zero(main.row_slice(*offset).unwrap()[*index].clone());
                    }
                    _ => panic!("Test only supports Main entry"),
                }
            }
        }
    }

    #[test]
    fn test_get_log_quotient_degree_no_constraints() {
        let air = MockAir {
            constraint_specs: vec![],
            width: 4,
        };
        let log_degree = get_log_quotient_degree::<Goldilocks, Goldilocks, _>(&air, 3, 2, 0, 0, 0);
        assert_eq!(log_degree, 0);
    }

    #[test]
    fn test_get_log_quotient_degree_single_constraint() {
        let air = MockAir {
            constraint_specs: vec![(Entry::Main { offset: 0 }, 0)],
            width: 4,
        };
        let log_degree = get_log_quotient_degree::<Goldilocks, Goldilocks, _>(&air, 3, 2, 0, 0, 0);
        assert_eq!(log_degree, log2_ceil_usize(1));
    }

    #[test]
    fn test_get_log_quotient_degree_multiple_constraints() {
        let air = MockAir {
            constraint_specs: vec![
                (Entry::Main { offset: 0 }, 0),
                (Entry::Main { offset: 1 }, 1),
                (Entry::Main { offset: 0 }, 2),
            ],
            width: 4,
        };
        let log_degree = get_log_quotient_degree::<Goldilocks, Goldilocks, _>(&air, 3, 2, 0, 0, 0);
        assert_eq!(log_degree, log2_ceil_usize(1));
    }

    #[test]
    fn test_get_max_constraint_degree_no_constraints() {
        let air = MockAir {
            constraint_specs: vec![],
            width: 4,
        };
        let max_degree = get_max_constraint_degree::<Goldilocks, Goldilocks, _>(&air, 3, 2, 0, 0);
        assert_eq!(
            max_degree, 0,
            "No constraints should result in a degree of 0"
        );
    }

    #[test]
    fn test_get_max_constraint_degree_multiple_constraints() {
        let air = MockAir {
            constraint_specs: vec![
                (Entry::Main { offset: 0 }, 0),
                (Entry::Main { offset: 1 }, 1),
                (Entry::Main { offset: 0 }, 2),
            ],
            width: 4,
        };
        let max_degree = get_max_constraint_degree::<Goldilocks, Goldilocks, _>(&air, 3, 2, 0, 0);
        assert_eq!(max_degree, 1, "Max constraint degree should be 1");
    }

    #[test]
    fn test_get_symbolic_constraints() {
        let c1: SymbolicVariable<Goldilocks> = SymbolicVariable::new(Entry::Main { offset: 0 }, 0);
        let c2: SymbolicVariable<Goldilocks> = SymbolicVariable::new(Entry::Main { offset: 1 }, 1);

        let air = MockAir {
            constraint_specs: vec![
                (Entry::Main { offset: 0 }, 0),
                (Entry::Main { offset: 1 }, 1),
            ],
            width: 4,
        };

        let constraints = get_symbolic_constraints::<Goldilocks, Goldilocks, _>(&air, 3, 2, 0, 0);

        assert_eq!(constraints.len(), 2, "Should return exactly 2 constraints");

        assert!(
            constraints.iter().any(|x| matches!(x, SymbolicExpression::Variable(v) if v.index == c1.index && v.entry == c1.entry)),
            "Expected constraint {c1:?} was not found"
        );

        assert!(
            constraints.iter().any(|x| matches!(x, SymbolicExpression::Variable(v) if v.index == c2.index && v.entry == c2.entry)),
            "Expected constraint {c2:?} was not found"
        );
    }

    #[test]
    fn test_symbolic_air_builder_initialization() {
        let builder = SymbolicAirBuilder::<Goldilocks>::new(2, 4, 0, 0, 3);

        let expected_main = [
            SymbolicVariable::<Goldilocks>::new(Entry::Main { offset: 0 }, 0),
            SymbolicVariable::<Goldilocks>::new(Entry::Main { offset: 0 }, 1),
            SymbolicVariable::<Goldilocks>::new(Entry::Main { offset: 0 }, 2),
            SymbolicVariable::<Goldilocks>::new(Entry::Main { offset: 0 }, 3),
            SymbolicVariable::<Goldilocks>::new(Entry::Main { offset: 1 }, 0),
            SymbolicVariable::<Goldilocks>::new(Entry::Main { offset: 1 }, 1),
            SymbolicVariable::<Goldilocks>::new(Entry::Main { offset: 1 }, 2),
            SymbolicVariable::<Goldilocks>::new(Entry::Main { offset: 1 }, 3),
        ];

        let builder_main = builder.main.values;

        assert_eq!(
            builder_main.len(),
            expected_main.len(),
            "Main matrix should have the expected length"
        );

        for (expected, actual) in expected_main.iter().zip(builder_main.iter()) {
            assert_eq!(expected.index, actual.index, "Index mismatch");
            assert_eq!(expected.entry, actual.entry, "Entry mismatch");
        }
    }

    #[test]
    fn test_symbolic_air_builder_is_first_last_row() {
        let builder = SymbolicAirBuilder::<Goldilocks>::new(2, 4, 0, 0, 3);

        assert!(
            matches!(builder.is_first_row(), SymbolicExpression::IsFirstRow),
            "First row condition did not match"
        );

        assert!(
            matches!(builder.is_last_row(), SymbolicExpression::IsLastRow),
            "Last row condition did not match"
        );
    }

    #[test]
    fn test_symbolic_air_builder_assert_zero() {
        let mut builder = SymbolicAirBuilder::<Goldilocks>::new(2, 4, 0, 0, 3);
        let expr = SymbolicExpression::Constant(Goldilocks::new(5));
        builder.assert_zero(expr);

        let constraints = builder.constraints();
        assert_eq!(constraints.len(), 1, "One constraint should be recorded");

        assert!(
            constraints.iter().any(
                |x| matches!(x, SymbolicExpression::Constant(val) if *val == Goldilocks::new(5))
            ),
            "Constraint should match the asserted one"
        );
    }
}
