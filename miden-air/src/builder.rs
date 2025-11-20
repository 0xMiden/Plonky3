use core::ops::{Add, Mul, Sub};

use crate::{
    Algebra, ExtensionField, Field, FilteredMidenAirBuilder, Matrix, PrimeCharacteristicRing,
};

/// Super trait for all AIR builders in the Miden VM ecosystem.
///
/// This trait contains all methods from `AirBuilder`, `AirBuilderWithPublicValues`,
/// `PairBuilder`, `ExtensionBuilder`, and `PermutationAirBuilder`. Implementers only
/// need to implement this single trait.
///
/// To use your builder with the STARK prover/verifier, you'll also need to implement
/// the p3-air traits using the `impl_p3_air_builder_traits!` macro.
///
/// # Type Parameters
///
/// All type parameters are associated types that must be specified by implementers.
///
/// # Required Methods
///
/// Core methods that must be implemented:
/// - `main()` - Access to main trace
/// - `is_first_row()` - First row indicator
/// - `is_last_row()` - Last row indicator
/// - `is_transition_window()` - Transition window indicator
/// - `assert_zero()` - Core constraint assertion
/// - `public_values()` - Access to public values
/// - `preprocessed()` - Access to preprocessed columns
/// - `assert_zero_ext()` - Extension field constraint assertion
/// - `permutation()` - Access to permutation trace
/// - `permutation_randomness()` - Access to randomness
///
/// # Optional Methods (with default implementations)
///
/// All other methods have default implementations.
pub trait MidenAirBuilder: Sized {
    // ==================== Associated Types from AirBuilder ====================

    /// Underlying field type.
    type F: Field + Sync;

    /// Serves as the output type for an AIR constraint evaluation.
    type Expr: Algebra<Self::F> + Algebra<Self::Var>;

    /// The type of the variable appearing in the trace matrix.
    type Var: Into<Self::Expr>
        + Clone
        + Send
        + Sync
        + Add<Self::F, Output = Self::Expr>
        + Add<Self::Var, Output = Self::Expr>
        + Add<Self::Expr, Output = Self::Expr>
        + Sub<Self::F, Output = Self::Expr>
        + Sub<Self::Var, Output = Self::Expr>
        + Sub<Self::Expr, Output = Self::Expr>
        + Mul<Self::F, Output = Self::Expr>
        + Mul<Self::Var, Output = Self::Expr>
        + Mul<Self::Expr, Output = Self::Expr>;

    /// Matrix type holding variables.
    type M: Matrix<Self::Var>;

    // ==================== Associated Types from AirBuilderWithPublicValues ====================

    /// Type representing a public variable.
    type PublicVar: Into<Self::Expr> + Copy;

    // ==================== Associated Types from ExtensionBuilder ====================

    /// Extension field type.
    type EF: ExtensionField<Self::F>;

    /// Expression type over extension field elements.
    type ExprEF: Algebra<Self::Expr> + Algebra<Self::EF>;

    /// Variable type over extension field elements.
    type VarEF: Into<Self::ExprEF> + Copy + Send + Sync;

    // ==================== Associated Types from PermutationAirBuilder ====================

    /// Matrix type over extension-field variables representing permutation registers.
    type MP: Matrix<Self::VarEF>;

    /// Randomness variable type used in permutation commitments.
    type RandomVar: Into<Self::ExprEF> + Copy;

    // ==================== Core AirBuilder Methods ====================

    /// Return the matrix representing the main (primary) trace registers.
    fn main(&self) -> Self::M;

    /// Expression evaluating to 1 on the first row, 0 elsewhere.
    fn is_first_row(&self) -> Self::Expr;

    /// Expression evaluating to 1 on the last row, 0 elsewhere.
    fn is_last_row(&self) -> Self::Expr;

    /// Expression evaluating to 1 on all transition rows (not last row), 0 on last row.
    fn is_transition(&self) -> Self::Expr {
        self.is_transition_window(2)
    }

    /// Expression evaluating to 1 on rows except the last `size - 1` rows, 0 otherwise.
    fn is_transition_window(&self, size: usize) -> Self::Expr;

    /// Returns a sub-builder whose constraints are enforced only when `condition` is nonzero.
    fn when<I: Into<Self::Expr>>(&mut self, condition: I) -> FilteredMidenAirBuilder<'_, Self> {
        FilteredMidenAirBuilder {
            inner: self,
            condition: condition.into(),
        }
    }

    /// Returns a sub-builder whose constraints are enforced only when `x != y`.
    fn when_ne<I1: Into<Self::Expr>, I2: Into<Self::Expr>>(
        &mut self,
        x: I1,
        y: I2,
    ) -> FilteredMidenAirBuilder<'_, Self> {
        self.when(x.into() - y.into())
    }

    /// Returns a sub-builder whose constraints are enforced only on the first row.
    fn when_first_row(&mut self) -> FilteredMidenAirBuilder<'_, Self> {
        self.when(self.is_first_row())
    }

    /// Returns a sub-builder whose constraints are enforced only on the last row.
    fn when_last_row(&mut self) -> FilteredMidenAirBuilder<'_, Self> {
        self.when(self.is_last_row())
    }

    /// Returns a sub-builder whose constraints are enforced on all rows except the last.
    fn when_transition(&mut self) -> FilteredMidenAirBuilder<'_, Self> {
        self.when(self.is_transition())
    }

    /// Returns a sub-builder whose constraints are enforced on all rows except the last `size - 1`.
    fn when_transition_window(&mut self, size: usize) -> FilteredMidenAirBuilder<'_, Self> {
        self.when(self.is_transition_window(size))
    }

    /// Assert that the given element is zero.
    ///
    /// Where possible, batching multiple assert_zero calls
    /// into a single assert_zeros call will improve performance.
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I);

    /// Assert that every element of a given array is 0.
    ///
    /// This should be preferred over calling `assert_zero` multiple times.
    fn assert_zeros<const N: usize, I: Into<Self::Expr>>(&mut self, array: [I; N]) {
        for elem in array {
            self.assert_zero(elem);
        }
    }

    /// Assert that a given array consists of only boolean values.
    fn assert_bools<const N: usize, I: Into<Self::Expr>>(&mut self, array: [I; N]) {
        let zero_array = array.map(|x| x.into().bool_check());
        self.assert_zeros(zero_array);
    }

    /// Assert that `x` element is equal to `1`.
    fn assert_one<I: Into<Self::Expr>>(&mut self, x: I) {
        self.assert_zero(x.into() - Self::Expr::ONE);
    }

    /// Assert that the given elements are equal.
    fn assert_eq<I1: Into<Self::Expr>, I2: Into<Self::Expr>>(&mut self, x: I1, y: I2) {
        self.assert_zero(x.into() - y.into());
    }

    /// Assert that `x` is a boolean, i.e. either `0` or `1`.
    ///
    /// Where possible, batching multiple assert_bool calls
    /// into a single assert_bools call will improve performance.
    fn assert_bool<I: Into<Self::Expr>>(&mut self, x: I) {
        self.assert_zero(x.into().bool_check());
    }

    // ==================== AirBuilderWithPublicValues Methods ====================

    /// Return the list of public variables.
    fn public_values(&self) -> &[Self::PublicVar];

    // ==================== AirBuilderWithPeriodicValues Methods ====================

    /// Return the list of periodic values.
    fn periodic_evals(&self) -> &[Self::F];

    // ==================== PairBuilder Methods ====================

    /// Return a matrix of preprocessed registers.
    fn preprocessed(&self) -> Self::M;

    // ==================== ExtensionBuilder Methods ====================

    /// Assert that an extension field expression is zero.
    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>;

    /// Assert that two extension field expressions are equal.
    fn assert_eq_ext<I1, I2>(&mut self, x: I1, y: I2)
    where
        I1: Into<Self::ExprEF>,
        I2: Into<Self::ExprEF>,
    {
        self.assert_zero_ext(x.into() - y.into());
    }

    /// Assert that an extension field expression is equal to one.
    fn assert_one_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        self.assert_eq_ext(x, Self::ExprEF::ONE)
    }

    // ==================== PermutationAirBuilder Methods ====================

    /// Return the matrix representing permutation registers over EF variables.
    fn permutation(&self) -> Self::MP;

    /// Return the list of randomness values for permutation argument.
    fn permutation_randomness(&self) -> &[Self::RandomVar];

    // ==================== AuxAirBuilder Methods ====================

    /// Aux bus boundary values: EF finals, one per aux/bus column, carried in the proof.
    fn aux_bus_boundary_values(&self) -> &[Self::VarEF];
}

/// Helper macro to implement p3-air builder traits by delegating to MidenAirBuilder.
///
/// This macro generates the boilerplate implementations of `AirBuilder`, `AirBuilderWithPublicValues`,
/// `PairBuilder`, `ExtensionBuilder`, and `PermutationAirBuilder` that delegate to your
/// `MidenAirBuilder` implementation.
///
/// # Usage
///
/// ```rust,ignore
/// use miden_air::{MidenAirBuilder, impl_p3_air_builder_traits};
///
/// struct MyBuilder { /* ... */ }
///
/// impl MidenAirBuilder for MyBuilder {
///     // Implement all required methods...
/// }
///
/// // Generate all p3-air builder trait implementations
/// impl_p3_air_builder_traits!(MyBuilder);
/// ```
#[macro_export]
macro_rules! impl_p3_air_builder_traits {
    // Non-generic type
    ($builder_type:ty) => {
        impl $crate::AirBuilder for $builder_type {
            type F = <Self as $crate::MidenAirBuilder>::F;
            type Expr = <Self as $crate::MidenAirBuilder>::Expr;
            type Var = <Self as $crate::MidenAirBuilder>::Var;
            type M = <Self as $crate::MidenAirBuilder>::M;

            fn main(&self) -> Self::M {
                <Self as $crate::MidenAirBuilder>::main(self)
            }

            fn is_first_row(&self) -> Self::Expr {
                <Self as $crate::MidenAirBuilder>::is_first_row(self)
            }

            fn is_last_row(&self) -> Self::Expr {
                <Self as $crate::MidenAirBuilder>::is_last_row(self)
            }

            fn is_transition_window(&self, size: usize) -> Self::Expr {
                <Self as $crate::MidenAirBuilder>::is_transition_window(self, size)
            }

            fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
                <Self as $crate::MidenAirBuilder>::assert_zero(self, x)
            }
        }

        impl $crate::AirBuilderWithPublicValues for $builder_type {
            type PublicVar = <Self as $crate::MidenAirBuilder>::PublicVar;

            fn public_values(&self) -> &[Self::PublicVar] {
                <Self as $crate::MidenAirBuilder>::public_values(self)
            }
        }

        impl $crate::PairBuilder for $builder_type {
            fn preprocessed(&self) -> Self::M {
                <Self as $crate::MidenAirBuilder>::preprocessed(self)
            }
        }

        impl $crate::ExtensionBuilder for $builder_type
        where
            Self::F: $crate::Field,
        {
            type EF = <Self as $crate::MidenAirBuilder>::EF;
            type ExprEF = <Self as $crate::MidenAirBuilder>::ExprEF;
            type VarEF = <Self as $crate::MidenAirBuilder>::VarEF;

            fn assert_zero_ext<I>(&mut self, x: I)
            where
                I: Into<Self::ExprEF>,
            {
                <Self as $crate::MidenAirBuilder>::assert_zero_ext(self, x)
            }
        }

        impl $crate::PermutationAirBuilder for $builder_type
        where
            Self::F: $crate::Field,
        {
            type MP = <Self as $crate::MidenAirBuilder>::MP;
            type RandomVar = <Self as $crate::MidenAirBuilder>::RandomVar;

            fn permutation(&self) -> Self::MP {
                <Self as $crate::MidenAirBuilder>::permutation(self)
            }

            fn permutation_randomness(&self) -> &[Self::RandomVar] {
                <Self as $crate::MidenAirBuilder>::permutation_randomness(self)
            }
        }
    };
    // Generic type with bounds (supports lifetimes and types)
    ($builder_type:ident<$($gen:tt),+> where $($bound:tt)+) => {
        impl<$($gen),+> $crate::AirBuilder for $builder_type<$($gen),+>
        where
            $($bound)+
        {
            type F = <Self as $crate::MidenAirBuilder>::F;
            type Expr = <Self as $crate::MidenAirBuilder>::Expr;
            type Var = <Self as $crate::MidenAirBuilder>::Var;
            type M = <Self as $crate::MidenAirBuilder>::M;

            fn main(&self) -> Self::M {
                <Self as $crate::MidenAirBuilder>::main(self)
            }

            fn is_first_row(&self) -> Self::Expr {
                <Self as $crate::MidenAirBuilder>::is_first_row(self)
            }

            fn is_last_row(&self) -> Self::Expr {
                <Self as $crate::MidenAirBuilder>::is_last_row(self)
            }

            fn is_transition_window(&self, size: usize) -> Self::Expr {
                <Self as $crate::MidenAirBuilder>::is_transition_window(self, size)
            }

            fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
                <Self as $crate::MidenAirBuilder>::assert_zero(self, x)
            }
        }

        impl<$($gen),+> $crate::AirBuilderWithPublicValues for $builder_type<$($gen),+>
        where
            $($bound)+
        {
            type PublicVar = <Self as $crate::MidenAirBuilder>::PublicVar;

            fn public_values(&self) -> &[Self::PublicVar] {
                <Self as $crate::MidenAirBuilder>::public_values(self)
            }
        }

        impl<$($gen),+> $crate::PairBuilder for $builder_type<$($gen),+>
        where
            $($bound)+
        {
            fn preprocessed(&self) -> Self::M {
                <Self as $crate::MidenAirBuilder>::preprocessed(self)
            }
        }

        impl<$($gen),+> $crate::ExtensionBuilder for $builder_type<$($gen),+>
        where
            $($bound)+
        {
            type EF = <Self as $crate::MidenAirBuilder>::EF;
            type ExprEF = <Self as $crate::MidenAirBuilder>::ExprEF;
            type VarEF = <Self as $crate::MidenAirBuilder>::VarEF;

            fn assert_zero_ext<I>(&mut self, x: I)
            where
                I: Into<Self::ExprEF>,
            {
                <Self as $crate::MidenAirBuilder>::assert_zero_ext(self, x)
            }
        }

        impl<$($gen),+> $crate::PermutationAirBuilder for $builder_type<$($gen),+>
        where
            $($bound)+
        {
            type MP = <Self as $crate::MidenAirBuilder>::MP;
            type RandomVar = <Self as $crate::MidenAirBuilder>::RandomVar;

            fn permutation(&self) -> Self::MP {
                <Self as $crate::MidenAirBuilder>::permutation(self)
            }

            fn permutation_randomness(&self) -> &[Self::RandomVar] {
                <Self as $crate::MidenAirBuilder>::permutation_randomness(self)
            }
        }
    };
}
