use crate::{MidenAirBuilder, RowMajorMatrix};

/// Super trait for all AIR definitions in the Miden VM ecosystem.
///
/// This trait contains all methods from `BaseAir`, `BaseAirWithPublicValues`,
/// `BaseAirWithAuxTrace`, and `Air`. Implementers only need to implement this
/// single trait.
///
/// To use your AIR with the STARK prover/verifier, you'll need to also implement
/// the p3-air traits using the `impl_p3_air_traits!` macro.
///
/// # Type Parameters
///
/// - `F`: The base field type
/// - `EF`: The extension field type (used for auxiliary traces like LogUp)
///
/// # Required Methods
///
/// - [`width`](MidenAir::width) - Number of columns in the main trace
/// - [`eval`](MidenAir::eval) - Constraint evaluation logic
///
/// # Optional Methods (with default implementations)
///
/// All other methods have default implementations that can be overridden as needed.
pub trait MidenAir<F, EF>: Sync {
    // ==================== BaseAir Methods ====================

    /// The number of columns (a.k.a. registers) in this AIR.
    fn width(&self) -> usize;

    /// Return an optional preprocessed trace matrix to be included in the prover's trace.
    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        None
    }

    // ==================== BaseAirWithPublicValues Methods ====================

    /// Return the number of expected public values.
    fn num_public_values(&self) -> usize {
        0
    }

    // ==================== BaseAirWithAuxTrace Methods ====================

    /// Number of challenges (extension fields) that is required to compute the aux trace
    fn num_randomness(&self) -> usize {
        0
    }

    /// Number of columns (in based field) that is required for aux trace
    fn aux_width(&self) -> usize {
        0
    }

    /// Build an aux trace (EF-based) given the main trace and EF challenges.
    /// Return None to indicate no aux or to fall back to legacy behavior.
    fn build_aux_trace(
        &self,
        _main: &RowMajorMatrix<F>,
        _challenges: &[EF],
    ) -> Option<RowMajorMatrix<F>> {
        None
    }

    /// Load an aux builder.
    ///
    /// An aux builder takes in a main matrix and a randomness, and generate a aux matrix.
    fn with_aux_builder<Builder>(&mut self, _builder: Builder)
    where
        Builder: Fn(&RowMajorMatrix<F>, &[EF]) -> RowMajorMatrix<F> + Send + Sync + 'static,
    {
        // default: do nothing
    }

    // ==================== Air Methods ====================

    /// Evaluate all AIR constraints using the provided builder.
    ///
    /// The builder provides both the trace on which the constraints
    /// are evaluated on as well as the method of accumulating the
    /// constraint evaluations.
    ///
    /// # Arguments
    /// - `builder`: Mutable reference to a `MidenAirBuilder` for defining constraints.
    fn eval<AB: MidenAirBuilder<F = F, EF = EF>>(&self, builder: &mut AB);
}

/// Helper macro to implement p3-air traits by delegating to MidenAir.
///
/// This macro generates the boilerplate implementations of `BaseAir`, `BaseAirWithPublicValues`,
/// `BaseAirWithAuxTrace`, and `Air` that simply delegate to your `MidenAir` implementation.
///
/// # Usage
///
/// ```rust,ignore
/// use miden_air::{MidenAir, MidenAirBuilder, impl_p3_air_traits};
/// use p3_field::extension::BinomialExtensionField;
///
/// struct MyAir { width: usize }
///
/// impl<F: Field, EF: ExtensionField<F>> MidenAir<F, EF> for MyAir {
///     fn width(&self) -> usize { self.width }
///     fn eval<AB: MidenAirBuilder<F = F, EF = EF>>(&self, builder: &mut AB) {
///         // constraints...
///     }
/// }
///
/// // Generate all p3-air trait implementations
/// impl_p3_air_traits!(MyAir, BinomialExtensionField<_, 2>);
/// ```
#[macro_export]
macro_rules! impl_p3_air_traits {
    ($air_type:ty, $ef_type:ty) => {
        impl<F: $crate::Field> $crate::BaseAir<F> for $air_type
        where
            $ef_type: $crate::ExtensionField<F>,
        {
            fn width(&self) -> usize {
                <Self as $crate::MidenAir<F, $ef_type>>::width(self)
            }

            fn preprocessed_trace(&self) -> Option<$crate::p3_matrix::dense::RowMajorMatrix<F>> {
                <Self as $crate::MidenAir<F, $ef_type>>::preprocessed_trace(self)
            }
        }

        impl<F: $crate::Field> $crate::BaseAirWithPublicValues<F> for $air_type
        where
            $ef_type: $crate::ExtensionField<F>,
        {
            fn num_public_values(&self) -> usize {
                <Self as $crate::MidenAir<F, $ef_type>>::num_public_values(self)
            }
        }

        impl<F: $crate::Field, EF: $crate::ExtensionField<F>> $crate::BaseAirWithAuxTrace<F, EF>
            for $air_type
        {
            fn num_randomness(&self) -> usize {
                <Self as $crate::MidenAir<F, EF>>::num_randomness(self)
            }

            fn aux_width(&self) -> usize {
                <Self as $crate::MidenAir<F, EF>>::aux_width(self)
            }

            fn build_aux_trace(
                &self,
                main: &$crate::p3_matrix::dense::RowMajorMatrix<F>,
                challenges: &[EF],
            ) -> Option<$crate::p3_matrix::dense::RowMajorMatrix<F>> {
                <Self as $crate::MidenAir<F, EF>>::build_aux_trace(self, main, challenges)
            }

            fn with_aux_builder<Builder>(&mut self, builder: Builder)
            where
                Builder: Fn(
                        &$crate::p3_matrix::dense::RowMajorMatrix<F>,
                        &[EF],
                    ) -> $crate::p3_matrix::dense::RowMajorMatrix<F>
                    + Send
                    + Sync
                    + 'static,
            {
                <Self as $crate::MidenAir<F, EF>>::with_aux_builder(self, builder)
            }
        }

        impl<AB: $crate::MidenAirBuilder> $crate::Air<AB> for $air_type {
            fn eval(&self, builder: &mut AB) {
                <Self as $crate::MidenAir<AB::F, AB::EF>>::eval(self, builder)
            }
        }
    };
}
