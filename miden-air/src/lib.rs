//! # Miden AIR Super Traits
//!
//! This crate provides unified super traits for defining AIRs (Algebraic Intermediate Representations)
//! in the Miden VM ecosystem. Instead of implementing multiple individual traits from `p3-air`,
//! implementers only need to implement these super traits which bundle all necessary functionality.
//!
//! ## Core Traits
//!
//! - [`MidenAirBuilder`]: Super trait for all AIR builders, containing all builder methods
//! - [`MidenAir`]: Super trait for all AIR definitions, containing all AIR methods
//!
//! ## Design Philosophy
//!
//! The goal is to provide single traits to implement. Implementers should only work with
//! `MidenAir` and `MidenAirBuilder`, without needing to understand or import the underlying
//! p3-air traits.
//!
//! ## Usage
//!
//! ### Defining an AIR
//!
//! ```rust,ignore
//! use miden_air::{MidenAir, MidenAirBuilder};
//! use p3_field::{Field, ExtensionField};
//!
//! struct MyAir { width: usize }
//!
//! impl<F: Field, EF: ExtensionField<F>> MidenAir<F, EF> for MyAir {
//!     fn width(&self) -> usize { self.width }
//!     fn eval<AB: MidenAirBuilder<F = F, EF = EF>>(&self, builder: &mut AB) {
//!         // All builder methods available directly on builder
//!         let main = builder.main();
//!         builder.when_first_row().assert_zero(main.row_slice(0).unwrap()[0]);
//!     }
//! }
//!
//! // Use macro to implement p3-air traits
//! impl_p3_air_traits!(MyAir, BinomialExtensionField<_, 2>);
//! ```

mod air;
mod builder;
mod filtered_builder;

pub use air::MidenAir;
pub use builder::MidenAirBuilder;
pub use filtered_builder::FilteredMidenAirBuilder;


// Re-export for convenience
pub use p3_air::{
    Air, AirBuilder, AirBuilderWithPublicValues, BaseAir, BaseAirWithAuxTrace,
    BaseAirWithPublicValues, ExtensionBuilder, FilteredAirBuilder, PairBuilder,
    PermutationAirBuilder,
};
pub use p3_field::{Algebra, ExtensionField, Field, PrimeCharacteristicRing};
pub use p3_matrix;
pub use p3_matrix::Matrix;
pub use p3_matrix::dense::RowMajorMatrix;
