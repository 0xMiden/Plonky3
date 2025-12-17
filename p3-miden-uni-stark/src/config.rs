//! Re-export config types from p3-uni-stark.
//!
//! This module re-exports the core STARK configuration types from p3-uni-stark,
//! ensuring we use a single StarkGenericConfig trait across the codebase.

pub use p3_uni_stark::{
    Domain, PackedChallenge, PackedVal, PcsError, StarkConfig, StarkGenericConfig, Val,
};
