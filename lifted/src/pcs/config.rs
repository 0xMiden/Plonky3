//! PCS configuration.

use crate::fri::FriParams;

/// Complete PCS configuration combining FRI parameters with additional settings.
///
/// Groups all parameters needed for `open` and `verify` into a single struct,
/// reducing the number of function arguments and ensuring consistent configuration.
pub struct PcsConfig {
    /// FRI protocol parameters (blowup, folding, final degree, queries).
    pub fri: FriParams,

    /// Column alignment for batching in DEEP quotient construction.
    ///
    /// Typically set to the hasher's rate (e.g., 8 for Poseidon2 with WIDTH=16, RATE=8).
    /// Ensures coefficients are aligned for efficient hashing.
    pub alignment: usize,
}
