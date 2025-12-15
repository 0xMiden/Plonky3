//! # Lifted PCS
//!
//! A polynomial commitment scheme (PCS) combining DEEP quotient construction with FRI
//! for efficient low-degree testing over two-adic fields.
//!
//! ## Overview
//!
//! This crate provides:
//!
//! - **[`deep`]**: DEEP (Dimension Extension of Evaluation Protocol) quotient construction
//!   for batching polynomial evaluation claims into a single low-degree polynomial.
//!
//! - **[`fri`]**: Fast Reed-Solomon Interactive Oracle Proof for proving that a committed
//!   polynomial has degree below a target bound.
//!
//! - **[`merkle_tree`]**: Lifted Merkle tree commitments supporting matrices of varying
//!   heights via upsampling or cyclic lifting strategies.
//!
//! - **[`pcs`]**: The complete PCS interface combining DEEP and FRI for opening and
//!   verifying polynomial evaluations.
//!
//! ## Architecture
//!
//! The PCS follows this flow:
//!
//! 1. **Commit**: Commit to polynomial evaluations using a lifted Merkle tree (MMCS).
//!
//! 2. **Open** (Prover):
//!    - Construct DEEP quotient `Q(X) = (f(X) - f(z)) / (X - z)` combining all claims.
//!    - Run FRI commit phase to fold `Q(X)` down to a low-degree polynomial.
//!    - Answer query challenges with Merkle openings.
//!
//! 3. **Verify** (Verifier):
//!    - Replay challenger to derive the same folding challenges.
//!    - For each query: verify Merkle openings, recompute DEEP quotient, check FRI folding.
//!    - Verify final polynomial evaluation.
//!
//! ## Feature Flags
//!
//! - `parallel`: Enable parallel computation via Rayon (recommended for production).

#![no_std]

extern crate alloc;

/// DEEP quotient construction for batched polynomial evaluation.
pub mod deep;

/// FRI protocol for low-degree testing.
pub mod fri;

/// Lifted Merkle tree commitments for matrices of varying heights.
pub mod merkle_tree;

/// Complete PCS interface combining DEEP and FRI.
pub mod pcs;

mod utils;

pub use merkle_tree::*;
