#![no_std]

extern crate alloc;

pub mod deep;
pub mod fri;
mod merkle_tree;

pub use merkle_tree::*;
