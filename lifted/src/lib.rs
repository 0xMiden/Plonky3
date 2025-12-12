#![no_std]

extern crate alloc;

pub mod deep;
pub mod fri;
mod merkle_tree;
mod utils;

pub use merkle_tree::*;
