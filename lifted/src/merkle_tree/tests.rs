//! Integration tests for Merkle tree LMCS commit/open/verify.

use alloc::vec;
use alloc::vec::Vec;

use p3_commit::{BatchOpeningRef, Mmcs};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::tests::{F, base_lmcs};

// ============================================================================
// Integration test
// ============================================================================

#[test]
fn commit_open_verify_roundtrip() {
    let lmcs = base_lmcs();

    let mut rng = SmallRng::seed_from_u64(9);
    let matrices = vec![
        RowMajorMatrix::rand(&mut rng, 2, 3),
        RowMajorMatrix::rand(&mut rng, 4, 5),
        RowMajorMatrix::rand(&mut rng, 8, 7),
    ];
    let dims: Vec<_> = matrices
        .iter()
        .map(|m: &RowMajorMatrix<F>| m.dimensions())
        .collect();

    let (commitment, tree) = lmcs.commit(matrices);
    let final_height = dims.last().unwrap().height;
    let index = final_height - 1; // valid index within range

    let opening = lmcs.open_batch(index, &tree);
    let opening_ref: BatchOpeningRef<'_, F, _> = (&opening).into();
    assert!(
        lmcs.verify_batch(&commitment, &dims, index, opening_ref)
            .is_ok()
    );
}
