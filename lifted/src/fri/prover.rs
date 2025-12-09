use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field};
use p3_matrix::Matrix;

struct VirtualPoly<'a, F: Field, EF: ExtensionField<F>, M: Matrix<F>, Commit: Mmcs<F>> {
    matrices: Vec<&'a Commit::ProverData<M>>,
    poly: Vec<EF>,
    _marker: PhantomData<F>,
}
