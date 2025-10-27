use alloc::vec::Vec;

use p3_field::{ExtensionField, Field, batch_multiplicative_inverse};

// A very simple permutation
pub fn permute<F: Field>(x: &Vec<F>) -> Vec<F> {
    x.iter().rev().cloned().collect::<Vec<F>>()
}

// Give a column m, for each i, generate aux[i] = 1/(r-m[i])
pub fn gen_logup_col<F, EF>(main_col: &[F], randomness: &EF) -> Vec<EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let res = main_col
        .iter()
        .map(|&x| (*randomness - EF::from(x)))
        .collect::<Vec<EF>>();

    batch_multiplicative_inverse(&res)
}
