use alloc::vec;
use alloc::vec::Vec;

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_field::{Field, PackedValue, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::{Dimensions, Matrix};
use p3_symmetric::{PaddingFreeSponge, StatefulHasher, TruncatedPermutation};
use rand::rngs::SmallRng;
use rand::{RngCore, SeedableRng};

use crate::merkle_tree::Lifting;

pub(crate) type F = BabyBear;
pub(crate) type P = <F as Field>::Packing;
pub(crate) const WIDTH: usize = 16;
pub(crate) const RATE: usize = 8;
pub(crate) const DIGEST: usize = 8;
pub(crate) type Sponge = PaddingFreeSponge<Poseidon2BabyBear<WIDTH>, WIDTH, RATE, DIGEST>;
pub(crate) type Compressor = TruncatedPermutation<Poseidon2BabyBear<WIDTH>, 2, DIGEST, WIDTH>;

pub(crate) fn components() -> (Sponge, Compressor) {
    let mut rng = SmallRng::seed_from_u64(123);
    let perm = Poseidon2BabyBear::<WIDTH>::new_from_rng_128(&mut rng);
    let sponge = PaddingFreeSponge::<_, WIDTH, RATE, DIGEST>::new(perm.clone());
    let compress = TruncatedPermutation::<_, 2, DIGEST, WIDTH>::new(perm);
    (sponge, compress)
}

pub(crate) fn rand_matrix(h: usize, w: usize, rng: &mut SmallRng) -> RowMajorMatrix<F> {
    let vals = (0..h * w).map(|_| F::new(rng.next_u32())).collect();
    RowMajorMatrix::new(vals, w)
}

pub(crate) fn concatenate_matrices<const RATE: usize>(
    matrices: &[RowMajorMatrix<F>],
) -> RowMajorMatrix<F> {
    let max_height = matrices.last().unwrap().height();
    let width: usize = matrices
        .iter()
        .map(|m| m.width().next_multiple_of(RATE))
        .sum();

    let concatenated_data: Vec<_> = (0..max_height)
        .flat_map(|idx| {
            matrices.iter().flat_map(move |m| {
                let mut row = m.row_slice(idx).unwrap().to_vec();
                let padded_width = row.len().next_multiple_of(RATE);
                row.resize(padded_width, F::ZERO);
                row
            })
        })
        .collect();
    RowMajorMatrix::new(concatenated_data, width)
}

pub(crate) fn build_leaves_single(matrix: &RowMajorMatrix<F>, sponge: &Sponge) -> Vec<[F; DIGEST]> {
    matrix
        .rows()
        .map(|row| {
            let mut state = [F::ZERO; WIDTH];
            sponge.absorb_into(&mut state, row);
            sponge.squeeze(&state)
        })
        .collect()
}

pub(crate) fn matrix_scenarios() -> Vec<Vec<(usize, usize)>> {
    let pack_width = <P as PackedValue>::WIDTH;
    vec![
        vec![(1, 1)],
        vec![(1, RATE - 1)],
        vec![(2, 3), (4, 5), (8, RATE)],
        vec![(1, 5), (1, 3), (2, 7), (4, 1), (8, RATE + 1)],
        vec![
            (pack_width / 2, RATE - 1),
            (pack_width, RATE),
            (pack_width * 2, RATE + 3),
        ],
        vec![(pack_width, RATE + 5), (pack_width * 2, 25)],
        vec![
            (1, RATE * 2),
            (pack_width / 2, RATE * 2 - 1),
            (pack_width, RATE * 2),
            (pack_width * 2, RATE * 3 - 2),
        ],
        vec![(4, RATE - 1), (4, RATE), (8, RATE + 3), (8, RATE * 2)],
        vec![(pack_width * 2, RATE - 1)],
    ]
}

pub(crate) fn lift_matrix(
    matrix: &RowMajorMatrix<F>,
    lifting: Lifting,
    max_height: usize,
) -> RowMajorMatrix<F> {
    let Dimensions { height, width } = matrix.dimensions();
    let data = (0..max_height)
        .flat_map(|index| {
            let mapped_index = lifting.map_index(index, height, max_height);
            matrix.row(mapped_index).unwrap()
        })
        .collect();
    RowMajorMatrix::new(data, width)
}
