use p3_field::{BasedVectorSpace, ExtensionField, Field, PrimeCharacteristicRing};
use p3_uni_stark::StarkGenericConfig;

use crate::Val;

/// Helper: convert a flattened base-field row (slice of `F`) into a Vec<EF>
#[allow(dead_code)]
pub(crate) fn prover_row_to_ext<F, EF>(row: &[F]) -> Vec<EF>
where
    F: Field,
    EF: ExtensionField<F> + BasedVectorSpace<F>,
{
    row.chunks(EF::DIMENSION)
        .map(|chunk| EF::from_basis_coefficients_slice(chunk).unwrap())
        .collect()
}

// Helper: convert a flattened base-field row into EF elements.
pub(crate) fn verifier_row_to_ext<F, EF>(row: &[EF]) -> Option<Vec<EF>>
where
    F: Field,
    EF: ExtensionField<F> + BasedVectorSpace<F>,
{
    let dim = EF::DIMENSION;
    if !row.len().is_multiple_of(dim) {
        return None;
    }

    let mut out = Vec::with_capacity(row.len() / dim);
    for chunk in row.chunks(dim) {
        let mut acc = EF::ZERO;
        for (i, limb) in chunk.iter().enumerate() {
            let basis = EF::ith_basis_element(i).unwrap();
            acc += basis * *limb;
        }
        out.push(acc);
    }
    Some(out)
}
