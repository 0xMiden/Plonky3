use alloc::vec::Vec;

use p3_field::{BasedVectorSpace, ExtensionField, Field};

/// Helper: convert a flattened base-field row (slice of `F`) into a Vec<EF>
#[cfg(debug_assertions)]
pub(crate) fn prover_row_to_ext<F, EF>(row: &[F]) -> Vec<EF>
where
    F: Field,
    EF: ExtensionField<F> + BasedVectorSpace<F>,
{
    row.chunks(EF::DIMENSION)
        .map(|chunk| EF::from_basis_coefficients_slice(chunk).unwrap())
        .collect()
}

// NOTE: these helpers use different representations and semantics.
// `prover_row_to_ext` accepts `&[F]` (raw base-field coefficients) and
// builds `EF` via `EF::from_basis_coefficients_slice(chunk)` (trusted,
// panicking on malformation). `verifier_row_to_ext` accepts `&[EF]`
// limbs and reconstructs each element by summing `EF::ith_basis_element(i) * limb`
// (validating length and returning `Option`). Unifying them would
// require conversions/extra trait bounds and would change verification
// semantics and performance, so they remain separate.

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
