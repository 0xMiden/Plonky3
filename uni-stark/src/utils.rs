use alloc::vec;
use alloc::vec::Vec;
use p3_air::ExtensionBuilder;
use p3_field::Field;
use p3_field::PrimeCharacteristicRing;
use p3_field::BasedVectorSpace;
use p3_matrix::Matrix;
use p3_air::PermutationAirBuilder;

/// Multiplication in a binomial extension field defined by X^d = W.
/// Works for any extension degree `d` (length of the input slices).
/// Example: BabyBearExt4 is defined by X^4 = W, where W = 11.
pub fn ext_field_mul<AB: ExtensionBuilder>(a: &[AB::ExprEF], b: &[AB::ExprEF], w: &AB::F) -> Vec<AB::ExprEF>
where
    AB::F: Field,
{
    assert_eq!(a.len(), b.len(), "Mismatched extension element degrees");
    let d = a.len();
    let mut res = vec![AB::ExprEF::ZERO; d];

    // Expanding the multiplication:
    // res[0] = a0*b0 + W*(a1*b3 + a2*b2 + a3*b1)
    // res[1] = a0*b1 + a1*b0 + W*(a2*b3 + a3*b2)
    // res[2] = a0*b2 + a1*b1 + a2*b0 + W*a3*b3
    // res[3] = a0*b3 + a1*b2 + a2*b1 + a3*b0
    for i in 0..d {
        for j in 0..d {
            let prod = a[i].clone() * b[j].clone();
            if i + j < d {
                res[i + j] = res[i + j].clone() + prod;
            } else {
                // i + j >= 4, multiply by W since X^(i+j) = X^(i+j-4) * W
                res[i + j - d] =
                    res[i + j - d].clone() + prod * <AB as p3_air::AirBuilder>::Expr::from(w.clone());
            }
        }
    }

    res
}

/// Addition in a binomial extension field (component-wise over basis coefficients)
pub fn ext_field_add<AB: ExtensionBuilder>(a: &[AB::ExprEF], b: &[AB::ExprEF]) -> Vec<AB::ExprEF> {
    assert_eq!(a.len(), b.len(), "Mismatched extension element degrees");
    a.iter()
        .zip(b.iter())
        .map(|(a, b)| a.clone() + b.clone())
        .collect()
}

/// Subtraction in a binomial extension field (component-wise over basis coefficients)
pub fn ext_field_sub<AB: ExtensionBuilder>(a: &[AB::ExprEF], b: &[AB::ExprEF]) -> Vec<AB::ExprEF> {
    assert_eq!(a.len(), b.len(), "Mismatched extension element degrees");
    a.iter()
        .zip(b.iter())
        .map(|(a, b)| a.clone() - b.clone())
        .collect()
}

/// Convenience: assert an extension element (provided as limbs) equals one.
/// Assumes canonical basis with first limb as constant term.
pub fn assert_ext_one_from_limbs<AB: ExtensionBuilder>(builder: &mut AB, el: &[AB::ExprEF])
where
    AB::F: Field,
{
    if el.is_empty() {
        return;
    }
    builder.assert_one_ext(el[0].clone());
    for limb in &el[1..] {
        builder.assert_zero_ext(limb.clone());
    }
}

/// Combine basis coefficients (limbs) into a single extension-field expression.
/// This is degree-agnostic and uses the canonical basis of `AB::EF`.
pub fn ext_from_limbs<AB: ExtensionBuilder>(limbs: &[AB::ExprEF]) -> AB::ExprEF
where
    AB::F: Field,
    AB::EF: BasedVectorSpace<AB::F>,
{
    let d = <AB::EF as BasedVectorSpace<AB::F>>::DIMENSION;
    // Sum_i limb_i * basis_i, where basis_i has 1 at position i.
    // If `limbs.len() != d`, combine up to `min(d, limbs.len())` to support builders
    // that model EF differently (e.g., base-field EF where d = 1).
    let mut acc = AB::ExprEF::ZERO;
    let m = core::cmp::min(d, limbs.len());
    for (i, limb) in limbs.iter().take(m).enumerate() {
        // Build one-hot basis element in EF.
        let mut coeffs = alloc::vec![AB::F::ZERO; d];
        coeffs[i] = AB::F::ONE;
        let basis_i = <AB::EF as BasedVectorSpace<AB::F>>::from_basis_coefficients_slice(&coeffs)
            .expect("invalid basis coeffs");
        acc += limb.clone() * basis_i;
    }
    acc
}

/// Convert a base-field row slice into extension elements using the canonical basis.
/// Groups every `EF::DIMENSION` base entries into one extension element.
pub fn row_to_ef_elems<AB: ExtensionBuilder>(row: &[AB::Var]) -> Vec<AB::ExprEF>
where
    AB::F: Field,
    AB::EF: BasedVectorSpace<AB::F>,
{
    let d = <AB::EF as BasedVectorSpace<AB::F>>::DIMENSION;
    if d == 0 {
        return vec![];
    }
    let elems = row.chunks(d).map(|chunk| {
        let limbs: Vec<AB::ExprEF> = chunk
            .iter()
            .map(|v| AB::ExprEF::from(Into::<AB::Expr>::into((*v).clone())))
            .collect();
        ext_from_limbs::<AB>(&limbs)
    });
    elems.collect()
}

/// Convenience: get EF elements for current and next aux rows from the builder.
pub fn permutation_rows_ext<AB>(builder: &AB) -> (Vec<AB::ExprEF>, Vec<AB::ExprEF>)
where
    AB: ExtensionBuilder + PermutationAirBuilder,
    AB::F: Field,
{
    let aux = builder.permutation();
    let local = aux.row_slice(0).expect("aux matrix is empty");
    let next = aux.row_slice(1).expect("aux matrix has only 1 row");
    (
        local.iter().cloned().map(Into::into).collect(),
        next.iter().cloned().map(Into::into).collect(),
    )
}
