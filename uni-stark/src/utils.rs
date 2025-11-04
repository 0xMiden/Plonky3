use alloc::vec;
use alloc::vec::Vec;
use p3_air::AirBuilder;
use p3_field::Field;
use p3_field::PrimeCharacteristicRing;

/// Multiplication in BinomialExtensionField<F, W>
/// Hardcoded for degree 4 extension field.
// Example: BabyBearExt4 is defined by X^4 = W, where W = 11
// TODO: maybe this already exits somewhere?
pub fn ext_field_mul<AB: AirBuilder>(a: &[AB::Expr], b: &[AB::Expr], w: &AB::F) -> Vec<AB::Expr>
where
    AB::F: Field,
{
    assert_eq!(a.len(), 4, "Expected degree 4 extension field element");
    assert_eq!(b.len(), 4, "Expected degree 4 extension field element");

    let mut res = vec![
        AB::Expr::ZERO,
        AB::Expr::ZERO,
        AB::Expr::ZERO,
        AB::Expr::ZERO,
    ];

    // Expanding the multiplication:
    // res[0] = a0*b0 + W*(a1*b3 + a2*b2 + a3*b1)
    // res[1] = a0*b1 + a1*b0 + W*(a2*b3 + a3*b2)
    // res[2] = a0*b2 + a1*b1 + a2*b0 + W*a3*b3
    // res[3] = a0*b3 + a1*b2 + a2*b1 + a3*b0
    for i in 0..4 {
        for j in 0..4 {
            let prod = a[i].clone() * b[j].clone();
            if i + j < 4 {
                res[i + j] = res[i + j].clone() + prod;
            } else {
                // i + j >= 4, multiply by W since X^(i+j) = X^(i+j-4) * W
                res[i + j - 4] = res[i + j - 4].clone() + prod * w.clone();
            }
        }
    }

    res
}

/// Addition in BinomialExtensionField<F, W>
/// Hardcoded for degree 4 extension field.
pub fn ext_field_add<AB: AirBuilder>(a: &[AB::Expr], b: &[AB::Expr]) -> Vec<AB::Expr> {
    assert_eq!(a.len(), 4, "Expected degree 4 extension field element");
    assert_eq!(b.len(), 4, "Expected degree 4 extension field element");
    a.iter()
        .zip(b.iter())
        .map(|(a, b)| a.clone() + b.clone())
        .collect()
}

/// Subtraction in BinomialExtensionField<F, W>
/// Hardcoded for degree 4 extension field.
pub fn ext_field_sub<AB: AirBuilder>(a: &[AB::Expr], b: &[AB::Expr]) -> Vec<AB::Expr> {
    assert_eq!(a.len(), 4, "Expected degree 4 extension field element");
    assert_eq!(b.len(), 4, "Expected degree 4 extension field element");
    a.iter()
        .zip(b.iter())
        .map(|(a, b)| a.clone() - b.clone())
        .collect()
}
