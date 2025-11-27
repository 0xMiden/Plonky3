use alloc::vec::Vec;
use p3_air::ExtensionBuilder;
use p3_air::PermutationAirBuilder;
use p3_field::Field;
use p3_matrix::Matrix;

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

/// Enforce LogUp-style permutation constraints using the builder’s permutation aux view and randomness.
///
/// Given xi, yi from main trace, this asserts over the extension field:
/// - t_i * (r - xi) == 1
/// - w_i * (r - yi) == 1
/// - s_0 = t_0 - w_0; s' = s + t' - w'; and s_last = 0
///
/// Works whether the aux rows are provided as EF elements or as flattened base limbs.
pub fn enforce_logup_permutation<AB, X, Y>(builder: &mut AB, xi: X, yi: Y)
where
    AB: ExtensionBuilder + PermutationAirBuilder,
    AB::F: Field,
    X: Into<AB::Expr>,
    Y: Into<AB::Expr>,
{
    let (local, next) = permutation_rows_ext(builder);
    let t_i = local[0].clone();
    let w_i = local[1].clone();
    let s_i = local[2].clone();
    let t_next = next[0].clone();
    let w_next = next[1].clone();
    let s_next = next[2].clone();

    // EF randomness
    let r_expr: AB::ExprEF = builder.permutation_randomness()[0].into();

    // Convert base exprs xi/yi into EF exprs
    let xi_ext: AB::ExprEF = Into::<AB::Expr>::into(xi).into();
    let yi_ext: AB::ExprEF = Into::<AB::Expr>::into(yi).into();

    // t * (r - x_i) == 1 and w * (r - y_i) == 1
    builder.assert_one_ext(t_i.clone() * (r_expr.clone() - xi_ext));
    builder.assert_one_ext(w_i.clone() * (r_expr - yi_ext));

    // Running sum constraints
    builder
        .when_first_row()
        .assert_eq_ext(s_i.clone(), t_i.clone() - w_i.clone());
    builder
        .when_transition()
        .assert_eq_ext(s_next, s_i.clone() + t_next - w_next);
    builder.when_last_row().assert_zero_ext(s_i);
}
