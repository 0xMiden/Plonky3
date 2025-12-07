//! Arity-4 FRI folding via inverse FFT.
//!
//! Given evaluations of a polynomial `f` on a coset `s·⟨ω⟩` where `ω = i` is a primitive
//! 4th root of unity, we recover `f(β)` for an arbitrary challenge point `β`.
//!
//! ## Setup
//!
//! Let `f(X) = c₀ + c₁X + c₂X² + c₃X³` with evaluations on the coset `s·⟨ω⟩`:
//!
//! ```text
//! y₀ = f(s),   y₁ = f(ωs),   y₂ = f(ω²s),   y₃ = f(ω³s)
//! ```
//!
//! We store these in **bit-reversed order**: `[y₀, y₂, y₁, y₃]`.
//!
//! ## Algorithm
//!
//! 1. **Inverse FFT**: Recover coefficients of `f(sX)` from evaluations on `⟨ω⟩`.
//! 2. **Evaluate**: Compute `f(sX)` at `X = β/s`, yielding `f(β)`.

use p3_field::{ExtensionField, TwoAdicField};

/// Size-4 inverse FFT (unscaled), input in bit-reversed order.
///
/// Returns coefficients `[c₀, c₁, c₂, c₃]` of `4·f(sX) = c₀ + c₁X + c₂X² + c₃X³`.
#[inline(always)]
fn ifft4<F: TwoAdicField, EF: ExtensionField<F>>(evals: [EF; 4]) -> [EF; 4] {
    let w = F::two_adic_generator(2); // ω = i, primitive 4th root of unity

    // Input (bit-reversed): [y₀, y₂, y₁, y₃]
    let (y0, y2, y1, y3) = (evals[0], evals[1], evals[2], evals[3]);

    // Inverse DFT formula (without 1/N normalization):
    //   4cⱼ = Σₖ yₖ · ω^(−jk)
    //
    // Expanded for each coefficient:
    //   4c₀ = y₀ + y₁ + y₂ + y₃
    //   4c₁ = y₀ − iy₁ − y₂ + iy₃
    //   4c₂ = y₀ − y₁ + y₂ − y₃
    //   4c₃ = y₀ + iy₁ − y₂ − iy₃

    // -------------------------------------------------------------------------
    // Stage 0: length-2 butterflies on bit-reversed pairs
    // -------------------------------------------------------------------------
    let s02 = y0 + y2; // y₀ + y₂  (used in c₀, c₂)
    let d02 = y0 - y2; // y₀ − y₂  (used in c₁, c₃)
    let s13 = y1 + y3; // y₁ + y₃  (used in c₀, c₂)
    let d31 = y3 - y1; // y₃ − y₁  (note: negated so we can multiply by ω instead of ω⁻¹)

    // -------------------------------------------------------------------------
    // Stage 1: combine via length-4 butterflies
    //
    // Rewriting the target formulas using stage 0 results:
    //   4c₀ = (y₀ + y₂) + (y₁ + y₃)           = s02 + s13
    //   4c₂ = (y₀ + y₂) − (y₁ + y₃)           = s02 − s13
    //   4c₁ = (y₀ − y₂) + i(y₃ − y₁)          = d02 + i·d31
    //   4c₃ = (y₀ − y₂) − i(y₃ − y₁)          = d02 − i·d31
    // -------------------------------------------------------------------------
    let d31_w = d31 * w; // i · (y₃ − y₁)

    [
        s02 + s13,   // 4c₀
        d02 + d31_w, // 4c₁
        s02 - s13,   // 4c₂
        d02 - d31_w, // 4c₃
    ]
}

/// Evaluate `f(β)` from evaluations on a coset.
///
/// ## Inputs
///
/// - `evals`: evaluations `[f(s), f(ω²s), f(ωs), f(ω³s)]` in bit-reversed order,
///   equivalently `[f(s), f(−s), f(is), f(−is)]` since `ω = i`.
/// - `s_inv`: the inverse of the coset generator `s`.
/// - `beta`: the FRI folding challenge `β`.
///
/// ## FRI Context
///
/// In arity-4 FRI, the polynomial `f` is evaluated on cosets of the form `s·⟨ω⟩`.
/// The verifier needs to check that `f(β)` equals the claimed folded value.
/// This function recovers `f(β)` from the four coset evaluations via interpolation.
#[inline(always)]
pub fn fold_evals<F: TwoAdicField, EF: ExtensionField<F>>(
    evals: [EF; 4],
    s_inv: F,
    beta: EF,
) -> EF {
    // Recover coefficients [c₀, c₁, c₂, c₃] of 4·f(sX) via inverse FFT.
    let coeffs = ifft4::<F, EF>(evals);

    // f(β) = f(s · β/s) = (1/4) · (c₀ + c₁·x + c₂·x² + c₃·x³)  where x = β/s.
    //
    // Powers of x are computed independently for instruction-level parallelism.
    let x = beta * s_inv;
    let terms = [
        coeffs[0],              // c₀
        coeffs[1] * x,          // c₁ · x
        coeffs[2] * x.square(), // c₂ · x²
        coeffs[3] * x.cube(),   // c₃ · x³
    ];

    // Divide by 4: use base field 1/4 since EF×F is cheaper than EF.halve().halve().
    let four_inv = F::ONE.halve().halve();
    EF::sum_array::<4>(&terms) * four_inv
}

#[cfg(test)]
mod tests {
    use core::array;

    use p3_baby_bear::BabyBear as F;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_util::log2_strict_usize;
    use rand::distr::StandardUniform;
    use rand::prelude::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::*;
    type EF = BinomialExtensionField<F, 4>;

    #[test]
    fn test() {
        let rng = &mut SmallRng::seed_from_u64(1);
        let beta: EF = rng.sample(StandardUniform);

        let horner = |coeffs: &[EF], x: EF| -> EF {
            coeffs
                .iter()
                .rev()
                .copied()
                .reduce(|acc, c| acc * x + c)
                .unwrap_or_default()
        };

        let n = 64;
        let log_n = log2_strict_usize(n);
        let row_idx = 42;
        let h = F::two_adic_generator(log_n);
        let h_row = h.exp_u64(row_idx);
        let h_row_inv = h_row.inverse();

        // 4-th root of unity: i
        let w = F::two_adic_generator(2);

        // random poly p(X)
        let poly: [EF; 4] = array::from_fn(|_| rng.sample(StandardUniform));

        // order <i> in bit-reversed order
        // [1, -1, i, -i]
        let roots = [F::ONE, F::NEG_ONE, w, -w];
        // [p(h), p(-h), p(i*h), p(-i*h)
        let evals = roots.map(|root| horner(&poly, EF::from(root * h_row)));

        // p(beta)
        let expected = horner(&poly, beta);

        let result = fold_evals(evals, h_row_inv, beta);
        assert_eq!(result, expected);
    }
}
