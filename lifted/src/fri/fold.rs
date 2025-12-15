//! FRI folding via polynomial interpolation.
//!
//! FRI (Fast Reed-Solomon IOP of Proximity) requires computing `f(β)` from evaluations
//! of a polynomial `f` on a coset. This module provides a trait-based abstraction for
//! FRI folding at different arities.
//!
//! ## Arity
//!
//! The **arity** determines how many evaluations are folded together in each round:
//! - **Arity 2**: Fold pairs `{f(s), f(-s)}` using even-odd decomposition
//! - **Arity 4**: Fold quadruples `{f(s), f(-s), f(is), f(-is)}` using inverse FFT
//!
//! Higher arity reduces the number of FRI rounds but increases per-round work.

use alloc::vec::Vec;

use p3_field::{
    Algebra, ExtensionField, PackedField, PackedFieldExtension, PackedValue, TwoAdicField,
};
use p3_matrix::dense::RowMajorMatrixView;
use p3_maybe_rayon::prelude::*;

// ============================================================================
// Trait Definition
// ============================================================================

/// FRI folding strategy for evaluating `f(β)` from coset evaluations.
///
/// Given evaluations of a polynomial `f` on a coset of size `ARITY`, this trait
/// provides a method to recover `f(β)` for an arbitrary challenge point `β`.
pub trait FriFold<const ARITY: usize> {
    /// Evaluate `f(β)` from evaluations on a coset.
    ///
    /// ## Inputs
    ///
    /// - `evals`: evaluations in bit-reversed order
    /// - `s_inv`: inverse of the coset generator `s`
    /// - `beta`: the FRI folding challenge `β`
    fn fold_evals<PF, EF, PEF>(evals: [PEF; ARITY], s_inv: PF, beta: EF) -> PEF
    where
        PF: PackedField,
        PF::Scalar: TwoAdicField,
        EF: ExtensionField<PF::Scalar>,
        PEF: Algebra<PF> + Algebra<EF>;

    /// Fold a matrix of coset evaluations using the challenge `beta`.
    ///
    /// Each row contains evaluations on a coset `s·⟨ω⟩`. Returns folded
    /// evaluations, one per row, maintaining bit-reversed order.
    fn fold_matrix<F: TwoAdicField, EF: ExtensionField<F>>(
        input: RowMajorMatrixView<'_, EF>,
        s_invs: &[F],
        beta: EF,
    ) -> Vec<EF> {
        assert_eq!(input.width, ARITY);
        let (evals, _) = input.values.as_chunks::<ARITY>();

        evals
            .par_iter()
            .zip(s_invs.par_iter())
            .map(|(evals, s_inv)| {
                // Scalar mode: PF=F, EF=EF, PEF=EF
                Self::fold_evals::<F, EF, EF>(*evals, *s_inv, beta)
            })
            .collect()
    }

    /// SIMD-optimized matrix folding using packed field operations.
    ///
    /// Processes multiple rows in parallel using horizontal SIMD packing.
    /// Equivalent to [`Self::fold_matrix`] but faster for large matrices.
    fn fold_matrix_packed<F: TwoAdicField, EF: ExtensionField<F>>(
        input: RowMajorMatrixView<'_, EF>,
        s_invs: &[F],
        beta: EF,
    ) -> Vec<EF> {
        assert_eq!(input.width, ARITY);
        let (evals, _) = input.values.as_chunks::<ARITY>();
        let width = F::Packing::WIDTH;
        if evals.len() < width || width == 1 {
            return Self::fold_matrix(input, s_invs, beta);
        }

        assert!(evals.len().is_multiple_of(width));

        let mut new_evals = EF::zero_vec(evals.len());

        new_evals
            .par_chunks_exact_mut(width)
            .zip(evals.par_chunks_exact(width))
            .zip(s_invs.par_chunks_exact(width))
            .for_each(|((new_evals_chunk, evals_chunk), s_inv_chunk)| {
                let evals_packed = EF::ExtensionPacking::pack_ext_columns(evals_chunk);
                let s_invs_packed = F::Packing::from_slice(s_inv_chunk);
                let new_evals_packed = Self::fold_evals::<F::Packing, EF, EF::ExtensionPacking>(
                    evals_packed,
                    *s_invs_packed,
                    beta,
                );
                new_evals_packed.to_ext_slice(new_evals_chunk);
            });
        new_evals
    }
}

/// Marker type for arity-2 FRI folding.
///
/// Folds pairs of evaluations using the even-odd decomposition:
/// `f(β) = (f(s) + f(-s))/2 + β/s · (f(s) - f(-s))/2`
pub struct FriFold2;

// ============================================================================
// Arity-2 Implementation: Even-Odd Decomposition
// ============================================================================
//
// Any polynomial `f(X)` can be uniquely decomposed into even and odd parts:
//
// ```text
// f(X) = fₑ(X²) + X · fₒ(X²)
// ```
//
// where `fₑ` contains the even-degree coefficients and `fₒ` the odd-degree coefficients.
//
// ## Key Identity
//
// From evaluations at `s` and `−s`, we can recover `fₑ(s²)` and `fₒ(s²)`:
//
// ```text
// f(s)  = fₑ(s²) + s · fₒ(s²)
// f(−s) = fₑ(s²) − s · fₒ(s²)
// ```
//
// Solving:
//
// ```text
// fₑ(s²) = (f(s) + f(−s)) / 2
// fₒ(s²) = (f(s) − f(−s)) / (2s)
// ```
//
// ## FRI Folding
//
// Given a challenge `β`, FRI computes:
//
// ```text
// f(β) = fₑ(β²) + β · fₒ(β²)
// ```
//
// Since we only have evaluations on the coset `{s, −s}`, we interpolate using the identity
// above, noting that `fₑ` and `fₒ` are constant on this coset (they depend only on `s²`).

impl FriFold<2> for FriFold2 {
    /// Evaluate `f(β)` from evaluations on a coset `{s, −s}`.
    ///
    /// ## Inputs
    ///
    /// - `evals`: evaluations `[f(s), f(−s)]` in bit-reversed order.
    /// - `s_inv`: the inverse of the coset generator `s`.
    /// - `beta`: the FRI folding challenge `β`.
    ///
    /// ## Algorithm
    ///
    /// Using the even-odd decomposition `f(X) = fₑ(X²) + X · fₒ(X²)`:
    ///
    /// 1. Compute `fₑ(s²) = (f(s) + f(−s)) / 2`
    /// 2. Compute `fₒ(s²) = (f(s) − f(−s)) / (2s)`
    /// 3. Return `f(β) = fₑ(s²) + β · fₒ(s²)` (valid since `β² = s²` in the folded domain)
    #[inline(always)]
    fn fold_evals<PF, EF, PEF>(evals: [PEF; 2], s_inv: PF, beta: EF) -> PEF
    where
        PF: PackedField,
        PF::Scalar: TwoAdicField,
        EF: ExtensionField<PF::Scalar>,
        PEF: Algebra<PF> + Algebra<EF>,
    {
        // y₀ = f(s), y₁ = f(−s)
        let [y0, y1] = evals;

        // Broadcast beta to PEF
        let beta_packed: PEF = beta.into();

        // f(β) = fₑ(s²) + β · fₒ(s²)
        // Even part: fₑ(s²) = (f(s) + f(−s)) / 2
        // Odd part: fₒ(s²) = (f(s) − f(−s)) / (2s)
        // Combined: ((y0 + y1) + (y0 - y1) * beta * s_inv) / 2
        let sum = y0.clone() + y1.clone();
        let diff = y0 - y1;
        let result = sum + diff * beta_packed * s_inv;

        // Divide by 2
        result.halve()
    }
}

/// Marker type for arity-4 FRI folding.
///
/// Folds quadruples of evaluations via size-4 inverse FFT followed by Horner evaluation.
/// More efficient than two rounds of arity-2 folding.
pub struct FriFold4;

// ============================================================================
// Arity-4 Implementation: Inverse FFT
// ============================================================================
//
// Given evaluations of a polynomial `f` on a coset `s·⟨ω⟩` where `ω = i` is a primitive
// 4th root of unity, we recover `f(β)` for an arbitrary challenge point `β`.
//
// ## Setup
//
// Let `f(X) = c₀ + c₁X + c₂X² + c₃X³` with evaluations on the coset `s·⟨ω⟩`:
//
// ```text
// y₀ = f(s),   y₁ = f(ωs),   y₂ = f(ω²s),   y₃ = f(ω³s)
// ```
//
// We store these in **bit-reversed order**: `[y₀, y₂, y₁, y₃]`.
//
// ## Algorithm
//
// 1. **Inverse FFT**: Recover coefficients of `f(sX)` from evaluations on `⟨ω⟩`.
// 2. **Evaluate**: Compute `f(sX)` at `X = β/s`, yielding `f(β)`.

/// Size-4 inverse FFT (unscaled), input in bit-reversed order.
///
/// Returns coefficients `[c₀, c₁, c₂, c₃]` of `4·f(sX) = c₀ + c₁X + c₂X² + c₃X³`.
///
/// ## Type Parameters
///
/// - `PF`: Packed base field (can be `F` for scalar or `F::Packing` for SIMD)
/// - `PEF`: Packed extension field (can be `EF` for scalar or `EF::ExtensionPacking` for SIMD)
///
/// The caller chooses whether to operate on scalars or packed values by selecting
/// the appropriate type parameters.
#[inline(always)]
fn ifft4<PF, PEF>(evals: [PEF; 4]) -> [PEF; 4]
where
    PF: PackedField,
    PF::Scalar: TwoAdicField,
    PEF: Algebra<PF>,
{
    // ω = i, primitive 4th root of unity
    let w: PF = PF::Scalar::two_adic_generator(2).into();

    // Input (bit-reversed): [y₀, y₂, y₁, y₃]
    let [y0, y2, y1, y3] = evals;

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
    let s02 = y0.clone() + y2.clone(); // y₀ + y₂  (used in c₀, c₂)
    let d02 = y0 - y2; // y₀ − y₂  (used in c₁, c₃)
    let s13 = y1.clone() + y3.clone(); // y₁ + y₃  (used in c₀, c₂)
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
        s02.clone() + s13.clone(),   // 4c₀
        d02.clone() + d31_w.clone(), // 4c₁
        s02 - s13,                   // 4c₂
        d02 - d31_w,                 // 4c₃
    ]
}

impl FriFold<4> for FriFold4 {
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
    fn fold_evals<PF, EF, PEF>(evals: [PEF; 4], s_inv: PF, beta: EF) -> PEF
    where
        PF: PackedField,
        PF::Scalar: TwoAdicField,
        EF: ExtensionField<PF::Scalar>,
        PEF: Algebra<PF> + Algebra<EF>,
    {
        // Recover coefficients [c₀, c₁, c₂, c₃] of 4·f(sX) via inverse FFT.
        let [c0, c1, c2, c3] = ifft4::<PF, PEF>(evals);

        // f(β) = f(s · β/s) = (1/4) · (c₀ + c₁·x + c₂·x² + c₃·x³)  where x = β/s.
        let x = PEF::from(beta) * s_inv;
        let terms = [
            c0,              // c₀
            c1 * x.clone(),  // c₁ · x
            c2 * x.square(), // c₂ · x²
            c3 * x.cube(),   // c₃ · x³
        ];

        // Divide by 4: use base field 1/4 since EF×F is cheaper than EF.halve().halve().
        PEF::sum_array::<4>(&terms).halve().halve()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use core::array;

    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{Field, PrimeCharacteristicRing, TwoAdicField};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_util::reverse_slice_index_bits;
    use rand::distr::{Distribution, StandardUniform};
    use rand::prelude::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Pf = <F as Field>::Packing;
    type Pef = <EF as ExtensionField<F>>::ExtensionPacking;

    /// Test that ifft4 compiles with scalar types (F, EF).
    #[test]
    fn test_ifft4_scalar_types() {
        let evals: [EF; 4] = [EF::ZERO; 4];
        let _coeffs: [EF; 4] = ifft4::<F, EF>(evals);
    }

    /// Test that ifft4 compiles with packed types (Pf, Pef).
    #[test]
    fn test_ifft4_packed_types() {
        let evals: [Pef; 4] = [Pef::ZERO; 4];
        let _coeffs: [Pef; 4] = ifft4::<Pf, Pef>(evals);
    }

    /// Test that fold_evals (arity 2) compiles with scalar types.
    #[test]
    fn test_fold_evals_arity2_scalar_types() {
        let evals: [EF; 2] = [EF::ZERO; 2];
        let s_inv = F::ONE;
        let beta = EF::ONE;
        let _result: EF = FriFold2::fold_evals::<F, EF, EF>(evals, s_inv, beta);
    }

    /// Test that fold_evals (arity 2) compiles with packed types.
    #[test]
    fn test_fold_evals_arity2_packed_types() {
        let evals: [Pef; 2] = [Pef::ZERO; 2];
        let s_inv = Pf::ZERO;
        let beta = EF::ONE;
        let _result: Pef = FriFold2::fold_evals::<Pf, EF, Pef>(evals, s_inv, beta);
    }

    /// Test that fold_evals (arity 4) compiles with scalar types.
    #[test]
    fn test_fold_evals_arity4_scalar_types() {
        let evals: [EF; 4] = [EF::ZERO; 4];
        let s_inv = F::ONE;
        let beta = EF::ONE;
        let _result: EF = FriFold4::fold_evals::<F, EF, EF>(evals, s_inv, beta);
    }

    /// Test that fold_evals (arity 4) compiles with packed types.
    #[test]
    fn test_fold_evals_arity4_packed_types() {
        let evals: [Pef; 4] = [Pef::ZERO; 4];
        let s_inv = Pf::ZERO;
        let beta = EF::ONE;
        let _result: Pef = FriFold4::fold_evals::<Pf, EF, Pef>(evals, s_inv, beta);
    }

    /// Evaluate polynomial using Horner's method.
    fn horner<F: Field, EF: ExtensionField<F>>(coeffs: &[EF], x: F) -> EF {
        coeffs
            .iter()
            .rev()
            .copied()
            .reduce(|acc, c| acc * x + c)
            .unwrap_or(EF::ZERO)
    }

    /// Generic test for FRI folding at any arity.
    ///
    /// Creates a random polynomial of degree `ARITY - 1`, evaluates it on a coset
    /// of size `ARITY`, then verifies that `fold_evals` correctly recovers `f(β)`.
    fn test_fold<F, EF, FF: FriFold<ARITY>, const ARITY: usize>()
    where
        F: TwoAdicField,
        EF: ExtensionField<F>,
        StandardUniform: Distribution<EF> + Distribution<F>,
    {
        let rng = &mut SmallRng::seed_from_u64(1);
        let beta: EF = rng.sample(StandardUniform);

        // Random polynomial of degree ARITY - 1
        let poly: [EF; ARITY] = array::from_fn(|_| rng.sample(StandardUniform));

        // Compute roots of unity in bit-reversed order for this arity
        // For ARITY=2: [1, -1]
        // For ARITY=4: [1, -1, w, -w] = [w^0, w^2, w^1, w^3]
        let roots: [F; ARITY] = {
            let log_arity = ARITY.ilog2() as usize;
            let mut points = F::two_adic_generator(log_arity).powers().collect_n(ARITY);
            reverse_slice_index_bits(&mut points);
            points.try_into().unwrap()
        };

        let s: F = rng.sample(StandardUniform);
        let s_inv = s.inverse();

        // Evaluate polynomial at coset points: [f(s·root) for root in roots]
        let evals: [EF; ARITY] = roots.map(|root| horner(&poly, root * s));

        // Expected: f(beta)
        let expected = horner::<EF, EF>(&poly, beta);

        // Test fold_evals with scalar types: PF=F, EF=EF, PEF=EF
        let result = FF::fold_evals::<F, EF, EF>(evals, s_inv, beta);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_arity_2_babybear() {
        type F = BabyBear;
        type EF = BinomialExtensionField<F, 4>;
        test_fold::<F, EF, FriFold2, 2>();
    }

    #[test]
    fn test_arity_4_babybear() {
        type F = BabyBear;
        type EF = BinomialExtensionField<F, 4>;
        test_fold::<F, EF, FriFold4, 4>();
    }

    /// Test that `fold_matrix` and `fold_matrix_packed` produce identical results.
    fn test_fold_matrix_packed_equivalence<FF: FriFold<ARITY>, const ARITY: usize>() {
        let rng = &mut SmallRng::seed_from_u64(42);

        // Create input matrix with height = multiple of packing width
        let height = Pf::WIDTH * 4; // 4 packed rows worth
        let width = ARITY;
        let values: Vec<EF> = (0..height * width)
            .map(|_| rng.sample(StandardUniform))
            .collect();
        let input = RowMajorMatrix::new(values, width);

        // Generate s_invs (one per row)
        let s_invs: Vec<F> = (0..height)
            .map(|_| rng.sample::<F, _>(StandardUniform).inverse())
            .collect();

        let beta: EF = rng.sample(StandardUniform);

        // Call both implementations
        let result_scalar = FF::fold_matrix::<F, EF>(input.as_view(), &s_invs, beta);
        let result_packed = FF::fold_matrix_packed::<F, EF>(input.as_view(), &s_invs, beta);

        // They should be identical
        assert_eq!(result_scalar, result_packed);
    }

    #[test]
    fn test_fold_matrix_arity2_packed_equivalence() {
        test_fold_matrix_packed_equivalence::<FriFold2, 2>();
    }

    #[test]
    fn test_fold_matrix_arity4_packed_equivalence() {
        test_fold_matrix_packed_equivalence::<FriFold4, 4>();
    }

    /// Test that our folding matches the reference implementation from two_adic_pcs.rs.
    #[test]
    fn test_fold_matches_reference() {
        let mut rng = SmallRng::seed_from_u64(99);

        let log_height = 4; // 16 evaluations
        let num_evals = 1 << log_height;
        let log_arity = 1; // arity 2
        let arity = 1 << log_arity;
        let log_num_cosets = log_height - log_arity;
        let num_cosets = 1 << log_num_cosets;

        // Generate random low-degree polynomial and compute evaluations
        let log_poly_degree = 3;
        let poly_degree = 1 << log_poly_degree;
        let coeffs: Vec<EF> = (0..poly_degree)
            .map(|_| rng.sample(StandardUniform))
            .collect();

        // DFT and bit-reverse to match FRI's expectation
        let mut full_coeffs = coeffs;
        full_coeffs.resize(num_evals, EF::ZERO);
        let dft = p3_dft::Radix2DFTSmallBatch::<EF>::default();
        let mut evals = p3_dft::TwoAdicSubgroupDft::dft_algebra(&dft, full_coeffs);
        reverse_slice_index_bits(&mut evals);

        let beta: EF = rng.sample(StandardUniform);

        // Our implementation
        let g_inv = F::two_adic_generator(log_num_cosets + log_arity).inverse();
        let mut s_invs: Vec<F> = g_inv.powers().take(num_cosets).collect();
        reverse_slice_index_bits(&mut s_invs);

        let matrix = RowMajorMatrix::new(evals.clone(), arity);
        let my_folded = FriFold2::fold_matrix::<F, EF>(matrix.as_view(), &s_invs, beta);

        // Reference implementation from two_adic_pcs.rs
        let ref_g_inv = F::two_adic_generator(log_num_cosets + 1).inverse();
        let mut ref_halve_inv_powers: Vec<F> = ref_g_inv
            .shifted_powers(F::ONE.halve())
            .take(num_cosets)
            .collect();
        reverse_slice_index_bits(&mut ref_halve_inv_powers);

        let ref_folded: Vec<EF> = evals
            .chunks(arity)
            .zip(ref_halve_inv_powers.iter())
            .map(|(row, &halve_inv_power)| {
                let lo = row[0];
                let hi = row[1];
                (lo + hi).halve() + (lo - hi) * beta * halve_inv_power
            })
            .collect();

        // Assert all values match
        assert_eq!(my_folded.len(), ref_folded.len());
        for (my_val, ref_val) in my_folded.iter().zip(ref_folded.iter()) {
            assert_eq!(my_val, ref_val, "Folded value mismatch");
        }
    }

    /// Test that FRI folding preserves the low-degree structure.
    ///
    /// After folding a degree-d polynomial, the result should have degree d/arity.
    /// This test verifies by checking that high coefficients are zero after IDFT.
    #[test]
    fn test_folding_preserves_low_degree() {
        let mut rng = SmallRng::seed_from_u64(42);

        let log_blowup = 2;
        let log_poly_degree = 4; // degree 16 polynomial
        let poly_degree = 1 << log_poly_degree;
        let log_lde_size = log_poly_degree + log_blowup;
        let lde_size = 1 << log_lde_size;
        let log_arity = 1; // arity 2
        let arity = 1 << log_arity;

        // Generate random low-degree polynomial
        let coeffs: Vec<EF> = (0..poly_degree)
            .map(|_| rng.sample(StandardUniform))
            .collect();

        // Compute LDE in bit-reversed order
        let mut full_coeffs = coeffs;
        full_coeffs.resize(lde_size, EF::ZERO);
        let dft = p3_dft::Radix2DFTSmallBatch::<EF>::default();
        let mut evals = p3_dft::TwoAdicSubgroupDft::dft_algebra(&dft, full_coeffs);
        reverse_slice_index_bits(&mut evals);

        // Compute s_invs
        let log_num_cosets = log_lde_size - log_arity;
        let num_cosets = 1 << log_num_cosets;
        let g_inv = F::two_adic_generator(log_lde_size).inverse();
        let mut s_invs: Vec<F> = g_inv.powers().take(num_cosets).collect();
        reverse_slice_index_bits(&mut s_invs);

        // Fold with random beta
        let beta: EF = rng.sample(StandardUniform);
        let matrix = RowMajorMatrix::new(evals, arity);
        let folded = FriFold2::fold_matrix::<F, EF>(matrix.as_view(), &s_invs, beta);

        // IDFT the result to get coefficients
        let mut folded_for_idft = folded;
        reverse_slice_index_bits(&mut folded_for_idft);
        let folded_coeffs = p3_dft::TwoAdicSubgroupDft::idft_algebra(&dft, folded_for_idft);

        // Check that all coefficients beyond degree/arity are zero
        let expected_degree = poly_degree / arity;
        for (i, coeff) in folded_coeffs.iter().enumerate().skip(expected_degree) {
            assert_eq!(
                *coeff,
                EF::ZERO,
                "High coefficient c[{i}] should be zero but was {:?}",
                coeff
            );
        }
    }
}
