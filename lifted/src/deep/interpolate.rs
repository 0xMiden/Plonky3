//! # Barycentric Interpolation for Lifted Polynomials
//!
//! Evaluates `f(z)` from samples `{f(xᵢ)}` in O(d) time via the barycentric formula,
//! versus O(d²) for naive Lagrange interpolation.
//!
//! ## Barycentric Formula
//!
//! For a polynomial `f` of degree < d sampled on coset `gH` of order d:
//!
//! ```text
//! f(z) = s(z) · Σᵢ wᵢ(z) · f(gHᵢ)
//! ```
//!
//! where the **scaling factor** and **barycentric weights** are:
//!
//! ```text
//! s_{gH}(z) = V_{gH}(z) / d = ((z/g)^d - 1) / d
//! w_{gH,i}(z) = (gHᵢ) / (z - gHᵢ)
//! ```
//!
//! Here `V_{gH}(X) = (X/g)^d - 1` is the vanishing polynomial of coset `gH`.
//!
//! ## The Point Quotient
//!
//! Since `wᵢ(z) = xᵢ · 1/(z - xᵢ)`, precomputing `qᵢ = 1/(z - xᵢ)` via batch inversion
//! lets us both evaluate polynomials and construct DEEP quotients `(f(z) - f(X))/(z - X)`.
//! Montgomery's trick computes all n inverses with 3n multiplications + 1 inversion.
//!
//! ## Lifting and Weight Folding
//!
//! For polynomials of varying degrees, we "lift" smaller polynomials to the largest
//! domain. A degree-d' polynomial `f` lifts to `f'(X) = f(Xʳ)` on a domain of size
//! r·d'. To evaluate `f` at point `z`, we equivalently evaluate `f'` at `z^{1/r}`—
//! but since we want all evaluations at the *same* point z, we instead evaluate
//! `f(zʳ)`, which equals `f'(z)`.
//!
//! ### Bit-Reversed Domain Structure
//!
//! In bit-reversed order, the coset `gH` satisfies:
//! - **Adjacent negation**: `gH[2i+1] = -gH[2i]`
//! - **Squaring gives prefix**: `(gH[2i])² = (gH)²[i]`
//!
//! This means lifted polynomial `f(X²)` has the same value at indices `2i` and `2i+1`.
//!
//! ### Weight Folding Derivation
//!
//! For the squared domain, adjacent weights combine:
//!
//! ```text
//! w_{gH,2i}(z) + w_{gH,2i+1}(z)
//!   = gH[2i]/(z - gH[2i]) + gH[2i+1]/(z - gH[2i+1])
//!   = gH[2i]/(z - gH[2i]) + (-gH[2i])/(z + gH[2i])      [since gH[2i+1] = -gH[2i]]
//!   = gH[2i] · (z + gH[2i] - z + gH[2i]) / (z² - gH[2i]²)
//!   = 2·(gH[2i])² / (z² - (gH[2i])²)
//!   = 2 · w_{(gH)²,i}(z²)
//! ```
//!
//! The factor of 2 cancels with the scaling factor:
//!
//! ```text
//! s_{(gH)²}(z²) = ((z²/g²)^{d/2} - 1) / (d/2)
//!              = ((z/g)^d - 1) / (d/2)
//!              = 2 · s_{gH}(z)
//! ```
//!
//! Therefore: `s_{(gH)²}(z²) · w_{(gH)²,i}(z²) = s_{gH}(z) · [w_{gH,2i}(z) + w_{gH,2i+1}(z)]`
//!
//! This lets us fold weights iteratively: sum pairs to halve the domain size, with
//! the 2× factors canceling at each step.
//!
//! ### Uniform Evaluation via Lifting
//!
//! The key insight: to make all evaluations "look" uniform at point z:
//! - For a degree-d polynomial on full domain: evaluate at z directly
//! - For a degree-d' polynomial (d' < d) with lift factor r = d/d': evaluate at zʳ
//!
//! From the verifier's perspective, evaluating `f(zʳ)` is equivalent to evaluating
//! the lifted polynomial `f'(X) = f(Xʳ)` at z. This makes all polynomials appear
//! to live on the same domain, simplifying the DEEP quotient construction.

use alloc::collections::BTreeSet;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_field::{
    ExtensionField, TwoAdicField, batch_multiplicative_inverse, scale_slice_in_place_single_core,
};
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_util::linear_map::LinearMap;
use p3_util::log2_strict_usize;

use super::MatrixGroupEvals;

/// Precomputed `1/(z - xᵢ)` for all domain points, enabling O(n) barycentric
/// evaluation and DEEP quotient construction.
///
/// Batch inversion (Montgomery's trick) computes all n inverses with 3n muls + 1 inv,
/// amortized across all polynomials opened at this point.
pub struct SinglePointQuotient<F: TwoAdicField, EF: ExtensionField<F>> {
    /// The evaluation point `z`.
    point: EF,
    /// Evaluations of `1/(z-X)` over `gK`, used for both barycentric weights and DEEP quotients.
    point_quotient: Vec<EF>,
    _marker: PhantomData<F>,
}

impl<F: TwoAdicField, EF: ExtensionField<F>> SinglePointQuotient<F, EF> {
    /// Create precomputation for point `z`.
    ///
    /// Computes the point quotient `1/(z-X)` over the coset `gK`.
    /// Use [`Self::batch_eval_lifted`] to evaluate matrices at `z`.
    pub fn new(point: EF, coset_points: &[F]) -> Self {
        // q(X) = 1 / (z - X) evaluated over gK
        let point_quotient: Vec<EF> = {
            let diffs: Vec<EF> = coset_points.par_iter().map(|&x| point - x).collect();
            batch_multiplicative_inverse(&diffs)
        };

        Self {
            point,
            point_quotient,
            _marker: PhantomData,
        }
    }

    /// Returns the precomputed point quotients `1/(z - xᵢ)` for each domain point.
    ///
    /// Used both for barycentric interpolation weights and DEEP quotient construction.
    /// The domain points `xᵢ` are the coset `gK` in bit-reversed order.
    pub fn point_quotient(&self) -> &[EF] {
        &self.point_quotient
    }

    /// Evaluate all matrix columns at `zʳ` where `r = domain_size / matrix_height`.
    ///
    /// Exploits weight folding: since lifted polynomials repeat values in bit-reversed
    /// order, we sum weights for r consecutive indices rather than computing r× more
    /// dot products. Matrices must be sorted by ascending height to enable progressive
    /// folding from largest to smallest domain.
    pub fn batch_eval_lifted<M: Matrix<F>>(
        &self,
        matrices_groups: &[Vec<&M>],
        coset_points: &[F],
        log_blowup: usize,
    ) -> Vec<MatrixGroupEvals<EF>> {
        let n = coset_points.len();
        let d = n >> log_blowup;
        let log_d = log2_strict_usize(d);

        let shift = coset_points[0]; // g in bit-reversed order
        let shift_inverse = shift.inverse();

        // s(z) = ((z/g)^d - 1) / d
        let barycentric_scaling = {
            let z_over_shift = self.point * shift_inverse;
            let t = z_over_shift.exp_power_of_2(log_d) - EF::ONE;
            t.div_2exp_u64(log_d as u64)
        };

        let used_heights: BTreeSet<usize> = matrices_groups
            .iter()
            .flat_map(|g| g.iter().map(|m| m.height()))
            .collect();

        // wᵢ(z) = xᵢ / (z - xᵢ) = xᵢ · point_quotient[i]
        // For smaller domains, sum chunks (weight folding).
        let barycentric_weights: LinearMap<usize, Vec<EF>> = {
            let top_weights: Vec<EF> = coset_points[..d]
                .par_iter()
                .zip(self.point_quotient[..d].par_iter())
                .map(|(&k, &inv)| inv * k)
                .collect();

            let mut result = LinearMap::new();
            let mut current_weights = top_weights;

            // Descending order: progressively sum chunks to shrink weights
            for &target_height in used_heights.iter().rev() {
                let target_d = target_height >> log_blowup;
                let chunk_size = current_weights.len() / target_d;
                current_weights = current_weights
                    .par_chunks_exact(chunk_size)
                    .map(|chunk| chunk.iter().copied().sum())
                    .collect();
                result.insert(target_height, current_weights.clone());
            }
            result
        };

        // f(zʳ) = s(z) · Σ wᵢ(z) · f(xᵢ)
        matrices_groups
            .iter()
            .map(|group| {
                let evals = group
                    .iter()
                    .map(|m| {
                        let weights = &barycentric_weights[&m.height()];
                        let mut evals = m.columnwise_dot_product(weights);
                        scale_slice_in_place_single_core(&mut evals, barycentric_scaling);
                        evals
                    })
                    .collect();
                MatrixGroupEvals::new(evals)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_dft::{NaiveDft, TwoAdicSubgroupDft};
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_interpolation::{interpolate_coset, interpolate_coset_with_precomputation};
    use p3_matrix::bitrev::BitReversibleMatrix;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_util::reverse_slice_index_bits;
    use rand::distr::StandardUniform;
    use rand::prelude::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::SinglePointQuotient;
    use crate::tests::{EF, F};
    use crate::utils::bit_reversed_coset_points;

    /// Verify `batch_eval_lifted` matches `interpolate_coset` for various lift factors.
    #[test]
    fn batch_eval_matches_interpolate_coset() {
        let rng = &mut SmallRng::seed_from_u64(42);
        let log_blowup = 2;
        let log_n = 8; // Full LDE domain size = 256
        let n = 1 << log_n;
        let shift = F::GENERATOR;

        // Coset points in bit-reversed order for our barycentric evaluation
        let coset_points_br = bit_reversed_coset_points::<F>(log_n);

        // Random out-of-domain evaluation point
        let z: EF = rng.sample(StandardUniform);

        // Test multiple polynomial degrees
        for log_scaling in 0..=2 {
            // Polynomial degree (trace height before LDE)
            let poly_degree = (n >> log_blowup) >> log_scaling;
            // LDE evaluation count = poly_degree * blowup
            let lde_height = poly_degree << log_blowup;
            let width = 3;

            // For lifted polynomials, the coset becomes (gK)ʳ = gʳ · Kʳ
            // So the shift for the smaller coset is shiftʳ
            let lifted_shift = shift.exp_power_of_2(log_scaling);

            // Generate random polynomial coefficients and pad to LDE size
            let mut coeffs_values = RowMajorMatrix::<F>::rand(rng, poly_degree, width).values;
            coeffs_values.resize(lde_height * width, F::ZERO);
            let padded_coeffs = RowMajorMatrix::new(coeffs_values, width);

            // Compute evaluations on the lifted coset via DFT (standard order)
            let evals_std = NaiveDft.coset_dft_batch(padded_coeffs, lifted_shift);

            // Convert to bit-reversed order for our evaluation
            let evals_br = evals_std.clone().bit_reverse_rows();

            // Our method computes f(zʳ) where r = n / lde_height = 2^log_scaling
            let z_lifted = z.exp_power_of_2(log_scaling);

            // Our barycentric evaluation
            let quotient = SinglePointQuotient::<F, EF>::new(z, &coset_points_br);
            let result =
                quotient.batch_eval_lifted(&[vec![&evals_br]], &coset_points_br, log_blowup);
            let our_evals = &result[0].0[0];

            // Standard interpolation on the lifted coset
            let expected_evals = interpolate_coset(&evals_std, lifted_shift, z_lifted);

            assert_eq!(
                our_evals.len(),
                expected_evals.len(),
                "log_scaling={log_scaling}: length mismatch"
            );
            for (col, (our, expected)) in our_evals.iter().zip(expected_evals.iter()).enumerate() {
                assert_eq!(
                    our, expected,
                    "log_scaling={log_scaling}, col={col}: evaluation mismatch"
                );
            }
        }
    }

    /// Verify `batch_eval_lifted` matches `interpolate_coset_with_precomputation`.
    #[test]
    fn batch_eval_matches_interpolate_with_precomputation() {
        let rng = &mut SmallRng::seed_from_u64(123);
        let log_blowup = 2;
        let log_n = 8;
        let n = 1 << log_n;
        let shift = F::GENERATOR;

        // Coset points in both orderings
        let coset_points_br = bit_reversed_coset_points::<F>(log_n);
        let mut coset_points_std = coset_points_br.clone();
        reverse_slice_index_bits(&mut coset_points_std); // Convert to standard order

        // Random out-of-domain evaluation point
        let z: EF = rng.sample(StandardUniform);

        // Create quotient for bit-reversed coset
        let quotient = SinglePointQuotient::<F, EF>::new(z, &coset_points_br);

        // Test polynomial with no lifting (log_scaling = 0, full LDE domain)
        let poly_degree = n >> log_blowup; // = 64
        let lde_height = n; // = 256, full LDE
        let width = 4;

        // Generate random polynomial coefficients and pad to LDE size
        let mut coeffs_values = RowMajorMatrix::<F>::rand(rng, poly_degree, width).values;
        coeffs_values.resize(lde_height * width, F::ZERO);
        let padded_coeffs = RowMajorMatrix::new(coeffs_values, width);

        // Compute evaluations on coset via DFT (standard order)
        let evals_std = NaiveDft.coset_dft_batch(padded_coeffs, shift);

        // Convert to bit-reversed order
        let evals_br = evals_std.clone().bit_reverse_rows();

        // Our barycentric evaluation (no lifting since lde_height = n)
        let result = quotient.batch_eval_lifted(&[vec![&evals_br]], &coset_points_br, log_blowup);
        let our_evals = &result[0].0[0];

        // Convert our diff_invs from bit-reversed to standard order for precomputation
        let mut diff_invs_std = quotient.point_quotient()[..lde_height].to_vec();
        reverse_slice_index_bits(&mut diff_invs_std);

        // Interpolation with precomputation (both in standard order)
        let expected_evals = interpolate_coset_with_precomputation(
            &evals_std,
            shift,
            z,
            &coset_points_std[..lde_height],
            &diff_invs_std,
        );

        assert_eq!(our_evals.len(), expected_evals.len(), "length mismatch");
        for (col, (our, expected)) in our_evals.iter().zip(expected_evals.iter()).enumerate() {
            assert_eq!(our, expected, "col={col}: evaluation mismatch");
        }
    }
}
