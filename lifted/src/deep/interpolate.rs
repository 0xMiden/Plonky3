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
    ExtensionField, FieldArray, TwoAdicField, batch_multiplicative_inverse,
    scale_slice_in_place_single_core,
};
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_util::linear_map::LinearMap;
use p3_util::{flatten_to_base, log2_strict_usize, reconstitute_from_base};

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

/// Precomputed `1/(zⱼ - xᵢ)` for N evaluation points, enabling batched O(n) barycentric
/// evaluation and DEEP quotient construction.
///
/// Unlike [`SinglePointQuotient`] which handles one point at a time, this struct batches
/// N evaluation points together, sharing a single batch inversion across all points.
/// This is more efficient when opening at multiple points (e.g., z₁ and z₂ in DEEP).
pub struct MultiPointQuotient<F: TwoAdicField, EF: ExtensionField<F>, const N: usize> {
    /// The evaluation points `[z₀, z₁, ..., z_{N-1}]`.
    points: FieldArray<EF, N>,
    /// Batched quotients: `point_quotient[i][j] = 1/(zⱼ - xᵢ)` for domain point xᵢ and eval point zⱼ.
    point_quotient: Vec<FieldArray<EF, N>>,
    _marker: PhantomData<F>,
}

impl<F: TwoAdicField, EF: ExtensionField<F>, const N: usize> MultiPointQuotient<F, EF, N> {
    /// Create precomputation for N evaluation points.
    ///
    /// Computes `1/(zⱼ - xᵢ)` for all domain points xᵢ and evaluation points zⱼ
    /// using a single batched inversion via Montgomery's trick.
    ///
    /// The differences are stored as `Vec<[EF; N]>`, flattened for batch inversion,
    /// then reconstituted back to the array layout.
    pub fn new(points: FieldArray<EF, N>, coset_points: &[F]) -> Self {
        let n_points = coset_points.len();

        // Compute differences in parallel: for each domain point x, compute [z₀ - x, z₁ - x, ...]
        let diffs: Vec<FieldArray<EF, N>> = coset_points
            .par_iter()
            .map(|&x| points.map(|z| z - x))
            .collect();

        // Flatten to Vec<EF> for batch inversion, invert, then reconstitute
        // SAFETY: [EF; N] has same alignment as EF and size is N * size_of::<EF>()
        let diffs_flat: Vec<EF> = unsafe { flatten_to_base(diffs) };
        let invs_flat: Vec<EF> = batch_multiplicative_inverse(&diffs_flat);
        let point_quotient: Vec<FieldArray<EF, N>> = unsafe { reconstitute_from_base(invs_flat) };

        debug_assert_eq!(point_quotient.len(), n_points);

        Self {
            points,
            point_quotient,
            _marker: PhantomData,
        }
    }

    /// Returns the evaluation points `[z₀, z₁, ..., z_{N-1}]`.
    pub const fn points(&self) -> &FieldArray<EF, N> {
        &self.points
    }

    /// Returns the precomputed point quotients `1/(zⱼ - xᵢ)` for each domain point.
    ///
    /// `point_quotient[i][j]` is `1/(zⱼ - xᵢ)` where xᵢ is the i-th domain point
    /// and zⱼ is the j-th evaluation point.
    pub fn point_quotient(&self) -> &[FieldArray<EF, N>] {
        &self.point_quotient
    }

    /// Evaluate all matrix columns at `[z₀ʳ, z₁ʳ, ..., z_{N-1}ʳ]` where `r = domain_size / matrix_height`.
    ///
    /// Returns evaluations grouped by commitment: `result[group_idx][matrix_idx][col_idx]`
    /// where each element is a `FieldArray<EF, N>` containing evaluations at all N points.
    /// This batches N evaluation points together, using `columnwise_dot_product_batched<N>`
    /// for better cache utilization than N separate calls.
    pub fn batch_eval_lifted<M: Matrix<F>>(
        &self,
        matrices_groups: &[Vec<&M>],
        coset_points: &[F],
        log_blowup: usize,
    ) -> Vec<MatrixGroupEvals<FieldArray<EF, N>>> {
        let n = coset_points.len();
        let d = n >> log_blowup;
        let log_d = log2_strict_usize(d);

        let shift = coset_points[0]; // g in bit-reversed order
        let shift_inverse = shift.inverse();

        // Compute barycentric scaling factors for each point:
        // s_j(z_j) = ((z_j/g)^d - 1) / d
        let barycentric_scalings = self.points.map(|point| {
            let z_over_shift = point * shift_inverse;
            let t = z_over_shift.exp_power_of_2(log_d) - EF::ONE;
            t.div_2exp_u64(log_d as u64)
        });

        let used_degrees: BTreeSet<usize> = matrices_groups
            .iter()
            .flat_map(|g| g.iter().map(|m| m.height() >> log_blowup))
            .collect();

        // Compute barycentric weights for each point at each height:
        // w_{i,j}(z_j) = x_i / (z_j - x_i) = x_i · point_quotient[i][j]
        // For smaller domains, sum chunks (weight folding).
        let barycentric_weights: LinearMap<usize, Vec<FieldArray<EF, N>>> = {
            assert_eq!(*used_degrees.last().unwrap(), d);
            // Initial weights at full domain size
            let top_weights: Vec<FieldArray<EF, N>> = coset_points[..d]
                .par_iter()
                .zip(self.point_quotient[..d].par_iter())
                .map(|(&x, invs)| (*invs).map(|inv| inv * x))
                .collect();

            let mut weights = Vec::with_capacity(used_degrees.len());
            weights.push(top_weights);

            // Descending order: progressively sum chunks to shrink weights
            for &next_degree in used_degrees.iter().rev().skip(1) {
                let prev_weights = weights.last().unwrap();
                let chunk_size = prev_weights.len() / next_degree;
                let next_weights = prev_weights
                    .par_chunks_exact(chunk_size)
                    .map(|chunk| chunk.iter().copied().sum())
                    .collect();
                weights.push(next_weights);
            }

            weights.into_iter().map(|w| (w.len(), w)).collect()
        };

        // f(z_j^r) = s_j(z_j) · Σ w_{i,j}(z_j) · f(x_i)
        // For each group, evaluate at all N points using columnwise_dot_product_batched
        // Returns Vec<[EF; N]> where result[col][point] = eval of column col at point point
        matrices_groups
            .iter()
            .map(|group| {
                // Evaluate each matrix in the group at all N points
                let group_evals: Vec<_> = group
                    .iter()
                    .map(|m| {
                        let weights = &barycentric_weights[&(m.height() >> log_blowup)];
                        let mut results = m.columnwise_dot_product_batched(weights);
                        for batch_evals in results.iter_mut() {
                            *batch_evals *= barycentric_scalings;
                        }
                        results
                    })
                    .collect();
                MatrixGroupEvals::new(group_evals)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_dft::{NaiveDft, TwoAdicSubgroupDft};
    use p3_field::{Field, FieldArray, PrimeCharacteristicRing};
    use p3_interpolation::{interpolate_coset, interpolate_coset_with_precomputation};
    use p3_matrix::bitrev::BitReversibleMatrix;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_util::reverse_slice_index_bits;
    use rand::distr::StandardUniform;
    use rand::prelude::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::{MultiPointQuotient, SinglePointQuotient};
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

    /// Verify `MultiPointQuotient` produces the same results as separate `SinglePointQuotient` calls.
    #[test]
    fn multi_point_matches_single_point() {
        use alloc::vec::Vec;

        use p3_matrix::Matrix;

        let rng = &mut SmallRng::seed_from_u64(999);
        let log_blowup = 2;
        let log_n = 8;
        let n = 1 << log_n;
        let shift = F::GENERATOR;

        // Coset points in bit-reversed order
        let coset_points_br = bit_reversed_coset_points::<F>(log_n);

        // Two random out-of-domain evaluation points
        let z1: EF = rng.sample(StandardUniform);
        let z2: EF = rng.sample(StandardUniform);

        // Generate test matrices of varying heights
        let poly_degree_1 = n >> log_blowup; // Full size
        let poly_degree_2 = poly_degree_1 >> 1; // Half size
        let width = 3;

        let lifted_shift_1 = shift;
        let lifted_shift_2 = shift.square();

        let mut coeffs1 = RowMajorMatrix::<F>::rand(rng, poly_degree_1, width).values;
        coeffs1.resize(n * width, F::ZERO);
        let evals1_std =
            NaiveDft.coset_dft_batch(RowMajorMatrix::new(coeffs1, width), lifted_shift_1);
        let evals1_br: RowMajorMatrix<F> = evals1_std.bit_reverse_rows().to_row_major_matrix();

        let mut coeffs2 = RowMajorMatrix::<F>::rand(rng, poly_degree_2, width).values;
        coeffs2.resize((n >> 1) * width, F::ZERO);
        let evals2_std =
            NaiveDft.coset_dft_batch(RowMajorMatrix::new(coeffs2, width), lifted_shift_2);
        let evals2_br: RowMajorMatrix<F> = evals2_std.bit_reverse_rows().to_row_major_matrix();

        let matrices_groups: Vec<Vec<&RowMajorMatrix<F>>> = vec![vec![&evals1_br, &evals2_br]];

        // --- Single-point evaluation (baseline) ---
        let sq1 = SinglePointQuotient::<F, EF>::new(z1, &coset_points_br);
        let sq2 = SinglePointQuotient::<F, EF>::new(z2, &coset_points_br);
        let single_evals1 = sq1.batch_eval_lifted(&matrices_groups, &coset_points_br, log_blowup);
        let single_evals2 = sq2.batch_eval_lifted(&matrices_groups, &coset_points_br, log_blowup);

        // --- Multi-point evaluation ---
        let mq = MultiPointQuotient::<F, EF, 2>::new(FieldArray([z1, z2]), &coset_points_br);
        let multi_evals = mq.batch_eval_lifted(&matrices_groups, &coset_points_br, log_blowup);

        // Verify point_quotient matches
        for (i, (sq1_q, sq2_q)) in sq1
            .point_quotient()
            .iter()
            .zip(sq2.point_quotient().iter())
            .enumerate()
        {
            let mq_q = &mq.point_quotient()[i];
            assert_eq!(*sq1_q, mq_q[0], "point_quotient mismatch at {i} for z1");
            assert_eq!(*sq2_q, mq_q[1], "point_quotient mismatch at {i} for z2");
        }

        // Verify batch_eval_lifted results match
        // multi_evals[group][point] vs single_evals[point][group]
        for (group_idx, multi_group) in multi_evals.iter().enumerate() {
            // multi_group is [MatrixGroupEvals; 2] - one per point
            let single_group1 = &single_evals1[group_idx];
            let single_group2 = &single_evals2[group_idx];

            // Compare point 0 (z1)
            for (matrix_idx, (multi_mat, single_mat)) in
                multi_group.0.iter().zip(single_group1.0.iter()).enumerate()
            {
                assert_eq!(
                    multi_mat.len(),
                    single_mat.len(),
                    "length mismatch for z1 at group {group_idx}, matrix {matrix_idx}"
                );
                for (col, (m, s)) in multi_mat.iter().zip(single_mat.iter()).enumerate() {
                    assert_eq!(
                        m[0], *s,
                        "mismatch at group {group_idx}, matrix {matrix_idx}, col {col} for z1"
                    );
                }
            }

            // Compare point 1 (z2)
            for (matrix_idx, (multi_mat, single_mat)) in
                multi_group.0.iter().zip(single_group2.0.iter()).enumerate()
            {
                assert_eq!(
                    multi_mat.len(),
                    single_mat.len(),
                    "length mismatch for z2 at group {group_idx}, matrix {matrix_idx}"
                );
                for (col, (m, s)) in multi_mat.iter().zip(single_mat.iter()).enumerate() {
                    assert_eq!(
                        m[1], *s,
                        "mismatch at group {group_idx}, matrix {matrix_idx}, col {col} for z2"
                    );
                }
            }
        }
    }
}
