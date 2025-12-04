//! ## Purpose
//!
//! Precompute data to evaluate polynomials at a single extension‑field point `z` using
//! barycentric weights, and to reuse single‑point quotients needed for the DEEP polynomial.
//!
//! ## Setting and Notation
//! - Work with functions `f: H → F`, where `H` is a smooth subgroup of order `d`.
//! - Use a larger domain `gK` with `|K| = b × |H|` (blowup factor `b`), so `H = K^b`.
//! - For subgroups `H^r ≤ H`, extend functions to `(gK)^r` and “lift” them to the common top
//!   domain `H` via
//!   `( f(X) : H^r → F ) ↦ ( f′(X) = f(X^r) : H → F )`.
//! - We evaluate all lifted functions `f′` at `z`, so we want an O(domain) method to compute
//!   `f(z^r)` for the smaller domain size.
//!
//! ## DEEP Quotient
//! - Precompute the single‑point quotient `q(X) = 1/(z − X)` over `gK`.
//! - This supports efficient computation of the DEEP quotient `(f(z) − f(X)) / (z − X)` by
//!   point‑wise multiplication.
//! - These values are reused when constructing barycentric evaluation weights.
//!
//! ## Barycentric Evaluation on `gH`
//! For a smooth coset `gH` of order `d`, a polynomial `f` with `deg f < d` can be evaluated
//! in `O(d)` time from its samples on `gH`, using precomputed weights.
//!
//! Definitions
//! ```text
//! V_H(X) = X^d − 1                                    (vanishing polynomial of H)
//! L_{H,i}(X) = (V_H(X)/d) ⋅ h_i / (X − h_i)           (Lagrange basis on H)
//! L_{H,i}(X/g) = ((X/g)^d − 1)/d ⋅ h_i / (X/g − h_i)
//! L_{gH,i}(X) = V_{gH}(X)/d ⋅ (g h_i)/(X − g h_i)
//! ```
//!
//! We use
//! ```text
//! w_{gH,i}(z) = (g h_i)/(z − g h_i)   (unscaled weights)
//! s_{gH}(z)   = V_{gH}(z)/d           (global scale)
//! ```
//! and the evaluation identity
//! ```text
//! f(z) = s_{gH}(z) ⋅ Σ_{i<d} w_{gH,i}(z) ⋅ f(g h_i).
//! ```
//!
//! ## Lifting (Squaring Step)
//! Let `d` be even and `H = g⟨h⟩` a smooth coset of the same order, ordered in bit‑reversed
//! order. For all `0 ≤ i < d/2` we assume:
//! - `H[2i+1] = −H[2i]`
//! - `H^2[i] = H[2i]^2`
//!
//! Given `f: H^2 → F`, lift to `f′: H → F` with `f′(X) = f(X^2)`. Using the weights over `H`,
//! the barycentric formula gives
//! ```text
//! f(z^2) = f′(z)
//!        = s_H(z) ⋅ Σ_{i<d/2} [ w_{H,2i}(z) ⋅ f′(H[2i]) + w_{H,2i+1}(z) ⋅ f′(H[2i+1]) ].
//! ```
//!
//! Since `f′` is even and using the bit‑reversed order of `H`:
//! - `f′(H[2i])   = f(H[2i]^2) = f(H^2[i])`
//! - `f′(H[2i+1]) = f′(−H[2i]) = f((−H[2i])^2) = f(H^2[i])`
//!
//! Taking `w_{H^2,i}(z) = w_{H,2i}(z) + w_{H,2i+1}(z)`, we obtain
//! ```text
//! f(z^2) = s_H(z) ⋅ Σ_{i<d/2} w_{H^2,i}(z) ⋅ f(H^2[i]).
//! ```
//!
//! ## Consistency Check
//! Using the explicit formulas for the weights over the coset `gH`:
//! ```text
//! w_{gH,2i}(X) + w_{gH,2i+1}(X)
//!   = gH[2i]/(X − gH[2i]) + gH[2i+1]/(X − gH[2i+1])
//!   = 2 ⋅ (gH[2i])^2 / (X^2 − (gH[2i])^2)
//!   = 2 ⋅ w_{(gH)^2,i}(X)
//!
//! s_{(gH)^2}(X^2)
//!   = ((X^2/g^2)^{d/2} − 1)/(d/2)
//!   = ((X/g)^d − 1)/(d/2)
//!   = s_H(X)/2
//! ```

use alloc::vec::Vec;
use core::iter;
use core::iter::zip;
use core::marker::PhantomData;

use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{
    ExtensionField, Field, PackedFieldExtension, PackedValue, TwoAdicField,
    batch_multiplicative_inverse, scale_slice_in_place_single_core,
};
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_util::linear_map::LinearMap;
use p3_util::{log2_strict_usize, reverse_slice_index_bits};

/// Precomputed barycentric/DEEP data for evaluating polynomials at `z`.
///
/// For each degree bound `d' ≤ d` (powers of two) and polynomial `f` of degree `d'`,
/// with `r = d/d'`, this enables evaluating `f(z^r)` from evaluations `f((gK)^r)`.
/// Also supports computing the DEEP quotient `(f(z^r) - f(X^r)) / (z - X)` over `gK`.
pub struct Precomputation<F: TwoAdicField, EF: ExtensionField<F>> {
    log_blowup: usize,
    // evaluations of 1/(z-X) over gK
    point_quotient: Vec<EF>,

    // s_{D}(z)
    // note that this scaling is valid for all domains since s_{D^2}(z^2) = s(z)
    barycentric_scaling: EF,

    // for each key log_d, gives the weight for evaluating a polynomial of degree < 2^log_d,
    // let r = d_max/d
    // at f(z^r) provided f((gH)^r)
    // w_{D,i}(z)
    barycentric_weights: LinearMap<usize, Vec<EF>>,
    _marker: PhantomData<F>,
}

impl<F: TwoAdicField, EF: ExtensionField<F>> Precomputation<F, EF> {
    /// Create precomputation for point `z` with max degree `d` and blowup `log_blowup`.
    ///
    /// The evaluation domain has order `n = d << log_blowup`.
    pub fn new(point: EF, d: usize, log_blowup: usize) -> Self {
        let log_d = log2_strict_usize(d);
        let n = d << log_blowup;
        let log_n = log2_strict_usize(n);

        // Coset gK in bit-reversed order. This ensures that
        // - gK[2i+1] = -gK[2i]
        // - gH = gK[..d]
        let coset = TwoAdicMultiplicativeCoset::new(F::GENERATOR, log_n).unwrap();
        let coset_points = {
            let mut pts: Vec<F> = coset.iter().collect();
            reverse_slice_index_bits(&mut pts);
            pts
        };

        // V_{H}(X/g) / d = ( (X/g)^d - 1)/d
        let barycentric_scaling = {
            let z_over_shift = point * coset.shift_inverse();
            let t = z_over_shift.exp_power_of_2(log_d) - EF::ONE;
            t.div_2exp_u64(log_d as u64)
        };

        // q(X) = 1 / (z - X) evaluated over gK
        let point_quotient = {
            let diffs: Vec<EF> = coset_points.par_iter().map(|&x| point - x).collect();
            batch_multiplicative_inverse(&diffs)
        };

        // w_{gH, i}(z) = w_{H,i}(z/g) = h^i / ( (z/g) - h^i ) = gh^i / (z - gh^i)
        // Reuse the point quotient evaluations since gH = gK[..d] in bit-reversed order.
        let top_weights: Vec<EF> = coset_points[..d]
            .par_iter()
            .zip(point_quotient[..d].par_iter())
            .map(|(&k, &inv)| inv * k)
            .collect();

        // For each degree bound d' < d (powers of 2), let r = d/d' be the shrinking factor
        // get the weights for evaluating f(z^r) over (gH)^r
        let barycentric_weights: LinearMap<usize, Vec<EF>> =
            iter::successors(Some((log_d, top_weights)), |(prev_log_d, prev_weights)| {
                if prev_weights.len() == 1 {
                    None
                } else {
                    let new_weights = prev_weights
                        .par_chunks_exact(2)
                        .map(|pair| pair[0] + pair[1])
                        .collect();
                    Some((*prev_log_d - 1, new_weights))
                }
            })
            .collect();

        Self {
            log_blowup,
            point_quotient,
            barycentric_scaling,
            barycentric_weights,
            _marker: Default::default(),
        }
    }

    /// Evaluate polynomials at `z` from their coset evaluations.
    ///
    /// Given a matrix with columns `[f_1(gH), ..., f_m(gH)]`, returns `[f_1(z), ..., f_m(z)]`.
    pub fn eval_matrix<M: Matrix<F>>(&self, m: &M) -> Vec<EF> {
        let log_d = log2_strict_usize(m.height()) - self.log_blowup;
        let d = 1 << log_d;
        let weights = &self.barycentric_weights[&log_d];
        let mut evals = m.columnwise_dot_product(&weights[..d]);
        scale_slice_in_place_single_core(&mut evals, self.barycentric_scaling);
        evals
    }

    /// Accumulate the DEEP quotient into `acc` in-place.
    ///
    /// For each row `i`, adds `(eval_reduced + neg_reduced_matrix[i]) * point_quotient[i]` to `acc[i]`,
    /// where `neg_reduced_matrix = -f_reduced` (pre-computed with negated coefficients) and
    /// `eval_reduced = reduce_evals(evals_groups, coeffs_groups)` is a single scalar.
    ///
    /// Requires `n >= WIDTH` and `n % WIDTH == 0`.
    pub fn accumulate_deep_quotient(
        &self,
        acc: &mut [EF],
        neg_reduced_matrix: &[EF],
        evals_groups: &[Vec<Vec<EF>>],
        coeffs_groups: &[Vec<Vec<EF>>],
    ) {
        let n = acc.len();
        debug_assert_eq!(neg_reduced_matrix.len(), n);
        debug_assert_eq!(self.point_quotient.len(), n);

        let w = F::Packing::WIDTH;
        debug_assert!(
            n >= w && n.is_multiple_of(w),
            "accumulate_deep_quotient requires n >= WIDTH and aligned"
        );

        let eval_reduced: EF = reduce_evals(evals_groups, coeffs_groups);
        let eval_reduced_p = EF::ExtensionPacking::from(eval_reduced);

        acc.par_chunks_exact_mut(w)
            .zip(neg_reduced_matrix.par_chunks_exact(w))
            .zip(self.point_quotient[..n].par_chunks_exact(w))
            .for_each(|((acc_chunk, neg_chunk), q_chunk)| {
                let acc_p = EF::ExtensionPacking::from_ext_slice(acc_chunk);
                let neg_p = EF::ExtensionPacking::from_ext_slice(neg_chunk);
                let q_p = EF::ExtensionPacking::from_ext_slice(q_chunk);

                let res_p = acc_p + q_p * (neg_p + eval_reduced_p);
                res_p.to_ext_slice(acc_chunk);
            });
    }
}

/// Accumulate weighted matrix rows with upsampling for height differences.
///
/// Matrices must be sorted by height (ascending, powers of two). For each matrix,
/// computes the dot product of each row with its coefficients and adds to an accumulator.
/// When height increases, the accumulator is upsampled by repeating each entry.
///
/// Requires all matrix heights to be `>= WIDTH` and divisible by `WIDTH`.
pub fn accumulate_matrices<F: Field, EF: ExtensionField<F>, M: Matrix<F>>(
    matrices: &[M],
    coeffs: &[Vec<EF>],
) -> Vec<EF> {
    let n = matrices.last().unwrap().height();

    let mut acc = EF::zero_vec(n);
    let mut scratch = EF::zero_vec(n);

    let mut active_height = matrices.first().unwrap().height();

    for (matrix, coeffs) in zip(matrices, coeffs) {
        let height = matrix.height();
        debug_assert!(
            height.is_power_of_two(),
            "matrix height must be a power of two"
        );

        // Upsample if height increased (repeat each entry scaling_factor times)
        // E.g., [a, b] with scaling=2 → [a, a, b, b]
        if height > active_height {
            let scaling_factor = height / active_height;
            scratch[..height]
                .par_chunks_mut(scaling_factor)
                .zip(acc[..active_height].par_iter())
                .for_each(|(chunk, &val)| chunk.fill(val));
            acc[..height].swap_with_slice(&mut scratch[..height]);
        }

        // SIMD path using horizontal packing
        // Pack coefficients: group WIDTH coefficients into each ExtensionPacking
        let w = F::Packing::WIDTH;
        let packed_coeffs: Vec<EF::ExtensionPacking> = coeffs
            .chunks(w)
            .map(|chunk| {
                if chunk.len() == w {
                    EF::ExtensionPacking::from_ext_slice(chunk)
                } else {
                    // Pad with zeros for the last chunk
                    let mut padded = EF::zero_vec(w);
                    padded[..chunk.len()].copy_from_slice(chunk);
                    EF::ExtensionPacking::from_ext_slice(&padded)
                }
            })
            .collect();

        matrix
            .rowwise_packed_dot_product::<EF>(&packed_coeffs)
            .zip(acc[..height].par_iter_mut())
            .for_each(|(dot_result, acc_val)| {
                *acc_val += dot_result;
            });

        active_height = height;
    }

    acc
}

/// Compute the reduced matrix by accumulating all matrices with negated coefficients.
/// Returns `-f_reduced` where f_reduced = Σ coeff_i * matrix_i(row).
///
/// Requires `n >= WIDTH` and `n % WIDTH == 0`.
pub fn reduce_matrices<F: Field, EF: ExtensionField<F>, M: Matrix<F>>(
    matrices_groups: &[Vec<M>],
    coeffs_groups: &[Vec<Vec<EF>>],
    n: usize,
) -> Vec<EF> {
    // Negate all coefficients
    let neg_coeffs_groups: Vec<Vec<Vec<EF>>> = coeffs_groups
        .iter()
        .map(|group| {
            group
                .iter()
                .map(|coeffs| coeffs.iter().copied().map(EF::neg).collect())
                .collect()
        })
        .collect();

    let w = F::Packing::WIDTH;
    zip(matrices_groups, &neg_coeffs_groups)
        .map(|(matrices_group, coeffs_group)| accumulate_matrices(matrices_group, coeffs_group))
        .reduce(|mut acc, next| {
            debug_assert_eq!(acc.len(), next.len());
            acc.par_chunks_mut(w)
                .zip(next.par_chunks(w))
                .for_each(|(acc_chunk, next_chunk)| {
                    EF::add_slices(acc_chunk, next_chunk);
                });
            acc
        })
        .unwrap_or_else(|| EF::zero_vec(n))
}

/// Compute the reduced evals by summing Σ coeff_i * eval_i across all groups/matrices/columns.
pub fn reduce_evals<EF: Field>(
    evals_groups: &[Vec<Vec<EF>>],
    coeffs_groups: &[Vec<Vec<EF>>],
) -> EF {
    zip(evals_groups, coeffs_groups)
        .flat_map(|(evals_group, coeffs_group)| {
            zip(evals_group, coeffs_group)
                .flat_map(|(evals, coeffs)| zip(evals, coeffs).map(|(&e, &c)| e * c))
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear as F;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_matrix::dense::RowMajorMatrix;
    use rand::distr::StandardUniform;
    use rand::prelude::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::*;
    type EF = BinomialExtensionField<F, 4>;

    // Evaluate polynomial with base-field coefficients at an extension-field point via Horner.
    fn horner_ext<F: Field, EF: ExtensionField<F>>(coeffs: &[F], x: EF) -> EF {
        coeffs
            .iter()
            .copied()
            .rev()
            .fold(EF::ZERO, |acc, c| acc * x + c)
    }

    // Evaluate polynomial with base-field coefficients at an extension-field point via Horner.
    fn horner_base<F: Field>(coeffs: &[F], x: F) -> F {
        coeffs
            .iter()
            .copied()
            .rev()
            .fold(F::ZERO, |acc, c| acc * x + c)
    }

    #[test]
    fn check_quotient() {
        let rng = &mut SmallRng::seed_from_u64(1);

        let z: EF = rng.sample(StandardUniform);
        // Domain sizes
        let d: usize = 16;
        let log_blowup: usize = 2;
        let n = d << log_blowup;
        let log_n = log2_strict_usize(n);

        // Build precomputation at z.
        let pre = Precomputation::<F, EF>::new(z, d, log_blowup);

        // Reconstruct gK in the same bit-reversed order as Precomputation.
        let gk = TwoAdicMultiplicativeCoset::new(F::GENERATOR, log_n).unwrap();
        let mut gk_points: Vec<F> = gk.iter().collect();
        reverse_slice_index_bits(&mut gk_points);

        // Verify the point quotient q(X) = 1/(z - X) over gK.
        assert_eq!(gk_points.len(), pre.point_quotient.len());
        for (x, &q) in gk_points.iter().zip(pre.point_quotient.iter()) {
            let expected = (z - *x).inverse();
            assert_eq!(q, expected, "q(x) mismatch at x={:?}", x);
        }
    }

    #[test]
    fn deep_precomputation_matches_horner() {
        let rng = &mut SmallRng::seed_from_u64(1);

        // Domain sizes
        let d: usize = 64;
        let log_blowup: usize = 4;

        let n: usize = d << log_blowup;
        let log_n = log2_strict_usize(n);

        let z: EF = rng.sample(StandardUniform);

        // Build precomputation at z.
        let pre = Precomputation::<F, EF>::new(z, d, log_blowup);

        // Reconstruct gK in the same bit-reversed order as Precomputation.
        let gk = TwoAdicMultiplicativeCoset::new(F::GENERATOR, log_n).unwrap();
        let mut gk_points: Vec<F> = gk.iter().collect();
        reverse_slice_index_bits(&mut gk_points);

        for log_scaling in [0, 1, 2] {
            let d_lift = d >> log_scaling;

            let gk_lift = gk.exp_power_of_2(log_scaling).unwrap();

            let mut gk_lift_points = gk_lift.iter().collect();
            reverse_slice_index_bits(&mut gk_lift_points);
            let poly: Vec<F> = rng.sample_iter(StandardUniform).take(d_lift).collect();

            let z_lift = z.exp_power_of_2(log_scaling);

            let evals_gk: Vec<_> = gk_lift_points
                .iter()
                .map(|pt| horner_base(&poly, *pt))
                .collect();
            let evals_gk = RowMajorMatrix::new_col(evals_gk);

            let eval_expected = horner_ext(&poly, z_lift);

            let eval = pre.eval_matrix(&evals_gk)[0];
            assert_eq!(eval_expected, eval);
        }
    }

    /// End-to-end test for DEEP quotient accumulation.
    ///
    /// Creates 3 groups with [3, 3, 2] matrices of varying polynomial degrees,
    /// uses two evaluation points, and verifies the accumulated DEEP quotient
    /// matches the naive computation.
    #[test]
    fn deep_quotient_end_to_end() {
        let rng = &mut SmallRng::seed_from_u64(42);

        // Parameters: d_max = 64, blowup = 4, n = 256
        let log_d_max = 10;
        let d_max: usize = 1 << log_d_max;
        let log_blowup: usize = 2;
        let n = d_max << log_blowup;
        let log_n = log_d_max + log_blowup;

        // Two evaluation points
        let z1: EF = rng.sample(StandardUniform);
        let z2: EF = rng.sample(StandardUniform);

        // Coset gK in bit-reversed order
        let coset = TwoAdicMultiplicativeCoset::new(F::GENERATOR, log_n).unwrap();
        let mut domain: Vec<F> = coset.iter().collect();
        reverse_slice_index_bits(&mut domain);

        // Create 3 groups with [3, 3, 2] matrices
        // specs: (log_scaling, width) where degree = d_max >> log_scaling
        // Group 0: degrees 16, 32, 64 (log_scaling 2, 1, 0) with widths 2, 3, 4
        // Group 1: degrees 16, 32, 64 with widths 1, 2, 3
        // Group 2: degrees 32, 64 with widths 2, 3
        let group_specs: Vec<Vec<(usize, usize)>> = vec![
            vec![(2, 2), (1, 3), (0, 4)],
            vec![(2, 1), (1, 2), (0, 3)],
            vec![(1, 2), (0, 3)],
        ];

        // Store polynomial coefficients for naive verification
        struct PolyData {
            coeffs: Vec<F>,
        }

        let mut matrices_groups: Vec<Vec<RowMajorMatrix<F>>> = Vec::new();
        let mut coeffs_groups: Vec<Vec<Vec<EF>>> = Vec::new();
        let mut polys_data: Vec<Vec<Vec<PolyData>>> = Vec::new();

        for specs in &group_specs {
            let mut group_matrices = Vec::new();
            let mut group_coeffs = Vec::new();
            let mut group_polys = Vec::new();

            for &(log_scaling, width) in specs {
                let degree = d_max >> log_scaling;

                // Get the lifted coset for evaluation
                let lifted_coset = coset.exp_power_of_2(log_scaling).unwrap();
                let mut lifted_points: Vec<F> = lifted_coset.iter().collect();
                reverse_slice_index_bits(&mut lifted_points);
                let height = lifted_points.len();

                // Generate polynomials
                let mut matrix_polys = Vec::new();
                for _ in 0..width {
                    let poly: Vec<F> = rng.sample_iter(StandardUniform).take(degree).collect();
                    matrix_polys.push(PolyData { coeffs: poly });
                }

                // Build matrix in row-major order: for each domain point, evaluate all polynomials
                let mut matrix_data = Vec::with_capacity(height * width);
                for &pt in &lifted_points {
                    for poly_data in &matrix_polys {
                        matrix_data.push(horner_base(&poly_data.coeffs, pt));
                    }
                }

                group_matrices.push(RowMajorMatrix::new(matrix_data, width));
                group_coeffs.push(rng.sample_iter(StandardUniform).take(width).collect());
                group_polys.push(matrix_polys);
            }

            matrices_groups.push(group_matrices);
            coeffs_groups.push(group_coeffs);
            polys_data.push(group_polys);
        }

        // Create precomputations for both points
        let pre1 = Precomputation::<F, EF>::new(z1, d_max, log_blowup);
        let pre2 = Precomputation::<F, EF>::new(z2, d_max, log_blowup);

        // Compute polynomial evaluations at z1 and z2 using barycentric interpolation
        let evals_groups_1: Vec<Vec<Vec<EF>>> = matrices_groups
            .iter()
            .map(|group| group.iter().map(|m| pre1.eval_matrix(m)).collect())
            .collect();
        let evals_groups_2: Vec<Vec<Vec<EF>>> = matrices_groups
            .iter()
            .map(|group| group.iter().map(|m| pre2.eval_matrix(m)).collect())
            .collect();

        // Compute neg_reduced_matrix = -Σ coeff * f(X) with upsampling
        let neg_reduced = reduce_matrices(&matrices_groups, &coeffs_groups, n);

        // Accumulate DEEP quotients for both points
        let mut acc = EF::zero_vec(n);
        pre1.accumulate_deep_quotient(&mut acc, &neg_reduced, &evals_groups_1, &coeffs_groups);
        pre2.accumulate_deep_quotient(&mut acc, &neg_reduced, &evals_groups_2, &coeffs_groups);

        // Naive verification at each domain point
        for i in 0..n {
            let x = domain[i];
            let mut expected = EF::ZERO;

            for (z, evals_groups) in [(z1, &evals_groups_1), (z2, &evals_groups_2)] {
                let mut eval_at_z = EF::ZERO;
                let mut eval_at_x = EF::ZERO;

                for (g, specs) in group_specs.iter().enumerate() {
                    for (m, &(log_scaling, _)) in specs.iter().enumerate() {
                        let x_lifted: F = x.exp_u64(1u64 << log_scaling);

                        for (col, poly_data) in polys_data[g][m].iter().enumerate() {
                            let coeff = coeffs_groups[g][m][col];

                            // Evaluation at z from barycentric interpolation
                            eval_at_z += coeff * evals_groups[g][m][col];

                            // Evaluation at x (lifted)
                            let f_x = horner_base(&poly_data.coeffs, x_lifted);
                            eval_at_x += coeff * f_x;
                        }
                    }
                }

                // DEEP quotient: (f(z) - f(x)) / (z - x)
                expected += (eval_at_z - eval_at_x) / (z - x);
            }

            assert_eq!(acc[i], expected, "Mismatch at domain point {}", i);
        }
    }
}
