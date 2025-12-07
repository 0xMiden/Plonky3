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

use alloc::collections::BTreeSet;
use alloc::vec::Vec;
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
    /// log2 of the blowup factor.
    log_blowup: usize,
    /// Evaluations of 1/(z-X) over gK.
    point_quotient: Vec<EF>,
    /// Evaluated matrices: evals_groups[g][m][col] = f_{g,m,col}(z^r).
    evals_groups: Vec<Vec<Vec<EF>>>,
    _marker: PhantomData<F>,
}

impl<F: TwoAdicField, EF: ExtensionField<F>> Precomputation<F, EF> {
    /// Create precomputation for point `z`, evaluating all matrices.
    ///
    /// Computes the point quotient 1/(z-X) over gK, and evaluates all matrices at z
    /// using barycentric interpolation. The barycentric weights are computed only
    /// for heights that appear in the matrices.
    pub fn new<M: Matrix<F>>(point: EF, matrices_groups: &[Vec<M>], log_blowup: usize) -> Self {
        // Determine n from largest matrix height
        let n = matrices_groups
            .iter()
            .flat_map(|g| g.iter().map(|m| m.height()))
            .max()
            .expect("matrices_groups must not be empty");
        let d = n >> log_blowup;
        let log_d = log2_strict_usize(d);
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

        // Collect unique heights that need weights (keyed by matrix height for easy lookup)
        let used_heights: BTreeSet<usize> = matrices_groups
            .iter()
            .flat_map(|g| g.iter().map(|m| m.height()))
            .collect();

        // Compute barycentric weights for each used height.
        // w_{gH, i}(z) = w_{H,i}(z/g) = h^i / ( (z/g) - h^i ) = gh^i / (z - gh^i)
        // Reuse the point quotient evaluations since gH = gK[..d] in bit-reversed order.
        //
        // Note: Matrix height = d_m * blowup, but we only need d_m weights for degree d_m polynomials.
        let barycentric_weights: LinearMap<usize, Vec<EF>> = {
            let top_weights: Vec<EF> = coset_points[..d]
                .par_iter()
                .zip(point_quotient[..d].par_iter())
                .map(|(&k, &inv)| inv * k)
                .collect();

            let mut result = LinearMap::new();
            let mut current_weights = top_weights;

            // Iterate in descending order (BTreeSet iterates ascending, so reverse)
            for &target_height in used_heights.iter().rev() {
                let target_d = target_height >> log_blowup;
                // Shrink weights by summing chunks: w_{H^r,i}(z) = Σ_{j} w_{H,r*i+j}(z)
                let chunk_size = current_weights.len() / target_d;
                current_weights = current_weights
                    .par_chunks_exact(chunk_size)
                    .map(|chunk| chunk.iter().copied().sum())
                    .collect();
                result.insert(target_height, current_weights.clone());
            }
            result
        };

        // Evaluate all matrices at point using precomputed barycentric weights.
        let evals_groups: Vec<Vec<Vec<EF>>> = matrices_groups
            .iter()
            .map(|group| {
                group
                    .iter()
                    .map(|m| {
                        let weights = &barycentric_weights[&m.height()];
                        let mut evals = m.columnwise_dot_product(weights);
                        scale_slice_in_place_single_core(&mut evals, barycentric_scaling);
                        evals
                    })
                    .collect()
            })
            .collect();

        Self {
            log_blowup,
            point_quotient,
            evals_groups,
            _marker: Default::default(),
        }
    }

    /// Get the evaluated matrices: `evals()[g][m][col] = f_{g,m,col}(z^r)`.
    pub fn evals(&self) -> &[Vec<Vec<EF>>] {
        &self.evals_groups
    }

    /// Compute DEEP quotient for multiple evaluation points.
    ///
    /// Given polynomials `f_i` (columns of matrices) and evaluation points `z_j`, computes:
    /// ```text
    /// Q(X) = Σ_i c_i · Σ_j (f_i(z_j) - f_i(X)) / (z_j - X)
    /// ```
    /// where `c_i = challenge^i` are the batching coefficients.
    ///
    /// The formula is rearranged for efficiency:
    /// ```text
    /// Q(X) = Σ_j (Σ_i c_i · f_i(z_j) - Σ_i c_i · f_i(X)) / (z_j - X)
    ///      = Σ_j (f_reduced(z_j) - f_reduced(X)) · q_j(X)
    /// ```
    /// where:
    /// - `f_reduced(X) = Σ_i c_i · f_i(X)` is the batched polynomial (computed once)
    /// - `q_j(X) = 1/(z_j - X)` is precomputed in each `Precomputation`
    pub fn compute_deep_quotient<M: Matrix<F>>(
        precomputations: &[Self],
        matrices_groups: &[Vec<M>],
        challenge: EF,
        padding: usize,
    ) -> Vec<EF> {
        assert!(
            !precomputations.is_empty(),
            "precomputations must not be empty"
        );

        let w = F::Packing::WIDTH;
        let n = precomputations[0].point_quotient.len();
        let log_blowup = precomputations[0].log_blowup;

        // Compute padded widths and group sizes for splitting coeffs later
        let group_sizes: Vec<usize> = matrices_groups.iter().map(|g| g.len()).collect();
        let padded_widths: Vec<usize> = matrices_groups
            .iter()
            .flat_map(|g| g.iter().map(|m| m.width().next_multiple_of(padding)))
            .collect();

        // Step 1: Derive batching coefficients c_i = challenge^i (flat)
        let coeffs: Vec<Vec<EF>> = derive_coeffs_from_challenge(&padded_widths, challenge);

        // Step 2: Compute -f_reduced(X) = -Σ_i c_i · f_i(X) over the domain
        // We negate here so that the inner loop computes: f_reduced(z_j) + (-f_reduced(X))
        let neg_coeffs: Vec<Vec<EF>> = coeffs
            .iter()
            .map(|c| c.iter().copied().map(EF::neg).collect())
            .collect();

        // Split neg_coeffs back into groups for accumulate_matrices
        let mut neg_coeffs_iter = neg_coeffs.iter();
        let neg_f_reduced = zip(matrices_groups, &group_sizes)
            .map(|(matrices_group, &size)| {
                let group_coeffs: Vec<&Vec<EF>> = neg_coeffs_iter.by_ref().take(size).collect();
                accumulate_matrices(matrices_group, &group_coeffs)
            })
            .reduce(|mut acc, next| {
                debug_assert_eq!(acc.len(), next.len());
                acc.par_chunks_mut(w).zip(next.par_chunks(w)).for_each(
                    |(acc_chunk, next_chunk)| {
                        EF::add_slices(acc_chunk, next_chunk);
                    },
                );
                acc
            })
            .unwrap_or_else(|| EF::zero_vec(n));

        // Step 3: For each evaluation point z_j, accumulate:
        //   Q(X) += (f_reduced(z_j) - f_reduced(X)) · q_j(X)
        //         = (f_reduced(z_j) + neg_f_reduced(X)) · q_j(X)
        let mut acc = EF::zero_vec(n);
        for pre in precomputations {
            debug_assert_eq!(pre.point_quotient.len(), n);
            debug_assert_eq!(pre.log_blowup, log_blowup);

            // f_reduced(z_j) = Σ_i c_i · f_i(z_j), a single scalar
            let evals_flat = pre.evals_groups.iter().flatten().map(|v| v.as_slice());
            let coeffs_flat = coeffs.iter().map(|v| v.as_slice());
            let f_reduced_at_z: EF = reduce_evals(evals_flat, coeffs_flat);
            let f_reduced_at_z_packed = EF::ExtensionPacking::from(f_reduced_at_z);

            // acc[k] += (f_reduced(z_j) + neg_f_reduced[k]) · q_j[k]
            acc.par_chunks_exact_mut(w)
                .zip(neg_f_reduced.par_chunks_exact(w))
                .zip(pre.point_quotient[..n].par_chunks_exact(w))
                .for_each(|((acc_chunk, neg_chunk), q_chunk)| {
                    let acc_p = EF::ExtensionPacking::from_ext_slice(acc_chunk);
                    let neg_p = EF::ExtensionPacking::from_ext_slice(neg_chunk);
                    let q_p = EF::ExtensionPacking::from_ext_slice(q_chunk);

                    let res_p = acc_p + q_p * (f_reduced_at_z_packed + neg_p);
                    res_p.to_ext_slice(acc_chunk);
                });
        }

        acc
    }
}

/// Derive batching coefficients from a single challenge using powers.
///
/// Returns coefficients in reverse order: `[..., c², c, 1]` for each width.
/// This ordering enables efficient recursive verification.
///
/// Takes padded widths directly. This accounts for matrices that were
/// committed with zero-padded columns for alignment.
pub fn derive_coeffs_from_challenge<EF: Field>(
    padded_widths: &[usize],
    challenge: EF,
) -> Vec<Vec<EF>> {
    // Compute total number of coefficients needed
    let total: usize = padded_widths.iter().sum();

    // Collect all powers at once, then reverse
    let all_powers: Vec<EF> = challenge.powers().take(total).collect();
    let mut rev_powers_iter = all_powers.into_iter().rev();

    // Assign reversed powers to each width
    padded_widths
        .iter()
        .map(|&width| rev_powers_iter.by_ref().take(width).collect())
        .collect()
}

/// Compute the reduced evals by summing Σ coeff_i * eval_i across all matrices/columns.
///
/// Evals can be over base field `F` while coeffs are over extension field `EF`.
/// This allows calling with matrix rows (over `F`) or precomputed evals (over `EF`).
pub fn reduce_evals<'a, F: Field, EF: ExtensionField<F>>(
    evals: impl IntoIterator<Item = &'a [F]>,
    coeffs: impl IntoIterator<Item = &'a [EF]>,
) -> EF {
    zip(evals, coeffs)
        .flat_map(|(evals, coeffs)| {
            debug_assert!(
                evals.len() <= coeffs.len(),
                "evals length {} exceeds coeffs length {}",
                evals.len(),
                coeffs.len()
            );
            zip(evals, coeffs).map(|(&e, &c)| c * e)
        })
        .sum()
}

/// Accumulate weighted matrix rows with upsampling for height differences.
///
/// Matrices must be sorted by height (ascending, powers of two). For each matrix,
/// computes the dot product of each row with its coefficients and adds to an accumulator.
/// When height increases, the accumulator is upsampled by repeating each entry.
///
/// Requires all matrix heights to be `>= WIDTH` and divisible by `WIDTH`.
fn accumulate_matrices<F: Field, EF: ExtensionField<F>, M: Matrix<F>, C: AsRef<[EF]>>(
    matrices: &[M],
    coeffs: &[C],
) -> Vec<EF> {
    let n = matrices.last().unwrap().height();

    let mut acc = EF::zero_vec(n);
    let mut scratch = EF::zero_vec(n);

    let mut active_height = matrices.first().unwrap().height();

    for (matrix, coeffs) in zip(matrices, coeffs) {
        let coeffs = coeffs.as_ref();
        let height = matrix.height();
        debug_assert!(
            height.is_power_of_two(),
            "matrix height must be a power of two"
        );
        debug_assert!(
            matrix.width() <= coeffs.len(),
            "matrix width {} exceeds coeffs length {}",
            matrix.width(),
            coeffs.len()
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

    // Evaluate polynomial with base-field coefficients at a base-field point via Horner.
    fn horner_base<F: Field>(coeffs: &[F], x: F) -> F {
        coeffs
            .iter()
            .copied()
            .rev()
            .fold(F::ZERO, |acc, c| acc * x + c)
    }

    #[test]
    fn check_point_quotient() {
        let rng = &mut SmallRng::seed_from_u64(1);

        let z: EF = rng.sample(StandardUniform);
        let d: usize = 16;
        let log_blowup: usize = 2;
        let n = d << log_blowup;
        let log_n = log2_strict_usize(n);

        // Create a dummy matrix to call new()
        let matrix = RowMajorMatrix::<F>::rand(rng, n, 1);
        let pre = Precomputation::<F, EF>::new(z, &[vec![matrix]], log_blowup);

        // Reconstruct gK in bit-reversed order
        let gk = TwoAdicMultiplicativeCoset::new(F::GENERATOR, log_n).unwrap();
        let mut gk_points: Vec<F> = gk.iter().collect();
        reverse_slice_index_bits(&mut gk_points);

        // Verify point quotient q(X) = 1/(z - X) over gK
        assert_eq!(gk_points.len(), pre.point_quotient.len());
        for (x, &q) in gk_points.iter().zip(pre.point_quotient.iter()) {
            let expected = (z - *x).inverse();
            assert_eq!(q, expected, "q(x) mismatch at x={x:?}");
        }
    }

    #[test]
    fn check_evals_match_horner() {
        let rng = &mut SmallRng::seed_from_u64(2);

        let d: usize = 64;
        let log_blowup: usize = 4;
        let n = d << log_blowup;
        let log_n = log2_strict_usize(n);

        let z: EF = rng.sample(StandardUniform);

        // Build coset gK
        let gk = TwoAdicMultiplicativeCoset::new(F::GENERATOR, log_n).unwrap();

        // Test multiple degrees (log_scaling = 0, 1, 2 means degree d, d/2, d/4)
        let specs: Vec<(usize, usize)> = vec![(0, 3), (1, 2), (2, 1)]; // (log_scaling, width)

        // Generate polynomials and build matrices
        let mut polys: Vec<Vec<Vec<F>>> = Vec::new(); // polys[m][col] = coefficients
        let mut matrices: Vec<RowMajorMatrix<F>> = Vec::new();

        for &(log_scaling, width) in &specs {
            let degree = d >> log_scaling;
            let lifted_coset = gk.exp_power_of_2(log_scaling).unwrap();
            let mut lifted_points: Vec<F> = lifted_coset.iter().collect();
            reverse_slice_index_bits(&mut lifted_points);
            let height = lifted_points.len();

            let mut matrix_polys = Vec::new();
            let mut matrix_data = Vec::with_capacity(height * width);

            for _ in 0..width {
                let poly: Vec<F> = rng.sample_iter(StandardUniform).take(degree).collect();
                for &pt in &lifted_points {
                    matrix_data.push(horner_base(&poly, pt));
                }
                matrix_polys.push(poly);
            }

            // RowMajorMatrix is row-major, so we need to interleave columns
            let mut interleaved = Vec::with_capacity(height * width);
            for row in 0..height {
                for col in 0..width {
                    interleaved.push(matrix_data[col * height + row]);
                }
            }

            matrices.push(RowMajorMatrix::new(interleaved, width));
            polys.push(matrix_polys);
        }

        // Create precomputation
        let pre = Precomputation::<F, EF>::new(z, &[matrices], log_blowup);

        // Verify evals match Horner evaluation
        for (m, (&(log_scaling, _), matrix_polys)) in specs.iter().zip(polys.iter()).enumerate() {
            let z_lifted = z.exp_power_of_2(log_scaling);
            for (col, poly) in matrix_polys.iter().enumerate() {
                let expected = horner_ext(poly, z_lifted);
                let actual = pre.evals_groups[0][m][col];
                assert_eq!(actual, expected, "Mismatch at matrix {m}, col {col}");
            }
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
        let mut polys_data: Vec<Vec<Vec<PolyData>>> = Vec::new();

        for specs in &group_specs {
            let mut group_matrices = Vec::new();
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
                group_polys.push(matrix_polys);
            }

            matrices_groups.push(group_matrices);
            polys_data.push(group_polys);
        }

        // Create precomputations for both points (with matrix evaluations)
        let pre1 = Precomputation::<F, EF>::new(z1, &matrices_groups, log_blowup);
        let pre2 = Precomputation::<F, EF>::new(z2, &matrices_groups, log_blowup);

        // Generate a random challenge and compute DEEP quotient using the new API
        let challenge: EF = rng.sample(StandardUniform);
        let acc =
            Precomputation::compute_deep_quotient(&[pre1, pre2], &matrices_groups, challenge, 1);

        // Derive coefficients (no padding needed for scalar reduce_evals verification)
        let widths: Vec<usize> = matrices_groups
            .iter()
            .flat_map(|g| g.iter().map(|m| m.width()))
            .collect();
        let coeffs = derive_coeffs_from_challenge(&widths, challenge);

        // For naive verification, we need the evaluations at z1 and z2
        // These are stored in pre1.evals_groups and pre2.evals_groups, but we can
        // recompute them naively using Horner's method
        let compute_evals = |z: EF| -> Vec<Vec<Vec<EF>>> {
            group_specs
                .iter()
                .zip(polys_data.iter())
                .map(|(specs, group_polys)| {
                    specs
                        .iter()
                        .zip(group_polys.iter())
                        .map(|(&(log_scaling, _), matrix_polys)| {
                            let z_lifted = z.exp_power_of_2(log_scaling);
                            matrix_polys
                                .iter()
                                .map(|poly_data| horner_ext(&poly_data.coeffs, z_lifted))
                                .collect()
                        })
                        .collect()
                })
                .collect()
        };

        let evals_groups_1 = compute_evals(z1);
        let evals_groups_2 = compute_evals(z2);

        // Naive verification at each domain point
        for i in 0..n {
            let x = domain[i];
            let mut expected = EF::ZERO;

            for (z, evals_groups) in [(z1, &evals_groups_1), (z2, &evals_groups_2)] {
                let mut eval_at_z = EF::ZERO;
                let mut eval_at_x = EF::ZERO;

                let mut coeff_idx = 0;
                for (g, specs) in group_specs.iter().enumerate() {
                    for (m, &(log_scaling, _)) in specs.iter().enumerate() {
                        let x_lifted: F = x.exp_u64(1u64 << log_scaling);

                        for (col, poly_data) in polys_data[g][m].iter().enumerate() {
                            let coeff = coeffs[coeff_idx][col];

                            // Evaluation at z from naive Horner
                            eval_at_z += coeff * evals_groups[g][m][col];

                            // Evaluation at x (lifted)
                            let f_x = horner_base(&poly_data.coeffs, x_lifted);
                            eval_at_x += coeff * f_x;
                        }
                        coeff_idx += 1;
                    }
                }

                // DEEP quotient: (f(z) - f(x)) / (z - x)
                expected += (eval_at_z - eval_at_x) / (z - x);
            }

            assert_eq!(acc[i], expected, "Mismatch at domain point {}", i);
        }
    }
}
