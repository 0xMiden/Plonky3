//! DEEP/Barycentric precomputation helpers for lifted FRI.
//!
//! Overview
//! - Goal: precompute weights to efficiently evaluate, for any power‑of‑two degree `d ≤ d_max`, a
//!   polynomial `p` at `p(z^r)` using only its evaluations over the coset `(g*H)^r`, where
//!   `r = d_max / d` and `|H| = d_max`. Intuitively, this is “lifting”: extending polynomials to a
//!   common top degree and evaluating them at a consistent lifted point.
//!
//! - Setup: let `K = gH` be a two‑adic coset with `|H| = d_max` (up to an optional blowup in this
//!   crate’s representation; see below). Let `K_base = K[..d_max]` denote the degree‑max domain in
//!   bit‑reversed order. For a point `z ∈ EF` with `z ∉ K`, the Lagrange basis on `K_base` is
//!
//! ```text
//! L_{K_base, i}(X) = ((X/g)^{d_max} - 1)/d_max * k_i/(X - k_i),   with 0 <= i < d_max and k_i in K_base.
//! ```
//!
//!   Hence, given a column of `f`‑values over `K_base`, we compute `f(z)` as
//!
//! ```text
//! f(z) = s_max(z) * sum_{i=0}^{d_max-1} ( k_i/(z - k_i) * f(k_i) ),
//! where s_max(z) = ((z/g)^{d_max} - 1)/d_max.
//!
//! - Lifting to smaller domains: for a degree `d`, set `r = d_max / d` (a power of two). We want
//!   `p(z^r)` from evaluations over `(g*H)^r`, using only the first `d` points in bit‑reversed
//!   order. We precompute the unscaled weights `w_i(z) = k_i/(z - k_i)` on the degree‑max domain
//!   and obtain the weights for the degree‑`d` domain by repeated neighbor additions:
//! ```
//!
//! - Storage is bit‑reversed: adjacent entries of `K` are `(x, -x)`. This lets us descend to
//!   squared cosets `(K_base)^{2^j}` by pairing neighbors.
//!
//! Key identity used to descend heights
//! - Unscaled Lagrange weights pair:
//!
//! ```text
//! x/(z - x) + (-x)/(z + x) = 2 * x^2 / (z^2 - x^2).
//!
//! Repeating this `log2(r)` times halves the length each step and yields the degree‑`d` weights.
//! The same scaling factor `s_max(z)` applies at all subdomains; the required factor `r` is baked
//! into the repeated neighbor additions.
//! ```
//!
//! Because we store `(x, -x)` as neighbors, each descent step is just a pairwise sum. Repeating
//! produces the entire tower down to height 1. We reuse the same `s(z)` at every level—pairwise
//! sums already encode the required `2^j` factor.

use alloc::vec::Vec;
use core::iter;
use core::marker::PhantomData;

use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{BasedVectorSpace, ExtensionField, Field, PackedFieldExtension, PackedFieldPow2, PackedValue, TwoAdicField, batch_multiplicative_inverse, scale_slice_in_place_single_core};
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_util::linear_map::LinearMap;
use p3_util::{log2_strict_usize, reverse_slice_index_bits};

struct BarycentricWeights<F, EF> {
    scaling: EF,
    unscaled_weights: Vec<EF>,
    _marker: PhantomData<F>,
}

impl<F: Field, EF: ExtensionField<F>> BarycentricWeights<F, EF> {
    fn shrink(&self) -> Option<Self> {
        let len = self.unscaled_weights.len();
        if len == 1 {
            return None;
        }

        let scaling = self.scaling.halve();
        let unscaled_weights: Vec<EF> = if len < 2 * F::Packing::WIDTH {
            let (pairs, _) = self.unscaled_weights.as_chunks::<2>();
            pairs.iter().map(|&[x_0, x_1]| x_0 * x_1).collect()
        } else {
            let packing_width = F::Packing::WIDTH;
            self.unscaled_weights
                .par_chunks_exact(2 * packing_width)
                .flat_map(|chunk| {
                    let left: Vec<_> = (0..packing_width).map(|i| chunk[2 * i]).collect();
                    let left = EF::ExtensionPacking::from_ext_slice(&left);
                    let right: Vec<_> = (0..packing_width).map(|i| chunk[2 * i + 1]).collect();
                    let right = EF::ExtensionPacking::from_ext_slice(&right);
                    let sum = left + right;
                    EF::ExtensionPacking::to_ext_iter([sum])
                })
                .collect()
        };

        Some(Self {
            scaling,
            unscaled_weights,
            _marker: Default::default(),
        })
    }

    fn eval_matrix<M: Matrix<F>>(&self, m: &M) -> Vec<EF> {
        let mut evals = m.columnwise_dot_product(&self.unscaled_weights);
        scale_slice_in_place_single_core(&mut evals, self.scaling);
        evals
    }
}

/// Precomputed barycentric/DEEP data at a point `z` for lifting over the coset tower rooted at `K = gH`.
///
/// Usage at a glance
/// - To evaluate a column `f(·)` of degree `d` (with `r = d_max/d`) using evaluations over
///   `(g*H)^r`, compute
///
/// ```text
/// value = s_max(z) * dot( unscaled_lagrange[log_d], f_values_on_(K^r)[..d] )
/// ```
///
///   where `unscaled_lagrange[log_d][i]` are the unscaled weights at degree `d`, derived by
///   neighbor sums from the degree‑max unscaled weights. The scaling factor `s_max(z)` is shared
///   by all subdomains.
///
/// Why this helps
/// - The vectors are reusable for many columns and matrices once `z` is fixed.
/// - We store `1/(z - K[i])` for the full coset to support DEEP/quotient steps.
/// - The tower by `log_h` gives O(N) total precompute, O(N) memory, and O(N) dot per column.
pub struct Precomputation<F: TwoAdicField, EF: ExtensionField<F>> {
    /// Bit‑reversed points of the coset `K = gH`, length `N = d * 2^{log_blowup}`.
    /// Neighboring entries are negations: `K[2i+1] = -K[2i]`.
    coset_points: Vec<F>,

    /// Evaluations of `1/(z-X)` over the largest coset `K`:
    ///   `inverse_vanishing_evals[i] = 1/(z - K[i])`.
    /// Stored once for later DEEP/quotient computations.
    inverse_vanishing_evals: Vec<EF>,

    /// Unscaled Lagrange weights by log‑degree, anchored at the maximum degree `d_max`.
    ///
    /// Top level (key `log_d_max`) stores `[ k_i/(z - k_i) ]` for `i < d_max` over `K_base = K[..d_max]`.
    /// Each descent level halves via neighbor addition, e.g.
    /// `w'_{i} = w_{2i} + w_{2i+1} = 2*x^2/(z^2 - x^2)`.
    unscaled_lagrange: LinearMap<usize, Vec<EF>>,

    /// Barycentric scaling for the maximum degree domain `K_base = K[..d_max]`:
    ///   `s_max(z) = ((z/g)^{d_max} - 1) / d_max = (z^{d_max} - g^{d_max})/(d_max * g^{d_max})`.
    /// The same `s_max(z)` multiplies unscaled weights at all subdomains; neighbor additions bake in `r = d_max/d`.
    scaling_factor: EF,
}

impl<F: TwoAdicField, EF: ExtensionField<F>> Precomputation<F, EF> {
    /// Build the precomputation for a point `z` over the ambient coset used for lifting.
    ///
    /// Inputs:
    /// - `point`: the evaluation point `z ∈ EF` with `z ∉ K` (no poles)
    /// - `d`: the maximum input degree bound `d_max`; pass the largest degree for which you want reuse.
    /// - `log_blowup`: optional blowup exponent; the internal coset size is `N = d_max * 2^{log_blowup}`.
    ///
    /// Output caches:
    /// - `inverse_vanishing_evals[i] = 1/(z - K[i])` for the entire coset `K` (length `N`).
    /// - `unscaled_lagrange[log_h]` holds unscaled weights for degree `2^{log_h}`, anchored at `log_d_max`.
    /// - `scaling_factor` is the barycentric scalar on `K_base = K[..d_max]` and is reused for all subdomains.
    pub fn new(point: EF, d: usize, log_blowup: usize) -> Self {
        let log_d = log2_strict_usize(d);
        let n = d << log_blowup; // N = d · 2^{log_blowup}
        let log_n = log2_strict_usize(n);

        // Build the coset K = gH at size N, then bit‑reverse it so neighbors are negatives.
        let coset_points = {
            let coset = TwoAdicMultiplicativeCoset::new(F::GENERATOR, log_n).unwrap();
            let mut pts: Vec<F> = coset.iter().collect();
            reverse_slice_index_bits(&mut pts);
            pts
        };

        // Inverse differences on the full coset: inv_diffs_full[i] = 1/(z - K[i]).
        let inv_diffs_full = {
            let diffs: Vec<EF> = coset_points.iter().map(|&x| point - x).collect();
            batch_multiplicative_inverse(&diffs)
        };

        // Unscaled Lagrange weights at the top degree bound d: w_i(z) = k_i/(z - k_i) for i < d.
        // Reuse the prefix of inv_diffs_full to avoid recomputing inverses.
        let top_unscaled: Vec<EF> = coset_points
            .iter()
            .zip(inv_diffs_full.iter())
            .take(d)
            .map(|(&k, &inv)| EF::from(k) * inv)
            .collect();

        // Descend to squared sub‑cosets by pairing neighbors.
        // Bit‑reversed neighbors are (x, -x), so
        //   w_{2i}(z) + w_{2i+1}(z) = x/(z - x) + (-x)/(z + x) = 2*x^2/(z^2 - x^2).
        // Repeating this halves the length at each level and yields the entire tower.

        let unscaled_lagrange: LinearMap<usize, Vec<EF>> =
            iter::successors(Some((log_d, top_unscaled)), |(prev_log, prev)| {
                if prev.len() == 1 {
                    return None;
                }
                let next: Vec<EF> = prev.par_chunks_exact(2).map(|c| c[0] + c[1]).collect();
                Some((prev_log - 1, next))
            })
            .collect();

        // Barycentric scaling over the maximum‑degree domain K_base (size d_max = d):
        //   s_max(z) = ((z/shift)^{d_max} - 1) / d_max, using the same shift as K.
        // The same s_max(z) multiplies unscaled weights at all subdomains.
        let scaling_factor = {
            let z_over_shift = point * EF::from(F::GENERATOR.inverse());
            let t = z_over_shift.exp_power_of_2(log_d) - EF::ONE;
            t.div_2exp_u64(log_d as u64)
        };

        Self {
            coset_points,
            inverse_vanishing_evals: inv_diffs_full,
            unscaled_lagrange,
            scaling_factor,
        }
    }

    /// Return the bit‑reversed coset points `K = gH` at the tallest height.
    #[inline]
    pub fn coset_points(&self) -> &[F] {
        &self.coset_points
    }

    /// Get the unscaled Lagrange weights at a specific degree‑bound log‑height.
    /// Returns `Some(&[EF])` representing `k/(z - k)` folded by neighbor sums down to the requested height.
    #[inline]
    pub fn unscaled_lagrange_at(&self, log_h: usize) -> Option<&[EF]> {
        self.unscaled_lagrange.get(&log_h).map(|v| v.as_slice())
    }

    /// Return the barycentric scaling factor `s(z)` on `K_base = K[..d]`.
    #[inline]
    pub fn scaling_factor(&self) -> EF {
        self.scaling_factor
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear as F;
    use p3_field::extension::BinomialExtensionField as EFx;
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_util::log2_strict_usize;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::Precomputation;

    type EF = EFx<F, 4>;

    fn horner_eval(coeffs: &[EF], x: EF) -> EF {
        coeffs.iter().rev().fold(EF::ZERO, |acc, &c| acc * x + c)
    }

    #[test]
    fn pairwise_identities_hold() {
        let mut rng = SmallRng::seed_from_u64(0xC0FFEE);
        let n = 16usize; // degree bound
        let log_blowup = 0usize; // so N = n
        let log_n = log2_strict_usize(n << log_blowup);

        // Choose z randomly in EF and ensure z ∉ gH by checking against first few points
        let mut z: EF = EF::from(F::new(rng.random::<u32>()));
        let pre = Precomputation::<F, EF>::new(z, n, log_blowup);
        // If unlucky, resample a couple of times
        for _ in 0..3 {
            if pre.coset_points().iter().take(8).all(|&y| z != EF::from(y)) {
                break;
            }
            z = EF::from(F::new(rng.random::<u32>()));
        }

        let coset = pre.coset_points();
        let var_top = pre.unscaled_lagrange_at(log2_strict_usize(n)).unwrap();
        let var_next = pre.unscaled_lagrange_at(log2_strict_usize(n) - 1).unwrap();

        for i in 0..(n / 2) {
            // Adjacent pair (x, -x) in bit-reversed order
            let x = coset[2 * i];
            let neg_x = coset[2 * i + 1];
            assert_eq!(neg_x, -x, "bit-reversed neighbors must be negatives");

            // Variable parts sum: x/(z - x) + (-x)/(z + x)
            let sum = var_top[2 * i] + var_top[2 * i + 1];
            assert_eq!(var_next[i], sum, "variable parts pairwise sum");

            // And equals 2 x^2 /(z^2 - x^2)
            let two = EF::TWO;
            let x2 = EF::from(x * x);
            let z2 = z * z;
            let denom = z2 - x2;
            let expected = two * x2 * denom.inverse();
            assert_eq!(var_next[i], expected, "variable parts closed form");
        }
    }

    #[test]
    fn barycentric_matches_polynomial_evaluation() {
        let mut rng = SmallRng::seed_from_u64(0xBADA55);
        let d = 16usize; // degree bound for test
        let log_blowup = 0usize;
        let _log_n = log2_strict_usize(d << log_blowup);
        let z = EF::from(F::new(1234567));
        let pre = Precomputation::<F, EF>::new(z, d, log_blowup);

        let coset = pre.coset_points();
        let var_top = pre.unscaled_lagrange_at(log2_strict_usize(d)).unwrap();
        let s = pre.scaling_factor();

        // Random polynomial of degree < d in EF
        let mut coeffs = vec![EF::ZERO; d];
        for c in &mut coeffs {
            *c = EF::from(F::new(rng.random::<u32>()));
        }

        // Evaluate polynomial at the first d coset points in EF
        let mut evals = vec![EF::ZERO; d];
        for (i, &y) in coset.iter().take(d).enumerate() {
            evals[i] = horner_eval(&coeffs, EF::from(y));
        }

        // Barycentric evaluation via precomputation: s · sum var_i * f(y_i)
        let mut dot = EF::ZERO;
        for i in 0..d {
            dot += var_top[i] * evals[i];
        }
        let bary = s * dot;

        // Direct polynomial evaluation at z
        let direct = horner_eval(&coeffs, z);
        assert_eq!(bary, direct, "barycentric equals direct evaluation");
    }

    #[test]
    fn barycentric_with_blowup_matches_polynomial_evaluation() {
        // Use a non-zero blowup to ensure we operate on the prefix K[..d].
        let mut rng = SmallRng::seed_from_u64(0xF00DCAFE);
        let d = 16usize; // degree bound
        let log_blowup = 2usize; // N = d * 4
        let _log_n = log2_strict_usize(d << log_blowup);

        // Pick a z unlikely to be in the prefix; resample a bit if needed.
        let mut z: EF = EF::from(F::new(rng.random::<u32>()));
        let pre = Precomputation::<F, EF>::new(z, d, log_blowup);
        for _ in 0..3 {
            if pre.coset_points().iter().take(d).all(|&y| z != EF::from(y)) {
                break;
            }
            z = EF::from(F::new(rng.random::<u32>()));
        }
        let pre = Precomputation::<F, EF>::new(z, d, log_blowup);

        let coset = pre.coset_points(); // length N
        let var_top = pre.unscaled_lagrange_at(log2_strict_usize(d)).unwrap(); // length d
        let s = pre.scaling_factor();

        // Random polynomial of degree < d in EF
        let mut coeffs = vec![EF::ZERO; d];
        for c in &mut coeffs {
            *c = EF::from(F::new(rng.random::<u32>()));
        }

        // Evaluate polynomial at the first d points of the coset
        let mut evals = vec![EF::ZERO; d];
        for (i, &y) in coset.iter().take(d).enumerate() {
            evals[i] = horner_eval(&coeffs, EF::from(y));
        }

        // Compute barycentric via precomputation
        let mut dot = EF::ZERO;
        for i in 0..d {
            dot += var_top[i] * evals[i];
        }
        let bary = s * dot;

        // Direct polynomial evaluation at z
        let direct = horner_eval(&coeffs, z);
        assert_eq!(
            bary, direct,
            "barycentric with blowup equals direct evaluation"
        );
    }
}
