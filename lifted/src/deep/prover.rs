use alloc::vec::Vec;
use core::iter::zip;
use core::marker::PhantomData;

use p3_commit::{BatchOpening, Mmcs};
use p3_field::{
    ExtensionField, Field, PackedFieldExtension, PackedValue, TwoAdicField, dot_product,
};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrixView;
use p3_maybe_rayon::prelude::*;

use super::MatrixGroupEvals;
use super::interpolate::SinglePointQuotient;

/// The DEEP quotient `Q(X)` evaluated over the LDE domain.
///
/// Combines all polynomial evaluation claims into a single low-degree polynomial.
/// See module documentation for the construction and soundness argument.
pub struct DeepPoly<'a, F: TwoAdicField, EF: ExtensionField<F>, M: Matrix<F>, Commit: Mmcs<F>> {
    /// References to the committed prover data for each matrix group.
    matrices: Vec<&'a Commit::ProverData<M>>,

    /// The DEEP quotient polynomial evaluated over the domain.
    /// `deep_poly[i]` is the evaluation at the i-th domain point (bit-reversed order).
    deep_poly: Vec<EF>,

    _marker: PhantomData<F>,
}

impl<'a, F: TwoAdicField, EF: ExtensionField<F>, M: Matrix<F>, Commit: Mmcs<F>>
    DeepPoly<'a, F, EF, M, Commit>
{
    /// Construct `Q(X)` from committed matrices and their evaluations at opening points.
    ///
    /// # Arguments
    /// - `c`: The MMCS used for commitment (extracts matrices from prover data)
    /// - `openings`: Pairs `(quotient, evals_groups)` where:
    ///   - `quotient`: Precomputed `1/(zⱼ - X)` from [`SinglePointQuotient`]
    ///   - `evals_groups[commit][matrix][col] = fᵢ(zⱼʳ)` are evaluations at the opening point
    /// - `prover_data`: References to committed matrix data
    /// - `challenge_points`: Challenge `β` for batching opening points
    /// - `challenge_columns`: Challenge `α` for batching columns
    /// - `alignment`: Width for coefficient alignment (must match commitment)
    #[allow(clippy::type_complexity)]
    pub fn new(
        c: &Commit,
        openings: &[(&SinglePointQuotient<F, EF>, Vec<MatrixGroupEvals<EF>>)],
        prover_data: Vec<&'a Commit::ProverData<M>>,
        challenge_points: EF,
        challenge_columns: EF,
        alignment: usize,
    ) -> Self {
        assert!(!openings.is_empty(), "openings must not be empty");

        let matrices_groups: Vec<Vec<&M>> = prover_data
            .iter()
            .map(|data| c.get_matrices(*data))
            .collect();

        let w = F::Packing::WIDTH;
        let n = openings[0].0.point_quotient().len();

        let group_sizes: Vec<usize> = matrices_groups.iter().map(|g| g.len()).collect();
        let widths: Vec<usize> = matrices_groups
            .iter()
            .flat_map(|g| g.iter().map(|m| m.width()))
            .collect();

        let coeffs_columns: Vec<Vec<EF>> =
            derive_coeffs_from_challenge(&widths, challenge_columns, alignment);

        // Negate coefficients so inner loop computes f_reduced(z) - f_reduced(X) via addition
        let neg_column_coeffs: Vec<Vec<EF>> = coeffs_columns
            .iter()
            .map(|c| c.iter().copied().map(EF::neg).collect())
            .collect();

        // Compute -f_reduced(X) = -Σᵢ αⁱ · fᵢ(X) over the LDE domain.
        // Negating here lets the inner loop compute f_reduced(zⱼ) - f_reduced(X) via addition.
        let mut neg_column_coeffs_iter = neg_column_coeffs.iter();
        let neg_f_reduced = zip(&matrices_groups, &group_sizes)
            .map(|(matrices_group, &size)| {
                let group_coeffs: Vec<&Vec<EF>> =
                    neg_column_coeffs_iter.by_ref().take(size).collect();
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

        // Q(X) = Σⱼ βʲ · (f_reduced(zⱼ) - f_reduced(X)) · 1/(zⱼ - X)
        let mut deep_poly = EF::zero_vec(n);
        let mut point_coeff = EF::ONE;
        for (quotient, evals_groups) in openings {
            let point_quotient = quotient.point_quotient();
            debug_assert_eq!(point_quotient.len(), n);

            let coeffs_flat = coeffs_columns.iter().flatten().copied();
            let evals_flat = evals_groups.iter().flat_map(|g| g.flatten()).copied();
            let f_reduced_at_z: EF = dot_product(coeffs_flat, evals_flat);
            let f_reduced_at_z_packed = EF::ExtensionPacking::from(f_reduced_at_z);
            let point_coeff_ef = EF::ExtensionPacking::from(point_coeff);
            deep_poly
                .par_chunks_exact_mut(w)
                .zip(neg_f_reduced.par_chunks_exact(w))
                .zip(point_quotient[..n].par_chunks_exact(w))
                .for_each(|((acc_chunk, neg_chunk), q_chunk)| {
                    let acc_p = EF::ExtensionPacking::from_ext_slice(acc_chunk);
                    let neg_p = EF::ExtensionPacking::from_ext_slice(neg_chunk);
                    let q_p = EF::ExtensionPacking::from_ext_slice(q_chunk);

                    // Q(X) += βʲ · (f_reduced(zⱼ) - f_reduced(X)) / (zⱼ - X)
                    //       = βʲ · q(X) · (f_reduced(zⱼ) + neg_f_reduced(X))
                    // where q(X) = 1/(zⱼ - X) is the point quotient.
                    let res_p = acc_p + point_coeff_ef * q_p * (f_reduced_at_z_packed + neg_p);
                    res_p.to_ext_slice(acc_chunk);
                });
            point_coeff *= challenge_points;
        }

        Self {
            matrices: prover_data,
            deep_poly,
            _marker: PhantomData,
        }
    }
    
    pub fn folded(&self, arity: usize) -> RowMajorMatrixView<'_, EF> {
        assert!(arity.is_power_of_two());
        RowMajorMatrixView::new(&self.deep_poly, arity)
    }

    pub fn commit<LdeMmcs: Mmcs<EF>>(
        &self,
        c: &LdeMmcs,
        arity: usize,
    ) -> (
        LdeMmcs::Commitment,
        LdeMmcs::ProverData<RowMajorMatrixView<'_, EF>>,
    ) {
        assert!(arity.is_power_of_two());
        let folded_matrix = RowMajorMatrixView::new(&self.deep_poly, arity);
        c.commit_matrix(folded_matrix)
    }

    pub fn open(&self, c: &Commit, index: usize) -> (EF, Vec<BatchOpening<F, Commit>>) {
        let openings = self
            .matrices
            .iter()
            .map(|m| c.open_batch(index, m))
            .collect();
        let eval = self.deep_poly[index];
        (eval, openings)
    }
}

/// Accumulate `f_reduced(X) = Σᵢ αⁱ · fᵢ(X)` across matrices of varying heights.
///
/// In bit-reversed order, lifting `f(X)` to `f(Xʳ)` repeats each value r times.
/// We exploit this: when crossing a height boundary, upsample by repeating entries,
/// then continue accumulating. Matrices must be sorted by ascending height.
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

        // Upsample: [a, b] → [a, a, b, b] when height doubles
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

/// Derive coefficients `[αⁿ⁻¹, ..., α, 1]` (reversed) for batching.
///
/// Reversed order enables Horner evaluation: the verifier processes values
/// left-to-right computing `α·acc + val`, which produces `Σ αⁿ⁻¹⁻ⁱ·vᵢ`.
///
/// # Alignment
///
/// Each matrix's coefficient range is padded to a multiple of `alignment`.
/// This is equivalent to an implementation that:
/// 1. Pads each row with zeros to the alignment width
/// 2. Computes the linear combination including those zeros
///
/// The zeros don't affect the sum, but they do affect coefficient indexing.
/// By aligning indices, we ensure the prover's explicit coefficients match
/// the verifier's Horner reduction (which skips the implicit zeros via `α^gap`).
pub(crate) fn derive_coeffs_from_challenge<EF: Field>(
    widths: &[usize],
    challenge: EF,
    alignment: usize,
) -> Vec<Vec<EF>> {
    let total: usize = widths.iter().map(|w| w.next_multiple_of(alignment)).sum();
    let all_powers: Vec<EF> = challenge.powers().collect_n(total);
    let rev_powers_iter = &mut all_powers.into_iter().rev();

    widths
        .iter()
        .map(|&width| {
            let padded = width.next_multiple_of(alignment);
            let mut coeffs: Vec<EF> = rev_powers_iter.take(padded).collect();
            coeffs.truncate(width); // drop alignment padding
            coeffs
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear as F;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{PrimeCharacteristicRing, dot_product};

    use super::derive_coeffs_from_challenge;
    use crate::deep::verifier::reduce_with_powers;

    type EF = BinomialExtensionField<F, 4>;

    /// `reduce_with_powers` (Horner) must match explicit `derive_coeffs` + dot product.
    #[test]
    fn reduce_evals_matches_reduce_with_powers() {
        let c: EF = EF::from_u64(2);
        let alignment = 3;
        let widths = [2usize, 3];
        let rows: Vec<Vec<F>> = vec![
            vec![F::from_u64(1), F::from_u64(2)],
            vec![F::from_u64(3), F::from_u64(4), F::from_u64(5)],
        ];

        let coeffs = derive_coeffs_from_challenge(&widths, c, alignment);

        // Explicit coefficient sum: Σᵢ coeffs[i] · rows[i]
        let explicit: EF = dot_product(
            coeffs.iter().flatten().copied(),
            rows.iter().flatten().copied(),
        );

        // Horner using reduce_with_powers (same as used in verifier)
        let horner: EF = reduce_with_powers(rows.iter().map(|r| r.as_slice()), c, alignment);

        assert_eq!(explicit, horner);
    }

    /// Alignment: coeffs match Horner for various width/alignment combos.
    #[test]
    fn derive_coeffs_alignment() {
        let c: EF = EF::from_u64(7);
        let alignment = 4;
        let widths = [3usize, 5, 2];

        let coeffs = derive_coeffs_from_challenge(&widths, c, alignment);

        // Verify lengths match widths
        assert_eq!(coeffs[0].len(), 3);
        assert_eq!(coeffs[1].len(), 5);
        assert_eq!(coeffs[2].len(), 2);

        // Verify this matches Horner reduction with arbitrary test data
        let rows: Vec<Vec<F>> = vec![
            vec![F::from_u64(10), F::from_u64(20), F::from_u64(30)],
            vec![
                F::from_u64(1),
                F::from_u64(2),
                F::from_u64(3),
                F::from_u64(4),
                F::from_u64(5),
            ],
            vec![F::from_u64(100), F::from_u64(200)],
        ];

        let explicit: EF = dot_product(
            coeffs.iter().flatten().copied(),
            rows.iter().flatten().copied(),
        );
        let horner: EF = reduce_with_powers(rows.iter().map(|r| r.as_slice()), c, alignment);

        assert_eq!(explicit, horner);
    }
}
