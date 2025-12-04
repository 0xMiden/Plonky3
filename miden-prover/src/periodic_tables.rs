//! Efficient evaluation of periodic columns in STARKs.
//!
//! # Overview
//!
//! Periodic columns are trace columns that repeat with a period dividing the trace length.
//! Instead of committing to these columns, both prover and verifier compute them independently.
//! This module provides efficient algorithms to evaluate periodic columns at:
//! - All points in the quotient domain (during proving)
//! - A single out-of-domain challenge point (during verification)
//!
//! # Mathematical Foundation
//!
//! For a periodic column with period `P` and trace height `N` where `P | N`:
//!
//! The column repeats `N/P` times over the trace domain. Instead of interpolating over the
//! full trace height `N`, we leverage this periodicity to interpolate only over the minimal
//! repeating cycle of size `P`.
//!
//! ## Key Insight
//!
//! If a polynomial `f` is periodic with period `P` over a trace domain `g·H` (where `g` is a
//! shift and `H` is a two-adic subgroup of size `N`), then for any point `z`:
//!
//! ```text
//! f(z) = f(y) where y = (z/g)^(N/P)
//! ```
//!
//! This maps evaluation points to positions within a single period, allowing us to:
//! 1. Store only the minimal cycle (period `P` elements)
//! 2. Interpolate over a subgroup of size `P` rather than `N`
//! 3. Reuse interpolation work for columns with the same period
//!
//! ## Efficiency
//!
//! The work scales as `sum(period_i) + num_points * (#period_groups)` rather than `N²`:
//! - We only interpolate over each column's minimal period `P_i`, not the full trace height
//! - Columns with the same period share a subgroup and batch interpolation work
//! - For each evaluation point, we do O(period) work per group
//!
//! This assumes a two-adic trace domain so each period divides the trace height
//! and `rate_bits = log2(N/P)` is integral.
//!
//! # Functions
//!
//! - [`compute_periodic_on_quotient`]: Evaluates periodic columns over all quotient domain points.
//!   Called by the prover during quotient polynomial computation.
//! - [`evaluate_periodic_at_point`]: Evaluates periodic columns at a single challenge point.
//!   Called by the verifier to check constraint satisfaction.

use std::collections::BTreeMap;

use p3_commit::PolynomialSpace;
use p3_field::{ExtensionField, TwoAdicField, batch_multiplicative_inverse};
use p3_interpolation::interpolate_coset_with_precomputation;
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_strict_usize;

/// Computes evaluations of periodic columns over the entire quotient domain.
///
/// Used by the prover during quotient polynomial computation. This function evaluates
/// all periodic columns at every point in the quotient domain.
///
/// # Implementation Details
///
/// 1. Groups columns by their period to batch interpolation for efficiency
/// 2. For each group with period `P`:
///    - For each quotient point `z`, computes `y = (z / shift)^(N/P)`
///    - Interpolates the column over its minimal cycle (subgroup of size `P`)
///      using barycentric Lagrange interpolation to evaluate at `y`
///
/// # Arguments
///
/// * `periodic_table` - Vector of periodic columns, where each column is a vector
///   of length equal to its period (a power of 2 that divides trace height)
/// * `trace_domain` - The domain over which the trace is defined
/// * `quotient_points` - Pre-computed evaluation points in the quotient domain
///
/// # Returns
///
/// `Some(Vec<Vec<EF>>)` where the outer vector corresponds to periodic columns
/// and the inner vector contains evaluations over all quotient points. Returns `None`
/// if the periodic table is empty.
pub(crate) fn compute_periodic_on_quotient<F, EF>(
    periodic_table: Vec<Vec<F>>,
    trace_domain: impl PolynomialSpace<Val = F>,
    quotient_points: &[EF],
) -> Option<Vec<Vec<EF>>>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
{
    if periodic_table.is_empty() {
        return None;
    }

    let trace_height = trace_domain.size();
    let log_trace_height = log2_strict_usize(trace_height); // panics if trace_height is not a power of 2
    let quotient_size = quotient_points.len();

    // Undo the trace-domain shift to map points to the unshifted subgroup via multiplying shift_inv.
    // If trace_domain = g·H for generator g, we need points in H for interpolation.
    let shift_inv = trace_domain.first_point().inverse();
    // Group columns by period to batch interpolation per unique cycle size.
    // we batch all columns with the same period and reuse the same subgroup, diffs, and inverse computations.
    let mut grouped: BTreeMap<usize, Vec<(usize, Vec<F>)>> = BTreeMap::new();
    // Allocate output slots per periodic column; filled group by group below.
    let mut evals = vec![Vec::new(); periodic_table.len()];

    // first, let's group the columns of same length together.
    // they share a same subgroup.
    for (idx, col) in periodic_table.into_iter().enumerate() {
        if col.is_empty() {
            // Question: prob we should panic here? Is there a case where any column is empty?
            evals[idx] = vec![EF::ZERO; quotient_size];
            continue;
        }

        debug_assert!(
            trace_height.is_multiple_of(col.len()),
            "Periodic column length must divide trace length"
        );

        grouped.entry(col.len()).or_default().push((idx, col));
    }

    // for each subgroup, compute the eval via interpolation
    for (period, cols) in grouped {
        let log_period = log2_strict_usize(period);
        debug_assert!(
            log_trace_height >= log_period,
            "Periodic column period cannot exceed trace height"
        );
        // rate_bits = log2(trace_height / period); i.e. rate = 2^{rate_bits} so y = z^{rate}.
        let rate_bits = log_trace_height - log_period;
        let subgroup: Vec<_> = F::two_adic_generator(log_period)
            .powers()
            .take(period)
            .collect();

        let num_cols = cols.len();
        let mut subgroup_values = Vec::with_capacity(period * num_cols);
        for row in 0..period {
            for (_, col) in cols.iter() {
                subgroup_values.push(col[row]);
            }
        }
        let subgroup_matrix = RowMajorMatrix::new(subgroup_values, num_cols);

        let mut group_evals = vec![Vec::with_capacity(quotient_size); num_cols];
        for &z in quotient_points {
            let unshifted = z * EF::from(shift_inv);
            let y = unshifted.exp_power_of_2(rate_bits); // y = (z / shift)^{trace_height / period}
            let diffs: Vec<_> = subgroup.iter().map(|&g| y - EF::from(g)).collect();
            let diff_invs = batch_multiplicative_inverse(&diffs);

            let values_at_y = interpolate_coset_with_precomputation(
                &subgroup_matrix,
                F::ONE,
                y,
                &subgroup,
                &diff_invs,
            );

            // group_evals is column-major for this period group:
            // - rows: quotient points (iterate all quotient_points)
            // - cols: columns in this group (order matches `cols`)
            // After filling:
            //   group_evals[c] = [value_at(z0), value_at(z1), ..., value_at(zQ-1)]
            // for column c in the group.
            for (col_idx, value) in values_at_y.into_iter().enumerate() {
                group_evals[col_idx].push(value);
            }
        }

        for (local_idx, (orig_idx, _)) in cols.iter().enumerate() {
            evals[*orig_idx] = group_evals[local_idx].clone();
        }
    }

    Some(evals)
}

/// Evaluates periodic columns at an out-of-domain challenge point `zeta`.
///
/// Used by the verifier to check constraint satisfaction. This function evaluates
/// all periodic columns at a single random challenge point.
///
/// # Implementation Details
///
/// For each periodic column with period `P` and trace height `N`:
/// 1. Shifts `zeta` by the trace domain's offset to get `unshifted_zeta`
/// 2. Computes `y = unshifted_zeta^(N/P)`, mapping `zeta` to its position within one period
/// 3. Interpolates the column over its minimal cycle (subgroup of size `P`)
///    using barycentric Lagrange interpolation to evaluate at `y`
///
/// # Arguments
///
/// * `periodic_table` - Vector of periodic columns, where each column is a vector
///   of length equal to its period (a power of 2 that divides trace height)
/// * `trace_domain` - The domain over which the trace is defined
/// * `zeta` - The out-of-domain challenge point at which to evaluate
///
/// # Returns
///
/// A vector containing the evaluation of each periodic column at `zeta`
pub(crate) fn evaluate_periodic_at_point<F, EF>(
    periodic_table: Vec<Vec<F>>,
    trace_domain: impl PolynomialSpace<Val = F>,
    zeta: EF,
) -> Vec<EF>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
{
    if periodic_table.is_empty() {
        return Vec::new();
    }

    let trace_height = trace_domain.size();
    let log_trace_height = log2_strict_usize(trace_height); // panics if trace_height is not a power of 2
    // Multiplier to move into the unshifted subgroup.
    let shift_inv = trace_domain.first_point().inverse();
    let unshifted_zeta = zeta * EF::from(shift_inv);

    periodic_table
        .into_iter()
        .map(|col| {
            if col.is_empty() {
                return EF::ZERO;
            }

            debug_assert!(
                trace_height.is_multiple_of(col.len()),
                "Periodic column length must divide trace length"
            );

            let log_period = log2_strict_usize(col.len());
            debug_assert!(
                log_trace_height >= log_period,
                "Periodic column period cannot exceed trace height"
            );
            // rate_bits = log2(trace_height / period); rate = 2^{rate_bits} so y = (zeta/shift)^{rate}.
            let rate_bits = log_trace_height - log_period;
            let y = unshifted_zeta.exp_power_of_2(rate_bits); // y = (zeta / shift)^{trace_height / period}

            let subgroup: Vec<_> = F::two_adic_generator(log_period)
                .powers()
                .take(col.len())
                .collect();
            let diffs: Vec<_> = subgroup.iter().map(|&g| y - EF::from(g)).collect();
            let diff_invs = batch_multiplicative_inverse(&diffs);

            interpolate_coset_with_precomputation(
                &RowMajorMatrix::new(col, 1),
                F::ONE,
                y,
                &subgroup,
                &diff_invs,
            )
            .pop()
            .expect("single-column interpolation should return one value")
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use p3_field::coset::TwoAdicMultiplicativeCoset;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_goldilocks::Goldilocks;
    use p3_interpolation::interpolate_coset;
    use p3_matrix::dense::RowMajorMatrix;

    use super::*;

    type Val = Goldilocks;
    type Challenge = BinomialExtensionField<Val, 2>;

    /// Test that compute_periodic_on_quotient produces the same results as the naive method
    /// where we unpack the periodic table into a full column and do interpolation for the whole column
    #[test]
    fn test_compute_periodic_on_quotient_correctness() {
        // Test parameters
        let trace_height = 16; // Must be a power of 2
        let log_quotient_degree = 2;
        let quotient_size = trace_height << log_quotient_degree;

        // Create test periodic columns with different periods
        let periodic_table = vec![
            // Period 2: [10, 20]
            vec![Val::from_u32(10), Val::from_u32(20)],
            // Period 4: [1, 2, 3, 4]
            vec![
                Val::from_u32(1),
                Val::from_u32(2),
                Val::from_u32(3),
                Val::from_u32(4),
            ],
            // Period 8: [5, 6, 7, 8, 9, 10, 11, 12]
            vec![
                Val::from_u32(5),
                Val::from_u32(6),
                Val::from_u32(7),
                Val::from_u32(8),
                Val::from_u32(9),
                Val::from_u32(10),
                Val::from_u32(11),
                Val::from_u32(12),
            ],
        ];

        // Get the trace domain
        let log_trace_height = log2_strict_usize(trace_height);
        let trace_domain = TwoAdicMultiplicativeCoset::new(Val::GENERATOR, log_trace_height)
            .expect("valid trace domain");
        let quotient_domain = trace_domain.create_disjoint_domain(quotient_size);

        // Generate quotient points
        let quotient_points: Vec<Challenge> = {
            let mut pts = Vec::with_capacity(quotient_size);
            let mut point = Challenge::from(quotient_domain.first_point());
            pts.push(point);
            for _ in 1..quotient_size {
                point = quotient_domain
                    .next_point(point)
                    .expect("quotient_domain should yield enough points");
                pts.push(point);
            }
            pts
        };

        // Method 1: Optimized method (compute_periodic_on_quotient)
        let optimized_result = compute_periodic_on_quotient::<Val, Challenge>(
            periodic_table.clone(),
            trace_domain,
            &quotient_points,
        )
        .expect("periodic_table should not be empty");

        // Method 2: Naive method - unpack each periodic column to full trace height and interpolate
        let shift = trace_domain.first_point();
        let naive_result: Vec<Vec<Challenge>> = periodic_table
            .iter()
            .map(|periodic_col| {
                let period = periodic_col.len();

                // Unpack: repeat the periodic column to fill the entire trace height
                let mut unpacked = Vec::with_capacity(trace_height);
                for i in 0..trace_height {
                    unpacked.push(periodic_col[i % period]);
                }

                // Create a matrix from the unpacked column
                let unpacked_matrix = RowMajorMatrix::new(unpacked, 1);

                // For each quotient point, interpolate the full column
                let mut evals = Vec::with_capacity(quotient_size);
                for &z in &quotient_points {
                    // Interpolate the full unpacked column at this point
                    let result = interpolate_coset(&unpacked_matrix, shift, z);
                    evals.push(result[0]);
                }

                evals
            })
            .collect();

        // Compare the results
        assert_eq!(optimized_result, naive_result);
    }

    /// Test with edge case: single period equals trace height
    #[test]
    fn test_compute_periodic_on_quotient_full_period() {
        let trace_height = 8;
        let log_quotient_degree = 1;
        let quotient_size = trace_height << log_quotient_degree;

        // Periodic column with period = trace_height (no repetition)
        let periodic_table = vec![vec![
            Val::from_u32(1),
            Val::from_u32(2),
            Val::from_u32(3),
            Val::from_u32(4),
            Val::from_u32(5),
            Val::from_u32(6),
            Val::from_u32(7),
            Val::from_u32(8),
        ]];

        let trace_domain =
            TwoAdicMultiplicativeCoset::new(Val::GENERATOR, log2_strict_usize(trace_height))
                .expect("valid trace domain");
        let quotient_domain = trace_domain.create_disjoint_domain(quotient_size);

        let quotient_points: Vec<Challenge> = {
            let mut pts = Vec::with_capacity(quotient_size);
            let mut point = Challenge::from(quotient_domain.first_point());
            pts.push(point);
            for _ in 1..quotient_size {
                point = quotient_domain
                    .next_point(point)
                    .expect("quotient_domain should yield enough points");
                pts.push(point);
            }
            pts
        };

        let optimized_result = compute_periodic_on_quotient::<Val, Challenge>(
            periodic_table.clone(),
            trace_domain,
            &quotient_points,
        )
        .expect("periodic_table should not be empty");

        // Naive method
        let shift = trace_domain.first_point();
        let naive_result: Vec<Vec<Challenge>> = periodic_table
            .iter()
            .map(|periodic_col| {
                let unpacked_matrix = RowMajorMatrix::new(periodic_col.clone(), 1);
                let mut evals = Vec::with_capacity(quotient_size);
                for &z in &quotient_points {
                    let result = interpolate_coset(&unpacked_matrix, shift, z);
                    evals.push(result[0]);
                }
                evals
            })
            .collect();

        // Compare results
        assert_eq!(optimized_result, naive_result);
    }
}
