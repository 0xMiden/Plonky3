//! Example demonstrating PermutationAirBuilder usage
//!
//! This example shows how to:
//! 1. Define an AIR using PermutationAirBuilder
//! 2. Generate execution traces
//! 3. Compute permutation traces with randomness
//! 4. Verify permutation arguments work correctly
//!
//! Run with: cargo run --example permutation_air

use p3_air::{Air, BaseAir, ExtensionBuilder, PermutationAirBuilder};
use p3_baby_bear::BabyBear;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use tracing_forest::ForestLayer;
use tracing_subscriber::{EnvFilter, Registry};
use tracing_subscriber::layer::SubscriberExt;

type F = BabyBear;

fn main() {
    // Initialize tracing
    let forest = ForestLayer::default();
    let subscriber = Registry::default()
        .with(EnvFilter::from_default_env())
        .with(forest);
    tracing::subscriber::set_global_default(subscriber).expect("Failed to set subscriber");

    println!("=== Permutation AIR Builder Example ===\n");

    example_permutation_check();
    println!();
    example_lookup();
    println!();
    example_non_permutation();
}

// ============================================================================
// AIR Definitions
// ============================================================================

/// AIR that checks if sequence B is a permutation of sequence A
///
/// This uses the permutation argument with running products:
/// - For sequence A: compute ∏(α - a_i)
/// - For sequence B: compute ∏(α - b_i)
/// - If A and B are permutations, the products are equal
#[derive(Debug, Clone)]
pub struct PermutationCheckAir {
    /// Number of elements in each sequence
    pub num_elements: usize,
}

impl PermutationCheckAir {
    pub fn new(num_elements: usize) -> Self {
        Self { num_elements }
    }
}

impl<F: Field> BaseAir<F> for PermutationCheckAir {
    fn width(&self) -> usize {
        // We need columns for both sequences A and B
        self.num_elements * 2
    }
}

impl<AB: PermutationAirBuilder> Air<AB> for PermutationCheckAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let perm = builder.permutation();
        let randomness = builder.permutation_randomness();

        // Get current and next rows
        let main_local = main.row_slice(0).expect("main trace is empty");
        let main_local = &*main_local;
        let perm_local = perm.row_slice(0).expect("perm trace is empty");
        let perm_local = &*perm_local;
        let perm_next = perm.row_slice(1).expect("perm trace has only 1 row");
        let perm_next = &*perm_next;

        let n = self.num_elements;

        // We need at least one random value for fingerprinting
        assert!(
            !randomness.is_empty(),
            "Need at least one random value for permutation argument"
        );
        let alpha = randomness[0];

        // Process each element
        for i in 0..n {
            let a_value = main_local[i].clone();
            let b_value = main_local[n + i].clone();

            // Compute fingerprint factors: (α - value)
            let a_factor = alpha.into() - a_value.into();
            let b_factor = alpha.into() - b_value.into();

            // Running product accumulation for sequence A
            // running_product[i+1] = running_product[i] * (α - a_i)
            if i == 0 {
                // Initialize first running product
                builder.when_first_row().assert_eq_ext(perm_local[i], a_factor.clone());
            } else {
                builder.when_first_row().assert_eq_ext(
                    perm_local[i].into(),
                    perm_local[i - 1].into() * a_factor.clone()
                );
            }

            // Similarly for sequence B (stored in second half of perm trace)
            if i == 0 {
                builder.when_first_row().assert_eq_ext(perm_local[n + i], b_factor.clone());
            } else {
                builder.when_first_row().assert_eq_ext(
                    perm_local[n + i].into(),
                    perm_local[n + i - 1].into() * b_factor.clone()
                );
            }

            // On subsequent rows, products should stay the same (single-row example)
            // In a multi-row scenario, you'd continue accumulating
            builder.when_transition().assert_eq_ext(
                perm_local[i],
                perm_next[i]
            );
            builder.when_transition().assert_eq_ext(
                perm_local[n + i],
                perm_next[n + i]
            );
        }

        // Final constraint: the two products must be equal
        builder.assert_eq_ext(
            perm_local[n - 1],
            perm_local[2 * n - 1]
        );
    }
}

/// AIR for lookup table constraints using permutation arguments
///
/// This demonstrates using permutation arguments to prove that all
/// lookup queries match entries in a lookup table.
#[derive(Debug, Clone)]
pub struct LookupAir {
    /// Number of lookup operations
    pub num_lookups: usize,
}

impl LookupAir {
    pub fn new(num_lookups: usize) -> Self {
        Self { num_lookups }
    }
}

impl<F: Field> BaseAir<F> for LookupAir {
    fn width(&self) -> usize {
        // Columns: [query_values..., table_values...]
        self.num_lookups * 2
    }
}

impl<AB: PermutationAirBuilder> Air<AB> for LookupAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let perm = builder.permutation();
        let randomness = builder.permutation_randomness();

        let main_local = main.row_slice(0).expect("main trace is empty");
        let main_local = &*main_local;
        let perm_local = perm.row_slice(0).expect("perm trace is empty");
        let perm_local = &*perm_local;
        let perm_next = perm.row_slice(1).expect("perm trace has only 1 row");
        let perm_next = &*perm_next;

        // Need at least 2 random values for lookup arguments
        assert!(
            randomness.len() >= 2,
            "Need at least 2 random values for lookup argument"
        );
        let alpha = randomness[0];
        let beta = randomness[1];

        let n = self.num_lookups;

        // For each lookup operation
        for i in 0..n {
            let query_value = main_local[i].clone();
            let table_value = main_local[n + i].clone();

            // Create fingerprints: α + value * β
            // Need to convert to ExprEF for extension field operations
            let alpha_ef: AB::ExprEF = alpha.into();
            let beta_ef: AB::ExprEF = beta.into();

            // Convert Var -> Expr -> ExprEF
            let query_expr: AB::Expr = query_value.into();
            let table_expr: AB::Expr = table_value.into();
            let query_value_ef: AB::ExprEF = query_expr.into();
            let table_value_ef: AB::ExprEF = table_expr.into();

            let query_fingerprint = alpha_ef.clone() + query_value_ef * beta_ef.clone();
            let table_fingerprint = alpha_ef + table_value_ef * beta_ef;

            // Build running products
            if i == 0 {
                // Initialize
                builder.when_first_row().assert_eq_ext(
                    perm_local[i],
                    query_fingerprint.clone()
                );
                builder.when_first_row().assert_eq_ext(
                    perm_local[n + i],
                    table_fingerprint.clone()
                );
            } else {
                // Accumulate
                builder.when_first_row().assert_eq_ext(
                    perm_local[i].into(),
                    perm_local[i - 1].into() * query_fingerprint.clone()
                );
                builder.when_first_row().assert_eq_ext(
                    perm_local[n + i].into(),
                    perm_local[n + i - 1].into() * table_fingerprint.clone()
                );
            }

            // Maintain products across rows
            builder.when_transition().assert_eq_ext(
                perm_local[i],
                perm_next[i]
            );
            builder.when_transition().assert_eq_ext(
                perm_local[n + i],
                perm_next[n + i]
            );
        }

        // Final check: products should match (meaning queries ⊆ table)
        builder.assert_eq_ext(
            perm_local[n - 1],
            perm_local[2 * n - 1]
        );
    }
}

// ============================================================================
// Trace Generation Functions
// ============================================================================

/// Generate a trace for the PermutationCheckAir
pub fn generate_permutation_trace<F: Field>(
    sequence_a: Vec<F>,
    sequence_b: Vec<F>,
) -> RowMajorMatrix<F> {
    assert_eq!(sequence_a.len(), sequence_b.len());
    let n = sequence_a.len();

    // Create a single-row trace with both sequences
    let mut trace_values = Vec::with_capacity(n * 2);
    trace_values.extend(sequence_a);
    trace_values.extend(sequence_b);

    RowMajorMatrix::new(trace_values, n * 2)
}

/// Generate a permutation trace (the extension field running products)
pub fn generate_permutation_trace_for_check<F: Field>(
    sequence_a: &[F],
    sequence_b: &[F],
    alpha: F,
) -> RowMajorMatrix<F> {
    assert_eq!(sequence_a.len(), sequence_b.len());
    let n = sequence_a.len();

    let mut perm_values = Vec::with_capacity(n * 2);

    // Compute running products for sequence A
    let mut running_product = F::ONE;
    for &a_val in sequence_a {
        running_product *= alpha - a_val;
        perm_values.push(running_product);
    }

    // Compute running products for sequence B
    running_product = F::ONE;
    for &b_val in sequence_b {
        running_product *= alpha - b_val;
        perm_values.push(running_product);
    }

    RowMajorMatrix::new(perm_values, n * 2)
}

// ============================================================================
// Example Functions
// ============================================================================

/// Example 1: Check that two sequences are permutations of each other
fn example_permutation_check() {
    println!("Example 1: Permutation Check");
    println!("------------------------------");

    let sequence_a = vec![
        F::from_u32(5),
        F::from_u32(3),
        F::from_u32(7),
        F::from_u32(1),
    ];

    let sequence_b = vec![
        F::from_u32(1),
        F::from_u32(7),
        F::from_u32(5),
        F::from_u32(3),
    ];

    println!("Sequence A: {:?}", sequence_a.iter().map(|x| x.as_canonical_u64()).collect::<Vec<_>>());
    println!("Sequence B: {:?}", sequence_b.iter().map(|x| x.as_canonical_u64()).collect::<Vec<_>>());

    // Create the AIR
    let air = PermutationCheckAir::new(sequence_a.len());
    println!("AIR width: {}", <PermutationCheckAir as BaseAir<F>>::width(&air));

    // Generate main trace
    let trace = generate_permutation_trace(sequence_a.clone(), sequence_b.clone());
    println!("Main trace dimensions: {}x{}", trace.height(), trace.width());

    // Simulate randomness (in a real proof system, this comes from the verifier)
    let alpha = F::from_u32(42);
    println!("Random challenge α: {}", alpha.as_canonical_u64());

    // Generate permutation trace
    let perm_trace = generate_permutation_trace_for_check(&sequence_a, &sequence_b, alpha);
    println!("Permutation trace dimensions: {}x{}", perm_trace.height(), perm_trace.width());

    // Check the final products
    let n = sequence_a.len();
    let final_product_a = perm_trace.get(0, n - 1).expect("invalid index");
    let final_product_b = perm_trace.get(0, 2 * n - 1).expect("invalid index");

    println!("\nRunning products:");
    println!("  Product for A: {} (mod p)", final_product_a.as_canonical_u64());
    println!("  Product for B: {} (mod p)", final_product_b.as_canonical_u64());

    if final_product_a == final_product_b {
        println!("✓ Products match! Sequences are permutations of each other.");
    } else {
        println!("✗ Products differ! Sequences are NOT permutations.");
    }

    // Manually verify the computation
    println!("\nManual verification:");
    let mut product_a = F::ONE;
    for &val in &sequence_a {
        product_a *= alpha - val;
    }
    println!("  ∏(α - a_i) = {}", product_a.as_canonical_u64());

    let mut product_b = F::ONE;
    for &val in &sequence_b {
        product_b *= alpha - val;
    }
    println!("  ∏(α - b_i) = {}", product_b.as_canonical_u64());
    println!("  Match: {}", product_a == product_b);
}

/// Example 2: Lookup table constraint
fn example_lookup() {
    println!("Example 2: Lookup Table");
    println!("------------------------");

    // Lookup table: [1, 2, 3, 4]
    let table = vec![
        F::from_u32(1),
        F::from_u32(2),
        F::from_u32(3),
        F::from_u32(4),
    ];

    // Queries: all values are in the table
    let queries = vec![
        F::from_u32(3),
        F::from_u32(1),
        F::from_u32(4),
        F::from_u32(2),
    ];

    println!("Lookup table: {:?}", table.iter().map(|x| x.as_canonical_u64()).collect::<Vec<_>>());
    println!("Queries:      {:?}", queries.iter().map(|x| x.as_canonical_u64()).collect::<Vec<_>>());

    // Simulate randomness
    let alpha = F::from_u32(42);
    let beta = F::from_u32(17);
    println!("Random challenges: α={}, β={}", alpha.as_canonical_u64(), beta.as_canonical_u64());

    // Compute fingerprints and products
    println!("\nFingerprints (α + value * β):");

    let mut product_queries = F::ONE;
    for &query in &queries {
        let fingerprint = alpha + query * beta;
        product_queries *= fingerprint;
        println!("  Query {}: fingerprint = {}", query.as_canonical_u64(), fingerprint.as_canonical_u64());
    }

    let mut product_table = F::ONE;
    for &entry in &table {
        let fingerprint = alpha + entry * beta;
        product_table *= fingerprint;
        println!("  Table {}: fingerprint = {}", entry.as_canonical_u64(), fingerprint.as_canonical_u64());
    }

    println!("\nRunning products:");
    println!("  Product of query fingerprints: {}", product_queries.as_canonical_u64());
    println!("  Product of table fingerprints: {}", product_table.as_canonical_u64());

    if product_queries == product_table {
        println!("✓ Products match! All queries are in the table.");
    } else {
        println!("✗ Products differ! Some queries not in table.");
    }
}

/// Example 3: Show that non-permutations fail the check
fn example_non_permutation() {
    println!("Example 3: Non-Permutation Detection");
    println!("--------------------------------------");

    let sequence_a = vec![
        F::from_u32(1),
        F::from_u32(2),
        F::from_u32(3),
        F::from_u32(4),
    ];

    let sequence_b = vec![
        F::from_u32(1),
        F::from_u32(2),
        F::from_u32(5), // Different value!
        F::from_u32(6),
    ];

    println!("Sequence A: {:?}", sequence_a.iter().map(|x| x.as_canonical_u64()).collect::<Vec<_>>());
    println!("Sequence B: {:?}", sequence_b.iter().map(|x| x.as_canonical_u64()).collect::<Vec<_>>());
    println!("(Note: B is NOT a permutation of A)");

    let alpha = F::from_u32(42);
    println!("Random challenge α: {}", alpha.as_canonical_u64());

    let perm_trace = generate_permutation_trace_for_check(&sequence_a, &sequence_b, alpha);

    let n = sequence_a.len();
    let final_product_a = perm_trace.get(0, n - 1).expect("invalid index");
    let final_product_b = perm_trace.get(0, 2 * n - 1).expect("invalid index");

    println!("\nRunning products:");
    println!("  Product for A: {}", final_product_a.as_canonical_u64());
    println!("  Product for B: {}", final_product_b.as_canonical_u64());

    if final_product_a == final_product_b {
        println!("✗ ERROR: Products match, but sequences are not permutations!");
    } else {
        println!("✓ Correctly detected: Products differ, sequences are not permutations.");
    }
}
