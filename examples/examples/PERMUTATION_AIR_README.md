# Permutation AIR Builder Example

This example demonstrates how to use the `PermutationAirBuilder` trait from Plonky3 to implement permutation arguments in Algebraic Intermediate Representations (AIRs).

## Location

This example is located in the `examples` crate at:
- [`examples/examples/permutation_air.rs`](permutation_air.rs)

## Overview

Permutation arguments are a powerful cryptographic technique used in zero-knowledge proof systems to prove relationships between sequences of values. Common use cases include:

- **Permutation checks**: Proving that one sequence is a permutation of another
- **Lookup arguments**: Proving that all queried values exist in a lookup table
- **Memory consistency**: Ensuring reads and writes to memory are consistent

## What's Included

This example provides two AIR implementations demonstrating `PermutationAirBuilder`:

### 1. PermutationCheckAir

Checks whether two sequences are permutations of each other using the product argument:
- Computes ∏(α - a_i) for sequence A
- Computes ∏(α - b_i) for sequence B
- If the products are equal, sequences are permutations (with high probability)

### 2. LookupAir

Demonstrates a lookup table constraint:
- Proves that all query values appear in a lookup table
- Uses fingerprinting: α + value * β
- Compares products of fingerprints

## Running the Example

```bash
# From the root of the Plonky3 repository
cargo run --example permutation_air
```

This will run three demonstrations:
1. Two sequences that ARE permutations (products match)
2. A lookup table example (all queries in table)
3. Two sequences that are NOT permutations (products differ)

## Example Output

```
=== Permutation AIR Builder Example ===

Example 1: Permutation Check
------------------------------
Sequence A: [5, 3, 7, 1]
Sequence B: [1, 7, 5, 3]
AIR width: 8
Main trace dimensions: 1x8
Random challenge α: 42
Permutation trace dimensions: 1x8

Running products:
  Product for A: 2070705 (mod p)
  Product for B: 2070705 (mod p)
✓ Products match! Sequences are permutations of each other.

...
```

## Key Concepts

### Permutation Argument

The permutation argument uses the fact that for any random α:
- If A and B are permutations: ∏(α - a_i) = ∏(α - b_i)
- If A and B are NOT permutations: products differ (with high probability)

This works because multiplication is commutative, so the order doesn't matter.

### PermutationAirBuilder Methods

When implementing an AIR with `PermutationAirBuilder`:

- **`builder.permutation()`** - Returns matrix of permutation trace columns
- **`builder.permutation_randomness()`** - Returns random challenges from verifier
- **`builder.assert_zero_ext()`** - Assert extension field expression equals zero
- **`builder.assert_eq_ext()`** - Assert two extension field expressions are equal
- **`builder.assert_one_ext()`** - Assert extension field expression equals one

### Trace Structure

**Main Trace**: Contains the actual computation values
- Sequence A values
- Sequence B values

**Permutation Trace**: Contains running products in extension field
- Running products for sequence A
- Running products for sequence B

## Implementation Pattern

```rust
impl<AB: PermutationAirBuilder> Air<AB> for MyAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();           // Main trace
        let perm = builder.permutation();    // Permutation trace
        let randomness = builder.permutation_randomness(); // Random challenges

        // Get rows from traces
        let main_local = main.row_slice(0).expect("trace is empty");
        let main_local = &*main_local;
        let perm_local = perm.row_slice(0).expect("perm trace is empty");
        let perm_local = &*perm_local;

        // Access randomness
        let alpha = randomness[0];

        // Build constraints using extension field operations
        builder.assert_eq_ext(expr1, expr2);
    }
}
```

## Key Implementation Details

### Type Conversions

When working with extension field expressions in `PermutationAirBuilder`:

```rust
// For base field values
let value: AB::Var = ...;
let expr: AB::Expr = value.into();
let expr_ef: AB::ExprEF = expr.into();  // Convert to extension field

// For randomness (already in extension field)
let alpha: AB::RandomVar = randomness[0];
let alpha_ef: AB::ExprEF = alpha.into();
```

### Row Access

Rows are accessed via `row_slice()` which returns `Option<impl Deref<Target = [T]>>`:

```rust
let main_local = main.row_slice(0).expect("trace is empty");
let main_local = &*main_local;  // Deref to get &[T]
```

## Understanding the Math

### Permutation Check (Example 1)

For sequences A = [5, 3, 7, 1] and B = [1, 7, 5, 3]:

```
Product_A = (42-5) * (42-3) * (42-7) * (42-1) = 37 * 39 * 35 * 41 = 2070705
Product_B = (42-1) * (42-7) * (42-5) * (42-3) = 41 * 35 * 37 * 39 = 2070705
```

Products match → B is a permutation of A ✓

### Lookup Table (Example 2)

For table [1, 2, 3, 4] and queries [3, 1, 4, 2]:

```
Fingerprints = α + value * β

Query fingerprints: (42+3*17), (42+1*17), (42+4*17), (42+2*17)
                  = 93, 59, 110, 76

Table fingerprints: (42+1*17), (42+2*17), (42+3*17), (42+4*17)
                  = 59, 76, 93, 110

Product of queries = 93 * 59 * 110 * 76 = 45871320
Product of table   = 59 * 76 * 93 * 110 = 45871320
```

Products match → All queries are in table ✓

## References

- [Plonky3 Documentation](https://github.com/Plonky3/Plonky3)
- [AIR Arithmetic](https://aszepieniec.github.io/stark-anatomy/arithmetic)
- [Permutation Arguments (Plookup)](https://eprint.iacr.org/2020/315)
- [LogUp: Logarithmic Lookups](https://eprint.iacr.org/2023/1284)

## Further Exploration

To extend this example, you could:

1. **Multi-row traces**: Extend the examples to accumulate products across multiple rows
2. **Batch lookups**: Implement batched lookup arguments for efficiency
3. **Grand product**: Implement the complete grand product argument with carries
4. **LogUp**: Implement the logarithmic derivative lookup argument
5. **Integration**: Integrate with the full STARK prover in `p3-uni-stark`

## Notes

- This example demonstrates the **AIR definition** only, not the full proving system
- In a complete STARK system, the permutation trace would be computed by the prover
- The verifier's random challenges would come from the Fiat-Shamir transform
- The trait currently serves as a design interface; concrete implementations will follow
