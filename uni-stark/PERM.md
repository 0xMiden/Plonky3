# Permutation Arguments using LogUp in Plonky3

This guide explains how to use the LogUp (Logarithmic Derivative) protocol to prove that two columns in your execution trace form a permutation.

## Table of Contents

1. [Overview](#overview)
2. [What is LogUp?](#what-is-logup)
3. [API Components](#api-components)
4. [Step-by-Step Guide](#step-by-step-guide)
5. [Complete Example](#complete-example)
6. [Common Pitfalls](#common-pitfalls)

## Overview

The LogUp protocol is a zero-knowledge proof technique that proves two columns contain the same multiset of elements (i.e., one is a permutation of the other). This is particularly useful for:

- Proving memory consistency (addresses match values)
- Proving lookup arguments (all lookups exist in a table)
- Proving range checks (all values are in a valid range)

In Plonky3's uni-stark, the LogUp implementation is **hardcoded** to check that **the last two columns** of your main trace form a permutation.

## What is LogUp?

Given two columns `A = [a₀, a₁, ..., aₙ₋₁]` and `B = [b₀, b₁, ..., bₙ₋₁]` that should be permutations, and a random challenge `r`:

1. Compute `tᵢ = 1/(r - aᵢ)` for each element in column A
2. Compute `wᵢ = 1/(r - bᵢ)` for each element in column B
3. Maintain a running sum: `Sᵢ = Σⱼ₌₀ⁱ (tⱼ - wⱼ)`
4. If A and B are truly permutations, then `Sₙ₋₁ = 0`

The soundness relies on the Schwartz-Zippel lemma: if the columns aren't permutations, the probability that the final sum equals zero is negligibly small (approximately `n/|F|`).

## API Components

### 1. Trait Implementations

Your AIR must implement these traits:

```rust
// Basic AIR structure
impl<F> BaseAir<F> for MyAir {
    fn width(&self) -> usize { 3 }
}

// AIR evaluation with LogUp support (EF-first)
impl<AB: AirBuilderWithPublicValues + PermutationAirBuilder> Air<AB> for MyAir
where
    AB::F: Field,
{
    fn eval(&self, builder: &mut AB) {
        // Your constraint implementation
    }
}
```

The permutation interface follows the logup-p3 style and works directly over the extension field:

```rust
pub trait PermutationAirBuilder: ExtensionBuilder {
    type MP: Matrix<Self::VarEF>;                 // EF aux rows
    type RandomVar: Into<Self::ExprEF> + Copy;    // EF randomness

    fn permutation(&self) -> Self::MP;            // EF aux matrix view
    fn permutation_randomness(&self) -> &[Self::RandomVar];
}
```

### 2. Main Trace Generation

Your main trace **must** have the permutation columns as the **last two columns**:

```rust
pub fn generate_main_trace<F: Field>(data: &[u64], n: usize) -> RowMajorMatrix<F> {
    let mut trace = RowMajorMatrix::new(F::zero_vec(n * 3), 3);

    for i in 0..n {
        // ... your computation logic ...

        // CRITICAL: Last two columns must be permutations
        trace.set(i, 1, F::from_u64(column_a[i]));  // Second-to-last column
        trace.set(i, 2, F::from_u64(column_b[i]));  // Last column (permutation of column_a)
    }

    trace
}
```

### 3. AIR Constraints

In your `eval` function, you need to:

#### a. Access the auxiliary trace and randomness

```rust
let aux = builder.permutation();               // EF rows
let r = builder.permutation_randomness();      // EF challenges
```

#### b. Extract the permutation values from your main trace

```rust
let main = builder.main();
let (local, next) = (
    main.row_slice(0).expect("Matrix is empty?"),
    main.row_slice(1).expect("Matrix only has 1 row?"),
);

// Get the values from the last two columns
let xi = local[width - 2].clone().into();  // Second-to-last column
let yi = local[width - 1].clone().into();  // Last column
```

#### c. Constrain the auxiliary trace structure (EF-first)

The auxiliary trace has 3 extension field columns (t, w, running sum), already exposed as EF elements:

```rust
let local = aux.row_slice(0).unwrap();
let next = aux.row_slice(1).unwrap();

let t_i: AB::ExprEF = local[0].into();
let w_i: AB::ExprEF = local[1].into();
let s_i: AB::ExprEF = local[2].into();
let t_next: AB::ExprEF = next[0].into();
let w_next: AB::ExprEF = next[1].into();
let s_next: AB::ExprEF = next[2].into();

let r_expr = r[0].into();

// t * (r - x_i) == 1
builder.assert_one_ext(t_i.clone() * (r_expr.clone() - AB::Expr::from(xi).into()));
// w * (r - y_i) == 1
builder.assert_one_ext(w_i.clone() * (r_expr - AB::Expr::from(yi).into()));

// Running sums
builder.when_first_row().assert_eq_ext(s_i.clone(), t_i.clone() - w_i.clone());
builder.when_transition().assert_eq_ext(s_next, s_i.clone() + t_next - w_next);
builder.when_last_row().assert_zero_ext(s_i);
```

### 4. Proving and Verifying

The prover and verifier handle the auxiliary trace generation automatically. Configure EF challenges and aux building via `with_aux_builder`:

```rust
use p3_uni_stark::{prove, verify, StarkConfig, generate_logup_trace};

// Configure EF challenge count and aux width (in base field limbs)
let config = MyConfig::new(pcs, challenger)
    .with_aux_builder(1, 12, |main, challenges| {
        // Build aux trace over EF, but commit its flattened base view internally
        generate_logup_trace::<Challenge, _>(main, &challenges[0])
    });

// Generate your main trace (with permutation in last two columns)
let trace = generate_main_trace(data, n);
let public_values = vec![/* your public inputs */];

// Prove (auxiliary trace is generated internally)
let proof = prove(&config, &MyAir {}, trace, &public_values);

// Verify
verify(&config, &MyAir {}, &proof, &public_values)
    .expect("verification failed");
```

## Step-by-Step Guide

### Step 1: Design Your Main Trace

Decide what data you want to prove has a permutation relationship. Arrange your trace so the **last two columns** contain these values.

**Example:** Proving Fibonacci numbers have a permutation
```
| computation | column_a | column_b |
|-------------|----------|----------|
| fib(0) = 1  |    1     |    8     |
| fib(1) = 1  |    1     |    5     |
| fib(2) = 2  |    2     |    3     |
| fib(3) = 3  |    3     |    2     |
| fib(4) = 5  |    5     |    1     |
| fib(5) = 8  |    8     |    1     |
```

Here, column_b is a permutation of column_a: both contain {1,1,2,3,5,8}.

### Step 2: Implement the Required Traits

```rust
impl<F> BaseAir<F> for MyAir {
    fn width(&self) -> usize { 3 }
}
```

### Step 3: Write Your Trace Generator

```rust
fn generate_trace<F: Field>(n: usize) -> RowMajorMatrix<F> {
    let mut trace = RowMajorMatrix::new(F::zero_vec(n * 3), 3);

    // Generate your computation in column 0
    for i in 0..n {
        trace.set(i, 0, /* computation */);
    }

    // Generate permutation columns
    let mut column_a = vec![];
    let mut column_b = vec![];
    for i in 0..n {
        column_a.push(/* some value */);
        column_b.push(/* permutation of column_a */);
    }

    for i in 0..n {
        trace.set(i, 1, F::from(column_a[i]));
        trace.set(i, 2, F::from(column_b[i]));
    }

    trace
}
```

### Step 4: Implement AIR Constraints

Follow the pattern in [section 3c](#c-constrain-the-auxiliary-trace-structure) above.

### Step 5: Prove and Verify

```rust
let trace = generate_trace(n);
let proof = prove(&config, &MyAir {}, trace, &public_values);
verify(&config, &MyAir {}, &proof, &public_values).unwrap();
```

## Complete Example

See [tests/perm_air.rs](tests/perm_air.rs) for a complete working example that:
- Implements a Fibonacci computation
- Proves a permutation between the last two columns
- Includes all necessary trait implementations
- Shows how to set up the configuration and run the prover/verifier

Key points from the example:
- Main trace width: 3 (one computation column + two permutation columns)
- Auxiliary trace width: 12 base field elements (3 extension field columns × 4)
- Randomness width: 4 base field elements (1 extension field challenge)
- Extension field: BabyBearExt4 with constant `w = 11`
