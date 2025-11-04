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
    fn width(&self) -> usize {
        // Return the number of columns in your main trace
        3  // Example: includes the two permutation columns
    }
}

// Multi-phase support (required for LogUp)
impl<F> MultiPhaseBaseAir<F> for MyAir {
    fn aux_width_in_base_field(&self) -> usize {
        // For LogUp: 3 extension field columns = 3 * EF::DIMENSION base field columns
        // For BabyBearExt4: 3 * 4 = 12
        12
    }

    fn num_randomness_in_base_field(&self) -> usize {
        // Number of base field elements in the random challenge
        // For BabyBearExt4: 4
        4
    }
}

// AIR evaluation with LogUp support
impl<AB: AirBuilderWithPublicValues + AirBuilderWithLogUp> Air<AB> for MyAir
where
    AB::F: Field,
{
    fn eval(&self, builder: &mut AB) {
        // Your constraint implementation
    }
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
let aux = builder.logup_permutation();
let randomnesses = builder.logup_permutation_randomness();
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

#### c. Constrain the auxiliary trace structure

The auxiliary trace has 3 extension field columns (in base field representation):
- Columns 0..r_width: `tᵢ = 1/(r - xᵢ)`
- Columns r_width..2*r_width: `wᵢ = 1/(r - yᵢ)`
- Columns 2*r_width..3*r_width: Running sum

```rust
let (aux_local, aux_next) = (
    aux.row_slice(0).expect("Matrix is empty?"),
    aux.row_slice(1).expect("Matrix only has 1 row?"),
);

let r_width = <MyAir as MultiPhaseBaseAir<AB::F>>::num_randomness_in_base_field(self);
let w = AB::F::from_i8(11); // Extension field constant (e.g., 11 for BabyBearExt4)

// Constraint 1: t * (r - x_i) == 1
{
    let r_min_xi = ext_field_sub::<AB>(
        &randomnesses,
        &[xi.clone(), AB::Expr::ZERO, AB::Expr::ZERO, AB::Expr::ZERO],
    );

    let t_mul_r_min_xi = ext_field_mul::<AB>(
        &aux_local[..r_width]
            .iter()
            .map(|x| x.clone().into())
            .collect::<Vec<_>>(),
        &r_min_xi,
        &w,
    );

    builder.assert_one(t_mul_r_min_xi[0].clone());
    builder.assert_zero(t_mul_r_min_xi[1].clone());
    builder.assert_zero(t_mul_r_min_xi[2].clone());
    builder.assert_zero(t_mul_r_min_xi[3].clone());
}

// Constraint 2: w * (r - y_i) == 1
{
    let r_min_yi = ext_field_sub::<AB>(
        &randomnesses,
        &[yi.clone(), AB::Expr::ZERO, AB::Expr::ZERO, AB::Expr::ZERO],
    );

    let w_mul_r_min_yi = ext_field_mul::<AB>(
        &aux_local[r_width..2 * r_width]
            .iter()
            .map(|x| x.clone().into())
            .collect::<Vec<_>>(),
        &r_min_yi,
        &w,
    );

    builder.assert_one(w_mul_r_min_yi[0].clone());
    builder.assert_zero(w_mul_r_min_yi[1].clone());
    builder.assert_zero(w_mul_r_min_yi[2].clone());
    builder.assert_zero(w_mul_r_min_yi[3].clone());
}

// Constraint 3: Running sum constraints
for i in 0..r_width {
    let ti = aux_local[i].clone().into();
    let wi = aux_local[r_width + i].clone().into();
    let next_ti = aux_next[i].clone().into();
    let next_wi = aux_next[r_width + i].clone().into();
    let running_sum = aux_local[2 * r_width + i].clone().into();
    let next_running_sum = aux_next[2 * r_width + i].clone().into();

    // First row: running_sum = ti - wi
    builder
        .when_first_row()
        .assert_eq(running_sum.clone(), ti - wi);

    // Transition: next_running_sum = running_sum + next_ti - next_wi
    builder
        .when_transition()
        .assert_eq(next_running_sum, running_sum.clone() + next_ti - next_wi);

    // Last row: running_sum == 0 (proves permutation)
    builder.when_last_row().assert_zero(running_sum);
}
```

### 4. Proving and Verifying

The prover and verifier handle the auxiliary trace generation automatically:

```rust
use p3_uni_stark::{prove, verify};

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

impl<F> MultiPhaseBaseAir<F> for MyAir {
    fn aux_width_in_base_field(&self) -> usize { 12 }  // For BabyBearExt4
    fn num_randomness_in_base_field(&self) -> usize { 4 }  // For BabyBearExt4
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
