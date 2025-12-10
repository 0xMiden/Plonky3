# Miden-Prover Integration Guide

This guide explains how to integrate with the `miden-prover` and `miden-air` crates in this Plonky3 fork. These crates implement **multi-phase proving** with auxiliary traces for permutation arguments and lookups.

## Table of Contents
- [Overview](#overview)
- [Core Concepts](#core-concepts)
- [Current API (Today)](#current-api-today)
- [Complete Example](#complete-example)
- [Future Changes](#future-changes)
- [Migration Strategy](#migration-strategy)

---

## Overview

### What's Different in This Fork

This fork adds **two-phase proving** to Plonky3:

1. **Phase 1**: Generate and commit to main trace
2. **Challenge sampling**: Get random challenges from verifier
3. **Phase 2**: Build auxiliary trace using main trace + challenges
4. **Quotient generation**: Evaluate constraints over both traces

This enables:
- LogUp permutation arguments
- Lookup arguments with bus columns
- Multi-set equality checks
- Any constraint system needing verifier randomness

### Key Additions

**New crates:**
- `miden-air` - Unified AIR trait for Miden VM
- `miden-prover` - Prover/verifier with Miden-specific optimizations

**Extended traits:**
- `BaseAirWithAuxTrace` - Adds auxiliary trace support to any AIR
- Periodic column support in folders

**Proven in:**
- PR #4, #13, #14
- Used by Miden VM integration

---

## Core Concepts

### Trace Types

Your STARK proof can have up to 4 trace types:

1. **Preprocessed (optional)**: Fixed columns known to both prover and verifier
2. **Main (required)**: Execution trace computed by prover
3. **Auxiliary (optional)**: Columns computed after challenge sampling
4. **Randomness (optional)**: Challenge values from verifier

### Auxiliary Trace Flow

```
┌─────────────┐
│ Main Trace  │  Prover computes execution trace
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Commit    │  Send commitment to verifier/challenger
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Challenges │  Sample random values via Fiat-Shamir
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Aux Trace  │  Build auxiliary columns using main + challenges
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Commit    │  Send aux commitment
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Quotient    │  Evaluate constraints on all traces
└─────────────┘
```

---

## Current API (Today)

### Step 1: Implement Your AIR

All AIRs must implement `BaseAirWithAuxTrace`:

```rust
use p3_air::{Air, AirBuilder, BaseAir, BaseAirWithAuxTrace};
use p3_matrix::dense::RowMajorMatrix;

struct MyAir {
    // Your AIR parameters
}

// Basic trace dimensions
impl<F: Field> BaseAir<F> for MyAir {
    fn width(&self) -> usize {
        4  // Number of main trace columns
    }
}

// Auxiliary trace support
impl<F: Field, EF: ExtensionField<F>> BaseAirWithAuxTrace<F, EF> for MyAir {
    fn num_randomness(&self) -> usize {
        2  // Number of challenge values needed (0 for single-phase)
    }

    fn aux_width(&self) -> usize {
        2  // Number of auxiliary columns in EF (0 for single-phase)
    }

    fn build_aux_trace(
        &self,
        main_trace: &RowMajorMatrix<F>,
        challenges: &[EF],
    ) -> RowMajorMatrix<EF> {
        // Build auxiliary trace from main trace and challenges
        let height = main_trace.height();
        let mut aux_trace = RowMajorMatrix::new(vec![EF::ZERO; height * self.aux_width()], self.aux_width());

        // Your logic here: compute aux columns using main trace + challenges

        aux_trace
    }
}

// Constraints
impl<AB: AirBuilder> Air<AB> for MyAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let next = main.row_slice(1);

        // Main trace constraints
        // ...

        // If you have auxiliary trace:
        if self.aux_width() > 0 {
            let aux = builder.aux();  // Currently returns Option<ViewPair> ⚠️
            let randomness = builder.randomness();  // Challenge values

            // Auxiliary trace constraints
            // ...
        }
    }
}
```

**⚠️ Current State Notes:**

1. **All AIRs must implement `BaseAirWithAuxTrace`** even if they don't use auxiliary traces. Return 0 for both methods:
   ```rust
   impl<F, EF> BaseAirWithAuxTrace<F, EF> for MySimpleAir {}  // Uses defaults
   ```
   See [Issue #29](https://github.com/0xMiden/Plonky3/issues/29) - may split into separate prove functions in future.

2. **Metadata appears in two places** ⚠️
   - You implement it on your AIR (shown above)
   - `StarkGenericConfig` also has these methods

   See [Issue #21](https://github.com/0xMiden/Plonky3/issues/21) - will move exclusively to AIR trait.

3. **Option types for aux traces** ⚠️
   - `builder.aux()` currently returns `Option<ViewPair>`
   - Must pattern match even if you know aux trace exists

   See [Issue #22](https://github.com/0xMiden/Plonky3/issues/22) - may change to zero-width views.

### Step 2: Configure Your STARK

```rust
use p3_uni_stark::{StarkConfig, StarkGenericConfig};
use p3_commit::ExtensionMmcs;
use p3_field::extension::BinomialExtensionField;
use p3_goldilocks::Goldilocks;

type Val = Goldilocks;
type Challenge = BinomialExtensionField<Val, 2>;

// Your config (PCS, hash functions, etc.)
let config = StarkConfig::new(/* ... */);
```

**⚠️ Current State:** Config currently has `aux_width()` and `num_randomness()` methods. These will be removed ([Issue #21](https://github.com/0xMiden/Plonky3/issues/21)).

### Step 3: Prove

```rust
use p3_uni_stark::prove;

let air = MyAir { /* ... */ };
let main_trace: RowMajorMatrix<Val> = /* your execution trace */;
let public_values = vec![/* public inputs */];

let proof = prove(
    &config,
    &air,
    main_trace,
    &public_values,
);
```

The prover automatically:
- Commits to main trace
- Samples `num_randomness()` challenges if > 0
- Calls `build_aux_trace()` if `aux_width() > 0`
- Commits to aux trace
- Generates quotient polynomial over all traces

### Step 4: Verify

```rust
use p3_uni_stark::verify;

verify(
    &config,
    &air,
    &proof,
    &public_values,
)?;
```

---

## Complete Example

### Single-Phase AIR (No Aux Trace)

```rust
use p3_air::{Air, AirBuilder, BaseAir, BaseAirWithAuxTrace};
use p3_field::Field;

struct FibonacciAir;

impl<F: Field> BaseAir<F> for FibonacciAir {
    fn width(&self) -> usize { 2 }
}

// Dummy implementation - uses defaults (both return 0)
impl<F: Field, EF> BaseAirWithAuxTrace<F, EF> for FibonacciAir {}

impl<AB: AirBuilder> Air<AB> for FibonacciAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let next = main.row_slice(1);

        // local[1] = local[0] + local[1]
        builder.when_transition().assert_eq(
            next[0],
            local[0] + local[1]
        );
    }
}
```

### Multi-Phase AIR (With Aux Trace)

```rust
use p3_air::{Air, AirBuilder, BaseAir, BaseAirWithAuxTrace};
use p3_matrix::dense::RowMajorMatrix;
use p3_field::{Field, ExtensionField};

struct PermutationAir {
    alpha_width: usize,  // Num challenge values
    bus_width: usize,    // Num bus columns
}

impl<F: Field> BaseAir<F> for PermutationAir {
    fn width(&self) -> usize { 4 }  // Main trace columns
}

impl<F: Field, EF: ExtensionField<F>> BaseAirWithAuxTrace<F, EF> for PermutationAir {
    fn num_randomness(&self) -> usize {
        self.alpha_width  // How many challenges we need
    }

    fn aux_width(&self) -> usize {
        self.bus_width  // Auxiliary columns (in EF)
    }

    fn build_aux_trace(
        &self,
        main_trace: &RowMajorMatrix<F>,
        challenges: &[EF],
    ) -> RowMajorMatrix<EF> {
        let height = main_trace.height();
        let width = self.aux_width();
        let mut aux_values = vec![EF::ZERO; height * width];

        // Example: Running sum for permutation check
        let alpha = challenges[0];
        let mut running_sum = EF::ZERO;

        for i in 0..height {
            let main_row = main_trace.row_slice(i);

            // Compute fingerprint: sum of (value - alpha)
            let fingerprint = EF::from_base(main_row[0]) - alpha;
            running_sum += fingerprint.inverse();

            aux_values[i * width] = running_sum;
        }

        RowMajorMatrix::new(aux_values, width)
    }
}

impl<AB: AirBuilder> Air<AB> for PermutationAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_main = main.row_slice(0);
        let next_main = main.row_slice(1);

        // Main trace constraints
        // ...

        // Auxiliary trace constraints
        let aux = builder.aux();
        let randomness = builder.randomness();

        if let Some(aux_trace) = aux {  // ⚠️ Pattern match currently required
            let local_aux = aux_trace.row_slice(0);
            let next_aux = aux_trace.row_slice(1);
            let alpha = randomness[0];

            // Running sum constraint
            let fingerprint = AB::Expr::from_canonical_u32(1)
                / (local_main[0] - alpha);

            builder.when_transition().assert_eq(
                next_aux[0],
                local_aux[0] + fingerprint
            );
        }
    }
}
```

### Using Miden-Specific Traits

For Miden VM integration, use the unified `MidenAir` trait:

```rust
use miden_air::MidenAir;
use miden_prover::{prove, verify};

struct MyMidenAir {
    // Your AIR
}

// Implement MidenAir (combines all required trait bounds)
impl MidenAir for MyMidenAir {
    // Single unified trait for prover/verifier
}

// Use Miden prover/verifier
let proof = miden_prover::prove(&config, &air, trace, &public_values);
miden_prover::verify(&config, &air, &proof, &public_values)?;
```

See `miden-air/src/air.rs` and `miden-prover/src/prover.rs` for details.

---

## Future Changes

### High Priority API Changes

#### 1. Metadata Location ([Issue #21](https://github.com/0xMiden/Plonky3/issues/21))

**What will change:**
`StarkGenericConfig` will no longer have `aux_width()` and `num_randomness()`. These will only exist on `BaseAirWithAuxTrace`.

**Migration:**
```rust
// Before (current):
let num_challenges = config.num_randomness();

// After (future):
let num_challenges = air.num_randomness();
```

**Impact:** Breaking change for config implementations. Prover/verifier code updates needed.

**Timeline:** Likely before AirScript integration (Issue #5).

#### 2. Option vs Zero-Width ([Issue #22](https://github.com/0xMiden/Plonky3/issues/22))

**What will change:**
`builder.aux()` will return `ViewPair` (width=0 when absent) instead of `Option<ViewPair>`.

**Migration:**
```rust
// Before (current):
if let Some(aux) = builder.aux() {
    let local = aux.row_slice(0);
    // ...
}

// After (future):
let aux = builder.aux();  // Always present
if aux.width() > 0 {      // Check width if needed
    let local = aux.row_slice(0);
    // ...
}
// Or just use it directly - zero-width is safe
```

**Impact:** Breaking change for all AIR implementations with auxiliary traces.

**Timeline:** Under discussion. Team deciding between backward compatibility and simplicity.

#### 3. Extension Field Type Safety ([Issue #24](https://github.com/0xMiden/Plonky3/issues/24))

**What will change:**
`SymbolicAirBuilder` may use `EF` (extension field) throughout instead of `F` (base field).

**Migration:** Likely transparent if you use `AB::Expr` throughout. May affect manual type annotations.

**Impact:** Medium - affects symbolic constraint building.

**Timeline:** After checking if upstream Plonky3 resolved the generic issues.

### Medium Priority Changes

#### 4. Prove Function Split ([Issue #29](https://github.com/0xMiden/Plonky3/issues/29))

**What might change:**
Two separate prove functions to avoid dummy trait implementations:

```rust
// For single-phase AIRs
prove(&config, &air, trace, &public_values);

// For multi-phase AIRs
prove_with_aux(&config, &air, trace, &public_values);
```

**Impact:** Would eliminate need for dummy `BaseAirWithAuxTrace` implementations.

**Timeline:** Design discussion ongoing. Current unified approach may stay.

#### 5. Separate Miden Concerns ([Issue #28](https://github.com/0xMiden/Plonky3/issues/28))

**What will change:**
`miden-prover` folder logic may move to separate module/crate (like Winterfell).

**Impact:** Import paths change, cleaner architecture for Miden VM.

**Timeline:** Follow-up to current integration work.

#### 6. Code Quality Issues

- **Dead code cleanup** ([Issue #27](https://github.com/0xMiden/Plonky3/issues/27)) - `#[allow(dead_code)]` → `#[cfg(debug_assertions)]`
- **Dependency cleanup** ([Issue #30](https://github.com/0xMiden/Plonky3/issues/30)) - Remove unused crates from `miden-prover`
- **Function duplication** ([Issue #23](https://github.com/0xMiden/Plonky3/issues/23)) - Unify `row_to_ext` implementations

**Impact:** Low - internal changes, mostly transparent to API users.

### Low Priority (Performance)

- **Periodic column optimization** ([Issue #25](https://github.com/0xMiden/Plonky3/issues/25))
- **FRI folding optimization** ([Issue #26](https://github.com/0xMiden/Plonky3/issues/26))

**Impact:** Performance improvements, no API changes.

---

## Migration Strategy

### For New Projects (Starting Today)

✅ **Safe to use now:**
- Multi-phase proving with auxiliary traces
- Periodic columns
- LogUp and lookup arguments
- Miden-specific prover/verifier

⚠️ **Expect changes:**
- Where you query `aux_width()` / `num_randomness()` (config vs air)
- Pattern matching on `Option<ViewPair>` for aux traces
- Import paths if using `miden-prover` internals

**Recommendation:**
- Implement `BaseAirWithAuxTrace` on your AIR (required)
- Avoid querying config metadata directly - use air methods
- Write tests that exercise both single-phase and multi-phase paths
- Follow issues #21 and #22 for API updates

### For Existing Uni-Stark Code

**Migration from upstream Plonky3:**

Your single-phase AIRs need minimal changes:

```rust
// Add this to your existing AIR:
impl<F: Field, EF> BaseAirWithAuxTrace<F, EF> for YourAir {}
```

That's it. Defaults handle everything else.

**If you want multi-phase:**
1. Add `num_randomness()` - how many challenges you need
2. Add `aux_width()` - how many auxiliary columns (in EF)
3. Implement `build_aux_trace()` - construct aux from main + challenges
4. Update `eval()` to include auxiliary trace constraints

### Tracking Changes

**Subscribe to these issues:**
- [#21 - Metadata location](https://github.com/0xMiden/Plonky3/issues/21) - HIGH: Will break config implementers
- [#22 - Option removal](https://github.com/0xMiden/Plonky3/issues/22) - HIGH: Will break constraint evaluation
- [#24 - Extension field types](https://github.com/0xMiden/Plonky3/issues/24) - HIGH: Potential soundness fix
- [#29 - Prove function split](https://github.com/0xMiden/Plonky3/issues/29) - MEDIUM: May change prove() API

**Backport tracking:**
- [#18 - Upstream changes](https://github.com/0xMiden/Plonky3/issues/18) - Lookup integration from upstream may conflict

### Upstream Synchronization

This fork diverged at upstream commit `5132bc78` (Nov 13, 2025). 24 upstream commits not yet backported.

**Critical upcoming:**
- Upstream PR #1165 - Lookup integration (different design than our aux traces)
- Upstream PR #1150 - Preprocessed trace/VK infrastructure

See [Issue #18](https://github.com/0xMiden/Plonky3/issues/18) for backport tracking.

---

## Additional Resources

### Examples

- **Single-phase:** `blake3-air/src/air.rs`, `poseidon2-air/src/air.rs`
- **Multi-phase:** See test suite in `uni-stark/tests/two_phase.rs` (429 lines)
- **Miden integration:** `miden-prover/src/prover.rs`, `miden-air/src/air.rs`

### Key Files

**AIR traits:**
- `air/src/air.rs` - Core trait definitions including `BaseAirWithAuxTrace`
- `air/src/virtual_column.rs` - Extended column system (preprocessed, main, aux, randomness)

**Prover/Verifier:**
- `uni-stark/src/prover.rs:132-303` - Two-phase proving flow
- `uni-stark/src/verifier.rs:230-238` - Auxiliary trace verification
- `uni-stark/src/folder.rs` - Constraint folders with aux support

**Miden-specific:**
- `miden-air/src/air.rs` - Unified `MidenAir` trait
- `miden-prover/src/prover.rs` - Miden prover implementation
- `miden-prover/src/periodic_tables.rs` - Periodic column optimization

### Development Commands

```bash
# Build
cargo build --all-targets

# Test (basic)
cargo test

# Test with parallel features
cargo test --features parallel

# Run clippy (CI requirement)
cargo +stable clippy --all-targets -- -D warnings

# Format check
cargo +nightly fmt --all -- --check

# Run examples
RUSTFLAGS="-Ctarget-cpu=native" cargo run --example prove_prime_field_31 --release --features parallel
```

See `CLAUDE.md` for full development setup.

---

## Questions?

- **General Plonky3:** See upstream docs at https://github.com/Plonky3/Plonky3
- **Multi-phase proving:** Review PR #4 design doc and discussion
- **AirScript integration:** Track Issue #5
- **API changes:** Follow issues #21, #22, #24

For Miden VM specific integration questions, see `miden-air` and `miden-prover` crate documentation.
