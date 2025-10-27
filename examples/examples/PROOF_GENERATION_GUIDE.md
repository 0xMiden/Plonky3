# Complete Guide to STARK Proof Generation in Plonky3

This guide demonstrates how to generate and verify STARK proofs using the Plonky3 framework, with a focus on permutation arguments and the `PermutationAirBuilder` trait.

## Overview

We provide three examples demonstrating different aspects of proof generation:

1. **`permutation_air.rs`** - Demonstrates `PermutationAirBuilder` usage (AIR definition only)
2. **`permutation_air_prove.rs`** - Shows the structure for full proof generation with `PermutationAirBuilder`
3. **`permutation_proof_complete.rs`** - **Complete working example** of proof generation and verification

## Running the Examples

```bash
# Example 1: PermutationAirBuilder demonstration (no proving)
cargo run --example permutation_air

# Example 2: Proof structure with PermutationAirBuilder (setup only)
cargo run --example permutation_air_prove

# Example 3: Complete working proof (generates and verifies!)
cargo run --example permutation_proof_complete
```

## Complete Proof Generation Pipeline

Based on `permutation_proof_complete.rs`, here's the complete pipeline:

### Step 1: Set up Type Aliases

```rust
type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;
type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs = MerkleTreeMmcs<
    <Val as Field>::Packing,
    <Val as Field>::Packing,
    MyHash,
    MyCompress,
    8
>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Dft = Radix2DitParallel<Val>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;
```

### Step 2: Create STARK Configuration

```rust
fn create_stark_config() -> MyConfig {
    // 1. Initialize permutation for Poseidon2
    let mut rng = SmallRng::seed_from_u64(42);
    let perm = Perm::new_from_rng_128(&mut rng);

    // 2. Set up hash function and compression
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());

    // 3. Create Merkle tree commitment scheme
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    // 4. Create DFT (Discrete Fourier Transform)
    let dft = Dft::default();

    // 5. Configure FRI parameters
    let fri_params = FriParameters {
        log_blowup: 1,           // Blowup factor: 2x
        log_final_poly_len: 0,   // Final polynomial length
        num_queries: 40,         // Number of queries (security)
        proof_of_work_bits: 8,   // PoW difficulty
        mmcs: challenge_mmcs,
    };

    // 6. Create PCS (Polynomial Commitment Scheme)
    let pcs = Pcs::new(dft, val_mmcs, fri_params);

    // 7. Create challenger (Fiat-Shamir)
    let challenger = Challenger::new(perm);

    // 8. Combine into config
    MyConfig::new(pcs, challenger)
}
```

### Step 3: Define the AIR

```rust
#[derive(Debug, Clone)]
pub struct SimplePermutationAir {
    pub num_elements: usize,
}

impl<F: Field> BaseAir<F> for SimplePermutationAir {
    fn width(&self) -> usize {
        self.num_elements * 2
    }
}

impl<AB: AirBuilder> Air<AB> for SimplePermutationAir
where
    AB::F: Field,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let main_local = main.row_slice(0).expect("main trace is empty");
        let main_local = &*main_local;
        let main_next = main.row_slice(1).expect("has only 1 row");
        let main_next = &*main_next;

        // Define constraints...
        builder.when_first_row().assert_eq(/* ... */);
        builder.when_transition().assert_eq(/* ... */);
        // etc.
    }
}
```

### Step 4: Generate Execution Trace

```rust
fn generate_trace() -> RowMajorMatrix<Val> {
    let trace_height = 8;  // Must be power of 2
    let trace_width = 16;

    let mut trace_values = Vec::with_capacity(trace_height * trace_width);

    // Fill trace with values that satisfy the AIR constraints
    for _ in 0..trace_height {
        // Add row values...
        trace_values.extend(/* your values */);
    }

    RowMajorMatrix::new(trace_values, trace_width)
}
```

### Step 5: Generate Proof

```rust
use p3_uni_stark::{prove, verify};

let config = create_stark_config();
let air = SimplePermutationAir { num_elements: 8 };
let trace = generate_trace();
let public_values = vec![];  // Optional public inputs

// Generate proof
let proof = prove(&config, &air, trace, &public_values);
```

### Step 6: Verify Proof

```rust
let result = verify(&config, &air, &proof, &public_values);

match result {
    Ok(()) => println!("✓ Proof verified!"),
    Err(e) => println!("✗ Verification failed: {:?}", e),
}
```

## Key Configuration Choices

### Field Selection

- **BabyBear** (`p = 2^31 - 2^27 + 1`): Fast, good for most applications
- **Goldilocks** (`p = 2^64 - 2^32 + 1`): More security, larger field
- **Mersenne31** (`p = 2^31 - 1`): Alternative 31-bit field

### Hash Function Selection

- **Poseidon2**: Fast, ZK-friendly (recommended)
- **Keccak**: Standard, well-studied
- **SHA256**: Maximum compatibility

### PCS (Polynomial Commitment Scheme) Selection

- **TwoAdicFriPcs**: Standard FRI-based PCS (recommended)
- **CirclePcs**: Circle STARK variant
- **TrivialPcs**: Testing only, not secure

### FRI Parameters

```rust
FriParameters {
    log_blowup: 1,              // Blowup = 2^log_blowup
    log_final_poly_len: 0,      // Final polynomial size = 2^log_final_poly_len
    num_queries: 40,            // More queries = more security (typ. 20-80)
    proof_of_work_bits: 8,      // PoW bits (typ. 8-16)
    mmcs: challenge_mmcs,
}
```

**Important**: Must satisfy `log_trace_height > log_final_poly_len + log_blowup`

For trace height 8: `log2(8) = 3`, so `3 > 0 + 1` ✓

## PermutationAirBuilder Integration

While `PermutationAirBuilder` is defined, full prover support requires additional infrastructure. When implemented, the pattern will be:

```rust
impl<AB: PermutationAirBuilder> Air<AB> for MyAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let perm = builder.permutation();           // Extension field trace
        let randomness = builder.permutation_randomness();  // Random challenges

        // Use extension field operations
        builder.assert_eq_ext(expr1, expr2);
        builder.assert_zero_ext(expr);
    }
}
```

## Complete Example Output

```
=== Complete STARK Proof Example ===

Example: Proving and Verifying a Simple AIR
=============================================

✓ STARK configuration created
✓ AIR created with width: 16
✓ Trace generated: 8x16
  Trace values: [5, 3, 7, 1, 9, 11, 13, 2]

Generating proof...
✓ Proof generated successfully!
  Estimated proof size: 288 bytes

Verifying proof...
✓ Proof verified successfully!

✓ All examples completed successfully!
```

## Debugging Tips

### Constraint Checking

In debug mode, Plonky3 automatically checks constraints:

```rust
// Automatically runs in debug builds
// Checks all constraints evaluate to 0
prove(&config, &air, trace, &public_values);
```

If constraints fail:
```
assertion `left == right` failed: constraints had nonzero value on row 0
  left: 24
 right: 0
```

This means constraint evaluated to 24 instead of 0 on row 0.

### Common Issues

1. **Trace height not power of 2**: Use `8, 16, 32, 64...`

2. **FRI parameter constraint violation**:
   ```
   assertion failed: log_min_height > params.log_final_poly_len + params.log_blowup
   ```
   Solution: Reduce `log_final_poly_len` or increase trace height

3. **Type annotation issues**: Use fully qualified syntax:
   ```rust
   <MyAir as BaseAir<Val>>::width(&air)
   ```

## Performance Considerations

### Proof Size

Typical proof sizes:
- Small traces (8-64 rows): 200-500 bytes
- Medium traces (1K-8K rows): 1-10 KB
- Large traces (64K+ rows): 10-100 KB

Proof size depends on:
- `num_queries` (more queries = larger proof)
- Trace width (more columns = larger proof)
- `log_blowup` (higher blowup = slightly larger)

### Proving Time

Factors affecting proving time:
- Trace size (height × width)
- Constraint complexity
- Hash function choice (Poseidon2 fastest)
- Hardware (CPU cores for parallel operations)

### Security Level

For production use:
```rust
FriParameters {
    log_blowup: 1,           // 2x blowup (minimum)
    log_final_poly_len: 0,   // Adjust based on trace size
    num_queries: 40,         // 40+ queries recommended
    proof_of_work_bits: 16,  // 16 bits for production
    mmcs: challenge_mmcs,
}
```

## References

- [Plonky3 Repository](https://github.com/Plonky3/Plonky3)
- [STARK Anatomy](https://aszepieniec.github.io/stark-anatomy/)
- [FRI Protocol](https://drops.dagstuhl.de/opus/volltexte/2018/9018/)
- [Permutation Arguments (Plookup)](https://eprint.iacr.org/2020/315)

## Next Steps

1. **Modify the AIR**: Add your own constraints in `eval()`
2. **Generate custom traces**: Create traces for your computation
3. **Experiment with parameters**: Try different field sizes, hash functions
4. **Implement PermutationAirBuilder**: Add support in the prover infrastructure
5. **Benchmark**: Measure proof size and generation time for your use case

## Complete Working Examples in This Directory

- ✅ **permutation_proof_complete.rs** - Full working proof pipeline
- 📚 **permutation_air.rs** - PermutationAirBuilder demonstration
- 🏗️ **permutation_air_prove.rs** - Proof structure template
