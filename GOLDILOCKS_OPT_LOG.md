# Goldilocks Scalar Optimization Log

Platform: Apple M4 (aarch64), macOS

## Baseline (Commit 0)

### Benchmarks

```
mul-latency/100 Goldilocks          time: [354-365 ns]    (~3.6 ns/op)
mul-throughput/25 Goldilocks         time: [197-198 ns]    (~0.8 ns/op)
Goldilocks square                    time: [895-910 ps]
Goldilocks inv                       time: [127-129 ns]
Goldilocks sum/200, 4                time: [362-369 ns]
Goldilocks tree sum/200, 4           time: [361-370 ns]
Goldilocks tree sum/200, 5           time: [426-430 ns]
Goldilocks tree sum/200, 6           time: [494-513 ns]
Goldilocks tree sum/200, 7           time: [549-559 ns]
Goldilocks dot product/1             time: [1.10-1.13 ns]
Goldilocks dot product/2             time: [1.51-1.54 ns]
Goldilocks dot product/3             time: [2.39-2.42 ns]
Goldilocks dot product/4             time: [3.05-3.10 ns]
Goldilocks dot product/5             time: [3.66-3.69 ns]
Goldilocks dot product/6             time: [4.35-4.41 ns]
add-latency/2000 Goldilocks         time: [2.55-2.67 µs]  (~1.3 ns/op)
add-throughput/200 Goldilocks        time: [985-1016 ns]   (~0.5 ns/op)
sub-latency/2000 Goldilocks          time: [2.55-2.64 µs]  (~1.3 ns/op)
sub-throughput/200 Goldilocks        time: [981-1001 ns]   (~0.5 ns/op)
Goldilocks halve. Num Reps: 200      time: [53.1-54.1 ns]  (~0.27 ns/op)
Goldilocks mul_2exp_u64 1            time: [177-184 ns]
Goldilocks mul_2exp_u64 10           time: [177-180 ns]
Goldilocks mul_2exp_u64 32           time: [177-182 ns]
Goldilocks mul_2exp_u64 63           time: [714-780 ns]
Goldilocks div_2exp_u64 1            time: [241-248 ns]
Goldilocks div_2exp_u64 10           time: [241-249 ns]
neg-latency/2000 Goldilocks         time: [5.04-5.05 µs]  (~2.5 ns/op)
neg-throughput/200 Goldilocks        time: [597-718 ns]    (~3.0-3.6 ns/op)
double-latency/2000 Goldilocks       time: [1.70-1.75 µs]  (~0.85 ns/op)
double-throughput/200 Goldilocks     time: [597-603 ns]    (~3.0 ns/op)
Goldilocks batched_lc chunk=8        time: [113.4-113.5 ns]
Goldilocks batched_lc chunk=16       time: [111.3-111.6 ns]
7th_root                             time: [299-303 ns]
```

### Assembly Notes

All scalar Goldilocks operations are inlined by LLVM on aarch64. Assembly
for each operation is captured inline with the corresponding optimization
commit below by inspecting the benchmark assembly or using `--emit=asm`.

---

## Opt 1: Branchless `neg()` -- SKIP

### Approach A: `ORDER.wrapping_sub(value)` (no canonicalization)
- **Result:** INCORRECT. Fails for non-canonical inputs (v > ORDER).
  `wrapping_sub` gives `2^64 - (v - ORDER)`, which represents `NEG_ORDER - (v - ORDER) mod p`,
  not `-(v mod p)`. The packed NEON neg also canonicalizes before negating.

### Approach B: Branchless canonicalization + subtract
```rust
let mask = ((c >= ORDER) as u64).wrapping_neg();
let canonical = c.wrapping_sub(ORDER & mask);
Self::new(ORDER - canonical)
```
- **Benchmarks (before -> after):**
  - neg-latency/2000: 5.04 µs -> 5.92 µs (**+17% REGRESSION**)
  - neg-throughput/200: ~710 ns -> ~711 ns (neutral)

### Conclusion: SKIP
The branch in `as_canonical_u64` is taken ~2^{-32} of the time, making it
almost perfectly predicted. Branchless mask arithmetic adds 2-3 cycles of
latency to every call for no benefit. On aarch64 (Apple M4), the branch
predictor wins decisively.

---

## Opt 2: Branchless `halve()` -- KEEP

### Implementation
Replaced `halve_u64::<P>(self.value)` (which uses `if x & 1 == 0`) with
branchless mask arithmetic: `(x >> 1) + ((0u64.wrapping_sub(x & 1)) & HALF_P_PLUS_1)`.

### Benchmarks (before -> after)
- Goldilocks halve/200: 53.1 ns -> 42.3 ns (**-20.3%**)

### Conclusion: KEEP
The parity branch is 50/50 unpredictable for random inputs, causing ~50%
mispredict rate. Branchless mask form eliminates all mispredictions. This
matches the pattern already used in NEON and AVX2 packed halve.

---

## Opt 3: Specialized `double()` -- KEEP (neutral, cleaner)

### Benchmarks (before -> after)
- double-latency/2000: 1.70-1.75 µs -> 1.71 µs (neutral)
- double-throughput/200: 597-603 ns -> 600 ns (neutral)

### Conclusion: KEEP
Neutral performance. LLVM already optimized the default `self + self` well.
Override is kept for code clarity (explicit same-register assumption) and to
provide stronger `assume` hints for callers with canonical inputs.

---

## Opt 4: Branchless `is_zero()` -- KEEP

Changed `||` to `|` to avoid short-circuit branch.

### Conclusion: KEEP (trivial change, eliminates branch)

---

## Opt 5: Specialized `square()` -- KEEP (neutral, cleaner)

### Benchmarks (before -> after)
- Goldilocks square: 895-910 ps -> 899 ps (neutral)
- 7th_root: 299-303 ns -> 300 ns (neutral)

### Conclusion: KEEP
Neutral performance. LLVM already recognized the squaring pattern.
Override kept for explicitness.

---

## Opt 6+7: Extend `sum_array` N=4..7 and `dot_product` N=3,4,5 -- SKIP

### sum_array tree summation (before -> after)
- tree sum/200, N=4: 361-370 ns -> 398-401 ns (**+10% regression**)
- tree sum/200, N=5: 426-430 ns -> 522-524 ns (**+22% regression**)
- tree sum/200, N=6: 494-513 ns -> 586-590 ns (**+15% regression**)
- tree sum/200, N=7: 549-559 ns -> 682-687 ns (**+24% regression**)

### dot_product OFFSET-correction unrolling (before -> after)
- dot product/3: 2.39-2.42 ns -> 3.24 ns (**+35% regression**)
- dot product/4: 3.05-3.10 ns -> 4.49 ns (**+45% regression**)
- dot product/5: 3.66-3.69 ns -> 5.83 ns (**+58% regression**)

### Conclusion: SKIP
Unlike MontyField31 (32-bit field where scalar add is cheap), Goldilocks (64-bit)
benefits more from the u128 delayed-reduction path in `Sum` and the generic fold
with the `2^96 ≡ -1` split/accumulate trick. The overhead of individual u64 adds
(with carry folding) is larger for 64-bit fields than for 32-bit fields, making
tree summation slower than batch u128 accumulation. Similarly, the per-pair
OFFSET-correction overhead in dot_product exceeds the savings from avoiding
the fold's split/accumulate.

---

## Opt 8: `div_2exp_u64(1)` -> `halve()` -- KEEP

### Benchmarks (before -> after)
- div_2exp_u64(1): 241-248 ns -> 42.7 ns (**-83%**)

### Conclusion: KEEP
Massive improvement. The current path for div_2exp(1) goes through
`mul_2exp_u64(191)` = full field multiply by `POWERS_OF_TWO[95]`.
Direct `halve()` avoids the multiply entirely.

---

## Opt 9: `mul_2exp_u64` shift fast paths -- SKIP

### Benchmarks (before -> after)
- mul_2exp(1): 177-184 ns -> 179-180 ns (neutral)
- mul_2exp(10): 177-180 ns -> 336-339 ns (**+88% regression**)
- mul_2exp(32): 177-182 ns -> 337-343 ns (**+89% regression**)
- mul_2exp(63): 714-780 ns -> 1124-1138 ns (**+44% regression**)

### Conclusion: SKIP
The `reduce128((value as u128) << exp)` path is significantly slower than
multiplying by a precomputed table entry. The table multiply benefits from
the compiler's highly optimized 64x64->128 multiply path, while shift+reduce128
introduces extra u128 arithmetic overhead. The existing table approach is already optimal.

---

## Opt 10: Branchless `reduce128` borrow path -- SKIP

### Benchmarks (before -> after)
- mul-latency/100: 354-365 ns -> 436-442 ns (**+21% regression**)
- mul-throughput/25: 197-198 ns -> 227-228 ns (**+15% regression**)

### Conclusion: SKIP
The borrow is exceedingly rare (~2^{-32}). The branched version with
`branch_hint()` is perfectly predicted. Making it branchless adds
`csel` latency to every multiply, which is a significant regression.
The comment "It is faster to branch" is confirmed correct on Apple M4.

---

## Opt GCD: Branchless `gcd_inner` for inversion -- SKIP

### Assembly analysis (branched version, aarch64)
LLVM already compiles `gcd_inner` to a semi-branchless form:
- The `if a < b` swap uses 4 `csel` instructions (no branch)
- Only the parity check `if a & 1 == 0` uses a real branch (`tbz`)
- Even path: 4 instructions; odd path: 14 instructions

The `tbz` branch has regularity (after subtraction, result is always even,
leading to runs of even iterations), so the predictor handles it well.

### Fully branchless version
Replaced both branches with mask arithmetic (XOR-swap, conditional subtract).
Each iteration: 20 instructions (first loop), 26 instructions (second loop).

### Benchmarks (branched -> branchless)
- Goldilocks inv: 128 ns -> 479 ns (**+274% regression**)

### Why branchless is slower
The branched version averages ~9 instructions/iteration (mix of 4-insn even
path and 14-insn odd path). The branchless version forces 20 insns on EVERY
iteration. Apple M4's branch predictor handles the `tbz` parity branch well
enough that the branch penalty is small compared to the extra instruction cost.

### Conclusion: SKIP
The existing `gcd_inner` is already well-optimized by LLVM (using `csel` for
the swap, only branching on parity). The branchless approach triples the cost.

---

## Opt 11: `mul_pow2_raw<K>` const-generic helper + `mul_2exp_u64` dispatch -- PARTIAL

### `mul_pow2_raw<K>` for K in 1..32
Added a `pub(crate)` const-generic helper that multiplies by `2^K` using pure
u64 shift+fold: `(x << K) + (x >> (64-K)) * NEG_ORDER`. No u128 arithmetic.
For K < 32, `hi * NEG_ORDER < 2^64`, so the final addition uses
`add_no_canonicalize_trashing_input` (pure u64 add with carry folding).

### Wiring into `mul_2exp_u64` -- SKIP
Attempted dispatching to `mul_pow2_raw::<K>` for K=2..8 via a match table
with `exp % 192` normalization. Regressed +168-208% due to the modulo
overhead and large match table penalizing the runtime benchmark.

The existing table-multiply approach (`*self * POWERS_OF_TWO[exp]`) is
optimal for runtime exp because LLVM resolves table lookups well and avoids
u128 widening for the power. The `mul_pow2_raw` helper is kept for future
callers that know K at compile time (const-generic context).

---

## Opt 12: `reduce128` shift-sub for `x_hi_lo * NEG_ORDER` -- SKIP (LLVM no-op)

### Assembly check
Inspecting `try_inverse` assembly shows LLVM already converts
`x * NEG_ORDER` (where `NEG_ORDER = 0xFFFFFFFF = 2^32 - 1`) into
`lsl + sub` (shift-left-32 then subtract original). No code change needed.

---

## Opt 13: Scalar `BATCHED_LC_CHUNK` tuning -- SKIP (marginal)

### Benchmarks
```
chunk=1:  139 ns
chunk=2:  128 ns
chunk=4:  122 ns
chunk=8:  114 ns (current default)
chunk=16: 112 ns (best)
chunk=32: 121 ns
chunk=64: 133 ns
```

chunk=16 is 2% better than chunk=8. Too marginal to justify overriding
the Algebra trait for scalar Goldilocks. The packed types already override
BATCHED_LC_CHUNK (AVX2=2, NEON=2, AVX512=4) where it matters.

---

## Representation & Trait Audit

### Methods already optimal (no change needed)
- `from_bool`: overridden with branchless `Self::new(b.into())`
- `zero_vec`: overridden with `flatten_to_base(vec![0u64; len])`
- `two_adic_generator`: O(1) table lookup `TWO_ADIC_GENERATORS[bits]`
- `to_unique_u64`: defaults to `as_canonical_u64()`, correct since
  representation is non-canonical (0 and ORDER both represent zero)
- `exp_power_of_2`: repeated `square()` -- no Goldilocks shortcut exists
- `exp_const_u64`: handles 0..7 with optimal addition chains

### Serialization (`RawDataSerializable`)
Uses `impl_raw_serializable_primefield64!()` macro. All stream methods call
`to_unique_u64()` which calls `as_canonical_u64()`. The canonicalization
branch is well-predicted (~2^{-32} taken rate). No scalar optimization
available -- vectorized canonicalization would be a packed-field change.

---

## Opt 14: aarch64 inline asm for `add_no_canonicalize_trashing_input` -- KEEP

### Assembly (Rust fallback -> inline asm)
Rust fallback compiles to `adds + mov #-1 + csel + add` (4 instructions).
Inline asm uses `adds + csetm + add` (3 instructions). Saves 1 instruction
per call by using `csetm` to produce `0xFFFFFFFF` directly from the carry flag,
matching the x86_64 `sbb` trick.

### Benchmarks
Neutral on Apple M4 (M4 can fuse/parallel mov+csel). But the instruction
reduction may help on narrower aarch64 chips.

### Conclusion: KEEP
Cleaner code, 1 fewer instruction, matches the pattern already used
in aarch64_neon/utils.rs `add_asm`.

---

### Representation invariant
- `value: u64` can be any value in `[0, 2^64)`
- Canonical range: `[0, ORDER)` where `ORDER = 2^64 - 2^32 + 1`
- Only redundancy: `0` and `ORDER` both represent zero
- `as_canonical_u64()`: single conditional subtract, well-predicted
- All arithmetic operations work correctly on non-canonical inputs
- Canonicalization only needed for: `PartialEq`, `Hash`, `Ord`,
  `Display`, `Debug`, `to_unique_u64`, `neg` (calls `as_canonical_u64`)

---

## Final State: Assembly Snapshot (aarch64)

### `add` (6 instructions, 0 branches)
```
adds   x8, x1, x0          ; sum = a + b
mov    w9, #-1              ; NEG_ORDER
csel   x10, x9, xzr, hs    ; adj = carry ? NEG_ORDER : 0
adds   x8, x10, x8         ; sum += adj
add    x9, x8, x9          ; sum_corr = sum + NEG_ORDER
csel   x0, x9, x8, hs      ; result = carry2 ? sum_corr : sum
```

### `reduce128` inner `add_no_canonicalize` (3 instructions)
```
adds   {result}, {x}, {y}
csetm  {adj:w}, cs          ; 0xFFFFFFFF on carry
add    {result}, {result}, {adj}
```

### `halve` (branchless, 5 instructions)
```
and    x8, x0, #1           ; lo_bit = x & 1
lsr    x9, x0, #1           ; half = x >> 1
neg    x8, x8               ; mask = -lo_bit
and    x8, x8, HALF         ; mask & HALF_P_PLUS_1
add    x0, x9, x8           ; half + correction
```

### GCD inner loop (parity branch + csel swap, ~9 avg insns/iter)
```
tbz    w9, #0, even_path    ; branch on parity (well-predicted)
cmp    x9, x13              ; compare a, b
csel   ...                  ; 4x conditional swap (branchless)
sub    ...                  ; a -= b
lsr    ...                  ; a >>= 1
sub    ...                  ; f0 -= f1
lsl    ...                  ; f1 <<= 1
```

---

## Opt 15: Multiply by small constants (4, 7) -- SKIP (LLVM already optimal)

### Assembly analysis
LLVM already converts `reduce128((x as u128) * c)` for small constant `c`
into near-optimal code:

| Constant | Low part | High part | Total |
|----------|----------|-----------|-------|
| ×3 | `add x, x, x lsl #1` | `umulh` + shift-sub | 8 insns |
| ×4 | `lsl #2` | `lsr #62` + shift-sub | 7 insns (no umulh!) |
| ×7 | `lsl #3; sub` | `umulh` + shift-sub | 8 insns |

Attempted umulh-free alternatives (shift + borrow detection) produced the
same or more instructions (8-9). Apple M4 `umulh` is 1-cycle pipelined,
so there is no benefit to avoiding it.

### Conclusion: SKIP
LLVM already generates optimal code for multiplication by compile-time
constants through the `reduce128` path. Extension field W=7 multiplication
is already 8 instructions with no room for improvement.

### double().double() vs shift+fold vs table multiply for ×4

| Method | Insns | Notes |
|--------|-------|-------|
| `mul_pow2_raw` (shift+fold) | 7 | `lsl #2 + lsr #62 + shift-sub + add_no_canon` |
| `reduce128(x * 4)` (const inlined) | 7 | LLVM produces identical code to shift+fold |
| `double().double()` | 11 | Two chained add-carry sequences |
| `*self * POWERS_OF_TWO[exp]` (runtime) | ~15 | Full `mul + umulh + reduce128` when constant not inlined |

**Key finding:** When LLVM can see the constant (compile-time known), it generates
optimal shift+fold code (7 insns). When the constant comes from a runtime table
lookup, it falls back to full `mul + umulh` (15 insns). `double().double()` is
the middle ground at 11 insns but has a serial dependency.

Adding runtime dispatch (if/else chain) to `mul_2exp_u64` regressed because
the extra branches add latency that exceeds the savings. The `mul_pow2_raw`
and `mul_pow2_raw_dyn` helpers are available for direct callers that know K.

---

## Comparison: Plonky3 vs Lambdaworks Goldilocks

Source: https://github.com/lambdaclass/lambdaworks/blob/main/crates/math/src/field/fields/u64_goldilocks_field.rs

### Identical implementations
- add, sub, mul, square, reduce128, canonicalize: same algorithms
- Both use branch_hint + assume for rare double-overflow paths
- Both use `reduce128((a as u128) * (b as u128))` for multiplication

### Plonky3 advantages
| Feature | Plonky3 | Lambdaworks |
|---------|---------|-------------|
| aarch64 `add_no_canonicalize` | `adds + csetm + add` (3 insns) | Rust fallback (4 insns) |
| x86 `add_no_canonicalize` | single correction (3 insns) | double correction (5 insns) |
| Inverse | Binary GCD (128 ns) | Fermat FLT chain 63sq+9mul (~72 ops) |
| halve | branchless mask | not implemented |
| double | specialized with assume hints | delegates to add(a,a) |
| mul_by_7 | LLVM-optimized `x * F::new(7)` (8 insns) | `double.double.double - x` (3 doubles + sub) |
| mul_2exp_u64 | precomputed 96-entry table | not implemented |

### Lambdaworks advantages
| Feature | Lambdaworks | Plonky3 |
|---------|-------------|---------|
| x86 full mul asm | hand-written `mul` + reduction | relies on LLVM (similar quality) |
| x86 MULX (BMI2) | available | not implemented |
| Fp2 mul | Karatsuba (3 base muls) | dot_product (4 base muls, but u128 batched) |
| Fp2 square | hand-written (2sq+1mul) | generic (uses dot_product) |
| Fp3 mul | Karatsuba (6 muls) | not specifically optimized |

### Key takeaway
The implementations are very close. Plonky3's main edge is the binary GCD
inversion (much faster than Fermat FLT) and the branchless halve. Lambdaworks'
edge is the Karatsuba extension field arithmetic (fewer base muls) and the
x86 MULX support for BMI2 CPUs.

The Fp2 Karatsuba vs dot_product tradeoff is roughly neutral: Karatsuba saves
1 base mul but adds 3 add/sub ops. The dot_product::<2> path batches two
products in u128 with a single reduce128, partially offsetting the extra mul.

---

## Remaining Opportunities (Beyond Scalar Scope)

1. **7th root addition chain**: Current 71 ops (63 sq + 8 mul) is already
   24 ops better than naive. Exhaustive search tool needed to find shorter.
2. **Packed field (NEON/AVX2)**: The main throughput path uses SIMD.
   Scalar optimizations affect the tail/remainder processing only.
3. **Extension field**: Degree-2 and degree-5 operations are base-field-bound.
   The generic BinomialExtensionField implementation is already well-optimized.
4. **`neg` without canonicalization**: Impossible with current non-canonical
   representation. Would require switching to always-canonical representation
   (tradeoff: slower add/sub).
