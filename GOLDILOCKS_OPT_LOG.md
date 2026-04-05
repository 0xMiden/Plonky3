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
