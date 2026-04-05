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
