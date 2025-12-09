# Lifted FRI — Protocol Notes and Implementation Plan

This document captures how Plonky3’s existing FRI works today, and how we will simplify it for the lifted LMCS-only variant under `lifted/src/fri`. It focuses on regular FRI (batched LDT on a single two‑adic domain) and ignores Circle PCS. We also specialize openings to two points per matrix: `(z^r, (g·z)^r)` where `r = H / h` is the scaling factor from the tallest height `H` to the matrix’s height `h`.

## Scope and Simplifications

- Use LMCS (lifted Merkle tree) exclusively for openings and commitments where applicable; ignore Circle PCS.
- Do not implement the `Pcs` trait initially; expose a minimal, direct API (commit, open, prove, verify) tailored to LMCS.
- All matrices are opened at exactly two points derived from a single global query point `z` on the tallest domain: `z^r` and `(g·z)^r` per matrix, where `g` is the domain coset shift on the tallest domain (see Cosets per height below).
- Stick to two‑adic subgroups and bit‑reversed storage, mirroring Plonky3.
- No hiding for now (omit randomized blinding codewords and hiding LMCS wrapper).

Cosets per height (different from current approach):
- Let matrices be sorted by nondecreasing height, tallest height `H` (power of two). Let `H` also denote a 2‑adic subgroup of order `H`.
- Define tallest LDE coset `D* = g · H`.
- For a matrix of height `h ≤ H`, with scaling factor `r = H / h`, define its LDE coset as `(g · H)^r = g^r · H^r`, which has size `h`. Projection from `D*` is `π_r(x) = x^r ∈ (g · H)^r`.
- Two points per matrix derived from the global point `z ∈ D*`: `z^r` and `(g·z)^r = g^r · z^r`. This matches the “current” and “next‑row” points for that matrix’s domain.


## Quick Recap: FRI Protocol

- Goal: Prove that a function (here, a DEEP quotient or combinations thereof) is close to a low‑degree polynomial over a 2‑adic domain.
- Commit phase (folding):
  - Given evaluations of `f_i` on domain `H` (2^k points), commit to pairs `(f_i(x), f_i(-x))` as a 2‑wide matrix, sample `β_i`, and fold to size |H|/2 via
    - `f_{i+1}(x^2) = (f_i(x) + f_i(-x))/2 + β_i · (f_i(x) - f_i(-x)) / (2x)`.
  - Repeat until the domain shrinks to the final size; send coefficients of the final polynomial.
- Query phase:
  - Sample random indices; for each query, open sibling entries from each commit phase, verify paths, and apply the same fold chain to reach the final layer.
  - Separately, compute reduced openings from input polynomials (DEEP quotients etc.) as `(f(z) − f(x)) / (z − x)` and “roll” them into the fold chain when the domain matches.
  - Finally, evaluate the sent final polynomial at `x` for the queried index and check equality.

Soundness (conjectured) scales like `log_blowup * num_queries + pow_bits` bits.


## How Plonky3 Implements FRI Today

High‑level structure lives in these files:
- `fri/src/prover.rs:43` — `prove_fri`: orchestrates commit and query phases.
- `fri/src/prover.rs:156` — `commit_phase`: commits 2‑wide codewords and folds using sampled `β`.
- `fri/src/prover.rs:244` — `answer_query`: returns sibling value + opening proof for each fold step at a query index.
- `fri/src/verifier.rs:43` — `verify_fri`: end‑to‑end verification entry.
- `fri/src/verifier.rs:220` — `verify_query`: replays the folding chain and rolls in reduced openings at matching heights.
- `fri/src/verifier.rs:327` — `open_input`: checks batch openings and builds reduced openings `(f(z) − f(x)) / (z − x)` grouped by log‑height.
- `fri/src/two_adic_pcs.rs:440` — reduction pipeline that batches many `(f, z)` pairs per height and produces the FRI inputs.

Key details reflected in code:
- Inputs sorted by descending length, stored bit‑reversed. Fold operates on adjacent pairs, hence 2‑wide matrices (`fri/src/prover.rs:174–183`).
- Final polynomial obtained by truncation, bit‑reversal, then IDFT (`fri/src/prover.rs:205–213`).
- Query indices: sample LS bits (optionally with extra bits), then walk the fold chain. At each step, verify sibling via MMCS `verify_batch`, fold with the same `β`, and roll in any pending reduced openings with weight `β²` (`fri/src/verifier.rs:268–294`).
- Reduced openings computed per height by batching across both polynomials and opening points with a single challenge `α` (`fri/src/verifier.rs:346–438` and `fri/src/two_adic_pcs.rs:440–506`).
- Two points per DEEP in practice: the current implementation commonly opens at `z` and `g·z` (next‑row shift), so “k is two times the trace width plus the number of quotient polynomials” (`fri/src/two_adic_pcs.rs:430–439`). In the lifted design, we keep this spirit but use per‑height cosets `(g·H)^r` and the projected points `(z^r, (g·z)^r)`.


## LMCS — What It Buys Us

LMCS commits multiple matrices of differing heights as if all were virtually lifted to the tallest height `H` (upsample or cyclic), then absorbs each lifted row into a single sponge state to form a Merkle tree over `H` leaves:
- Implementation: `lifted/src/merkle_tree` with `MerkleTreeLmcs` (`lifted/src/merkle_tree/mod.rs`).
- The lifting map induces a natural projection for points: if a matrix has height `h`, with scaling `r = H/h`, an evaluation at `x ∈ D*` (tallest domain) maps to the original domain by `x ↦ x^r` (see `lifted/docs.md`).
- Verification simplifies because every opening is a leaf on the same height `H` tree, and row padding is uniform.


## Lifted FRI Design (Simplified)

We adopt regular, batched FRI over a single tallest domain `D*` and use LMCS exclusively for openings. We keep the same two‑adic fold formula and commit‑phase proof shape, but we simplify inputs and openings:

- Single global query point per round: let `z ∈ D* = g·H` be the global point derived from the index. For a matrix of height `h`, define `r = H/h` and open only at two points:
  - Current row point: `z^r`.
  - Next‑row point: `(ω·z)^r = ω^r · z^r`, where `ω` is the generator of the largest subgroup `H`. This aligns the second point with the true next‑row relation at height `h`.
- For each matrix, compress columns with `α` into `M_red`, and form reduced openings
  - `(M_red(z^r) − M_red(x^r)) / (z^r − x^r)` and `(M_red((g·z)^r) − M_red((g·x)^r)) / ((g·z)^r − (g·x)^r)`
  - Combine those two using `β²` (reuse as in current code) and aggregate across matrices by advancing powers of `α` by width per height.
- Roll‑in schedule: the roll‑in of reduced openings occurs exactly when the fold chain’s current height equals the matrix’s lifted height (same as Plonky3’s `reduced_openings` by log‑height).
- Commit‑phase: commit 2‑wide codewords and fold with `β` exactly as today. We can use LMCS for these 2‑wide commitments as well (two columns, height halved each round), or keep a minimal MMCS just for FRI codewords.
- Final check: evaluate the sent final polynomial at `x` computed from the (post‑shift) domain index on `D*` and compare with the folded value.

Notes:
- Because all matrices are conceptually lifted to `H`, one global index/point suffices; per‑matrix indices are derived by right shifts (bit‑reversal aligns automatically) and the projection `x ↦ x^r`.
- Choosing two points `(z^r, (g·z)^r)` per matrix mirrors existing DEEP constraints for current/next row and keeps the DEEP polynomial computation unchanged while simplifying openings.

Intra‑ and inter‑matrix random linear combination with padding:
- Let `pad = StatefulHasher::PADDING_WIDTH` (the hasher's absorption rate; e.g., `RATE` for `PaddingFreeSponge`). For matrix `M_j` with width `w_j`, define `w'_j = ceil(w_j / pad) · pad`.
- Within a matrix, define the cached combination
  `p_j(X) = Σ_{i=0}^{w_j-1} β^{2i} · p_{j,i}(X)`, treating the padded lanes `i ∈ [w_j, w'_j)` as zeros.
- Define the inter‑matrix offset in padded lanes `U_j = Σ_{ℓ<j} w'_ℓ`, and set `β_j = β^{2·U_j}`.
- The global combination across all (padded) columns in the lifted view is
  `p(X) = Σ_j β_j · p_j(X) = Σ_j Σ_i β^{2·(U_j + i)} · p_{j,i}(X)`.
- Two‑point DEEP combination uses the same `β` to keep the two fractions disjoint in exponent space:
  `(p(X) − v_z)/(X − z) + β · (p(X) − v_{ωz})/(X − ωz)`.

Deep polynomial optimization via caching:
- Precompute and cache the random linear combination `R(x) = Σ_i β^{2·i} · f_i(x)` over columns `f_i` (or over per‑matrix groups), once per height. This lets both prover and verifier evaluate DEEP terms using two evaluations of `R` instead of iterating all `f_i` each time:
  - For a given matrix/height: compute `R(z^r)` and `R((ω·z)^r)` once from the sent claims; at a query index with row value `R(x^r)`, the verifier computes
    - `(R(z^r) − R(x^r)) / (z^r − x^r)` and `(R((ω·z)^r) − R((ω·x)^r)) / ((ω·z)^r − (ω·x)^r)`
  - The row values `R(x^r)` and `R((ω·x)^r)` are obtained by dotting the opened row(s) with the fixed weights `β^{2·i}`; this avoids re‑summing per column for each point and each query.
- This reduces per‑query work and transcript size, while preserving identical soundness (we are merely linear‑combining before quotienting).


## Algorithm Sketch (Prover)

- Inputs: for each height bucket (matrices of height `h`), a row‑major matrix of evaluations on `D_h` with common blowup; tallest height is `H`.
- LMCS commit: commit all matrices (sorted by height) via `MerkleTreeLmcs` (upsample or cyclic) to get a root and a prover handle. Use per‑height cosets `(g·H)^r` to define evaluation domains for each matrix’s LDE.
- Send claimed values at the two points per matrix: for each matrix, send `R(z^r)` and `R((ω·z)^r)` where `R = Σ β^{2·i} f_i`, in canonical height order.
- Build reduced openings per height using the verified rows at `x` (from LMCS `open_batch`), the cached `R(·)` values, batching with `α` and combining the two points with `β²`.
- Commit phase: run the standard FRI fold over the vector(s) of reduced openings (possibly multiple height vectors, largest to smallest), committing the intermediate 2‑wide codewords and sending the final polynomial coefficients.
- Query phase: for each random index, open LMCS at the appropriate height‑adjusted row index and provide sibling openings for all commit‑phase trees; return `pow` witness if enabled.


## Algorithm Sketch (Verifier)

- Recompute `α` (and `β`) from the challenger transcript.
- For each query index:
  - Derive `x ∈ D*` (coset shift times the two‑adic generator at the reversed index) and per‑height reduced indices by right shift.
  - Verify LMCS batch openings at the reduced index; compute per‑height reduced openings by projecting points `x ↦ x^r`, `z ↦ z^r` and using the two claimed values `R(z^r), R((ω·z)^r)` plus the dot‑product row values `R(x^r), R((ω·x)^r)`.
  - Replay the fold chain with the commit‑phase sibling openings, rolling in reduced openings at matching heights using the `β²` weighting rule (current Plonky3 behavior).
  - Evaluate final polynomial at `x` and check equality with the folded value.


## Mapping to Today’s Code

What we’ll keep (same mechanics, new wrappers):
- Fold formula and per‑round commit/open steps (`fri/src/prover.rs:156` and `fri/src/verifier.rs:220`).
- “Roll‑in at height” pattern and `β²` multiplier (`fri/src/verifier.rs:289–294`).
- Bit‑reversed storage, index shifts, and two‑adic `x` reconstruction (`fri/src/verifier.rs:167–171`).

What we’ll simplify/replace:
- Replace `two_adic_pcs`’s generic batching of arbitrary point sets with a fixed two‑point schedule derived from `z` via projection (`lifted` encodes the `x ↦ x^r` rule; see `lifted/docs.md`).
- Use LMCS for input openings and (optionally) for FRI codeword commitments; unify all openings against a single height `H` tree (`lifted/src/merkle_tree/mod.rs`). Different LDE cosets `(g·H)^r` are handled by projection `x ↦ x^r`.
- Drop the `Pcs` trait initially; provide a direct API that accepts LMCS commitments and returns a FRI proof with commit‑phase openings specialized to LMCS.


## Open Points / Clarifications

- Cosets: we set `D* = g·H` for the tallest height, and use projected cosets `(g·H)^r` for smaller heights. The “next‑row” point uses the subgroup generator `ω`: second point is `(ω·z)^r`.
- Combine the two DEEP fractions with `β` (per your formula); keep `β²` only for fold roll‑ins as in the current verifier.
- Commit‑phase trees: prefer reusing LMCS for simplicity (or keep a tiny MMCS if this proves cleaner), without hiding.
- Defaults: set `log_final_poly_len = 0`, keep proof‑of‑work enabled unless otherwise requested.
- First iteration: single batched quotient input plus the two points per matrix; extend later if needed.


## Next Steps

- Scaffold `lifted/src/fri` with:
  - Two‑adic folding strategy (same formula, specialized types).
  - Minimal parameters struct mirroring `FriParameters` without Circle‑PCS/trait dependencies.
  - LMCS‑backed openings that project points via `r = H/h` and always use two points.
  - Proof structs parallel to `FriProof` but specialized to LMCS commitments.
- Port/adapt commit/query logic from `fri/src/prover.rs` and `fri/src/verifier.rs`, trimming generic PCS hooks.
- Add end‑to‑end tests that commit a small batch via LMCS, run the lifted FRI, and verify.

References: see `lifted/docs.md` for the lifting map, domain alignment, and two‑point DEEP strategy; and the current FRI implementation in `fri/src/*` for concrete mechanics and transcript ordering.
