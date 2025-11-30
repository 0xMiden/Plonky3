# Lifted FRI — Implementation Plan (Pseudocode + Math)

This plan specifies a simplified, lifted FRI over LMCS with two-point openings per matrix and height-specific cosets. It reuses the standard two-adic FRI folding and commits, but removes Circle PCS and hiding.

We assume matrices are sorted by height (nondecreasing), and we work over two-adic groups with bit-reversed storage.

## Domains, Cosets, and Projection

- Let the tallest height be `H = 2^k` with two-adic subgroup `\langle \omega_H \rangle`.
- Define tallest (lifted) LDE coset
  $$ D^* = g\cdot H = \{ g \cdot \omega_H^i \mid i=0,\dots,H-1 \}. $$
- For a matrix of height `h = 2^m \le H`, define the scaling factor `r = H/h = 2^{k-m}` and its LDE coset as
  $$ D_h = (g\cdot H)^r = g^r \cdot H^r, \quad |D_h| = h. $$
- Projection map from `D^*` to `D_h` is
  $$ \pi_r(x) = x^r. $$
- Two canonical points per matrix (for DEEP-like openings), derived from a single global `z \in D^*`:
  $$ z_h := z^r, \qquad z'_h := (g\cdot z)^r = g^r\,z^r. $$

Interpretation under lifting: conceptually, we commit to one height-`H` matrix that vertically lifts all input matrices and concatenates them (with zero-padding to hasher width for absorption). FRI then operates over this single, uniform domain.

## Cached Random Linear Combination

To optimize the DEEP terms, fix a challenge `\beta` (sampled by the transcript) and define the cached combination
$$
R(X) = \sum_{i \ge 0} \beta^{2i} \cdot f_i(X),
$$
where `f_i` are a fixed column ordering within a height (or within a matrix), consistent across prover and verifier. At runtime we maintain this per height bucket (or per matrix), so that for the two points we only evaluate `R` twice:
$$
R(z_h), \quad R(z'_h).
$$
Given an opened row `x_h := x^r` at `D_h`, the verifier computes row values
$$
R(x_h) = \sum_i \beta^{2i} f_i(x_h), \qquad R((g\cdot x)_h) = \sum_i \beta^{2i} f_i((g\cdot x)^r),
$$
via a single dot product of the opened row(s) with the weights `\beta^{2i}`.

We then form per-height reduced openings using two DEEP terms combined with `\beta^2` (reusing the folding challenge square per current Plonky3 practice):
$$
\mathrm{RO}_h(x) = \frac{R(z_h) - R(x_h)}{z_h - x_h} \;\; + \;\; \beta\, \frac{R(z'_h) - R((\omega\cdot x)_h)}{z'_h - (\omega\cdot x)_h}.
$$
Finally we batch across matrices of the same height using `\alpha` by advancing powers proportional to widths (as in existing code).

Padding‑aware offsets across matrices:
- Let `pad = InputMmcs::ROW_PADDING` (defaults to 1; LMCS uses its hasher padding). For matrix `M_j` with width `w_j`, padded width is `w'_j = \lceil w_j / pad \rceil · pad`.
- Define intra‑matrix cache `p_j(X) = \sum_{i=0}^{w_j-1} \beta^{2i} p_{j,i}(X)`.
- Define offset in padded lanes `U_j = \sum_{\ell<j} w'_{\ell}` and `\beta_j = \beta^{2\,U_j}`.
- Global combination `p(X) = \sum_j \beta_j p_j(X)` ensures verifier can recover `p(x)` by a dot product over the LMCS‑opened, padded row using the same exponent schedule.

## Transcript and Challenges

- `\alpha \leftarrow \mathcal{C}`: batches functions and points per height.
- `z \leftarrow \mathcal{C}`: a fresh extension-field challenge for DEEP. Sampled after LMCS
  commitments are observed and after `\alpha`, before building reduced FRI inputs. With
  overwhelming probability `z` is out-of-domain (not equal to any evaluation point).
- Commit phase repeats `t` times (from `H` down to final height), each time:
  - Observe commitment to 2-wide codeword, then sample `\beta_i \leftarrow \mathcal{C}`.
- Proof-of-work witness as configured (can keep enabled).
- Query indices: sample `q` indices from transcript in `[0, H)`.

## Data Structures (sketch)

- `FriParamsLifted { log_blowup, log_final_poly_len (≈ 0), num_queries, pow_bits }`.
- `LmcsCommitment` and `LmcsProverData` from `lifted/src/merkle_tree`.
- `ProofLiftedFri`:
  - `commit_phase_commits: Vec<CodewordCommitment>`
  - `final_poly: Vec<EF>`
  - `query_proofs: Vec<QueryProofLifted>`
  - `pow_witness`
- `QueryProofLifted`:
  - `input_openings: LmcsBatchOpening` (LMCS rows at the reduced index for all matrices)
  - `input_claims: Vec<(height, R(z_h), R(z'_h))>` (claimed two-point values per height)
  - `commit_phase_openings: Vec<CodewordSiblingOpening>` (sibling value + Merkle path per fold)

We may implement a small internal Merkle for 2-wide codewords (height varies each fold). LMCS can also be reused with width=2 at each fold if convenient.

## Prover — Pseudocode

Notation: matrices sorted by ascending height, tallest height `H`. Let `EF` be the extension field.

```
ProveLiftedFRI(matrices: [Matrix<F>], params): ProofLiftedFri
  assert sorted_by_height(matrices)
  H := height(matrices.last)
  // Commit inputs via LMCS (conceptually a single lifted matrix)
  (lmcs_root, lmcs_data) := LMCS.commit(matrices)

  // Generate transcript challenges
  α := Challenger.sample_algebra_element()
  z := Challenger.sample_algebra_element()    // DEEP point
  ω := Challenge::from(Val::two_adic_generator(log2(H)))

  // Precompute β-weights per height/matrix for R(X) caching
  // Define a stable column enumeration f_i within each height bucket.

  // Precompute per-height DEEP points derived from z:
  // For each height h (descending), r := H / h; zh := z^r; wzh := (ω * z)^r.
  // We can compute R(zh), R(wzh) once per height and reuse across queries.
  // Build reduced openings vectors per height conceptually, but fill their values per query
  // after obtaining p(x_h) and p((ωx)_h) from opened rows.
  reduced_openings := []  // shapes recorded; values computed per query

  // Commit phase over reduced openings
  (cw_commits, cw_data, final_poly) := CommitPhase(reduced_openings, params)

  // Produce PoW witness before queries
  pow := Challenger.grind(params.pow_bits)

  // Query phase
  queries := []
  for t in 1..params.num_queries:
    idx := Challenger.sample_bits(log2(H))

    // Open LMCS at reduced index for each height:
    // reduced_index(h) = idx >> (log2(H) - log2(h))
    lmcs_open := LMCS.open_batch(idx, lmcs_data)  // internally shifts per height

    // For each height h, we already have zh = z^r, wzh = (ω*z)^r.
    // Prover includes the (constant across queries) claims R(zh), R(wzh) in the query,
    // or amortizes them at the proof level if desired.
    claims := []
    for each height h descending:
      // Prover sends R(zh), R(wzh) based on cached R
      claims.push((h, R(zh), R(wzh)))

    // Sibling openings for commit-phase codewords at the index chain idx, idx>>1, ...
    siblings := AnswerQuery(cw_data, idx)

    queries.push({ input_openings: lmcs_open, input_claims: claims, commit_phase_openings: siblings })

  return { commit_phase_commits: cw_commits, final_poly, query_proofs: queries, pow_witness: pow }
```

Commit phase (folding) matches standard FRI:

```
CommitPhase(inputs: Vec<Vec<EF>>, params): (commits, data, final_poly)
  folded := inputs[0]  // largest height vector
  commits := []; data := []
  while |folded| > params.blowup * params.final_poly_len:
    // reshape to height × 2 (bit-reversed order makes siblings adjacent)
    cw := reshape_as_2wide(folded)
    (cmt, cdata) := CodewordMMCS.commit(cw)
    Challenger.observe(cmt)
    β := Challenger.sample_algebra_element()
    folded := FoldMatrix(β, cw)  // element-wise folding
    data.push(cdata); commits.push(cmt)
    // If next inputs[k] matches length, add with β^2 multiplier
    if exists v in inputs with |v| == |folded|: folded := folded + β^2 * v
  // Truncate, bit-unreverse, IDFT to coefficients
  final_poly := IDFT(bit_unreverse(truncate(folded, params.final_poly_len)))
  Challenger.observe_all(final_poly)
  return (commits, data, final_poly)
```

## Verifier — Pseudocode

```
VerifyLiftedFRI(matrices_dims, lmcs_root, proof, params): bool
  H := matrices_dims.last.height

  // Batch challenges; replay transcript in the same order
  α := Challenger.sample_algebra_element()
  z := Challenger.sample_algebra_element()    // DEEP point
  ω := Challenge::from(Val::two_adic_generator(log2(H)))
  betas := []
  for cmt in proof.commit_phase_commits:
    Challenger.observe(cmt)
    betas.push(Challenger.sample_algebra_element())

  // Check final_poly length and absorb coefficients
  assert |final_poly| == params.final_poly_len
  Challenger.observe_all(proof.final_poly)

  // Check PoW
  assert Challenger.check_witness(params.pow_bits, proof.pow_witness)

  for q in proof.query_proofs:
    idx := Challenger.sample_bits(log2(H))

    // Verify LMCS openings at reduced index per height
    LMCS.verify_batch(lmcs_root, matrices_dims, idx, q.input_openings)

    // Build reduced openings per height using cached R claims and row values
    ro_list := []  // pairs (log2(h), RO_h(idx)) in descending log-height

    // Compute subgroup point x at this index (no coset shift); x ∈ H
    k := reverse_bits_len(idx, log2(H))
    x := Challenge::from(Val::two_adic_generator(log2(H)).exp_u64(k as u64))

    for each height h descending:
      r := H / h
      // Project x and z to D_h
      xh   := x^r
      z_h  := z^r
      wzh  := (ω*z)^r
      wxh  := (ω*x)^r
      // Row values assembled by dot product with weights β^{2i}
      R_xh  := dot(weights=β^{2i}, row_at(q.input_openings, h, idx_reduced))
      R_wxh := dot(weights=β^{2i}, row_at(q.input_openings, h, idx_reduced_shifted))
      // Use claimed two-point values
      (R_zh, R_wzh) := q.input_claims[height=h]
      // Reduced opening
      RO := (R_zh - R_xh) / (z_h - xh) + β * (R_wzh - R_wxh) / (wzh - wxh)
      ro_list.push((log2(h), RO))

    // Replay the fold chain: start from the largest-height RO value
    folded := take_first(ro_list).value
    domain_idx := idx

    for i from 0 .. betas.len-1:
      (β, cmt) := (betas[i], proof.commit_phase_commits[i])
      opening := q.commit_phase_openings[i]
      // sibling index and parent index
      sib_idx := domain_idx ^ 1
      parent  := domain_idx >> 1
      // Verify 2-wide codeword opening at parent for the row [folded, sibling]
      CodewordMMCS.verify(cmt, parent, row=[folded at idx%2, opening.sibling_value], opening.path)
      // Fold to parent value
      folded := FoldRow(parent, log_height_from_round, β, [folded, opening.sibling_value])
      // Roll-in another RO if ro_list has an entry at the current folded height
      if exists (lh, RO) with lh == current_log_height: folded := folded + β^2 * RO
      domain_idx := parent

    // Final layer check: evaluate final_poly at x_final and compare
    x_final := ω_H^{rev(domain_idx,H)} * g  // same reconstruction as prover
    if HornerEval(proof.final_poly, x_final) != folded: return false

  return true
```

Notes:
- `idx_reduced` per height is `idx >> (log2(H) - log2(h))` due to projection; bit-reversal must be applied consistently for `x` reconstruction.
- The “shifted” row for next-row value can be taken by index xor 1 at that height (or by appropriate domain shift), consistent with commit-phase layout. Alternatively compute `(g*x)^r` explicitly and derive the matching row index.

## Correctness and Soundness

- Folding and commit-phase match standard two-adic FRI; soundness comes from conjectured bound `log_blowup * num_queries + pow_bits` bits.
- Caching `R` is a linear preprocessing; DEEP terms remain linear and do not change completeness or soundness.
- Coset projection ensures the two points per matrix come from a single global `z ∈ D^*`, respecting per-height domains `(g·H)^r`.

## Complexity

- Prover:
  - LMCS commit is linear in total area (sum of matrix areas after padding); no hiding overhead.
  - Caching `R`: one pass per height/matrix; amortizes per-query DEEP evaluation to two scalar values plus two dot products.
  - Commit-phase: identical to standard FRI over the concatenated reduced openings.
- Verifier:
  - Per query: one LMCS batch verification, a small number of codeword Merkle verifications (one per fold), and O(1) DEEP math per height (thanks to cached `R`).

## Testing Strategy

- Unit tests for projection `x ↦ x^r` and index reduction.
- End-to-end small instance with multiple heights and widths: commit via LMCS, run prover, verify proof.
- Negative tests: perturb claimed `R(z_h)`, break sibling proofs, and mismatched final polynomial.

## Implementation Notes

- Implement a tiny `CodewordMMCS` (binary Merkle) for 2-wide codewords, or reuse LMCS with width=2 per fold.
- Keep `log_final_poly_len = 0` for now; later support nonzero stopping points.
- No hiding in LMCS or FRI commit-phase initially.



Let's implement a deep.rs file which encapsulates the logic for computing the quotient
we have a prove/verify paradigm
there is a "config" struct containing the mmcs and maybe otherthings
on the prover side, we have a method which takes as input a sequence of ProverData (a list of multiple committed matrices), and a vector of opening points (in practice we'd give z and omega*z, where omega is the generator for the subgroup H of the tallest matrix).
It then uses barycentric evaluation to efficiently evaluate all columns at the points, taking into account the scaling factors of each matrix (ie eval at x^r where x is the opening and r is scaling factor)
it then "yeilds" allowing the caller to sample beta
in the second step, it would take the opening points and evals again, and compute the deepquotient as a vecotr of challenge

set beta' = beta^{num_openings}
for each matrix M=p1...pn, we compute the random linear q = ∑_i beta'^i * pi
assume in the first step we had v_i_j = pi(x_j) where x_j is the j-th opening point (there are num_openings of them)

now, we do this for all matrices indexed by k. that is, we have q_k which are all of different heights, grouped by sets of prover data
for each group, we do a random linear combination of the q_k as ∑_k beta^{trace_k_width_padded} * q_k. except that we take advantage of lifting. 
we iterate from smallest to tallest matrix, 
