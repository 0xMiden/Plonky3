use alloc::vec::Vec;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_util::log2_strict_usize;
use p3_util::{reverse_bits_len, reverse_slice_index_bits};
use p3_dft::{Radix2DFTSmallBatch, TwoAdicSubgroupDft};

use p3_commit::Mmcs;
use super::config::LiftedFriParams;
use super::proof::{CommitPhaseProofStep, LiftedFriProof, QueryProof};

/// Prover entry-point for lifted FRI using LMCS and two-point DEEP per height.
///
/// Inputs:
/// - `folding`: two-adic folding strategy (standard FRI fold).
/// - `params`: lifted FRI parameters (includes commit-phase MMCS and padding width).
/// - `lmcs`: LMCS used for input openings (commitment must have been observed earlier in transcript).
/// - `lmcs_input`: LMCS prover data and matrix dimensions; matrices sorted by height.
/// - `challenger`: Fiat–Shamir challenger (observes commitments and samples α, β’s, PoW witness).
///
/// Output:
/// - A lifted FRI proof containing commit-phase codeword commitments, per-query LMCS batch openings
///   and codeword sibling openings, and the final polynomial coefficients.
pub fn prove_lifted_fri<Val, Challenge, InputMmcs, CodeMmcs, Challenger>(
    params: &LiftedFriParams<CodeMmcs>,
    input_mmcs: &InputMmcs,
    input_prover_data: &InputMmcs::ProverData<RowMajorMatrix<Val>>,
    challenger: &mut Challenger,
) -> LiftedFriProof<Val, Challenge, CodeMmcs, InputMmcs, Challenger::Witness>
where
    Val: TwoAdicField,
    Challenge: ExtensionField<Val>,
    InputMmcs: Mmcs<Val>,
    CodeMmcs: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger + CanObserve<CodeMmcs::Commitment>,
{
    // TODO: High-level flow with detailed steps:
    // 1) Sample α (batching across heights/matrices) and prepare per-height cached combinations.
    //    - Compute padding-aware offsets U_j per matrix using params.padding_width.
    //    - Define β_j = β^{2 * U_j} (β sampled later during query; constants can be defined symbolically here).
    //    - Define p_j(X) = Σ_i β^{2i} p_{j,i}(X) for each matrix j (implicitly via column weights).
    //    - Define global p(X) = Σ_j β_j p_j(X).
    // 2) Build initial reduced_openings vectors per height (largest to smallest) over domain size H:
    //    - For each height h, reduced_openings_h[x] will hold RO_h(x) after we have row openings.
    //    - At this stage, we can leave them empty and fill per query, or precompute structure only.
    //    - Skeleton: we will defer concrete values to query-time; we still need correct shapes for commit phase.
    // 3) Commit phase over vectors (fold 2-wide with β’s) until final_poly_len; observe commitments and coefficients.
    //    - Use params.mmcs.commit_matrix for 2-wide codewords and observe commitments via challenger.
    //    - Sample β per round from challenger.
    //    - Truncate + bit-unreverse + IDFT to get final_poly; observe coefficients.
    // 4) Grind PoW before receiving queries.
    // 5) For each of params.num_queries queries:
    //    - Sample index (with folding.extra_query_index_bits LSB if needed; shifted off later).
    //    - Open LMCS at reduced index per height: input_mmcs.open_batch(index, input_prover_data)
    //      (LMCS handles lifted mapping and padding shape).
    //    - Compute claims per height: (R(z^r), R((ω·z)^r)) using barycentric evaluation on Dh points
    //      per lifted/docs.md and existing PCS code; cache them in the proof.
    //    - Produce commit-phase sibling openings for the 2-wide trees at each fold step.
    //    - Package QueryProof with LMCS opening, claims, and sibling openings.
    // 6) Return LiftedFriProof with all pieces.

    // NOTE: Skeleton – we return todo! to be filled; leave the type-level plumbing ready.
    todo!("Implement lifted FRI prover: α sampling, padding-aware β_j, LMCS open, two-point claims, commit phase, β roll-ins, PoW, and queries.")
}

/// Perform the commit phase of FRI over the (conceptual) vectors of reduced openings per height.
///
/// Skeleton: lays out what needs to be computed and observed.
pub fn commit_phase<F, EF, CodeMmcs, Challenger>(
    params: &LiftedFriParams<CodeMmcs>,
    inputs_descending: Vec<Vec<EF>>,
    challenger: &mut Challenger,
) -> (
    Vec<CodeMmcs::Commitment>,                // commit_phase_commits
    Vec<CodeMmcs::ProverData<RowMajorMatrix<EF>>>, // commit_phase prover data to answer queries
    Vec<EF>,                           // final_poly coefficients
)
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    CodeMmcs: Mmcs<EF>,
    Challenger: FieldChallenger<F> + CanObserve<CodeMmcs::Commitment>,
{
    assert!(!inputs_descending.is_empty());
    for w in inputs_descending.windows(2) {
        debug_assert!(w[0].len() >= w[1].len(), "inputs not sorted by descending length");
    }

    let mut inputs_iter = inputs_descending.into_iter().peekable();
    let mut folded = inputs_iter.next().unwrap();
    let mut commits = Vec::new();
    let mut data = Vec::new();

    while folded.len() > params.blowup() * params.final_poly_len() {
        // Adjacent pairs are the two-point rows in bit-reversed layout.
        let leaves = RowMajorMatrix::new(folded, 2);

        // Commit and observe.
        let (commit, prover_data) = params.mmcs.commit_matrix(leaves);
        challenger.observe(commit.clone());
        commits.push(commit);

        // Sample beta for this round.
        let beta: EF = challenger.sample_algebra_element();

        // Fold to the next height using our two-adic fold.
        let leaves_view = params.mmcs.get_matrices(&prover_data).pop().unwrap();
        folded = fold_matrix_two_adic::<F, EF, _>(beta, leaves_view.as_view());
        data.push(prover_data);

        // If another input matches the current length, roll it in with beta^2.
        if let Some(v) = inputs_iter.next_if(|v| v.len() == folded.len()) {
            for (c, x) in folded.iter_mut().zip(v.into_iter()) {
                *c += beta.square() * x;
            }
        }
    }

    // Extract final polynomial coefficients.
    folded.truncate(params.final_poly_len());
    reverse_slice_index_bits(&mut folded);
    let final_poly = Radix2DFTSmallBatch::default().idft_algebra(folded);

    // Observe coefficients.
    for &coeff in &final_poly {
        challenger.observe_algebra_element(coeff);
    }

    (commits, data, final_poly)
}

/// Given an index, produce commit-phase sibling openings across all fold rounds.
pub fn answer_query<EF, CodeMmcs>(
    params: &LiftedFriParams<CodeMmcs>,
    folded_commits: &[CodeMmcs::ProverData<RowMajorMatrix<EF>>],
    start_index: usize,
) -> Vec<CommitPhaseProofStep<EF, CodeMmcs>>
where
    EF: Field,
    CodeMmcs: Mmcs<EF>,
{
    folded_commits
        .iter()
        .enumerate()
        .map(|(i, commit)| {
            // After i folds, current index and parent.
            let index_i = start_index >> i;
            let index_i_sibling = index_i ^ 1;
            let parent = index_i >> 1;

            // Open the 2-wide row at the parent.
            let (mut opened_rows, opening_proof) = params.mmcs.open_batch(parent, commit).unpack();
            assert_eq!(opened_rows.len(), 1);
            let opened_row = opened_rows.pop().unwrap();
            assert_eq!(opened_row.len(), 2);
            let sibling_value = opened_row[index_i_sibling % 2];

            CommitPhaseProofStep { sibling_value, opening_proof }
        })
        .collect()
}

/// Two-adic FRI fold of a single pair (row-level), using the standard formula:
/// f_{i+1}(x^2) = (a + b)/2 + beta * (a - b) / (2x)
/// where (a, b) = (f_i(x), f_i(-x)).
///
/// NOTE: Computing x depends on the enumeration of points for the given row. Here we leave a
/// placeholder using 1/x = 1 (i.e., omitting the x factor) with a TODO to wire in correct x.
fn fold_row_two_adic<F, EF>(index: usize, log_height: usize, beta: EF, pair: [EF; 2]) -> EF
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
{
    // Interpolate at `beta` between the pair's x-values (as in fri::two_adic_pcs::fold_row).
    let a = pair[0];
    let b = pair[1];
    let arity = 2;
    let log_arity = 1usize;
    let subgroup_start = F::two_adic_generator(log_height + log_arity)
        .exp_u64(reverse_bits_len(index, log_height) as u64);
    let mut xs = F::two_adic_generator(log_arity)
        .shifted_powers(subgroup_start)
        .collect_n(arity);
    reverse_slice_index_bits(&mut xs);
    a + (beta - xs[0]) * (b - a) * (xs[1] - xs[0]).inverse()
}

/// Apply `fold_row_two_adic` across all rows in a 2-wide matrix.
fn fold_matrix_two_adic<F, EF, M>(beta: EF, m: M) -> Vec<EF>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    M: Matrix<EF>,
{
    assert_eq!(m.width(), 2);
    let h = m.height();
    let log_h = log2_strict_usize(h);
    (0..h)
        .map(|row| {
            let idx = row;
            let pair = [m.get(row, 0).unwrap(), m.get(row, 1).unwrap()];
            fold_row_two_adic::<F, EF>(idx, log_h, beta, pair)
        })
        .collect()
}
