use alloc::vec::Vec;

use p3_commit::{BatchOpening, Mmcs};
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

/// A single step in the commit-phase query: sibling value and opening proof
/// at the current node for the 2-wide codeword tree.
#[derive(Clone, Debug)]
pub struct CommitPhaseProofStep<F: Field, M: Mmcs<F>> {
    pub sibling_value: F,
    pub opening_proof: M::Proof,
}

/// Proof object for lifted FRI.
///
/// Generics:
/// - `Challenge`: extension field element type.
/// - `CodeMmcs`: commitment scheme used for commit-phase codewords (2-wide trees).
/// - `Witness`: PoW/grinding witness type from the challenger.
/// - `Val`, `InputMmcs`: base field and LMCS used for input openings.
#[derive(Clone)]
pub struct LiftedFriProof<Val, Challenge, CodeMmcs, InputMmcs, Witness>
where
    Val: Field + Send + Sync + Clone,
    Challenge: Field + Send + Sync + Clone,
    CodeMmcs: Mmcs<Challenge>,
    InputMmcs: Mmcs<Val>,
{
    pub commit_phase_commits: Vec<CodeMmcs::Commitment>,
    pub final_poly: Vec<Challenge>,
    pub query_proofs: Vec<QueryProof<Val, Challenge, CodeMmcs, InputMmcs>>,
    pub pow_witness: Witness,
}

/// Per-query proof components for lifted FRI.
#[derive(Clone)]
pub struct QueryProof<Val, Challenge, CodeMmcs, InputMmcs>
where
    Val: Field + Send + Sync + Clone,
    Challenge: Field + Send + Sync + Clone,
    CodeMmcs: Mmcs<Challenge>,
    InputMmcs: Mmcs<Val>,
{
    /// LMCS opening of all matrices at the appropriate reduced index.
    /// In LMCS, this is a single batch opening against the lifted tree.
    pub input_opening: BatchOpening<Val, InputMmcs>,

    /// Claimed cached combination values per height (descending by height):
    /// (height, R(z^r), R((ω·z)^r)).
    pub input_claims_per_height: Vec<(usize, Challenge, Challenge)>,

    /// Sibling values and opening proofs for the 2-wide codewords at each fold round.
    pub commit_phase_openings: Vec<CommitPhaseProofStep<Challenge, CodeMmcs>>,
}
