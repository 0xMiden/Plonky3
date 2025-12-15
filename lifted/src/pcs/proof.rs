use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_commit::{BatchOpening, Mmcs};
use p3_field::{ExtensionField, Field};
use thiserror::Error;

use crate::deep::{DeepError, DeepQuery, MatrixGroupEvals};
use crate::fri::CommitPhaseProof;

/// Complete PCS opening proof.
///
/// Contains all information needed by the verifier to check polynomial
/// evaluation claims against a commitment.
pub struct Proof<F: Field, EF: ExtensionField<F>, InputMmcs: Mmcs<F>, FriMmcs: Mmcs<EF>> {
    /// Claimed evaluations at each opening point.
    /// Structure: `evals[point_idx][commit_idx]` is a `MatrixGroupEvals` containing
    /// `evals[point_idx][commit_idx][matrix_idx][col_idx]`
    pub(crate) evals: Vec<Vec<MatrixGroupEvals<EF>>>,

    /// FRI commit phase proof (intermediate commitments + final polynomial)
    pub(crate) fri_commit_proof: CommitPhaseProof<EF, FriMmcs>,

    /// Query phase proofs, one per query index
    pub(crate) query_proofs: Vec<QueryProof<F, EF, InputMmcs, FriMmcs>>,
}

/// Proof for a single FRI query index.
///
/// Contains Merkle openings for both the input matrices (via DEEP)
/// and each FRI folding round, allowing the verifier to check consistency.
pub struct QueryProof<F: Field, EF: ExtensionField<F>, InputMmcs: Mmcs<F>, FriMmcs: Mmcs<EF>> {
    /// Openings of the input matrices at this query index
    /// (one BatchOpening per committed matrix group)
    pub(crate) input_openings: DeepQuery<F, InputMmcs>,

    /// Openings for each FRI folding round
    pub(crate) fri_round_openings: Vec<BatchOpening<EF, FriMmcs>>,

    _marker: PhantomData<F>,
}

impl<F: Field, EF: ExtensionField<F>, InputMmcs: Mmcs<F>, FriMmcs: Mmcs<EF>>
    QueryProof<F, EF, InputMmcs, FriMmcs>
{
    /// Create a new query proof from input and FRI round openings.
    pub const fn new(
        input_openings: DeepQuery<F, InputMmcs>,
        fri_round_openings: Vec<BatchOpening<EF, FriMmcs>>,
    ) -> Self {
        Self {
            input_openings,
            fri_round_openings,
            _marker: PhantomData,
        }
    }
}

/// Errors that can occur during PCS verification.
///
/// Verification can fail due to invalid Merkle proofs, inconsistent folding,
/// or mismatched polynomial evaluations.
#[derive(Debug, Error)]
pub enum PcsError<InputMmcsError, FriMmcsError> {
    /// Input MMCS verification failed.
    #[error("input MMCS error: {0:?}")]
    InputMmcsError(InputMmcsError),
    /// FRI MMCS verification failed.
    #[error("FRI MMCS error: {0:?}")]
    FriMmcsError(FriMmcsError),
    /// FRI folding verification failed.
    #[error("FRI folding error at query {query_index}")]
    FriFoldingError { query_index: usize },
    /// Final polynomial evaluation mismatch.
    #[error("final polynomial mismatch at query {query_index}")]
    FinalPolyMismatch { query_index: usize },
    /// Wrong number of queries in proof.
    #[error("wrong number of queries: expected {expected}, got {actual}")]
    WrongNumQueries { expected: usize, actual: usize },
    /// DEEP oracle construction failed.
    #[error("DEEP error: {0}")]
    DeepError(#[from] DeepError),
}
