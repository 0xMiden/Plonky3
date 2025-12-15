use alloc::vec::Vec;
use core::fmt;
use core::marker::PhantomData;

use p3_commit::{BatchOpening, Mmcs};
use p3_field::{ExtensionField, Field};

use crate::deep::{DeepQuery, MatrixGroupEvals};
use crate::fri::CommitPhaseProof;

/// Complete PCS opening proof.
///
/// Contains all information needed by the verifier to check polynomial
/// evaluation claims against a commitment.
pub struct Proof<F: Field, EF: ExtensionField<F>, InputMmcs: Mmcs<F>, FriMmcs: Mmcs<EF>> {
    /// Claimed evaluations at each opening point.
    /// Structure: `evals[point_idx][commit_idx]` is a `MatrixGroupEvals` containing
    /// `evals[point_idx][commit_idx][matrix_idx][col_idx]`
    pub evals: Vec<Vec<MatrixGroupEvals<EF>>>,

    /// FRI commit phase proof (intermediate commitments + final polynomial)
    pub fri_commit_proof: CommitPhaseProof<EF, FriMmcs>,

    /// Query phase proofs, one per query index
    pub query_proofs: Vec<QueryProof<F, EF, InputMmcs, FriMmcs>>,
}

/// Proof for a single FRI query index.
///
/// Contains Merkle openings for both the input matrices (via DEEP)
/// and each FRI folding round, allowing the verifier to check consistency.
pub struct QueryProof<F: Field, EF: ExtensionField<F>, InputMmcs: Mmcs<F>, FriMmcs: Mmcs<EF>> {
    /// Openings of the input matrices at this query index
    /// (one BatchOpening per committed matrix group)
    pub input_openings: DeepQuery<F, InputMmcs>,

    /// Openings for each FRI folding round
    pub fri_round_openings: Vec<BatchOpening<EF, FriMmcs>>,

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
#[derive(Debug)]
pub enum PcsError<InputMmcsError: fmt::Debug, FriMmcsError: fmt::Debug> {
    /// Input MMCS verification failed
    InputMmcsError(InputMmcsError),
    /// FRI MMCS verification failed
    FriMmcsError(FriMmcsError),
    /// DEEP quotient evaluation mismatch
    DeepQuotientMismatch { query_index: usize },
    /// FRI folding verification failed
    FriFoldingError { query_index: usize, round: usize },
    /// Final polynomial evaluation mismatch
    FinalPolyMismatch { query_index: usize },
    /// Wrong number of queries in proof
    WrongNumQueries { expected: usize, actual: usize },
}

impl<E1: fmt::Debug, E2: fmt::Debug> fmt::Display for PcsError<E1, E2> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputMmcsError(e) => write!(f, "Input MMCS error: {:?}", e),
            Self::FriMmcsError(e) => write!(f, "FRI MMCS error: {:?}", e),
            Self::DeepQuotientMismatch { query_index } => {
                write!(f, "DEEP quotient mismatch at query {}", query_index)
            }
            Self::FriFoldingError { query_index, round } => {
                write!(
                    f,
                    "FRI folding error at query {}, round {}",
                    query_index, round
                )
            }
            Self::FinalPolyMismatch { query_index } => {
                write!(f, "Final polynomial mismatch at query {}", query_index)
            }
            Self::WrongNumQueries { expected, actual } => {
                write!(
                    f,
                    "Wrong number of queries: expected {}, got {}",
                    expected, actual
                )
            }
        }
    }
}
