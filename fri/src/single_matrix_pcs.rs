//! A simplified Polynomial Commitment Scheme (PCS) for single matrices.
//!
//! This module provides a commitment scheme based on the FRI protocol but simplified
//! to handle only a single matrix at a time, removing the complexity of batch commitments.

use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Debug;
use core::marker::PhantomData;

use itertools::Itertools;
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{BatchOpening, Mmcs, OpenedValues, Pcs};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{
    ExtensionField, PackedFieldExtension, TwoAdicField, batch_multiplicative_inverse, dot_product,
};
use p3_interpolation::interpolate_coset_with_precomputation;
use p3_matrix::Matrix;
use p3_matrix::bitrev::{BitReversedMatrixView, BitReversibleMatrix};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_maybe_rayon::prelude::*;
use p3_util::linear_map::LinearMap;
use p3_util::{log2_strict_usize, reverse_slice_index_bits};

use crate::verifier::{self, FriError};
use crate::{FriParameters, FriProof, TwoAdicFriFolding, TwoAdicFriFoldingForMmcs, prover};

/// A simplified polynomial commitment scheme for single matrices using FRI.
///
/// Unlike the full `TwoAdicFriPcs` which handles batches of matrices across multiple rounds,
/// this implementation commits to exactly one matrix at a time. This simplifies the API
/// while maintaining the security and efficiency of the FRI protocol.
#[derive(Debug)]
pub struct SingleMatrixPcs<Val, Dft, InputMmcs, FriMmcs> {
    pub(crate) dft: Dft,
    pub(crate) mmcs: InputMmcs,
    pub(crate) fri: FriParameters<FriMmcs>,
    _phantom: PhantomData<Val>,
}

impl<Val, Dft, InputMmcs, FriMmcs> SingleMatrixPcs<Val, Dft, InputMmcs, FriMmcs> {
    /// Creates a new single matrix PCS with the given DFT, MMCS, and FRI parameters.
    pub const fn new(dft: Dft, mmcs: InputMmcs, fri: FriParameters<FriMmcs>) -> Self {
        Self {
            dft,
            mmcs,
            fri,
            _phantom: PhantomData,
        }
    }
}

impl<Val, Dft, InputMmcs, FriMmcs, Challenge, Challenger> Pcs<Challenge, Challenger>
    for SingleMatrixPcs<Val, Dft, InputMmcs, FriMmcs>
where
    Val: TwoAdicField,
    Dft: TwoAdicSubgroupDft<Val>,
    InputMmcs: Mmcs<Val>,
    FriMmcs: Mmcs<Challenge>,
    Challenge: ExtensionField<Val>,
    Challenger:
        FieldChallenger<Val> + CanObserve<FriMmcs::Commitment> + GrindingChallenger<Witness = Val>,
{
    type Domain = TwoAdicMultiplicativeCoset<Val>;
    type Commitment = InputMmcs::Commitment;
    type ProverData = InputMmcs::ProverData<RowMajorMatrix<Val>>;
    type EvaluationsOnDomain<'a> = BitReversedMatrixView<RowMajorMatrixView<'a, Val>>;
    type Proof = FriProof<Challenge, FriMmcs, Val, Vec<BatchOpening<Val, InputMmcs>>>;
    type Error = FriError<FriMmcs::Error, InputMmcs::Error>;
    const ZK: bool = false;

    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain {
        TwoAdicMultiplicativeCoset::new(Val::ONE, log2_strict_usize(degree)).unwrap()
    }

    /// Commits to a single matrix.
    ///
    /// This is a simplified version of the batch commit that processes exactly one
    /// (domain, matrix) pair.
    fn commit(
        &self,
        evaluations: impl IntoIterator<Item = (Self::Domain, RowMajorMatrix<Val>)>,
    ) -> (Self::Commitment, Self::ProverData) {
        let mut eval_vec: Vec<_> = evaluations.into_iter().collect();

        // Enforce single matrix constraint
        assert_eq!(
            eval_vec.len(),
            1,
            "SingleMatrixPcs only supports committing to exactly one matrix"
        );

        let (domain, evals) = eval_vec.pop().unwrap();
        assert_eq!(domain.size(), evals.height());

        // Compute LDE and bit reverse
        let shift = Val::GENERATOR / domain.shift();
        let lde = self
            .dft
            .coset_lde_batch(evals, self.fri.log_blowup, shift)
            .bit_reverse_rows()
            .to_row_major_matrix();

        // Commit to the single matrix
        self.mmcs.commit(vec![lde])
    }

    fn get_evaluations_on_domain<'a>(
        &self,
        prover_data: &'a Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> Self::EvaluationsOnDomain<'a> {
        assert_eq!(idx, 0, "SingleMatrixPcs only has one matrix (idx=0)");
        assert_eq!(domain.shift(), Val::GENERATOR);

        let matrices = self.mmcs.get_matrices(prover_data);
        assert_eq!(matrices.len(), 1, "Expected exactly one matrix");

        let lde = matrices[0];
        assert!(lde.height() >= domain.size());
        lde.split_rows(domain.size()).0.bit_reverse_rows()
    }

    /// Opens the committed matrix at the specified points.
    ///
    /// The input should contain exactly one prover data entry with one matrix's worth of points.
    fn open(
        &self,
        commitment_data_with_opening_points: Vec<(&Self::ProverData, Vec<Vec<Challenge>>)>,
        challenger: &mut Challenger,
    ) -> (OpenedValues<Challenge>, Self::Proof) {
        // Enforce single matrix constraint
        assert_eq!(
            commitment_data_with_opening_points.len(),
            1,
            "SingleMatrixPcs only supports opening one commitment"
        );

        let (data, points) = &commitment_data_with_opening_points[0];
        assert_eq!(
            points.len(),
            1,
            "SingleMatrixPcs only supports one matrix per commitment"
        );

        // Extract the single matrix
        let matrices = self.mmcs.get_matrices(data);
        assert_eq!(matrices.len(), 1, "Expected exactly one matrix");
        let mat = matrices[0].as_view();

        let points_for_mat = &points[0];

        // Compute global dimensions
        let global_max_height = mat.height();
        let global_max_width = mat.width();
        let log_global_max_height = log2_strict_usize(global_max_height);

        // Build bit-reversed coset
        let coset = {
            let coset =
                TwoAdicMultiplicativeCoset::new(Val::GENERATOR, log_global_max_height).unwrap();
            let mut coset_points = coset.iter().collect();
            reverse_slice_index_bits(&mut coset_points);
            coset_points
        };

        // Compute inverse denominators for each unique opening point
        let inv_denoms: LinearMap<Challenge, Vec<Challenge>> =
            compute_inverse_denominators_single(&mat, points_for_mat, &coset);

        // Evaluate and send openings to challenger
        let h = mat.height() >> self.fri.log_blowup;
        let (low_coset, _) = mat.split_rows(h);
        let coset_h = &coset[..h];

        let openings_for_mat: Vec<Vec<Challenge>> = points_for_mat
            .iter()
            .map(|&point| {
                let inv_denoms = &inv_denoms.get(&point).unwrap()[..h];
                let ys = interpolate_coset_with_precomputation(
                    &low_coset,
                    Val::GENERATOR,
                    point,
                    coset_h,
                    inv_denoms,
                );
                ys.iter()
                    .for_each(|&y| challenger.observe_algebra_element(y));
                ys
            })
            .collect();

        let all_opened_values = vec![vec![openings_for_mat]];

        // Batch combination challenge
        let alpha: Challenge = challenger.sample_algebra_element();

        // Compute alpha powers
        let packed_alpha_powers =
            Challenge::ExtensionPacking::packed_ext_powers_capped(alpha, global_max_width)
                .collect_vec();
        let alpha_powers =
            Challenge::ExtensionPacking::to_ext_iter(packed_alpha_powers.iter().copied())
                .collect_vec();

        // Reduce the opening proof
        let mut reduced_opening = vec![Challenge::ZERO; mat.height()];

        let mat_compressed: Vec<Challenge> = mat
            .rowwise_packed_dot_product::<Challenge>(&packed_alpha_powers)
            .collect();

        let mut num_reduced = 0; // Track total number of columns we've reduced

        for (&point, openings) in points_for_mat.iter().zip(&all_opened_values[0][0]) {
            // Apply alpha offset for multiple opening points
            let alpha_pow_offset = alpha.exp_u64(num_reduced as u64);

            let reduced_openings: Challenge =
                dot_product(alpha_powers.iter().copied(), openings.iter().copied());

            mat_compressed
                .par_iter()
                .zip(reduced_opening.par_iter_mut())
                .zip(inv_denoms.get(&point).unwrap().par_iter())
                .for_each(|((&reduced_row, ro), &inv_denom)| {
                    *ro += alpha_pow_offset * (reduced_openings - reduced_row) * inv_denom
                });

            num_reduced += mat.width();
        }

        // FRI expects Vec<Vec<Challenge>> where outer vec is for different heights
        // Since we have only one matrix, we have only one height
        let fri_input = vec![reduced_opening];

        let folding: TwoAdicFriFoldingForMmcs<Val, InputMmcs> = TwoAdicFriFolding(PhantomData);

        let fri_proof = prover::prove_fri(
            &folding,
            &self.fri,
            fri_input,
            challenger,
            log_global_max_height,
            &commitment_data_with_opening_points,
            &self.mmcs,
        );

        (all_opened_values, fri_proof)
    }

    fn verify(
        &self,
        commitments_with_opening_points: Vec<(
            Self::Commitment,
            Vec<(Self::Domain, Vec<(Challenge, Vec<Challenge>)>)>,
        )>,
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        // Enforce single matrix constraint
        assert_eq!(
            commitments_with_opening_points.len(),
            1,
            "SingleMatrixPcs only supports one commitment"
        );
        assert_eq!(
            commitments_with_opening_points[0].1.len(),
            1,
            "SingleMatrixPcs only supports one matrix"
        );

        // Write all evaluations to challenger
        for (_, round) in &commitments_with_opening_points {
            for (_, mat) in round {
                for (_, point) in mat {
                    point
                        .iter()
                        .for_each(|&opening| challenger.observe_algebra_element(opening));
                }
            }
        }

        let folding: TwoAdicFriFoldingForMmcs<Val, InputMmcs> = TwoAdicFriFolding(PhantomData);

        verifier::verify_fri(
            &folding,
            &self.fri,
            proof,
            challenger,
            &commitments_with_opening_points,
            &self.mmcs,
        )?;

        Ok(())
    }
}

/// Compute inverse denominators for a single matrix.
///
/// This is a simplified version that doesn't need to handle multiple matrices.
fn compute_inverse_denominators_single<F: TwoAdicField, EF: ExtensionField<F>, M: Matrix<F>>(
    mat: &M,
    points: &[EF],
    coset: &[F],
) -> LinearMap<EF, Vec<EF>> {
    let log_height = log2_strict_usize(mat.height());
    let height = 1 << log_height;

    points
        .iter()
        .map(|&z| {
            (
                z,
                batch_multiplicative_inverse(&coset[..height].iter().map(|&x| z - x).collect_vec()),
            )
        })
        .collect()
}
