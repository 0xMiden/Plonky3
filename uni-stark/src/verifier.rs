//! See `prover.rs` for an overview of the protocol and a more detailed soundness analysis.

use alloc::vec;
use alloc::vec::Vec;

use itertools::Itertools;
use p3_air::Air;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::stack::VerticalPair;
use p3_util::zip_eq::zip_eq;
use tracing::instrument;

use crate::symbolic_builder::{SymbolicAirBuilder, get_log_quotient_degree};
use crate::{PcsError, Proof, StarkGenericConfig, Val, VerifierConstraintFolder};

#[instrument(skip_all)]
pub fn verify<SC, A>(
    config: &SC,
    air: &A,
    proof: &Proof<SC>,
    public_values: &Vec<Val<SC>>,
) -> Result<(), VerificationError<PcsError<SC>>>
where
    SC: StarkGenericConfig,
    A: Air<SymbolicAirBuilder<Val<SC>>> + for<'a> Air<VerifierConstraintFolder<'a, SC>>,
{
    let Proof {
        commitments,
        opened_values,
        opening_proof,
        degree_bits,
    } = proof;

    let pcs = config.pcs();

    let degree = 1 << degree_bits;
    let log_quotient_degree =
        get_log_quotient_degree::<Val<SC>, A>(air, 0, public_values.len(), config.is_zk());
    let quotient_degree = 1 << (log_quotient_degree + config.is_zk());

    ark_std::println!("verifier: degree_bits={}, degree={}, log_quotient_degree={}", degree_bits, degree, log_quotient_degree);

    let mut challenger = config.initialise_challenger();
    let trace_domain = pcs.natural_domain_for_degree(degree);
    let init_trace_domain = pcs.natural_domain_for_degree(degree >> (config.is_zk()));

    let quotient_domain =
        trace_domain.create_disjoint_domain(1 << (degree_bits + log_quotient_degree));
    let quotient_chunks_domains = quotient_domain.split_domains(quotient_degree);

    ark_std::println!("quotient_domain: size={}, shift={:?}", quotient_domain.size(), quotient_domain.first_point());
    for (i, d) in quotient_chunks_domains.iter().enumerate() {
        ark_std::println!("quotient_chunks_domains[{}]: size={}, shift={:?}", i, d.size(), d.first_point());
    }

    let randomized_quotient_chunks_domains = quotient_chunks_domains
        .iter()
        .map(|domain| pcs.natural_domain_for_degree(domain.size() << (config.is_zk())))
        .collect_vec();

    for (i, d) in randomized_quotient_chunks_domains.iter().enumerate() {
        ark_std::println!("randomized_quotient_chunks_domains[{}]: size={}, shift={:?}", i, d.size(), d.first_point());
    }

    // Check that the random commitments are/are not present depending on the ZK setting.
    // - If ZK is enabled, the prover should have random commitments.
    // - If ZK is not enabled, the prover should not have random commitments.
    if (opened_values.random.is_some() != SC::Pcs::ZK)
        || (commitments.random.is_some() != SC::Pcs::ZK)
    {
        return Err(VerificationError::RandomizationError);
    }

    let air_width = A::width(air);
    let valid_shape = opened_values.trace_local.len() == air_width
        && opened_values.trace_next.len() == air_width
        && opened_values.quotient_chunks.len() == quotient_degree
        && opened_values
            .quotient_chunks
            .iter()
            .all(|qc| qc.len() == SC::Challenge::DIMENSION)
        // We've already checked that opened_values.random is present if and only if ZK is enabled.
        && if let Some(r_comm) = &opened_values.random {
            r_comm.len() == SC::Challenge::DIMENSION
        } else {
            true
        };
    ark_std::println!("valid shape: {}", valid_shape);

    if !valid_shape {
        return Err(VerificationError::InvalidProofShape);
    }

    // Observe the instance.
    challenger.observe(Val::<SC>::from_usize(proof.degree_bits));
    challenger.observe(Val::<SC>::from_usize(proof.degree_bits - config.is_zk()));
    // TODO: Might be best practice to include other instance data here in the transcript, like some
    // encoding of the AIR. This protects against transcript collisions between distinct instances.
    // Practically speaking though, the only related known attack is from failing to include public
    // values. It's not clear if failing to include other instance data could enable a transcript
    // collision, since most such changes would completely change the set of satisfying witnesses.

    challenger.observe(commitments.trace.clone());
    challenger.observe_slice(public_values);

    // begin processing aux trace
    let num_randomness = air.num_randomness();
    let randomness = if num_randomness != 0 {
        let randomness: Vec<SC::Challenge> = (0..num_randomness)
            .map(|_| challenger.sample_algebra_element())
            .collect();

        challenger.observe(commitments.aux.clone());
        randomness
    } else {
        vec![]
    };

    // Get the first Fiat Shamir challenge which will be used to combine all constraint polynomials
    // into a single polynomial.
    //
    // Soundness Error: n/|EF| where n is the number of constraints.
    let alpha = challenger.sample_algebra_element();
    ark_std::println!("verifier alpha: {:?}", alpha);

    challenger.observe(commitments.quotient_chunks.clone());

    // We've already checked that commitments.random is present if and only if ZK is enabled.
    // Observe the random commitment if it is present.
    if let Some(r_commit) = commitments.random.clone() {
        challenger.observe(r_commit);
    }

    // Get an out-of-domain point to open our values at.
    //
    // Soundness Error: dN/|EF| where `N` is the trace length and our constraint polynomial has degree `d`.
    let zeta = challenger.sample_algebra_element();
    let zeta_next = init_trace_domain.next_point(zeta).unwrap();

    ark_std::println!("verifier zeta: {:?}", zeta);
    ark_std::println!("verifier zeta_next: {:?}", zeta_next);

    // We've already checked that commitments.random and opened_values.random are present if and only if ZK is enabled.
    let mut coms_to_verify = if let Some(random_commit) = &commitments.random {
        let random_values = opened_values
            .random
            .as_ref()
            .ok_or(VerificationError::RandomizationError)?;
        vec![(
            random_commit.clone(),
            vec![(trace_domain, vec![(zeta, random_values.clone())])],
        )]
    } else {
        vec![]
    };

    ark_std::println!("prepare com for pcs");

    // The aux trace was committed as flattened base field values, so we need to
    // flatten the extension field opened values back to base field for PCS verification
    let aux_local_base: Vec<SC::Challenge> = opened_values
        .aux_trace_local
        .iter()
        .flat_map(|ef| {
            ef.as_basis_coefficients_slice()
                .iter()
                .map(|&coef| SC::Challenge::from(coef))
        })
        .collect();

    let aux_next_base: Vec<SC::Challenge> = opened_values
        .aux_trace_next
        .iter()
        .flat_map(|ef| {
            ef.as_basis_coefficients_slice()
                .iter()
                .map(|&coef| SC::Challenge::from(coef))
        })
        .collect();

    coms_to_verify.extend(vec![
        (
            commitments.trace.clone(),
            vec![(
                trace_domain,
                vec![
                    (zeta, opened_values.trace_local.clone()),
                    (zeta_next, opened_values.trace_next.clone()),
                ],
            )],
        ),
        (
            commitments.quotient_chunks.clone(),
            // Check the commitment on the randomized domains.
            zip_eq(
                randomized_quotient_chunks_domains.iter(),
                &opened_values.quotient_chunks,
                VerificationError::InvalidProofShape,
            )?
            .map(|(domain, values)| (*domain, vec![(zeta, values.clone())]))
            .collect_vec(),
        ),
        (
            commitments.aux.clone(),
            vec![(
                trace_domain,
                vec![
                    (zeta, aux_local_base),
                    (zeta_next, aux_next_base),
                ],
            )],
        ),
    ]);

    ark_std::println!("\n========== VERIFIER: Polynomial Evaluations ==========");
    ark_std::println!("Evaluation points:");
    ark_std::println!("  zeta: {:?}", zeta);
    ark_std::println!("  zeta_next: {:?}", zeta_next);
    ark_std::println!("\nTrace evaluations:");
    ark_std::println!("  trace_local: {:?}", opened_values.trace_local);
    ark_std::println!("  trace_next: {:?}", opened_values.trace_next);
    ark_std::println!("\nAux trace evaluations:");
    ark_std::println!("  aux_trace_local: {:?}", opened_values.aux_trace_local);
    ark_std::println!("  aux_trace_next: {:?}", opened_values.aux_trace_next);
    ark_std::println!("\nQuotient polynomial evaluations:");
    for (i, chunk) in opened_values.quotient_chunks.iter().enumerate() {
        ark_std::println!("  quotient_chunk[{}]: {:?}", i, chunk);
    }
    if let Some(ref r) = opened_values.random {
        ark_std::println!("\nRandom polynomial evaluation:");
        ark_std::println!("  random: {:?}", r);
    }
    ark_std::println!("====================================================\n");

    ark_std::println!("start to verify pcs");
    pcs.verify(coms_to_verify, opening_proof, &mut challenger)
        .map_err(VerificationError::InvalidOpeningArgument)?;

    ark_std::println!("pcs verified");
    let zps = quotient_chunks_domains
        .iter()
        .enumerate()
        .map(|(i, domain)| {
            quotient_chunks_domains
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, other_domain)| {
                    other_domain.vanishing_poly_at_point(zeta)
                        * other_domain
                            .vanishing_poly_at_point(domain.first_point())
                            .inverse()
                })
                .product::<SC::Challenge>()
        })
        .collect_vec();

    ark_std::println!("start to compute quotient");
    ark_std::println!("zps: {:?}", zps);
    let zps_sum: SC::Challenge = zps.iter().copied().sum();
    ark_std::println!("zps sum: {:?}", zps_sum);
    ark_std::println!("quotient_chunks: {:?}", opened_values.quotient_chunks);
    let quotient = opened_values
        .quotient_chunks
        .iter()
        .enumerate()
        .map(|(ch_i, ch)| {
            // We checked in valid_shape the length of "ch" is equal to
            // <SC::Challenge as BasedVectorSpace<Val<SC>>>::DIMENSION. Hence
            // the unwrap() will never panic.
            let chunk_val = ch.iter()
                .enumerate()
                .map(|(e_i, &c)| SC::Challenge::ith_basis_element(e_i).unwrap() * c)
                .sum::<SC::Challenge>();
            ark_std::println!("chunk {}: value={:?}, zps={:?}, contribution={:?}",
                ch_i, chunk_val, zps[ch_i], zps[ch_i] * chunk_val);
            zps[ch_i] * chunk_val
        })
        .sum::<SC::Challenge>();

    ark_std::println!("verifier: trace_domain size={}, shift={:?}, init_trace_domain size={}, shift={:?}",
        trace_domain.size(), trace_domain.first_point(), init_trace_domain.size(), init_trace_domain.first_point());

    let sels = init_trace_domain.selectors_at_point(zeta);

    let main = VerticalPair::new(
        RowMajorMatrixView::new_row(&opened_values.trace_local),
        RowMajorMatrixView::new_row(&opened_values.trace_next),
    );

    let aux = VerticalPair::new(
        RowMajorMatrixView::new_row(&opened_values.aux_trace_local),
        RowMajorMatrixView::new_row(&opened_values.aux_trace_next),
    );

    ark_std::println!("\n========== VERIFIER: Constraint Evaluation ==========");
    ark_std::println!("Selectors at zeta:");
    ark_std::println!("  is_first_row: {:?}", sels.is_first_row);
    ark_std::println!("  is_last_row: {:?}", sels.is_last_row);
    ark_std::println!("  is_transition: {:?}", sels.is_transition);
    ark_std::println!("  inv_vanishing: {:?}", sels.inv_vanishing);
    ark_std::println!("\nRandomness:");
    ark_std::println!("  {:?}", randomness);

    let mut folder = VerifierConstraintFolder {
        main,
        aux,
        randomness: &randomness,
        public_values,
        is_first_row: sels.is_first_row,
        is_last_row: sels.is_last_row,
        is_transition: sels.is_transition,
        alpha,
        accumulator: SC::Challenge::ZERO,
    };
    air.eval(&mut folder);
    let folded_constraints = folder.accumulator;

    ark_std::println!("folded_constraints: {:?}", folded_constraints);
    let expected_constraints = quotient * sels.is_transition;  // Z_H(zeta) = zeta - 1 = is_transition for size-1 domain
    ark_std::println!("expected_constraints (quotient * Z_H): {:?}", expected_constraints);
    ark_std::println!(
        "prod: {:?}\nquotient: {:?}",
        folded_constraints * sels.inv_vanishing,
        quotient
    );
    // Finally, check that
    //     folded_constraints(zeta) / Z_H(zeta) = quotient(zeta)
    if folded_constraints * sels.inv_vanishing != quotient {
        return Err(VerificationError::OodEvaluationMismatch);
    }

    Ok(())
}

#[derive(Debug)]
pub enum VerificationError<PcsErr> {
    InvalidProofShape,
    /// An error occurred while verifying the claimed openings.
    InvalidOpeningArgument(PcsErr),
    /// Out-of-domain evaluation mismatch, i.e. `constraints(zeta)` did not match
    /// `quotient(zeta) Z_H(zeta)`.
    OodEvaluationMismatch,
    /// The FRI batch randomization does not correspond to the ZK setting.
    RandomizationError,
}
