use core::marker::PhantomData;

use p3_challenger::{CanObserve, CanSample, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrix;

pub type PcsError<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::Error;

pub type Domain<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::Domain;

pub type Val<SC> = <Domain<SC> as PolynomialSpace>::Val;

pub type PackedVal<SC> = <Val<SC> as Field>::Packing;

pub type PackedChallenge<SC> =
    <<SC as StarkGenericConfig>::Challenge as ExtensionField<Val<SC>>>::ExtensionPacking;

pub trait StarkGenericConfig {
    /// The PCS used to commit to trace polynomials.
    type Pcs: Pcs<Self::Challenge, Self::Challenger>;

    /// The field from which most random challenges are drawn.
    type Challenge: ExtensionField<Val<Self>>;

    /// The challenger (Fiat-Shamir) implementation used.
    type Challenger: FieldChallenger<Val<Self>>
        + CanObserve<<Self::Pcs as Pcs<Self::Challenge, Self::Challenger>>::Commitment>
        + CanSample<Self::Challenge>;

    /// Get a reference to the PCS used by this proof configuration.
    fn pcs(&self) -> &Self::Pcs;

    /// Get an initialisation of the challenger used by this proof configuration.
    fn initialise_challenger(&self) -> Self::Challenger;

    /// Returns 1 if the PCS is zero-knowledge, 0 otherwise.
    fn is_zk(&self) -> usize {
        Self::Pcs::ZK as usize
    }

    /// Number of EF challenges used to build the aux trace (LogUp/perm arguments).
    /// Default is 0 (no aux).
    fn aux_challenges(&self) -> usize {
        0
    }

    /// Optionally build an aux trace (EF-based) given the main trace and EF challenges.
    /// Return None to indicate no aux or to fall back to legacy behavior.
    fn build_aux_trace(
        &self,
        _main: &RowMajorMatrix<
            <<Self::Pcs as Pcs<Self::Challenge, Self::Challenger>>::Domain as PolynomialSpace>::Val,
        >,
        _challenges: &[Self::Challenge],
    ) -> Option<
        RowMajorMatrix<
            <<Self::Pcs as Pcs<Self::Challenge, Self::Challenger>>::Domain as PolynomialSpace>::Val,
        >,
    > {
        None
    }

    /// Optional: width of the aux trace when flattened in base field elements.
    fn aux_width_in_base_field(&self) -> usize {
        0
    }
}

pub struct StarkConfig<Pcs, Challenge, Challenger>
where
    Pcs: p3_commit::Pcs<Challenge, Challenger>,
    Challenge:
        ExtensionField<<<Pcs as p3_commit::Pcs<Challenge, Challenger>>::Domain as PolynomialSpace>::Val>,
{
    /// The PCS used to commit polynomials and prove opening proofs.
    pcs: Pcs,
    /// An initialised instance of the challenger.
    challenger: Challenger,
    _phantom: PhantomData<Challenge>,
    /// Optional: number of EF challenges used to build aux trace.
    aux_challenges: usize,
    /// Optional: aux width (flattened in base field elements).
    aux_width: usize,
    /// Optional: aux trace builder callback.
    aux_builder: Option<
        alloc::boxed::Box<
            dyn Fn(
                    &RowMajorMatrix<<Pcs::Domain as PolynomialSpace>::Val>,
                    &[Challenge],
                ) -> RowMajorMatrix<<Pcs::Domain as PolynomialSpace>::Val>
                + Send
                + Sync,
        >,
    >,
}

impl<Pcs, Challenge, Challenger> StarkConfig<Pcs, Challenge, Challenger>
where
    Pcs: p3_commit::Pcs<Challenge, Challenger>,
    Challenge:
        ExtensionField<<<Pcs as p3_commit::Pcs<Challenge, Challenger>>::Domain as PolynomialSpace>::Val>,
{
    pub const fn new(pcs: Pcs, challenger: Challenger) -> Self {
        Self {
            pcs,
            challenger,
            _phantom: PhantomData,
            aux_challenges: 0,
            aux_width: 0,
            aux_builder: None,
        }
    }
}

impl<Pcs, Challenge, Challenger> StarkGenericConfig for StarkConfig<Pcs, Challenge, Challenger>
where
    Challenge: ExtensionField<<Pcs::Domain as PolynomialSpace>::Val>,
    Pcs: p3_commit::Pcs<Challenge, Challenger>,
    Challenger: FieldChallenger<<Pcs::Domain as PolynomialSpace>::Val>
        + CanObserve<Pcs::Commitment>
        + CanSample<Challenge>
        + Clone,
{
    type Pcs = Pcs;
    type Challenge = Challenge;
    type Challenger = Challenger;

    fn pcs(&self) -> &Self::Pcs {
        &self.pcs
    }

    fn initialise_challenger(&self) -> Self::Challenger {
        self.challenger.clone()
    }

    fn aux_challenges(&self) -> usize {
        self.aux_challenges
    }

    fn build_aux_trace(
        &self,
        main: &RowMajorMatrix<<Pcs::Domain as PolynomialSpace>::Val>,
        challenges: &[Challenge],
    ) -> Option<RowMajorMatrix<<Pcs::Domain as PolynomialSpace>::Val>> {
        self.aux_builder
            .as_ref()
            .map(|f| (f)(main, challenges))
    }

    fn aux_width_in_base_field(&self) -> usize {
        self.aux_width
    }
}

impl<Pcs, Challenge, Challenger> StarkConfig<Pcs, Challenge, Challenger>
where
    Pcs: p3_commit::Pcs<Challenge, Challenger>,
    Challenge:
        ExtensionField<<<Pcs as p3_commit::Pcs<Challenge, Challenger>>::Domain as PolynomialSpace>::Val>,
{
    pub fn with_aux_builder<F>(mut self, aux_challenges: usize, aux_width: usize, f: F) -> Self
    where
        F: Fn(
                &RowMajorMatrix<<Pcs::Domain as PolynomialSpace>::Val>,
                &[Challenge],
            ) -> RowMajorMatrix<<Pcs::Domain as PolynomialSpace>::Val>
            + Send
            + Sync
            + 'static,
    {
        self.aux_challenges = aux_challenges;
        self.aux_width = aux_width;
        self.aux_builder = Some(alloc::boxed::Box::new(f));
        self
    }
}
