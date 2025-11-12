use alloc::vec::Vec;

use p3_air::{AirBuilder, AirBuilderWithPublicValues, ExtensionBuilder, PermutationAirBuilder};
use p3_field::{BasedVectorSpace, PackedField, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::stack::ViewPair;

use crate::{PackedChallenge, PackedVal, StarkGenericConfig, Val};

/// Handles constraint accumulation for the prover in a STARK system.
///
/// This struct is responsible for evaluating constraints corresponding to a given row in the trace matrix.
/// It accumulates them into a single value using a randomized challenge.
/// `C_0 + alpha C_1 + alpha^2 C_2 + ...`
#[derive(Debug)]
pub struct ProverConstraintFolder<'a, SC: StarkGenericConfig> {
    /// The matrix containing rows on which the constraint polynomial is to be evaluated
    pub main: RowMajorMatrixView<'a, PackedVal<SC>>,
    /// Optional: the matrix containing rows on which the aux constraint polynomial is to be evaluated
    pub aux: Option<RowMajorMatrixView<'a, PackedVal<SC>>>,
    /// Optional: the randomness used to compute the aux tract
    /// Cached EF randomness packed from base randomness to avoid temporary leaks
    pub packed_randomness: Vec<PackedChallenge<SC>>,
    /// Public inputs to the AIR
    pub public_values: &'a Vec<Val<SC>>,
    /// Evaluations of the Selector polynomial for the first row of the trace
    pub is_first_row: PackedVal<SC>,
    /// Evaluations of the Selector polynomial for the last row of the trace
    pub is_last_row: PackedVal<SC>,
    /// Evaluations of the Selector polynomial for rows where transition constraints should be applied
    pub is_transition: PackedVal<SC>,
    /// Challenge powers used for randomized constraint combination
    pub alpha_powers: &'a [SC::Challenge],
    /// Challenge powers decomposed into their base field component.
    pub decomposed_alpha_powers: &'a [Vec<Val<SC>>],
    /// Running accumulator for all constraints multiplied by challenge powers
    /// `C_0 + alpha C_1 + alpha^2 C_2 + ...`
    pub accumulator: PackedChallenge<SC>,
    /// Current constraint index being processed
    pub constraint_index: usize,
}

/// Handles constraint verification for the verifier in a STARK system.
///
/// Similar to ProverConstraintFolder but operates on committed values rather than the full trace,
/// using a more efficient accumulation method for verification.
#[derive(Debug)]
pub struct VerifierConstraintFolder<'a, SC: StarkGenericConfig> {
    /// Pair of consecutive rows from the committed polynomial evaluations
    pub main: ViewPair<'a, SC::Challenge>,
    /// Optional: pair of consecutive rows from the committed polynomial evaluations
    pub aux: Option<ViewPair<'a, SC::Challenge>>,
    /// Optional: the randomness used to compute the aux tract
    pub randomness: &'a [SC::Challenge],
    /// Public values that are inputs to the computation
    pub public_values: &'a Vec<Val<SC>>,
    /// Evaluations of the Selector polynomial for the first row of the trace
    pub is_first_row: SC::Challenge,
    /// Evaluations of the Selector polynomial for the last row of the trace
    pub is_last_row: SC::Challenge,
    /// Evaluations of the Selector polynomial for rows where transition constraints should be applied
    pub is_transition: SC::Challenge,
    /// Single challenge value used for constraint combination
    pub alpha: SC::Challenge,
    /// Running accumulator for all constraints
    pub accumulator: SC::Challenge,
}

#[derive(Clone, Copy, Debug)]
pub struct EfAuxView<'a, SC: StarkGenericConfig> {
    inner: RowMajorMatrixView<'a, PackedVal<SC>>,
}

impl<'a, SC: StarkGenericConfig> EfAuxView<'a, SC> {
    pub fn new(inner: RowMajorMatrixView<'a, PackedVal<SC>>) -> Self {
        #[cfg(debug_assertions)]
        {
            let d = <SC::Challenge as BasedVectorSpace<Val<SC>>>::DIMENSION;
            debug_assert!(
                inner.is_multiple_of(d),
                "aux trace width must be a multiple of EF dimension"
            );
        }
        Self { inner }
    }
}

/// Verifier-side EF view over aux rows committed as flattened base limbs.
///
/// PCS commits and opens base-field polynomials. Our aux EF columns are flattened into
/// `d = [E:F]` base columns using a fixed F-basis `{e_i}` for the extension field `E`.
/// At verification, PCS returns openings for each of those base columns (typed as
/// `SC::Challenge` values in E via the canonical embedding). To recover the EF value of one
/// aux column at the point `ζ ∈ E`, we recombine the opened limbs using the F-linear identity:
///   Y(ζ) = Σ_i e_i · Y_i(ζ),  where Y(x) = Σ_i e_i · Y_i(x), Y_i ∈ F[x].
///
/// This view performs exactly that recombination, grouping every `d` consecutive base limbs
/// into a single EF element. This keeps the AIR EF-only, while preserving PCS compatibility.
#[derive(Clone, Copy, Debug)]
pub struct VerifierEfAuxView<'a, SC: StarkGenericConfig> {
    inner: ViewPair<'a, SC::Challenge>,
}

impl<'a, SC: StarkGenericConfig> VerifierEfAuxView<'a, SC> {
    pub fn new(inner: ViewPair<'a, SC::Challenge>) -> Self {
        #[cfg(debug_assertions)]
        {
            let d = <SC::Challenge as BasedVectorSpace<Val<SC>>>::DIMENSION;
            debug_assert!(
                inner.width() % d == 0,
                "aux trace width must be a multiple of EF dimension"
            );
        }
        Self { inner }
    }
}

impl<'a, SC: StarkGenericConfig> Matrix<SC::Challenge> for VerifierEfAuxView<'a, SC> {
    fn width(&self) -> usize {
        let d = <SC::Challenge as BasedVectorSpace<Val<SC>>>::DIMENSION;
        self.inner.width() / d
    }

    fn height(&self) -> usize {
        2
    }

    fn get(&self, r: usize, c: usize) -> Option<SC::Challenge> {
        if r >= 2 || c >= self.width() {
            return None;
        }
        let row = if r == 0 {
            self.inner.top
        } else {
            self.inner.bottom
        };
        let d = <SC::Challenge as BasedVectorSpace<Val<SC>>>::DIMENSION;
        // Recombine EF element at column `c` from its `d` base limbs using the canonical basis.
        let mut acc = SC::Challenge::ZERO;
        for i in 0..d {
            let limb = row.get(0, c * d + i)?;
            let basis_i = SC::Challenge::ith_basis_element(i).unwrap();
            acc += basis_i * limb;
        }
        Some(acc)
    }

    fn row_slice(&self, r: usize) -> Option<impl core::ops::Deref<Target = [SC::Challenge]>> {
        if r >= 2 {
            return None;
        }
        let d = <SC::Challenge as BasedVectorSpace<Val<SC>>>::DIMENSION;
        let ef_w = self.width();
        let row = if r == 0 {
            self.inner.top
        } else {
            self.inner.bottom
        };
        // Recombine every EF column in this row from its `d` base limbs.
        let mut v: alloc::vec::Vec<SC::Challenge> = alloc::vec::Vec::with_capacity(ef_w);
        for ec in 0..ef_w {
            let mut acc = SC::Challenge::ZERO;
            for i in 0..d {
                let limb = row.get(0, ec * d + i).unwrap();
                let basis_i = SC::Challenge::ith_basis_element(i).unwrap();
                acc += basis_i * limb;
            }
            v.push(acc);
        }
        Some(v)
    }
}

impl<'a, SC: StarkGenericConfig> Matrix<PackedChallenge<SC>> for EfAuxView<'a, SC> {
    fn width(&self) -> usize {
        let d = <SC::Challenge as BasedVectorSpace<Val<SC>>>::DIMENSION;
        self.inner.width() / d
    }

    fn height(&self) -> usize {
        self.inner.height()
    }

    fn get(&self, r: usize, c: usize) -> Option<PackedChallenge<SC>> {
        if r >= self.height() || c >= self.width() {
            return None;
        }
        let d = <SC::Challenge as BasedVectorSpace<Val<SC>>>::DIMENSION;
        Some(PackedChallenge::<SC>::from_basis_coefficients_fn(|i| {
            self.inner.get(r, c * d + i).unwrap()
        }))
    }

    fn row_slice(&self, r: usize) -> Option<impl core::ops::Deref<Target = [PackedChallenge<SC>]>> {
        if r >= self.height() {
            return None;
        }
        let d = <SC::Challenge as BasedVectorSpace<Val<SC>>>::DIMENSION;
        let ef_w = self.width();
        let mut row: alloc::vec::Vec<PackedChallenge<SC>> = alloc::vec::Vec::with_capacity(ef_w);
        for ec in 0..ef_w {
            row.push(PackedChallenge::<SC>::from_basis_coefficients_fn(|i| {
                self.inner.get(r, ec * d + i).unwrap()
            }));
        }
        Some(row)
    }
}

impl<'a, SC: StarkGenericConfig> AirBuilder for ProverConstraintFolder<'a, SC> {
    type F = Val<SC>;
    type Expr = PackedVal<SC>;
    type Var = PackedVal<SC>;
    type M = RowMajorMatrixView<'a, PackedVal<SC>>;

    #[inline]
    fn main(&self) -> Self::M {
        self.main
    }

    #[inline]
    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row
    }

    #[inline]
    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row
    }

    /// Returns an expression indicating rows where transition constraints should be checked.
    ///
    /// # Panics
    /// This function panics if `size` is not `2`.
    #[inline]
    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            self.is_transition
        } else {
            panic!("uni-stark only supports a window size of 2")
        }
    }

    #[inline]
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let alpha_power = self.alpha_powers[self.constraint_index];
        self.accumulator += Into::<PackedChallenge<SC>>::into(alpha_power) * x.into();
        self.constraint_index += 1;
    }

    #[inline]
    fn assert_zeros<const N: usize, I: Into<Self::Expr>>(&mut self, array: [I; N]) {
        let expr_array = array.map(Into::into);
        self.accumulator += PackedChallenge::<SC>::from_basis_coefficients_fn(|i| {
            let alpha_powers = &self.decomposed_alpha_powers[i]
                [self.constraint_index..(self.constraint_index + N)];
            PackedVal::<SC>::packed_linear_combination::<N>(alpha_powers, &expr_array)
        });
        self.constraint_index += N;
    }
}

impl<SC: StarkGenericConfig> AirBuilderWithPublicValues for ProverConstraintFolder<'_, SC> {
    type PublicVar = Self::F;

    #[inline]
    fn public_values(&self) -> &[Self::F] {
        self.public_values
    }
}

impl<SC: StarkGenericConfig> ExtensionBuilder for ProverConstraintFolder<'_, SC> {
    type EF = SC::Challenge;
    type ExprEF = PackedChallenge<SC>;
    type VarEF = PackedChallenge<SC>;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        let alpha_power = self.alpha_powers[self.constraint_index];
        self.accumulator += Into::<PackedChallenge<SC>>::into(alpha_power) * x.into();
        self.constraint_index += 1;
    }
}

impl<'a, SC: StarkGenericConfig> PermutationAirBuilder for ProverConstraintFolder<'a, SC> {
    type MP = EfAuxView<'a, SC>;
    type RandomVar = PackedChallenge<SC>;

    fn permutation(&self) -> Self::MP {
        EfAuxView::new(self.aux.expect("permutation called but aux trace is None - AIR should check num_randomness > 0 before using permutation columns"))
    }

    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        self.packed_randomness.as_slice()
    }
}

impl<'a, SC: StarkGenericConfig> AirBuilder for VerifierConstraintFolder<'a, SC> {
    type F = Val<SC>;
    type Expr = SC::Challenge;
    type Var = SC::Challenge;
    type M = ViewPair<'a, SC::Challenge>;

    fn main(&self) -> Self::M {
        self.main
    }

    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row
    }

    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row
    }

    /// Returns an expression indicating rows where transition constraints should be checked.
    ///
    /// # Panics
    /// This function panics if `size` is not `2`.
    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            self.is_transition
        } else {
            panic!("uni-stark only supports a window size of 2")
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.accumulator *= self.alpha;
        self.accumulator += x.into();
    }
}

impl<SC: StarkGenericConfig> AirBuilderWithPublicValues for VerifierConstraintFolder<'_, SC> {
    type PublicVar = Self::F;

    fn public_values(&self) -> &[Self::F] {
        self.public_values
    }
}

impl<SC: StarkGenericConfig> ExtensionBuilder for VerifierConstraintFolder<'_, SC> {
    type EF = SC::Challenge;
    type ExprEF = SC::Challenge;
    type VarEF = SC::Challenge;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        self.accumulator *= self.alpha;
        self.accumulator += x.into();
    }
}

impl<'a, SC: StarkGenericConfig> PermutationAirBuilder for VerifierConstraintFolder<'a, SC> {
    type MP = VerifierEfAuxView<'a, SC>;
    type RandomVar = SC::Challenge;

    fn permutation(&self) -> Self::MP {
        VerifierEfAuxView::new(self.aux.expect("permutation called but aux trace is None - AIR should check num_randomness > 0 before using permutation columns"))
    }

    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        self.randomness
    }
}
