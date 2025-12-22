use core::iter::chain;

use p3_field::Field;

use crate::CryptographicHasher;
use crate::permutation::CryptographicPermutation;

/// Trait for stateful sponge-like hashers.
///
/// A stateful hasher maintains an external state value that evolves as input is
/// absorbed, and from which fixed-size outputs can be squeezed. This interface
/// is used pervasively by commitment schemes and Merkle trees to incrementally
/// absorb rows of matrices and later read out the final digest.
///
/// Padding semantics:
/// - Each implementation exposes a constant `PADDING_WIDTH` that counts how many
///   input items constitute one horizontal "padding unit". Callers may treat each
///   input slice as implicitly padded with zeros to a multiple of `PADDING_WIDTH`.
/// - Importantly, `PADDING_WIDTH` is measured in units of `Item`, not bytes.
///   For example, a field-to-bytes adapter has `PADDING_WIDTH = 1`
///   (one more field element extends the input by one item), while a field-native
///   sponge with rate `R` has `PADDING_WIDTH = R` (in field elements).
pub trait StatefulHasher<Item, State, Out>: Clone {
    /// The horizontal padding width for absorption, expressed in `Item` units.
    /// Default is 1.
    const PADDING_WIDTH: usize = 1;

    /// Returns the default/zero state for this hasher.
    fn default_state(&self) -> State;

    /// Absorb elements into the state with overwrite-mode and zero-padding semantics if applicable.
    fn absorb_into<I>(&self, state: &mut State, input: I)
    where
        I: IntoIterator<Item = Item>;

    /// Squeeze an output from the current state.
    fn squeeze(&self, state: &State) -> Out;
}

// =============================================================================
// StatefulSponge - wraps CryptographicPermutation
// =============================================================================

/// A stateful sponge wrapper around a cryptographic permutation.
///
/// This implements proper sponge absorption semantics where the state evolves
/// with each absorption. Unlike `PaddingFreeSponge` (which implements `CryptographicHasher`),
/// this struct only implements `StatefulHasher`.
///
/// `WIDTH` is the sponge's rate plus capacity.
/// `RATE` is the number of elements absorbed per permutation.
/// `OUT` is the number of elements squeezed from the state.
#[derive(Copy, Clone, Debug)]
pub struct StatefulSponge<P, const WIDTH: usize, const RATE: usize, const OUT: usize> {
    permutation: P,
}

impl<P, const WIDTH: usize, const RATE: usize, const OUT: usize>
    StatefulSponge<P, WIDTH, RATE, OUT>
{
    pub const fn new(permutation: P) -> Self {
        Self { permutation }
    }
}

impl<P, T, const WIDTH: usize, const RATE: usize, const OUT: usize>
    StatefulHasher<T, [T; WIDTH], [T; OUT]> for StatefulSponge<P, WIDTH, RATE, OUT>
where
    T: Default + Clone,
    P: CryptographicPermutation<[T; WIDTH]>,
{
    const PADDING_WIDTH: usize = RATE;

    fn default_state(&self) -> [T; WIDTH] {
        core::array::from_fn(|_| T::default())
    }

    fn absorb_into<I>(&self, state: &mut [T; WIDTH], input: I)
    where
        I: IntoIterator<Item = T>,
    {
        const { assert!(OUT < WIDTH) }
        let mut input = input.into_iter();

        'outer: loop {
            for i in 0..RATE {
                if let Some(x) = input.next() {
                    state[i] = x;
                } else {
                    if i != 0 {
                        state[i..RATE].fill(T::default());
                        self.permutation.permute_mut(state);
                    }
                    break 'outer;
                }
            }
            self.permutation.permute_mut(state);
        }
    }

    fn squeeze(&self, state: &[T; WIDTH]) -> [T; OUT] {
        const { assert!(OUT < WIDTH) }
        core::array::from_fn(|i| state[i].clone())
    }
}

// =============================================================================
// SerializingStatefulSponge - wraps binary StatefulSponge
// =============================================================================

/// An adapter that serializes field elements to binary and delegates to an inner `StatefulHasher`.
///
/// This mirrors `SerializingHasher`'s conversions from fields to bytes/u32/u64 streams,
/// but implements the `StatefulHasher` interface by delegating to an inner stateful hasher
/// that operates on binary data.
///
/// Unlike `ChainingHasher` (which uses chaining mode `H(state || input)`), this adapter
/// preserves proper sponge absorption semantics by directly calling the inner hasher's
/// `absorb_into` method.
#[derive(Copy, Clone, Debug)]
pub struct SerializingStatefulSponge<Inner> {
    inner: Inner,
}

impl<Inner> SerializingStatefulSponge<Inner> {
    pub const fn new(inner: Inner) -> Self {
        Self { inner }
    }
}

// -----------------------------------------------------------------------------
// Scalar implementations: F -> binary
// -----------------------------------------------------------------------------

// Scalar field -> u8 based inner
impl<F, Inner, const WIDTH: usize, const OUT: usize> StatefulHasher<F, [u8; WIDTH], [u8; OUT]>
    for SerializingStatefulSponge<Inner>
where
    F: Field,
    Inner: StatefulHasher<u8, [u8; WIDTH], [u8; OUT]>,
{
    fn default_state(&self) -> [u8; WIDTH] {
        self.inner.default_state()
    }

    fn absorb_into<I>(&self, state: &mut [u8; WIDTH], input: I)
    where
        I: IntoIterator<Item = F>,
    {
        self.inner.absorb_into(state, F::into_byte_stream(input));
    }

    fn squeeze(&self, state: &[u8; WIDTH]) -> [u8; OUT] {
        self.inner.squeeze(state)
    }
}

// Scalar field -> u32 based inner
impl<F, Inner, const WIDTH: usize, const OUT: usize> StatefulHasher<F, [u32; WIDTH], [u32; OUT]>
    for SerializingStatefulSponge<Inner>
where
    F: Field,
    Inner: StatefulHasher<u32, [u32; WIDTH], [u32; OUT]>,
{
    fn default_state(&self) -> [u32; WIDTH] {
        self.inner.default_state()
    }

    fn absorb_into<I>(&self, state: &mut [u32; WIDTH], input: I)
    where
        I: IntoIterator<Item = F>,
    {
        self.inner.absorb_into(state, F::into_u32_stream(input));
    }

    fn squeeze(&self, state: &[u32; WIDTH]) -> [u32; OUT] {
        self.inner.squeeze(state)
    }
}

// Scalar field -> u64 based inner
impl<F, Inner, const WIDTH: usize, const OUT: usize> StatefulHasher<F, [u64; WIDTH], [u64; OUT]>
    for SerializingStatefulSponge<Inner>
where
    F: Field,
    Inner: StatefulHasher<u64, [u64; WIDTH], [u64; OUT]>,
{
    fn default_state(&self) -> [u64; WIDTH] {
        self.inner.default_state()
    }

    fn absorb_into<I>(&self, state: &mut [u64; WIDTH], input: I)
    where
        I: IntoIterator<Item = F>,
    {
        self.inner.absorb_into(state, F::into_u64_stream(input));
    }

    fn squeeze(&self, state: &[u64; WIDTH]) -> [u64; OUT] {
        self.inner.squeeze(state)
    }
}

// -----------------------------------------------------------------------------
// Parallel implementations: [F; M] -> [binary; M]
// -----------------------------------------------------------------------------

// Parallel [F; M] -> [u8; M] based inner
impl<F, Inner, const WIDTH: usize, const OUT: usize, const M: usize>
    StatefulHasher<[F; M], [[u8; M]; WIDTH], [[u8; M]; OUT]> for SerializingStatefulSponge<Inner>
where
    F: Field,
    Inner: StatefulHasher<[u8; M], [[u8; M]; WIDTH], [[u8; M]; OUT]>,
{
    fn default_state(&self) -> [[u8; M]; WIDTH] {
        self.inner.default_state()
    }

    fn absorb_into<I>(&self, state: &mut [[u8; M]; WIDTH], input: I)
    where
        I: IntoIterator<Item = [F; M]>,
    {
        self.inner
            .absorb_into(state, F::into_parallel_byte_streams(input));
    }

    fn squeeze(&self, state: &[[u8; M]; WIDTH]) -> [[u8; M]; OUT] {
        self.inner.squeeze(state)
    }
}

// Parallel [F; M] -> [u32; M] based inner
impl<F, Inner, const WIDTH: usize, const OUT: usize, const M: usize>
    StatefulHasher<[F; M], [[u32; M]; WIDTH], [[u32; M]; OUT]> for SerializingStatefulSponge<Inner>
where
    F: Field,
    Inner: StatefulHasher<[u32; M], [[u32; M]; WIDTH], [[u32; M]; OUT]>,
{
    fn default_state(&self) -> [[u32; M]; WIDTH] {
        self.inner.default_state()
    }

    fn absorb_into<I>(&self, state: &mut [[u32; M]; WIDTH], input: I)
    where
        I: IntoIterator<Item = [F; M]>,
    {
        self.inner
            .absorb_into(state, F::into_parallel_u32_streams(input));
    }

    fn squeeze(&self, state: &[[u32; M]; WIDTH]) -> [[u32; M]; OUT] {
        self.inner.squeeze(state)
    }
}

// Parallel [F; M] -> [u64; M] based inner
impl<F, Inner, const WIDTH: usize, const OUT: usize, const M: usize>
    StatefulHasher<[F; M], [[u64; M]; WIDTH], [[u64; M]; OUT]> for SerializingStatefulSponge<Inner>
where
    F: Field,
    Inner: StatefulHasher<[u64; M], [[u64; M]; WIDTH], [[u64; M]; OUT]>,
{
    fn default_state(&self) -> [[u64; M]; WIDTH] {
        self.inner.default_state()
    }

    fn absorb_into<I>(&self, state: &mut [[u64; M]; WIDTH], input: I)
    where
        I: IntoIterator<Item = [F; M]>,
    {
        self.inner
            .absorb_into(state, F::into_parallel_u64_streams(input));
    }

    fn squeeze(&self, state: &[[u64; M]; WIDTH]) -> [[u64; M]; OUT] {
        self.inner.squeeze(state)
    }
}

// =============================================================================
// ChainingHasher - wraps CryptographicHasher
// =============================================================================

/// An adapter that chains state with new input, hashing `state || encode(input)`.
///
/// This mirrors `SerializingHasher`'s conversions from fields to bytes/u32/u64 streams,
/// but implements the `StatefulHasher` interface where the state is the digest itself.
///
/// Unlike `SerializingStatefulSponge` (which preserves proper sponge semantics),
/// this adapter uses chaining mode where each absorption computes `H(state || input)`.
#[derive(Copy, Clone, Debug)]
pub struct ChainingHasher<Inner> {
    inner: Inner,
}

impl<Inner> ChainingHasher<Inner> {
    pub const fn new(inner: Inner) -> Self {
        Self { inner }
    }
}

// -----------------------------------------------------------------------------
// Scalar implementations: F -> digest
// -----------------------------------------------------------------------------

// Scalar field -> byte digest
impl<F, Inner, const N: usize> StatefulHasher<F, [u8; N], [u8; N]> for ChainingHasher<Inner>
where
    F: Field,
    Inner: CryptographicHasher<u8, [u8; N]>,
{
    fn default_state(&self) -> [u8; N] {
        [0u8; N]
    }

    fn absorb_into<I>(&self, state: &mut [u8; N], input: I)
    where
        I: IntoIterator<Item = F>,
    {
        let prev = *state;
        *state = self
            .inner
            .hash_iter(chain(prev, F::into_byte_stream(input)));
    }

    fn squeeze(&self, state: &[u8; N]) -> [u8; N] {
        *state
    }
}

// Scalar field -> u32 digest
impl<F, Inner, const N: usize> StatefulHasher<F, [u32; N], [u32; N]> for ChainingHasher<Inner>
where
    F: Field,
    Inner: CryptographicHasher<u32, [u32; N]>,
{
    fn default_state(&self) -> [u32; N] {
        [0u32; N]
    }

    fn absorb_into<I>(&self, state: &mut [u32; N], input: I)
    where
        I: IntoIterator<Item = F>,
    {
        let prev = *state;
        *state = self.inner.hash_iter(chain(prev, F::into_u32_stream(input)));
    }

    fn squeeze(&self, state: &[u32; N]) -> [u32; N] {
        *state
    }
}

// Scalar field -> u64 digest
impl<F, Inner, const N: usize> StatefulHasher<F, [u64; N], [u64; N]> for ChainingHasher<Inner>
where
    F: Field,
    Inner: CryptographicHasher<u64, [u64; N]>,
{
    fn default_state(&self) -> [u64; N] {
        [0u64; N]
    }

    fn absorb_into<I>(&self, state: &mut [u64; N], input: I)
    where
        I: IntoIterator<Item = F>,
    {
        let prev = *state;
        *state = self.inner.hash_iter(chain(prev, F::into_u64_stream(input)));
    }

    fn squeeze(&self, state: &[u64; N]) -> [u64; N] {
        *state
    }
}

// -----------------------------------------------------------------------------
// Parallel implementations: [F; M] -> [[binary; M]; OUT]
// -----------------------------------------------------------------------------

// Parallel lanes (array-based) implemented via per-lane scalar hashing.
impl<F, Inner, const OUT: usize, const M: usize>
    StatefulHasher<[F; M], [[u8; M]; OUT], [[u8; M]; OUT]> for ChainingHasher<Inner>
where
    F: Field,
    Inner: CryptographicHasher<[u8; M], [[u8; M]; OUT]>,
{
    fn default_state(&self) -> [[u8; M]; OUT] {
        [[0u8; M]; OUT]
    }

    fn absorb_into<I>(&self, state: &mut [[u8; M]; OUT], input: I)
    where
        I: IntoIterator<Item = [F; M]>,
    {
        let prev = *state;
        *state = self
            .inner
            .hash_iter(chain(prev, F::into_parallel_byte_streams(input)));
    }

    fn squeeze(&self, state: &[[u8; M]; OUT]) -> [[u8; M]; OUT] {
        *state
    }
}

impl<F, Inner, const OUT: usize, const M: usize>
    StatefulHasher<[F; M], [[u32; M]; OUT], [[u32; M]; OUT]> for ChainingHasher<Inner>
where
    F: Field,
    Inner: CryptographicHasher<[u32; M], [[u32; M]; OUT]>,
{
    fn default_state(&self) -> [[u32; M]; OUT] {
        [[0u32; M]; OUT]
    }

    fn absorb_into<I>(&self, state: &mut [[u32; M]; OUT], input: I)
    where
        I: IntoIterator<Item = [F; M]>,
    {
        let prev = *state;
        *state = self
            .inner
            .hash_iter(chain(prev, F::into_parallel_u32_streams(input)));
    }

    fn squeeze(&self, state: &[[u32; M]; OUT]) -> [[u32; M]; OUT] {
        *state
    }
}

impl<F, Inner, const OUT: usize, const M: usize>
    StatefulHasher<[F; M], [[u64; M]; OUT], [[u64; M]; OUT]> for ChainingHasher<Inner>
where
    F: Field,
    Inner: CryptographicHasher<[u64; M], [[u64; M]; OUT]>,
{
    fn default_state(&self) -> [[u64; M]; OUT] {
        [[0u64; M]; OUT]
    }

    fn absorb_into<I>(&self, state: &mut [[u64; M]; OUT], input: I)
    where
        I: IntoIterator<Item = [F; M]>,
    {
        let prev = *state;
        *state = self
            .inner
            .hash_iter(chain(prev, F::into_parallel_u64_streams(input)));
    }

    fn squeeze(&self, state: &[[u64; M]; OUT]) -> [[u64; M]; OUT] {
        *state
    }
}

#[cfg(test)]
mod tests {
    use core::array;

    use p3_bn254::Bn254;
    use p3_field::{Field, RawDataSerializable};
    use p3_goldilocks::Goldilocks;
    use p3_koala_bear::KoalaBear;
    use p3_mersenne_31::Mersenne31;

    use super::*;
    use crate::testing::{MockHasher, MockPermutation};

    // =========================================================================
    // StatefulSponge tests
    // =========================================================================

    #[test]
    fn stateful_sponge_basic() {
        const WIDTH: usize = 4;
        const RATE: usize = 2;
        const OUT: usize = 2;

        let sponge = StatefulSponge::<MockPermutation, WIDTH, RATE, OUT>::new(MockPermutation);

        let input = [1u64, 2, 3, 4, 5];
        let mut state = sponge.default_state();
        sponge.absorb_into(&mut state, input);
        let output: [u64; OUT] = sponge.squeeze(&state);

        // StatefulSponge pads partial blocks to rate boundary (proper sponge semantics):
        // Initial state: [0, 0, 0, 0]
        // First input chunk [1, 2] overwrites first two positions: [1, 2, 0, 0]
        // Apply permutation (sum all elements and overwrite): [3, 3, 3, 3]
        // Second input chunk [3, 4] overwrites first two positions: [3, 4, 3, 3]
        // Apply permutation: [13, 13, 13, 13] (3 + 4 + 3 + 3 = 13)
        // Third input chunk [5] overwrites first position: [5, 13, 13, 13]
        // Pad remaining rate positions: [5, 0, 13, 13]
        // Apply permutation: [31, 31, 31, 31] (5 + 0 + 13 + 13 = 31)
        assert_eq!(output, [31; OUT]);
    }

    #[test]
    fn stateful_sponge_empty_input() {
        const WIDTH: usize = 4;
        const RATE: usize = 2;
        const OUT: usize = 2;

        let sponge = StatefulSponge::<MockPermutation, WIDTH, RATE, OUT>::new(MockPermutation);

        let mut state = sponge.default_state();
        sponge.absorb_into(&mut state, []);
        let output: [u64; OUT] = sponge.squeeze(&state);

        assert_eq!(
            output, [0; OUT],
            "Should return default values when input is empty."
        );
    }

    #[test]
    fn stateful_sponge_exact_block_size() {
        const WIDTH: usize = 6;
        const RATE: usize = 3;
        const OUT: usize = 2;

        let sponge = StatefulSponge::<MockPermutation, WIDTH, RATE, OUT>::new(MockPermutation);

        let input = [10u64, 20, 30];
        let mut state = sponge.default_state();
        sponge.absorb_into(&mut state, input);
        let output: [u64; OUT] = sponge.squeeze(&state);

        let expected_sum = 10 + 20 + 30;
        assert_eq!(output, [expected_sum; OUT]);
    }

    // =========================================================================
    // SerializingStatefulSponge tests
    // =========================================================================

    fn test_serializing_stateful_basic_generic<F: Field>() {
        // StatefulSponge with WIDTH=4, RATE=2, OUT=2
        let inner = StatefulSponge::<MockPermutation, 4, 2, 2>::new(MockPermutation);
        let hasher: SerializingStatefulSponge<_> = SerializingStatefulSponge::new(inner);

        let inputs: [F; 5] = array::from_fn(|i| F::from_usize(i));

        let mut state = StatefulHasher::<F, _, _>::default_state(&hasher);
        StatefulHasher::<F, _, _>::absorb_into(&hasher, &mut state, inputs);
        let output: [u64; 2] = StatefulHasher::<F, _, _>::squeeze(&hasher, &state);

        // The mock permutation sums all elements.
        // Field values serialized to u64s and absorbed into sponge.
        // Exact output depends on serialization and absorption pattern.
        // Just verify it runs without panicking and produces non-zero output.
        assert!(output.iter().any(|&x| x != 0), "Output should be non-zero");
    }

    fn test_serializing_stateful_multiple_absorptions_generic<F: Field>() {
        // Verify that multiple absorptions work correctly
        let inner = StatefulSponge::<MockPermutation, 4, 2, 2>::new(MockPermutation);
        let hasher: SerializingStatefulSponge<_> = SerializingStatefulSponge::new(inner);

        let inputs: [F; 10] = array::from_fn(|i| F::from_usize(i));

        // Split into two absorptions
        let mut state = StatefulHasher::<F, _, _>::default_state(&hasher);
        StatefulHasher::<F, _, _>::absorb_into(&hasher, &mut state, inputs[..5].iter().copied());
        StatefulHasher::<F, _, _>::absorb_into(&hasher, &mut state, inputs[5..].iter().copied());
        let output: [u64; 2] = StatefulHasher::<F, _, _>::squeeze(&hasher, &state);

        // Verify multiple absorptions work and produce non-zero output
        assert!(
            output.iter().any(|&x| x != 0),
            "Output should be non-zero after multiple absorptions"
        );
    }

    #[test]
    fn serializing_stateful_sponge() {
        test_serializing_stateful_basic_generic::<Mersenne31>();
        test_serializing_stateful_basic_generic::<KoalaBear>();
        test_serializing_stateful_basic_generic::<Goldilocks>();
        test_serializing_stateful_basic_generic::<Bn254>();

        test_serializing_stateful_multiple_absorptions_generic::<Mersenne31>();
        test_serializing_stateful_multiple_absorptions_generic::<KoalaBear>();
        test_serializing_stateful_multiple_absorptions_generic::<Goldilocks>();
        test_serializing_stateful_multiple_absorptions_generic::<Bn254>();
    }

    // =========================================================================
    // ChainingHasher tests
    // =========================================================================

    fn test_chaining_scalar_matches_manual_generic<F: Field + RawDataSerializable>() {
        let hasher = ChainingHasher::new(MockHasher);
        let inputs: [F; 17] = array::from_fn(|i| F::from_usize(i * 7 + 3));
        let segments: &[core::ops::Range<usize>] = &[0..3, 3..5, 5..9, 9..17];

        // Test u8 path
        let mut state_adapter_u8: [u8; 4] = StatefulHasher::<F, _, _>::default_state(&hasher);
        for seg in segments {
            hasher.absorb_into(&mut state_adapter_u8, inputs[seg.clone()].iter().copied());
        }

        let mut state_manual_u8 = [0u8; 4];
        for seg in segments {
            let prefix = state_manual_u8.into_iter();
            let bytes = F::into_byte_stream(inputs[seg.clone()].iter().copied());
            state_manual_u8 = MockHasher.hash_iter(prefix.chain(bytes));
        }

        assert_eq!(state_adapter_u8, state_manual_u8, "u8 path mismatch");

        // Test u64 path
        let mut state_adapter_u64: [u64; 4] = StatefulHasher::<F, _, _>::default_state(&hasher);
        for seg in segments {
            hasher.absorb_into(&mut state_adapter_u64, inputs[seg.clone()].iter().copied());
        }

        let mut state_manual_u64 = [0u64; 4];
        for seg in segments {
            let prefix = state_manual_u64.into_iter();
            let words = F::into_u64_stream(inputs[seg.clone()].iter().copied());
            state_manual_u64 = MockHasher.hash_iter(prefix.chain(words));
        }

        assert_eq!(state_adapter_u64, state_manual_u64, "u64 path mismatch");
    }

    #[test]
    fn chaining_scalar_matches_manual() {
        test_chaining_scalar_matches_manual_generic::<Mersenne31>();
        test_chaining_scalar_matches_manual_generic::<KoalaBear>();
        test_chaining_scalar_matches_manual_generic::<Goldilocks>();
        test_chaining_scalar_matches_manual_generic::<Bn254>();
    }

    /// Generic test that ChainingHasher parallel matches per-lane scalar for all binary types.
    ///
    /// Tests u8, u32, and u64 serialization paths.
    fn test_chaining_parallel_matches_scalar_generic<F: Field>() {
        let hasher = ChainingHasher::new(MockHasher);

        // Create 64 field elements with distinct values
        let input: [F; 64] = array::from_fn(|i| F::from_usize(i * 7 + 3));

        // Reshape to [[F; 4]; 16] for parallel processing
        let parallel_input: [[F; 4]; 16] = array::from_fn(|i| array::from_fn(|j| input[i * 4 + j]));
        let unzipped_input: [[F; 16]; 4] = array::from_fn(|i| parallel_input.map(|x| x[i]));

        // Test u8 path
        let mut state_parallel_u8: [[u8; 4]; 4] =
            StatefulHasher::<[F; 4], _, _>::default_state(&hasher);
        hasher.absorb_into(&mut state_parallel_u8, parallel_input);

        let per_lane_u8: [[u8; 4]; 4] = array::from_fn(|lane| {
            let mut s: [u8; 4] = StatefulHasher::<F, _, _>::default_state(&hasher);
            hasher.absorb_into(&mut s, unzipped_input[lane]);
            s
        });
        let per_lane_u8_transposed: [[u8; 4]; 4] = array::from_fn(|i| per_lane_u8.map(|x| x[i]));
        assert_eq!(
            state_parallel_u8, per_lane_u8_transposed,
            "u8 path mismatch"
        );

        // Test u32 path
        let mut state_parallel_u32: [[u32; 4]; 4] =
            StatefulHasher::<[F; 4], _, _>::default_state(&hasher);
        hasher.absorb_into(&mut state_parallel_u32, parallel_input);

        let per_lane_u32: [[u32; 4]; 4] = array::from_fn(|lane| {
            let mut s: [u32; 4] = StatefulHasher::<F, _, _>::default_state(&hasher);
            hasher.absorb_into(&mut s, unzipped_input[lane]);
            s
        });
        let per_lane_u32_transposed: [[u32; 4]; 4] = array::from_fn(|i| per_lane_u32.map(|x| x[i]));
        assert_eq!(
            state_parallel_u32, per_lane_u32_transposed,
            "u32 path mismatch"
        );

        // Test u64 path
        let mut state_parallel_u64: [[u64; 4]; 4] =
            StatefulHasher::<[F; 4], _, _>::default_state(&hasher);
        hasher.absorb_into(&mut state_parallel_u64, parallel_input);

        let per_lane_u64: [[u64; 4]; 4] = array::from_fn(|lane| {
            let mut s: [u64; 4] = StatefulHasher::<F, _, _>::default_state(&hasher);
            hasher.absorb_into(&mut s, unzipped_input[lane]);
            s
        });
        let per_lane_u64_transposed: [[u64; 4]; 4] = array::from_fn(|i| per_lane_u64.map(|x| x[i]));
        assert_eq!(
            state_parallel_u64, per_lane_u64_transposed,
            "u64 path mismatch"
        );
    }

    #[test]
    fn chaining_parallel_matches_scala() {
        test_chaining_parallel_matches_scalar_generic::<Mersenne31>();
        test_chaining_parallel_matches_scalar_generic::<KoalaBear>();
        test_chaining_parallel_matches_scalar_generic::<Goldilocks>();
        test_chaining_parallel_matches_scalar_generic::<Bn254>();
    }
}
