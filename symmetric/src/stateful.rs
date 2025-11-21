use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_field::{Field, PackedValue};

// No need for permutation types here; implemented in sponge.rs where needed.

/// Trait for stateful sponge-like hashers.
///
/// A stateful hasher maintains an external state value that evolves as input is
/// absorbed, and from which fixed-size outputs can be squeezed. This interface
/// is used pervasively by commitment schemes and Merkle trees to incrementally
/// absorb rows of matrices and later read out the final digest.
///
/// Padding semantics:
/// - Each implementation exposes a constant `PADDING_WIDTH` that counts how many
///   input items constitute one horizontal “padding unit”. Callers may treat each
///   input slice as implicitly padded with zeros to a multiple of `PADDING_WIDTH`.
/// - Importantly, `PADDING_WIDTH` is measured in units of `Item`, not bytes.
///   For example, a field-to-bytes chaining adapter has `PADDING_WIDTH = 1`
///   (one more field element extends the input by one item), while a field-native
///   sponge with rate `R` has `PADDING_WIDTH = R` (in field elements).
///
/// Implementations provided in this crate:
/// - [`StatefulSponge`] over an algebraic permutation (field-native). Its
///   `PADDING_WIDTH` is the sponge rate, expressed in `Item` units.
/// - [`ChainedStateful`] adapters over u8/u32/u64 words (and their parallel
///   array forms). These adapt a stateless hasher `H` to the stateful interface
///   with the rule `state <- H(state || encode(input))` for non-empty inputs.
///   Their `PADDING_WIDTH` is always 1 (one more item extends the input stream).
///
/// # Examples
///
/// Using Keccak-256 (bytes) as a stateful adapter over the BabyBear field:
/// ```ignore
/// use p3_baby_bear::BabyBear;
/// use p3_keccak::Keccak256Hash;
/// use p3_symmetric::{ChainedStateful, StatefulHasher};
///
/// // State is the digest itself: 32 bytes
/// let keccak = ChainedStateful::<Keccak256Hash, u8, 32>::new(Keccak256Hash {});
/// let mut state = [0u8; 32];
///
/// // Absorb multiple segments of field elements
/// keccak.absorb_into(&mut state, [BabyBear::from_u32(1), BabyBear::from_u32(2)]);
/// keccak.absorb_into(&mut state, [BabyBear::from_u32(3)]);
///
/// // Read the digest (first 32 bytes of the running state)
/// let digest = keccak.squeeze(&state);
/// assert_eq!(digest.len(), 32);
/// ```
///
/// Using a field-native Poseidon2 sponge over BabyBear (rate-based padding):
/// ```ignore
/// use rand::{SeedableRng, rngs::SmallRng};
/// use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
/// use p3_symmetric::{StatefulHasher, PaddingFreeSponge};
/// const WIDTH: usize = 16; const RATE: usize = 8; const DIGEST: usize = 8;
/// let mut rng = SmallRng::seed_from_u64(1);
/// let perm = Poseidon2BabyBear::<WIDTH>::new_from_rng_128(&mut rng);
/// let sponge = PaddingFreeSponge::<_, WIDTH, RATE, DIGEST>::new(perm);
/// let mut state = [BabyBear::ZERO; WIDTH];
/// sponge.absorb_into(&mut state, [BabyBear::from_u32(7); 5]);
/// let digest = sponge.squeeze(&state);
/// assert_eq!(digest.len(), DIGEST);
/// ```
pub trait StatefulHasher<Item, State, Out>
where
    Item: Clone,
{
    /// The horizontal padding width for absorption, in elements.
    /// Default is 1.
    const PADDING_WIDTH: usize = 1;
    /// Absorb elements into the state with overwrite-mode and zero-padding
    /// semantics if applicable to the implementation.
    fn absorb_into<I>(&self, state: &mut State, input: I)
    where
        I: IntoIterator<Item = Item>;

    /// Squeeze an output from the current state.
    fn squeeze(&self, state: &State) -> Out;
}

/// A generic chaining adapter turning a stateless hasher `H` over words `I`
/// into a stateful interface whose state is the digest itself: `[I; N]`.
///
/// Absorb rule: `state <- H(state || encode(input))`. Empty input is a no-op.
/// Stateful chaining adapter that turns a stateless hasher `H` over words `I`
/// into a stateful interface whose state is the digest itself: `[I; N]`.
///
/// Absorb rule: `state <- H(state || encode(input))`. Empty input is a no-op.
/// `PADDING_WIDTH` is always 1, since each absorbed item contributes directly to
/// the serialized stream.
#[derive(Clone, Debug)]
pub struct ChainedStateful<H, I, const N: usize> {
    pub h: H,
    _phantom: PhantomData<I>,
}

impl<H, I, const N: usize> ChainedStateful<H, I, N> {
    pub const fn new(h: H) -> Self {
        Self {
            h,
            _phantom: PhantomData,
        }
    }
}

// Field -> bytes variant.
impl<F, H, const N: usize> StatefulHasher<F, [u8; N], [u8; N]> for ChainedStateful<H, u8, N>
where
    F: Field,
    H: crate::CryptographicHasher<u8, [u8; N]>,
{
    const PADDING_WIDTH: usize = 1;

    fn absorb_into<I>(&self, state: &mut [u8; N], input: I)
    where
        I: IntoIterator<Item = F>,
    {
        let mut bytes_iter = F::into_byte_stream(input).into_iter().peekable();
        if bytes_iter.peek().is_none() {
            return;
        }
        let chained = state.iter().copied().chain(bytes_iter);
        *state = self.h.hash_iter(chained);
    }

    fn squeeze(&self, state: &[u8; N]) -> [u8; N] {
        *state
    }
}

// Field -> u64s variant.
impl<F, H, const N: usize> StatefulHasher<F, [u64; N], [u64; N]> for ChainedStateful<H, u64, N>
where
    F: Field,
    H: crate::CryptographicHasher<u64, [u64; N]>,
{
    const PADDING_WIDTH: usize = 1;

    fn absorb_into<I>(&self, state: &mut [u64; N], input: I)
    where
        I: IntoIterator<Item = F>,
    {
        let mut words_iter = F::into_u64_stream(input).into_iter().peekable();
        if words_iter.peek().is_none() {
            return;
        }
        let chained = state.iter().copied().chain(words_iter);
        *state = self.h.hash_iter(chained);
    }

    fn squeeze(&self, state: &[u64; N]) -> [u64; N] {
        *state
    }
}

// Field -> u32s variant (scalar)
impl<F, H, const N: usize> StatefulHasher<F, [u32; N], [u32; N]> for ChainedStateful<H, u32, N>
where
    F: Field,
    H: crate::CryptographicHasher<u32, [u32; N]>,
{
    const PADDING_WIDTH: usize = 1;

    fn absorb_into<I>(&self, state: &mut [u32; N], input: I)
    where
        I: IntoIterator<Item = F>,
    {
        let mut words_iter = F::into_u32_stream(input).into_iter().peekable();
        if words_iter.peek().is_none() {
            return;
        }
        let chained = state.iter().copied().chain(words_iter);
        *state = self.h.hash_iter(chained);
    }

    fn squeeze(&self, state: &[u32; N]) -> [u32; N] {
        *state
    }
}

// Packed Item variant: PF as Item with per-lane chaining to bytes
impl<F, PF, H, const N: usize, const M: usize> StatefulHasher<PF, [[u8; M]; N], [[u8; M]; N]>
    for ChainedStateful<H, u8, N>
where
    F: Field + Clone,
    PF: PackedValue<Value = F>,
    H: crate::CryptographicHasher<u8, [u8; N]>,
{
    const PADDING_WIDTH: usize = 1;

    fn absorb_into<I>(&self, state: &mut [[u8; M]; N], input: I)
    where
        I: IntoIterator<Item = PF>,
    {
        let mut lanes: Vec<Vec<F>> = (0..M).map(|_| Vec::new()).collect();
        for packed in input {
            let s = packed.as_slice();
            debug_assert_eq!(s.len(), M);
            for j in 0..M {
                lanes[j].push(s[j]);
            }
        }
        if lanes.iter().all(|v| v.is_empty()) {
            return;
        }
        for j in 0..M {
            let lane_bytes = F::into_byte_stream(lanes[j].clone()).into_iter();
            let mut lane_state = [0u8; N];
            for i in 0..N {
                lane_state[i] = state[i][j];
            }
            let new_lane = self.h.hash_iter(lane_state.into_iter().chain(lane_bytes));
            for i in 0..N {
                state[i][j] = new_lane[i];
            }
        }
    }

    fn squeeze(&self, state: &[[u8; M]; N]) -> [[u8; M]; N] {
        *state
    }
}

// Packed Item variant: PF as Item with per-lane chaining to u32
impl<F, PF, H, const N: usize, const M: usize> StatefulHasher<PF, [[u32; M]; N], [[u32; M]; N]>
    for ChainedStateful<H, u32, N>
where
    F: Field + Clone,
    PF: PackedValue<Value = F>,
    H: crate::CryptographicHasher<u32, [u32; N]>,
{
    const PADDING_WIDTH: usize = 1;

    fn absorb_into<I>(&self, state: &mut [[u32; M]; N], input: I)
    where
        I: IntoIterator<Item = PF>,
    {
        let mut lanes: Vec<Vec<F>> = (0..M).map(|_| Vec::new()).collect();
        for packed in input {
            let s = packed.as_slice();
            debug_assert_eq!(s.len(), M);
            for j in 0..M {
                lanes[j].push(s[j]);
            }
        }
        if lanes.iter().all(|v| v.is_empty()) {
            return;
        }
        for j in 0..M {
            let lane_words = F::into_u32_stream(lanes[j].clone()).into_iter();
            let mut lane_state = [0u32; N];
            for i in 0..N {
                lane_state[i] = state[i][j];
            }
            let new_lane = self.h.hash_iter(lane_state.into_iter().chain(lane_words));
            for i in 0..N {
                state[i][j] = new_lane[i];
            }
        }
    }

    fn squeeze(&self, state: &[[u32; M]; N]) -> [[u32; M]; N] {
        *state
    }
}

// Packed Item variant: PF as Item with per-lane chaining to u64
impl<F, PF, H, const N: usize, const M: usize> StatefulHasher<PF, [[u64; M]; N], [[u64; M]; N]>
    for ChainedStateful<H, u64, N>
where
    F: Field + Clone,
    PF: PackedValue<Value = F>,
    H: crate::CryptographicHasher<u64, [u64; N]>,
{
    const PADDING_WIDTH: usize = 1;

    fn absorb_into<I>(&self, state: &mut [[u64; M]; N], input: I)
    where
        I: IntoIterator<Item = PF>,
    {
        let mut lanes: Vec<Vec<F>> = (0..M).map(|_| Vec::new()).collect();
        for packed in input {
            let s = packed.as_slice();
            debug_assert_eq!(s.len(), M);
            for j in 0..M {
                lanes[j].push(s[j]);
            }
        }
        if lanes.iter().all(|v| v.is_empty()) {
            return;
        }
        for j in 0..M {
            let lane_words = F::into_u64_stream(lanes[j].clone()).into_iter();
            let mut lane_state = [0u64; N];
            for i in 0..N {
                lane_state[i] = state[i][j];
            }
            let new_lane = self.h.hash_iter(lane_state.into_iter().chain(lane_words));
            for i in 0..N {
                state[i][j] = new_lane[i];
            }
        }
    }

    fn squeeze(&self, state: &[[u64; M]; N]) -> [[u64; M]; N] {
        *state
    }
}
