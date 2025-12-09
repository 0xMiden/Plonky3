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
///   For example, a field-to-bytes adapter has `PADDING_WIDTH = 1`
///   (one more field element extends the input by one item), while a field-native
///   sponge with rate `R` has `PADDING_WIDTH = R` (in field elements).
pub trait StatefulHasher<Item, State, Out>: Clone {
    /// The horizontal padding width for absorption, expressed in `Item` units.
    /// Default is 1.
    const PADDING_WIDTH: usize = 1;

    /// Absorb elements into the state with overwrite-mode and zero-padding semantics if applicable.
    fn absorb_into<I>(&self, state: &mut State, input: I)
    where
        I: IntoIterator<Item = Item>;

    /// Squeeze an output from the current state.
    fn squeeze(&self, state: &State) -> Out;
}
