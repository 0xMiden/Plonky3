use std::vec::Vec;

use p3_baby_bear::BabyBear as F;
use p3_field::{PrimeCharacteristicRing, RawDataSerializable};
use p3_keccak::{Keccak256Hash, KeccakF};
use p3_symmetric::{
    ChainingHasher, CryptographicHasher, PaddingFreeSponge, SerializingHasher, StatefulHasher,
};

#[test]
fn chaining_keccak_u8_matches_manual() {
    type H = Keccak256Hash;
    const N: usize = 32; // Keccak-256 output size in bytes
    let h = ChainingHasher::new(H {});

    // Deterministic BabyBear inputs
    let inputs: Vec<F> = (0..37).map(|i| F::from_u32(i as u32)).collect();

    // Partition the inputs into segments, including empty ones
    let segments: &[&[F]] = &[
        &inputs[..0],    // empty
        &inputs[0..1],   // single element
        &inputs[1..5],   // small chunk
        &inputs[5..5],   // empty
        &inputs[5..17],  // larger chunk
        &inputs[17..17], // empty
        &inputs[17..37], // remainder
    ];

    // Adapter stateful path
    let mut state_adapter = [0u8; N];
    for seg in segments {
        h.absorb_into(&mut state_adapter, seg.iter().copied());
    }

    // Manual chaining: state <- H(state || encode(seg))
    let mut state_manual = [0u8; N];
    for seg in segments {
        let prefix = state_manual.into_iter();
        let bytes = F::into_byte_stream(seg.iter().copied());
        state_manual = H {}.hash_iter(prefix.chain(bytes));
    }
    let out_manual = state_manual;

    assert_eq!(state_adapter, out_manual);
}

#[test]
fn chaining_keccak_u64_matches_manual() {
    // Build a Keccak-based sponge hasher over u64s: WIDTH=25, RATE=17, OUT=4
    type H = PaddingFreeSponge<KeccakF, 25, 17, 4>;
    const N: usize = 4;
    let h = ChainingHasher::new(H::new(KeccakF {}));

    // Deterministic BabyBear inputs
    let inputs: Vec<F> = (0..41).map(|i| F::from_u32((i * 3 + 1) as u32)).collect();

    // Partition the inputs into segments, include empties
    let segments: &[&[F]] = &[
        &inputs[..0],
        &inputs[0..2],
        &inputs[2..2],
        &inputs[2..9],
        &inputs[9..24],
        &inputs[24..24],
        &inputs[24..41],
    ];

    let mut state_adapter = [0u64; N];
    for seg in segments {
        h.absorb_into(&mut state_adapter, seg.iter().copied());
    }

    // Manual chaining using the same hasher
    let mut state_manual = [0u64; N];
    let h2 = H::new(KeccakF {});
    for seg in segments {
        let prefix = state_manual.into_iter();
        let words = F::into_u64_stream(seg.iter().copied());
        state_manual = h2.hash_iter(prefix.chain(words));
    }
    let out_manual = state_manual;

    assert_eq!(state_adapter, out_manual);
}

#[test]
fn serializing_hasher_matches_inner_u8_stream() {
    type ByteHash = Keccak256Hash;
    type FieldHash = SerializingHasher<ByteHash>;
    const N: usize = 32;
    let inputs: Vec<F> = (0..17).map(|i| F::from_u32((i * 9 + 4) as u32)).collect();

    let bytes: Vec<u8> = F::into_byte_stream(inputs.clone()).into_iter().collect();
    let inner = ByteHash {};
    let expected: [u8; N] = inner.hash_iter(bytes);

    let field = FieldHash::new(inner);
    let actual: [u8; N] = field.hash_iter(inputs);

    assert_eq!(actual, expected);
}

#[test]
fn serializing_hasher_matches_inner_u64_stream() {
    type U64Hash = PaddingFreeSponge<KeccakF, 25, 17, 4>;
    type FieldHash = SerializingHasher<U64Hash>;
    const N: usize = 4;
    let inputs: Vec<F> = (0..19).map(|i| F::from_u32((i * 5 + 7) as u32)).collect();

    let words: Vec<u64> = F::into_u64_stream(inputs.clone()).into_iter().collect();
    let inner = U64Hash::new(KeccakF {});
    let expected: [u64; N] = inner.hash_iter(words);

    let field = FieldHash::new(inner);
    let actual: [u64; N] = field.hash_iter(inputs);

    assert_eq!(actual, expected);
}
