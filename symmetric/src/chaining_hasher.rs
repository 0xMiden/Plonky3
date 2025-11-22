use core::iter::chain;

use p3_field::Field;

use crate::{CryptographicHasher, StatefulHasher};

/// An adapter that chains state with new input, hashing `state || encode(input)`.
///
/// This mirrors `SerializingHasher`'s conversions from fields to bytes/u32/u64 streams,
/// but implements the `StatefulHasher` interface where the state is the digest itself.
#[derive(Copy, Clone, Debug)]
pub struct ChainingHasher<Inner> {
    inner: Inner,
}

impl<Inner> ChainingHasher<Inner> {
    pub const fn new(inner: Inner) -> Self {
        Self { inner }
    }
}

// Scalar field -> byte digest
impl<F, Inner, const N: usize> StatefulHasher<F, [u8; N], [u8; N]> for ChainingHasher<Inner>
where
    F: Field,
    Inner: CryptographicHasher<u8, [u8; N]>,
{
    const PADDING_WIDTH: usize = 1;

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
    const PADDING_WIDTH: usize = 1;

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
    const PADDING_WIDTH: usize = 1;

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

// Parallel lanes (array-based) implemented via per-lane scalar hashing.
impl<F, Inner, const OUT: usize, const M: usize>
    StatefulHasher<[F; M], [[u8; M]; OUT], [[u8; M]; OUT]> for ChainingHasher<Inner>
where
    F: Field,
    Inner: CryptographicHasher<[u8; M], [[u8; M]; OUT]>,
{
    const PADDING_WIDTH: usize = 1;

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
    const PADDING_WIDTH: usize = 1;

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
    const PADDING_WIDTH: usize = 1;

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

// Remove generic PackedValue impls to avoid overlaps; array-based variants above suffice for benches/tests.

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use core::array;

    use p3_field::RawDataSerializable;
    use p3_koala_bear::KoalaBear;

    use crate::{ChainingHasher, CryptographicHasher, StatefulHasher};

    #[derive(Clone)]
    struct MockHasher;

    impl CryptographicHasher<u8, [u8; 4]> for MockHasher {
        fn hash_iter<I: IntoIterator<Item = u8>>(&self, iter: I) -> [u8; 4] {
            let sum: u8 = iter.into_iter().fold(0, |acc, x| acc.wrapping_add(x));
            [sum; 4]
        }
    }

    impl CryptographicHasher<[u8; 4], [[u8; 4]; 4]> for MockHasher {
        fn hash_iter<I: IntoIterator<Item = [u8; 4]>>(&self, iter: I) -> [[u8; 4]; 4] {
            let sum: [u8; 4] = iter.into_iter().fold([0, 0, 0, 0], |acc, x| {
                [
                    acc[0].wrapping_add(x[0]),
                    acc[1].wrapping_add(x[1]),
                    acc[2].wrapping_add(x[2]),
                    acc[3].wrapping_add(x[3]),
                ]
            });
            [sum; 4]
        }
    }

    impl CryptographicHasher<u32, [u32; 4]> for MockHasher {
        fn hash_iter<I: IntoIterator<Item = u32>>(&self, iter: I) -> [u32; 4] {
            let sum: u32 = iter.into_iter().fold(0, |acc, x| acc.wrapping_add(x));
            [sum; 4]
        }
    }

    impl CryptographicHasher<[u32; 4], [[u32; 4]; 4]> for MockHasher {
        fn hash_iter<I: IntoIterator<Item = [u32; 4]>>(&self, iter: I) -> [[u32; 4]; 4] {
            let sum: [u32; 4] = iter.into_iter().fold([0, 0, 0, 0], |acc, x| {
                [
                    acc[0].wrapping_add(x[0]),
                    acc[1].wrapping_add(x[1]),
                    acc[2].wrapping_add(x[2]),
                    acc[3].wrapping_add(x[3]),
                ]
            });
            [sum; 4]
        }
    }

    impl CryptographicHasher<u64, [u64; 4]> for MockHasher {
        fn hash_iter<I: IntoIterator<Item = u64>>(&self, iter: I) -> [u64; 4] {
            let sum: u64 = iter.into_iter().fold(0, |acc, x| acc.wrapping_add(x));
            [sum; 4]
        }
    }

    impl CryptographicHasher<[u64; 4], [[u64; 4]; 4]> for MockHasher {
        fn hash_iter<I: IntoIterator<Item = [u64; 4]>>(&self, iter: I) -> [[u64; 4]; 4] {
            let sum: [u64; 4] = iter.into_iter().fold([0, 0, 0, 0], |acc, x| {
                [
                    acc[0].wrapping_add(x[0]),
                    acc[1].wrapping_add(x[1]),
                    acc[2].wrapping_add(x[2]),
                    acc[3].wrapping_add(x[3]),
                ]
            });
            [sum; 4]
        }
    }

    #[test]
    fn chaining_scalar_u8_matches_manual() {
        let hasher = ChainingHasher::new(MockHasher {});
        let inputs: Vec<KoalaBear> = (0..17).map(KoalaBear::new).collect();
        let segments: &[core::ops::Range<usize>] = &[0..3, 3..5, 5..9, 9..17];

        let mut state_adapter = [0u8; 4];
        for seg in segments {
            hasher.absorb_into(&mut state_adapter, inputs[seg.clone()].iter().copied());
        }

        let mut state_manual = [0u8; 4];
        for seg in segments {
            let prefix = state_manual.into_iter();
            let bytes = KoalaBear::into_byte_stream(inputs[seg.clone()].iter().copied());
            state_manual = MockHasher {}.hash_iter(prefix.chain(bytes));
        }

        assert_eq!(state_adapter, state_manual);
    }

    #[test]
    fn chaining_scalar_u64_matches_manual() {
        let hasher = ChainingHasher::new(MockHasher {});
        let inputs: Vec<KoalaBear> = (0..19).map(|i| KoalaBear::new(i * 7 + 3)).collect();
        let segments: &[core::ops::Range<usize>] = &[0..0, 0..2, 2..9, 9..19];

        let mut state_adapter = [0u64; 4];
        for seg in segments {
            hasher.absorb_into(&mut state_adapter, inputs[seg.clone()].iter().copied());
        }

        let mut state_manual = [0u64; 4];
        for seg in segments {
            let prefix = state_manual.into_iter();
            let words = KoalaBear::into_u64_stream(inputs[seg.clone()].iter().copied());
            state_manual = MockHasher {}.hash_iter(prefix.chain(words));
        }

        assert_eq!(state_adapter, state_manual);
    }

    #[test]
    fn chaining_parallel_matches_per_lane_u8_u32_u64() {
        let mock_hash = MockHasher {};
        let hasher = ChainingHasher::new(mock_hash);
        let input: [KoalaBear; 256] = KoalaBear::new_array(array::from_fn(|x| x as u32));

        let parallel_input: [[KoalaBear; 4]; 64] = unsafe { core::mem::transmute(input) };
        let unzipped_input: [[KoalaBear; 64]; 4] = array::from_fn(|i| parallel_input.map(|x| x[i]));

        // u8 path
        let mut state_parallel_u8 = [[0u8; 4]; 4];
        hasher.absorb_into(&mut state_parallel_u8, parallel_input);
        let out_parallel_u8 = state_parallel_u8;

        let per_lane_u8: [[u8; 4]; 4] = array::from_fn(|lane| {
            let mut s = [0u8; 4];
            hasher.absorb_into(&mut s, unzipped_input[lane]);
            s
        });
        let per_lane_u8_transposed = array::from_fn(|i| per_lane_u8.map(|x| x[i]));
        assert_eq!(out_parallel_u8, per_lane_u8_transposed);

        // u32 path
        let mut state_parallel_u32 = [[0u32; 4]; 4];
        hasher.absorb_into(&mut state_parallel_u32, parallel_input);
        let out_parallel_u32 = state_parallel_u32;

        let per_lane_u32: [[u32; 4]; 4] = array::from_fn(|lane| {
            let mut s = [0u32; 4];
            hasher.absorb_into(&mut s, unzipped_input[lane]);
            s
        });
        let per_lane_u32_transposed = array::from_fn(|i| per_lane_u32.map(|x| x[i]));
        assert_eq!(out_parallel_u32, per_lane_u32_transposed);

        // u64 path
        let mut state_parallel_u64 = [[0u64; 4]; 4];
        hasher.absorb_into(&mut state_parallel_u64, parallel_input);
        let out_parallel_u64 = state_parallel_u64;

        let per_lane_u64: [[u64; 4]; 4] = array::from_fn(|lane| {
            let mut s = [0u64; 4];
            hasher.absorb_into(&mut s, unzipped_input[lane]);
            s
        });
        let per_lane_u64_transposed = array::from_fn(|i| per_lane_u64.map(|x| x[i]));
        assert_eq!(out_parallel_u64, per_lane_u64_transposed);
    }
}
