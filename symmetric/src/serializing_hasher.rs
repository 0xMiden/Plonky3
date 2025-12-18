use p3_field::Field;

use crate::CryptographicHasher;

/// Converts a hasher which can hash bytes, u32's or u64's into a hasher which can hash field elements.
///
/// Supports two types of hashing.
/// - Hashing a sequence of field elements.
/// - Hashing a sequence of arrays of `N` field elements as if we are hashing `N` sequences of field elements in parallel.
///   This is useful when the inner hash is able to use vectorized instructions to compute multiple hashes at once.
#[derive(Copy, Clone, Debug)]
pub struct SerializingHasher<Inner> {
    inner: Inner,
}

impl<Inner> SerializingHasher<Inner> {
    pub const fn new(inner: Inner) -> Self {
        Self { inner }
    }
}

impl<F, Inner, const N: usize> CryptographicHasher<F, [u8; N]> for SerializingHasher<Inner>
where
    F: Field,
    Inner: CryptographicHasher<u8, [u8; N]>,
{
    fn hash_iter<I>(&self, input: I) -> [u8; N]
    where
        I: IntoIterator<Item = F>,
    {
        self.inner.hash_iter(F::into_byte_stream(input))
    }
}

impl<F, Inner, const N: usize> CryptographicHasher<F, [u32; N]> for SerializingHasher<Inner>
where
    F: Field,
    Inner: CryptographicHasher<u32, [u32; N]>,
{
    fn hash_iter<I>(&self, input: I) -> [u32; N]
    where
        I: IntoIterator<Item = F>,
    {
        self.inner.hash_iter(F::into_u32_stream(input))
    }
}

impl<F, Inner, const N: usize> CryptographicHasher<F, [u64; N]> for SerializingHasher<Inner>
where
    F: Field,
    Inner: CryptographicHasher<u64, [u64; N]>,
{
    fn hash_iter<I>(&self, input: I) -> [u64; N]
    where
        I: IntoIterator<Item = F>,
    {
        self.inner.hash_iter(F::into_u64_stream(input))
    }
}

impl<F, Inner, const N: usize, const M: usize> CryptographicHasher<[F; M], [[u8; M]; N]>
    for SerializingHasher<Inner>
where
    F: Field,
    Inner: CryptographicHasher<[u8; M], [[u8; M]; N]>,
{
    fn hash_iter<I>(&self, input: I) -> [[u8; M]; N]
    where
        I: IntoIterator<Item = [F; M]>,
    {
        self.inner.hash_iter(F::into_parallel_byte_streams(input))
    }
}

impl<F, Inner, const N: usize, const M: usize> CryptographicHasher<[F; M], [[u32; M]; N]>
    for SerializingHasher<Inner>
where
    F: Field,
    Inner: CryptographicHasher<[u32; M], [[u32; M]; N]>,
{
    fn hash_iter<I>(&self, input: I) -> [[u32; M]; N]
    where
        I: IntoIterator<Item = [F; M]>,
    {
        self.inner.hash_iter(F::into_parallel_u32_streams(input))
    }
}

impl<F, Inner, const N: usize, const M: usize> CryptographicHasher<[F; M], [[u64; M]; N]>
    for SerializingHasher<Inner>
where
    F: Field,
    Inner: CryptographicHasher<[u64; M], [[u64; M]; N]>,
{
    fn hash_iter<I>(&self, input: I) -> [[u64; M]; N]
    where
        I: IntoIterator<Item = [F; M]>,
    {
        self.inner.hash_iter(F::into_parallel_u64_streams(input))
    }
}

#[cfg(test)]
mod tests {
    use core::array;

    use p3_bn254::Bn254;
    use p3_goldilocks::Goldilocks;
    use p3_koala_bear::KoalaBear;
    use p3_mersenne_31::Mersenne31;

    use super::*;
    use crate::testing::MockHasher;

    /// Generic test that parallel hashing matches scalar hashing for all binary types.
    ///
    /// Tests u8, u32, and u64 serialization paths.
    fn test_parallel_matches_scalar_generic<F: Field>() {
        let hasher = SerializingHasher::new(MockHasher);

        // Create 64 field elements with distinct values
        let input: [F; 64] = array::from_fn(|i| F::from_usize(i * 7 + 3));

        // Reshape to [[F; 4]; 16] for parallel processing
        let parallel_input: [[F; 4]; 16] = array::from_fn(|i| array::from_fn(|j| input[i * 4 + j]));
        let unzipped_input: [[F; 16]; 4] = array::from_fn(|i| parallel_input.map(|x| x[i]));

        // Test u8 path
        let u8_parallel: [[u8; 4]; 4] = hasher.hash_iter(parallel_input);
        let u8_scalar: [[u8; 4]; 4] = unzipped_input.map(|lane| hasher.hash_iter(lane));
        let u8_scalar_transposed: [[u8; 4]; 4] = array::from_fn(|i| u8_scalar.map(|x| x[i]));
        assert_eq!(u8_parallel, u8_scalar_transposed, "u8 path mismatch");

        // Test u32 path
        let u32_parallel: [[u32; 4]; 4] = hasher.hash_iter(parallel_input);
        let u32_scalar: [[u32; 4]; 4] = unzipped_input.map(|lane| hasher.hash_iter(lane));
        let u32_scalar_transposed: [[u32; 4]; 4] = array::from_fn(|i| u32_scalar.map(|x| x[i]));
        assert_eq!(u32_parallel, u32_scalar_transposed, "u32 path mismatch");

        // Test u64 path
        let u64_parallel: [[u64; 4]; 4] = hasher.hash_iter(parallel_input);
        let u64_scalar: [[u64; 4]; 4] = unzipped_input.map(|lane| hasher.hash_iter(lane));
        let u64_scalar_transposed: [[u64; 4]; 4] = array::from_fn(|i| u64_scalar.map(|x| x[i]));
        assert_eq!(u64_parallel, u64_scalar_transposed, "u64 path mismatch");
    }

    #[test]
    fn parallel_matches_scalar() {
        test_parallel_matches_scalar_generic::<KoalaBear>();
        test_parallel_matches_scalar_generic::<Goldilocks>();
        test_parallel_matches_scalar_generic::<Mersenne31>();
        test_parallel_matches_scalar_generic::<Bn254>();
    }
}
