//! Criterion benches for RPO, xHash, and Poseidon1 permutations.
//!
//! Small-prime variants use width 24, Goldilocks width 12. The exception is
//! Poseidon1-Mersenne31, which has no width-24 MDS and so runs at width 16.

use core::array;

use criterion::{criterion_group, criterion_main, Criterion};
use p3_baby_bear::{
    BabyBear, MdsMatrixBabyBear, Poseidon1BabyBear, BABYBEAR_POSEIDON1_HALF_FULL_ROUNDS,
    BABYBEAR_POSEIDON1_PARTIAL_ROUNDS_24,
};
use p3_goldilocks::poseidon1::default_goldilocks_poseidon1_12;
use p3_goldilocks::Goldilocks;
use p3_koala_bear::{
    KoalaBear, MdsMatrixKoalaBear, Poseidon1KoalaBear, KOALABEAR_POSEIDON_HALF_FULL_ROUNDS,
    KOALABEAR_POSEIDON_PARTIAL_ROUNDS_24,
};
use p3_mersenne_31::{default_mersenne31_poseidon1_16, Mersenne31};
use p3_rpo_xhash::rpo::babybear::rpo_babybear;
use p3_rpo_xhash::rpo::goldilocks::rpo_goldilocks;
use p3_rpo_xhash::rpo::koalabear::rpo_koalabear;
use p3_rpo_xhash::rpo::m31::{rpo_m31_bb_mds, rpo_m31_cir};
use p3_rpo_xhash::xhash::babybear::xhash_babybear;
use p3_rpo_xhash::xhash::goldilocks::xhash_goldilocks;
use p3_rpo_xhash::xhash::koalabear::xhash_koalabear;
use p3_rpo_xhash::xhash::m31::{xhash_m31_bb_mds, xhash_m31_cir};
use p3_symmetric::Permutation;
use rand::rngs::SmallRng;
use rand::SeedableRng;

fn bench_rpo(c: &mut Criterion) {
    let mut g = c.benchmark_group("rpo/permute24");

    let mut rng = SmallRng::seed_from_u64(1);
    let hash = rpo_babybear(&mut rng);
    let input: [BabyBear; 24] = array::from_fn(|i| BabyBear::new((i as u32).wrapping_add(1)));
    g.bench_function("babybear", |b| b.iter(|| hash.permute(input)));

    let mut rng = SmallRng::seed_from_u64(1);
    let hash = rpo_koalabear(&mut rng);
    let input: [KoalaBear; 24] = array::from_fn(|i| KoalaBear::new((i as u32).wrapping_add(1)));
    g.bench_function("koalabear", |b| b.iter(|| hash.permute(input)));

    let mut rng = SmallRng::seed_from_u64(1);
    let hash = rpo_m31_cir(&mut rng);
    let input: [Mersenne31; 24] = array::from_fn(|i| Mersenne31::new((i as u32).wrapping_add(1)));
    g.bench_function("m31_cir", |b| b.iter(|| hash.permute(input)));

    let mut rng = SmallRng::seed_from_u64(1);
    let hash = rpo_m31_bb_mds(&mut rng);
    let input: [Mersenne31; 24] = array::from_fn(|i| Mersenne31::new((i as u32).wrapping_add(1)));
    g.bench_function("m31_bb_mds", |b| b.iter(|| hash.permute(input)));

    g.finish();
}

fn bench_rpo_goldilocks(c: &mut Criterion) {
    let mut g = c.benchmark_group("rpo/permute12");
    let input: [Goldilocks; 12] =
        array::from_fn(|i| Goldilocks::new((i as u64 + 1) * 1_000_000_007));

    let mut rng = SmallRng::seed_from_u64(3);
    let hash = rpo_goldilocks(&mut rng);
    g.bench_function("goldilocks", |b| b.iter(|| hash.permute(input)));

    g.finish();
}

fn bench_xhash(c: &mut Criterion) {
    let mut g = c.benchmark_group("xhash/permute24");

    let mut rng = SmallRng::seed_from_u64(2);
    let hash = xhash_babybear(&mut rng);
    let input: [BabyBear; 24] = array::from_fn(|i| BabyBear::new((i as u32).wrapping_add(1)));
    g.bench_function("babybear", |b| b.iter(|| hash.permute(input)));

    let mut rng = SmallRng::seed_from_u64(2);
    let hash = xhash_koalabear(&mut rng);
    let input: [KoalaBear; 24] = array::from_fn(|i| KoalaBear::new((i as u32).wrapping_add(1)));
    g.bench_function("koalabear", |b| b.iter(|| hash.permute(input)));

    let mut rng = SmallRng::seed_from_u64(2);
    let hash = xhash_m31_cir(&mut rng);
    let input: [Mersenne31; 24] = array::from_fn(|i| Mersenne31::new((i as u32).wrapping_add(1)));
    g.bench_function("m31_cir", |b| b.iter(|| hash.permute(input)));

    let mut rng = SmallRng::seed_from_u64(2);
    let hash = xhash_m31_bb_mds(&mut rng);
    let input: [Mersenne31; 24] = array::from_fn(|i| Mersenne31::new((i as u32).wrapping_add(1)));
    g.bench_function("m31_bb_mds", |b| b.iter(|| hash.permute(input)));

    g.finish();
}

fn bench_xhash_goldilocks(c: &mut Criterion) {
    let mut g = c.benchmark_group("xhash/permute12");
    let input: [Goldilocks; 12] =
        array::from_fn(|i| Goldilocks::new((i as u64 + 1) * 1_000_000_007));

    let mut rng = SmallRng::seed_from_u64(4);
    let hash = xhash_goldilocks(&mut rng);
    g.bench_function("goldilocks", |b| b.iter(|| hash.permute(input)));

    g.finish();
}

fn bench_poseidon1(c: &mut Criterion) {
    let mut g = c.benchmark_group("poseidon1/permute");

    let mut rng = SmallRng::seed_from_u64(5);

    let mds_bb: MdsMatrixBabyBear = Default::default();
    let p1_bb = Poseidon1BabyBear::<24>::new_from_rng(
        BABYBEAR_POSEIDON1_HALF_FULL_ROUNDS,
        BABYBEAR_POSEIDON1_PARTIAL_ROUNDS_24,
        &mds_bb,
        &mut rng,
    );
    let input: [BabyBear; 24] = array::from_fn(|i| BabyBear::new((i as u32).wrapping_add(1)));
    g.bench_function("babybear_w24", |b| b.iter(|| p1_bb.permute(input)));

    let mds_kb: MdsMatrixKoalaBear = Default::default();
    let p1_kb = Poseidon1KoalaBear::<24>::new_from_rng(
        KOALABEAR_POSEIDON_HALF_FULL_ROUNDS,
        KOALABEAR_POSEIDON_PARTIAL_ROUNDS_24,
        &mds_kb,
        &mut rng,
    );
    let input: [KoalaBear; 24] = array::from_fn(|i| KoalaBear::new((i as u32).wrapping_add(1)));
    g.bench_function("koalabear_w24", |b| b.iter(|| p1_kb.permute(input)));

    // Mersenne31 has no width-24 MDS; 16 is the tuned set.
    let p1_m31 = default_mersenne31_poseidon1_16();
    let input: [Mersenne31; 16] = array::from_fn(|i| Mersenne31::new((i as u32).wrapping_add(1)));
    g.bench_function("m31_w16", |b| b.iter(|| p1_m31.permute(input)));

    let p1_gl = default_goldilocks_poseidon1_12();
    let input: [Goldilocks; 12] =
        array::from_fn(|i| Goldilocks::new((i as u64 + 1) * 1_000_000_007));
    g.bench_function("goldilocks_w12", |b| b.iter(|| p1_gl.permute(input)));

    g.finish();
}

criterion_group!(
    benches,
    bench_rpo,
    bench_rpo_goldilocks,
    bench_xhash,
    bench_xhash_goldilocks,
    bench_poseidon1
);
criterion_main!(benches);
