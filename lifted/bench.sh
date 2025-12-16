#!/usr/bin/env bash
# Benchmark commands for the lifted crate.
# This file is NOT meant to be executed directly - copy/paste individual commands.
#
# Default features: bench-goldilocks, bench-poseidon2
#
# Feature flags (mutually exclusive within each group):
#   Field:  bench-babybear | bench-goldilocks (default)
#   Hash:   bench-poseidon2 (default) | bench-keccak
#
# Note: deep_quotient, lmcs_vs_mmcs, and pcs require bench-poseidon2

# =============================================================================
# DEFAULT: RUN ALL (Goldilocks + Poseidon2 + Parallel)
# =============================================================================

RUSTFLAGS="-Ctarget-cpu=native" cargo bench -p p3-lifted --features parallel

# =============================================================================
# MERKLE COMMIT BENCHMARKS
# =============================================================================
# Benchmarks: scalar, packed, ext_mmcs/arity2, ext_mmcs/arity4

# Goldilocks + Poseidon2 + Parallel (default)
RUSTFLAGS="-Ctarget-cpu=native" cargo bench -p p3-lifted --bench merkle_commit \
    --features parallel

# BabyBear + Poseidon2 + Parallel
RUSTFLAGS="-Ctarget-cpu=native" cargo bench -p p3-lifted --bench merkle_commit \
    --no-default-features --features bench-babybear,bench-poseidon2,parallel

# BabyBear + Keccak
RUSTFLAGS="-Ctarget-cpu=native" cargo bench -p p3-lifted --bench merkle_commit \
    --no-default-features --features bench-babybear,bench-keccak

# =============================================================================
# LMCS vs MMCS COMPARISON (Poseidon2 only)
# =============================================================================
# Benchmarks: lmcs, mmcs

# Goldilocks + Parallel (default)
RUSTFLAGS="-Ctarget-cpu=native" cargo bench -p p3-lifted --bench lmcs_vs_mmcs \
    --features parallel

# BabyBear + Parallel
RUSTFLAGS="-Ctarget-cpu=native" cargo bench -p p3-lifted --bench lmcs_vs_mmcs \
    --no-default-features --features bench-babybear,bench-poseidon2,parallel

# =============================================================================
# FRI FOLD BENCHMARKS
# =============================================================================
# Benchmarks: lifted/2/scalar, lifted/2/packed, lifted/4/scalar, lifted/4/packed

# Goldilocks + Poseidon2 + Parallel (default)
RUSTFLAGS="-Ctarget-cpu=native" cargo bench -p p3-lifted --bench fri_fold \
    --features parallel

# BabyBear + Poseidon2 + Parallel
RUSTFLAGS="-Ctarget-cpu=native" cargo bench -p p3-lifted --bench fri_fold \
    --no-default-features --features bench-babybear,bench-poseidon2,parallel

# BabyBear + Keccak
RUSTFLAGS="-Ctarget-cpu=native" cargo bench -p p3-lifted --bench fri_fold \
    --no-default-features --features bench-babybear,bench-keccak

# =============================================================================
# PCS OPEN BENCHMARKS (Poseidon2 only)
# =============================================================================
# Benchmarks: workspace, lifted/arity2, lifted/arity4

# Goldilocks + Parallel (default)
RUSTFLAGS="-Ctarget-cpu=native" cargo bench -p p3-lifted --bench pcs \
    --features parallel

# BabyBear + Parallel
RUSTFLAGS="-Ctarget-cpu=native" cargo bench -p p3-lifted --bench pcs \
    --no-default-features --features bench-babybear,bench-poseidon2,parallel

# =============================================================================
# DEEP QUOTIENT BENCHMARKS (Poseidon2 only)
# =============================================================================
# Benchmarks: batch_eval, combined

# Goldilocks + Parallel (default)
RUSTFLAGS="-Ctarget-cpu=native" cargo bench -p p3-lifted --bench deep_quotient \
    --features parallel

# BabyBear + Parallel
RUSTFLAGS="-Ctarget-cpu=native" cargo bench -p p3-lifted --bench deep_quotient \
    --no-default-features --features bench-babybear,bench-poseidon2,parallel

# =============================================================================
# BASELINE COMPARISONS
# =============================================================================
# Save baseline, then compare after changes.
# Note: Must specify --bench to avoid passing criterion args to lib tests.

# Save baseline (run each benchmark individually)
for bench in merkle_commit fri_fold deep_quotient lmcs_vs_mmcs pcs; do
    RUSTFLAGS="-Ctarget-cpu=native" cargo bench -p p3-lifted --bench $bench \
        --features parallel -- --save-baseline main
done

# Compare against baseline
for bench in merkle_commit fri_fold deep_quotient lmcs_vs_mmcs pcs; do
    RUSTFLAGS="-Ctarget-cpu=native" cargo bench -p p3-lifted --bench $bench \
        --features parallel -- --baseline main
done
