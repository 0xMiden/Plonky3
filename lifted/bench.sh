RUSTFLAGS="-Ctarget-cpu=native" cargo bench -p p3-lifted --bench deep_quotient  --features parallel -- --save-baseline deep-parallel --measurement-time 30
RUSTFLAGS="-Ctarget-cpu=native" cargo bench -p p3-lifted --bench deep_quotient -- --save-baseline deep-serial --measurement-time 30

RUSTFLAGS="-Ctarget-cpu=native" cargo bench -p p3-lifted --bench deep_quotient  --features parallel -- --baseline deep-parallel --measurement-time 30
RUSTFLAGS="-Ctarget-cpu=native" cargo bench -p p3-lifted --bench deep_quotient -- --baseline deep-serial --measurement-time 30

RUSTFLAGS="-Ctarget-cpu=native" cargo bench -p p3-lifted --bench deep_quotient  --features parallel -- --save-baseline deep-combined-parallel --measurement-time 30
RUSTFLAGS="-Ctarget-cpu=native" cargo bench -p p3-lifted --bench deep_quotient -- --save-baseline deep-combined-serial --measurement-time 30

RUSTFLAGS="-Ctarget-cpu=native" cargo bench -p p3-lifted --bench deep_quotient  --features parallel -- --baseline deep-combined-parallel --measurement-time 30
RUSTFLAGS="-Ctarget-cpu=native" cargo bench -p p3-lifted --bench deep_quotient -- --baseline deep-combined-serial --measurement-time 30