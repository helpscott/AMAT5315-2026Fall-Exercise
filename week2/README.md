# Week 2: Agentic coding with Rust

## Part 1: Rust ABC

The `md` crate contains a library and an executable. The executable and the
unit test both call `greeting()` from `md/src/lib.rs`.

From the repository root:

```bash
cd week2/md
cargo run
cargo test --lib
```

Expected output: `Hello, world!` and `1 passed; 0 failed`.

- `md/Cargo.toml`: package metadata and dependencies.
- `md/src/lib.rs`: the shared greeting function and its unit test.
- `md/src/main.rs`: the command-line entry point, which prints the greeting.
- `md/Cargo.lock`: the locked dependency graph, committed for reproducibility.

This first stage uses only the Rust standard library. Build output under
`md/target/` is excluded from Git. The molecular-dynamics implementation will
extend this same crate in later stages.

After verifying Part 1, return to `week2/` with `cd ..`. Later Cargo commands
run there with `--manifest-path md/Cargo.toml`.
