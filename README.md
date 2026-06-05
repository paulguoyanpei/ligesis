# ligesis

A Rust implementation of **code-based multilinear polynomial commitment schemes (PCS)** over
[Goldilocks](https://docs.rs/p3-goldilocks): a standalone **Basefold** PCS, and **LigeSIS** — a
Ligero-style RS-code PCS whose column digests are committed with a subset-sum (binary-SIS) hash.
LigeSIS follows the single-machine **Protocol 1** of the paper [`LigeSIS_SP27.pdf`](LigeSIS_SP27.pdf).

> Research/experimental code. The Fiat–Shamir transcript is a precomputed randomness oracle
> (`RandomOracle`), not a hash-absorbing transcript; soundness parameters are set for structure,
> not tuned to the paper's proven bounds. Single-machine only (the distributed protocol is not
> implemented).

## Layout

```
utils/                shared primitives
  src/poly.rs         MlPoly<F> — multilinear poly as hypercube evals (eval / fold / new_eq)
  src/sumcheck.rs     product-of-multilinears sumcheck (prove / verify)
  src/merkle.rs       Blake3 Merkle tree (rs_merkle) + field serialization
  src/oracle.rs       RandomOracle<F> — precomputed Fiat–Shamir stand-in
pcs/
  src/basefold.rs     Basefold PCS: commit / prove / verify + shared-domain batched open
  src/subset_sum.rs   SubsetSumHash — H = A·B, the binary-SIS hash (C = 32 digest rows)
  src/ligesis.rs      LigeSIS PCS: setup / commit / prove / verify
  benches/            single-threaded wall-clock benches (basefold, ligesis, sumcheck)
```

## Build, test, bench

All code is **single-threaded** by design (keep the `p3` `parallel` feature off; run tests/benches
single-threaded).

```sh
cargo build
cargo test  -p pcs -- --test-threads=1      # round-trip + tamper tests
cargo bench -p pcs --bench basefold         # Basefold commit/eval/verify + proof size
cargo bench -p pcs --bench ligesis          # LigeSIS commit/eval/verify + proof size
```

## Parameters

| | |
|---|---|
| base field | `Goldilocks` (p = 2⁶⁴ − 2³² + 1) |
| opening / challenge field | `BinomialExtensionField<Goldilocks, 2>` (λ ≈ 100 over `F_{p²}`) |
| secondary-PCS rate | ρ = 1/2 (`code_rate = 1`, 100 FRI queries) |
| bit width `η` | 64 (bit decomposition of each field element) |
| SIS digest rows `C` | 32 |
| opened columns `s` | 128 |

## LigeSIS in one page

A multilinear `f` (`2^μ` evals) is arranged into an `m × n` matrix `F`.

**Commit**
1. RS-encode each row at ρ=1/2 → `F' ∈ F^{m×2n}`.
2. Bit-decompose → `B ∈ {0,1}^{ηm×2n}`.
3. Subset-sum hash the columns → **`H = A·B ∈ F^{C×2n}`** (`A` public).
4. Commit `H` **column-wise with a Merkle tree**. The root is the commitment to `f`.

**Eval** — prove `f̄(z) = y`, with `z = (z2, z1)` (`z1` = rows, `z2` = columns):
- Send `ā = ēq_{z1}^T F` (so `y = ā(z2)`); sample a random column set `I` (|I| = `s`); open the
  queried digest columns `H[:,I]` directly from the Merkle tree; commit the queried bit-columns
  `B_I`.
- Prove three checks over `B_I`, each reduced to multilinear evaluations by sumcheck:
  1. **binary** — `B_I ∈ {0,1}`;
  2. **hash consistency** — `A·B_I = H[:,I]` (binds opened columns to the commitment);
  3. **reconstruction** — the bit-columns rebuild `RS(ā)` at `I` (binds them to the claim `y`).
- Collapse all auxiliary openings (`Ā`, `B_I`, `ā`) into **one shared-domain Basefold batched
  open**.

**Verify** replays the transcript, Merkle-checks `H[:,I]`, checks every sumcheck, and batch-verifies
the openings.

Soundness is the usual Ligero argument: the Merkle root + SIS collision-resistance bind `f`;
hash + reconstruction tie the opened columns to both the commitment and the claim; a random `I`
plus the binary check give proximity.

See [`pcs/src/ligesis.rs`](pcs/src/ligesis.rs) (module docs) for the full protocol and conventions.

### Notable deviations from the paper
- **Bit** decomposition (binary `B`), matching the protocol's `B·(B−1)=0` check and the
  subset-sum (binary) hash definition.
- The secondary PCS is the in-repo **Basefold** (ρ=1/2), not Deepfold; the multiple openings are
  served by one **shared-RS-domain batched FRI**.
- **`H` is committed by a plain Merkle tree** and its queried columns opened directly, replacing the
  paper's "commit `H` with a secondary PCS + tiny-lookup" (`H` is `O(λ)` to reveal on a single
  machine, so the lookup isn't needed). This also removes `H` from the FRI batch, which lifts the
  shape restriction below.

## Benchmarks (indicative)

Single-threaded; numbers vary with machine load (commit is dominated by the un-accelerated
subset-sum hash and swings run-to-run). `Basefold` opens the full `2^μ` polynomial; `LigeSIS` rows
list `2^{log_m} × 2^{log_n}`.

| scheme / shape | μ | commit | eval | verify | proof |
|---|---|---|---|---|---|
| Basefold | 22 | ~1 s | ~0.85 s | ~7 ms | 391 KB |
| LigeSIS `2⁷×2¹⁵` | 22 | ~35 s | ~2.5 s | ~0.1 s | 455 KB |
| Basefold | 24 | ~5 s | ~3.5 s | ~9 ms | 527 KB |
| LigeSIS `2⁷×2¹⁷` | 24 | ~140 s | ~2.9 s | ~0.5 s | 502 KB |

For skinny matrices LigeSIS eval/proof are competitive with (sometimes better than) Basefold,
because its FRI runs over the small auxiliary polys (`B_I`, `Ā`, `ā`) rather than the full `f`.
Commit is far higher than Basefold's: the subset-sum hash is `Θ(2^μ · η · C)` field additions and
is not yet preprocessing-accelerated. LigeSIS's real advantage is distribution-friendliness, not
single-node speed.

## Limitations / out of scope
- Distributed LigeSIS (Protocol 5) and load balancing.
- Subset-sum hash preprocessing acceleration (the paper's ~8× speedup) and `Z_{2⁶⁴}` overflow
  handling — `A` is sampled uniformly and hashing is plain field addition.
- The batched open requires the committed polys `{Ā, B_I, ā}` to be within `LOG_INTERLEAVE = 6`
  variables of each other (so they share one RS domain), i.e. roughly `7 + log_m ≤ log_n ≤ 17 +
  log_m`. Shapes outside this window are rejected at `setup`.
- `RandomOracle` is a precomputed randomness source, not a real Fiat–Shamir transcript.
