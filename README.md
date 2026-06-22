# ligesis

A Rust workspace for **code-based multilinear polynomial commitment schemes (PCS)** over
[Goldilocks](https://docs.rs/p3-goldilocks) and a **faithful zk-PIOP for integer GPT-2 inference**
built on top of them.

- **`pcs`** — multilinear PCS over Goldilocks: a standalone **Basefold** (FRI-style); **LigeSIS** —
  a Ligero-style RS-code PCS whose column digests are committed with a subset-sum (binary-SIS) hash,
  following the single-machine **Protocol 1** of [`LigeSIS_SP27.pdf`](LigeSIS_SP27.pdf); and a
  small **`PolyCommitmentScheme` interface** with a transparent placeholder backend.
- **`logup`** — a GKR-LogUp lookup (log-derivative argument), the network-wide table lookup.
- **`proofsys`** — proves one integer GPT-2 inference as a **commit → reduce → batch-open** PIOP
  over the PCS interface (see [proofsys](#proofsys--faithful-gpt-2-piop) below).

> Research/experimental code. The Fiat–Shamir transcript is a precomputed randomness oracle
> (`RandomOracle`), not a hash-absorbing transcript; soundness parameters are set for structure,
> not tuned to proven bounds. Single-machine only (the distributed LigeSIS protocol is not
> implemented).

## Layout

```
utils/                shared primitives
  src/poly.rs         MlPoly<F> — multilinear poly as hypercube evals (eval / fold / new_eq)
  src/sumcheck.rs     product-of-multilinears sumcheck (prove / verify)
  src/merkle.rs       Blake3 Merkle tree (rs_merkle) + field serialization
  src/oracle.rs       RandomOracle<F> — precomputed Fiat–Shamir stand-in
pcs/
  src/scheme.rs       PolyCommitmentScheme trait + transparent PlaceholderPcs
  src/basefold.rs     Basefold PCS: commit / prove / verify + shared-domain batched open
  src/subset_sum.rs   SubsetSumHash — H = A·B, the binary-SIS hash (C = 32 digest rows)
  src/ligesis.rs      LigeSIS PCS: setup / commit / prove / verify
  benches/            single-threaded wall-clock benches (basefold, ligesis, sumcheck)
logup/
  src/lookup.rs       GKR-LogUp lookup: two fraction-sum passes + cross-multiply
proofsys/             faithful PIOP for integer GPT-2 inference (commit → reduce → batch-open)
  src/model.rs        integer GPT-2 forward pass — the witness generator
  src/canonical.rs    one merged commitment per witness type (+ offline weights)
  src/commit.rs       canonical layout, OracleSource point-maps, claim accumulator, CommitSet
  src/reduce.rs       matmul (Thaler13) + eq-product sumcheck reductions → opening claims
  src/faithful.rs     unified lookup + all gadget reductions; prove / verify
  examples/bench_faithful.rs   end-to-end forward → commit → prove → verify benchmark
```

## Build, test, bench

All code is **single-threaded** by design (keep the `p3` `parallel` feature off; run tests/benches
single-threaded).

```sh
cargo build
cargo test  -p pcs -- --test-threads=1      # PCS round-trip + tamper tests
cargo bench -p pcs --bench basefold         # Basefold commit/eval/verify + proof size
cargo bench -p pcs --bench ligesis          # LigeSIS commit/eval/verify + proof size

cargo test  -p proofsys                     # PIOP reductions + accept/tamper tests
cargo run --release -p proofsys --example bench_faithful            # synthetic config
cargo run --release -p proofsys --example bench_faithful -- ../int_gpt/export   # real GPT-2
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

## proofsys — faithful GPT-2 PIOP

`proofsys` proves one integer GPT-2 inference (the `int_gpt` fixed-point model) as a **faithful**
PIOP over the PCS interface: the prover commits the witness **once**, every sumcheck and a single
network-wide LogUp lookup reduce to **opening claims** against those commitments (committing nothing
more), and all claims are discharged by one **PCS batch-open** at the end. The committed witness set
is the single source of truth; every other polynomial — matmul products, division remainders,
reshapes/transposes, the residual stream — is *virtual*, reconstructed at the queried point from
opened values + public data (no extra commitments).

The PCS is pluggable through `pcs::PolyCommitmentScheme`. The current backend is the transparent
**`PlaceholderPcs`** (commit = ship the polynomial, open = the verifier re-evaluates it, empty
proof), so the whole architecture is testable end-to-end; swapping in `Basefold::batch_prove` is a
localized change since every gadget already emits `(commitment, point, value)` claims.

**Reductions** (batched across the 12 layers / 144 heads via per-instance segments):
- **matmul** — Thaler13 `C̃(z) = Σ_k Ã·B̃`, one sumcheck over the contraction, fused with the
  following rescale; the product `C` is never committed. Operands resolve to canonical openings via
  affine / head-slice / weight point-maps (all 7 GPT-2 matmuls, regular and transposed).
- **division** — Euclidean `a = q·b + r`: commit only `q`, range-check `r` and `b−1−r`. Constant
  divisors (`//SCALE`, `//(√D·SCALE)`) hit a range table directly; the witness divisor (`//std`)
  and the ≈2³⁰ LayerNorm remainder are **limb-decomposed** (16-bit limbs vs a shared 2¹⁶ table, plus
  a recomposition check). Every committed quotient also carries a limb quotient bound.
- **sqrt** — `y²−y ≤ x ≤ y²+y` via an eq-weighted product (`y²`) and limb brackets.
- **gelu** — indexed LUT (`act = GELU_LUT[fc + offset]`).
- **unified lookup** — every range / LUT / limb query folds into **one** LogUp lookup against a
  single `(in, out, type)` table; only the multiplicity vector `e` is committed online, the table
  side is public and verifier-reconstructed. The table is sized to the config.

`model::forward` is the bit-exact integer reference that generates the witness. The pipeline
(forward → commit → prove → verify) verifies on small and multi-layer/-head configs;
`bench_faithful` reports prover/verifier time, transcript size, and Fiat-Shamir draw counts.

> Status: the reductions above are implemented and accept/tamper-tested. Not yet wired (the
> remaining soundness bindings): the softmax-max grand-product, the exp masked-index LUT, the
> sum-over-features half-points / `var_sum` binding, and the residual-stream wiring. Full-scale
> GPT-2 (12 layers, vocab 50257) is being brought up; small and multi-instance configs pass today.

## PCS benchmarks (indicative)

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
