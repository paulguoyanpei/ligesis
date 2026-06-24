# `bench_faithful` progression

Single-threaded, release. Tracks the cost of wiring the four unfinished soundness bindings
(plan: `~/.claude/plans/optimized-squishing-stardust.md`). One row per milestone; deltas are
vs the previous row. Times are wall-clock ms (machine-load sensitive — the prover/commit
numbers swing run-to-run; transcript size and FS draw count are deterministic).

Run with:
```sh
cargo run --release -p proofsys --example bench_faithful                  # synthetic_2layer
cargo run --release -p proofsys --example bench_faithful -- ../int_gpt/export   # gpt2_31
```

## synthetic_2layer  (n_layer=2 n_head=1 d_model=2 n_seq=2 vocab=4)

| milestone | prove_ms | verify_ms | transcript_bytes | FS_field_draws | num_segments | verify_ok |
|---|---|---|---|---|---|---|
| baseline | 54.6 | 0.28 | 17680 | 321 | 31 | ✓ |
| Phase 1 (residual) | 49.8 | 0.28 | 17872 | 323 | 31 | ✓ |
| Phase 2 (var_sum/sum_exp) | 54.3 | 0.31 | 19600 | 345 | 31 | ✓ |
| Phase 3 (softmax-max) | 58.1 | 0.33 | 19920 | 352 | 31 | ✓ |
| Phase 4 (exp-LUT) | 54.9 | 0.35 | 20176 | 355 | 31 | ✓ |

## gpt2_31  (n_layer=12 n_head=12 d_model=768 n_seq=31 vocab=50257)

| milestone | prove_ms | verify_ms | transcript_bytes | FS_field_draws | num_segments | verify_ok |
|---|---|---|---|---|---|---|
| baseline | 19595.7 | 442.8 | 177952 | 2999 | 639 | ✓ |
| Phase 1 (residual) | 20618.2 | 449.7 | 179104 | 3014 | 639 | ✓ |
| Phase 2 (var_sum/sum_exp) | 22589.1 | 610.7 | 185584 | 3100 | 639 | ✓ |
| Phase 3 (softmax-max) | 22703.4 | 599.8 | 191088 | 3203 | 639 | ✓ |
| Phase 4 (exp-LUT) | 22536.1 | 616.6 | 193472 | 3244 | 639 | ✓ |

### baseline transcript breakdown (gpt2_31)
- lookup_gkr 35648 (20.0%)
- segment_matmul 116880 (65.7%)
- typeb_batched_prod 3504 (2.0%)
- segment_scalar_openings 17152 (9.6%)
- limb_recomp 4720 (2.7%)
- table_col_openings 48 (0.0%)

### notes
- Baseline = the four bindings (softmax-max grand-product, exp masked-index LUT,
  sum-over-features/var_sum, residual-stream) NOT yet wired. Phases 1–4 wire them in turn.
- All four now active (Phase 4 row). Net cost vs baseline on gpt2_31: prove ≈ flat (noise),
  verify +174ms (442→617), transcript +15.5KB (178→193KB), FS draws +245 (2999→3244). The
  bindings are dominated by the existing matmul sumchecks; the new reductions are cheap because
  each is batched (one grand-product GKR / one masked sumcheck / γ-RLC'd LayerNorm sumchecks
  across all instances).
- Possible follow-up optimization: batch the 3 per-LayerNorm-type sumchecks in Phase 2 are
  already γ-RLC'd across the 25 instances; the residual identity (Phase 1b) is a single affine
  batch. No obvious super-linear blowup remains.
