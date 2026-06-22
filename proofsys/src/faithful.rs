//! Faithful PIOP assembly: commit-once → reduce-to-claims → batch-open, over the PCS interface.
//!
//! The unified lookup is built from **segments**, each a contiguous block of query rows living in
//! the canonical layout of its source witness type. A segment's `in`/`out` columns are virtual
//! views of committed canonical polynomials; the lookup proves every folded row
//! `in + α·out + α²·type` lies in the public unified table, and the query opening at `zq` reduces
//! to canonical opening claims discharged by one `Pcs::batch_*`. Only the multiplicity vector `e`
//! is committed online; the table (ranges + LUTs) is public.
//!
//! Segment kinds:
//! - `Direct` — `in = canonical_in + bias + offset`, `out = 0` or a committed canonical column
//!   (range / quotient-bound / indexed LUT).
//! - `TypeA` — matmul+rescale: `in = r = C − divisor·q`, where `C` is the Thaler13 product (a
//!   per-segment sumcheck over the contraction, run after the lookup at the query point) and `q` is
//!   the canonical quotient. The matmul operands `A`/`B` reduce to canonical/weight openings via
//!   [`OracleSource`]s. (Single-instance for now; multi-layer batching is a follow-on.)

use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, PrimeField64};

use logup::lookup::{self, LookupProof, Poly};
use p3_goldilocks::Goldilocks;
use pcs::PolyCommitmentScheme;
use utils::oracle::RandomOracle;
use utils::poly::MlPoly;
use utils::sumcheck::{self, SumcheckProof};

use crate::canonical::{
    self, Canonical, ACT, A_LN, EXP, Q_FC, Q_FPR, Q_LN, Q_PROB, Q_QKV, STD, SUM_EXP, VAR,
};
use crate::commit::{
    eq_at_index, BiasPoly, ClaimAccumulator, CommitSet, Coord, OracleSource, Pcs, PointMap,
};
use crate::model::ModelWeights;
use crate::protocol::EF;
use crate::reduce::{self, MatmulKind, MatmulProof, ProdProof};
use crate::tensor::Matrix;
use crate::witness::{encode_i64, Config, Witness};

// Lookup column types (mirrors protocol.rs).
const TYPE_GELU: u64 = 2;
const TYPE_LIMB16: u64 = 3;
const TYPE_RANGE12: u64 = 4;
const TYPE_RANGE15: u64 = 5;
const TYPE_RANGE13: u64 = 6;
const TYPE_RANGE20: u64 = 7;
const TYPE_DUMMY: u64 = 99;

const RANGE12_SIZE: usize = 4096;
const RANGE13_SIZE: usize = 8192;
const RANGE15_SIZE: usize = 32768;
const LIMB16_SIZE: usize = 1 << 16;
const LIMB_BITS: usize = 16;
const NUM_LIMBS: usize = 2; // 2×16-bit limbs cover [0, 2^32) ⊃ the ≈2^30 LayerNorm remainders.
const R12_BASE: usize = 0;
const R13_BASE: usize = RANGE12_SIZE;
const R15_BASE: usize = R13_BASE + RANGE13_SIZE;
const R20_BASE: usize = R15_BASE + RANGE15_SIZE;

/// Size of the softmax-division remainder range table (`TYPE_RANGE20`). The remainder is
/// `< sum_exp ≤ n_seq·scale` (exp values are `≤ scale`), so this is config-sized rather than a
/// fixed `2²⁰` — keeping the unified table (and thus the lookup cost) proportional to the model.
fn range_b_size(config: &Config) -> usize {
    (config.n_seq * config.scale as usize).next_power_of_two()
}
fn limb16_base(config: &Config) -> usize {
    R20_BASE + range_b_size(config)
}

/// The committed multiplicity vector for the unified lookup.
pub const E: &str = "e";
/// The three committed (public) unified-table columns: `in`, `out`, `type`. Committed offline (the
/// table is data-independent); the lookup runs over the folded `in + α·out + α²·type`.
pub const TABLE_IN: &str = "table_in";
pub const TABLE_OUT: &str = "table_out";
pub const TABLE_TYPE: &str = "table_type";
/// The committed limb decomposition of the LayerNorm `// std` remainder (lower `r ≥ 0` check).
pub const LN_REM_LIMB: &str = "ln_rem_limb";
/// Limb decomposition of the LayerNorm upper bound `std − 1 − r` (asserts `r < std`).
pub const LN_REM_HI_LIMB: &str = "ln_rem_hi_limb";
/// The committed limb decomposition of the sqrt lower bracket `x − (y²−y)` (`x = var_sum + 1`).
pub const SQRT_LO_LIMB: &str = "sqrt_lo_limb";
/// Limb decomposition of the sqrt upper bracket `(y²+y) − x` (asserts `x ≤ y²+y`).
pub const SQRT_HI_LIMB: &str = "sqrt_hi_limb";
/// Offset added to a signed quotient before its non-negative limb decomposition.
const QUOTIENT_LIMB_OFFSET: u64 = 1 << 24;

fn ef_u64(x: u64) -> EF {
    EF::from(Goldilocks::new(x))
}
fn ef_i64(x: i64) -> EF {
    EF::from(encode_i64(x))
}

/// Extract the non-negative integer represented by a base-field-valued `EF`.
fn ef_to_u64(x: EF) -> u64 {
    let coeffs: &[Goldilocks] = x.as_basis_coefficients_slice();
    assert!(coeffs[1] == Goldilocks::ZERO, "non-base field element");
    coeffs[0].as_canonical_u64()
}

/// Folded table/query row `in + α·out + α²·type`.
fn fold_row(alpha: EF, input: EF, output: EF, ty: u64) -> EF {
    input + alpha * output + alpha * alpha * ef_u64(ty)
}

fn gelu_base(config: &Config) -> usize {
    limb16_base(config) + LIMB16_SIZE
}
fn table_len(config: &Config, weights: &ModelWeights) -> usize {
    (gelu_base(config) + weights.gelu_lut.len()).next_power_of_two()
}

/// The three unfolded public table columns `(in, out, type)` in row order. These are committed
/// offline (data-independent); the folded table the lookup runs over is `in + α·out + α²·type`.
fn table_columns(config: &Config, weights: &ModelWeights) -> (Vec<EF>, Vec<EF>, Vec<EF>) {
    type Cols = (Vec<EF>, Vec<EF>, Vec<EF>);
    let len = table_len(config, weights);
    let mut cols: Cols = (
        Vec::with_capacity(len),
        Vec::with_capacity(len),
        Vec::with_capacity(len),
    );
    let push = |c: &mut Cols, i: u64, o: EF, t: u64| {
        c.0.push(ef_u64(i));
        c.1.push(o);
        c.2.push(ef_u64(t));
    };
    for v in 0..RANGE12_SIZE as u64 {
        push(&mut cols, v, EF::ZERO, TYPE_RANGE12);
    }
    for v in 0..RANGE13_SIZE as u64 {
        push(&mut cols, v, EF::ZERO, TYPE_RANGE13);
    }
    for v in 0..RANGE15_SIZE as u64 {
        push(&mut cols, v, EF::ZERO, TYPE_RANGE15);
    }
    for v in 0..range_b_size(config) as u64 {
        push(&mut cols, v, EF::ZERO, TYPE_RANGE20);
    }
    for v in 0..LIMB16_SIZE as u64 {
        push(&mut cols, v, EF::ZERO, TYPE_LIMB16);
    }
    for (idx, &out) in weights.gelu_lut.iter().enumerate() {
        push(&mut cols, idx as u64, ef_i64(out), TYPE_GELU);
    }
    while cols.0.len() < len {
        let idx = cols.0.len() as u64;
        push(&mut cols, idx, EF::ZERO, TYPE_DUMMY);
    }
    cols
}

/// The folded public table `in + α·out + α²·type` (prover side, to run the lookup).
fn build_table(alpha: EF, config: &Config, weights: &ModelWeights) -> Vec<EF> {
    let (tin, tout, tty) = table_columns(config, weights);
    (0..tin.len())
        .map(|i| tin[i] + alpha * tout[i] + alpha * alpha * tty[i])
        .collect()
}

/// What a segment's `in`/`out` columns are.
enum SegKind {
    /// `in = canonical(oracle_in) + bias + offset`; `out = 0` or `out` canonical column.
    Direct {
        bias: Option<BiasPoly>,
        offset: u64,
        out: Option<String>,
    },
    /// `in = C − divisor·q`, with `q` opened via `q_src` (a fixed-instance slice of the merged
    /// quotient) and `C` the Thaler13 matmul product over `a_src`/`b_src`. One segment per matmul
    /// instance (`elem_vars`-sized).
    TypeA {
        divisor: u64,
        mk: MatmulKind,
        q_src: OracleSource,
        a_src: OracleSource,
        b_src: OracleSource,
        op: TypeAOp,
    },
    /// A prod-plus-affine direct range segment: `in = prod_coeff·(qL·qR) + Σ coeff·linear + const`,
    /// range-checked into the segment's table. `prod_left`/`prod_right` (`q_prob`, `sum_exp`) build
    /// the query column `in = prod_coeff·(qL·qR) + Σ coeff·linear + const`; the `qL·qR` product is
    /// reduced by the batched [`prove_typeb_batch`] (one sumcheck for all heads), and the linears
    /// are opened directly. Covers softmax Type-B division lower (`r ≥ 0`) and upper (`r < sum_exp`).
    /// One segment per head (`elem_vars`-sized).
    TypeB {
        prod_left: OracleSource,
        prod_right: OracleSource,
        prod_coeff: EF,
        linears: Vec<(OracleSource, EF)>,
        constant: EF,
        op: TypeBOp,
    },
}

/// Identifies a Type-B division instance (softmax, per head) for the prover.
#[derive(Clone, Copy)]
struct TypeBOp {
    layer: usize,
    head: usize,
}

/// Identifies which model matmul a TypeA segment proves, so the prover can fetch its `A`/`B`
/// matrices and quotient from the witness/weights.
#[derive(Clone, Copy)]
struct TypeAOp {
    mm: Mm,
    layer: usize,
    head: usize,
}

/// The matmuls whose rescale quotient is committed (single-instance/head for now).
#[derive(Clone, Copy, PartialEq)]
enum Mm {
    Qkv,
    Fc,
    Fpr,
    Apr,
    Log,
    Sc, // qs@ksᵀ → scores (per head)
    Ao, // probs@vs → attn out (per head)
}

struct MmDims {
    k: usize,
    row: usize,
    col: usize,
}

fn vbits(n: usize) -> usize {
    n.next_power_of_two().trailing_zeros() as usize
}

fn mm_dims(config: &Config, mm: Mm) -> MmDims {
    let s = config.n_seq;
    let d = config.d_model;
    match mm {
        Mm::Qkv => MmDims { k: d, row: s, col: 3 * d },
        Mm::Fc => MmDims { k: d, row: s, col: config.mlp_hidden },
        Mm::Fpr => MmDims { k: config.mlp_hidden, row: s, col: d },
        Mm::Apr => MmDims { k: d, row: s, col: d },
        Mm::Log => MmDims { k: d, row: s, col: config.vocab },
        Mm::Sc => MmDims { k: config.d_head, row: s, col: s },
        Mm::Ao => MmDims { k: s, row: s, col: config.d_head },
    }
}

fn mm_divisor(config: &Config, mm: Mm) -> u64 {
    match mm {
        Mm::Sc => config.sqrt_d_scale() as u64,
        _ => config.scale as u64,
    }
}

/// Table (type, base) for a constant-divisor rescale remainder in `[0, divisor)`.
fn const_div_range(divisor: u64) -> (u64, usize) {
    match divisor {
        0..=4096 => (TYPE_RANGE12, R12_BASE),
        4097..=32768 => (TYPE_RANGE15, R15_BASE),
        _ => panic!("unsupported constant divisor {divisor}"),
    }
}

fn mm_kind(mm: Mm) -> MatmulKind {
    match mm {
        Mm::Log | Mm::Sc => MatmulKind::Transposed,
        _ => MatmulKind::Regular,
    }
}

struct Descriptor {
    oracle_in: String,
    ty: u64,
    table_base: usize,
    vars: usize,
    start: usize,
    kind: SegKind,
}

impl Descriptor {
    fn len(&self) -> usize {
        1 << self.vars
    }
}

/// `fc_b` bias over the `q_fc` canonical layout (column low, row middle, layer high). The bias is
/// added only on the **real** rows `< n_seq`; padding rows are zeroed so that there (where the
/// committed `q_fc`/`act` are both 0) the gelu `in` is exactly `offset` and the table `out` is
/// `gelu_lut[offset] = gelu(0) = 0`, matching the committed `act`. (A row-broadcast bias would put
/// `fc_b + offset` on padding rows with `out = 0`, which is not a table row — see the unified
/// lookup.) Padding columns/instances are already 0 since `fc_b`/the blocks don't fill them.
fn fc_bias(config: &Config, weights: &ModelWeights) -> BiasPoly {
    let lay = canonical::layout_for(config, Q_FC);
    let (cv, rv, nv) = (lay.col_vars(), lay.row_vars(), lay.inst_vars());
    let col_pow = 1usize << cv;
    let row_pow = 1usize << rv;
    let mut values = vec![EF::ZERO; col_pow * row_pow * (1usize << nv)];
    for (layer, bw) in weights.blocks.iter().enumerate() {
        for row in 0..config.n_seq {
            for (col, &b) in bw.fc_b.iter().enumerate() {
                values[col + col_pow * row + col_pow * row_pow * layer] = ef_i64(b);
            }
        }
    }
    BiasPoly {
        values,
        vars: (0..cv + rv + nv).collect(),
    }
}

/// A matmul **A** operand bias on the `k ++ row` operand point (feature `b[k]` on the low feature
/// vars). The bias is applied only on the **real** rows `< n_seq`: the matmul contraction sums just
/// those rows, so `Σ_r eq(row_point,r)` is `< 1` when `n_seq` isn't a power of two; a row-broadcast
/// bias would add the missing padding rows' share and the emitted operand opening would not match
/// the committed (padding-zero) quotient. Zeroing padding rows makes the virtual operand poly equal
/// the contracted `avec` everywhere.
fn feature_bias(b: &[i64], k: usize, config: &Config) -> BiasPoly {
    let k_pow = k.next_power_of_two();
    let row_pow = config.n_seq.next_power_of_two();
    let mut values = vec![EF::ZERO; k_pow * row_pow];
    for row in 0..config.n_seq {
        for (i, &x) in b.iter().enumerate() {
            values[i + k_pow * row] = ef_i64(x);
        }
    }
    BiasPoly {
        values,
        vars: (0..vbits(k) + vbits(config.n_seq)).collect(),
    }
}

/// Fixed-instance slice source of a merged poly: maps an `elem_vars`-point to
/// `[elem ++ bits(inst)]` of the merged commitment (identity over the element variables).
fn slice_src(config: &Config, oracle: &str, inst: usize) -> OracleSource {
    let lay = canonical::layout_for(config, oracle);
    OracleSource::committed(
        oracle,
        PointMap::identity(lay.elem_vars()).with_const_suffix(inst, lay.inst_vars()),
    )
}

/// Quotient source for the attention-output matmul `Ao`: the per-head matmul output is `S × d_head`
/// but the committed `q_ao` is the per-layer `S × d_model` (heads concatenated on the column). Maps
/// an operand point `[d(d_head) ++ row(S)]` to `q_ao[layer]` at column `head·d_head + d`.
fn ao_out_slice_src(config: &Config, layer: usize, head: usize) -> OracleSource {
    let lay = canonical::layout_for(config, crate::canonical::Q_AO);
    let dh_vars = vbits(config.d_head);
    let row_vars = vbits(config.n_seq);
    let col_vars = lay.col_vars(); // d_model column bits
    let head_vars = col_vars - dh_vars; // high column bits select the head
    let mut coords = Vec::with_capacity(lay.num_vars());
    for i in 0..dh_vars {
        coords.push(Coord::Var(i)); // within-head column d
    }
    for b in 0..head_vars {
        coords.push(Coord::Const((head >> b) & 1 == 1));
    }
    for i in 0..row_vars {
        coords.push(Coord::Var(dh_vars + i)); // sequence row
    }
    for b in 0..lay.inst_vars() {
        coords.push(Coord::Const((layer >> b) & 1 == 1));
    }
    OracleSource::committed(crate::canonical::Q_AO, PointMap(coords))
}

/// Matmul `A` source: canonical `a_oracle` at instance `inst`, identity over `k ++ row`, + bias.
fn a_source(
    config: &Config,
    a_oracle: &str,
    inst: usize,
    k: usize,
    row: usize,
    bias: Option<BiasPoly>,
) -> OracleSource {
    let inst_vars = canonical::layout_for(config, a_oracle).inst_vars();
    let map = PointMap::identity(vbits(k) + vbits(row)).with_const_suffix(inst, inst_vars);
    let s = OracleSource::committed(a_oracle, map);
    match bias {
        Some(b) => s.with_bias(b),
        None => s,
    }
}

/// Matmul `B` (weight) source: identity over `k`+`col` vars (works for regular and transposed —
/// the committed weight layout is built to match the `reduce::matmul` operand point).
fn b_source(b_name: String, k: usize, col: usize) -> OracleSource {
    OracleSource::committed(b_name, PointMap::identity(vbits(k) + vbits(col)))
}

/// A head-slice operand source: `qs/ks/vs[seq, d] = qkv[seq, part·D + head·d_head + d]`, a column
/// slice of canonical `q_qkv` plus the `attn_b` slice as bias. The operand point is
/// `[d_head_part ++ seq_part]` (the `d_head` index is `reduce::matmul`'s low operand vars; the
/// sequence index — query/key row — maps to the qkv row). Single layer.
fn head_slice_src(config: &Config, weights: &ModelWeights, layer: usize, part: usize, head: usize) -> OracleSource {
    let dh_vars = vbits(config.d_head);
    let seq_vars = vbits(config.n_seq);
    let lay = canonical::layout_for(config, Q_QKV);
    let col_vars = lay.col_vars();
    let inst_vars = lay.inst_vars();
    // High qkv-column bits select (part, head): col = (g << dh_vars) | d, g = part·n_head + head.
    let g = part * config.n_head + head;
    let mut coords = Vec::with_capacity(col_vars + seq_vars + inst_vars);
    for i in 0..dh_vars {
        coords.push(Coord::Var(i)); // d (the contraction-side feature)
    }
    for b in 0..(col_vars - dh_vars) {
        coords.push(Coord::Const((g >> b) & 1 == 1));
    }
    for i in 0..seq_vars {
        coords.push(Coord::Var(dh_vars + i)); // sequence index → qkv row
    }
    for b in 0..inst_vars {
        coords.push(Coord::Const((layer >> b) & 1 == 1));
    }
    let base = part * config.d_model + head * config.d_head;
    let slice = &weights.blocks[layer].attn_b[base..base + config.d_head];
    OracleSource::committed(Q_QKV, PointMap(coords))
        .with_bias(feature_bias(slice, config.d_head, config))
}

/// Broadcast source for `sum_exp` over the softmax `q_prob`/`exp` layout: the operand point is
/// `[key(col) ++ query(row)]`; `sum_exp` depends only on the query row (broadcast over keys), so
/// the column vars are dropped and the row vars map to `SUM_EXP`'s row, + the head instance.
fn sum_exp_bc_src(config: &Config, layer: usize, head: usize) -> OracleSource {
    let qp = canonical::layout_for(config, Q_PROB);
    let (col_vars, row_vars) = (qp.col_vars(), qp.row_vars());
    let se_inst_vars = canonical::layout_for(config, SUM_EXP).inst_vars();
    let inst = layer * config.n_head + head;
    let mut coords = Vec::with_capacity(row_vars + se_inst_vars);
    for i in 0..row_vars {
        coords.push(Coord::Var(col_vars + i)); // operand query row → SUM_EXP row
    }
    for b in 0..se_inst_vars {
        coords.push(Coord::Const((inst >> b) & 1 == 1));
    }
    OracleSource::committed(SUM_EXP, PointMap(coords))
}

/// Softmax Type-B division descriptors (per head): the lower `r = scale·exp − q_prob·sum_exp ≥ 0`
/// and upper `sum_exp − 1 − r = q_prob·sum_exp + sum_exp − scale·exp − 1 ≥ 0` checks, both into
/// RANGE20. The `q_prob·sum_exp` product (built into the query column here) is reduced for the
/// proof by the batched [`prove_typeb_batch`]; the affine linear terms (`exp`, `sum_exp`) are
/// opened per descriptor.
fn typeb_descriptors(config: &Config, layer: usize, head: usize) -> [Descriptor; 2] {
    let inst = layer * config.n_head + head;
    let scale = EF::from(p3_goldilocks::Goldilocks::new(config.scale as u64));
    let q_src = || slice_src(config, Q_PROB, inst);
    let sum_bc = || sum_exp_bc_src(config, layer, head);
    let exp_src = || slice_src(config, EXP, inst);
    let mk = |prod_coeff, linears, constant| Descriptor {
        oracle_in: EXP.to_owned(),
        ty: TYPE_RANGE20,
        table_base: R20_BASE,
        vars: canonical::layout_for(config, EXP).elem_vars(),
        start: 0,
        kind: SegKind::TypeB {
            prod_left: q_src(),
            prod_right: sum_bc(),
            prod_coeff,
            linears,
            constant,
            op: TypeBOp { layer, head },
        },
    };
    [
        // lower: scale·exp − q_prob·sum_exp.
        mk(-EF::ONE, vec![(exp_src(), scale)], EF::ZERO),
        // upper: q_prob·sum_exp + sum_exp − scale·exp − 1.
        mk(EF::ONE, vec![(sum_bc(), EF::ONE), (exp_src(), -scale)], -EF::ONE),
    ]
}

/// A limb-recomposition check: `recomp(limb_oracle) == prod_coeff·(prod_left·prod_right) +
/// Σ coeff·linear + constant`, all at a fresh point in the `value_oracle` layout.
struct LimbCheckSpec {
    limb_oracle: String,
    value_oracle: &'static str,
    /// Optional `(left, right, coeff)` product term. `None` = a pure affine range (e.g. a quotient
    /// bound), which decomposes into limbs without a `prove_prod`.
    prod: Option<(OracleSource, OracleSource, EF)>,
    linears: Vec<(OracleSource, EF)>,
    constant: EF,
}

/// The committed limb-poly name for a quotient bound on `q_oracle`.
fn qbound_limb_name(q_oracle: &str) -> String {
    format!("{q_oracle}_bound_limb")
}

/// The full ordered list of limb-recomposition checks (both prover and verifier build it
/// identically). Order fixes the `recomps` alignment.
fn limb_checks(config: &Config) -> Vec<LimbCheckSpec> {
    let one = EF::ONE;
    let aln_vars = canonical::layout_for(config, A_LN).num_vars();
    let std_vars = canonical::layout_for(config, STD).num_vars();
    let id_src = |name: &'static str, vars| OracleSource::committed(name, PointMap::identity(vars));
    let mut checks = vec![
        // r = a_ln − q_ln·std ≥ 0.
        LimbCheckSpec {
            limb_oracle: LN_REM_LIMB.to_owned(),
            value_oracle: A_LN,
            prod: Some((id_src(Q_LN, aln_vars), std_bc_src(config), -one)),
            linears: vec![(id_src(A_LN, aln_vars), one)],
            constant: EF::ZERO,
        },
        // std − 1 − r = q_ln·std + std − a_ln − 1 ≥ 0 (asserts r < std).
        LimbCheckSpec {
            limb_oracle: LN_REM_HI_LIMB.to_owned(),
            value_oracle: A_LN,
            prod: Some((id_src(Q_LN, aln_vars), std_bc_src(config), one)),
            linears: vec![(std_bc_src(config), one), (id_src(A_LN, aln_vars), -one)],
            constant: -one,
        },
        // x − (y²−y) = var + 1 − std² + std ≥ 0.
        LimbCheckSpec {
            limb_oracle: SQRT_LO_LIMB.to_owned(),
            value_oracle: STD,
            prod: Some((id_src(STD, std_vars), id_src(STD, std_vars), -one)),
            linears: vec![(id_src(VAR, std_vars), one), (id_src(STD, std_vars), one)],
            constant: one,
        },
        // (y²+y) − x = std² + std − var − 1 ≥ 0.
        LimbCheckSpec {
            limb_oracle: SQRT_HI_LIMB.to_owned(),
            value_oracle: STD,
            prod: Some((id_src(STD, std_vars), id_src(STD, std_vars), one)),
            linears: vec![(id_src(STD, std_vars), one), (id_src(VAR, std_vars), -one)],
            constant: -one,
        },
    ];
    // Quotient bounds (pin each committed quotient to [−OFFSET, 2^32−OFFSET) so `divisor·q` cannot
    // wrap): `q + QUOTIENT_LIMB_OFFSET ∈ [0, 2^32)`. A pure affine range — no product term.
    // `q_prob` has a small direct RANGE13 bound already.
    for q in [
        Q_LN,
        Q_QKV,
        crate::canonical::Q_SC,
        crate::canonical::Q_AO,
        crate::canonical::Q_APR,
        Q_FC,
        Q_FPR,
        crate::canonical::Q_LOG,
    ] {
        let vars = canonical::layout_for(config, q).num_vars();
        checks.push(LimbCheckSpec {
            limb_oracle: qbound_limb_name(q),
            value_oracle: q,
            prod: None,
            linears: vec![(id_src(q, vars), one)],
            constant: ef_u64(QUOTIENT_LIMB_OFFSET),
        });
    }
    checks
}

/// The prover's `(L, R)` product operand vectors for a limb check's `prove_prod` (in the value
/// layout). `prod_left`/`prod_right` may map to other layouts, so materialize each by evaluating
/// the source at every boolean vertex of the value cube.
fn limb_check_operands(spec: &LimbCheckSpec, set: &CommitSet, config: &Config) -> (Vec<EF>, Vec<EF>) {
    let vars = canonical::layout_for(config, spec.value_oracle).num_vars();
    let total = 1usize << vars;
    let (pl, pr, _) = spec.prod.as_ref().expect("prod present");
    let left = (0..total).map(|i| source_at_index(pl, set, i)).collect();
    let right = (0..total).map(|i| source_at_index(pr, set, i)).collect();
    (left, right)
}

/// Evaluate an `OracleSource` at a general `point` (the operand value at that point, bias included).
fn source_eval(src: &OracleSource, set: &CommitSet, point: &[EF]) -> EF {
    let mapped = src.map.apply(point);
    set.prover_data(&src.oracle).unwrap().clone().eval(&mapped) + src.bias_at(point)
}

/// The element block of merged poly `oracle` for instance `inst` (`elem_len` contiguous values).
fn instance_block(set: &CommitSet, oracle: &str, inst: usize, config: &Config) -> Vec<EF> {
    let lay = canonical::layout_for(config, oracle);
    let elem_len = lay.row_pow * lay.col_pow;
    let poly = &set.prover_data(oracle).unwrap().0;
    poly[inst * elem_len..(inst + 1) * elem_len].to_vec()
}

/// Evaluate an `OracleSource` at boolean vertex `i` of its value cube: index the committed poly at
/// the mapped boolean point and add the public bias.
fn source_at_index(src: &OracleSource, set: &CommitSet, i: usize) -> EF {
    // At a boolean vertex the mapped point is boolean, so `map.apply` + the `== ONE` scan is just an
    // integer index: a `Const(true)`/`Var(j)`-with-bit-set coord sets index bit `b`. Compute it
    // directly — no per-vertex `Vec<EF>` allocations.
    let mut idx = 0usize;
    for (b, coord) in src.map.0.iter().enumerate() {
        let bit = match *coord {
            Coord::Const(v) => v,
            Coord::Var(j) => (i >> j) & 1 == 1,
        };
        if bit {
            idx |= 1 << b;
        }
    }
    let mut val = set.prover_data(&src.oracle).unwrap().0[idx];
    if let Some(bias) = &src.bias {
        val += bias_value_at_index(bias, i);
    }
    val
}

/// The (non-negative integer) limb-check target at value vertex `i`:
/// `prod_coeff·(L·R) + Σ coeff·linear + constant`, all evaluated in the value layout.
fn limb_target_at(spec: &LimbCheckSpec, set: &CommitSet, i: usize) -> EF {
    let mut acc = spec.constant;
    if let Some((pl, pr, coeff)) = &spec.prod {
        acc += *coeff * source_at_index(pl, set, i) * source_at_index(pr, set, i);
    }
    for (src, coeff) in &spec.linears {
        acc += *coeff * source_at_index(src, set, i);
    }
    acc
}

/// Build a limb check's committed limb poly: `limbs[i·NUM_LIMBS + l]` = limb `l` of the target at
/// vertex `i`, computed by the SAME formula on every vertex (padding included), so the limb range
/// check + recomposition exactly reproduce the target everywhere.
fn limb_poly_for(spec: &LimbCheckSpec, set: &CommitSet, config: &Config) -> Vec<EF> {
    let vars = canonical::layout_for(config, spec.value_oracle).num_vars();
    let total = 1usize << vars;
    let mut out = vec![EF::ZERO; total * NUM_LIMBS];
    for i in 0..total {
        let target = ef_to_u64(limb_target_at(spec, set, i));
        debug_assert!((target as u128) < (1u128 << (LIMB_BITS * NUM_LIMBS)));
        let mut v = target;
        for l in 0..NUM_LIMBS {
            out[i * NUM_LIMBS + l] = ef_u64(v & 0xffff);
            v >>= LIMB_BITS;
        }
    }
    out
}

/// The limb poly evaluation point for limb index `l` at value point `p` (`[l_bits ++ p]`).
fn limb_point(l: usize, p: &[EF]) -> Vec<EF> {
    let limb_vars = vbits(NUM_LIMBS);
    let mut lp: Vec<EF> = (0..limb_vars)
        .map(|b| if (l >> b) & 1 == 1 { EF::ONE } else { EF::ZERO })
        .collect();
    lp.extend_from_slice(p);
    lp
}

/// `sum_exp` broadcast into the `exp`/`q_prob` layout (`[idx = query·col_pow + key]`). Must match
/// the committed `SUM_EXP` viewed through [`sum_exp_bc_src`]: broadcast over **all** key columns
/// (not just `< n_seq`), and the padding query rows `≥ n_seq` carry `1` (the committed `SUM_EXP`
/// pads with 1 — see [`canonical::build_online`]). Any mismatch would make the Type-B `prove_prod`
/// emit a `sum_exp` opening that disagrees with the commitment.
fn sum_exp_broadcast_vals(config: &Config, witness: &Witness, layer: usize, head: usize) -> Vec<EF> {
    let qp = canonical::layout_for(config, Q_PROB);
    let col_pow = qp.col_pow;
    // Single-instance elem block (row_pow × col_pow), matching the EXP/q_prob segment.
    let mut out = vec![EF::ZERO; qp.row_pow * col_pow];
    let sums = &witness.blocks[layer].attention.sum_exp[head];
    for i in 0..qp.row_pow {
        let s = if i < config.n_seq { ef_i64(sums.get(i, 0)) } else { EF::ONE };
        for j in 0..col_pow {
            out[i * col_pow + j] = s;
        }
    }
    out
}

/// Broadcast source for LayerNorm `std` over the `a_ln`/`q_ln` layout: `std` depends on the row
/// (and instance), broadcast over the feature column. The operand point is `[col ++ row ++ inst]`.
fn std_bc_src(config: &Config) -> OracleSource {
    let aln = canonical::layout_for(config, A_LN);
    let (cv, rv, nv) = (aln.col_vars(), aln.row_vars(), aln.inst_vars());
    let mut coords = Vec::with_capacity(rv + nv);
    for i in 0..rv {
        coords.push(Coord::Var(cv + i)); // a_ln row → STD row
    }
    for i in 0..nv {
        coords.push(Coord::Var(cv + rv + i)); // a_ln instance → STD instance
    }
    OracleSource::committed(STD, PointMap(coords))
}

/// Extract the `S × d_head` head slice `qkv[:, part·D + head·d_head ..]`.
fn head_mat(qkv: &Matrix, config: &Config, part: usize, head: usize) -> Matrix {
    let base = part * config.d_model + head * config.d_head;
    let mut data = Vec::with_capacity(config.n_seq * config.d_head);
    for i in 0..config.n_seq {
        for d in 0..config.d_head {
            data.push(qkv.get(i, base + d));
        }
    }
    Matrix::new(config.n_seq, config.d_head, data)
}

/// Build the TypeA descriptor for a matmul instance `(layer, head)` (head ignored for the
/// non-attention matmuls). The quotient `oracle_in`, the matmul operand sources, and the divisor
/// are all determined by `mm`.
fn typea_descriptor(
    config: &Config,
    weights: &ModelWeights,
    mm: Mm,
    layer: usize,
    head: usize,
) -> Descriptor {
    let d = mm_dims(config, mm);
    let bw = &weights.blocks[layer];
    let bias = |b: &[i64]| Some(feature_bias(b, d.k, config));
    // (quotient oracle, A source, B source).
    let (q_oracle, a_src, b_src): (&str, OracleSource, OracleSource) = match mm {
        Mm::Qkv => (
            Q_QKV,
            a_source(config, Q_LN, 2 * layer, d.k, d.row, bias(&bw.ln_1_b)),
            b_source(canonical::attn_w(layer), d.k, d.col),
        ),
        Mm::Fc => (
            Q_FC,
            a_source(config, Q_LN, 2 * layer + 1, d.k, d.row, bias(&bw.ln_2_b)),
            b_source(canonical::fc_w(layer), d.k, d.col),
        ),
        Mm::Fpr => (
            Q_FPR,
            a_source(config, ACT, layer, d.k, d.row, None),
            b_source(canonical::fproj_w(layer), d.k, d.col),
        ),
        Mm::Apr => (
            crate::canonical::Q_APR,
            a_source(config, crate::canonical::Q_AO, layer, d.k, d.row, None),
            b_source(canonical::attn_proj_w(layer), d.k, d.col),
        ),
        Mm::Log => (
            crate::canonical::Q_LOG,
            a_source(config, Q_LN, 2 * config.n_layer, d.k, d.row, bias(&weights.ln_f_b)),
            b_source(canonical::WTE.to_owned(), d.k, d.col),
        ),
        Mm::Sc => (
            crate::canonical::Q_SC,
            head_slice_src(config, weights, layer, 0, head),
            head_slice_src(config, weights, layer, 1, head),
        ),
        Mm::Ao => (
            crate::canonical::Q_AO,
            a_source(config, Q_PROB, layer * config.n_head + head, d.k, d.row, None),
            head_slice_src(config, weights, layer, 2, head),
        ),
    };
    // Quotient instance in the merged poly: per-layer matmuls use the layer; attention matmuls use
    // `layer·n_head + head`; the lnf→logits matmul is the single instance 0.
    let inst = match mm {
        Mm::Sc | Mm::Ao => layer * config.n_head + head,
        Mm::Log => 0,
        _ => layer,
    };
    // The rescale remainder lies in [0, divisor): RANGE12 for `//SCALE`, RANGE15 for the attention
    // `//(SQRT_D·SCALE)` rescale.
    let (ty, table_base) = const_div_range(mm_divisor(config, mm));
    // The remainder/quotient segment lives in the matmul *output* layout (row=S, col=mm output);
    // for `Ao` this is the per-head `S×d_head` sub-block of the per-layer `q_ao`, so its quotient
    // source is an output-column slice (others are a whole-instance slice).
    let q_src = match mm {
        Mm::Ao => ao_out_slice_src(config, layer, head),
        _ => slice_src(config, q_oracle, inst),
    };
    Descriptor {
        oracle_in: q_oracle.to_owned(),
        ty,
        table_base,
        vars: vbits(d.col) + vbits(d.row),
        start: 0,
        kind: SegKind::TypeA {
            divisor: mm_divisor(config, mm),
            mk: mm_kind(mm),
            q_src,
            a_src,
            b_src,
            op: TypeAOp { mm, layer, head },
        },
    }
}

fn descriptors(config: &Config, weights: &ModelWeights) -> Vec<Descriptor> {
    let vars_of = |name: &str| canonical::layout_for(config, name).num_vars();
    let gelu_offset = (config.max_v * config.scale) as u64;
    let mut descs = vec![
        Descriptor {
            oracle_in: Q_PROB.to_owned(),
            ty: TYPE_RANGE13,
            table_base: R13_BASE,
            vars: vars_of(Q_PROB),
            start: 0,
            kind: SegKind::Direct {
                bias: None,
                offset: 0,
                out: None,
            },
        },
        // (The LayerNorm quotient bound on q_ln is a limb check — `QBoundQln` — since real q_ln
        // exceeds RANGE20; see `all_limb_checks`.)
        Descriptor {
            oracle_in: Q_FC.to_owned(),
            ty: TYPE_GELU,
            table_base: gelu_base(config),
            vars: vars_of(Q_FC),
            start: 0,
            kind: SegKind::Direct {
                bias: Some(fc_bias(config, weights)),
                offset: gelu_offset,
                out: Some(ACT.to_owned()),
            },
        },
    ];
    // Each limb check contributes a LIMB16 range segment on its committed limb poly; the
    // recomposition tying the limbs to the target value is checked separately (see `recomps`).
    for spec in limb_checks(config) {
        descs.push(Descriptor {
            vars: vars_of(spec.value_oracle) + vbits(NUM_LIMBS),
            oracle_in: spec.limb_oracle,
            ty: TYPE_LIMB16,
            table_base: limb16_base(config),
            start: 0,
            kind: SegKind::Direct {
                bias: None,
                offset: 0,
                out: None,
            },
        });
    }
    // Per-layer matmuls (one segment per instance).
    for mm in [Mm::Qkv, Mm::Fc, Mm::Fpr, Mm::Apr] {
        for layer in 0..config.n_layer {
            descs.push(typea_descriptor(config, weights, mm, layer, 0));
        }
    }
    // The lnf→logits matmul is a single instance.
    descs.push(typea_descriptor(config, weights, Mm::Log, 0, 0));
    // Per-head attention matmuls + softmax division (one segment per head).
    for layer in 0..config.n_layer {
        for head in 0..config.n_head {
            descs.push(typea_descriptor(config, weights, Mm::Sc, layer, head));
            descs.push(typea_descriptor(config, weights, Mm::Ao, layer, head));
            descs.extend(typeb_descriptors(config, layer, head));
        }
    }
    descs.sort_by_key(|d| core::cmp::Reverse(d.len()));
    let mut start = 0;
    for d in &mut descs {
        debug_assert_eq!(start % d.len(), 0);
        d.start = start;
        start += d.len();
    }
    descs
}

/// Per-segment opening values the prover sends. `in_value` is the canonical-`in` opening (the
/// quotient `q` for TypeA); `matmul` carries the contraction sumcheck for TypeA segments. The
/// `TypeB` product `q_prob·sum_exp` is *not* here — it is proven once for all heads in
/// [`UnifiedProof::typeb`]; each `TypeB` segment reads its per-head value from `head_evals`.
pub struct SegmentOpening {
    pub in_value: EF,
    pub out_value: Option<EF>,
    pub matmul: Option<MatmulProof>,
    /// Linear-term openings for a `TypeB` prod-plus-affine segment (in `linears` order).
    pub linear_openings: Vec<EF>,
}

/// The batched Type-B product proof: every head's softmax-division product `q_prob·sum_exp` shares
/// the same lookup point, so all are reduced by **one** sumcheck over the merged `(elem ++ inst)`
/// cube weighted by `eq(local, x)·γ^i` (a `γ`-RLC over the head instances). `head_evals[i]` is the
/// per-head product evaluation `(q_prob_i·sum_exp_i)(local)` (`i = layer·n_head + head`); the
/// sumcheck binds their RLC `Σ_i γ^i·head_evals[i]` to the committed `q_prob`/`sum_exp`.
pub struct BatchProdProof {
    pub head_evals: Vec<EF>,
    pub proof: SumcheckProof<EF>,
}

impl BatchProdProof {
    pub fn size_bytes(&self) -> usize {
        let ef = core::mem::size_of::<EF>();
        ef * self.head_evals.len() + self.proof.size_bytes()
    }
}

/// A limb-recomposition check: ties a committed limb poly to a target value
/// `prod_coeff·(L·R) + Σ coeff·linear + constant` at a fresh point. `prod` reduces the `L·R`
/// product; `openings` are the linear-term canonical openings; `limb_values` open the limb poly at
/// the per-limb shifted points.
pub struct RecompProof {
    pub prod: Option<ProdProof>,
    pub openings: Vec<EF>,
    pub limb_values: Vec<EF>,
}

pub struct UnifiedProof {
    pub lookup: LookupProof<EF>,
    pub segment_openings: Vec<SegmentOpening>,
    /// The single batched Type-B softmax-division product (all heads).
    pub typeb: BatchProdProof,
    pub recomps: Vec<RecompProof>,
    /// Openings of the three committed public-table columns at the lookup's table point `z_t`,
    /// combined as `in + α·out + α²·type` to reconstruct the table value.
    pub table_cols: [EF; 3],
    pub batch: <Pcs as PolyCommitmentScheme>::Proof,
}

impl UnifiedProof {
    /// Transcript size in bytes: the lookup PIOP, every per-segment matmul sumcheck, the batched
    /// Type-B product, the recomposition sumchecks + openings, and the scalar openings. (The
    /// placeholder PCS `batch` proof is empty; a real PCS opening would be added here.)
    pub fn size_bytes(&self) -> usize {
        let ef = core::mem::size_of::<EF>();
        let mut s = self.lookup.size_bytes();
        for so in &self.segment_openings {
            s += ef * (1 + so.out_value.is_some() as usize);
            if let Some(m) = &so.matmul {
                s += m.proof.size_bytes() + ef;
            }
            s += ef * so.linear_openings.len();
        }
        s += self.typeb.size_bytes();
        s += 3 * ef; // table column openings at z_t
        for r in &self.recomps {
            if let Some(p) = &r.prod {
                s += p.proof.size_bytes() + ef;
            }
            s += ef * (r.openings.len() + r.limb_values.len());
        }
        s
    }
}

/// Low-bit-first eq weights over `len` indices.
fn eq_weights(point: &[EF], len: usize) -> Vec<EF> {
    (0..len).map(|i| eq_at_index(point, i)).collect()
}

/// The `A@B` regular contraction vectors at `(row_point, col_point)`.
fn contract_regular(a: &Matrix, b: &Matrix, row_point: &[EF], col_point: &[EF]) -> (Vec<EF>, Vec<EF>) {
    let k_pow = a.cols().next_power_of_two();
    let row_eq = eq_weights(row_point, a.rows());
    let col_eq = eq_weights(col_point, b.cols());
    let mut avec = vec![EF::ZERO; k_pow];
    let mut bvec = vec![EF::ZERO; k_pow];
    for k in 0..a.cols() {
        avec[k] = (0..a.rows()).map(|r| row_eq[r] * ef_i64(a.get(r, k))).sum();
        bvec[k] = (0..b.cols()).map(|c| col_eq[c] * ef_i64(b.get(k, c))).sum();
    }
    (avec, bvec)
}

/// A TypeA segment's matmul operands `A`/`B` and quotient `q` (all owned), from witness/weights.
fn typea_inputs(
    op: &TypeAOp,
    config: &Config,
    weights: &ModelWeights,
    witness: &Witness,
) -> (Matrix, Matrix, Matrix) {
    let l = op.layer;
    let h = op.head;
    let blk = &witness.blocks[l];
    let bw = &weights.blocks[l];
    match op.mm {
        Mm::Qkv => (blk.ln1.output.clone(), bw.attn_w.clone(), blk.attention.q_qkv.clone()),
        Mm::Fc => (blk.ln2.output.clone(), bw.fc_w.clone(), blk.mlp.q_fc.clone()),
        Mm::Fpr => (blk.mlp.act.clone(), bw.fproj_w.clone(), blk.mlp.q_fpr.clone()),
        Mm::Apr => (
            blk.attention.q_ao.clone(),
            bw.attn_proj_w.clone(),
            blk.attention.q_apr.clone(),
        ),
        Mm::Log => (witness.lnf.output.clone(), weights.wte.clone(), witness.q_log.clone()),
        Mm::Sc => (
            head_mat(&blk.attention.qkv, config, 0, h),
            head_mat(&blk.attention.qkv, config, 1, h),
            blk.attention.scores[h].head_matrix(0),
        ),
        Mm::Ao => (
            blk.attention.q_prob[h].head_matrix(0),
            head_mat(&blk.attention.qkv, config, 2, h),
            blk.attention.q_ao_heads[h].head_matrix(0),
        ),
    }
}

/// The rescale-fusion remainder matrix `r = C − divisor·q` for a TypeA matmul instance, in the
/// matmul *output* layout (row=S, col=mm output). Retained by the forward pass (see
/// [`crate::witness::AttentionWitness`]), so this is a cheap lookup — no matmul-product recompute.
fn typea_remainder(op: &TypeAOp, witness: &Witness) -> Matrix {
    let (l, h) = (op.layer, op.head);
    let blk = &witness.blocks[l];
    match op.mm {
        Mm::Qkv => blk.attention.rem_qkv.clone(),
        Mm::Fc => blk.mlp.rem_fc.clone(),
        Mm::Fpr => blk.mlp.rem_fpr.clone(),
        Mm::Apr => blk.attention.rem_apr.clone(),
        Mm::Log => witness.rem_log.clone(),
        Mm::Sc => blk.attention.rem_sc[h].head_matrix(0),
        Mm::Ao => blk.attention.rem_ao_heads[h].head_matrix(0),
    }
}

/// `A@Bᵀ` transposed contraction vectors (`B` stored `col × k`).
fn contract_transposed(
    a: &Matrix,
    bt: &Matrix,
    row_point: &[EF],
    col_point: &[EF],
) -> (Vec<EF>, Vec<EF>) {
    let k_pow = a.cols().next_power_of_two();
    let row_eq = eq_weights(row_point, a.rows());
    let col_eq = eq_weights(col_point, bt.rows());
    let mut avec = vec![EF::ZERO; k_pow];
    let mut bvec = vec![EF::ZERO; k_pow];
    for k in 0..a.cols() {
        avec[k] = (0..a.rows()).map(|r| row_eq[r] * ef_i64(a.get(r, k))).sum();
        bvec[k] = (0..bt.rows()).map(|c| col_eq[c] * ef_i64(bt.get(c, k))).sum();
    }
    (avec, bvec)
}

/// Compute a segment's `in`/`out`/`table-index` columns (prover side, α-independent).
fn segment_columns(
    set: &CommitSet,
    config: &Config,
    witness: &Witness,
    d: &Descriptor,
) -> (Vec<EF>, Vec<EF>, Vec<usize>) {
    let len = d.len();
    let mut in_vals = vec![EF::ZERO; len];
    let mut out_vals = vec![EF::ZERO; len];
    let mut indices = vec![d.table_base; len]; // padding entries (value 0) hit table_base
    match &d.kind {
        SegKind::Direct { bias, offset, out } => {
            let in_poly = &set.prover_data(&d.oracle_in).unwrap().0;
            let out_poly = out.as_ref().map(|o| &set.prover_data(o).unwrap().0);
            for i in 0..len {
                let mut v = in_poly[i] + ef_u64(*offset);
                if let Some(b) = bias {
                    v += bias_value_at_index(b, i);
                }
                in_vals[i] = v;
                out_vals[i] = out_poly.map_or(EF::ZERO, |p| p[i]);
                indices[i] = d.table_base + ef_to_u64(v) as usize;
            }
        }
        SegKind::TypeA { divisor, op, .. } => {
            // r = C − divisor·q in the matmul *output* layout (row=S, col=mm output; padding ⇒ 0).
            // The remainder is retained by the forward pass, so no matmul-product recompute here.
            let rem = typea_remainder(op, witness);
            let col_pow = mm_dims(config, op.mm).col.next_power_of_two();
            for r in 0..rem.rows() {
                for col in 0..rem.cols() {
                    let idx = r * col_pow + col;
                    let v = rem.get(r, col);
                    debug_assert!((0..*divisor as i64).contains(&v));
                    in_vals[idx] = ef_i64(v);
                    indices[idx] = d.table_base + v as usize;
                }
            }
        }
        SegKind::TypeB { prod_left, prod_right, prod_coeff, linears, constant, .. } => {
            // in = prod_coeff·(L·R) + Σ coeff·linear + const, by the SAME formula on every vertex.
            for i in 0..len {
                let mut v = *prod_coeff
                    * source_at_index(prod_left, set, i)
                    * source_at_index(prod_right, set, i)
                    + *constant;
                for (src, coeff) in linears {
                    v += *coeff * source_at_index(src, set, i);
                }
                debug_assert!((ef_to_u64(v) as usize) < range_b_size(config));
                in_vals[i] = v;
                indices[i] = d.table_base + ef_to_u64(v) as usize;
            }
        }
    }
    (in_vals, out_vals, indices)
}

/// Bias value at boolean vertex `i`. At a boolean point `MlPoly(values).eval` is just an array
/// index, so select `values[idx]` directly (`idx` packs the bits `i` at the bias's `vars`),
/// avoiding the per-vertex `values.clone()` + `2^|vars|` eval that dominated commit/query build.
fn bias_value_at_index(bias: &BiasPoly, i: usize) -> EF {
    let mut idx = 0usize;
    for (k, &var) in bias.vars.iter().enumerate() {
        if (i >> var) & 1 == 1 {
            idx |= 1 << k;
        }
    }
    bias.values[idx]
}

fn multiplicity_counts(
    set: &CommitSet,
    config: &Config,
    weights: &ModelWeights,
    witness: &Witness,
    descs: &[Descriptor],
) -> Vec<u64> {
    let total = total_query_len(descs);
    let mut counts = vec![0u64; table_len(config, weights)];
    for d in descs {
        let (_, _, indices) = segment_columns(set, config, witness, d);
        for &idx in &indices {
            counts[idx] += 1;
        }
    }
    let padded = total.next_power_of_two();
    counts[R12_BASE] += (padded - total) as u64; // padding rows → RANGE12 row 0 (the default fold)
    counts
}

fn total_query_len(descs: &[Descriptor]) -> usize {
    descs.iter().map(Descriptor::len).sum()
}

/// Commit the **offline** (data-independent) polynomials: the model weights and the public unified
/// table columns (`in`/`out`/`type`). These depend only on `config`/`weights` — not the witness —
/// so they are committed *before* the forward pass and can be reused across inferences.
pub fn build_offline_commitments(config: &Config, weights: &ModelWeights) -> Canonical {
    let mut canon = canonical::build_offline(config, weights);
    // The public unified-table columns, committed as MLEs (opened at the lookup's table point like
    // any other oracle, rather than reconstructed by the verifier).
    let (tin, tout, tty) = table_columns(config, weights);
    canon.set.commit(TABLE_IN, tin);
    canon.set.commit(TABLE_OUT, tout);
    canon.set.commit(TABLE_TYPE, tty);
    canon
}

/// Commit the **online** (witness-dependent) polynomials on top of an offline set: the merged
/// canonical witness types, the limb-decomposition polys, and the lookup multiplicity `e`. Consumes
/// the offline `Canonical` (from [`build_offline_commitments`]) and folds it into the result.
pub fn build_online_commitments(
    offline: Canonical,
    config: &Config,
    weights: &ModelWeights,
    witness: &Witness,
) -> Canonical {
    let mut canon = canonical::build_online(config, witness);
    canon.set.extend(offline.set);
    canon.layouts.extend(offline.layouts);
    for spec in limb_checks(config) {
        let limbs = limb_poly_for(&spec, &canon.set, config);
        canon.set.commit(spec.limb_oracle, limbs);
    }
    let descs = descriptors(config, weights);
    let counts = multiplicity_counts(&canon.set, config, weights, witness, &descs);
    let e: Vec<EF> = counts.into_iter().map(ef_u64).collect();
    canon.set.commit(E, e);
    canon
}

/// Commit offline + online in one call (offline weights/table, then witness/limbs/`e`). A
/// convenience wrapper; the pipeline commits the offline set before the forward pass via
/// [`build_offline_commitments`] / [`build_online_commitments`].
pub fn build_commitments(config: &Config, weights: &ModelWeights, witness: &Witness) -> Canonical {
    let offline = build_offline_commitments(config, weights);
    build_online_commitments(offline, config, weights, witness)
}

fn build_query(
    set: &CommitSet,
    config: &Config,
    witness: &Witness,
    descs: &[Descriptor],
    alpha: EF,
) -> Vec<EF> {
    let total = total_query_len(descs);
    let padded = total.next_power_of_two();
    let mut q = vec![EF::ZERO; padded];
    for d in descs {
        let (in_vals, out_vals, _) = segment_columns(set, config, witness, d);
        for i in 0..d.len() {
            q[d.start + i] = fold_row(alpha, in_vals[i], out_vals[i], d.ty);
        }
    }
    let default = fold_row(alpha, EF::ZERO, EF::ZERO, TYPE_RANGE12);
    for slot in q.iter_mut().take(padded).skip(total) {
        *slot = default;
    }
    q
}

/// The merged Type-B operand sources over the `q_prob` `(elem ++ inst)` cube: `left` is committed
/// `q_prob` (identity); `right` is `sum_exp` broadcast over the key columns + the head instance
/// (drop the key vars, map query-row + instance to `SUM_EXP`). Used to emit the batched product's
/// two operand openings.
fn typeb_batch_sources(config: &Config) -> (OracleSource, OracleSource) {
    let qp = canonical::layout_for(config, Q_PROB);
    let (cv, rv, nv) = (qp.col_vars(), qp.row_vars(), qp.inst_vars());
    let left = OracleSource::committed(Q_PROB, PointMap::identity(cv + rv + nv));
    let mut coords = Vec::with_capacity(rv + nv);
    for i in 0..rv {
        coords.push(Coord::Var(cv + i)); // operand query row → SUM_EXP row
    }
    for i in 0..nv {
        coords.push(Coord::Var(cv + rv + i)); // operand instance → SUM_EXP instance
    }
    (left, OracleSource::committed(SUM_EXP, PointMap(coords)))
}

/// The full `sum_exp` broadcast over the `q_prob` `(elem ++ inst)` cube (prover operand):
/// `R[elem ++ inst] = SUM_EXP[query_row(elem), inst]`, broadcast over the key columns (padding rows
/// carry the committed 1). Matches [`typeb_batch_sources`]'s `right` as a polynomial.
fn typeb_full_right(set: &CommitSet, config: &Config) -> Vec<EF> {
    let qp = canonical::layout_for(config, Q_PROB);
    let se = canonical::layout_for(config, SUM_EXP);
    let elem_size = qp.row_pow * qp.col_pow;
    let inst_pow = 1usize << qp.inst_vars();
    let se_poly = &set.prover_data(SUM_EXP).unwrap().0;
    let mut out = vec![EF::ZERO; elem_size * inst_pow];
    for inst in 0..inst_pow {
        for query in 0..qp.row_pow {
            let s = se_poly[inst * se.row_pow + query]; // SUM_EXP[query, inst] (col_pow = 1)
            for key in 0..qp.col_pow {
                out[inst * elem_size + query * qp.col_pow + key] = s;
            }
        }
    }
    out
}

/// The Type-B batching weight `eq(local, x)·γ^i` over the `(elem ++ inst)` cube (`x` = elem index
/// on the low vars, `i` = instance on the high vars), so the batched sumcheck `Σ weight·q_prob·R`
/// equals `Σ_i γ^i·(q_prob_i·sum_exp_i)(local)`.
fn typeb_weight(local: &[EF], gamma: EF, elem_vars: usize, inst_vars: usize) -> Vec<EF> {
    let elem_size = 1usize << elem_vars;
    let eqv = eq_weights(local, elem_size);
    let inst_pow = 1usize << inst_vars;
    let mut out = vec![EF::ZERO; elem_size * inst_pow];
    let mut gpow = EF::ONE;
    for i in 0..inst_pow {
        for x in 0..elem_size {
            out[i * elem_size + x] = eqv[x] * gpow;
        }
        gpow *= gamma;
    }
    out
}

/// Prove all heads' Type-B products `q_prob_i·sum_exp_i` at the shared lookup point in **one**
/// sumcheck (a `γ`-RLC over the head instances), emitting the two merged operand openings.
fn prove_typeb_batch(
    set: &CommitSet,
    config: &Config,
    witness: &Witness,
    zq: &[EF],
    acc: &mut ClaimAccumulator,
    oracle: &mut RandomOracle<EF>,
) -> BatchProdProof {
    let qp = canonical::layout_for(config, Q_PROB);
    let (elem_vars, inst_vars) = (qp.elem_vars(), qp.inst_vars());
    let local = &zq[..elem_vars];
    let n_heads = config.n_layer * config.n_head;
    // Per-head product evaluation qb_i = (q_prob_i · sum_exp_i)(local).
    let head_evals: Vec<EF> = (0..n_heads)
        .map(|i| {
            let l = instance_block(set, Q_PROB, i, config);
            let r = sum_exp_broadcast_vals(config, witness, i / config.n_head, i % config.n_head);
            let prod: Vec<EF> = l.iter().zip(&r).map(|(&a, &b)| a * b).collect();
            MlPoly(prod).eval(local)
        })
        .collect();
    let gamma = oracle.next_field();
    let left = set.prover_data(Q_PROB).unwrap().0.clone();
    let right = typeb_full_right(set, config);
    let weight = typeb_weight(local, gamma, elem_vars, inst_vars);
    let (proof, sc) = sumcheck::prove(vec![weight, left, right], oracle);
    let open = reduce::low_first(&sc);
    let (l_src, r_src) = typeb_batch_sources(config);
    l_src.emit(acc, &open, proof.final_evals[1]);
    r_src.emit(acc, &open, proof.final_evals[2]);
    BatchProdProof { head_evals, proof }
}

/// Verify the batched Type-B product: rebuild `Σ_i γ^i·head_evals[i]`, check the sumcheck + the
/// public weight factor, and emit the merged `q_prob`/`sum_exp` openings. Returns `false` on any
/// failure.
fn verify_typeb_batch(
    batch: &BatchProdProof,
    config: &Config,
    zq: &[EF],
    acc: &mut ClaimAccumulator,
    oracle: &mut RandomOracle<EF>,
) -> bool {
    let qp = canonical::layout_for(config, Q_PROB);
    let (elem_vars, inst_vars) = (qp.elem_vars(), qp.inst_vars());
    if batch.head_evals.len() != config.n_layer * config.n_head {
        return false;
    }
    let local = &zq[..elem_vars];
    let gamma = oracle.next_field();
    let mut claim = EF::ZERO;
    let mut gpow = EF::ONE;
    for &qb in &batch.head_evals {
        claim += gpow * qb;
        gpow *= gamma;
    }
    let Some(sc) = sumcheck::verify(claim, &batch.proof, oracle) else {
        return false;
    };
    if batch.proof.final_evals.len() != 3 {
        return false;
    }
    let open = reduce::low_first(&sc);
    // The weight factor is public; its terminal eval must match before trusting the operand evals.
    let weight = typeb_weight(local, gamma, elem_vars, inst_vars);
    if batch.proof.final_evals[0] != MlPoly(weight).eval(&open) {
        return false;
    }
    let (l_src, r_src) = typeb_batch_sources(config);
    l_src.emit(acc, &open, batch.proof.final_evals[1]);
    r_src.emit(acc, &open, batch.proof.final_evals[2]);
    true
}

pub fn prove(
    set: &CommitSet,
    config: &Config,
    weights: &ModelWeights,
    witness: &Witness,
    oracle: &mut RandomOracle<EF>,
) -> UnifiedProof {
    let descs = descriptors(config, weights);
    let alpha = oracle.next_field();
    let query = build_query(set, config, witness, &descs, alpha);
    let table = build_table(alpha, config, weights);
    let e = set.prover_data(E).unwrap().0.clone();

    let (lookup, zq, zt) = lookup::prove_with_opening_points(&query, &table, &e, oracle);

    let mut acc = ClaimAccumulator::new();
    let mut segment_openings = Vec::with_capacity(descs.len());
    for d in &descs {
        let local = zq[..d.vars].to_vec();
        let mut out_value = None;
        let mut matmul = None;
        let mut linear_openings = Vec::new();
        // `in_value` is the segment's primary canonical opening: the merged `in` (Direct) or the
        // instance-sliced quotient `q` (TypeA). TypeB reconstructs `in` from prod + linears.
        let in_value = match &d.kind {
            SegKind::Direct { out, .. } => {
                let v = set.prover_data(&d.oracle_in).unwrap().clone().eval(&local);
                acc.open(d.oracle_in.clone(), local.clone(), v);
                if let Some(o) = out {
                    let ov = set.prover_data(o).unwrap().clone().eval(&local);
                    acc.open(o.clone(), local.clone(), ov);
                    out_value = Some(ov);
                }
                v
            }
            SegKind::TypeA { mk, q_src, a_src, b_src, op, .. } => {
                let q = source_eval(q_src, set, &local);
                q_src.emit(&mut acc, &local, q);
                let (row_point, col_point) = typea_points(config, op.mm, &local);
                let (a, b, _q) = typea_inputs(op, config, weights, witness);
                let (avec, bvec) = match mk {
                    MatmulKind::Regular => contract_regular(&a, &b, &row_point, &col_point),
                    MatmulKind::Transposed => contract_transposed(&a, &b, &row_point, &col_point),
                };
                matmul = Some(reduce::prove_matmul(
                    *mk, a_src, b_src, avec, bvec, &row_point, &col_point, &mut acc, oracle,
                ));
                q
            }
            SegKind::TypeB { linears, .. } => {
                // The `q_prob·sum_exp` product is proven once for all heads after this loop (see
                // `prove_typeb_batch`); here we only open the affine linear terms.
                for (src, _) in linears {
                    let v = source_eval(src, set, &local);
                    src.emit(&mut acc, &local, v);
                    linear_openings.push(v);
                }
                EF::ZERO // unused for TypeB (in is reconstructed from the batched prod + linears)
            }
        };
        segment_openings.push(SegmentOpening {
            in_value,
            out_value,
            matmul,
            linear_openings,
        });
    }

    // Batched Type-B softmax-division product: all heads' `q_prob·sum_exp` at the shared lookup
    // point in one sumcheck. Drawn after the per-segment matmul sumchecks, before the recomps.
    let typeb = prove_typeb_batch(set, config, witness, &zq, &mut acc, oracle);

    // Limb-recomposition checks (LayerNorm `// std` remainder, sqrt lower bracket) at fresh points.
    let mut recomps = Vec::new();
    for spec in limb_checks(config) {
        let vars = canonical::layout_for(config, spec.value_oracle).num_vars();
        let p = oracle.next_n_fields(vars);
        let prod = spec.prod.as_ref().map(|(pl, pr, _)| {
            let (left, right) = limb_check_operands(&spec, set, config);
            reduce::prove_prod(left, right, pl, pr, p.clone(), &mut acc, oracle)
        });
        let openings: Vec<EF> = spec
            .linears
            .iter()
            .map(|(src, _)| {
                let mapped = src.map.apply(&p);
                let v = set.prover_data(&src.oracle).unwrap().clone().eval(&mapped)
                    + src.bias_at(&p);
                src.emit(&mut acc, &p, v);
                v
            })
            .collect();
        let limb_values: Vec<EF> = (0..NUM_LIMBS)
            .map(|l| {
                let lp = limb_point(l, &p);
                let v = set.prover_data(&spec.limb_oracle).unwrap().clone().eval(&lp);
                acc.open(spec.limb_oracle.clone(), lp, v);
                v
            })
            .collect();
        recomps.push(RecompProof {
            prod,
            openings,
            limb_values,
        });
    }

    // Public table columns + multiplicity `e`, opened at the lookup's table point `z_t`.
    let table_cols = [TABLE_IN, TABLE_OUT, TABLE_TYPE].map(|name| {
        let v = set.prover_data(name).unwrap().clone().eval(&zt);
        acc.open(name, zt.clone(), v);
        v
    });
    let e_val = set.prover_data(E).unwrap().clone().eval(&zt);
    acc.open(E, zt, e_val);

    let tasks: Vec<_> = acc
        .claims
        .iter()
        .map(|c| (set.prover_data(&c.oracle).unwrap(), c.point.clone()))
        .collect();

    UnifiedProof {
        lookup,
        segment_openings,
        typeb,
        recomps,
        table_cols,
        batch: Pcs::batch_prove(&tasks, oracle),
    }
}

/// Split a TypeA segment's local point into `(row_point, col_point)` in the matmul *output* layout
/// (column low, row high).
fn typea_points(config: &Config, mm: Mm, local: &[EF]) -> (Vec<EF>, Vec<EF>) {
    let d = mm_dims(config, mm);
    let col_vars = vbits(d.col);
    let row_vars = vbits(d.row);
    let col_point = local[..col_vars].to_vec();
    let row_point = local[col_vars..col_vars + row_vars].to_vec();
    (row_point, col_point)
}

fn segment_selector(zq: &[EF], start: usize, vars: usize) -> EF {
    eq_at_index(&zq[vars..], start >> vars)
}

pub fn verify(
    set: &CommitSet,
    config: &Config,
    weights: &ModelWeights,
    proof: &UnifiedProof,
    oracle: &mut RandomOracle<EF>,
) -> bool {
    let descs = descriptors(config, weights);
    if proof.segment_openings.len() != descs.len() {
        return false;
    }
    let alpha = oracle.next_field();

    let Some(claims) = lookup::verify(&proof.lookup, oracle) else {
        return false;
    };
    let find = |p: Poly| claims.iter().find(|c| c.poly == p).unwrap();
    let query = find(Poly::Query);
    let mult = find(Poly::Mult);
    let table_claim = find(Poly::Table);
    let zq = &query.point;

    if mult.point != table_claim.point
        || zq.len() != total_query_len(&descs).next_power_of_two().trailing_zeros() as usize
    {
        return false;
    }
    let mut acc = ClaimAccumulator::new();
    // Table side: the lookup's folded table value must equal `in + α·out + α²·type` of the three
    // committed public-table columns opened at `z_t`; emit those openings for the batch-open. (No
    // table reconstruction — with a real PCS the verifier just checks the three openings.)
    let [tin, tout, tty] = proof.table_cols;
    if tin + alpha * tout + alpha * alpha * tty != table_claim.value {
        return false;
    }
    let zt = table_claim.point.clone();
    acc.open(TABLE_IN, zt.clone(), tin);
    acc.open(TABLE_OUT, zt.clone(), tout);
    acc.open(TABLE_TYPE, zt, tty);

    let mut recon = EF::ZERO;
    for (d, opening) in descs.iter().zip(&proof.segment_openings) {
        let local = zq[..d.vars].to_vec();
        let in_full = match &d.kind {
            SegKind::Direct { bias, offset, out } => {
                if opening.out_value.is_some() != out.is_some() {
                    return false;
                }
                acc.open(d.oracle_in.clone(), local.clone(), opening.in_value);
                let mut v = opening.in_value + ef_u64(*offset);
                if let Some(b) = bias {
                    v += b.eval(&local);
                }
                if let (Some(o), Some(val)) = (out, opening.out_value) {
                    acc.open(o.clone(), local.clone(), val);
                }
                v
            }
            SegKind::TypeA { divisor, mk, q_src, a_src, b_src, op } => {
                let Some(mmp) = &opening.matmul else {
                    return false;
                };
                q_src.emit(&mut acc, &local, opening.in_value);
                let (row_point, col_point) = typea_points(config, op.mm, &local);
                let Some(c) =
                    reduce::verify_matmul(*mk, a_src, b_src, &row_point, &col_point, mmp, &mut acc, oracle)
                else {
                    return false;
                };
                c - ef_u64(*divisor) * opening.in_value
            }
            SegKind::TypeB { prod_coeff, linears, constant, op, .. } => {
                if opening.linear_openings.len() != linears.len() {
                    return false;
                }
                // The `q_prob·sum_exp` product is verified once for all heads below; read this
                // head's value from the batched proof.
                let head = op.layer * config.n_head + op.head;
                let Some(&qb) = proof.typeb.head_evals.get(head) else {
                    return false;
                };
                let mut v = *prod_coeff * qb + *constant;
                for ((src, coeff), &ov) in linears.iter().zip(&opening.linear_openings) {
                    src.emit(&mut acc, &local, ov);
                    v += *coeff * ov;
                }
                v
            }
        };
        let out_full = opening.out_value.unwrap_or(EF::ZERO);
        recon += segment_selector(zq, d.start, d.vars) * fold_row(alpha, in_full, out_full, d.ty);
    }
    let total = total_query_len(&descs);
    let padded = total.next_power_of_two();
    let default = fold_row(alpha, EF::ZERO, EF::ZERO, TYPE_RANGE12);
    for idx in total..padded {
        recon += eq_at_index(zq, idx) * default;
    }
    if recon != query.value {
        return false;
    }

    // Batched Type-B product (all heads), drawn after the per-segment matmul sumchecks.
    if !verify_typeb_batch(&proof.typeb, config, zq, &mut acc, oracle) {
        return false;
    }

    // Limb-recomposition checks at fresh points.
    let checks = limb_checks(config);
    if proof.recomps.len() != checks.len() {
        return false;
    }
    for (spec, recomp) in checks.iter().zip(&proof.recomps) {
        let vars = canonical::layout_for(config, spec.value_oracle).num_vars();
        let p = oracle.next_n_fields(vars);
        if recomp.prod.is_some() != spec.prod.is_some()
            || recomp.openings.len() != spec.linears.len()
            || recomp.limb_values.len() != NUM_LIMBS
        {
            return false;
        }
        let mut target = spec.constant;
        if let (Some((pl, pr, coeff)), Some(pp)) = (&spec.prod, &recomp.prod) {
            let Some(qb) = reduce::verify_prod(&p, pl, pr, pp, &mut acc, oracle) else {
                return false;
            };
            target += *coeff * qb;
        }
        for ((src, coeff), &v) in spec.linears.iter().zip(&recomp.openings) {
            src.emit(&mut acc, &p, v);
            target += *coeff * v;
        }
        let mut recomp_val = EF::ZERO;
        for (l, &v) in recomp.limb_values.iter().enumerate() {
            acc.open(spec.limb_oracle.clone(), limb_point(l, &p), v);
            recomp_val += ef_u64(1u64 << (LIMB_BITS * l)) * v;
        }
        if recomp_val != target {
            return false;
        }
    }

    acc.open(E, mult.point.clone(), mult.value);

    let pcs_claims: Vec<_> = acc
        .claims
        .iter()
        .map(|c| (set.commitment(&c.oracle).unwrap(), c.point.clone(), c.value))
        .collect();
    Pcs::batch_verify(&pcs_claims, &proof.batch, oracle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::BlockWeights;

    fn tiny() -> (Config, ModelWeights, Witness) {
        let cfg = Config {
            n_layer: 1,
            n_seq: 2,
            n_head: 1,
            d_head: 2,
            d_model: 2,
            mlp_hidden: 3,
            vocab: 4,
            scale: 4,
            max_v: 2,
        };
        let exp_lut = (0..=(cfg.max_v * cfg.scale))
            .map(|i| if i == 0 { 0 } else { 1 })
            .collect();
        let gelu_lut = (0..=(2 * cfg.max_v * cfg.scale))
            .map(|i| i - cfg.max_v * cfg.scale)
            .collect();
        let weights = ModelWeights {
            wte: Matrix::new(4, 2, vec![1, 0, 0, 1, 1, 1, -1, 2]),
            wpe: Matrix::zeros(1024, 2),
            has_wpe: true,
            ln_f_g: vec![4, 4],
            ln_f_b: vec![0, 0],
            exp_lut,
            gelu_lut,
            blocks: vec![BlockWeights {
                ln_1_g: vec![4, 4],
                ln_1_b: vec![0, 0],
                attn_w: Matrix::new(2, 6, vec![1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]),
                attn_b: vec![0; 6],
                attn_proj_w: Matrix::new(2, 2, vec![1, 0, 0, 1]),
                attn_proj_b: vec![0, 0],
                ln_2_g: vec![4, 4],
                ln_2_b: vec![0, 0],
                fc_w: Matrix::new(2, 3, vec![1, 0, 1, 0, 1, -1]),
                fc_b: vec![1, -1, 2],
                fproj_w: Matrix::new(3, 2, vec![1, 0, 0, 1, 1, -1]),
                fproj_b: vec![0, 0],
            }],
        };
        let witness = weights.forward(Matrix::new(2, 2, vec![1, -2, 3, 1]), &cfg);
        (cfg, weights, witness)
    }

    /// A 2-layer version (per-layer + per-head instance batching), built by cloning tiny's block.
    fn tiny_multi() -> (Config, ModelWeights, Witness) {
        let (mut cfg, mut weights, _) = tiny();
        cfg.n_layer = 2;
        let block = weights.blocks[0].clone();
        weights.blocks.push(block);
        let witness = weights.forward(Matrix::new(2, 2, vec![1, -2, 3, 1]), &cfg);
        (cfg, weights, witness)
    }

    /// A config with a **non-power-of-two** `n_seq` (so the canonical layout has padding rows) and
    /// **nonzero** layernorm / attention biases. This is the regression guard for the padding-bias
    /// bug: a row-broadcast `feature_bias`/`fc_bias`/`sum_exp` would put a nonzero bias on the
    /// padding rows the matmul contraction never sums (`Σ_r eq(row_point,r) ≠ 1`), making the
    /// unified lookup and the operand openings inconsistent. `tiny`/`tiny_multi` use `n_seq = 2`
    /// (a power of two) and so never exercise it.
    fn tiny_odd() -> (Config, ModelWeights, Witness) {
        let cfg = Config {
            n_layer: 1,
            n_seq: 3,
            n_head: 2,
            d_head: 2,
            d_model: 4,
            mlp_hidden: 3,
            vocab: 4,
            scale: 4,
            max_v: 64,
        };
        let exp_lut = (0..=(cfg.max_v * cfg.scale))
            .map(|i| if i == 0 { 0 } else { 1 })
            .collect();
        let gelu_lut = (0..=(2 * cfg.max_v * cfg.scale))
            .map(|i| i - cfg.max_v * cfg.scale)
            .collect();
        #[rustfmt::skip]
        let weights = ModelWeights {
            wte: Matrix::new(4, 4, vec![1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]),
            wpe: Matrix::zeros(1024, 4),
            has_wpe: true,
            ln_f_g: vec![4, 4, 4, 4],
            ln_f_b: vec![1, -1, 1, -1],
            exp_lut,
            gelu_lut,
            blocks: vec![BlockWeights {
                ln_1_g: vec![4, 4, 4, 4],
                ln_1_b: vec![1, -1, 2, 0],
                attn_w: Matrix::new(4, 12, vec![
                    4,0,0,0, 4,0,0,0, 4,0,0,0,
                    0,4,0,0, 0,4,0,0, 0,4,0,0,
                    0,0,4,0, 0,0,4,0, 0,0,4,0,
                    0,0,0,4, 0,0,0,4, 0,0,0,4,
                ]),
                attn_b: vec![1, 0, -1, 0, 0, 1, 0, -1, 1, 0, 0, 1],
                attn_proj_w: Matrix::new(4, 4, vec![4,0,0,0, 0,4,0,0, 0,0,4,0, 0,0,0,4]),
                attn_proj_b: vec![0, 0, 0, 0],
                ln_2_g: vec![4, 4, 4, 4],
                ln_2_b: vec![-1, 1, 0, 1],
                fc_w: Matrix::new(4, 3, vec![4,0,0, 0,4,0, 0,0,4, 4,0,0]),
                fc_b: vec![1, -1, 2],
                fproj_w: Matrix::new(3, 4, vec![4,0,0,0, 0,4,0,0, 0,0,4,0]),
                fproj_b: vec![0, 0, 0, 0],
            }],
        };
        let x0 = Matrix::new(3, 4, vec![1, -2, 3, 1, 0, 2, -1, 1, 2, 0, 1, -1]);
        let witness = weights.forward(x0, &cfg);
        (cfg, weights, witness)
    }

    // NOTE: these exercise the full PIOP (forward → commit → prove → verify) end to end. The
    // default PCS is [`pcs::NoopPcs`], whose `batch_verify` always accepts, so they assert the
    // honest proof *verifies* (lookup, recon, batched Type-B, recomps all run for real) but cannot
    // test rejection of a tampered commitment — that is bound only by a sound PCS opening.

    #[test]
    fn unified_odd_seq_accepts() {
        let (cfg, weights, witness) = tiny_odd();
        let canon = build_commitments(&cfg, &weights, &witness);

        let mut rng = rand::rng();
        let mut oracle = RandomOracle::<EF>::new(&mut rng);
        let proof = prove(&canon.set, &cfg, &weights, &witness, &mut oracle);
        oracle.restart();
        assert!(verify(&canon.set, &cfg, &weights, &proof, &mut oracle));
    }

    #[test]
    fn unified_multi_instance_accepts() {
        let (cfg, weights, witness) = tiny_multi();
        let canon = build_commitments(&cfg, &weights, &witness);

        let mut rng = rand::rng();
        let mut oracle = RandomOracle::<EF>::new(&mut rng);
        let proof = prove(&canon.set, &cfg, &weights, &witness, &mut oracle);
        oracle.restart();
        assert!(verify(&canon.set, &cfg, &weights, &proof, &mut oracle));
    }

    #[test]
    fn unified_typea_accepts() {
        let (cfg, weights, witness) = tiny();
        let canon = build_commitments(&cfg, &weights, &witness);

        let mut rng = rand::rng();
        let mut oracle = RandomOracle::<EF>::new(&mut rng);
        let proof = prove(&canon.set, &cfg, &weights, &witness, &mut oracle);
        oracle.restart();
        assert!(verify(&canon.set, &cfg, &weights, &proof, &mut oracle));
    }
}
