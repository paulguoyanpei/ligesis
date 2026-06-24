//! Sum-over-features / `var_sum` + `sum_exp` bindings (plan Phase 2).
//!
//! These bind the committed LayerNorm `var`/`a_ln` and softmax `sum_exp` to the underlying input,
//! which the earlier reductions never did. Three relations, each **nonlinear**, so each rides a
//! product sumcheck (a naive random-point MLE check is unsound — the squared / public-gain product
//! terms are not multilinear and only hold pointwise on the boolean cube):
//!
//! - **var_sum**: `var_sum[i,r] = D·Σ_c x[i,r,c]² − (Σ_c x[i,r,c])²`. The `Σx²` term is a batched
//!   self-product sumcheck over the gathered LN-input poly; the `(Σx)²` term is a batched
//!   self-product sumcheck over the gathered per-row sums (`sum_x`, virtual via the column
//!   half-point); `var` is a direct opening. Batched over the 25 LayerNorm instances by a γ-RLC.
//! - **a_ln**: `a_ln[i,r,c] = gain[c]·(D·x[i,r,c] − sum_x[i,r])`, a batched product sumcheck with
//!   the public per-instance `gain` as one factor.
//! - **sum_exp**: `sum_exp[i,r] = Σ_k exp[i,r,k]`, the key-column half-point of the committed `exp`,
//!   bound on the **real rows only** (the committed `sum_exp` pads masked rows with 1 while `exp`
//!   pads with 0).
//!
//! The LayerNorm input `x` is the residual stream: every reduced x-opening is resolved through the
//! Phase-1 [`crate::residual`] sources (committed `X_OUT` slices, or the public `x0`), combined
//! across instances by `eq`-weights (the gathered polys are materialized only by the prover).

use p3_field::{Field, PrimeCharacteristicRing};
use utils::oracle::RandomOracle;
use utils::poly::MlPoly;
use utils::sumcheck::{self, SumcheckProof};

use crate::canonical::{self, A_LN, EXP, SUM_EXP, VAR, X_OUT};
use crate::commit::{eq_at_index, instance_slice_point, ClaimAccumulator, CommitSet};
use crate::model::ModelWeights;
use crate::protocol::EF;
use crate::reduce::low_first;
use crate::residual::{ln_input_source, ResidualSource};
use crate::tensor::Matrix;
use crate::witness::{encode_i64, Config, Witness};

fn ef_i64(x: i64) -> EF {
    EF::from(encode_i64(x))
}
fn half() -> EF {
    (EF::ONE + EF::ONE).inverse()
}

/// `x0` padded into the `S×D` element layout (column low, row high) as an MLE eval table.
fn x0_elem(x0: &Matrix, row_pow: usize, col_pow: usize) -> Vec<EF> {
    let mut out = vec![EF::ZERO; row_pow * col_pow];
    for r in 0..x0.rows() {
        for c in 0..x0.cols() {
            out[r * col_pow + c] = ef_i64(x0.get(r, c));
        }
    }
    out
}

/// The per-LayerNorm-instance gain vector (`ln_1_g`/`ln_2_g`/`ln_f_g`).
fn gain_for<'a>(config: &Config, weights: &'a ModelWeights, ln_inst: usize) -> &'a [i64] {
    if ln_inst == 2 * config.n_layer {
        &weights.ln_f_g
    } else if ln_inst % 2 == 0 {
        &weights.blocks[ln_inst / 2].ln_1_g
    } else {
        &weights.blocks[ln_inst / 2].ln_2_g
    }
}

/// The LayerNorm input element block (column low, row high) for instance `ln_inst`: a committed
/// `X_OUT` slice, or the public `x0`. (Both equal the real LN input by the Phase-1 residual
/// identity.)
fn ln_input_block(set: &CommitSet, config: &Config, x0: &Matrix, ln_inst: usize) -> Vec<EF> {
    let lay = canonical::layout_for(config, X_OUT);
    let elem_len = lay.row_pow * lay.col_pow;
    match ln_input_source(config, ln_inst) {
        ResidualSource::XOut(j) => {
            set.prover_data(X_OUT).unwrap().0[j * elem_len..(j + 1) * elem_len].to_vec()
        }
        ResidualSource::X0 => x0_elem(x0, lay.row_pow, lay.col_pow),
    }
}

/// Prover: the per-instance x-input openings `x_input_i(elem_point)` (len `2L+1`), emitting an
/// `X_OUT` claim for each committed source (the `x0` source emits none — the verifier recomputes).
fn combine_x_prove(
    set: &CommitSet,
    config: &Config,
    x0: &Matrix,
    elem_point: &[EF],
    acc: &mut ClaimAccumulator,
) -> Vec<EF> {
    let lay = canonical::layout_for(config, X_OUT);
    let inst_vars = lay.inst_vars();
    let x0e = x0_elem(x0, lay.row_pow, lay.col_pow);
    (0..=2 * config.n_layer)
        .map(|i| match ln_input_source(config, i) {
            ResidualSource::XOut(j) => {
                let pt = instance_slice_point(elem_point, j, inst_vars);
                let v = set.prover_data(X_OUT).unwrap().clone().eval(&pt);
                acc.open(X_OUT, pt, v);
                v
            }
            ResidualSource::X0 => MlPoly(x0e.clone()).eval(elem_point),
        })
        .collect()
}

/// Verifier: emit the `X_OUT` claims from the prover's `sent` values, check each `x0` source against
/// the public embedding, and return `Σ_i eq(inst_point,i)·x_input_i(elem_point)`. `None` if an `x0`
/// value disagrees or `sent` is the wrong length.
fn combine_x_verify(
    config: &Config,
    x0: &Matrix,
    elem_point: &[EF],
    inst_point: &[EF],
    sent: &[EF],
    acc: &mut ClaimAccumulator,
) -> Option<EF> {
    if sent.len() != 2 * config.n_layer + 1 {
        return None;
    }
    let lay = canonical::layout_for(config, X_OUT);
    let inst_vars = lay.inst_vars();
    let x0e = x0_elem(x0, lay.row_pow, lay.col_pow);
    let mut acc_val = EF::ZERO;
    for (i, &v) in sent.iter().enumerate() {
        match ln_input_source(config, i) {
            ResidualSource::XOut(j) => {
                let pt = instance_slice_point(elem_point, j, inst_vars);
                acc.open(X_OUT, pt, v);
            }
            ResidualSource::X0 => {
                if v != MlPoly(x0e.clone()).eval(elem_point) {
                    return None;
                }
            }
        }
        acc_val += eq_at_index(inst_point, i) * v;
    }
    Some(acc_val)
}

pub struct SumFeatProof {
    // var_sum (Σx² and (Σx)² sumchecks, batched over LN instances by γ).
    pub q2_claim: EF,
    pub q2: SumcheckProof<EF>,
    pub q2_x: Vec<EF>,
    pub sq_claim: EF,
    pub sq: SumcheckProof<EF>,
    pub sq_x: Vec<EF>,
    pub var_v: Vec<EF>,
    // a_ln (batched product sumcheck with public gain).
    pub aln_claim: EF,
    pub aln: SumcheckProof<EF>,
    pub aln_x: Vec<EF>,
    pub aln_sumx: Vec<EF>,
    pub aln_v: Vec<EF>,
    // sum_exp (key half-point, real rows only).
    pub se: SumcheckProof<EF>,
    pub se_sum: EF,
    pub se_exp: EF,
}

impl SumFeatProof {
    pub fn size_bytes(&self) -> usize {
        let ef = core::mem::size_of::<EF>();
        let v = |x: &Vec<EF>| ef * x.len();
        2 * ef
            + self.q2.size_bytes()
            + self.sq.size_bytes()
            + v(&self.q2_x)
            + v(&self.sq_x)
            + v(&self.var_v)
            + ef
            + self.aln.size_bytes()
            + v(&self.aln_x)
            + v(&self.aln_sumx)
            + v(&self.aln_v)
            + self.se.size_bytes()
            + 2 * ef
    }
}

/// Gather the 25 LN inputs into one `(col ++ row ++ inst)` poly (instance high), padded.
fn gather_x_ln(set: &CommitSet, config: &Config, x0: &Matrix) -> Vec<EF> {
    let lay = canonical::layout_for(config, X_OUT);
    let elem_len = lay.row_pow * lay.col_pow;
    let inst_pow = 1usize << layout_inst_vars(config);
    let mut out = vec![EF::ZERO; inst_pow * elem_len];
    for i in 0..=2 * config.n_layer {
        let block = ln_input_block(set, config, x0, i);
        out[i * elem_len..(i + 1) * elem_len].copy_from_slice(&block);
    }
    out
}

/// Per-row column sums of the gathered LN input: `sum_x[i,r] = Σ_c x[i,r,c]`, a `(row ++ inst)` poly.
fn gather_sum_x(x_ln: &[EF], config: &Config) -> Vec<EF> {
    let lay = canonical::layout_for(config, X_OUT);
    let (row_pow, col_pow) = (lay.row_pow, lay.col_pow);
    let elem_len = row_pow * col_pow;
    let inst_pow = 1usize << layout_inst_vars(config);
    let mut out = vec![EF::ZERO; inst_pow * row_pow];
    for i in 0..inst_pow {
        for r in 0..row_pow {
            let mut s = EF::ZERO;
            for c in 0..col_pow {
                s += x_ln[i * elem_len + r * col_pow + c];
            }
            out[i * row_pow + r] = s;
        }
    }
    out
}

/// The instance-var count shared by the LayerNorm element layout (`A_LN`/`VAR` agree).
fn layout_inst_vars(config: &Config) -> usize {
    canonical::layout_for(config, A_LN).inst_vars()
}

pub fn prove_sumfeat(
    set: &CommitSet,
    config: &Config,
    weights: &ModelWeights,
    witness: &Witness,
    acc: &mut ClaimAccumulator,
    oracle: &mut RandomOracle<EF>,
) -> SumFeatProof {
    let x0 = &witness.x0;
    let aln = canonical::layout_for(config, A_LN);
    let (col_vars, row_vars) = (aln.col_vars(), aln.row_vars());
    let elem_vars = col_vars + row_vars;
    let inst_vars = aln.inst_vars();
    let inst_pow = 1usize << inst_vars;
    let d = ef_i64(config.d_model as i64);
    let half_col = vec![half(); col_vars];

    let x_ln = gather_x_ln(set, config, x0);
    let sum_x = gather_sum_x(&x_ln, config);

    // ---- var_sum: Σx² and (Σx)² batched over instances by γ_var ----
    let gamma_v = oracle.next_field();
    let rho_row = oracle.next_n_fields(row_vars);

    // Σx² sumcheck: weight eq(rho_row,r)·γ^i broadcast over col, factors [W, x, x].
    let wq2 = weight_eqrow_gamma(&rho_row, gamma_v, col_vars, row_vars, inst_pow, true);
    let q2_claim = weighted_self_prod(&wq2, &x_ln);
    let (q2, sc) = sumcheck::prove(vec![wq2, x_ln.clone(), x_ln.clone()], oracle);
    let open = low_first(&sc); // (col ++ row ++ inst)
    let q2_x = combine_x_prove(set, config, x0, &open[..elem_vars], acc);

    // (Σx)² sumcheck: weight eq(rho_row,r)·γ^i over (row ++ inst), factors [W, sum_x, sum_x].
    let wsq = weight_eqrow_gamma(&rho_row, gamma_v, 0, row_vars, inst_pow, true);
    let sq_claim = weighted_self_prod(&wsq, &sum_x);
    let (sq, scs) = sumcheck::prove(vec![wsq, sum_x.clone(), sum_x.clone()], oracle);
    let opens = low_first(&scs); // (row ++ inst)
    // sum_x(open) = col_pow · x(half_col ++ row_open); emit x at the half-point.
    let sq_half_pt = [half_col.clone(), opens[..row_vars].to_vec()].concat();
    let sq_x = combine_x_prove(set, config, x0, &sq_half_pt, acc);

    // var openings VAR(rho_row ++ inst_i).
    let var_v = open_var(set, config, &rho_row, acc);

    // ---- a_ln: batched product sumcheck with public gain ----
    let gamma_a = oracle.next_field();
    let rho_a = oracle.next_n_fields(elem_vars); // (col ++ row)
    let gain_ln = gather_gain(config, weights);
    let g_ln = build_g(&x_ln, &sum_x, config, d);
    let wa = weight_eqelem_gamma(&rho_a, gamma_a, elem_vars, inst_pow);
    let aln_v = open_a_ln_emit(set, config, &rho_a, acc);
    let mut aln_claim = EF::ZERO;
    let mut ga = EF::ONE;
    for &v in &aln_v {
        aln_claim += ga * v;
        ga *= gamma_a;
    }
    let (aln, sca) = sumcheck::prove(vec![wa, gain_ln, g_ln], oracle);
    let opena = low_first(&sca); // (col ++ row ++ inst)
    let aln_x = combine_x_prove(set, config, x0, &opena[..elem_vars], acc);
    let aln_half_pt = [half_col.clone(), opena[col_vars..elem_vars].to_vec()].concat();
    let aln_sumx = combine_x_prove(set, config, x0, &aln_half_pt, acc);

    // ---- sum_exp: key half-point, real rows only ----
    let (se, se_sum, se_exp) = prove_sum_exp(set, config, acc, oracle);

    SumFeatProof {
        q2_claim,
        q2,
        q2_x,
        sq_claim,
        sq,
        sq_x,
        var_v,
        aln_claim,
        aln,
        aln_x,
        aln_sumx,
        aln_v,
        se,
        se_sum,
        se_exp,
    }
}

/// Emit `A_LN(rho_a ++ inst_i)` openings and return their values (len `2L+1`).
fn open_a_ln_emit(
    set: &CommitSet,
    config: &Config,
    rho_a: &[EF],
    acc: &mut ClaimAccumulator,
) -> Vec<EF> {
    let inst_vars = layout_inst_vars(config);
    (0..=2 * config.n_layer)
        .map(|i| {
            let pt = instance_slice_point(rho_a, i, inst_vars);
            let v = set.prover_data(A_LN).unwrap().clone().eval(&pt);
            acc.open(A_LN, pt, v);
            v
        })
        .collect()
}

/// Emit `VAR(rho_row ++ inst_i)` openings and return their values (len `2L+1`).
fn open_var(
    set: &CommitSet,
    config: &Config,
    rho_row: &[EF],
    acc: &mut ClaimAccumulator,
) -> Vec<EF> {
    let inst_vars = canonical::layout_for(config, VAR).inst_vars();
    (0..=2 * config.n_layer)
        .map(|i| {
            let pt = instance_slice_point(rho_row, i, inst_vars);
            let v = set.prover_data(VAR).unwrap().clone().eval(&pt);
            acc.open(VAR, pt, v);
            v
        })
        .collect()
}

/// `gain_i[c]` over `(col ++ row ++ inst)`, broadcast over rows (padding cols/rows/insts = 0).
fn gather_gain(config: &Config, weights: &ModelWeights) -> Vec<EF> {
    let aln = canonical::layout_for(config, A_LN);
    let (row_pow, col_pow) = (aln.row_pow, aln.col_pow);
    let elem_len = row_pow * col_pow;
    let inst_pow = 1usize << aln.inst_vars();
    let mut out = vec![EF::ZERO; inst_pow * elem_len];
    for i in 0..=2 * config.n_layer {
        let gain = gain_for(config, weights, i);
        for r in 0..row_pow {
            for (c, &gc) in gain.iter().enumerate() {
                out[i * elem_len + r * col_pow + c] = ef_i64(gc);
            }
        }
    }
    out
}

/// `g[i,r,c] = D·x[i,r,c] − sum_x[i,r]` over `(col ++ row ++ inst)`.
fn build_g(x_ln: &[EF], sum_x: &[EF], config: &Config, d: EF) -> Vec<EF> {
    let aln = canonical::layout_for(config, A_LN);
    let (row_pow, col_pow) = (aln.row_pow, aln.col_pow);
    let elem_len = row_pow * col_pow;
    let inst_pow = 1usize << aln.inst_vars();
    let mut out = vec![EF::ZERO; inst_pow * elem_len];
    for i in 0..inst_pow {
        for r in 0..row_pow {
            let sx = sum_x[i * row_pow + r];
            for c in 0..col_pow {
                out[i * elem_len + r * col_pow + c] = d * x_ln[i * elem_len + r * col_pow + c] - sx;
            }
        }
    }
    out
}

/// Weight `eq(rho_row, r) · γ^i` over `(col ++ row ++ inst)` (col broadcast). `col_vars = 0` builds
/// the `(row ++ inst)` variant.
fn weight_eqrow_gamma(
    rho_row: &[EF],
    gamma: EF,
    col_vars: usize,
    row_vars: usize,
    inst_pow: usize,
    _batched: bool,
) -> Vec<EF> {
    let col_pow = 1usize << col_vars;
    let row_pow = 1usize << row_vars;
    let elem_len = col_pow * row_pow;
    let mut out = vec![EF::ZERO; inst_pow * elem_len];
    let mut g = EF::ONE;
    for i in 0..inst_pow {
        for r in 0..row_pow {
            let er = eq_at_index(rho_row, r);
            for c in 0..col_pow {
                out[i * elem_len + r * col_pow + c] = er * g;
            }
        }
        g *= gamma;
    }
    out
}

/// Weight `eq(rho_elem, elem) · γ^i` over `(elem ++ inst)`.
fn weight_eqelem_gamma(rho_elem: &[EF], gamma: EF, elem_vars: usize, inst_pow: usize) -> Vec<EF> {
    let elem_len = 1usize << elem_vars;
    let mut out = vec![EF::ZERO; inst_pow * elem_len];
    let mut g = EF::ONE;
    for i in 0..inst_pow {
        for e in 0..elem_len {
            out[i * elem_len + e] = eq_at_index(rho_elem, e) * g;
        }
        g *= gamma;
    }
    out
}

/// `Σ_x w[x]·p[x]²`.
fn weighted_self_prod(w: &[EF], p: &[EF]) -> EF {
    w.iter().zip(p).fold(EF::ZERO, |a, (&wi, &pi)| a + wi * pi * pi)
}

/// `Σ_k exp[i,r,k]` half-point binding (real rows only). Returns the sumcheck plus the SUM_EXP and
/// EXP openings at its reduced point.
fn prove_sum_exp(
    set: &CommitSet,
    config: &Config,
    acc: &mut ClaimAccumulator,
    oracle: &mut RandomOracle<EF>,
) -> (SumcheckProof<EF>, EF, EF) {
    let exp_lay = canonical::layout_for(config, EXP);
    let col_vars = exp_lay.col_vars();
    let row_vars = exp_lay.row_vars();
    let inst_vars = exp_lay.inst_vars();
    let row_pow = exp_lay.row_pow;
    let inst_pow = 1usize << inst_vars;
    let col_pow_exp = ef_i64(exp_lay.col_pow as i64);
    let half_col = vec![half(); col_vars];

    // EXP folded at the key half-point → (row ++ inst).
    let mut exp_half = MlPoly(set.prover_data(EXP).unwrap().0.clone());
    exp_half.fold(&half_col);
    let exp_half = exp_half.0;
    let sum_exp = set.prover_data(SUM_EXP).unwrap().0.clone();
    // D = SUM_EXP − col_pow·EXP_half; weight = realrow(r)·γ^idx.
    let d_poly: Vec<EF> = (0..inst_pow * row_pow)
        .map(|idx| sum_exp[idx] - col_pow_exp * exp_half[idx])
        .collect();
    let gamma = oracle.next_field();
    let w = realrow_gamma(config, gamma, row_pow, inst_pow);
    let (se, sc) = sumcheck::prove(vec![w, d_poly], oracle);
    let open = low_first(&sc); // (row ++ inst)
    // SUM_EXP at open; EXP at (half_col ++ open).
    let se_sum = set.prover_data(SUM_EXP).unwrap().clone().eval(&open);
    acc.open(SUM_EXP, open.clone(), se_sum);
    let exp_pt = [half_col.clone(), open.clone()].concat();
    let se_exp = set.prover_data(EXP).unwrap().clone().eval(&exp_pt);
    acc.open(EXP, exp_pt, se_exp);
    let _ = (col_vars, row_vars);
    (se, se_sum, se_exp)
}

/// `realrow(r)·realinst(i)·γ^idx` over `(row ++ inst)`, zero on padding rows `r ≥ n_seq` AND on
/// padding head instances `i ≥ n_layer·n_head`. The latter matters because `SUM_EXP` is committed
/// padded with **1** on both the row and instance axes (a valid sum), while `exp` pads with 0; the
/// binding must be enforced only where both are real.
fn realrow_gamma(config: &Config, gamma: EF, row_pow: usize, inst_pow: usize) -> Vec<EF> {
    let n_heads = config.n_layer * config.n_head;
    let mut out = vec![EF::ZERO; inst_pow * row_pow];
    let mut g = EF::ONE;
    for idx in 0..inst_pow * row_pow {
        let r = idx % row_pow;
        let i = idx / row_pow;
        if r < config.n_seq && i < n_heads {
            out[idx] = g;
        }
        g *= gamma;
    }
    out
}

pub fn verify_sumfeat(
    proof: &SumFeatProof,
    set: &CommitSet,
    config: &Config,
    weights: &ModelWeights,
    x0: &Matrix,
    acc: &mut ClaimAccumulator,
    oracle: &mut RandomOracle<EF>,
) -> bool {
    let _ = set;
    let aln = canonical::layout_for(config, A_LN);
    let (col_vars, row_vars) = (aln.col_vars(), aln.row_vars());
    let elem_vars = col_vars + row_vars;
    let inst_vars = aln.inst_vars();
    let inst_pow = 1usize << inst_vars;
    let col_pow = ef_i64(aln.col_pow as i64);
    let d = ef_i64(config.d_model as i64);
    let half_col = vec![half(); col_vars];
    let n = 2 * config.n_layer + 1;
    if proof.var_v.len() != n || proof.aln_v.len() != n {
        return false;
    }

    // ---- var_sum ----
    let gamma_v = oracle.next_field();
    let rho_row = oracle.next_n_fields(row_vars);
    // Σx² sumcheck.
    let Some(sc) = sumcheck::verify(proof.q2_claim, &proof.q2, oracle) else {
        return false;
    };
    if proof.q2.final_evals.len() != 3 {
        return false;
    }
    let open = low_first(&sc);
    let wq2 = weight_eqrow_gamma(&rho_row, gamma_v, col_vars, row_vars, inst_pow, true);
    if proof.q2.final_evals[0] != MlPoly(wq2).eval(&open) {
        return false;
    }
    let Some(xo) = combine_x_verify(config, x0, &open[..elem_vars], &open[elem_vars..], &proof.q2_x, acc)
    else {
        return false;
    };
    if proof.q2.final_evals[1] != xo || proof.q2.final_evals[2] != xo {
        return false;
    }
    // (Σx)² sumcheck.
    let Some(scs) = sumcheck::verify(proof.sq_claim, &proof.sq, oracle) else {
        return false;
    };
    if proof.sq.final_evals.len() != 3 {
        return false;
    }
    let opens = low_first(&scs);
    let wsq = weight_eqrow_gamma(&rho_row, gamma_v, 0, row_vars, inst_pow, true);
    if proof.sq.final_evals[0] != MlPoly(wsq).eval(&opens) {
        return false;
    }
    let sq_half_pt = [half_col.clone(), opens[..row_vars].to_vec()].concat();
    let Some(sxh) = combine_x_verify(config, x0, &sq_half_pt, &opens[row_vars..], &proof.sq_x, acc)
    else {
        return false;
    };
    let sum_x_open = col_pow * sxh;
    if proof.sq.final_evals[1] != sum_x_open || proof.sq.final_evals[2] != sum_x_open {
        return false;
    }
    // var openings + batched identity: Σγ^i·var = D·q2_claim − sq_claim.
    let mut lhs = EF::ZERO;
    let mut g = EF::ONE;
    let var_inst_vars = canonical::layout_for(config, VAR).inst_vars();
    for (i, &vv) in proof.var_v.iter().enumerate() {
        let pt = instance_slice_point(&rho_row, i, var_inst_vars);
        acc.open(VAR, pt, vv);
        lhs += g * vv;
        g *= gamma_v;
    }
    if lhs != d * proof.q2_claim - proof.sq_claim {
        return false;
    }

    // ---- a_ln ----
    let gamma_a = oracle.next_field();
    let rho_a = oracle.next_n_fields(elem_vars);
    // claim = Σγ^i·A_LN(rho_a ++ inst_i); emit openings.
    let mut claim = EF::ZERO;
    let mut g = EF::ONE;
    for (i, &av) in proof.aln_v.iter().enumerate() {
        let pt = instance_slice_point(&rho_a, i, inst_vars);
        acc.open(A_LN, pt, av);
        claim += g * av;
        g *= gamma_a;
    }
    if claim != proof.aln_claim {
        return false;
    }
    let Some(sca) = sumcheck::verify(proof.aln_claim, &proof.aln, oracle) else {
        return false;
    };
    if proof.aln.final_evals.len() != 3 {
        return false;
    }
    let opena = low_first(&sca);
    let wa = weight_eqelem_gamma(&rho_a, gamma_a, elem_vars, inst_pow);
    if proof.aln.final_evals[0] != MlPoly(wa).eval(&opena) {
        return false;
    }
    let gain_ln = gather_gain(config, weights);
    if proof.aln.final_evals[1] != MlPoly(gain_ln).eval(&opena) {
        return false;
    }
    let Some(ax) = combine_x_verify(config, x0, &opena[..elem_vars], &opena[elem_vars..], &proof.aln_x, acc)
    else {
        return false;
    };
    let aln_half_pt = [half_col.clone(), opena[col_vars..elem_vars].to_vec()].concat();
    let Some(asx) = combine_x_verify(config, x0, &aln_half_pt, &opena[elem_vars..], &proof.aln_sumx, acc)
    else {
        return false;
    };
    let g_open = d * ax - col_pow * asx;
    if proof.aln.final_evals[2] != g_open {
        return false;
    }

    // ---- sum_exp ----
    verify_sum_exp(proof, set, config, acc, oracle)
}

fn verify_sum_exp(
    proof: &SumFeatProof,
    set: &CommitSet,
    config: &Config,
    acc: &mut ClaimAccumulator,
    oracle: &mut RandomOracle<EF>,
) -> bool {
    let _ = set;
    let exp_lay = canonical::layout_for(config, EXP);
    let col_vars = exp_lay.col_vars();
    let row_vars = exp_lay.row_vars();
    let inst_vars = exp_lay.inst_vars();
    let row_pow = exp_lay.row_pow;
    let inst_pow = 1usize << inst_vars;
    let col_pow_exp = ef_i64(exp_lay.col_pow as i64);
    let half_col = vec![half(); col_vars];
    let _ = row_vars;

    let gamma = oracle.next_field();
    let Some(sc) = sumcheck::verify(EF::ZERO, &proof.se, oracle) else {
        return false;
    };
    if proof.se.final_evals.len() != 2 {
        return false;
    }
    let open = low_first(&sc);
    let w = realrow_gamma(config, gamma, row_pow, inst_pow);
    if proof.se.final_evals[0] != MlPoly(w).eval(&open) {
        return false;
    }
    // D(open) = SUM_EXP(open) − col_pow·EXP(half ++ open).
    let d_open = proof.se_sum - col_pow_exp * proof.se_exp;
    if proof.se.final_evals[1] != d_open {
        return false;
    }
    acc.open(SUM_EXP, open.clone(), proof.se_sum);
    let exp_pt = [half_col, open].concat();
    acc.open(EXP, exp_pt, proof.se_exp);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::BlockWeights;

    /// A config with `d_model = 6` (so `col_pow = 8 ≠ 6 = D`, the col_pow-vs-D regression) and a
    /// non-power-of-two `n_seq` (padding rows, exercising the sum_exp real-rows-only weight).
    /// `d_model = n_head·d_head = 6`, `mlp_hidden = 5` (also non-power-of-two).
    fn tiny_n(n_layer: usize) -> (Config, ModelWeights, Witness) {
        // n_head=3 ⇒ heads ∈ {3,6}, non-power-of-two, so the softmax instance axis HAS padding
        // instances (SUM_EXP pads them with 1) — the regression guard for the real-rows-AND-insts
        // weight. d_model = n_head·d_head = 6.
        let cfg = Config {
            n_layer,
            n_seq: 3,
            n_head: 3,
            d_head: 2,
            d_model: 6,
            mlp_hidden: 5,
            vocab: 4,
            scale: 4,
            max_v: 16,
        };
        let exp_lut: Vec<i64> = (0..=(cfg.max_v * cfg.scale))
            .map(|i| if i == 0 { 0 } else { 1 })
            .collect();
        let gelu_lut: Vec<i64> = (0..=(2 * cfg.max_v * cfg.scale))
            .map(|i| i - cfg.max_v * cfg.scale)
            .collect();
        // Deterministic small weights at the exact shapes.
        let mat = |rows: usize, cols: usize, seed: i64| {
            Matrix::new(
                rows,
                cols,
                (0..rows * cols).map(|i| ((i as i64 * seed + 1) % 3) - 1).collect(),
            )
        };
        let block = || BlockWeights {
            ln_1_g: vec![4, 3, 5, 4, 3, 5],
            ln_1_b: vec![1, -1, 0, 1, 0, -1],
            attn_w: mat(6, 18, 1),
            attn_b: vec![0; 18],
            attn_proj_w: mat(6, 6, 2),
            attn_proj_b: vec![1, -1, 0, 1, 0, -1],
            ln_2_g: vec![4, 4, 3, 4, 4, 3],
            ln_2_b: vec![-1, 1, 0, -1, 1, 0],
            fc_w: mat(6, 5, 1),
            fc_b: vec![1, -1, 2, 0, 1],
            fproj_w: mat(5, 6, 2),
            fproj_b: vec![2, 0, -1, 1, 0, -1],
        };
        let weights = ModelWeights {
            wte: mat(4, 6, 1),
            wpe: Matrix::zeros(1024, 6),
            has_wpe: true,
            ln_f_g: vec![4, 4, 4, 4, 4, 4],
            ln_f_b: vec![0, 0, 0, 0, 0, 0],
            exp_lut,
            gelu_lut,
            blocks: (0..n_layer).map(|_| block()).collect(),
        };
        let x0 = Matrix::new(3, 6, (0..18).map(|i| ((i as i64 * 2) % 5) - 2).collect());
        let witness = weights.forward(x0, &cfg);
        (cfg, weights, witness)
    }

    fn roundtrip(n_layer: usize) {
        let (cfg, weights, witness) = tiny_n(n_layer);
        let canon = canonical::build_online(&cfg, &witness);
        let mut rng = rand::rng();
        let mut oracle = RandomOracle::<EF>::new(&mut rng);
        let mut acc_p = ClaimAccumulator::new();
        let proof = prove_sumfeat(&canon.set, &cfg, &weights, &witness, &mut acc_p, &mut oracle);
        oracle.restart();
        let mut acc_v = ClaimAccumulator::new();
        assert!(verify_sumfeat(
            &proof, &canon.set, &cfg, &weights, &witness.x0, &mut acc_v, &mut oracle
        ));
        // Every emitted claim must be a true opening of its committed poly.
        for c in &acc_v.claims {
            let got = canon.set.prover_data(&c.oracle).unwrap().clone().eval(&c.point);
            assert_eq!(got, c.value, "claim on {} mismatched", c.oracle);
        }
    }

    #[test]
    fn sumfeat_accepts_single_layer() {
        roundtrip(1);
    }

    #[test]
    fn sumfeat_accepts_multi_layer() {
        roundtrip(2);
    }

    #[test]
    fn tampered_var_rejected() {
        let (cfg, weights, witness) = tiny_n(2);
        let canon = canonical::build_online(&cfg, &witness);
        let mut rng = rand::rng();
        let mut oracle = RandomOracle::<EF>::new(&mut rng);
        let mut acc_p = ClaimAccumulator::new();
        let mut proof =
            prove_sumfeat(&canon.set, &cfg, &weights, &witness, &mut acc_p, &mut oracle);
        proof.var_v[1] += EF::ONE;
        oracle.restart();
        let mut acc_v = ClaimAccumulator::new();
        assert!(!verify_sumfeat(
            &proof, &canon.set, &cfg, &weights, &witness.x0, &mut acc_v, &mut oracle
        ));
    }

    #[test]
    fn tampered_sum_exp_rejected() {
        let (cfg, weights, witness) = tiny_n(2);
        let canon = canonical::build_online(&cfg, &witness);
        let mut rng = rand::rng();
        let mut oracle = RandomOracle::<EF>::new(&mut rng);
        let mut acc_p = ClaimAccumulator::new();
        let mut proof =
            prove_sumfeat(&canon.set, &cfg, &weights, &witness, &mut acc_p, &mut oracle);
        proof.se_sum += EF::ONE;
        oracle.restart();
        let mut acc_v = ClaimAccumulator::new();
        assert!(!verify_sumfeat(
            &proof, &canon.set, &cfg, &weights, &witness.x0, &mut acc_v, &mut oracle
        ));
    }
}
