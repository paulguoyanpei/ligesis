//! Residual-stream wiring (plan Phase 1).
//!
//! The LayerNorm input `x` is **virtual**: it is the residual stream, which is the committed block
//! output `x_out` of an earlier sub-block (or the public embedding `x0` at the very start). This
//! module provides
//!
//! 1. an **x-opener** (`open_x` / `reconstruct_x`) that resolves a LayerNorm input opening at a
//!    point into an opening of the committed `X_OUT` (or a public `x0` evaluation) — consumed by the
//!    LayerNorm bindings of Phase 2; and
//! 2. the **residual identity** binding (`prove_xout_residual` / `verify_xout_residual`): each
//!    committed `x_out` equals `x_in + projected + bias`, where `projected` is the committed
//!    attn-proj / fproj rescale quotient (`Q_APR` / `Q_FPR`) and `x_in` is the residual source. This
//!    is the binding that gives the committed `X_OUT` meaning; without it `x_out` is unconstrained.
//!
//! Instance maps follow `canonical.rs`: LayerNorm instances are `[ln1_0, ln2_0, …, lnf]` (`2ℓ` =
//! ln1, `2ℓ+1` = ln2, `2L` = lnf); `X_OUT` instances are `[attn_0, mlp_0, …]` (`2ℓ` = attn out,
//! `2ℓ+1` = mlp out). `X_OUT` and `A_LN` share the `S×D` element layout (column low, row high).

use p3_field::PrimeCharacteristicRing;
use utils::oracle::RandomOracle;
use utils::poly::MlPoly;

use crate::canonical::{self, Q_APR, Q_FPR, X_OUT};
use crate::commit::{instance_slice_point, ClaimAccumulator, CommitSet};
use crate::model::ModelWeights;
use crate::protocol::EF;
use crate::tensor::Matrix;
use crate::witness::{encode_i64, Config, Witness};

fn ef_i64(x: i64) -> EF {
    EF::from(encode_i64(x))
}

/// Where a residual-stream value comes from: a committed `X_OUT` instance, or the public `x0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidualSource {
    XOut(usize),
    X0,
}

/// The residual source feeding LayerNorm instance `ln_inst` (in the `Q_LN`/`A_LN`/`STD`/`VAR`
/// instance order). `2ℓ` = ln1 input = block input; `2ℓ+1` = ln2 input = attn sub-block out; `2L`
/// = lnf input = last block's mlp out.
pub fn ln_input_source(config: &Config, ln_inst: usize) -> ResidualSource {
    let l = config.n_layer;
    if ln_inst == 2 * l {
        return ResidualSource::XOut(2 * l - 1); // lnf ← last mlp out
    }
    let layer = ln_inst / 2;
    if ln_inst % 2 == 0 {
        // ln1 of `layer`: block input = prev block's mlp out, or x0 for layer 0.
        if layer == 0 {
            ResidualSource::X0
        } else {
            ResidualSource::XOut(2 * layer - 1)
        }
    } else {
        // ln2 of `layer`: attn sub-block out of this layer.
        ResidualSource::XOut(2 * layer)
    }
}

/// The residual source `x_in` added into `x_out` instance `xout_inst`. `2ℓ` (attn out) ← block
/// input (prev mlp out, or x0); `2ℓ+1` (mlp out) ← this layer's attn out.
pub fn xout_input_source(_config: &Config, xout_inst: usize) -> ResidualSource {
    let layer = xout_inst / 2;
    if xout_inst % 2 == 0 {
        if layer == 0 {
            ResidualSource::X0
        } else {
            ResidualSource::XOut(2 * layer - 1)
        }
    } else {
        ResidualSource::XOut(2 * layer)
    }
}

/// `x0` padded into the `S×D` element layout (column low, row high), as a low-bit-first MLE eval
/// table. Public — evaluated identically by prover and verifier.
fn x0_elem(x0: &Matrix, row_pow: usize, col_pow: usize) -> Vec<EF> {
    let mut out = vec![EF::ZERO; row_pow * col_pow];
    for r in 0..x0.rows() {
        for c in 0..x0.cols() {
            out[r * col_pow + c] = ef_i64(x0.get(r, c));
        }
    }
    out
}

/// A projection bias `bias[c]` broadcast over the **real** rows `< n_seq` (padding rows zeroed, so
/// the residual identity reads `0 = 0` there), in the `S×D` element layout.
fn proj_bias_elem(config: &Config, bias: &[i64], row_pow: usize, col_pow: usize) -> Vec<EF> {
    let mut out = vec![EF::ZERO; row_pow * col_pow];
    for r in 0..config.n_seq {
        for (c, &b) in bias.iter().enumerate() {
            out[r * col_pow + c] = ef_i64(b);
        }
    }
    out
}

/// Resolve a LayerNorm-input opening at element point `elem_point` for LayerNorm instance `ln_inst`
/// (prover side): returns `x(elem_point)` and, when the source is a committed `X_OUT` slice, emits
/// that opening claim. For the public `x0` source no claim is emitted (the verifier recomputes it).
pub fn open_x(
    set: &CommitSet,
    config: &Config,
    x0: &Matrix,
    elem_point: &[EF],
    ln_inst: usize,
    acc: &mut ClaimAccumulator,
) -> EF {
    let lay = canonical::layout_for(config, X_OUT);
    match ln_input_source(config, ln_inst) {
        ResidualSource::XOut(j) => {
            let pt = instance_slice_point(elem_point, j, lay.inst_vars());
            let v = set.prover_data(X_OUT).unwrap().clone().eval(&pt);
            acc.open(X_OUT, pt, v);
            v
        }
        ResidualSource::X0 => {
            MlPoly(x0_elem(x0, lay.row_pow, lay.col_pow)).eval(elem_point)
        }
    }
}

/// Verifier mirror of [`open_x`]: for a committed `X_OUT` source, emit the claim from `sent_val`
/// (the prover-supplied opening) and return it; for the public `x0` source, recompute it directly.
pub fn reconstruct_x(
    config: &Config,
    x0: &Matrix,
    elem_point: &[EF],
    ln_inst: usize,
    sent_val: EF,
    acc: &mut ClaimAccumulator,
) -> EF {
    let lay = canonical::layout_for(config, X_OUT);
    match ln_input_source(config, ln_inst) {
        ResidualSource::XOut(j) => {
            let pt = instance_slice_point(elem_point, j, lay.inst_vars());
            acc.open(X_OUT, pt, sent_val);
            sent_val
        }
        ResidualSource::X0 => MlPoly(x0_elem(x0, lay.row_pow, lay.col_pow)).eval(elem_point),
    }
}

/// The residual-identity binding proof: per `x_out` instance (in instance order) the openings of
/// `x_out`, its projection quotient (`Q_APR`/`Q_FPR`), and its committed residual input (the
/// `x_in[k]` entry is unused for `X0` sources, where the verifier recomputes from `x0`), all at one
/// shared fresh point.
pub struct XOutResidualProof {
    pub xout: Vec<EF>,
    pub proj: Vec<EF>,
    pub x_in: Vec<EF>,
}

impl XOutResidualProof {
    pub fn size_bytes(&self) -> usize {
        core::mem::size_of::<EF>() * (self.xout.len() + self.proj.len() + self.x_in.len())
    }
}

/// The projection quotient oracle + layer for `x_out` instance `k`.
fn proj_for(k: usize) -> (&'static str, usize) {
    let layer = k / 2;
    if k % 2 == 0 {
        (Q_APR, layer)
    } else {
        (Q_FPR, layer)
    }
}

/// Prove `x_out_k = x_in_k + projected_k + bias_k` for every committed `x_out` instance at one
/// shared fresh point, emitting all openings into `acc`.
pub fn prove_xout_residual(
    set: &CommitSet,
    config: &Config,
    _weights: &ModelWeights,
    witness: &Witness,
    acc: &mut ClaimAccumulator,
    oracle: &mut RandomOracle<EF>,
) -> XOutResidualProof {
    let lay = canonical::layout_for(config, X_OUT);
    let elem_vars = lay.elem_vars();
    let inst_vars = lay.inst_vars();
    let p = oracle.next_n_fields(elem_vars);
    let n = 2 * config.n_layer;
    let mut xout = Vec::with_capacity(n);
    let mut proj = Vec::with_capacity(n);
    let mut x_in = Vec::with_capacity(n);
    for k in 0..n {
        // x_out_k.
        let xpt = instance_slice_point(&p, k, inst_vars);
        let xv = set.prover_data(X_OUT).unwrap().clone().eval(&xpt);
        acc.open(X_OUT, xpt, xv);
        xout.push(xv);
        // projected_k = q_apr/q_fpr at the layer slice.
        let (poracle, layer) = proj_for(k);
        let player = canonical::layout_for(config, poracle);
        let ppt = instance_slice_point(&p, layer, player.inst_vars());
        let pv = set.prover_data(poracle).unwrap().clone().eval(&ppt);
        acc.open(poracle, ppt, pv);
        proj.push(pv);
        // x_in_k.
        let xin = open_xout_input(set, config, &witness.x0, &p, k, acc);
        x_in.push(xin);
    }
    XOutResidualProof { xout, proj, x_in }
}

/// Resolve the residual input `x_in` of `x_out` instance `k` at point `p` (prover side).
fn open_xout_input(
    set: &CommitSet,
    config: &Config,
    x0: &Matrix,
    p: &[EF],
    k: usize,
    acc: &mut ClaimAccumulator,
) -> EF {
    let lay = canonical::layout_for(config, X_OUT);
    match xout_input_source(config, k) {
        ResidualSource::XOut(j) => {
            let pt = instance_slice_point(p, j, lay.inst_vars());
            let v = set.prover_data(X_OUT).unwrap().clone().eval(&pt);
            acc.open(X_OUT, pt, v);
            v
        }
        ResidualSource::X0 => MlPoly(x0_elem(x0, lay.row_pow, lay.col_pow)).eval(p),
    }
}

/// Verify the residual-identity binding, emitting all openings into `acc`. Returns `false` if any
/// per-instance affine identity fails or the proof is the wrong shape.
pub fn verify_xout_residual(
    proof: &XOutResidualProof,
    config: &Config,
    weights: &ModelWeights,
    x0: &Matrix,
    acc: &mut ClaimAccumulator,
    oracle: &mut RandomOracle<EF>,
) -> bool {
    let lay = canonical::layout_for(config, X_OUT);
    let elem_vars = lay.elem_vars();
    let inst_vars = lay.inst_vars();
    let n = 2 * config.n_layer;
    if proof.xout.len() != n || proof.proj.len() != n || proof.x_in.len() != n {
        return false;
    }
    let p = oracle.next_n_fields(elem_vars);
    for k in 0..n {
        let xpt = instance_slice_point(&p, k, inst_vars);
        acc.open(X_OUT, xpt, proof.xout[k]);
        let (poracle, layer) = proj_for(k);
        let player = canonical::layout_for(config, poracle);
        let ppt = instance_slice_point(&p, layer, player.inst_vars());
        acc.open(poracle, ppt, proof.proj[k]);
        // x_in_k: committed slice (emit) or public x0 (recompute).
        let xin = match xout_input_source(config, k) {
            ResidualSource::XOut(j) => {
                let pt = instance_slice_point(&p, j, lay.inst_vars());
                acc.open(X_OUT, pt, proof.x_in[k]);
                proof.x_in[k]
            }
            ResidualSource::X0 => MlPoly(x0_elem(x0, lay.row_pow, lay.col_pow)).eval(&p),
        };
        // bias_k: public projection bias, real-rows-only.
        let layer = k / 2;
        let bias = if k % 2 == 0 {
            &weights.blocks[layer].attn_proj_b
        } else {
            &weights.blocks[layer].fproj_b
        };
        let bias_v = MlPoly(proj_bias_elem(config, bias, lay.row_pow, lay.col_pow)).eval(&p);
        if proof.xout[k] != xin + proof.proj[k] + bias_v {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::BlockWeights;

    /// A compact 2-layer model (column/feature sizes small) for exercising the residual chain.
    fn tiny_n(n_layer: usize) -> (Config, ModelWeights, Witness) {
        let cfg = Config {
            n_layer,
            n_seq: 3,
            n_head: 1,
            d_head: 2,
            d_model: 2,
            mlp_hidden: 3,
            vocab: 4,
            scale: 4,
            max_v: 8,
        };
        let exp_lut: Vec<i64> = (0..=(cfg.max_v * cfg.scale))
            .map(|i| if i == 0 { 0 } else { 1 })
            .collect();
        let gelu_lut: Vec<i64> = (0..=(2 * cfg.max_v * cfg.scale))
            .map(|i| i - cfg.max_v * cfg.scale)
            .collect();
        let block = || BlockWeights {
            ln_1_g: vec![4, 4],
            ln_1_b: vec![1, -1],
            attn_w: Matrix::new(2, 6, vec![1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]),
            attn_b: vec![0; 6],
            attn_proj_w: Matrix::new(2, 2, vec![1, 0, 0, 1]),
            attn_proj_b: vec![1, -1],
            ln_2_g: vec![4, 4],
            ln_2_b: vec![-1, 1],
            fc_w: Matrix::new(2, 3, vec![1, 0, 1, 0, 1, -1]),
            fc_b: vec![1, -1, 2],
            fproj_w: Matrix::new(3, 2, vec![1, 0, 0, 1, 1, -1]),
            fproj_b: vec![2, 0],
        };
        let weights = ModelWeights {
            wte: Matrix::new(4, 2, vec![1, 0, 0, 1, 1, 1, -1, 2]),
            wpe: Matrix::zeros(1024, 2),
            has_wpe: true,
            ln_f_g: vec![4, 4],
            ln_f_b: vec![0, 0],
            exp_lut,
            gelu_lut,
            blocks: (0..n_layer).map(|_| block()).collect(),
        };
        let witness = weights.forward(Matrix::new(3, 2, vec![1, -2, 3, 1, 0, 2]), &cfg);
        (cfg, weights, witness)
    }

    /// The padded LayerNorm input matrix in the `S×D` element layout for LN instance `ln_inst`.
    fn real_ln_input(witness: &Witness, config: &Config, ln_inst: usize) -> Vec<EF> {
        let m = if ln_inst == 2 * config.n_layer {
            &witness.lnf.input
        } else if ln_inst % 2 == 0 {
            &witness.blocks[ln_inst / 2].ln1.input
        } else {
            &witness.blocks[ln_inst / 2].ln2.input
        };
        let lay = canonical::layout_for(config, X_OUT);
        x0_elem(m, lay.row_pow, lay.col_pow)
    }

    #[test]
    fn x_opener_matches_layernorm_input() {
        let (cfg, _w, witness) = tiny_n(2);
        let canon = canonical::build_online(&cfg, &witness);
        let lay = canonical::layout_for(&cfg, X_OUT);
        let elem_point: Vec<EF> = (0..lay.elem_vars())
            .map(|i| EF::from(p3_goldilocks::Goldilocks::new((7 * i as u64 + 3) % 101)))
            .collect();
        // ln1_0 (x0), ln2_0 (attn out L0), ln1_1 (mlp out L0), ln2_1 (attn out L1), lnf.
        for ln_inst in 0..=2 * cfg.n_layer {
            let mut acc_p = ClaimAccumulator::new();
            let v = open_x(&canon.set, &cfg, &witness.x0, &elem_point, ln_inst, &mut acc_p);
            let expected = MlPoly(real_ln_input(&witness, &cfg, ln_inst)).eval(&elem_point);
            assert_eq!(v, expected, "ln_inst {ln_inst}");
            // verifier mirror with the prover's sent value reproduces the same value.
            let mut acc_v = ClaimAccumulator::new();
            let vv = reconstruct_x(&cfg, &witness.x0, &elem_point, ln_inst, v, &mut acc_v);
            assert_eq!(vv, v, "reconstruct ln_inst {ln_inst}");
        }
    }

    fn xout_residual_roundtrip(n_layer: usize) {
        let (cfg, weights, witness) = tiny_n(n_layer);
        let canon = canonical::build_online(&cfg, &witness);
        let mut rng = rand::rng();
        let mut oracle = RandomOracle::<EF>::new(&mut rng);
        let mut acc_p = ClaimAccumulator::new();
        let proof = prove_xout_residual(&canon.set, &cfg, &weights, &witness, &mut acc_p, &mut oracle);
        oracle.restart();
        let mut acc_v = ClaimAccumulator::new();
        assert!(verify_xout_residual(
            &proof, &cfg, &weights, &witness.x0, &mut acc_v, &mut oracle
        ));
    }

    #[test]
    fn xout_residual_accepts_single_layer() {
        xout_residual_roundtrip(1); // exercises the x0 path
    }

    #[test]
    fn xout_residual_accepts_multi_layer() {
        xout_residual_roundtrip(2); // exercises the cross-block chain (size-generic K>1)
    }

    #[test]
    fn tampered_xout_rejected() {
        let (cfg, weights, witness) = tiny_n(2);
        let canon = canonical::build_online(&cfg, &witness);
        let mut rng = rand::rng();
        let mut oracle = RandomOracle::<EF>::new(&mut rng);
        let mut acc_p = ClaimAccumulator::new();
        let mut proof =
            prove_xout_residual(&canon.set, &cfg, &weights, &witness, &mut acc_p, &mut oracle);
        proof.xout[1] += EF::ONE; // break one residual identity
        oracle.restart();
        let mut acc_v = ClaimAccumulator::new();
        assert!(!verify_xout_residual(
            &proof, &cfg, &weights, &witness.x0, &mut acc_v, &mut oracle
        ));
    }
}
