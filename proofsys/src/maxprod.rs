//! Softmax-max grand-product binding (plan Phase 3).
//!
//! Proves, per (head, row): `∏_k ( M[r,k]·(x_max[i,r] − scores[i,r,k]) + (1−M[r,k]) ) = 0`, where
//! `scores = Q_SC` (the committed rescale quotient), `x_max = X_MAX`, and `M` is the public causal
//! lower-triangular mask. The product being 0 forces `x_max` to equal some unmasked score; combined
//! with `x_max ≥ all scores` (the exp-index range row, Phase 4 / the existing softmax range check),
//! `x_max` is the row max.
//!
//! The per-(head,row) products are batched into ONE [`crate::grandprod`] pass: the leaf cube is
//! `[batch = (row ++ head)][prod = key]` (key on the low variables), and the **public** top vector
//! pins every real (head,row) product to 0 (padding rows, where the mask is empty, to 1).
//!
//! The reduced leaf opening is NOT affine in the committed values: the mask `M(row,key)` couples
//! with both `q_sc(row,key,head)` (shared row AND key) and `x_max(row,head)` (shared row), so
//! `M̃·data ≠ MLE(M·data)` at a non-boolean point. We discharge it with one sumcheck over the
//! `(key, row)` variables the mask depends on (the head is folded at the grand-product's reduced
//! head point), reducing to single `X_MAX(R'++H)` / `Q_SC(K'++R'++H)` openings; the mask and the
//! off-mask constant enter as public per-`(key,row)` coefficients.

use p3_field::PrimeCharacteristicRing;
use utils::oracle::RandomOracle;
use utils::poly::MlPoly;
use utils::sumcheck::{self, SumcheckProof};

use crate::canonical::{self, Q_SC, X_MAX};
use crate::commit::{eq_at_index, ClaimAccumulator, CommitSet};
use crate::grandprod::{self, GrandProdProof};
use crate::protocol::EF;
use crate::reduce::low_first;
use crate::witness::Config;

pub struct SoftmaxMaxProof {
    pub gp: GrandProdProof<EF>,
    /// The `(key, row)`-variable sumcheck reconstructing the masked leaf opening.
    pub rs: SumcheckProof<EF>,
    /// `X_MAX(R' ++ H)` and `Q_SC(K' ++ R' ++ H)` at the sumcheck reduced point.
    pub x_max: EF,
    pub q_sc: EF,
}

impl SoftmaxMaxProof {
    pub fn size_bytes(&self) -> usize {
        self.gp.size_bytes() + self.rs.size_bytes() + 2 * core::mem::size_of::<EF>()
    }
}

/// The public top vector over `batch = (row ++ head)` (row low): every real `(head,row)` product is
/// 0; padding rows `r ≥ n_seq` (where the mask is empty so every leaf is 1) have product 1.
fn top_vector(config: &Config) -> Vec<EF> {
    let lay = canonical::layout_for(config, Q_SC);
    let row_pow = lay.row_pow;
    let inst_pow = 1usize << lay.inst_vars();
    (0..inst_pow * row_pow)
        .map(|b| if b % row_pow < config.n_seq { EF::ZERO } else { EF::ONE })
        .collect()
}

/// Build the leaf cube `leaf[head][row][key] = M·(x_max − q_sc) + (1−M)` over `(key ++ row ++ head)`.
fn build_leaf(set: &CommitSet, config: &Config) -> Vec<EF> {
    let lay = canonical::layout_for(config, Q_SC);
    let (row_pow, col_pow) = (lay.row_pow, lay.col_pow);
    let elem_len = row_pow * col_pow;
    let inst_pow = 1usize << lay.inst_vars();
    let q_sc = &set.prover_data(Q_SC).unwrap().0;
    let x_max = &set.prover_data(X_MAX).unwrap().0;
    let xmax_row_pow = canonical::layout_for(config, X_MAX).row_pow;
    let mut out = vec![EF::ONE; inst_pow * elem_len];
    for head in 0..inst_pow {
        for r in 0..row_pow {
            let xm = x_max[head * xmax_row_pow + r];
            for k in 0..col_pow {
                let idx = head * elem_len + r * col_pow + k;
                out[idx] = if r < config.n_seq && k <= r {
                    xm - q_sc[head * elem_len + r * col_pow + k]
                } else {
                    EF::ONE
                };
            }
        }
    }
    out
}

/// `col_vars` (key, the product dim), `row_vars`, `inst_vars` (head).
fn dims(config: &Config) -> (usize, usize, usize) {
    let lay = canonical::layout_for(config, Q_SC);
    (lay.col_vars(), lay.row_vars(), lay.inst_vars())
}

/// The public `(key, row)` weight `eq(K,k)·eq(R,r)·M[r,k]` and the off-mask constant
/// `C = Σ_{k,r} eq(K,k)eq(R,r)(1−M[r,k])`, for fixed grand-product key/row points `K`,`R`.
fn weight_and_const(config: &Config, key_pt: &[EF], row_pt: &[EF]) -> (Vec<EF>, EF) {
    let lay = canonical::layout_for(config, Q_SC);
    let (row_pow, col_pow) = (lay.row_pow, lay.col_pow);
    let eqk: Vec<EF> = (0..col_pow).map(|k| eq_at_index(key_pt, k)).collect();
    let eqr: Vec<EF> = (0..row_pow).map(|r| eq_at_index(row_pt, r)).collect();
    let mut w = vec![EF::ZERO; row_pow * col_pow];
    let mut c = EF::ZERO;
    for r in 0..row_pow {
        for k in 0..col_pow {
            let m = if r < config.n_seq && k <= r { EF::ONE } else { EF::ZERO };
            w[r * col_pow + k] = eqk[k] * eqr[r] * m;
            c += eqk[k] * eqr[r] * (EF::ONE - m);
        }
    }
    (w, c)
}

pub fn prove_softmax_max(
    set: &CommitSet,
    config: &Config,
    acc: &mut ClaimAccumulator,
    oracle: &mut RandomOracle<EF>,
) -> SoftmaxMaxProof {
    let (col_vars, row_vars, inst_vars) = dims(config);
    let lay = canonical::layout_for(config, Q_SC);
    let (row_pow, col_pow) = (lay.row_pow, lay.col_pow);
    let elem_len = row_pow * col_pow;
    let inst_pow = 1usize << inst_vars;
    let xmax_row_pow = canonical::layout_for(config, X_MAX).row_pow;

    let leaf = build_leaf(set, config);
    let batch_vars = leaf.len().trailing_zeros() as usize - col_vars;
    let (gp, leaf_pt, _leaf_eval) = grandprod::prove(&leaf, batch_vars, oracle);
    let key_pt = &leaf_pt[..col_vars];
    let row_pt = &leaf_pt[col_vars..col_vars + row_vars];
    let head_pt = &leaf_pt[col_vars + row_vars..];

    // operand[k,r] = Xrow(H,r) − Qfold(H,r,k), over (key ++ row) (key low). Head folded at H.
    let q_sc = &set.prover_data(Q_SC).unwrap().0;
    let x_max = &set.prover_data(X_MAX).unwrap().0;
    let eqh: Vec<EF> = (0..inst_pow).map(|h| eq_at_index(head_pt, h)).collect();
    let mut operand = vec![EF::ZERO; row_pow * col_pow];
    for r in 0..row_pow {
        let xrow = (0..inst_pow).fold(EF::ZERO, |a, h| a + eqh[h] * x_max[h * xmax_row_pow + r]);
        for k in 0..col_pow {
            let qfold =
                (0..inst_pow).fold(EF::ZERO, |a, h| a + eqh[h] * q_sc[h * elem_len + r * col_pow + k]);
            operand[r * col_pow + k] = xrow - qfold;
        }
    }
    let (weight, _c) = weight_and_const(config, key_pt, row_pt);
    let (rs, sc) = sumcheck::prove(vec![weight, operand], oracle);
    let red = low_first(&sc); // (K' ++ R')
    let kprime = &red[..col_vars];
    let rprime = &red[col_vars..];
    let xm_pt = [rprime.to_vec(), head_pt.to_vec()].concat();
    let xm = set.prover_data(X_MAX).unwrap().clone().eval(&xm_pt);
    acc.open(X_MAX, xm_pt, xm);
    let qs_pt = [kprime.to_vec(), rprime.to_vec(), head_pt.to_vec()].concat();
    let qv = set.prover_data(Q_SC).unwrap().clone().eval(&qs_pt);
    acc.open(Q_SC, qs_pt, qv);
    SoftmaxMaxProof { gp, rs, x_max: xm, q_sc: qv }
}

pub fn verify_softmax_max(
    proof: &SoftmaxMaxProof,
    set: &CommitSet,
    config: &Config,
    acc: &mut ClaimAccumulator,
    oracle: &mut RandomOracle<EF>,
) -> bool {
    let _ = set;
    let (col_vars, row_vars, _inst_vars) = dims(config);
    let top = top_vector(config);
    let Some((leaf_pt, leaf_eval)) = grandprod::verify(&proof.gp, &top, oracle) else {
        return false;
    };
    if leaf_pt.len() != col_vars + row_vars + canonical::layout_for(config, Q_SC).inst_vars() {
        return false;
    }
    let key_pt = &leaf_pt[..col_vars];
    let row_pt = &leaf_pt[col_vars..col_vars + row_vars];
    let head_pt = &leaf_pt[col_vars + row_vars..];

    // leaf_eval = Σ_{k,r} weight·operand + C ⇒ the sumcheck proves Σ weight·operand = leaf_eval − C.
    let (weight, c) = weight_and_const(config, key_pt, row_pt);
    let claim = leaf_eval - c;
    let Some(sc) = sumcheck::verify(claim, &proof.rs, oracle) else {
        return false;
    };
    if proof.rs.final_evals.len() != 2 {
        return false;
    }
    let red = low_first(&sc); // (K' ++ R')
    let kprime = &red[..col_vars];
    let rprime = &red[col_vars..];
    if proof.rs.final_evals[0] != MlPoly(weight).eval(&red) {
        return false;
    }
    // operand(K',R') = X_MAX(R'++H) − Q_SC(K'++R'++H).
    if proof.rs.final_evals[1] != proof.x_max - proof.q_sc {
        return false;
    }
    let xm_pt = [rprime.to_vec(), head_pt.to_vec()].concat();
    acc.open(X_MAX, xm_pt, proof.x_max);
    let qs_pt = [kprime.to_vec(), rprime.to_vec(), head_pt.to_vec()].concat();
    acc.open(Q_SC, qs_pt, proof.q_sc);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{BlockWeights, ModelWeights};
    use crate::tensor::Matrix;
    use crate::witness::Witness;

    fn tiny_n(n_layer: usize) -> (Config, Witness) {
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
        (cfg, witness)
    }

    fn roundtrip(n_layer: usize) {
        let (cfg, witness) = tiny_n(n_layer);
        let canon = canonical::build_online(&cfg, &witness);
        let mut rng = rand::rng();
        let mut oracle = RandomOracle::<EF>::new(&mut rng);
        let mut acc_p = ClaimAccumulator::new();
        let proof = prove_softmax_max(&canon.set, &cfg, &mut acc_p, &mut oracle);
        oracle.restart();
        let mut acc_v = ClaimAccumulator::new();
        assert!(verify_softmax_max(&proof, &canon.set, &cfg, &mut acc_v, &mut oracle));
        for c in &acc_v.claims {
            let got = canon.set.prover_data(&c.oracle).unwrap().clone().eval(&c.point);
            assert_eq!(got, c.value, "claim on {} mismatched", c.oracle);
        }
    }

    #[test]
    fn softmax_max_accepts_single() {
        roundtrip(1);
    }

    #[test]
    fn softmax_max_accepts_multi() {
        roundtrip(2);
    }

    #[test]
    fn non_max_rejected() {
        let (cfg, mut witness) = tiny_n(1);
        for h in 0..cfg.n_head {
            for r in 0..cfg.n_seq {
                let v = witness.blocks[0].attention.x_max[h].get(r, 0);
                witness.blocks[0].attention.x_max[h].set(r, 0, v + 1000);
            }
        }
        let canon = canonical::build_online(&cfg, &witness);
        let mut rng = rand::rng();
        let mut oracle = RandomOracle::<EF>::new(&mut rng);
        let mut acc_p = ClaimAccumulator::new();
        let proof = prove_softmax_max(&canon.set, &cfg, &mut acc_p, &mut oracle);
        oracle.restart();
        let mut acc_v = ClaimAccumulator::new();
        assert!(!verify_softmax_max(&proof, &canon.set, &cfg, &mut acc_v, &mut oracle));
    }
}
