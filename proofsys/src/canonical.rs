//! Canonical online commitments: one merged polynomial per witness type, with a **fixed** layout
//! every [`crate::commit::OracleSource`] map relies on.
//!
//! Layout (matching [`crate::commit::stack_instances`]): each instance is a matrix padded to
//! `row_pow × col_pow` with the **column on the low-order variables** and the row above it
//! (`flat = row * col_pow + col`); instances are stacked with the **instance index on the
//! high-order variables**. A vector witness (e.g. `std`) is a `col_pow = 1` matrix.
//!
//! Instance ordering follows protocol.md §1:
//! - LayerNorm types (`q_ln`, `a_ln`, `std`): `[ln1_0, ln2_0, ln1_1, ln2_1, …, ln1_11, ln2_11, lnf]`
//!   (25 instances; instance `2ℓ` = ln1 of layer `ℓ`, `2ℓ+1` = ln2, `24` = lnf).
//! - per-layer types (`q_qkv`, `q_ao`, `q_apr`, `q_fc`, `act`, `q_fpr`): 12 instances.
//! - per-head types (`q_sc`, `x_max`, `exp`, `sum_exp`, `q_prob`): 144 instances (`12ℓ + h`).
//! - `x_out`: 24 instances `[attn_0, mlp_0, attn_1, mlp_1, …]` (instance `2ℓ` = attn sub-block).
//! - `q_log`: 1 instance.

use p3_field::PrimeCharacteristicRing;
use std::collections::BTreeMap;

use crate::commit::{stack_instances, CommitSet};
use crate::protocol::EF;
use crate::tensor::Matrix;
use crate::witness::{encode_i64, Config, LayerNormWitness, Witness};

// Online committed oracle names.
pub const Q_LN: &str = "q_ln";
pub const A_LN: &str = "a_ln";
pub const STD: &str = "std";
pub const VAR: &str = "var";
pub const Q_QKV: &str = "q_qkv";
pub const Q_SC: &str = "q_sc";
pub const X_MAX: &str = "x_max";
pub const EXP: &str = "exp";
pub const SUM_EXP: &str = "sum_exp";
pub const Q_PROB: &str = "q_prob";
pub const Q_AO: &str = "q_ao";
pub const Q_APR: &str = "q_apr";
pub const Q_FC: &str = "q_fc";
pub const ACT: &str = "act";
pub const Q_FPR: &str = "q_fpr";
pub const X_OUT: &str = "x_out";
pub const Q_LOG: &str = "q_log";

/// The fixed per-type layout: instance count and the padded per-instance matrix shape.
#[derive(Clone, Copy, Debug)]
pub struct TypeLayout {
    pub instances: usize,
    pub row_pow: usize,
    pub col_pow: usize,
}

impl TypeLayout {
    pub fn col_vars(&self) -> usize {
        self.col_pow.trailing_zeros() as usize
    }
    pub fn row_vars(&self) -> usize {
        self.row_pow.trailing_zeros() as usize
    }
    pub fn elem_vars(&self) -> usize {
        self.col_vars() + self.row_vars()
    }
    pub fn inst_vars(&self) -> usize {
        self.instances.max(1).next_power_of_two().trailing_zeros() as usize
    }
}

impl TypeLayout {
    pub fn num_vars(&self) -> usize {
        self.inst_vars() + self.elem_vars()
    }
}

/// The fixed layout of a committed type, derived from `config` (public — no witness needed). Both
/// prover and verifier compute the identical layout.
pub fn layout_for(config: &Config, name: &str) -> TypeLayout {
    let l = config.n_layer;
    let s = config.n_seq;
    let d = config.d_model;
    let i_a = 3 * config.d_model;
    let i_m = config.mlp_hidden;
    let heads = l * config.n_head;
    let lns = 2 * l + 1;
    let (instances, rows, cols) = match name {
        Q_LN | A_LN => (lns, s, d),
        STD | VAR => (lns, s, 1),
        Q_QKV => (l, s, i_a),
        Q_SC | EXP | Q_PROB => (heads, s, s),
        X_MAX | SUM_EXP => (heads, s, 1),
        Q_AO | Q_APR | Q_FPR => (l, s, d),
        Q_FC | ACT => (l, s, i_m),
        X_OUT => (2 * l, s, d),
        Q_LOG => (1, s, config.vocab),
        other => panic!("unknown canonical type {other}"),
    };
    TypeLayout {
        instances,
        row_pow: rows.next_power_of_two(),
        col_pow: cols.next_power_of_two(),
    }
}

/// The committed online polynomials plus their layouts.
pub struct Canonical {
    pub set: CommitSet,
    pub layouts: BTreeMap<String, TypeLayout>,
}

impl Canonical {
    pub fn layout(&self, name: &str) -> TypeLayout {
        self.layouts[name]
    }
}

fn ef(x: i64) -> EF {
    EF::from(encode_i64(x))
}

/// Pad a matrix into the canonical element layout `row_pow × col_pow` (`flat = row*col_pow + col`).
fn pad_matrix(m: &Matrix, row_pow: usize, col_pow: usize) -> Vec<EF> {
    let mut out = vec![EF::ZERO; row_pow * col_pow];
    for r in 0..m.rows() {
        for c in 0..m.cols() {
            out[r * col_pow + c] = ef(m.get(r, c));
        }
    }
    out
}

struct Builder {
    set: CommitSet,
    layouts: BTreeMap<String, TypeLayout>,
}

impl Builder {
    fn commit_matrices(&mut self, name: &str, instances: &[&Matrix]) {
        let row_pow = instances[0].rows().next_power_of_two();
        let col_pow = instances[0].cols().next_power_of_two();
        let elem_len = row_pow * col_pow;
        let stacked: Vec<Vec<EF>> = instances
            .iter()
            .map(|m| {
                assert!(m.rows() <= row_pow && m.cols() <= col_pow, "{name} shape varies");
                pad_matrix(m, row_pow, col_pow)
            })
            .collect();
        self.set.commit(name, stack_instances(&stacked, elem_len));
        self.layouts.insert(
            name.to_owned(),
            TypeLayout {
                instances: instances.len(),
                row_pow,
                col_pow,
            },
        );
    }

    fn commit_vectors(&mut self, name: &str, instances: &[&[i64]]) {
        self.commit_vectors_padded(name, instances, 0);
    }

    /// Like `commit_vectors` but fills both the row-axis and instance-axis padding with `pad`
    /// (rather than 0). Used for `std`, where padding must be a *valid* std (≥1) so the sqrt /
    /// division limb brackets stay non-negative on padding vertices.
    fn commit_vectors_padded(&mut self, name: &str, instances: &[&[i64]], pad: i64) {
        let row_pow = instances[0].len().next_power_of_two();
        let inst_pow = instances.len().max(1).next_power_of_two();
        let mut data = vec![ef(pad); inst_pow * row_pow];
        for (i, v) in instances.iter().enumerate() {
            for (r, &x) in v.iter().enumerate() {
                data[i * row_pow + r] = ef(x);
            }
        }
        self.set.commit(name, data);
        self.layouts.insert(
            name.to_owned(),
            TypeLayout {
                instances: instances.len(),
                row_pow,
                col_pow: 1,
            },
        );
    }
}

// Offline weight oracle names (per layer where applicable).
pub fn attn_w(layer: usize) -> String {
    format!("attn_w_{layer}")
}
pub fn attn_proj_w(layer: usize) -> String {
    format!("attn_proj_w_{layer}")
}
pub fn fc_w(layer: usize) -> String {
    format!("fc_w_{layer}")
}
pub fn fproj_w(layer: usize) -> String {
    format!("fproj_w_{layer}")
}
pub const WTE: &str = "wte";

impl Builder {
    /// Commit a single weight matrix (no instance stacking): `row_pow × col_pow`, column low.
    fn commit_weight(&mut self, name: &str, m: &Matrix) {
        let row_pow = m.rows().next_power_of_two();
        let col_pow = m.cols().next_power_of_two();
        self.set.commit(name, pad_matrix(m, row_pow, col_pow));
        self.layouts.insert(
            name.to_owned(),
            TypeLayout {
                instances: 1,
                row_pow,
                col_pow,
            },
        );
    }
}

/// Build the offline weight commitments. (The unified table is public and reconstructed by the
/// verifier, so it is not committed here; only `e` is committed online.)
pub fn build_offline(_config: &Config, weights: &crate::model::ModelWeights) -> Canonical {
    let mut b = Builder {
        set: CommitSet::new(),
        layouts: BTreeMap::new(),
    };
    for (l, bw) in weights.blocks.iter().enumerate() {
        b.commit_weight(&attn_w(l), &bw.attn_w);
        b.commit_weight(&attn_proj_w(l), &bw.attn_proj_w);
        b.commit_weight(&fc_w(l), &bw.fc_w);
        b.commit_weight(&fproj_w(l), &bw.fproj_w);
    }
    b.commit_weight(WTE, &weights.wte);
    Canonical {
        set: b.set,
        layouts: b.layouts,
    }
}

/// All LayerNorm instances in canonical order: ln1/ln2 per block, then lnf.
fn layernorms(witness: &Witness) -> Vec<&LayerNormWitness> {
    let mut out = Vec::new();
    for block in &witness.blocks {
        out.push(&block.ln1);
        out.push(&block.ln2);
    }
    out.push(&witness.lnf);
    out
}

/// Build the online canonical commitments from a forward-pass witness.
pub fn build_online(_config: &Config, witness: &Witness) -> Canonical {
    let mut b = Builder {
        set: CommitSet::new(),
        layouts: BTreeMap::new(),
    };

    let lns = layernorms(witness);
    b.commit_matrices(Q_LN, &lns.iter().map(|ln| &ln.quotient).collect::<Vec<_>>());
    b.commit_matrices(A_LN, &lns.iter().map(|ln| &ln.dividend).collect::<Vec<_>>());
    // `std` pads with 1 (a valid std) so sqrt/division limb brackets stay ≥0 on padding vertices.
    b.commit_vectors_padded(STD, &lns.iter().map(|ln| ln.std.as_slice()).collect::<Vec<_>>(), 1);
    b.commit_vectors(VAR, &lns.iter().map(|ln| ln.var_sum.as_slice()).collect::<Vec<_>>());

    // Per-head matrices, flattened to S×S / S×1 via head_matrix / get.
    let head_mats = |sel: &dyn Fn(&crate::witness::AttentionWitness, usize) -> Matrix| {
        let mut v = Vec::new();
        for block in &witness.blocks {
            for h in 0..block.attention.scores.len() {
                v.push(sel(&block.attention, h));
            }
        }
        v
    };

    let q_sc = head_mats(&|a, h| a.scores[h].head_matrix(0));
    let exp = head_mats(&|a, h| a.exp[h].head_matrix(0));
    let q_prob = head_mats(&|a, h| a.q_prob[h].head_matrix(0));
    let x_max = head_mats(&|a, h| a.x_max[h].clone());
    let sum_exp = head_mats(&|a, h| a.sum_exp[h].clone());
    b.commit_matrices(Q_SC, &q_sc.iter().collect::<Vec<_>>());
    b.commit_matrices(EXP, &exp.iter().collect::<Vec<_>>());
    b.commit_matrices(Q_PROB, &q_prob.iter().collect::<Vec<_>>());
    b.commit_matrices(X_MAX, &x_max.iter().collect::<Vec<_>>());
    // `sum_exp` pads with 1 (a valid sum) so the softmax-division upper limb bracket
    // (`q·sum + sum − scale·exp − 1`) stays ≥0 on padding/masked vertices.
    let sum_exp_data: Vec<&[i64]> = sum_exp.iter().map(|m| m.data()).collect();
    b.commit_vectors_padded(SUM_EXP, &sum_exp_data, 1);

    let per_layer = |sel: &dyn Fn(&crate::witness::BlockWitness) -> &Matrix| {
        witness.blocks.iter().map(sel).collect::<Vec<_>>()
    };
    b.commit_matrices(Q_QKV, &per_layer(&|bl| &bl.attention.q_qkv));
    b.commit_matrices(Q_AO, &per_layer(&|bl| &bl.attention.q_ao));
    b.commit_matrices(Q_APR, &per_layer(&|bl| &bl.attention.q_apr));
    b.commit_matrices(Q_FC, &per_layer(&|bl| &bl.mlp.q_fc));
    b.commit_matrices(ACT, &per_layer(&|bl| &bl.mlp.act));
    b.commit_matrices(Q_FPR, &per_layer(&|bl| &bl.mlp.q_fpr));

    let mut x_out = Vec::new();
    for block in &witness.blocks {
        x_out.push(&block.attention.x_out);
        x_out.push(&block.mlp.x_out);
    }
    b.commit_matrices(X_OUT, &x_out);

    b.commit_matrices(Q_LOG, &[&witness.q_log]);

    Canonical {
        set: b.set,
        layouts: b.layouts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commit::instance_slice_point;
    use crate::model::{BlockWeights, ModelWeights};
    use utils::poly::MlPoly;

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
                fc_b: vec![0, 0, 0],
                fproj_w: Matrix::new(3, 2, vec![1, 0, 0, 1, 1, -1]),
                fproj_b: vec![0, 0],
            }],
        };
        let witness = weights.forward(Matrix::new(2, 2, vec![1, -2, 3, 1]), &cfg);
        (cfg, weights, witness)
    }

    /// A fixed-instance slice of each committed type must reproduce the witness matrix entries.
    #[test]
    fn instance_slices_match_witness() {
        let (cfg, _w, witness) = tiny();
        let canon = build_online(&cfg, &witness);

        // q_ln instance 1 = ln2 of layer 0; check entry (row 1, col 0).
        let lay = canon.layout(Q_LN);
        let ln2 = &witness.blocks[0].ln2;
        // element point for (row=1, col=0): col low (col_vars), row high (row_vars).
        let col = 0usize;
        let row = 1usize;
        let mut elem = Vec::new();
        for bit in 0..lay.col_vars() {
            elem.push(if (col >> bit) & 1 == 1 { EF::ONE } else { EF::ZERO });
        }
        for bit in 0..lay.row_vars() {
            elem.push(if (row >> bit) & 1 == 1 { EF::ONE } else { EF::ZERO });
        }
        let point = instance_slice_point(&elem, 1, lay.inst_vars());
        let poly = MlPoly(
            canon
                .set
                .commitment(Q_LN)
                .unwrap()
                .0
                 .0
                .clone(),
        );
        assert_eq!(poly.eval(&point), ef(ln2.quotient.get(row, col)));
    }
}
