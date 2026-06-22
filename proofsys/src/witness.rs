use p3_goldilocks::Goldilocks;

use crate::tensor::{Matrix, Tensor3};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Config {
    pub n_layer: usize,
    pub n_seq: usize,
    pub n_head: usize,
    pub d_head: usize,
    pub d_model: usize,
    pub mlp_hidden: usize,
    pub vocab: usize,
    pub scale: i64,
    pub max_v: i64,
}

impl Config {
    pub fn gpt2_31() -> Self {
        Self {
            n_layer: 12,
            n_seq: 31,
            n_head: 12,
            d_head: 64,
            d_model: 768,
            mlp_hidden: 3072,
            vocab: 50_257,
            scale: 1 << 12,
            max_v: 200,
        }
    }

    pub fn sqrt_d_scale(&self) -> i64 {
        self.scale * (self.d_head as f64).sqrt().round() as i64
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerNormWitness {
    pub input: Matrix,
    pub sum_x: Vec<i64>,
    pub sum_x2: Vec<i64>,
    pub var_sum: Vec<i64>,
    pub std: Vec<i64>,
    pub dividend: Matrix,
    pub quotient: Matrix,
    pub output: Matrix,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttentionWitness {
    pub q_qkv: Matrix,
    pub qkv: Matrix,
    pub scores: Vec<Tensor3>,
    pub x_max: Vec<Matrix>,
    pub exp: Vec<Tensor3>,
    pub sum_exp: Vec<Matrix>,
    pub q_prob: Vec<Tensor3>,
    pub q_ao_heads: Vec<Tensor3>,
    pub q_ao: Matrix,
    pub q_apr: Matrix,
    pub x_out: Matrix,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MlpWitness {
    pub q_fc: Matrix,
    pub fc: Matrix,
    pub act: Matrix,
    pub q_fpr: Matrix,
    pub x_out: Matrix,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockWitness {
    pub ln1: LayerNormWitness,
    pub attention: AttentionWitness,
    pub ln2: LayerNormWitness,
    pub mlp: MlpWitness,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Witness {
    pub x0: Matrix,
    pub blocks: Vec<BlockWitness>,
    pub lnf: LayerNormWitness,
    pub q_log: Matrix,
}

/// Encode a signed integer into Goldilocks (negatives as the additive inverse).
pub fn encode_i64(x: i64) -> Goldilocks {
    if x >= 0 {
        Goldilocks::new(x as u64)
    } else {
        -Goldilocks::new(x.unsigned_abs())
    }
}
