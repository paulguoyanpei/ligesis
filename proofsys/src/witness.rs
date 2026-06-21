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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MergedPolys {
    pub std: Vec<i64>,
    pub q_qkv: Vec<i64>,
    pub q_sc: Vec<i64>,
    pub x_max: Vec<i64>,
    pub exp: Vec<i64>,
    pub q_prob: Vec<i64>,
    pub q_ao: Vec<i64>,
    pub q_fc: Vec<i64>,
    pub act: Vec<i64>,
}

impl Witness {
    pub fn logits(&self) -> &Matrix {
        &self.q_log
    }

    pub fn merged_polys(&self) -> MergedPolys {
        let mut std = Vec::new();
        let mut q_qkv = Vec::new();
        let mut q_sc = Vec::new();
        let mut x_max = Vec::new();
        let mut exp = Vec::new();
        let mut q_prob = Vec::new();
        let mut q_ao = Vec::new();
        let mut q_fc = Vec::new();
        let mut act = Vec::new();

        for block in &self.blocks {
            push_ln(&block.ln1, &mut std);
            q_qkv.extend(block.attention.q_qkv.data());
            for scores in &block.attention.scores {
                q_sc.extend(scores.data());
            }
            for x in &block.attention.x_max {
                x_max.extend(x.data());
            }
            for e in &block.attention.exp {
                exp.extend(e.data());
            }
            for p in &block.attention.q_prob {
                q_prob.extend(p.data());
            }
            q_ao.extend(block.attention.q_ao.data());

            push_ln(&block.ln2, &mut std);
            q_fc.extend(block.mlp.q_fc.data());
            act.extend(block.mlp.act.data());
        }
        push_ln(&self.lnf, &mut std);

        MergedPolys {
            std: pad_power_of_two(std),
            q_qkv: pad_power_of_two(q_qkv),
            q_sc: pad_power_of_two(q_sc),
            x_max: pad_power_of_two(x_max),
            exp: pad_power_of_two(exp),
            q_prob: pad_power_of_two(q_prob),
            q_ao: pad_power_of_two(q_ao),
            q_fc: pad_power_of_two(q_fc),
            act: pad_power_of_two(act),
        }
    }
}

fn push_ln(ln: &LayerNormWitness, std: &mut Vec<i64>) {
    std.extend(&ln.std);
}

pub fn pad_power_of_two(mut values: Vec<i64>) -> Vec<i64> {
    let len = values.len().max(1).next_power_of_two();
    values.resize(len, 0);
    values
}

pub fn encode_i64(x: i64) -> Goldilocks {
    if x >= 0 {
        Goldilocks::new(x as u64)
    } else {
        -Goldilocks::new(x.unsigned_abs())
    }
}

pub fn encode_poly(values: &[i64]) -> Vec<Goldilocks> {
    values.iter().map(|&x| encode_i64(x)).collect()
}
