use crate::{
    model::ModelWeights,
    tensor::{floor_div_i128, round_sqrt, Matrix},
    witness::{Config, MergedPolys, Witness},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicInput {
    pub x0: Matrix,
    pub logits: Matrix,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InferencePiopProof {
    pub witness: Witness,
    pub merged_polys: MergedPolys,
}

pub fn prove(config: &Config, weights: &ModelWeights, x0: Matrix) -> InferencePiopProof {
    let witness = weights.forward(x0, config);
    let merged_polys = witness.merged_polys();
    InferencePiopProof {
        witness,
        merged_polys,
    }
}

pub fn verify(
    config: &Config,
    weights: &ModelWeights,
    public: &PublicInput,
    proof: &InferencePiopProof,
) -> bool {
    if proof.witness.x0 != public.x0 || proof.witness.q_log != public.logits {
        return false;
    }
    if proof.witness.merged_polys() != proof.merged_polys {
        return false;
    }
    verify_witness(config, weights, &proof.witness)
        && weights.forward(public.x0.clone(), config) == proof.witness
}

fn verify_witness(config: &Config, weights: &ModelWeights, witness: &Witness) -> bool {
    let mut x = &witness.x0;
    for (idx, block) in witness.blocks.iter().enumerate() {
        let bw = &weights.blocks[idx];
        if !verify_ln(x, &bw.ln_1_g, &bw.ln_1_b, &block.ln1) {
            return false;
        }
        if !verify_attention(config, weights, x, &block.ln1.output, bw, &block.attention) {
            return false;
        }
        if !verify_ln(&block.attention.x_out, &bw.ln_2_g, &bw.ln_2_b, &block.ln2) {
            return false;
        }
        if !verify_mlp(
            config,
            weights,
            &block.attention.x_out,
            &block.ln2.output,
            bw,
            &block.mlp,
        ) {
            return false;
        }
        x = &block.mlp.x_out;
    }
    if !verify_ln(x, &weights.ln_f_g, &weights.ln_f_b, &witness.lnf) {
        return false;
    }
    witness.q_log
        == witness
            .lnf
            .output
            .matmul_transposed_rhs(&weights.wte)
            .floor_div_const(config.scale)
}

fn verify_ln(
    input: &Matrix,
    gain: &[i64],
    bias: &[i64],
    ln: &crate::witness::LayerNormWitness,
) -> bool {
    if &ln.input != input {
        return false;
    }
    let d = input.cols() as i128;
    for r in 0..input.rows() {
        let sx: i128 = input.row(r).iter().map(|&v| v as i128).sum();
        let sx2: i128 = input
            .row(r)
            .iter()
            .map(|&v| (v as i128) * (v as i128))
            .sum();
        let vs = sx2 * d - sx * sx;
        let st = round_sqrt(vs + 1);
        if ln.sum_x[r] != sx as i64
            || ln.sum_x2[r] != sx2 as i64
            || ln.var_sum[r] != vs as i64
            || ln.std[r] != st
        {
            return false;
        }
        let lo = st as i128 * st as i128 - st as i128;
        let hi = st as i128 * st as i128 + st as i128;
        if vs + 1 < lo || vs + 1 > hi {
            return false;
        }
        for c in 0..input.cols() {
            let a = gain[c] as i128 * (d * input.get(r, c) as i128 - sx);
            let q = floor_div_i128(a, st as i128);
            let rem = a - q as i128 * st as i128;
            if rem < 0 || rem >= st as i128 {
                return false;
            }
            if ln.dividend.get(r, c) != a as i64
                || ln.quotient.get(r, c) != q
                || ln.output.get(r, c) != q + bias[c]
            {
                return false;
            }
        }
    }
    true
}

fn verify_attention(
    config: &Config,
    weights: &ModelWeights,
    residual: &Matrix,
    nx: &Matrix,
    bw: &crate::model::BlockWeights,
    attn: &crate::witness::AttentionWitness,
) -> bool {
    if attn.q_qkv != nx.matmul(&bw.attn_w).floor_div_const(config.scale) {
        return false;
    }
    if attn.qkv != attn.q_qkv.add_row_vector(&bw.attn_b) {
        return false;
    }
    let expected = crate::model::ModelWeights {
        wte: weights.wte.clone(),
        wpe: weights.wpe.clone(),
        has_wpe: weights.has_wpe,
        ln_f_g: weights.ln_f_g.clone(),
        ln_f_b: weights.ln_f_b.clone(),
        blocks: vec![bw.clone()],
        exp_lut: weights.exp_lut.clone(),
        gelu_lut: weights.gelu_lut.clone(),
    };
    let one_block = expected.forward(
        residual.clone(),
        &Config {
            n_layer: 1,
            ..config.clone()
        },
    );
    let got = &one_block.blocks[0].attention;
    attn == got
}

fn verify_mlp(
    config: &Config,
    weights: &ModelWeights,
    residual: &Matrix,
    nx: &Matrix,
    bw: &crate::model::BlockWeights,
    mlp: &crate::witness::MlpWitness,
) -> bool {
    if mlp.q_fc != nx.matmul(&bw.fc_w).floor_div_const(config.scale) {
        return false;
    }
    if mlp.fc != mlp.q_fc.add_row_vector(&bw.fc_b) {
        return false;
    }
    let offset = config.max_v * config.scale;
    for r in 0..mlp.fc.rows() {
        for c in 0..mlp.fc.cols() {
            let idx = mlp.fc.get(r, c) + offset;
            if idx < 0
                || idx as usize >= weights.gelu_lut.len()
                || mlp.act.get(r, c) != weights.gelu_lut[idx as usize]
            {
                return false;
            }
        }
    }
    if mlp.q_fpr != mlp.act.matmul(&bw.fproj_w).floor_div_const(config.scale) {
        return false;
    }
    residual.add_matrix(&mlp.q_fpr.add_row_vector(&bw.fproj_b)) == mlp.x_out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{model::BlockWeights, tensor::euclidean_remainder};

    #[test]
    fn signed_floor_division_matches_python_semantics() {
        let cases = [
            (-1, 4096, -1, 4095),
            (-4096, 4096, -1, 0),
            (-4097, 4096, -2, 4095),
            (4097, 4096, 1, 1),
        ];
        for (a, b, q, r) in cases {
            assert_eq!(floor_div_i128(a, b), q);
            assert_eq!(euclidean_remainder(a, b), r);
        }
    }

    #[test]
    fn tiny_end_to_end_piop_accepts_and_rejects_tamper() {
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
        let weights = tiny_weights(&cfg);
        let x0 = Matrix::new(2, 2, vec![1, -2, 3, 1]);
        let proof = prove(&cfg, &weights, x0.clone());
        let public = PublicInput {
            x0,
            logits: proof.witness.q_log.clone(),
        };
        assert!(verify(&cfg, &weights, &public, &proof));

        let mut bad = proof.clone();
        let tampered = bad.witness.blocks[0].ln1.quotient.get(0, 0) + 1;
        bad.witness.blocks[0].ln1.quotient.set(0, 0, tampered);
        assert!(!verify(&cfg, &weights, &public, &bad));
    }

    #[test]
    #[ignore = "loads and proves the full exported GPT-2 tensors; run manually when needed"]
    fn exported_gpt2_forward_matches_golden_logits() {
        let cfg = Config::gpt2_31();
        let export_dir = "../int_gpt/export";
        let weights = ModelWeights::load_export_dir(export_dir, &cfg).unwrap();
        let public_data = ModelWeights::load_exported_public_data(export_dir, &cfg).unwrap();
        if weights.has_wpe {
            assert_eq!(
                weights.x0_from_input_ids(&public_data.input_ids, &cfg),
                public_data.x0
            );
        }

        let proof = prove(&cfg, &weights, public_data.x0.clone());
        assert_eq!(proof.witness.lnf.output, public_data.final_x);
        assert_eq!(proof.witness.q_log, public_data.logits_int);
        let public = PublicInput {
            x0: public_data.x0,
            logits: public_data.logits_int,
        };
        assert!(verify(&cfg, &weights, &public, &proof));
    }

    fn tiny_weights(cfg: &Config) -> ModelWeights {
        let exp_lut = (0..=(cfg.max_v * cfg.scale))
            .map(|i| if i == 0 { 0 } else { 1 })
            .collect();
        let gelu_lut = (0..=(2 * cfg.max_v * cfg.scale))
            .map(|i| i - cfg.max_v * cfg.scale)
            .collect();
        ModelWeights {
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
        }
    }
}
