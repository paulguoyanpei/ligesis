use std::{collections::BTreeMap, fs, io, path::Path};

use crate::{
    tensor::{checked_i128_to_i64, floor_div_i128, round_sqrt, Matrix, Tensor3},
    witness::{AttentionWitness, BlockWitness, Config, LayerNormWitness, MlpWitness, Witness},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExportedPublicData {
    pub input_ids: Vec<i64>,
    pub x0: Matrix,
    pub logits_int: Matrix,
    pub final_x: Matrix,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockWeights {
    pub ln_1_g: Vec<i64>,
    pub ln_1_b: Vec<i64>,
    pub attn_w: Matrix,
    pub attn_b: Vec<i64>,
    pub attn_proj_w: Matrix,
    pub attn_proj_b: Vec<i64>,
    pub ln_2_g: Vec<i64>,
    pub ln_2_b: Vec<i64>,
    pub fc_w: Matrix,
    pub fc_b: Vec<i64>,
    pub fproj_w: Matrix,
    pub fproj_b: Vec<i64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelWeights {
    pub wte: Matrix,
    pub wpe: Matrix,
    pub has_wpe: bool,
    pub ln_f_g: Vec<i64>,
    pub ln_f_b: Vec<i64>,
    pub blocks: Vec<BlockWeights>,
    pub exp_lut: Vec<i64>,
    pub gelu_lut: Vec<i64>,
}

impl ModelWeights {
    pub fn load_export_dir(path: impl AsRef<Path>, config: &Config) -> io::Result<Self> {
        let path = path.as_ref();
        let manifest = fs::read_to_string(path.join("manifest.txt"))?;
        let mut arrays = BTreeMap::new();
        for line in manifest.lines().filter(|l| !l.trim().is_empty()) {
            let parts: Vec<_> = line.split_whitespace().collect();
            if parts.len() != 5 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("bad manifest line: {line}"),
                ));
            }
            let name = parts[0].to_owned();
            let dtype = parts[1];
            let d0: usize = parts[2].parse().map_err(invalid)?;
            let d1: usize = parts[3].parse().map_err(invalid)?;
            let file = path.join(parts[4]);
            let values = match dtype {
                "int32" => read_i32_as_i64(&file)?,
                "int64" => read_i64(&file)?,
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unsupported dtype {dtype}"),
                    ));
                }
            };
            let expected = d0 * if d1 == 0 { 1 } else { d1 };
            if values.len() != expected {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("{name} length {} != {expected}", values.len()),
                ));
            }
            arrays.insert(name, (d0, d1, values));
        }

        let take_vec = |arrays: &mut BTreeMap<String, (usize, usize, Vec<i64>)>,
                        name: &str,
                        len: usize|
         -> io::Result<Vec<i64>> {
            let (d0, d1, values) = arrays
                .remove(name)
                .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, name.to_owned()))?;
            if d1 != 0 || d0 != len {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("{name} shape mismatch"),
                ));
            }
            Ok(values)
        };
        let take_mat = |arrays: &mut BTreeMap<String, (usize, usize, Vec<i64>)>,
                        name: &str,
                        rows: usize,
                        cols: usize|
         -> io::Result<Matrix> {
            let (d0, d1, values) = arrays
                .remove(name)
                .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, name.to_owned()))?;
            if d0 != rows || d1 != cols {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("{name} shape {d0}x{d1} != {rows}x{cols}"),
                ));
            }
            Ok(Matrix::new(rows, cols, values))
        };

        let wte = take_mat(&mut arrays, "wte", config.vocab, config.d_model)?;
        let (has_wpe, wpe) = if arrays.contains_key("wpe") {
            (true, take_mat(&mut arrays, "wpe", 1024, config.d_model)?)
        } else {
            (false, Matrix::zeros(1024, config.d_model))
        };
        let ln_f_g = take_vec(&mut arrays, "ln_f_g", config.d_model)?;
        let ln_f_b = take_vec(&mut arrays, "ln_f_b", config.d_model)?;
        let exp_lut = take_vec(
            &mut arrays,
            "exp_lut",
            (config.max_v * config.scale + 1) as usize,
        )?;
        let gelu_lut = take_vec(
            &mut arrays,
            "gelu_lut",
            (2 * config.max_v * config.scale + 1) as usize,
        )?;

        let mut blocks = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            blocks.push(BlockWeights {
                ln_1_g: take_vec(&mut arrays, &format!("blk{i}_ln_1_g"), config.d_model)?,
                ln_1_b: take_vec(&mut arrays, &format!("blk{i}_ln_1_b"), config.d_model)?,
                attn_w: take_mat(
                    &mut arrays,
                    &format!("blk{i}_attn_w"),
                    config.d_model,
                    3 * config.d_model,
                )?,
                attn_b: take_vec(&mut arrays, &format!("blk{i}_attn_b"), 3 * config.d_model)?,
                attn_proj_w: take_mat(
                    &mut arrays,
                    &format!("blk{i}_attn_proj_w"),
                    config.d_model,
                    config.d_model,
                )?,
                attn_proj_b: take_vec(&mut arrays, &format!("blk{i}_attn_proj_b"), config.d_model)?,
                ln_2_g: take_vec(&mut arrays, &format!("blk{i}_ln_2_g"), config.d_model)?,
                ln_2_b: take_vec(&mut arrays, &format!("blk{i}_ln_2_b"), config.d_model)?,
                fc_w: take_mat(
                    &mut arrays,
                    &format!("blk{i}_fc_w"),
                    config.d_model,
                    config.mlp_hidden,
                )?,
                fc_b: take_vec(&mut arrays, &format!("blk{i}_fc_b"), config.mlp_hidden)?,
                fproj_w: take_mat(
                    &mut arrays,
                    &format!("blk{i}_fproj_w"),
                    config.mlp_hidden,
                    config.d_model,
                )?,
                fproj_b: take_vec(&mut arrays, &format!("blk{i}_fproj_b"), config.d_model)?,
            });
        }

        Ok(Self {
            wte,
            wpe,
            has_wpe,
            ln_f_g,
            ln_f_b,
            blocks,
            exp_lut,
            gelu_lut,
        })
    }

    pub fn load_exported_public_data(
        path: impl AsRef<Path>,
        config: &Config,
    ) -> io::Result<ExportedPublicData> {
        let path = path.as_ref();
        Ok(ExportedPublicData {
            input_ids: read_i64(&path.join("input_ids.bin"))?,
            x0: Matrix::new(
                config.n_seq,
                config.d_model,
                read_i64(&path.join("x0.bin"))?,
            ),
            logits_int: Matrix::new(
                config.n_seq,
                config.vocab,
                read_i64(&path.join("logits_int.bin"))?,
            ),
            final_x: Matrix::new(
                config.n_seq,
                config.d_model,
                read_i64(&path.join("final_x.bin"))?,
            ),
        })
    }

    pub fn x0_from_input_ids(&self, input_ids: &[i64], config: &Config) -> Matrix {
        assert_eq!(input_ids.len(), config.n_seq);
        let mut out = Matrix::zeros(config.n_seq, config.d_model);
        for (pos, &tok) in input_ids.iter().enumerate() {
            let tok = tok as usize;
            for d in 0..config.d_model {
                out.set(pos, d, self.wte.get(tok, d) + self.wpe.get(pos, d));
            }
        }
        out
    }

    pub fn forward(&self, x0: Matrix, config: &Config) -> Witness {
        assert_eq!(x0.rows(), config.n_seq);
        assert_eq!(x0.cols(), config.d_model);
        let mut x = x0.clone();
        let mut blocks = Vec::with_capacity(config.n_layer);
        for b in &self.blocks {
            let ln1 = layer_norm(&x, &b.ln_1_g, &b.ln_1_b);
            let attention = attention(&x, &ln1.output, b, self, config);
            let ln2 = layer_norm(&attention.x_out, &b.ln_2_g, &b.ln_2_b);
            let mlp = mlp(&attention.x_out, &ln2.output, b, self, config);
            x = mlp.x_out.clone();
            blocks.push(BlockWitness {
                ln1,
                attention,
                ln2,
                mlp,
            });
        }
        let lnf = layer_norm(&x, &self.ln_f_g, &self.ln_f_b);
        let q_log = lnf
            .output
            .matmul_transposed_rhs(&self.wte)
            .floor_div_const(config.scale);
        Witness {
            x0,
            blocks,
            lnf,
            q_log,
        }
    }
}

pub fn layer_norm(x: &Matrix, gain: &[i64], bias: &[i64]) -> LayerNormWitness {
    assert_eq!(x.cols(), gain.len());
    assert_eq!(gain.len(), bias.len());
    let d = x.cols() as i128;
    let mut sum_x = vec![0; x.rows()];
    let mut sum_x2 = vec![0; x.rows()];
    let mut var_sum = vec![0; x.rows()];
    let mut std = vec![0; x.rows()];
    let mut dividend = Matrix::zeros(x.rows(), x.cols());
    let mut quotient = Matrix::zeros(x.rows(), x.cols());
    let mut output = Matrix::zeros(x.rows(), x.cols());
    for r in 0..x.rows() {
        let sx: i128 = x.row(r).iter().map(|&v| v as i128).sum();
        let sx2: i128 = x.row(r).iter().map(|&v| (v as i128) * (v as i128)).sum();
        let vs = sx2 * d - sx * sx;
        let st = round_sqrt(vs + 1);
        sum_x[r] = checked_i128_to_i64(sx);
        sum_x2[r] = checked_i128_to_i64(sx2);
        var_sum[r] = checked_i128_to_i64(vs);
        std[r] = st;
        for c in 0..x.cols() {
            let a = gain[c] as i128 * (d * x.get(r, c) as i128 - sx);
            let q = floor_div_i128(a, st as i128);
            dividend.set(r, c, checked_i128_to_i64(a));
            quotient.set(r, c, q);
            output.set(r, c, q + bias[c]);
        }
    }
    LayerNormWitness {
        input: x.clone(),
        sum_x,
        sum_x2,
        var_sum,
        std,
        dividend,
        quotient,
        output,
    }
}

fn attention(
    residual: &Matrix,
    nx: &Matrix,
    b: &BlockWeights,
    weights: &ModelWeights,
    config: &Config,
) -> AttentionWitness {
    let q_qkv = nx.matmul(&b.attn_w).floor_div_const(config.scale);
    let qkv = q_qkv.add_row_vector(&b.attn_b);
    let (qs, ks, vs) = split_qkv_heads(&qkv, config);

    let mut scores = Vec::with_capacity(config.n_head);
    let mut x_max = Vec::with_capacity(config.n_head);
    let mut exp = Vec::with_capacity(config.n_head);
    let mut sum_exp = Vec::with_capacity(config.n_head);
    let mut q_prob = Vec::with_capacity(config.n_head);
    let mut q_ao_heads = Vec::with_capacity(config.n_head);
    let mut merged_ao = Matrix::zeros(config.n_seq, config.d_model);

    for h in 0..config.n_head {
        let q = qs.head_matrix(h);
        let k = ks.head_matrix(h);
        let v = vs.head_matrix(h);
        let sc = q
            .matmul_transposed_rhs(&k)
            .floor_div_const(config.sqrt_d_scale());
        let (xm, ex, se, prob) = softmax(&sc, weights, config);
        let ao = prob.head_matrix(0).matmul(&v).floor_div_const(config.scale);
        for i in 0..config.n_seq {
            for d in 0..config.d_head {
                merged_ao.set(i, h * config.d_head + d, ao.get(i, d));
            }
        }
        scores.push(Tensor3::new(1, config.n_seq, config.n_seq, sc.into_data()));
        x_max.push(xm);
        exp.push(ex);
        sum_exp.push(se);
        q_prob.push(prob);
        q_ao_heads.push(Tensor3::new(1, config.n_seq, config.d_head, ao.into_data()));
    }

    let q_apr = merged_ao
        .matmul(&b.attn_proj_w)
        .floor_div_const(config.scale);
    let projected = q_apr.add_row_vector(&b.attn_proj_b);
    let x_out = residual.add_matrix(&projected);
    AttentionWitness {
        q_qkv,
        qkv,
        scores,
        x_max,
        exp,
        sum_exp,
        q_prob,
        q_ao_heads,
        q_ao: merged_ao,
        q_apr,
        x_out,
    }
}

fn mlp(
    residual: &Matrix,
    nx: &Matrix,
    b: &BlockWeights,
    weights: &ModelWeights,
    config: &Config,
) -> MlpWitness {
    let q_fc = nx.matmul(&b.fc_w).floor_div_const(config.scale);
    let fc = q_fc.add_row_vector(&b.fc_b);
    let mut act = Matrix::zeros(config.n_seq, config.mlp_hidden);
    let offset = config.max_v * config.scale;
    for r in 0..fc.rows() {
        for c in 0..fc.cols() {
            let idx = fc.get(r, c) + offset;
            assert!(
                idx >= 0 && (idx as usize) < weights.gelu_lut.len(),
                "GELU index out of range: {idx}"
            );
            act.set(r, c, weights.gelu_lut[idx as usize]);
        }
    }
    let q_fpr = act.matmul(&b.fproj_w).floor_div_const(config.scale);
    let projected = q_fpr.add_row_vector(&b.fproj_b);
    let x_out = residual.add_matrix(&projected);
    MlpWitness {
        q_fc,
        fc,
        act,
        q_fpr,
        x_out,
    }
}

fn split_qkv_heads(qkv: &Matrix, config: &Config) -> (Tensor3, Tensor3, Tensor3) {
    let mut q = Tensor3::zeros(config.n_head, config.n_seq, config.d_head);
    let mut k = Tensor3::zeros(config.n_head, config.n_seq, config.d_head);
    let mut v = Tensor3::zeros(config.n_head, config.n_seq, config.d_head);
    for s in 0..config.n_seq {
        for h in 0..config.n_head {
            for d in 0..config.d_head {
                q.set(h, s, d, qkv.get(s, h * config.d_head + d));
                k.set(h, s, d, qkv.get(s, config.d_model + h * config.d_head + d));
                v.set(
                    h,
                    s,
                    d,
                    qkv.get(s, 2 * config.d_model + h * config.d_head + d),
                );
            }
        }
    }
    (q, k, v)
}

fn softmax(
    scores: &Matrix,
    weights: &ModelWeights,
    config: &Config,
) -> (Matrix, Tensor3, Matrix, Tensor3) {
    let mut x_max = Matrix::zeros(config.n_seq, 1);
    let mut exp = Tensor3::zeros(1, config.n_seq, config.n_seq);
    let mut sum_exp = Matrix::zeros(config.n_seq, 1);
    let mut probs = Tensor3::zeros(1, config.n_seq, config.n_seq);
    let offset = config.max_v * config.scale;
    for i in 0..config.n_seq {
        let mut mx = scores.get(i, 0);
        for j in 1..=i {
            mx = mx.max(scores.get(i, j));
        }
        x_max.set(i, 0, mx);
        let mut sum = 0i64;
        for j in 0..=i {
            let idx = scores.get(i, j) - mx + offset;
            assert!(
                idx >= 0 && (idx as usize) < weights.exp_lut.len(),
                "EXP index out of range: {idx}"
            );
            let e = weights.exp_lut[idx as usize];
            exp.set(0, i, j, e);
            sum += e;
        }
        sum_exp.set(i, 0, sum);
        for j in 0..=i {
            let p = floor_div_i128(exp.get(0, i, j) as i128 * config.scale as i128, sum as i128);
            probs.set(0, i, j, p);
        }
    }
    (x_max, exp, sum_exp, probs)
}

fn read_i32_as_i64(path: &Path) -> io::Result<Vec<i64>> {
    let bytes = fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "i32 file length is not divisible by 4",
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap()) as i64)
        .collect())
}

fn read_i64(path: &Path) -> io::Result<Vec<i64>> {
    let bytes = fs::read(path)?;
    if bytes.len() % 8 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "i64 file length is not divisible by 8",
        ));
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

fn invalid<E: ToString>(err: E) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, err.to_string())
}
