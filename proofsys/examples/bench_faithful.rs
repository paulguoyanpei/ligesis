//! End-to-end benchmark of the faithful commit → reduce → batch-open PIOP (over the placeholder
//! PCS). Runs the integer forward pass, commits the canonical witnesses, proves and verifies the
//! unified lookup + all reductions, and reports timing, transcript size, and Fiat-Shamir draws.
//!
//! Usage: `cargo run --release --example bench_faithful [export_dir]`
//! With no export dir (or if loading fails) it runs a small synthetic config so the harness is
//! always exercisable.

use std::time::Instant;

use proofsys::faithful;
use proofsys::model::{BlockWeights, ModelWeights};
use proofsys::protocol::EF;
use proofsys::tensor::Matrix;
use proofsys::witness::Config;
use utils::oracle::RandomOracle;

fn main() {
    let export_dir = std::env::args().nth(1);
    let (label, config, weights, x0) = match export_dir.as_deref() {
        Some(dir) => match load_real(dir) {
            Some(t) => t,
            None => {
                eprintln!("could not load export from {dir}; using synthetic config");
                synthetic()
            }
        },
        None => synthetic(),
    };

    println!("config={label}");
    println!(
        "n_layer={} n_head={} d_model={} n_seq={} vocab={}",
        config.n_layer, config.n_head, config.d_model, config.n_seq, config.vocab
    );

    let t = Instant::now();
    let witness = weights.forward(x0, &config);
    let fwd_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let canon = faithful::build_commitments(&config, &weights, &witness);
    let commit_ms = t.elapsed().as_secs_f64() * 1000.0;

    let mut rng = rand::rng();
    let mut oracle = RandomOracle::<EF>::new(&mut rng);
    let t = Instant::now();
    let proof = faithful::prove(&canon.set, &config, &weights, &witness, &mut oracle);
    let prove_ms = t.elapsed().as_secs_f64() * 1000.0;
    let (fs_field, fs_int) = oracle.drawn();

    oracle.restart();
    let t = Instant::now();
    let ok = faithful::verify(&canon.set, &config, &weights, &proof, &mut oracle);
    let verify_ms = t.elapsed().as_secs_f64() * 1000.0;
    assert!(ok, "faithful verify failed");

    println!("forward_ms={fwd_ms:.3}");
    println!("commit_ms={commit_ms:.3}");
    println!("prove_ms={prove_ms:.3}");
    println!("verify_ms={verify_ms:.3}");
    println!("transcript_size_bytes={}", proof.size_bytes());
    println!("num_segments={}", proof.segment_openings.len());
    println!("fiat_shamir_field_challenges={fs_field}");
    println!("fiat_shamir_int_challenges={fs_int}");
    println!("verify_ok={ok}");
}

fn load_real(dir: &str) -> Option<(String, Config, ModelWeights, Matrix)> {
    let config = Config::gpt2_31();
    let weights = ModelWeights::load_export_dir(dir, &config).ok()?;
    let public = ModelWeights::load_exported_public_data(dir, &config).ok()?;
    Some(("gpt2_31".to_owned(), config, weights, public.x0))
}

fn synthetic() -> (String, Config, ModelWeights, Matrix) {
    let config = Config {
        n_layer: 2,
        n_seq: 2,
        n_head: 1,
        d_head: 2,
        d_model: 2,
        mlp_hidden: 3,
        vocab: 4,
        scale: 4,
        max_v: 2,
    };
    let exp_lut = (0..=(config.max_v * config.scale))
        .map(|i| if i == 0 { 0 } else { 1 })
        .collect();
    let gelu_lut = (0..=(2 * config.max_v * config.scale))
        .map(|i| i - config.max_v * config.scale)
        .collect();
    let block = BlockWeights {
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
    };
    let weights = ModelWeights {
        wte: Matrix::new(4, 2, vec![1, 0, 0, 1, 1, 1, -1, 2]),
        wpe: Matrix::zeros(1024, 2),
        has_wpe: true,
        ln_f_g: vec![4, 4],
        ln_f_b: vec![0, 0],
        exp_lut,
        gelu_lut,
        blocks: vec![block.clone(), block],
    };
    (
        "synthetic_2layer".to_owned(),
        config,
        weights,
        Matrix::new(2, 2, vec![1, -2, 3, 1]),
    )
}
