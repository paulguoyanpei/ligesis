//! Wall-clock benchmark for the LigeSIS PCS (single-threaded).
//!
//! Run with: `cargo bench -p pcs --bench ligesis`
//!
//! Reports setup, commit, eval (prove) and verify times plus proof size for a μ-variable
//! polynomial arranged as `2^log_m × 2^log_n`.

use std::time::Instant;

use p3_field::extension::BinomialExtensionField;
use p3_goldilocks::Goldilocks;
use rand::RngExt;

use pcs::ligesis::Ligesis;
use utils::{oracle::RandomOracle, poly::MlPoly};

type GoldilocksExt2 = BinomialExtensionField<Goldilocks, 2>;
type LS = Ligesis<Goldilocks, GoldilocksExt2>;

fn bench_ligesis(log_m: usize, log_n: usize) {
    let mu = log_m + log_n;
    let mut rng = rand::rng();

    let f: Vec<Goldilocks> = (0..(1usize << mu)).map(|_| rng.random()).collect();
    let z: Vec<GoldilocksExt2> = (0..mu).map(|_| rng.random()).collect();

    let t = Instant::now();
    let (pk, vk) = LS::setup(log_m, log_n, &mut rng);
    let setup_time = t.elapsed();

    let t = Instant::now();
    let (commit, data) = LS::commit(&pk, MlPoly(f));
    let commit_time = t.elapsed();

    let mut oracle = RandomOracle::<GoldilocksExt2>::new(&mut rng);
    let t = Instant::now();
    let proof = LS::prove(&pk, &data, z.clone(), &mut oracle);
    let prove_time = t.elapsed();

    let proof_size = proof.size_bytes();

    oracle.restart();
    let t = Instant::now();
    let ok = LS::verify(&vk, &commit, z, &proof, &mut oracle);
    let verify_time = t.elapsed();
    assert!(ok, "LigeSIS verification failed");

    println!("=== LigeSIS PCS ===");
    println!(
        "variables:   {mu}  (2^{mu} evals), shape: 2^{log_m} rows x 2^{log_n} cols",
    );
    println!("setup time:  {setup_time:?}");
    println!("commit time: {commit_time:?}");
    println!("eval time:   {prove_time:?}");
    println!("verify time: {verify_time:?}");
    println!(
        "proof size:  {proof_size} bytes ({:.2} KB)",
        proof_size as f64 / 1024.0
    );
}

fn main() {
    bench_ligesis(7, 15);
    println!();
    bench_ligesis(6, 16);
}
