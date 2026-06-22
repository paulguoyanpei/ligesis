//! Wall-clock benchmark for the GKR-LogUp lookup PIOP (`logup::lookup`) over GoldilocksExt2,
//! single-threaded. Proves `m` queries against an `n`-entry table (the `m ≫ n` regime),
//! excluding witness-gen and the PCS opening (multiplicities are precomputed from query
//! indices; openings batch separately via Basefold).
//!
//! Run with: `cargo bench -p logup --bench logup`

use std::time::Instant;

use p3_field::PrimeCharacteristicRing;
use p3_field::extension::BinomialExtensionField;
use p3_goldilocks::Goldilocks;
use rand::RngExt;

use logup::frac_sum;
use logup::lookup;
use utils::{oracle::RandomOracle, sumcheck};

type EF = BinomialExtensionField<Goldilocks, 2>;

fn from_u64(mut x: u64) -> EF {
    let mut acc = EF::ZERO;
    let mut base = EF::ONE;
    while x > 0 {
        if x & 1 == 1 {
            acc += base;
        }
        base += base;
        x >>= 1;
    }
    acc
}

fn bench_lookup(mu: usize, nu: usize, reps: usize) {
    let (m, n) = (1usize << mu, 1usize << nu);
    let mut rng = rand::rng();

    let t: Vec<EF> = (0..n).map(|_| rng.random::<EF>()).collect();
    let idx: Vec<usize> = (0..m).map(|_| rng.random::<u64>() as usize % n).collect();
    let a: Vec<EF> = idx.iter().map(|&j| t[j]).collect();
    let mut counts = vec![0u64; n];
    for &j in &idx {
        counts[j] += 1;
    }
    let e: Vec<EF> = counts.into_iter().map(from_u64).collect();

    let mut prove_t = f64::INFINITY;
    let mut verify_t = f64::INFINITY;
    let mut q_t = f64::INFINITY;
    let mut size = 0;
    for _ in 0..reps {
        let mut oracle = RandomOracle::<EF>::new(&mut rng);
        let t0 = Instant::now();
        let proof = lookup::prove_with_mults(&a, &t, &e, &mut oracle);
        prove_t = prove_t.min(t0.elapsed().as_secs_f64() * 1e3);
        size = proof.size_bytes();

        oracle.restart();
        let t1 = Instant::now();
        let ok = lookup::verify(&proof, &mut oracle).is_some();
        verify_t = verify_t.min(t1.elapsed().as_secs_f64() * 1e3);
        assert!(ok);

        // Query-side fraction-sum alone (the m-sized pass that dominates).
        let mut o2 = RandomOracle::<EF>::new(&mut rng);
        let alpha = o2.next_field();
        let pq = vec![EF::ONE; m];
        let qq: Vec<EF> = a.iter().map(|&ai| alpha - ai).collect();
        let t2 = Instant::now();
        let _ = frac_sum::prove(&pq, &qq, &mut o2);
        q_t = q_t.min(t2.elapsed().as_secs_f64() * 1e3);
    }

    println!("=== GKR-LogUp: m=2^{mu} queries, n=2^{nu} table ===");
    println!("prove time:  {prove_t:.1} ms   (query-side frac-sum alone: {q_t:.1} ms)");
    println!("verify time: {verify_t:.3} ms");
    println!("proof size:  {size} bytes ({:.2} KB)", size as f64 / 1024.0);
}

/// A plain degree-3 product sumcheck over `2^k` for reference (3 random multilinears).
fn bench_sumcheck(k: usize, reps: usize) {
    let n = 1usize << k;
    let mut rng = rand::rng();
    let mut prove_t = f64::INFINITY;
    let mut size = 0;
    for _ in 0..reps {
        let polys: Vec<Vec<EF>> = (0..3)
            .map(|_| (0..n).map(|_| rng.random::<EF>()).collect())
            .collect();
        let mut oracle = RandomOracle::<EF>::new(&mut rng);
        let t0 = Instant::now();
        let (proof, _) = sumcheck::prove(polys, &mut oracle);
        prove_t = prove_t.min(t0.elapsed().as_secs_f64() * 1e3);
        size = proof.size_bytes();
    }
    println!("=== degree-3 product sumcheck over 2^{k} ===");
    println!("prove time:  {prove_t:.1} ms");
    println!("proof size:  {size} bytes ({:.2} KB)", size as f64 / 1024.0);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 3 {
        let mu = args[1].parse().expect("mu");
        let nu = args[2].parse().expect("nu");
        let reps = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
        bench_lookup(mu, nu, reps);
        return;
    }
    bench_lookup(20, 14, 3);
    println!();
    bench_lookup(22, 16, 3);
    println!();
    bench_sumcheck(22, 3);
}
