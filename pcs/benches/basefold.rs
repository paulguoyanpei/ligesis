//! Wall-clock benchmarks for the Basefold PCS (single-threaded).
//!
//! Run with: `cargo bench -p pcs --bench basefold`
//!
//! Reports the commit cost split into RS-encode time and Merkle-tree construction time,
//! plus evaluation time, verifier time, and proof size.

use std::time::Instant;

use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::PrimeCharacteristicRing;
use p3_field::extension::BinomialExtensionField;
use p3_goldilocks::Goldilocks;
use rand::RngExt;

use pcs::basefold::{Basefold, INTERLEAVE};
use utils::{
    merkle::{MerkleTreeProver, Serialize, hash_leaf},
    oracle::RandomOracle,
    poly::MlPoly,
};

type GoldilocksExt2 = BinomialExtensionField<Goldilocks, 2>;

/// Queries per the soundness target `2^{-100}` at RS rate `1/2^code_rate`.
fn query_num(code_rate: usize) -> usize {
    (100 + code_rate - 1) / code_rate
}

/// Times the two phases of `commit_base` separately: RS encoding (split + DFT per chunk)
/// and Merkle-tree construction (transpose into INTERLEAVE-element columns, serialize, hash/build).
/// Mirrors `Basefold::commit_base` + `to_commit`.
fn bench_commit_breakdown(n: usize, code_rate: usize) {
    let mut rng = rand::rng();
    let coeffs = (0..(1usize << n))
        .map(|_| rng.random::<Goldilocks>())
        .collect::<Vec<_>>();
    let poly = MlPoly(coeffs);

    // RS encode: INTERLEAVE interleaved chunks, each low-degree-extended by DFT.
    let t = Instant::now();
    let chunk_count = poly.0.len() / INTERLEAVE;
    let polies = poly.split(chunk_count);
    let code_length = polies[0].0.len() << code_rate;
    let dft = Radix2Dit::<Goldilocks>::default();
    // commit_base keeps the codewords in the base field (8 bytes), not upcast to EF.
    let codes = polies
        .iter()
        .map(|p| {
            let mut c = p.0.clone();
            c.resize(code_length, Goldilocks::ZERO);
            dft.dft(c)
        })
        .collect::<Vec<_>>();
    let encode_time = t.elapsed();

    // Merkle tree construction (mirrors the optimized `to_commit`): tile-by-tile column gather
    // with contiguous reads + fused serialize/hash (no full transpose, no big allocation),
    // then build the tree. Leaves are base-field (half the bytes of EF).
    const TILE: usize = 64;
    let t = Instant::now();
    let mut tile = vec![Goldilocks::ZERO; TILE * INTERLEAVE];
    let mut buf = Vec::with_capacity(INTERLEAVE * core::mem::size_of::<Goldilocks>());
    let mut leaf_hashes = Vec::with_capacity(code_length);
    let mut start = 0;
    while start < code_length {
        let end = (start + TILE).min(code_length);
        let width = end - start;
        for j in 0..INTERLEAVE {
            let cw = &codes[j];
            for local in 0..width {
                tile[local * INTERLEAVE + j] = cw[start + local];
            }
        }
        for local in 0..width {
            buf.clear();
            Serialize::serialize_fields_into(
                &tile[local * INTERLEAVE..(local + 1) * INTERLEAVE],
                &mut buf,
            );
            leaf_hashes.push(hash_leaf(&buf));
        }
        start = end;
    }
    let hash_time = t.elapsed();

    let t = Instant::now();
    let mt = MerkleTreeProver::from_leaf_hashes(leaf_hashes);
    let _root = mt.commit();
    let build_time = t.elapsed();

    let merkle_time = hash_time + build_time;

    println!("=== Basefold commit breakdown (commit_base) ===");
    println!("variables:   {n}  (2^{n} = {} coeffs)", 1usize << n);
    println!(
        "code_rate:   {code_rate}  (RS rate 1/{}), code length: {code_length} x {INTERLEAVE} cols",
        1usize << code_rate,
    );
    println!("RS encode time:           {encode_time:?}");
    println!("Merkle construction time: {merkle_time:?}");
    println!("  - gather + serialize + hash: {hash_time:?}");
    println!("  - build tree:                {build_time:?}");
    println!("commit total:             {:?}", encode_time + merkle_time);
}

/// Times prove/verify and reports proof size, using the real `commit_base`.
fn bench_prove_verify(n: usize, code_rate: usize) {
    let mut rng = rand::rng();
    let coeffs = (0..(1usize << n))
        .map(|_| rng.random::<Goldilocks>())
        .collect::<Vec<_>>();
    let poly = MlPoly(coeffs);
    let point = (0..n)
        .map(|_| rng.random::<GoldilocksExt2>())
        .collect::<Vec<_>>();

    let (state, commit) = Basefold::<Goldilocks, GoldilocksExt2>::commit_base(poly, code_rate);

    let mut oracle = RandomOracle::<GoldilocksExt2>::new(&mut rng);
    let t = Instant::now();
    let proof = Basefold::<Goldilocks, GoldilocksExt2>::prove(&state, point.clone(), &mut oracle);
    let prove_time = t.elapsed();

    let proof_size = proof.size_bytes();

    oracle.restart();
    let t = Instant::now();
    let ok =
        Basefold::<Goldilocks, GoldilocksExt2>::verify(&commit, point.clone(), &proof, &mut oracle);
    let verify_time = t.elapsed();
    assert!(ok);

    println!("=== Basefold prove/verify ===");
    println!("variables:   {n}, code_rate: {code_rate}, queries: {}", query_num(code_rate));
    println!("eval time:   {prove_time:?}");
    println!("verify time: {verify_time:?}");
    println!(
        "proof size:  {proof_size} bytes ({:.2} KB)",
        proof_size as f64 / 1024.0
    );
}

fn main() {
    // code_rate = 1 (ρ=1/2, 100 queries) to match LigeSIS's secondary-PCS setting.
    for n in [22usize, 24] {
        bench_commit_breakdown(n, 1);
        bench_prove_verify(n, 1);
        println!();
    }
}
