//! Wall-clock benchmark for the product sumcheck (single-threaded).
//!
//! Run with: `cargo bench -p pcs --bench sumcheck`
//!
//! Benchmarks proving and verifying `sum_x prod_j p_j(x)` for `degree` multilinear factors
//! (so the per-round univariate has that degree), over both the base field and the
//! extension field.

use std::time::Instant;

use p3_field::Field;
use p3_field::extension::BinomialExtensionField;
use p3_goldilocks::Goldilocks;
use rand::RngExt;
use rand::distr::{Distribution, StandardUniform};

use utils::{oracle::RandomOracle, sumcheck};

type GoldilocksExt2 = BinomialExtensionField<Goldilocks, 2>;

fn bench_product_sumcheck<F: Field>(field_name: &str, n: usize, degree: usize)
where
    StandardUniform: Distribution<F>,
{
    let mut rng = rand::rng();
    let polys = (0..degree)
        .map(|_| {
            (0..(1usize << n))
                .map(|_| rng.random::<F>())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    // Claimed sum over the hypercube.
    let claim = (0..(1usize << n))
        .map(|i| polys.iter().fold(F::ONE, |a, p| a * p[i]))
        .fold(F::ZERO, |a, b| a + b);

    let mut oracle = RandomOracle::<F>::new(&mut rng);
    let t = Instant::now();
    let (proof, _challenges) = sumcheck::prove(polys, &mut oracle);
    let prove_time = t.elapsed();

    let proof_size = proof.size_bytes();

    oracle.restart();
    let t = Instant::now();
    let ok = sumcheck::verify(claim, &proof, &mut oracle).is_some();
    let verify_time = t.elapsed();
    assert!(ok);

    println!("=== Sumcheck benchmark (product of {degree} multilinears) ===");
    println!("field:       {field_name}");
    println!("variables:   {n}  (2^{n} = {} evals/poly)", 1usize << n);
    println!("degree:      {degree}  (degree-{degree} round polynomials)");
    println!("prove time:  {prove_time:?}");
    println!("verify time: {verify_time:?}");
    println!(
        "proof size:  {proof_size} bytes ({:.2} KB)",
        proof_size as f64 / 1024.0
    );
}

fn main() {
    bench_product_sumcheck::<Goldilocks>("Goldilocks (base field)", 22, 3);
    println!();
    bench_product_sumcheck::<GoldilocksExt2>("GoldilocksExt2 (extension field)", 22, 3);
}
