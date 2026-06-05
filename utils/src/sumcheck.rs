use p3_field::Field;

use crate::oracle::RandomOracle;

/// A sumcheck proof for the sum over the boolean hypercube of a product of multilinear
/// polynomials. Each round message is the degree-`d` univariate `g_i` evaluated at the
/// points `0..=d`, where `d = polys.len()` is the number of factors.
pub struct SumcheckProof<F: Field> {
    pub round_polys: Vec<Vec<F>>,
    pub final_evals: Vec<F>,
}

impl<F: Field> SumcheckProof<F> {
    /// Serialized proof size in bytes: the `(d+1)` field elements per round plus the final
    /// per-factor evaluations.
    pub fn size_bytes(&self) -> usize {
        let f = core::mem::size_of::<F>();
        self.round_polys.iter().map(|r| r.len() * f).sum::<usize>() + self.final_evals.len() * f
    }
}

/// Field elements `0, 1, ..., count-1`.
fn eval_points<F: Field>(count: usize) -> Vec<F> {
    let mut pts = Vec::with_capacity(count);
    let mut p = F::ZERO;
    for _ in 0..count {
        pts.push(p);
        p += F::ONE;
    }
    pts
}

/// Prove `H = sum_{x in {0,1}^n} prod_j polys[j](x)`. Each polynomial is given as its `2^n`
/// hypercube evaluations (all the same length, a power of two). The per-round univariate has
/// degree `d = polys.len()`; each round binds the high variable. Challenges are drawn from
/// `oracle`. Returns the proof and the sampled challenge point.
pub fn prove<F: Field>(
    mut polys: Vec<Vec<F>>,
    oracle: &mut RandomOracle<F>,
) -> (SumcheckProof<F>, Vec<F>) {
    assert!(!polys.is_empty());
    let d = polys.len();
    let n = polys[0].len().trailing_zeros() as usize;
    let points = eval_points::<F>(d + 1);

    let mut round_polys = Vec::with_capacity(n);
    let mut challenges = Vec::with_capacity(n);
    for _ in 0..n {
        let half = polys[0].len() / 2;

        // g_i(t) = sum_i prod_j (lo_j + t*(hi_j - lo_j)) for each evaluation point t.
        let mut evals = vec![F::ZERO; d + 1];
        for i in 0..half {
            for (k, &t) in points.iter().enumerate() {
                let mut prod = F::ONE;
                for poly in &polys {
                    let lo = poly[i];
                    let hi = poly[i + half];
                    prod *= lo + t * (hi - lo);
                }
                evals[k] += prod;
            }
        }
        round_polys.push(evals);

        let r = oracle.next_field();
        challenges.push(r);
        for poly in polys.iter_mut() {
            for i in 0..half {
                poly[i] = poly[i] + r * (poly[i + half] - poly[i]);
            }
            poly.truncate(half);
        }
    }

    let final_evals = polys.iter().map(|p| p[0]).collect();
    (
        SumcheckProof {
            round_polys,
            final_evals,
        },
        challenges,
    )
}

/// Lagrange-interpolate the univariate through `(0, evals[0]), ..., (d, evals[d])` and
/// evaluate it at `r`.
fn interpolate<F: Field>(evals: &[F], r: F) -> F {
    let pts = eval_points::<F>(evals.len());
    let mut acc = F::ZERO;
    for k in 0..evals.len() {
        let mut num = F::ONE;
        let mut den = F::ONE;
        for m in 0..evals.len() {
            if m == k {
                continue;
            }
            num *= r - pts[m];
            den *= pts[k] - pts[m];
        }
        acc += evals[k] * num * den.inverse();
    }
    acc
}

/// Verify a sumcheck proof against the claimed sum `claim`. Returns the sampled challenge
/// point on success, or `None` if any round check fails. Only enforces that the product of
/// `final_evals` equals the final claim; the caller must separately check that `final_evals`
/// are the correct openings of each factor at the returned point (e.g. via a PCS).
pub fn verify<F: Field>(
    claim: F,
    proof: &SumcheckProof<F>,
    oracle: &mut RandomOracle<F>,
) -> Option<Vec<F>> {
    let mut cur = claim;
    let mut challenges = Vec::with_capacity(proof.round_polys.len());
    for evals in &proof.round_polys {
        if evals.len() < 2 || evals[0] + evals[1] != cur {
            return None;
        }
        let r = oracle.next_field();
        cur = interpolate(evals, r);
        challenges.push(r);
    }
    let prod = proof.final_evals.iter().fold(F::ONE, |a, &b| a * b);
    if prod != cur {
        return None;
    }
    Some(challenges)
}

#[cfg(test)]
mod tests {
    use super::*;

    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_goldilocks::Goldilocks;
    use rand::RngExt;

    type EF = BinomialExtensionField<Goldilocks, 2>;

    fn hypercube_sum<F: Field>(polys: &[Vec<F>]) -> F {
        (0..polys[0].len())
            .map(|i| polys.iter().fold(F::ONE, |a, p| a * p[i]))
            .fold(F::ZERO, |a, b| a + b)
    }

    fn random_polys(d: usize, n: usize, rng: &mut impl rand::Rng) -> Vec<Vec<EF>> {
        (0..d)
            .map(|_| (0..(1usize << n)).map(|_| rng.random::<EF>()).collect())
            .collect()
    }

    #[test]
    fn product_sumcheck_round_trip() {
        let mut rng = rand::rng();
        let polys = random_polys(3, 10, &mut rng);
        let claim = hypercube_sum(&polys);

        let mut oracle = RandomOracle::<EF>::new(&mut rng);
        let (proof, _challenges) = prove(polys, &mut oracle);

        oracle.restart();
        assert!(verify(claim, &proof, &mut oracle).is_some());

        oracle.restart();
        assert!(verify(claim + EF::ONE, &proof, &mut oracle).is_none());
    }

    #[test]
    fn tampered_round_is_rejected() {
        let mut rng = rand::rng();
        let polys = random_polys(3, 8, &mut rng);
        let claim = hypercube_sum(&polys);

        let mut oracle = RandomOracle::<EF>::new(&mut rng);
        let (mut proof, _) = prove(polys, &mut oracle);
        proof.round_polys[2][3] += EF::ONE;

        oracle.restart();
        assert!(verify(claim, &proof, &mut oracle).is_none());
    }
}
