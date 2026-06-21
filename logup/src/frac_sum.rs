//! GKR fraction-sum circuit: prove `Σ_i p_i/q_i = num/den` for committed numerator and
//! denominator MLEs `p, q` over `{0,1}^K`, reducing to a single evaluation of `p` and `q`
//! at a random point.
//!
//! The circuit is a binary tree of fraction additions. Layer `L_s` has `2^s` fractions; its
//! parent `L_{s-1}` halves the count by pairing along the **low bit**:
//!
//! ```text
//!   p_{s-1}[i] = p_s[2i]·q_s[2i+1] + p_s[2i+1]·q_s[2i],
//!   q_{s-1}[i] = q_s[2i]·q_s[2i+1].
//! ```
//!
//! The root `L_0` is the total fraction `(num, den)`. Verification walks root→input: each
//! step reduces a claim `(P,Q)` about `L_r` at a point `z` to a claim about `L_{r+1}` via a
//! degree-3 sumcheck on `eq(z,·)·(A0·B1 + A1·B0 + λ·B0·B1)`, where `A0,A1,B0,B1` are the
//! low-bit splits of `L_{r+1}`'s `p,q`. After `K` steps the claim lands on the input layer,
//! returned for the caller (PCS) to discharge.
//!
//! Conventions: a point is low-bit-first (`point[0]` is the adjacent-pair bit, matching
//! [`utils::poly::MlPoly::eval`]); every sumcheck binds the low bit first, so challenge
//! vectors need no reversal.

use p3_field::Field;
use utils::oracle::RandomOracle;

/// One layer reduction: the degree-3 sumcheck round polynomials (evaluations at `0,1,2,3`)
/// followed by the four low-bit-split leaf evaluations `[A0, A1, B0, B1]` at the sumcheck
/// point.
pub struct LayerProof<F: Field> {
    pub round_polys: Vec<[F; 4]>,
    pub leaves: [F; 4],
}

/// A fraction-sum proof: the revealed root fraction plus one [`LayerProof`] per reduction
/// (`K` of them, root→input).
pub struct FracSumProof<F: Field> {
    pub num: F,
    pub den: F,
    pub layers: Vec<LayerProof<F>>,
}

impl<F: Field> FracSumProof<F> {
    /// Serialized PIOP size in bytes (excludes the PCS opening discharged by the caller).
    pub fn size_bytes(&self) -> usize {
        let f = core::mem::size_of::<F>();
        2 * f
            + self
                .layers
                .iter()
                .map(|l| (l.round_polys.len() * 4 + 4) * f)
                .sum::<usize>()
    }
}

/// `eq_table(z)[i] = ∏_k eq(z[k], bit_k(i))` over `i ∈ {0,1}^{|z|}`, with `z[0]` the low bit.
fn eq_table<F: Field>(z: &[F]) -> Vec<F> {
    let mut tab = vec![F::ONE];
    for &zk in z {
        let half = tab.len();
        let mut next = vec![F::ZERO; half * 2];
        for i in 0..half {
            next[i] = tab[i] * (F::ONE - zk); // new (high) bit = 0
            next[i + half] = tab[i] * zk; // new (high) bit = 1
        }
        tab = next;
    }
    tab
}

/// `∏_k eq(z[k], z'[k])`.
fn eq_eval<F: Field>(z: &[F], zp: &[F]) -> F {
    z.iter()
        .zip(zp)
        .fold(F::ONE, |acc, (&a, &b)| acc * (a * b + (F::ONE - a) * (F::ONE - b)))
}

/// Bind the low bit: `arr[j] ← arr[2j] + c·(arr[2j+1] − arr[2j])`, halving the length.
fn fold_low<F: Field>(arr: &mut Vec<F>, c: F) {
    let half = arr.len() / 2;
    for j in 0..half {
        arr[j] = arr[2 * j] + c * (arr[2 * j + 1] - arr[2 * j]);
    }
    arr.truncate(half);
}

/// Lagrange-interpolate the degree-3 poly through `(0..3, evals)` and evaluate at `x`.
fn interp4<F: Field>(evals: &[F; 4], x: F) -> F {
    // Denominators ∏_{m≠k}(k−m) for nodes {0,1,2,3}: -6, 2, -2, 6.
    let two = F::ONE + F::ONE;
    let three = two + F::ONE;
    let six = two * three;
    let nodes = [F::ZERO, F::ONE, two, three];
    let denoms = [-six, two, -two, six];
    let mut acc = F::ZERO;
    for k in 0..4 {
        let mut num = F::ONE;
        for (m, &nm) in nodes.iter().enumerate() {
            if m != k {
                num *= x - nm;
            }
        }
        acc += evals[k] * num * denoms[k].inverse();
    }
    acc
}

/// Prove `Σ_i p_in[i]/q_in[i] = num/den`. Both inputs must have the same power-of-two
/// length. Returns the proof and the reduced input claim `(point, p_in(point), q_in(point))`
/// (low-bit-first point), which the caller must discharge against the committed `p_in, q_in`.
pub fn prove<F: Field>(
    p_in: &[F],
    q_in: &[F],
    oracle: &mut RandomOracle<F>,
) -> (FracSumProof<F>, Vec<F>, F, F) {
    let n = p_in.len();
    assert!(n.is_power_of_two() && q_in.len() == n);
    let k = n.trailing_zeros() as usize;

    // Build every layer L_s (size 2^s), s = k (input) down to 0 (root).
    let mut lp: Vec<Vec<F>> = vec![Vec::new(); k + 1];
    let mut lq: Vec<Vec<F>> = vec![Vec::new(); k + 1];
    lp[k] = p_in.to_vec();
    lq[k] = q_in.to_vec();
    for s in (0..k).rev() {
        let sz = 1usize << s;
        let mut p = vec![F::ZERO; sz];
        let mut q = vec![F::ZERO; sz];
        let pn = &lp[s + 1];
        let qn = &lq[s + 1];
        for i in 0..sz {
            p[i] = pn[2 * i] * qn[2 * i + 1] + pn[2 * i + 1] * qn[2 * i];
            q[i] = qn[2 * i] * qn[2 * i + 1];
        }
        lp[s] = p;
        lq[s] = q;
    }
    let num = lp[0][0];
    let den = lq[0][0];

    // Reduce root→input: step r turns a claim about L_r at point z into one about L_{r+1}.
    let mut z: Vec<F> = Vec::new();
    let mut layers = Vec::with_capacity(k);
    for r in 0..k {
        let lambda = oracle.next_field();
        let (layer, zp, leaves) = prove_layer(&z, lambda, &lp[r + 1], &lq[r + 1], oracle);
        let beta = oracle.next_field();
        let [a0, a1, b0, b1] = leaves;
        z = core::iter::once(beta).chain(zp).collect();
        layers.push(layer);
        // Folded input claim carried forward (overwritten until the last step).
        if r + 1 == k {
            return (
                FracSumProof { num, den, layers },
                z,
                (F::ONE - beta) * a0 + beta * a1,
                (F::ONE - beta) * b0 + beta * b1,
            );
        }
    }
    // k == 0: a single fraction; the "input" claim is the root itself at the empty point.
    (FracSumProof { num, den, layers }, z, num, den)
}

/// The per-layer degree-3 sumcheck (prover). Returns the layer proof, the sumcheck point
/// `z'` (low-bit-first), and the four leaf evaluations `[A0, A1, B0, B1]`.
fn prove_layer<F: Field>(
    z: &[F],
    lambda: F,
    p_next: &[F],
    q_next: &[F],
    oracle: &mut RandomOracle<F>,
) -> (LayerProof<F>, Vec<F>, [F; 4]) {
    let r = z.len();
    let half0 = p_next.len() / 2; // = 2^r
    let mut a0: Vec<F> = (0..half0).map(|i| p_next[2 * i]).collect();
    let mut a1: Vec<F> = (0..half0).map(|i| p_next[2 * i + 1]).collect();
    let mut b0: Vec<F> = (0..half0).map(|i| q_next[2 * i]).collect();
    let mut b1: Vec<F> = (0..half0).map(|i| q_next[2 * i + 1]).collect();
    let mut eqv = eq_table(z);

    let two = F::ONE + F::ONE;
    let three = two + F::ONE;
    let pts = [F::ZERO, F::ONE, two, three];

    let mut round_polys = Vec::with_capacity(r);
    let mut zprime = Vec::with_capacity(r);
    for _ in 0..r {
        let half = eqv.len() / 2;
        let mut g = [F::ZERO; 4];
        for j in 0..half {
            let (eq_lo, eq_hi) = (eqv[2 * j], eqv[2 * j + 1]);
            let (a0_lo, a0_hi) = (a0[2 * j], a0[2 * j + 1]);
            let (a1_lo, a1_hi) = (a1[2 * j], a1[2 * j + 1]);
            let (b0_lo, b0_hi) = (b0[2 * j], b0[2 * j + 1]);
            let (b1_lo, b1_hi) = (b1[2 * j], b1[2 * j + 1]);
            for (t, &p) in pts.iter().enumerate() {
                let e = eq_lo + p * (eq_hi - eq_lo);
                let av0 = a0_lo + p * (a0_hi - a0_lo);
                let av1 = a1_lo + p * (a1_hi - a1_lo);
                let bv0 = b0_lo + p * (b0_hi - b0_lo);
                let bv1 = b1_lo + p * (b1_hi - b1_lo);
                g[t] += e * (av0 * bv1 + av1 * bv0 + lambda * bv0 * bv1);
            }
        }
        round_polys.push(g);
        let c = oracle.next_field();
        zprime.push(c);
        fold_low(&mut a0, c);
        fold_low(&mut a1, c);
        fold_low(&mut b0, c);
        fold_low(&mut b1, c);
        fold_low(&mut eqv, c);
    }
    let leaves = [a0[0], a1[0], b0[0], b1[0]];
    (LayerProof { round_polys, leaves }, zprime, leaves)
}

/// Verify a fraction-sum proof. Returns the reduced input claim `(point, p_in(point),
/// q_in(point))`, or `None` if any layer sumcheck fails. The caller must confirm the two
/// returned evaluations against the committed `p_in, q_in`.
pub fn verify<F: Field>(
    proof: &FracSumProof<F>,
    oracle: &mut RandomOracle<F>,
) -> Option<(Vec<F>, F, F)> {
    let k = proof.layers.len();
    let mut z: Vec<F> = Vec::new();
    let mut p_cur = proof.num;
    let mut q_cur = proof.den;
    for r in 0..k {
        let lambda = oracle.next_field();
        let claim_val = p_cur + lambda * q_cur;
        let layer = &proof.layers[r];
        if layer.round_polys.len() != r {
            return None;
        }
        let mut cur = claim_val;
        let mut zprime = Vec::with_capacity(r);
        for poly in &layer.round_polys {
            if poly[0] + poly[1] != cur {
                return None;
            }
            let c = oracle.next_field();
            cur = interp4(poly, c);
            zprime.push(c);
        }
        let [a0, a1, b0, b1] = layer.leaves;
        let eqf = eq_eval(&z, &zprime);
        if cur != eqf * (a0 * b1 + a1 * b0 + lambda * b0 * b1) {
            return None;
        }
        let beta = oracle.next_field();
        z = core::iter::once(beta).chain(zprime).collect();
        p_cur = (F::ONE - beta) * a0 + beta * a1;
        q_cur = (F::ONE - beta) * b0 + beta * b1;
    }
    Some((z, p_cur, q_cur))
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_goldilocks::Goldilocks;
    use rand::RngExt;
    use utils::poly::MlPoly;

    type EF = BinomialExtensionField<Goldilocks, 2>;

    fn frac_sum<F: Field>(p: &[F], q: &[F]) -> (F, F) {
        // Σ p_i/q_i as a single reduced fraction (num, den) by repeated addition.
        let mut num = F::ZERO;
        let mut den = F::ONE;
        for (&pi, &qi) in p.iter().zip(q) {
            num = num * qi + pi * den;
            den *= qi;
        }
        (num, den)
    }

    #[test]
    fn round_trip() {
        let mut rng = rand::rng();
        for k in 0..8 {
            let n = 1usize << k;
            let p: Vec<EF> = (0..n).map(|_| rng.random::<EF>()).collect();
            let q: Vec<EF> = (0..n).map(|_| rng.random::<EF>()).collect();

            let mut oracle = RandomOracle::<EF>::new(&mut rng);
            let (proof, point, pe, qe) = prove(&p, &q, &mut oracle);

            // Reduced root matches the honest fraction sum (cross-multiplied).
            let (num, den) = frac_sum(&p, &q);
            assert_eq!(proof.num * den, num * proof.den);

            oracle.restart();
            let (vpoint, vpe, vqe) = verify(&proof, &mut oracle).unwrap();
            assert_eq!(point, vpoint);
            assert_eq!((pe, qe), (vpe, vqe));

            // Input claim is the genuine MLE opening.
            assert_eq!(MlPoly(p.clone()).eval(&point), pe);
            assert_eq!(MlPoly(q.clone()).eval(&point), qe);
        }
    }

    #[test]
    fn tampered_root_rejected() {
        let mut rng = rand::rng();
        let n = 1usize << 6;
        let p: Vec<EF> = (0..n).map(|_| rng.random::<EF>()).collect();
        let q: Vec<EF> = (0..n).map(|_| rng.random::<EF>()).collect();
        let mut oracle = RandomOracle::<EF>::new(&mut rng);
        let (mut proof, _, _, _) = prove(&p, &q, &mut oracle);
        proof.num += EF::ONE;
        oracle.restart();
        // A bogus root fraction breaks the very first reduction's leaf consistency check.
        assert!(verify(&proof, &mut oracle).is_none());
    }

    #[test]
    fn tampered_round_rejected() {
        let mut rng = rand::rng();
        let n = 1usize << 6;
        let p: Vec<EF> = (0..n).map(|_| rng.random::<EF>()).collect();
        let q: Vec<EF> = (0..n).map(|_| rng.random::<EF>()).collect();
        let mut oracle = RandomOracle::<EF>::new(&mut rng);
        let (mut proof, _, _, _) = prove(&p, &q, &mut oracle);
        proof.layers[4].round_polys[2][1] += EF::ONE;
        oracle.restart();
        assert!(verify(&proof, &mut oracle).is_none());
    }
}
