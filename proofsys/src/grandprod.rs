//! Batched grand-product GKR (plan Phase 3).
//!
//! Proves, for a leaf cube viewed as `[batch = 2^batch_vars][prod = 2^col_vars]` (the product
//! dimension on the LOW variables, the batch dimension on the HIGH variables), that each batch's
//! product equals a **public** top value: `top[b] = ∏_k leaf[b·2^col + k]`. Only the `col_vars`
//! product layers are contracted; the batch dimension is carried as the free reduction point. The
//! result is a single leaf opening `leaf̃(point)` at a point the caller discharges against whatever
//! the leaf is built from.
//!
//! This is the denominator (product) tree of [`logup::frac_sum`] with the numerator dropped and the
//! reduction **stopped after `col_vars` layers**, seeded from a random batch point against the
//! public top vector (rather than reduced all the way to a scalar root). Conventions match
//! `frac_sum`: each layer binds the LOW bit first, so the returned point is low-bit-first and needs
//! no reversal.
//!
//! Soundness for "every real batch's product is 0": set the public top to 0 on real batches (and to
//! the honest value, typically 1, on padding batches). Because the top is public and the GKR binds
//! the leaf cube to it, a leaf whose real-batch product is nonzero cannot match the public top.

use p3_field::Field;
use utils::oracle::RandomOracle;

/// One layer reduction: the degree-3 sumcheck round polynomials (evals at `0,1,2,3`) plus the two
/// low-bit-split leaf evaluations `[A0, A1]` at the sumcheck point.
pub struct LayerProof<F: Field> {
    pub round_polys: Vec<[F; 4]>,
    pub leaves: [F; 2],
}

/// A grand-product proof: one [`LayerProof`] per contracted product layer (`col_vars` of them,
/// top→leaf).
pub struct GrandProdProof<F: Field> {
    pub layers: Vec<LayerProof<F>>,
}

impl<F: Field> GrandProdProof<F> {
    pub fn size_bytes(&self) -> usize {
        let f = core::mem::size_of::<F>();
        self.layers
            .iter()
            .map(|l| (l.round_polys.len() * 4 + 2) * f)
            .sum()
    }
}

/// `eq_table(z)[i] = ∏_k eq(z[k], bit_k(i))`, `z[0]` the low bit.
fn eq_table<F: Field>(z: &[F]) -> Vec<F> {
    let mut tab = vec![F::ONE];
    for &zk in z {
        let half = tab.len();
        let mut next = vec![F::ZERO; half * 2];
        for i in 0..half {
            next[i] = tab[i] * (F::ONE - zk);
            next[i + half] = tab[i] * zk;
        }
        tab = next;
    }
    tab
}

fn eq_eval<F: Field>(z: &[F], zp: &[F]) -> F {
    z.iter()
        .zip(zp)
        .fold(F::ONE, |acc, (&a, &b)| acc * (a * b + (F::ONE - a) * (F::ONE - b)))
}

/// `tab[i] = ∏_k eq(z[k], bit_k(i))` reused to evaluate a public multilinear `vals` at `z`.
fn eval_mle<F: Field>(vals: &[F], z: &[F]) -> F {
    let tab = eq_table(z);
    assert_eq!(tab.len(), vals.len());
    tab.iter().zip(vals).fold(F::ZERO, |acc, (&e, &v)| acc + e * v)
}

fn fold_low<F: Field>(arr: &mut Vec<F>, c: F) {
    let half = arr.len() / 2;
    for j in 0..half {
        arr[j] = arr[2 * j] + c * (arr[2 * j + 1] - arr[2 * j]);
    }
    arr.truncate(half);
}

fn interp4<F: Field>(evals: &[F; 4], x: F) -> F {
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

/// Prove the batched grand product. `leaf` has length `2^(batch_vars + col_vars)`; the product
/// dimension (`col_vars`) is the LOW variables. Returns the proof, the reduced leaf point
/// (low-bit-first, length `batch_vars + col_vars`), and `leaf̃(point)`.
pub fn prove<F: Field>(
    leaf: &[F],
    batch_vars: usize,
    oracle: &mut RandomOracle<F>,
) -> (GrandProdProof<F>, Vec<F>, F) {
    let total = leaf.len();
    assert!(total.is_power_of_two());
    let total_vars = total.trailing_zeros() as usize;
    let col_vars = total_vars - batch_vars;

    // Build the product tree: tree[0] = leaf, tree[c+1][i] = tree[c][2i]·tree[c][2i+1], pairing the
    // low bit. tree[col_vars] = top (size 2^batch_vars).
    let mut tree: Vec<Vec<F>> = Vec::with_capacity(col_vars + 1);
    tree.push(leaf.to_vec());
    for c in 0..col_vars {
        let cur = &tree[c];
        let mut next = vec![F::ZERO; cur.len() / 2];
        for i in 0..next.len() {
            next[i] = cur[2 * i] * cur[2 * i + 1];
        }
        tree.push(next);
    }

    let z0 = oracle.next_n_fields(batch_vars);
    let mut z = z0;
    let mut claim;
    let mut layers = Vec::with_capacity(col_vars);
    for step in 0..col_vars {
        // Current layer = tree[col_vars - step] at point z; its product children = tree[col_vars-step-1].
        let parent = &tree[col_vars - step - 1];
        let (layer, zp, leaves) = prove_layer(&z, parent, oracle);
        let beta = oracle.next_field();
        let [a0, a1] = leaves;
        claim = (F::ONE - beta) * a0 + beta * a1;
        z = core::iter::once(beta).chain(zp).collect();
        layers.push(layer);
        if step + 1 == col_vars {
            return (GrandProdProof { layers }, z, claim);
        }
    }
    // col_vars == 0: the leaf already is the top; return its eval at the batch point.
    let leaf_eval = eval_mle(leaf, &z);
    (GrandProdProof { layers }, z, leaf_eval)
}

/// The per-layer degree-3 product sumcheck `Σ_i eq(z,i)·A0[i]·A1[i]` over the `|z|` sumcheck
/// variables, where `A0[i] = parent[2i]`, `A1[i] = parent[2i+1]`.
fn prove_layer<F: Field>(
    z: &[F],
    parent: &[F],
    oracle: &mut RandomOracle<F>,
) -> (LayerProof<F>, Vec<F>, [F; 2]) {
    let r = z.len();
    let half0 = parent.len() / 2;
    debug_assert_eq!(half0, 1 << r);
    let mut a0: Vec<F> = (0..half0).map(|i| parent[2 * i]).collect();
    let mut a1: Vec<F> = (0..half0).map(|i| parent[2 * i + 1]).collect();
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
            for (t, &p) in pts.iter().enumerate() {
                let e = eq_lo + p * (eq_hi - eq_lo);
                let av0 = a0_lo + p * (a0_hi - a0_lo);
                let av1 = a1_lo + p * (a1_hi - a1_lo);
                g[t] += e * av0 * av1;
            }
        }
        round_polys.push(g);
        let c = oracle.next_field();
        zprime.push(c);
        fold_low(&mut a0, c);
        fold_low(&mut a1, c);
        fold_low(&mut eqv, c);
    }
    let leaves = [a0[0], a1[0]];
    (LayerProof { round_polys, leaves }, zprime, leaves)
}

/// Verify the batched grand product against the **public** `top` vector (length `2^batch_vars`).
/// Returns the reduced leaf point (low-bit-first) and `leaf̃(point)`, or `None` on any failure. The
/// caller must confirm the returned eval against whatever the leaf is built from.
pub fn verify<F: Field>(
    proof: &GrandProdProof<F>,
    top: &[F],
    oracle: &mut RandomOracle<F>,
) -> Option<(Vec<F>, F)> {
    assert!(top.len().is_power_of_two());
    let batch_vars = top.len().trailing_zeros() as usize;

    let z0 = oracle.next_n_fields(batch_vars);
    let mut cur = eval_mle(top, &z0);
    let mut z = z0;
    for (step, layer) in proof.layers.iter().enumerate() {
        if layer.round_polys.len() != batch_vars + step {
            return None;
        }
        let mut c_claim = cur;
        let mut zprime = Vec::with_capacity(layer.round_polys.len());
        for poly in &layer.round_polys {
            if poly[0] + poly[1] != c_claim {
                return None;
            }
            let c = oracle.next_field();
            c_claim = interp4(poly, c);
            zprime.push(c);
        }
        let [a0, a1] = layer.leaves;
        if c_claim != eq_eval(&z, &zprime) * a0 * a1 {
            return None;
        }
        let beta = oracle.next_field();
        cur = (F::ONE - beta) * a0 + beta * a1;
        z = core::iter::once(beta).chain(zprime).collect();
    }
    Some((z, cur))
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

    fn top_of(leaf: &[EF], batch_vars: usize) -> Vec<EF> {
        let col = leaf.len().trailing_zeros() as usize - batch_vars;
        let key_pow = 1usize << col;
        (0..(1usize << batch_vars))
            .map(|b| {
                (0..key_pow).fold(EF::ONE, |acc, k| acc * leaf[b * key_pow + k])
            })
            .collect()
    }

    #[test]
    fn round_trip() {
        let mut rng = rand::rng();
        for batch_vars in 0..4 {
            for col_vars in 0..5 {
                let n = 1usize << (batch_vars + col_vars);
                let leaf: Vec<EF> = (0..n).map(|_| rng.random::<EF>()).collect();
                let top = top_of(&leaf, batch_vars);
                let mut oracle = RandomOracle::<EF>::new(&mut rng);
                let (proof, point, leaf_eval) = prove(&leaf, batch_vars, &mut oracle);
                oracle.restart();
                let (vpoint, veval) = verify(&proof, &top, &mut oracle).unwrap();
                assert_eq!(point, vpoint, "b{batch_vars} c{col_vars}");
                assert_eq!(leaf_eval, veval);
                // The reduced eval is the genuine leaf MLE opening.
                assert_eq!(MlPoly(leaf.clone()).eval(&point), leaf_eval);
            }
        }
    }

    #[test]
    fn zero_product_batches() {
        // Each batch has a zero factor ⇒ top is all-zero; the GKR still reduces to a leaf opening.
        let mut rng = rand::rng();
        let (batch_vars, col_vars) = (3, 4);
        let key_pow = 1usize << col_vars;
        let mut leaf: Vec<EF> = (0..(1usize << (batch_vars + col_vars)))
            .map(|_| rng.random::<EF>())
            .collect();
        for b in 0..(1usize << batch_vars) {
            leaf[b * key_pow] = EF::ZERO; // a zero factor in every batch
        }
        let top = top_of(&leaf, batch_vars);
        assert!(top.iter().all(|&t| t == EF::ZERO));
        let mut oracle = RandomOracle::<EF>::new(&mut rng);
        let (proof, point, leaf_eval) = prove(&leaf, batch_vars, &mut oracle);
        oracle.restart();
        let (_, veval) = verify(&proof, &top, &mut oracle).unwrap();
        assert_eq!(leaf_eval, veval);
        assert_eq!(MlPoly(leaf).eval(&point), leaf_eval);
    }

    #[test]
    fn wrong_top_rejected() {
        // A top inconsistent with the leaf product is rejected (this is the soundness path: a
        // claimed public top of 0 on a batch whose real product is nonzero fails).
        let mut rng = rand::rng();
        let (batch_vars, col_vars) = (2, 3);
        let leaf: Vec<EF> = (0..(1usize << (batch_vars + col_vars)))
            .map(|_| rng.random::<EF>())
            .collect();
        let mut top = top_of(&leaf, batch_vars);
        top[1] = EF::ZERO; // claim batch 1's product is 0 when it isn't
        let mut oracle = RandomOracle::<EF>::new(&mut rng);
        let (proof, _, _) = prove(&leaf, batch_vars, &mut oracle);
        oracle.restart();
        // Either a layer check fails (None) or the returned eval disagrees with the real leaf.
        match verify(&proof, &top, &mut oracle) {
            None => {}
            Some((point, veval)) => {
                assert_ne!(MlPoly(leaf).eval(&point), veval);
            }
        }
    }

    #[test]
    fn tampered_round_rejected() {
        let mut rng = rand::rng();
        let (batch_vars, col_vars) = (2, 4);
        let leaf: Vec<EF> = (0..(1usize << (batch_vars + col_vars)))
            .map(|_| rng.random::<EF>())
            .collect();
        let top = top_of(&leaf, batch_vars);
        let mut oracle = RandomOracle::<EF>::new(&mut rng);
        let (mut proof, _, _) = prove(&leaf, batch_vars, &mut oracle);
        proof.layers[2].round_polys[1][0] += EF::ONE;
        oracle.restart();
        assert!(verify(&proof, &top, &mut oracle).is_none());
    }
}
