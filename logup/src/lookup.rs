//! GKR-LogUp lookup PIOP — prove every query `a_i` lies in table `t`.
//!
//! At a random `α` the membership reduces to `Σ_i 1/(α−a_i) = Σ_j e_j/(α−t_j)`. We run two
//! independent [`frac_sum`] circuits — one over the queries (numerators `1`, denominators
//! `α−a_i`), one over the table (numerators `e_j`, denominators `α−t_j`) — obtaining root
//! fractions `(S_q, T_q)` and `(S_t, T_t)`. The identity holds iff `S_q/T_q = S_t/T_t`, i.e.
//! `S_q·T_t = S_t·T_q`, a single scalar check.
//!
//! Each GKR pass also reduces its input to one MLE evaluation, giving exactly three
//! [`OpeningClaim`]s for the PCS: `a` (queries, size `m`), `e` (multiplicities, size `n`),
//! and `t` (table, size `n`). The query-side numerator is the constant `1`, checked
//! directly with no commitment.

use p3_field::Field;
use utils::oracle::RandomOracle;

use crate::frac_sum::{self, FracSumProof};

/// Which committed polynomial an [`OpeningClaim`] refers to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Poly {
    /// The query vector `a` (size `m`).
    Query,
    /// The table `t` (size `n`).
    Table,
    /// The multiplicity vector `e` (size `n`).
    Mult,
}

/// An evaluation the PCS must confirm: `poly(point) == value`.
#[derive(Clone, Debug)]
pub struct OpeningClaim<F: Field> {
    pub poly: Poly,
    pub point: Vec<F>,
    pub value: F,
}

/// A GKR-LogUp lookup proof: the two fraction-sum passes.
pub struct LookupProof<F: Field> {
    pub query: FracSumProof<F>,
    pub table: FracSumProof<F>,
}

impl<F: Field> LookupProof<F> {
    /// Serialized PIOP size in bytes (excludes the PCS opening, batched separately).
    pub fn size_bytes(&self) -> usize {
        self.query.size_bytes() + self.table.size_bytes()
    }
}

/// Convert a `u64` to a field element (`F: Field` needn't expose a direct cast).
fn from_u64<F: Field>(mut x: u64) -> F {
    let mut acc = F::ZERO;
    let mut base = F::ONE;
    while x > 0 {
        if x & 1 == 1 {
            acc += base;
        }
        base += base;
        x >>= 1;
    }
    acc
}

/// Multiplicity of each table entry among the queries, by value match. Assumes distinct
/// table entries. Linear-scan (`O(mn)`); witness-gen only — optimize from query indices.
fn multiplicities<F: Field>(a: &[F], t: &[F]) -> Vec<F> {
    let mut e = vec![0u64; t.len()];
    for &ai in a {
        if let Some(j) = t.iter().position(|&tj| tj == ai) {
            e[j] += 1;
        }
        // A query not in the table leaves the LHS with an extra factor ⇒ verification fails.
    }
    e.into_iter().map(from_u64).collect()
}

/// The multiplicity vector `e` (field counts), exposed so the PCS / tests can commit it.
pub fn multiplicity_vector<F: Field>(a: &[F], t: &[F]) -> Vec<F> {
    multiplicities(a, t)
}

/// Prove that every `a_i ∈ t`. Both lengths must be powers of two; `t` must have distinct
/// entries. Returns the proof; [`verify`] reproduces the reduction and emits the openings.
pub fn prove<F: Field>(a: &[F], t: &[F], oracle: &mut RandomOracle<F>) -> LookupProof<F> {
    let e = multiplicities(a, t);
    prove_with_mults(a, t, &e, oracle)
}

/// Like [`prove`] but with the multiplicity vector supplied (e.g. computed cheaply from
/// query *indices* instead of the `O(mn)` value scan). `e` must be the genuine per-entry
/// counts, or the proof will not verify.
pub fn prove_with_mults<F: Field>(
    a: &[F],
    t: &[F],
    e: &[F],
    oracle: &mut RandomOracle<F>,
) -> LookupProof<F> {
    prove_with_query_point(a, t, e, oracle).0
}

/// Like [`prove_with_mults`] but also returns the query-side input point `zq` (low-bit-first)
/// at which the query MLE is opened. A gadget whose query is a *virtual* poly (e.g. a
/// division remainder `r = a − q·b`) needs `zq` to bind that reconstruction at the same
/// point the lookup opens — the verifier recovers the identical point from the returned
/// `Poly::Query` claim.
pub fn prove_with_query_point<F: Field>(
    a: &[F],
    t: &[F],
    e: &[F],
    oracle: &mut RandomOracle<F>,
) -> (LookupProof<F>, Vec<F>) {
    let (proof, query_point, _) = prove_with_opening_points(a, t, e, oracle);
    (proof, query_point)
}

/// As [`prove_with_query_point`], additionally returning the table-side opening point.  A PCS
/// integration needs that point to open an offline committed table after the GKR transcript.
pub fn prove_with_opening_points<F: Field>(
    a: &[F],
    t: &[F],
    e: &[F],
    oracle: &mut RandomOracle<F>,
) -> (LookupProof<F>, Vec<F>, Vec<F>) {
    assert!(a.len().is_power_of_two() && t.len().is_power_of_two());
    assert_eq!(e.len(), t.len());

    let alpha = oracle.next_field();

    // Query side: ∑ 1/(α − a_i).
    let pq = vec![F::ONE; a.len()];
    let qq: Vec<F> = a.iter().map(|&ai| alpha - ai).collect();
    let (query, zq, _pq_in, _qq_in) = frac_sum::prove(&pq, &qq, oracle);

    // Table side: ∑ e_j/(α − t_j).
    let qt: Vec<F> = t.iter().map(|&tj| alpha - tj).collect();
    let (table, zt, _pt_in, _qt_in) = frac_sum::prove(e, &qt, oracle);

    (LookupProof { query, table }, zq, zt)
}

/// Verify a lookup proof. Returns the [`OpeningClaim`]s the PCS must confirm, or `None` if
/// any internal check fails (the two GKR passes, the constant-`1` query numerator, or the
/// cross-multiplied root equality).
pub fn verify<F: Field>(
    proof: &LookupProof<F>,
    oracle: &mut RandomOracle<F>,
) -> Option<Vec<OpeningClaim<F>>> {
    let alpha = oracle.next_field();

    let (zq, pq_in, qq_in) = frac_sum::verify(&proof.query, oracle)?;
    let (zt, pt_in, qt_in) = frac_sum::verify(&proof.table, oracle)?;

    // Query numerator is the constant 1 ⇒ its MLE is 1 everywhere; no commitment needed.
    if pq_in != F::ONE {
        return None;
    }

    // Cross-multiplied root equality S_q/T_q = S_t/T_t.
    if proof.query.num * proof.table.den != proof.table.num * proof.query.den {
        return None;
    }

    // Discharge the input denominators / numerator to committed openings:
    //   qq_in = α − a(zq)  ⇒  a(zq) = α − qq_in
    //   pt_in = e(zt)
    //   qt_in = α − t(zt)  ⇒  t(zt) = α − qt_in
    Some(vec![
        OpeningClaim { poly: Poly::Query, point: zq, value: alpha - qq_in },
        OpeningClaim { poly: Poly::Mult, point: zt.clone(), value: pt_in },
        OpeningClaim { poly: Poly::Table, point: zt, value: alpha - qt_in },
    ])
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

    /// Discharge the returned opening claims directly against the witness MLEs.
    fn check_openings(claims: &[OpeningClaim<EF>], a: &[EF], t: &[EF], e: &[EF]) -> bool {
        claims.iter().all(|c| {
            let poly = match c.poly {
                Poly::Query => a,
                Poly::Table => t,
                Poly::Mult => e,
            };
            MlPoly(poly.to_vec()).eval(&c.point) == c.value
        })
    }

    fn sample(m: usize, n: usize, rng: &mut impl rand::Rng) -> (Vec<EF>, Vec<EF>, Vec<EF>) {
        let t: Vec<EF> = (0..n).map(|_| rng.random::<EF>()).collect();
        let a: Vec<EF> = (0..m).map(|_| t[rng.random::<u64>() as usize % n]).collect();
        let e = multiplicity_vector(&a, &t);
        (a, t, e)
    }

    #[test]
    fn lookup_round_trip() {
        let mut rng = rand::rng();
        for (mu, nu) in [(0, 0), (4, 2), (8, 4), (10, 6)] {
            let (a, t, e) = sample(1 << mu, 1 << nu, &mut rng);
            let mut oracle = RandomOracle::<EF>::new(&mut rng);
            let proof = prove(&a, &t, &mut oracle);
            oracle.restart();
            let claims = verify(&proof, &mut oracle).expect("valid lookup verifies");
            assert!(check_openings(&claims, &a, &t, &e));
        }
    }

    #[test]
    fn non_member_rejected() {
        let mut rng = rand::rng();
        let (mut a, t, _e) = sample(1 << 8, 1 << 4, &mut rng);
        // Replace one query with a value not in the table.
        a[3] = rng.random::<EF>();
        let e = multiplicity_vector(&a, &t); // a[3] contributes nothing
        let mut oracle = RandomOracle::<EF>::new(&mut rng);
        let proof = prove_with_mults(&a, &t, &e, &mut oracle);
        oracle.restart();
        // The cross-multiplied root check must fail.
        assert!(verify(&proof, &mut oracle).is_none());
    }

    #[test]
    fn wrong_multiplicity_rejected() {
        let mut rng = rand::rng();
        let (a, t, mut e) = sample(1 << 8, 1 << 4, &mut rng);
        e[2] += EF::ONE; // claim one extra hit on t[2]
        let mut oracle = RandomOracle::<EF>::new(&mut rng);
        let proof = prove_with_mults(&a, &t, &e, &mut oracle);
        oracle.restart();
        assert!(verify(&proof, &mut oracle).is_none());
    }

    #[test]
    fn openings_via_basefold() {
        use pcs::basefold::Basefold;

        let mut rng = rand::rng();
        let (a, t, e) = sample(1 << 10, 1 << 6, &mut rng);
        let mut oracle = RandomOracle::<EF>::new(&mut rng);
        let proof = prove(&a, &t, &mut oracle);
        oracle.restart();
        let claims = verify(&proof, &mut oracle).expect("valid lookup verifies");

        // Confirm each opening claim is consistent with a Basefold commitment of the witness.
        for c in &claims {
            let poly = match c.poly {
                Poly::Query => &a,
                Poly::Table => &t,
                Poly::Mult => &e,
            };
            let (st, commit) =
                Basefold::<Goldilocks, EF>::commit_ext_on_domain(MlPoly(poly.to_vec()), 1, 0);
            let mut po = RandomOracle::<EF>::new(&mut rng);
            let pf = Basefold::<Goldilocks, EF>::prove::<EF>(&st, c.point.clone(), &mut po);
            assert_eq!(pf.claimed_eval(), c.value);
            po.restart();
            assert!(Basefold::<Goldilocks, EF>::verify::<EF>(&commit, c.point.clone(), &pf, &mut po));
        }
    }
}
