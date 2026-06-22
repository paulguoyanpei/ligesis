//! Commitment + opening-claim infrastructure for the faithful PIOP.
//!
//! The protocol commits a fixed set of polynomials **once** (online: the merged witness types,
//! limb polys, and the lookup multiplicity `e`; offline: the table columns and weights), then runs
//! every sumcheck/lookup **without committing anything more** — each reduces to opening *claims*
//! `(oracle, point, value)` against the committed set, discharged by a single PCS batch-open at the
//! end. This module provides the commit backend (over [`pcs::PolyCommitmentScheme`]), the canonical
//! merged layout (instance index on the high-order variables, element index on the low-order
//! variables), the point-maps that reduce a per-instance/sub-cube view onto a merged commitment,
//! and the claim accumulator.

use std::collections::BTreeMap;

use p3_field::PrimeCharacteristicRing;
use pcs::PolyCommitmentScheme;
use utils::poly::MlPoly;

use crate::protocol::EF;

/// The PCS backend: [`pcs::NoopPcs`] — an empty commitment whose `batch_verify` always accepts.
/// This isolates the PIOP transcript size / verifier time (what the bench measures); it is **not**
/// sound (it discharges no opening), so it does not bind the committed polynomials. Swappable for a
/// real `pcs::basefold::Basefold` later.
pub type Pcs = pcs::NoopPcs<EF>;
pub type Commitment = pcs::NoopCommit;
pub type ProverData = <Pcs as PolyCommitmentScheme>::ProverData;

/// Stack per-instance evaluation vectors into one merged MLE: the element index occupies the
/// low-order variables, the instance index the high-order variables. Each instance is zero-padded
/// to `elem_len` (a power of two) and the instance count is zero-padded to a power of two, so the
/// merged poly has `log2(elem_len) + log2(#instances padded)` variables.
pub fn stack_instances(instances: &[Vec<EF>], elem_len: usize) -> Vec<EF> {
    assert!(elem_len.is_power_of_two());
    let k = instances.len().max(1).next_power_of_two();
    let mut out = vec![EF::ZERO; k * elem_len];
    for (i, inst) in instances.iter().enumerate() {
        assert!(inst.len() <= elem_len, "instance longer than elem_len");
        out[i * elem_len..i * elem_len + inst.len()].copy_from_slice(inst);
    }
    out
}

/// The point opening instance `j`'s slice of a merged poly: `elem_point ++ bits(j)`, with the
/// instance bits low-bit-first over the high-order variables.
pub fn instance_slice_point(elem_point: &[EF], inst: usize, inst_vars: usize) -> Vec<EF> {
    let mut p = elem_point.to_vec();
    for bit in 0..inst_vars {
        p.push(if (inst >> bit) & 1 == 1 {
            EF::ONE
        } else {
            EF::ZERO
        });
    }
    p
}

/// `eq(point, idx)` low-bit-first — the multilinear selector that picks hypercube vertex `idx`.
pub fn eq_at_index(point: &[EF], idx: usize) -> EF {
    let mut acc = EF::ONE;
    for (bit, &r) in point.iter().enumerate() {
        acc *= if (idx >> bit) & 1 == 1 {
            r
        } else {
            EF::ONE - r
        };
    }
    acc
}

/// A canonical-poly coordinate as an affine function of an operand point: either a fixed boolean
/// bit, or "equals operand variable `i`". To keep the composed map multilinear, each operand
/// variable must be referenced **at most once** across a [`PointMap`].
#[derive(Clone, Copy, Debug)]
pub enum Coord {
    Const(bool),
    Var(usize),
}

/// Maps an operand evaluation point (low-bit-first) to a canonical commitment's point. Used to
/// express head-slices, transposes, instance-fixing, and broadcasts/projections as a relabeling of
/// variables.
#[derive(Clone, Debug)]
pub struct PointMap(pub Vec<Coord>);

impl PointMap {
    pub fn apply(&self, operand_point: &[EF]) -> Vec<EF> {
        self.0
            .iter()
            .map(|c| match *c {
                Coord::Const(false) => EF::ZERO,
                Coord::Const(true) => EF::ONE,
                Coord::Var(i) => operand_point[i],
            })
            .collect()
    }

    /// `Var(0), Var(1), ..., Var(n-1)` — the identity relabeling on `n` operand variables.
    pub fn identity(n: usize) -> Self {
        PointMap((0..n).map(Coord::Var).collect())
    }

    /// Append constant bits encoding `idx` (low-bit-first) — e.g. to fix an instance slice.
    pub fn with_const_suffix(mut self, idx: usize, bits: usize) -> Self {
        for bit in 0..bits {
            self.0.push(Coord::Const((idx >> bit) & 1 == 1));
        }
        self
    }
}

/// A public affine bias added to a virtual operand: `+ MlPoly(values).eval(operand_point[vars])`,
/// where `vars` selects (in order) the operand variables the bias depends on. `values` holds
/// `2^{vars.len()}` low-bit-first evaluations.
#[derive(Clone, Debug)]
pub struct BiasPoly {
    pub values: Vec<EF>,
    pub vars: Vec<usize>,
}

impl BiasPoly {
    pub fn eval(&self, operand_point: &[EF]) -> EF {
        let sub: Vec<EF> = self.vars.iter().map(|&i| operand_point[i]).collect();
        MlPoly(self.values.clone()).eval(&sub)
    }
}

/// How a virtual operand polynomial reduces to a canonical commitment: an opening of `oracle` at
/// `map.apply(operand_point)`, plus an optional public `bias`. The operand value at a point is
/// `canonical(map(point)) + bias(point)`; the emitted canonical claim therefore carries value
/// `operand_value - bias(point)`.
#[derive(Clone, Debug)]
pub struct OracleSource {
    pub oracle: String,
    pub map: PointMap,
    pub bias: Option<BiasPoly>,
}

impl OracleSource {
    pub fn committed(oracle: impl Into<String>, map: PointMap) -> Self {
        Self {
            oracle: oracle.into(),
            map,
            bias: None,
        }
    }

    pub fn with_bias(mut self, bias: BiasPoly) -> Self {
        self.bias = Some(bias);
        self
    }

    /// Public bias contribution at `operand_point`.
    pub fn bias_at(&self, operand_point: &[EF]) -> EF {
        self.bias.as_ref().map_or(EF::ZERO, |b| b.eval(operand_point))
    }

    /// Emit the canonical opening claim for an `operand_value` observed at `operand_point` (the
    /// value includes the public bias; the claim subtracts it).
    pub fn emit(&self, acc: &mut ClaimAccumulator, operand_point: &[EF], operand_value: EF) {
        let point = self.map.apply(operand_point);
        acc.open(
            self.oracle.clone(),
            point,
            operand_value - self.bias_at(operand_point),
        );
    }
}

/// One opening claim against a named committed oracle.
#[derive(Clone, Debug)]
pub struct OracleClaim {
    pub oracle: String,
    pub point: Vec<EF>,
    pub value: EF,
}

/// Accumulates opening claims emitted during the protocol; discharged once at the end.
#[derive(Default)]
pub struct ClaimAccumulator {
    pub claims: Vec<OracleClaim>,
}

impl ClaimAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that committed oracle `oracle` evaluates to `value` at `point`.
    pub fn open(&mut self, oracle: impl Into<String>, point: Vec<EF>, value: EF) {
        self.claims.push(OracleClaim {
            oracle: oracle.into(),
            point,
            value,
        });
    }
}

/// The committed polynomials, by name. Prover holds `(ProverData, Commitment)`; the verifier holds
/// only `Commitment`. Built once during the commit phase.
#[derive(Default)]
pub struct CommitSet {
    prover: BTreeMap<String, ProverData>,
    commitments: BTreeMap<String, Commitment>,
}

impl CommitSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Commit `poly` under `name`, retaining prover data and the commitment.
    pub fn commit(&mut self, name: impl Into<String>, poly: Vec<EF>) {
        let name = name.into();
        let (data, commitment) = Pcs::commit(MlPoly(poly));
        self.prover.insert(name.clone(), data);
        self.commitments.insert(name, commitment);
    }

    /// Absorb another set's commitments (e.g. merging offline weights into the online set).
    pub fn extend(&mut self, other: CommitSet) {
        self.prover.extend(other.prover);
        self.commitments.extend(other.commitments);
    }

    pub fn commitment(&self, name: &str) -> Option<&Commitment> {
        self.commitments.get(name)
    }

    pub fn prover_data(&self, name: &str) -> Option<&ProverData> {
        self.prover.get(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use utils::oracle::RandomOracle;

    fn poly(vals: &[u64]) -> Vec<EF> {
        vals.iter().map(|&v| EF::from(p3_goldilocks::Goldilocks::new(v))).collect()
    }

    #[test]
    fn fixed_instance_slice_and_eq_combination_round_trip() {
        // Three instances of 4 elements each (elem_vars = 2); instance count padded to 4
        // (inst_vars = 2). Merged poly has 4 variables.
        let elem_len = 4;
        let w: Vec<Vec<EF>> = vec![
            poly(&[1, 2, 3, 4]),
            poly(&[5, 6, 7, 8]),
            poly(&[9, 10, 11, 12]),
        ];
        let merged = MlPoly(stack_instances(&w, elem_len));
        let inst_vars = 2;
        let elem_point = poly(&[13, 17]); // arbitrary

        // Fixed-instance slice: merged(elem_point ++ bits(j)) == w_j(elem_point).
        for (j, wj) in w.iter().enumerate() {
            let sliced = merged
                .clone()
                .eval(&instance_slice_point(&elem_point, j, inst_vars));
            assert_eq!(sliced, MlPoly(wj.clone()).eval(&elem_point), "slice {j}");
        }

        // eq-combination: merged(elem_point ++ inst_point) == sum_j eq(inst_point, j) w_j(elem).
        let inst_point = poly(&[19, 23]);
        let mut full = elem_point.clone();
        full.extend_from_slice(&inst_point);
        let lhs = merged.clone().eval(&full);
        let rhs = (0..(1usize << inst_vars))
            .map(|j| {
                let wj = w.get(j).cloned().unwrap_or_else(|| vec![EF::ZERO; elem_len]);
                eq_at_index(&inst_point, j) * MlPoly(wj).eval(&elem_point)
            })
            .fold(EF::ZERO, |a, b| a + b);
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn oracle_source_instance_slice_plus_bias_reduces_correctly() {
        // Canonical W: 2 instances of a 2(col)×2(row) block; col is the low bit, row the next,
        // instance the high bit (3 vars total).
        let elem_len = 4;
        let w0 = poly(&[1, 2, 3, 4]);
        let w1 = poly(&[5, 6, 7, 8]);
        let merged = MlPoly(stack_instances(&[w0.clone(), w1.clone()], elem_len));

        // Operand = instance-1 slice + a public bias that depends on the column (low) variable.
        let bias_vals = poly(&[10, 20]);
        let src = OracleSource::committed("W", PointMap::identity(2).with_const_suffix(1, 1))
            .with_bias(BiasPoly {
                values: bias_vals.clone(),
                vars: vec![0],
            });

        let operand_point = poly(&[6, 9]); // (col_var, row_var)
        let operand_value = MlPoly(w1.clone()).eval(&operand_point)
            + MlPoly(bias_vals).eval(&operand_point[..1].to_vec());

        let mut acc = ClaimAccumulator::new();
        src.emit(&mut acc, &operand_point, operand_value);
        assert_eq!(acc.claims.len(), 1);
        let claim = &acc.claims[0];
        assert_eq!(claim.oracle, "W");
        // The claim must be a true opening of the committed merged poly.
        assert_eq!(merged.clone().eval(&claim.point), claim.value);
        // And it equals the instance-1 slice (bias removed).
        assert_eq!(claim.value, MlPoly(w1).eval(&operand_point));
    }

    #[test]
    fn commit_set_round_trips_through_pcs() {
        let mut rng = rand::rng();
        let mut oracle = RandomOracle::<EF>::new(&mut rng);
        let mut set = CommitSet::new();
        let p = poly(&[3, 1, 4, 1, 5, 9, 2, 6]);
        set.commit("q_ln", p.clone());
        let point = poly(&[2, 7, 1]);
        let value = MlPoly(p).eval(&point);

        let proof = Pcs::batch_prove(
            &[(set.prover_data("q_ln").unwrap(), point.clone())],
            &mut oracle,
        );
        assert!(Pcs::batch_verify(
            &[(set.commitment("q_ln").unwrap(), point, value)],
            &proof,
            &mut oracle,
        ));
    }
}
