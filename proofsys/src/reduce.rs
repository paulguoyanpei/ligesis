//! Sumcheck-based reductions that emit opening claims against canonical commitments.
//!
//! Every reduction here commits **nothing**: it runs a sumcheck and turns the final per-factor
//! evaluations into [`crate::commit::OracleClaim`]s on already-committed polynomials, to be
//! discharged by the final PCS batch-open.
//!
//! ## Point conventions (critical)
//! All committed polynomials are evaluated **low-bit-first** (`MlPoly::eval`, `new_eq`,
//! `eq_weights`, the lookup query points). The sumcheck (`utils::sumcheck`) binds the **high**
//! variable each round, so its returned challenge point is **high-bit-first**. Therefore any
//! sumcheck point used to open a committed poly must be reversed first — see [`low_first`].

use utils::{
    oracle::RandomOracle,
    sumcheck::{self, SumcheckProof},
};

use crate::commit::{ClaimAccumulator, OracleSource};
use crate::protocol::EF;

/// Convert a sumcheck challenge point (high-bit-first) to the low-bit-first convention the PCS and
/// `MlPoly::eval` use.
pub fn low_first(sc: &[EF]) -> Vec<EF> {
    sc.iter().rev().copied().collect()
}

fn dot(a: &[EF], b: &[EF]) -> EF {
    use p3_field::PrimeCharacteristicRing;
    a.iter().zip(b).fold(EF::ZERO, |acc, (&x, &y)| acc + x * y)
}

#[derive(Clone, Copy)]
pub enum MatmulKind {
    /// `C[row,col] = sum_k A[row,k] * B[k,col]` with `B` stored `K x cols` (low bit = col).
    Regular,
    /// `C[row,col] = sum_k A[row,k] * B[col,k]` with `B` stored `cols x K` (low bit = k).
    Transposed,
}

/// A matmul reduction proof: the claimed `C(row_point,col_point)` and the contraction sumcheck.
/// The operand openings are emitted as claims into the accumulator, not stored here.
pub struct MatmulProof {
    pub claimed_eval: EF,
    pub proof: SumcheckProof<EF>,
}

/// Build the canonical opening points for `A` and `B` from a (low-bit-first) contraction point `k`,
/// per the matmul layout. `A` is `rows x K` (low bit = k); see [`MatmulKind`] for `B`.
fn operand_points(
    kind: MatmulKind,
    k: &[EF],
    row_point: &[EF],
    col_point: &[EF],
) -> (Vec<EF>, Vec<EF>) {
    let a_point = [k, row_point].concat();
    let b_point = match kind {
        MatmulKind::Regular => [col_point, k].concat(),
        MatmulKind::Transposed => [k, col_point].concat(),
    };
    (a_point, b_point)
}

/// Prove `C(row_point,col_point) = sum_k A(row_point,k)·B(k,col_point)` via the Thaler13 sumcheck
/// over the contraction variables, emitting opening claims on canonical `A`/`B`.
///
/// `avec[k] = Ã(row_point,k)` and `bvec[k] = B̃(k,col_point)` are the eq-contracted operand vectors
/// the prover computes; only the resulting `A`/`B` openings are committed (via the accumulator).
#[allow(clippy::too_many_arguments)]
pub fn prove_matmul(
    kind: MatmulKind,
    a_src: &OracleSource,
    b_src: &OracleSource,
    avec: Vec<EF>,
    bvec: Vec<EF>,
    row_point: &[EF],
    col_point: &[EF],
    acc: &mut ClaimAccumulator,
    oracle: &mut RandomOracle<EF>,
) -> MatmulProof {
    assert_eq!(avec.len(), bvec.len());
    let claimed_eval = dot(&avec, &bvec);
    let (proof, sc) = sumcheck::prove(vec![avec, bvec], oracle);
    let k = low_first(&sc);
    let (a_point, b_point) = operand_points(kind, &k, row_point, col_point);
    a_src.emit(acc, &a_point, proof.final_evals[0]);
    b_src.emit(acc, &b_point, proof.final_evals[1]);
    MatmulProof {
        claimed_eval,
        proof,
    }
}

/// Verify a matmul reduction: check the contraction sumcheck and emit the matching `A`/`B` opening
/// claims into the accumulator. Returns the (now reduced) `C(row_point,col_point)` value on
/// success, which downstream gadgets (e.g. the rescale remainder) consume.
#[allow(clippy::too_many_arguments)]
pub fn verify_matmul(
    kind: MatmulKind,
    a_src: &OracleSource,
    b_src: &OracleSource,
    row_point: &[EF],
    col_point: &[EF],
    proof: &MatmulProof,
    acc: &mut ClaimAccumulator,
    oracle: &mut RandomOracle<EF>,
) -> Option<EF> {
    let sc = sumcheck::verify(proof.claimed_eval, &proof.proof, oracle)?;
    if proof.proof.final_evals.len() != 2 {
        return None;
    }
    let k = low_first(&sc);
    let (a_point, b_point) = operand_points(kind, &k, row_point, col_point);
    a_src.emit(acc, &a_point, proof.proof.final_evals[0]);
    b_src.emit(acc, &b_point, proof.proof.final_evals[1]);
    Some(proof.claimed_eval)
}

/// An eq-weighted product reduction: proves `value = sum_x eq(eval_point,x)·L(x)·R(x)`
/// (= `L(eval_point)·R(eval_point)` since `eq` selects), reducing `L`/`R` to opening claims via
/// their [`OracleSource`]s. Used by Type-B division (`q·b`) and sqrt (`y²`, `x²`).
pub struct ProdProof {
    pub claimed_eval: EF,
    pub proof: SumcheckProof<EF>,
}

/// Prove `L(eval_point)·R(eval_point)` via the eq-weighted product sumcheck, emitting opening
/// claims for `L` and `R` through their sources. `left`/`right` are the prover's operand vectors.
pub fn prove_prod(
    left: Vec<EF>,
    right: Vec<EF>,
    left_src: &OracleSource,
    right_src: &OracleSource,
    eval_point: Vec<EF>,
    acc: &mut ClaimAccumulator,
    oracle: &mut RandomOracle<EF>,
) -> ProdProof {
    use utils::poly::MlPoly;
    assert_eq!(left.len(), right.len());
    assert_eq!(left.len(), 1 << eval_point.len());
    let product: Vec<EF> = left.iter().zip(&right).map(|(&l, &r)| l * r).collect();
    let claimed_eval = MlPoly(product).eval(&eval_point);
    let eq = MlPoly::new_eq(&eval_point).0;
    let (proof, sc) = sumcheck::prove(vec![eq, left, right], oracle);
    let open = low_first(&sc);
    left_src.emit(acc, &open, proof.final_evals[1]);
    right_src.emit(acc, &open, proof.final_evals[2]);
    ProdProof {
        claimed_eval,
        proof,
    }
}

/// Verify an eq-weighted product reduction: check the sumcheck and the public `eq` factor, then
/// emit the `L`/`R` opening claims. Returns the reduced `claimed_eval` on success.
pub fn verify_prod(
    eval_point: &[EF],
    left_src: &OracleSource,
    right_src: &OracleSource,
    proof: &ProdProof,
    acc: &mut ClaimAccumulator,
    oracle: &mut RandomOracle<EF>,
) -> Option<EF> {
    use utils::poly::MlPoly;
    let sc = sumcheck::verify(proof.claimed_eval, &proof.proof, oracle)?;
    if proof.proof.final_evals.len() != 3 {
        return None;
    }
    // The eq factor is public: its terminal eval must match.
    let eq_terminal = eval_sumcheck_order(MlPoly::new_eq(&eval_point.to_vec()).0, &sc);
    if proof.proof.final_evals[0] != eq_terminal {
        return None;
    }
    let open = low_first(&sc);
    left_src.emit(acc, &open, proof.proof.final_evals[1]);
    right_src.emit(acc, &open, proof.proof.final_evals[2]);
    Some(proof.claimed_eval)
}

/// Evaluate `evals` at `point` in the sumcheck (high-bit-first) fold order. Mirrors
/// `protocol::eval_sumcheck_order` for use by the public-factor checks here.
fn eval_sumcheck_order(mut evals: Vec<EF>, point: &[EF]) -> EF {
    assert_eq!(evals.len(), 1 << point.len());
    for &r in point {
        let half = evals.len() / 2;
        for i in 0..half {
            evals[i] = evals[i] + r * (evals[i + half] - evals[i]);
        }
        evals.truncate(half);
    }
    evals[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commit::{eq_at_index, CommitSet, PointMap, Pcs};
    use crate::tensor::Matrix;
    use crate::witness::encode_i64;
    use p3_field::PrimeCharacteristicRing;
    use pcs::PolyCommitmentScheme;
    use utils::poly::MlPoly;

    fn ef(x: i64) -> EF {
        EF::from(encode_i64(x))
    }

    /// Low-bit-first eq weights over `len` indices (matches `eq_weights` in protocol.rs).
    fn eq_weights(point: &[EF], len: usize) -> Vec<EF> {
        (0..len).map(|idx| eq_at_index(point, idx)).collect()
    }

    fn matrix_poly(m: &Matrix) -> Vec<EF> {
        // Row-major: low bits = column, high bits = row (dims assumed powers of two here).
        m.data().iter().map(|&v| ef(v)).collect()
    }

    #[test]
    fn regular_matmul_reduces_to_canonical_openings() {
        // A: 4x8, B: 8x2 (all powers of two so no padding).
        let rows = 4usize;
        let kdim = 8usize;
        let cols = 2usize;
        let a = Matrix::new(rows, kdim, (0..rows * kdim).map(|x| x as i64 - 5).collect());
        let b = Matrix::new(kdim, cols, (0..kdim * cols).map(|x| 2 * x as i64 - 3).collect());
        let c = a.matmul(&b);

        let row_vars = rows.trailing_zeros() as usize;
        let col_vars = cols.trailing_zeros() as usize;
        let row_point: Vec<EF> = (0..row_vars).map(|i| ef(7 + i as i64)).collect();
        let col_point: Vec<EF> = (0..col_vars).map(|i| ef(3 + i as i64)).collect();

        // Contraction vectors avec[k]=Ã(row,k), bvec[k]=B̃(k,col).
        let row_eq = eq_weights(&row_point, rows);
        let col_eq = eq_weights(&col_point, cols);
        let mut avec = vec![EF::ZERO; kdim];
        let mut bvec = vec![EF::ZERO; kdim];
        for k in 0..kdim {
            for r in 0..rows {
                avec[k] += row_eq[r] * ef(a.get(r, k));
            }
            for col in 0..cols {
                bvec[k] += col_eq[col] * ef(b.get(k, col));
            }
        }

        // Sanity: claimed C(row,col) equals MlPoly(C).eval(col ++ row) (low-bit-first).
        let c_point = [col_point.clone(), row_point.clone()].concat();
        let c_eval = MlPoly(matrix_poly(&c)).eval(&c_point);
        assert_eq!(dot(&avec, &bvec), c_eval, "claimed C mismatch");

        // Commit A, B canonically; run the reduction; discharge the emitted claims via the PCS.
        let mut set = CommitSet::new();
        set.commit("A", matrix_poly(&a));
        set.commit("B", matrix_poly(&b));

        let mut rng = rand::rng();
        let mut oracle = RandomOracle::<EF>::new(&mut rng);
        let a_src = OracleSource::committed("A", PointMap::identity(row_point.len() + 3));
        let b_src = OracleSource::committed("B", PointMap::identity(col_point.len() + 3));
        let mut acc = ClaimAccumulator::new();
        let mp = prove_matmul(
            MatmulKind::Regular,
            &a_src,
            &b_src,
            avec,
            bvec,
            &row_point,
            &col_point,
            &mut acc,
            &mut oracle,
        );
        assert_eq!(mp.claimed_eval, c_eval);

        // Verifier side re-derives the claims from the proof.
        oracle.restart();
        let mut vacc = ClaimAccumulator::new();
        let got = verify_matmul(
            MatmulKind::Regular,
            &a_src,
            &b_src,
            &row_point,
            &col_point,
            &mp,
            &mut vacc,
            &mut oracle,
        );
        assert_eq!(got, Some(c_eval));
        assert_eq!(vacc.claims.len(), 2);

        // The emitted claims must open the committed A/B through the PCS.
        let proof = {
            let tasks: Vec<_> = vacc
                .claims
                .iter()
                .map(|cl| (set.prover_data(&cl.oracle).unwrap(), cl.point.clone()))
                .collect();
            Pcs::batch_prove(&tasks, &mut oracle)
        };
        let claims: Vec<_> = vacc
            .claims
            .iter()
            .map(|cl| (set.commitment(&cl.oracle).unwrap(), cl.point.clone(), cl.value))
            .collect();
        assert!(Pcs::batch_verify(&claims, &proof, &mut oracle));

    }

    #[test]
    fn eq_product_reduces_to_canonical_openings() {
        // L, R committed directly (identity map, single instance). Check q·b style product.
        let l = vec![ef(2), ef(-3), ef(5), ef(7)];
        let r = vec![ef(1), ef(4), ef(-2), ef(6)];
        let vars = 2usize;
        let eval_point: Vec<EF> = (0..vars).map(|i| ef(11 + i as i64)).collect();

        // The prod sumcheck yields the pointwise-product MLE at eval_point (Σ_x eq·L·R), not
        // L(eval_point)·R(eval_point) — those agree only on the boolean cube.
        let product: Vec<EF> = l.iter().zip(&r).map(|(&a, &b)| a * b).collect();
        let expected = MlPoly(product).eval(&eval_point);

        let mut set = CommitSet::new();
        set.commit("L", l.clone());
        set.commit("R", r.clone());
        let l_src = OracleSource::committed("L", PointMap::identity(vars));
        let r_src = OracleSource::committed("R", PointMap::identity(vars));

        let mut rng = rand::rng();
        let mut oracle = RandomOracle::<EF>::new(&mut rng);
        let mut acc = ClaimAccumulator::new();
        let pp = prove_prod(
            l,
            r,
            &l_src,
            &r_src,
            eval_point.clone(),
            &mut acc,
            &mut oracle,
        );
        assert_eq!(pp.claimed_eval, expected);

        oracle.restart();
        let mut vacc = ClaimAccumulator::new();
        assert_eq!(
            verify_prod(&eval_point, &l_src, &r_src, &pp, &mut vacc, &mut oracle),
            Some(expected)
        );

        let proof = {
            let tasks: Vec<_> = vacc
                .claims
                .iter()
                .map(|cl| (set.prover_data(&cl.oracle).unwrap(), cl.point.clone()))
                .collect();
            Pcs::batch_prove(&tasks, &mut oracle)
        };
        let claims: Vec<_> = vacc
            .claims
            .iter()
            .map(|cl| (set.commitment(&cl.oracle).unwrap(), cl.point.clone(), cl.value))
            .collect();
        assert!(Pcs::batch_verify(&claims, &proof, &mut oracle));
    }

    #[test]
    fn transposed_matmul_reduces_to_canonical_openings() {
        // A: 4x8, B stored transposed as cols x K = 2x8 (so C = A · Bᵀ).
        let rows = 4usize;
        let kdim = 8usize;
        let cols = 2usize;
        let a = Matrix::new(rows, kdim, (0..rows * kdim).map(|x| 3 * x as i64 - 7).collect());
        let bt = Matrix::new(cols, kdim, (0..cols * kdim).map(|x| x as i64 - 4).collect());
        let c = a.matmul_transposed_rhs(&bt);

        let row_vars = rows.trailing_zeros() as usize;
        let col_vars = cols.trailing_zeros() as usize;
        let row_point: Vec<EF> = (0..row_vars).map(|i| ef(2 + i as i64)).collect();
        let col_point: Vec<EF> = (0..col_vars).map(|i| ef(9 + i as i64)).collect();

        let row_eq = eq_weights(&row_point, rows);
        let col_eq = eq_weights(&col_point, cols);
        let mut avec = vec![EF::ZERO; kdim];
        let mut bvec = vec![EF::ZERO; kdim];
        for k in 0..kdim {
            for r in 0..rows {
                avec[k] += row_eq[r] * ef(a.get(r, k));
            }
            for col in 0..cols {
                bvec[k] += col_eq[col] * ef(bt.get(col, k));
            }
        }

        let c_point = [col_point.clone(), row_point.clone()].concat();
        let c_eval = MlPoly(matrix_poly(&c)).eval(&c_point);
        assert_eq!(dot(&avec, &bvec), c_eval);

        let mut set = CommitSet::new();
        set.commit("A", matrix_poly(&a));
        set.commit("B", matrix_poly(&bt));

        let mut rng = rand::rng();
        let mut oracle = RandomOracle::<EF>::new(&mut rng);
        let a_src = OracleSource::committed("A", PointMap::identity(row_point.len() + 3));
        let b_src = OracleSource::committed("B", PointMap::identity(col_point.len() + 3));
        let mut acc = ClaimAccumulator::new();
        let mp = prove_matmul(
            MatmulKind::Transposed,
            &a_src,
            &b_src,
            avec,
            bvec,
            &row_point,
            &col_point,
            &mut acc,
            &mut oracle,
        );

        oracle.restart();
        let mut vacc = ClaimAccumulator::new();
        assert_eq!(
            verify_matmul(
                MatmulKind::Transposed,
                &a_src,
                &b_src,
                &row_point,
                &col_point,
                &mp,
                &mut vacc,
                &mut oracle,
            ),
            Some(c_eval)
        );

        let proof = {
            let tasks: Vec<_> = vacc
                .claims
                .iter()
                .map(|cl| (set.prover_data(&cl.oracle).unwrap(), cl.point.clone()))
                .collect();
            Pcs::batch_prove(&tasks, &mut oracle)
        };
        let claims: Vec<_> = vacc
            .claims
            .iter()
            .map(|cl| (set.commitment(&cl.oracle).unwrap(), cl.point.clone(), cl.value))
            .collect();
        assert!(Pcs::batch_verify(&claims, &proof, &mut oracle));
    }
}
