//! The polynomial-commitment-scheme abstraction plus a no-op backend.
//!
//! The [`PolyCommitmentScheme`] trait mirrors the batched shape of [`crate::basefold::Basefold`]
//! (`commit` → prover-data + commitment; `batch_prove(tasks)`; `batch_verify(claims, proof)`) so a
//! real Basefold backend can be dropped in later with localized changes.
//!
//! [`NoopPcs`] is a measurement backend: the commitment carries **nothing** and `batch_verify`
//! accepts **unconditionally**. The prover still retains the full polynomial in `ProverData` (the
//! PIOP prover evaluates it to produce its opening claims), but the verifier-visible commitment is
//! empty and the opening proof is `()`. It is therefore **not sound** — it exists only to benchmark
//! the PIOP transcript size and the verifier time without any PCS opening work. It draws no oracle
//! randomness, so the Fiat-Shamir transcript is identical whether or not callers route through it.

use core::marker::PhantomData;

use p3_field::Field;
use utils::{oracle::RandomOracle, poly::MlPoly};

/// A polynomial commitment scheme over multilinear polynomials in the field `Field`.
///
/// Points are low-bit-first, matching [`MlPoly::eval`]. The scheme is batched: several committed
/// polynomials, each opened at a single point, are proven and verified together.
pub trait PolyCommitmentScheme {
    /// The field the committed polynomials and challenges live in.
    type Field: Field;
    /// The verifier-visible commitment.
    type Commitment;
    /// Prover-side state retained after committing, consumed by [`Self::batch_prove`].
    type ProverData;
    /// The batched opening proof.
    type Proof;

    /// Commit to a multilinear polynomial (low-bit-first evaluations).
    fn commit(poly: MlPoly<Self::Field>) -> (Self::ProverData, Self::Commitment);

    /// Batch-open committed polynomials, each at one point. `tasks[i] = (prover_data, point)`.
    fn batch_prove(
        tasks: &[(&Self::ProverData, Vec<Self::Field>)],
        oracle: &mut RandomOracle<Self::Field>,
    ) -> Self::Proof;

    /// Batch-verify opening claims. `claims[i] = (commitment, point, claimed value)`.
    fn batch_verify(
        claims: &[(&Self::Commitment, Vec<Self::Field>, Self::Field)],
        proof: &Self::Proof,
        oracle: &mut RandomOracle<Self::Field>,
    ) -> bool;
}

/// A no-op PCS for measuring the PIOP cost in isolation: the commitment carries **nothing** and
/// `batch_verify` accepts **unconditionally**. See the module docs — it is *not* sound.
#[derive(Debug, Clone, Default)]
pub struct NoopPcs<F>(PhantomData<F>);

/// An empty commitment — nothing is sent to the verifier.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NoopCommit;

impl<F: Field> PolyCommitmentScheme for NoopPcs<F> {
    type Field = F;
    type Commitment = NoopCommit;
    type ProverData = MlPoly<F>;
    type Proof = ();

    fn commit(poly: MlPoly<F>) -> (Self::ProverData, Self::Commitment) {
        (poly, NoopCommit)
    }

    fn batch_prove(_tasks: &[(&Self::ProverData, Vec<F>)], _oracle: &mut RandomOracle<F>) {}

    fn batch_verify(
        _claims: &[(&Self::Commitment, Vec<F>, F)],
        _proof: &Self::Proof,
        _oracle: &mut RandomOracle<F>,
    ) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_goldilocks::Goldilocks;

    type F = Goldilocks;
    type Pcs = NoopPcs<F>;

    #[test]
    fn commit_is_empty_and_verify_always_accepts() {
        let mut rng = rand::rng();
        let mut oracle = RandomOracle::<F>::new(&mut rng);

        let poly = MlPoly((0..8).map(|i| F::new(i as u64 + 1)).collect::<Vec<_>>());
        let point: Vec<F> = [5u64, 9, 2].into_iter().map(F::new).collect();
        let (data, commit) = Pcs::commit(poly.clone());

        // The prover retains the polynomial; the commitment is the empty unit.
        assert_eq!(data.0, poly.0);
        assert_eq!(commit, NoopCommit);

        let proof = Pcs::batch_prove(&[(&data, point.clone())], &mut oracle);
        // `batch_verify` accepts unconditionally — even a bogus claimed value (it binds nothing).
        let bogus = poly.eval(&point) + F::new(1);
        assert!(Pcs::batch_verify(&[(&commit, point, bogus)], &proof, &mut oracle));
    }
}
