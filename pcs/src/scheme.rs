//! A minimal polynomial-commitment-scheme abstraction plus a transparent placeholder.
//!
//! The trait mirrors the batched shape of [`crate::basefold::Basefold`] (`commit` →
//! prover-data + commitment; `batch_prove(tasks)`; `batch_verify(claims, proof)`) so a real
//! Basefold backend can be dropped in later with localized changes. The placeholder
//! [`PlaceholderPcs`] is *not* hiding anything: a commitment carries the full polynomial, the
//! opening proof is empty, and verification re-evaluates the committed polynomial directly. It
//! exists so callers can route every commit/open site through the PCS boundary while keeping the
//! transparent behavior of the current PIOP.

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

/// A transparent placeholder PCS: the commitment is the polynomial itself.
///
/// `commit` ships the full polynomial, the opening `Proof` is `()`, and `batch_verify` checks each
/// claim by evaluating the carried polynomial. It draws **no** oracle randomness, so the
/// Fiat-Shamir transcript is identical whether or not callers route through it. Swapping in
/// [`crate::basefold::Basefold`] (via `commit_ext`) is then a localized change.
#[derive(Debug, Clone, Default)]
pub struct PlaceholderPcs<F>(PhantomData<F>);

/// A placeholder commitment — the full polynomial, sent to the verifier.
#[derive(Debug, Clone)]
pub struct PlaceholderCommit<F: Field>(pub MlPoly<F>);

// `MlPoly` doesn't derive equality, so compare the carried evaluations directly.
impl<F: Field> PartialEq for PlaceholderCommit<F> {
    fn eq(&self, other: &Self) -> bool {
        self.0 .0 == other.0 .0
    }
}
impl<F: Field> Eq for PlaceholderCommit<F> {}

impl<F: Field> PolyCommitmentScheme for PlaceholderPcs<F> {
    type Field = F;
    type Commitment = PlaceholderCommit<F>;
    type ProverData = MlPoly<F>;
    type Proof = ();

    fn commit(poly: MlPoly<F>) -> (Self::ProverData, Self::Commitment) {
        (poly.clone(), PlaceholderCommit(poly))
    }

    fn batch_prove(
        _tasks: &[(&Self::ProverData, Vec<F>)],
        _oracle: &mut RandomOracle<F>,
    ) -> Self::Proof {
    }

    fn batch_verify(
        claims: &[(&Self::Commitment, Vec<F>, F)],
        _proof: &Self::Proof,
        _oracle: &mut RandomOracle<F>,
    ) -> bool {
        // `MlPoly::eval` consumes `self`, so evaluate a clone of the carried polynomial.
        claims
            .iter()
            .all(|(commit, point, value)| commit.0.clone().eval(point) == *value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_goldilocks::Goldilocks;

    type F = Goldilocks;
    type Pcs = PlaceholderPcs<F>;

    #[test]
    fn batch_open_round_trips_and_rejects_tamper() {
        let mut rng = rand::rng();
        let mut oracle = RandomOracle::<F>::new(&mut rng);

        // Two committed polynomials of different sizes, each opened at one point.
        let poly_a = MlPoly((0..8).map(|i| F::new(i as u64 + 1)).collect::<Vec<_>>());
        let poly_b = MlPoly((0..4).map(|i| F::new(7 * i as u64 + 3)).collect::<Vec<_>>());
        let point_a: Vec<F> = [5u64, 9, 2].into_iter().map(F::new).collect();
        let point_b: Vec<F> = [3u64, 11].into_iter().map(F::new).collect();

        let (data_a, commit_a) = Pcs::commit(poly_a.clone());
        let (data_b, commit_b) = Pcs::commit(poly_b.clone());
        let val_a = poly_a.clone().eval(&point_a);
        let val_b = poly_b.clone().eval(&point_b);

        let proof = Pcs::batch_prove(
            &[(&data_a, point_a.clone()), (&data_b, point_b.clone())],
            &mut oracle,
        );

        assert!(Pcs::batch_verify(
            &[
                (&commit_a, point_a.clone(), val_a),
                (&commit_b, point_b.clone(), val_b),
            ],
            &proof,
            &mut oracle,
        ));

        // A tampered claimed value is rejected.
        assert!(!Pcs::batch_verify(
            &[
                (&commit_a, point_a, val_a + F::new(1)),
                (&commit_b, point_b, val_b),
            ],
            &proof,
            &mut oracle,
        ));
    }
}
