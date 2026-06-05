//! Subset-sum (binary-SIS) hash over the base field, the core commitment primitive of
//! LigeSIS (paper §3). A public matrix `A ∈ F^{c × W}` hashes a binary vector `x ∈ {0,1}^W`
//! to `A·x ∈ F^c` — i.e. the sum of the columns of `A` selected by the 1-bits of `x`. The
//! hash is *linear*, so per-column digests of an encoded matrix can later be aggregated and
//! evaluated by the secondary PCS.
//!
//! Here we apply it column-wise to the bit-decomposed, RS-encoded matrix `B ∈ {0,1}^{W × 2n}`
//! (`W = η·m`), yielding the digest matrix `H = A·B ∈ F^{c × 2n}`.
//!
//! The paper's bounded-domain sampling / `Z_{2^64}` overflow handling and the Figure-1
//! preprocessing-table acceleration are performance optimizations and are intentionally out
//! of scope here: `A` is sampled uniformly over the field and hashing is plain field addition.

use p3_field::Field;
use rand::distr::{Distribution, StandardUniform};
use rand::{Rng, RngExt};

/// `H` height (number of digest field elements per column); paper parameter `c`, updated to 32.
pub const C: usize = 32;

/// A subset-sum hash instance: the public matrix `A ∈ F^{C × width}`, stored row-major.
#[derive(Debug, Clone)]
pub struct SubsetSumHash<F: Field> {
    /// `a[r]` is row `r` of `A`, length `width`.
    pub a: Vec<Vec<F>>,
    pub width: usize,
}

impl<F: Field> SubsetSumHash<F> {
    /// Sample `A ∈ F^{C × width}` uniformly at random.
    pub fn new<R: Rng + ?Sized>(width: usize, rng: &mut R) -> Self
    where
        StandardUniform: Distribution<F>,
    {
        let a = (0..C)
            .map(|_| (0..width).map(|_| rng.random()).collect())
            .collect();
        Self { a, width }
    }

    /// Hash one binary column `x ∈ {0,1}^width` to `A·x ∈ F^C`: sum the columns of `A`
    /// selected by the set bits of `x`.
    pub fn hash_column(&self, x: &[bool]) -> Vec<F> {
        debug_assert_eq!(x.len(), self.width);
        let mut out = vec![F::ZERO; C];
        for (i, &bit) in x.iter().enumerate() {
            if bit {
                for r in 0..C {
                    out[r] += self.a[r][i];
                }
            }
        }
        out
    }

    /// Hash every column, returning the digest matrix `H = A·B ∈ F^{C × cols.len()}`,
    /// row-major (`h[r][j]`).
    pub fn hash_columns(&self, cols: &[Vec<bool>]) -> Vec<Vec<F>> {
        let mut h = vec![vec![F::ZERO; cols.len()]; C];
        for (j, col) in cols.iter().enumerate() {
            let d = self.hash_column(col);
            for r in 0..C {
                h[r][j] = d[r];
            }
        }
        h
    }

    /// `Ā` as a flat multilinear: `flat[r * width + i] = A[r][i]`, length `C * width`. The
    /// `width`-dimension occupies the low variables and the `C`-dimension the high variables,
    /// so `Ā(col_point ++ row_point)` evaluates `A` folded by `col_point` over `width` and by
    /// `row_point` over the `C` rows.
    pub fn a_mle_flat(&self) -> Vec<F> {
        let mut flat = Vec::with_capacity(C * self.width);
        for r in 0..C {
            flat.extend_from_slice(&self.a[r]);
        }
        flat
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_field::PrimeCharacteristicRing;
    use p3_goldilocks::Goldilocks;
    use rand::RngExt;

    #[test]
    fn hash_column_matches_naive_matrix_vector_product() {
        let mut rng = rand::rng();
        let width = 128;
        let h = SubsetSumHash::<Goldilocks>::new(width, &mut rng);
        let x: Vec<bool> = (0..width).map(|_| rng.random::<bool>()).collect();

        let got = h.hash_column(&x);
        // Naive A·x.
        for r in 0..C {
            let mut acc = Goldilocks::ZERO;
            for i in 0..width {
                if x[i] {
                    acc += h.a[r][i];
                }
            }
            assert_eq!(got[r], acc);
        }
    }

    #[test]
    fn hash_columns_is_columnwise_hash() {
        let mut rng = rand::rng();
        let width = 64;
        let h = SubsetSumHash::<Goldilocks>::new(width, &mut rng);
        let cols: Vec<Vec<bool>> = (0..10)
            .map(|_| (0..width).map(|_| rng.random::<bool>()).collect())
            .collect();
        let hm = h.hash_columns(&cols);
        for (j, col) in cols.iter().enumerate() {
            let d = h.hash_column(col);
            for r in 0..C {
                assert_eq!(hm[r][j], d[r]);
            }
        }
    }
}
