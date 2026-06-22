//! GKR-LogUp lookup argument for `int_gpt` inference.
//!
//! Replaces the Celer (products-of-powers) lookup. The membership of a length-`m` query
//! multiset `a` in a length-`n` table `t`, with multiplicities `e`, is the logarithmic-
//! derivative identity
//!
//! ```text
//!   Σ_{i<m} 1/(α − a_i)  =  Σ_{j<n} e_j/(α − t_j)
//! ```
//!
//! at a random `α` (Schwartz–Zippel ⇒ multiset equality ⇒ `a ⊆ t`). Each side is a sum of
//! fractions, evaluated by a [`frac_sum`] GKR circuit that folds pairs of fractions
//! `(p₀,q₀)+(p₁,q₁) = (p₀q₁+p₁q₀, q₀q₁)` down to a single root fraction. Proving the two
//! root fractions equal (cross-multiplied) certifies the lookup.
//!
//! Versus Celer this needs **no** multiplicity bit-decomposition, no `B`-fold of subset
//! products, no Hadamard sumchecks, and no bit-check: the multiplicities `e` are committed
//! directly as field values (size `n`), and the whole argument is two GKR passes plus one
//! scalar check. See [`lookup`].

pub mod frac_sum;
pub mod lookup;
