//! Shared field type for the PIOP. (The original transparent shipped-poly protocol lived here; it
//! has been removed in favor of the faithful commit→reduce→batch-open PIOP in [`crate::faithful`].)

use p3_goldilocks::Goldilocks;

/// Challenges and sumcheck arithmetic run in `GoldilocksExt2`.
pub type EF = p3_field::extension::BinomialExtensionField<Goldilocks, 2>;
