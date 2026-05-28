use core::marker::PhantomData;

use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::{ExtensionField, Field, TwoAdicField};
use utils::{
    merkle::{MerkleTreeProver, Serialize},
    poly::MlPoly,
};

#[derive(Debug, Clone, Default)]
pub struct Deepfold<BF, EF>
where
    BF: Field,
    EF: ExtensionField<BF>,
{
    _marker: PhantomData<(BF, EF)>,
}

pub struct DeepfoldProverState<F: Field> {
    codes: Vec<Vec<F>>,
    mt_prover: MerkleTreeProver,
}

pub struct DeepfoldCommit([u8; 32]);

impl<BF, EF> Deepfold<BF, EF>
where
    BF: TwoAdicField,
    EF: ExtensionField<BF>,
{
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }

    fn to_commit(codes: Vec<Vec<EF>>) -> (DeepfoldProverState<EF>, DeepfoldCommit) {
        let code_length = codes[0].len();
        let codes = (0..code_length)
            .map(|i| (0..16).map(|j| codes[j][i]).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let mt_prover = MerkleTreeProver::new(
            &codes
                .iter()
                .map(|v| Serialize::serialize_fields(v))
                .collect(),
        );
        let commit = mt_prover.commit();
        (
            DeepfoldProverState { codes, mt_prover },
            DeepfoldCommit(commit),
        )
    }

    pub fn commit_base(
        poly: MlPoly<BF>,
        code_rate: usize,
    ) -> (DeepfoldProverState<EF>, DeepfoldCommit) {
        let polies = poly.split(16);
        let code_length = polies[0].0.len() << code_rate;
        let dft = Radix2Dit::<BF>::default();
        let codes = polies
            .iter()
            .map(|poly| {
                let mut coeffs = poly.0.clone();
                coeffs.resize(code_length, BF::ZERO);
                dft.dft(coeffs)
                    .into_iter()
                    .map(|x| EF::from(x))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        Self::to_commit(codes)
    }

    pub fn commit_ext(
        poly: MlPoly<EF>,
        code_rate: usize,
    ) -> (DeepfoldProverState<EF>, DeepfoldCommit) {
        let polies = poly.split(16);
        let code_length = polies[0].0.len() << code_rate;
        let dft = Radix2Dit::<BF>::default();
        let codes = polies
            .iter()
            .map(|poly| {
                let mut coeffs = poly.0.clone();
                coeffs.resize(code_length, EF::ZERO);
                dft.dft_algebra(coeffs)
            })
            .collect::<Vec<_>>();

        Self::to_commit(codes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use p3_field::extension::BinomialExtensionField;
    use p3_goldilocks::Goldilocks;

    type GoldilocksExt2 = BinomialExtensionField<Goldilocks, 2>;

    #[test]
    fn prover_can_be_parameterized_by_base_and_extension_fields() {
        let _prover = Deepfold::<Goldilocks, GoldilocksExt2>::new();
    }
}
