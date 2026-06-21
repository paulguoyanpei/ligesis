use logup::lookup::{self, LookupProof, Poly};
use p3_field::{extension::BinomialExtensionField, Field, PrimeCharacteristicRing};
use p3_goldilocks::Goldilocks;
use utils::{
    oracle::RandomOracle,
    poly::MlPoly,
    sumcheck::{self, SumcheckProof},
};

use crate::{
    model::ModelWeights,
    witness::{encode_i64, Config, Witness},
};

pub type EF = BinomialExtensionField<Goldilocks, 2>;

const TYPE_EXP: u64 = 1;
const TYPE_GELU: u64 = 2;
const TYPE_LIMB16: u64 = 3;
const TYPE_RANGE12: u64 = 4;
const TYPE_RANGE15: u64 = 5;
const TYPE_RANGE13: u64 = 6;
const TYPE_RANGE20: u64 = 7;
const TYPE_DUMMY: u64 = 99;
const LIMB_BITS: usize = 16;
const LIMB_SIZE: usize = 1 << LIMB_BITS;
const RANGE20_SIZE: usize = 1 << 20;
const QUOTIENT_LIMB_OFFSET: u64 = 1 << 24;

pub struct ProtocolProof {
    pub initial_witness: InitialWitnessOracles,
    pub lookup: UnifiedLookupProof,
    pub softmax_sum: LinearEvalProof,
    pub softmax_max: LinearEvalProof,
    pub wiring: LinearEvalProof,
    pub fs_field_challenges: usize,
    pub fs_int_challenges: usize,
}

/// The only online witness polynomials sent by the transparent PIOP. Every other
/// witness value must be a virtual oracle derived from these MLEs and public data.
pub struct InitialWitnessOracles {
    pub q_ln: Vec<EF>,
    pub a_ln: Vec<EF>,
    pub std: Vec<EF>,
    pub q_qkv: Vec<EF>,
    pub q_sc: Vec<EF>,
    pub x_max: Vec<EF>,
    pub exp: Vec<EF>,
    pub sum_exp: Vec<EF>,
    pub q_prob: Vec<EF>,
    pub q_ao: Vec<EF>,
    pub q_apr: Vec<EF>,
    pub q_fc: Vec<EF>,
    pub act: Vec<EF>,
    pub q_fpr: Vec<EF>,
    pub x_out: Vec<EF>,
    pub q_log: Vec<EF>,
}

impl InitialWitnessOracles {
    fn size_bytes(&self) -> usize {
        [
            &self.q_ln, &self.a_ln, &self.std, &self.q_qkv, &self.q_sc, &self.x_max,
            &self.exp, &self.sum_exp, &self.q_prob, &self.q_ao, &self.q_apr, &self.q_fc,
            &self.act, &self.q_fpr, &self.x_out, &self.q_log,
        ]
        .iter()
        .map(|poly| poly.len() * core::mem::size_of::<EF>())
        .sum()
    }
}

pub struct ProductEvalProof {
    pub point: Vec<EF>,
    pub claimed_eval: EF,
    pub fixed_point_prefix_len: usize,
    pub left_poly: Vec<EF>,
    pub right_poly: Vec<EF>,
    pub proof: SumcheckProof<EF>,
}

pub struct LinearEvalProof {
    pub point: Vec<EF>,
    /// Actual operand oracles. The residual used by sumcheck is derived locally from these
    /// oracles and is deliberately not part of the proof.
    pub source_polys: Vec<Vec<EF>>,
    pub proof: SumcheckProof<EF>,
}

pub struct MatmulEvalProof {
    pub row_point: Vec<EF>,
    pub col_point: Vec<EF>,
    pub claimed_eval: EF,
    pub left_poly: Vec<EF>,
    pub right_poly: Vec<EF>,
    pub proof: SumcheckProof<EF>,
}

impl MatmulEvalProof {
    pub fn size_bytes(&self) -> usize {
        (1 + self.row_point.len() + self.col_point.len()) * core::mem::size_of::<EF>()
            + (self.left_poly.len() + self.right_poly.len()) * core::mem::size_of::<EF>()
            + self.proof.size_bytes()
    }
}

impl LinearEvalProof {
    pub fn size_bytes(&self) -> usize {
        (self.point.len()
            + self.source_polys.iter().map(Vec::len).sum::<usize>())
            * core::mem::size_of::<EF>()
            + self.proof.size_bytes()
    }
}

impl ProtocolProof {
    pub fn size_bytes(&self) -> usize {
        self.initial_witness.size_bytes()
            + self.lookup.size_bytes()
            + self.softmax_sum.size_bytes()
            + self.softmax_max.size_bytes()
            + self.wiring.size_bytes()
    }
}

impl ProductEvalProof {
    pub fn size_bytes(&self) -> usize {
        (1 + self.point.len() + self.left_poly.len() + self.right_poly.len())
            * core::mem::size_of::<EF>()
            + core::mem::size_of::<usize>()
            + self.proof.size_bytes()
    }
}

pub struct UnifiedLookupProof {
    pub table_poly: Vec<EF>,
    pub multiplicity_poly: Vec<EF>,
    pub proof: LookupProof<EF>,
    pub query: VirtualQueryProof,
    pub num_real_queries: usize,
    pub num_padded_queries: usize,
    pub num_table_rows: usize,
}

impl UnifiedLookupProof {
    pub fn size_bytes(&self) -> usize {
        self.proof.size_bytes()
            + self.query.size_bytes()
            + (self.table_poly.len() + self.multiplicity_poly.len()) * core::mem::size_of::<EF>()
    }
}

pub struct VirtualQueryProof {
    pub segments: Vec<QuerySegmentProof>,
    pub range_checks: Vec<RangeValueCheckProof>,
    pub padded_len: usize,
}

impl VirtualQueryProof {
    pub fn size_bytes(&self) -> usize {
        self.segments
            .iter()
            .map(QuerySegmentProof::size_bytes)
            .sum::<usize>()
            + self
                .range_checks
                .iter()
                .map(RangeValueCheckProof::size_bytes)
                .sum::<usize>()
            + core::mem::size_of::<usize>()
    }
}

pub enum QuerySegmentProof {
    Direct {
        start: usize,
        input_poly: Vec<EF>,
        output_poly: Vec<EF>,
        ty: u64,
    },
    TypeA {
        start: usize,
        q_poly: Vec<EF>,
        divisor: u64,
        ty: u64,
        matmul: MatmulEvalProof,
    },
    Padding {
        start: usize,
        len: usize,
        value: EF,
    },
}

impl QuerySegmentProof {
    fn start(&self) -> usize {
        match self {
            QuerySegmentProof::Direct { start, .. }
            | QuerySegmentProof::TypeA { start, .. }
            | QuerySegmentProof::Padding { start, .. } => *start,
        }
    }

    fn len(&self) -> usize {
        match self {
            QuerySegmentProof::Direct { input_poly, .. } => input_poly.len(),
            QuerySegmentProof::TypeA { q_poly, .. } => q_poly.len(),
            QuerySegmentProof::Padding { len, .. } => *len,
        }
    }

    fn size_bytes(&self) -> usize {
        match self {
            QuerySegmentProof::Direct {
                input_poly,
                output_poly,
                ..
            } => {
                (input_poly.len() + output_poly.len()) * core::mem::size_of::<EF>()
                    + 2 * core::mem::size_of::<usize>()
                    + core::mem::size_of::<u64>()
            }
            QuerySegmentProof::TypeA { q_poly, matmul, .. } => {
                q_poly.len() * core::mem::size_of::<EF>()
                    + matmul.size_bytes()
                    + 2 * core::mem::size_of::<usize>()
                    + 2 * core::mem::size_of::<u64>()
            }
            QuerySegmentProof::Padding { .. } => {
                2 * core::mem::size_of::<usize>() + core::mem::size_of::<EF>()
            }
        }
    }
}

pub struct RangeValueCheckProof {
    pub start: usize,
    pub len: usize,
    pub point: Vec<EF>,
    pub kind: RangeValueCheckKind,
    pub source: RangeValueProof,
}

#[derive(Clone, Copy)]
pub enum RangeValueCheckKind {
    Direct,
    Limbs { num_limbs: usize },
}

impl RangeValueCheckProof {
    fn size_bytes(&self) -> usize {
        self.point.len() * core::mem::size_of::<EF>()
            + self.source.poly_len() * core::mem::size_of::<EF>()
            + 2 * core::mem::size_of::<usize>()
            + match self.kind {
                RangeValueCheckKind::Direct => core::mem::size_of::<u8>(),
                RangeValueCheckKind::Limbs { .. } => core::mem::size_of::<usize>(),
            }
            + self.source.proof_size_bytes()
    }
}

pub enum RangeValueProof {
    QuotientBound {
        quotient_poly: Vec<EF>,
        offset: u64,
    },
    Sqrt {
        input_poly: Vec<EF>,
        feat_vars: usize,
        d_model: u64,
        std_poly: Vec<EF>,
        upper: bool,
        std_square: ProductEvalProof,
        input_square: ProductEvalProof,
        sum_x_square: VirtualSquareProof,
    },
    TypeB {
        dividend_poly: Vec<EF>,
        quotient_poly: Vec<EF>,
        divisor_poly: Vec<EF>,
        upper: bool,
        product: ProductEvalProof,
    },
}

impl RangeValueProof {
    fn poly_len(&self) -> usize {
        match self {
            RangeValueProof::QuotientBound { quotient_poly, .. } => quotient_poly.len(),
            RangeValueProof::Sqrt {
                input_poly,
                std_poly,
                ..
            } => input_poly.len() + std_poly.len(),
            RangeValueProof::TypeB {
                dividend_poly,
                quotient_poly,
                divisor_poly,
                ..
            } => dividend_poly.len() + quotient_poly.len() + divisor_poly.len(),
        }
    }

    fn proof_size_bytes(&self) -> usize {
        match self {
            RangeValueProof::QuotientBound { .. } => 0,
            RangeValueProof::Sqrt {
                input_square,
                std_square,
                sum_x_square,
                ..
            } => input_square.size_bytes() + std_square.size_bytes() + sum_x_square.size_bytes(),
            RangeValueProof::TypeB { product, .. } => product.size_bytes(),
        }
    }
}

pub struct VirtualSquareProof {
    pub claimed_eval: EF,
    pub proof: SumcheckProof<EF>,
}

impl VirtualSquareProof {
    fn size_bytes(&self) -> usize {
        core::mem::size_of::<EF>() + self.proof.size_bytes()
    }
}

pub fn prove_protocol(
    config: &Config,
    weights: &ModelWeights,
    witness: &Witness,
    oracle: &mut RandomOracle<EF>,
) -> ProtocolProof {
    let initial_witness = initial_witness_oracles(witness);
    let lookup = prove_unified_lut_lookup(config, weights, witness, oracle);
    let softmax_sum = prove_softmax_sum_check(config, witness, oracle);
    let softmax_max = prove_softmax_max_check(config, witness, oracle);
    let wiring = prove_wiring_check(config, weights, witness, oracle);
    let (fs_field_challenges, fs_int_challenges) = oracle.drawn();
    ProtocolProof {
        initial_witness,
        lookup,
        softmax_sum,
        softmax_max,
        wiring,
        fs_field_challenges,
        fs_int_challenges,
    }
}

pub fn verify_protocol(
    config: &Config,
    proof: &ProtocolProof,
    oracle: &mut RandomOracle<EF>,
) -> bool {
    let ok = verify_unified_lut_lookup(&proof.lookup, oracle)
        && verify_softmax_sum_check(&proof.softmax_sum, oracle)
        && verify_softmax_max_check(config, &proof.softmax_max, oracle)
        && verify_wiring_check(&proof.wiring, oracle);
    let drawn = oracle.drawn();
    ok && drawn == (proof.fs_field_challenges, proof.fs_int_challenges)
}

fn initial_witness_oracles(witness: &Witness) -> InitialWitnessOracles {
    let mut q_ln = Vec::new();
    let mut a_ln = Vec::new();
    let mut std = Vec::new();
    let mut q_qkv = Vec::new();
    let mut q_sc = Vec::new();
    let mut x_max = Vec::new();
    let mut exp = Vec::new();
    let mut sum_exp = Vec::new();
    let mut q_prob = Vec::new();
    let mut q_ao = Vec::new();
    let mut q_apr = Vec::new();
    let mut q_fc = Vec::new();
    let mut act = Vec::new();
    let mut q_fpr = Vec::new();
    let mut x_out = Vec::new();

    for block in &witness.blocks {
        for ln in [&block.ln1, &block.ln2] {
            q_ln.extend(ln.quotient.data().iter().copied().map(ef_from_i64));
            a_ln.extend(ln.dividend.data().iter().copied().map(ef_from_i64));
            std.extend(ln.std.iter().copied().map(ef_from_i64));
        }
        q_qkv.extend(block.attention.q_qkv.data().iter().copied().map(ef_from_i64));
        for head in 0..block.attention.scores.len() {
            q_sc.extend(block.attention.scores[head].data().iter().copied().map(ef_from_i64));
            x_max.extend(block.attention.x_max[head].data().iter().copied().map(ef_from_i64));
            exp.extend(block.attention.exp[head].data().iter().copied().map(ef_from_i64));
            sum_exp.extend(block.attention.sum_exp[head].data().iter().copied().map(ef_from_i64));
            q_prob.extend(block.attention.q_prob[head].data().iter().copied().map(ef_from_i64));
        }
        q_ao.extend(block.attention.q_ao.data().iter().copied().map(ef_from_i64));
        q_apr.extend(block.attention.q_apr.data().iter().copied().map(ef_from_i64));
        q_fc.extend(block.mlp.q_fc.data().iter().copied().map(ef_from_i64));
        act.extend(block.mlp.act.data().iter().copied().map(ef_from_i64));
        q_fpr.extend(block.mlp.q_fpr.data().iter().copied().map(ef_from_i64));
        x_out.extend(block.attention.x_out.data().iter().copied().map(ef_from_i64));
        x_out.extend(block.mlp.x_out.data().iter().copied().map(ef_from_i64));
    }
    q_ln.extend(witness.lnf.quotient.data().iter().copied().map(ef_from_i64));
    a_ln.extend(witness.lnf.dividend.data().iter().copied().map(ef_from_i64));
    std.extend(witness.lnf.std.iter().copied().map(ef_from_i64));

    InitialWitnessOracles {
        q_ln: pad_ef_poly(q_ln),
        a_ln: pad_ef_poly(a_ln),
        std: pad_ef_poly(std),
        q_qkv: pad_ef_poly(q_qkv),
        q_sc: pad_ef_poly(q_sc),
        x_max: pad_ef_poly(x_max),
        exp: pad_ef_poly(exp),
        sum_exp: pad_ef_poly(sum_exp),
        q_prob: pad_ef_poly(q_prob),
        q_ao: pad_ef_poly(q_ao),
        q_apr: pad_ef_poly(q_apr),
        q_fc: pad_ef_poly(q_fc),
        act: pad_ef_poly(act),
        q_fpr: pad_ef_poly(q_fpr),
        x_out: pad_ef_poly(x_out),
        q_log: pad_ef_poly(witness.q_log.data().iter().copied().map(ef_from_i64).collect()),
    }
}

fn pad_ef_poly(mut values: Vec<EF>) -> Vec<EF> {
    values.resize(values.len().max(1).next_power_of_two(), EF::ZERO);
    values
}

fn prove_product_eval_at(
    left: Vec<EF>,
    right: Vec<EF>,
    eval_point: Vec<EF>,
    fixed_point_prefix_len: usize,
    oracle: &mut RandomOracle<EF>,
) -> ProductEvalProof {
    assert_eq!(left.len(), right.len());
    assert!(left.len().is_power_of_two());
    assert_eq!(left.len(), 1 << eval_point.len());
    assert!(fixed_point_prefix_len <= eval_point.len());
    let product_poly = pointwise_product(&left, &right);
    let claimed_eval = MlPoly(product_poly).eval(&eval_point);
    let eq_poly = MlPoly::new_eq(&eval_point).0;
    let (proof, _) = sumcheck::prove(vec![eq_poly, left.clone(), right.clone()], oracle);
    ProductEvalProof {
        point: eval_point,
        claimed_eval,
        fixed_point_prefix_len,
        left_poly: left,
        right_poly: right,
        proof,
    }
}

fn verify_product_eval_at(
    expected_point: Vec<EF>,
    proof: &ProductEvalProof,
    oracle: &mut RandomOracle<EF>,
) -> bool {
    if proof.left_poly.len() != proof.right_poly.len()
        || !proof.left_poly.len().is_power_of_two()
        || proof.left_poly.len() != 1 << expected_point.len()
        || proof.point != expected_point
    {
        return false;
    }
    let sc_point = match sumcheck::verify(proof.claimed_eval, &proof.proof, oracle) {
        Some(point) => point,
        None => return false,
    };
    if proof.proof.final_evals.len() != 3 {
        return false;
    }
    let eq_poly = MlPoly::new_eq(&expected_point).0;
    proof.proof.final_evals[0] == eval_sumcheck_order(eq_poly, &sc_point)
        && proof.proof.final_evals[1] == eval_sumcheck_order(proof.left_poly.clone(), &sc_point)
        && proof.proof.final_evals[2] == eval_sumcheck_order(proof.right_poly.clone(), &sc_point)
}

fn prove_softmax_sum_check(
    config: &Config,
    witness: &Witness,
    oracle: &mut RandomOracle<EF>,
) -> LinearEvalProof {
    let (sum_poly, exp_poly) = encoded_softmax_sum_sources(config, witness);
    prove_linear_relation(
        vec![sum_poly, exp_poly],
        |sources| softmax_sum_residual(&sources[0], &sources[1]),
        oracle,
    )
}

fn prove_softmax_max_check(
    config: &Config,
    witness: &Witness,
    oracle: &mut RandomOracle<EF>,
) -> LinearEvalProof {
    let (x_max_poly, scores_poly) = encoded_softmax_max_sources(config, witness);
    let point = oracle.next_n_fields(x_max_poly.len().trailing_zeros() as usize);
    let mut factors = Vec::with_capacity(config.n_seq + 2);
    factors.push(MlPoly::new_eq(&point).0);
    factors.push(softmax_max_valid_rows(config, x_max_poly.len()));
    for j in 0..config.n_seq {
        factors.push(softmax_max_factor(config, &x_max_poly, &scores_poly, j));
    }
    let (proof, _) = sumcheck::prove(factors, oracle);
    LinearEvalProof {
        point,
        source_polys: vec![x_max_poly, scores_poly],
        proof,
    }
}

fn prove_wiring_check(
    config: &Config,
    weights: &ModelWeights,
    witness: &Witness,
    oracle: &mut RandomOracle<EF>,
) -> LinearEvalProof {
    let (lhs, rhs) = encoded_wiring_sources(config, weights, witness);
    prove_linear_relation(vec![lhs, rhs], |sources| {
        sources[0]
            .iter()
            .zip(&sources[1])
            .map(|(&left, &right)| left - right)
            .collect()
    }, oracle)
}

fn prove_linear_relation(
    source_polys: Vec<Vec<EF>>,
    residual: impl FnOnce(&[Vec<EF>]) -> Vec<EF>,
    oracle: &mut RandomOracle<EF>,
) -> LinearEvalProof {
    assert!(!source_polys.is_empty());
    let residual = residual(&source_polys);
    let len = residual.len();
    assert!(len.is_power_of_two());
    let point = oracle.next_n_fields(len.trailing_zeros() as usize);
    assert_eq!(residual.len(), len);
    let eq = MlPoly::new_eq(&point).0;
    let (proof, _) = sumcheck::prove(vec![eq, residual], oracle);
    LinearEvalProof {
        point,
        source_polys,
        proof,
    }
}

fn verify_linear_relation(
    proof: &LinearEvalProof,
    residual_at: impl FnOnce(&[Vec<EF>], &[EF]) -> Option<EF>,
    oracle: &mut RandomOracle<EF>,
) -> bool {
    if proof.source_polys.is_empty() {
        return false;
    }
    let residual_len = 1usize << proof.point.len();
    if residual_len == 0 {
        return false;
    }
    let expected = oracle.next_n_fields(proof.point.len());
    if proof.point != expected {
        return false;
    }
    let sc_point = match sumcheck::verify(EF::ZERO, &proof.proof, oracle) {
        Some(point) => point,
        None => return false,
    };
    if proof.proof.final_evals.len() != 2 {
        return false;
    }
    proof.proof.final_evals[0] == eval_sumcheck_order(MlPoly::new_eq(&proof.point).0, &sc_point)
        && residual_at(&proof.source_polys, &sc_point) == Some(proof.proof.final_evals[1])
}

fn verify_softmax_sum_check(proof: &LinearEvalProof, oracle: &mut RandomOracle<EF>) -> bool {
    verify_linear_relation(proof, |sources, point| {
        let [sum_poly, exp_poly] = sources else {
            return None;
        };
        softmax_sum_residual_at(sum_poly, exp_poly, point)
    }, oracle)
}

fn verify_softmax_max_check(
    config: &Config,
    proof: &LinearEvalProof,
    oracle: &mut RandomOracle<EF>,
) -> bool {
    let [x_max_poly, scores_poly] = proof.source_polys.as_slice() else {
        return false;
    };
    if !x_max_poly.len().is_power_of_two()
        || scores_poly.len() % x_max_poly.len() != 0
        || proof.point.len() != x_max_poly.len().trailing_zeros() as usize
    {
        return false;
    }
    let expected = oracle.next_n_fields(proof.point.len());
    if proof.point != expected {
        return false;
    }
    let sc_point = match sumcheck::verify(EF::ZERO, &proof.proof, oracle) {
        Some(point) => point,
        None => return false,
    };
    if proof.proof.final_evals.len() != config.n_seq + 2
        || proof.proof.final_evals[0]
            != eval_sumcheck_order(MlPoly::new_eq(&proof.point).0, &sc_point)
        || proof.proof.final_evals[1]
            != eval_sumcheck_order(softmax_max_valid_rows(config, x_max_poly.len()), &sc_point)
    {
        return false;
    }
    (0..config.n_seq).all(|j| {
        proof.proof.final_evals[j + 2]
            == softmax_max_factor_at(config, x_max_poly, scores_poly, j, &sc_point)
    })
}

fn verify_wiring_check(proof: &LinearEvalProof, oracle: &mut RandomOracle<EF>) -> bool {
    verify_linear_relation(proof, |sources, point| {
        let [lhs, rhs] = sources else {
            return None;
        };
        Some(eval_sumcheck_order(lhs.clone(), point) - eval_sumcheck_order(rhs.clone(), point))
    }, oracle)
}

#[cfg(any())]
fn verify_linear_pcs(proof: &ProtocolProof, oracle: &mut RandomOracle<EF>) -> bool {
    let sum_commit = BasefoldCommit(proof.linear_pcs.softmax_sum_commitment);
    let max_commit = BasefoldCommit(proof.linear_pcs.softmax_max_commitment);
    let wiring_commit = BasefoldCommit(proof.linear_pcs.wiring_commitment);
    Basefold::<Goldilocks, EF>::batch_verify(
        &[],
        &[
            (
                &sum_commit,
                proof.softmax_sum.point.clone(),
                proof.softmax_sum.claimed_eval,
                proof.linear_pcs.softmax_log_interleave,
            ),
            (
                &max_commit,
                proof.softmax_max.point.clone(),
                proof.softmax_max.claimed_eval,
                proof.linear_pcs.softmax_log_interleave,
            ),
        ],
        &proof.linear_pcs.softmax_opening,
        oracle,
    ) && Basefold::<Goldilocks, EF>::batch_verify(
        &[],
        &[(
            &wiring_commit,
            proof.wiring.point.clone(),
            proof.wiring.claimed_eval,
            proof.linear_pcs.wiring_log_interleave,
        )],
        &proof.linear_pcs.wiring_opening,
        oracle,
    )
}

fn transposed_matmul_contraction_vectors(
    a: &crate::tensor::Matrix,
    b_t_source: &crate::tensor::Matrix,
    row_point: &[EF],
    col_point: &[EF],
) -> (Vec<EF>, Vec<EF>) {
    assert_eq!(a.cols(), b_t_source.cols());
    let k_pow2 = a.cols().next_power_of_two();
    let mut avec = vec![EF::ZERO; k_pow2];
    let mut bvec = vec![EF::ZERO; k_pow2];
    let row_eq = eq_weights(row_point, a.rows());
    let col_eq = eq_weights(col_point, b_t_source.rows());
    for k in 0..a.cols() {
        let mut av = EF::ZERO;
        for (r, &eq) in row_eq.iter().enumerate() {
            av += eq * EF::from(encode_i64(a.get(r, k)));
        }
        let mut bv = EF::ZERO;
        for (c, &eq) in col_eq.iter().enumerate() {
            bv += eq * EF::from(encode_i64(b_t_source.get(c, k)));
        }
        avec[k] = av;
        bvec[k] = bv;
    }
    (avec, bvec)
}

fn regular_matmul_contraction_vectors(
    a: &crate::tensor::Matrix,
    b: &crate::tensor::Matrix,
    row_point: &[EF],
    col_point: &[EF],
) -> (Vec<EF>, Vec<EF>) {
    assert_eq!(a.cols(), b.rows());
    let k_pow2 = a.cols().next_power_of_two();
    let mut avec = vec![EF::ZERO; k_pow2];
    let mut bvec = vec![EF::ZERO; k_pow2];
    let row_eq = eq_weights(row_point, a.rows());
    let col_eq = eq_weights(col_point, b.cols());
    for k in 0..a.cols() {
        let mut av = EF::ZERO;
        for (r, &eq) in row_eq.iter().enumerate() {
            av += eq * EF::from(encode_i64(a.get(r, k)));
        }
        let mut bv = EF::ZERO;
        for (c, &eq) in col_eq.iter().enumerate() {
            bv += eq * EF::from(encode_i64(b.get(k, c)));
        }
        avec[k] = av;
        bvec[k] = bv;
    }
    (avec, bvec)
}

fn qkv_head_matrix(
    config: &Config,
    block: &crate::witness::BlockWitness,
    head: usize,
    part: usize,
) -> crate::tensor::Matrix {
    let mut data = Vec::with_capacity(config.n_seq * config.d_head);
    let base = part * config.d_model + head * config.d_head;
    for i in 0..config.n_seq {
        for d in 0..config.d_head {
            data.push(block.attention.qkv.get(i, base + d));
        }
    }
    crate::tensor::Matrix::new(config.n_seq, config.d_head, data)
}

fn eq_weights(point: &[EF], len: usize) -> Vec<EF> {
    let pow2 = 1usize << point.len();
    assert!(len <= pow2);
    let mut out = Vec::with_capacity(len);
    for idx in 0..len {
        let mut eq = EF::ONE;
        for (bit, &r) in point.iter().enumerate() {
            eq *= if (idx >> bit) & 1 == 1 {
                r
            } else {
                EF::ONE - r
            };
        }
        out.push(eq);
    }
    out
}

fn dot(a: &[EF], b: &[EF]) -> EF {
    a.iter().zip(b).fold(EF::ZERO, |acc, (&x, &y)| acc + x * y)
}

fn encoded_layernorm_input_poly(config: &Config, witness: &Witness) -> Vec<EF> {
    let feat_pow2 = config.d_model.next_power_of_two();
    let mut out = Vec::new();
    for block in &witness.blocks {
        push_layernorm_input_poly(&mut out, feat_pow2, &block.ln1);
        push_layernorm_input_poly(&mut out, feat_pow2, &block.ln2);
    }
    push_layernorm_input_poly(&mut out, feat_pow2, &witness.lnf);

    let actual_rows = out.len() / feat_pow2;
    let padded_rows = actual_rows.max(1).next_power_of_two();
    out.resize(padded_rows * feat_pow2, EF::ZERO);
    out
}

fn encoded_softmax_sum_sources(config: &Config, witness: &Witness) -> (Vec<EF>, Vec<EF>) {
    let row_len = (config.n_layer * config.n_head * config.n_seq)
        .max(1)
        .next_power_of_two();
    let seq_len = config.n_seq.max(1).next_power_of_two();
    let mut sums = Vec::with_capacity(row_len);
    let mut exp = Vec::with_capacity(row_len * seq_len);
    for block in &witness.blocks {
        for h in 0..block.attention.exp.len() {
            let head_exp = &block.attention.exp[h];
            let head_sums = &block.attention.sum_exp[h];
            for i in 0..config.n_seq {
                for j in 0..config.n_seq {
                    exp.push(ef_from_i64(head_exp.get(0, i, j)));
                }
                exp.resize(exp.len() + seq_len - config.n_seq, EF::ZERO);
                sums.push(ef_from_i64(head_sums.get(i, 0)));
            }
        }
    }
    sums.resize(row_len, EF::ZERO);
    exp.resize(row_len * seq_len, EF::ZERO);
    (sums, exp)
}

fn softmax_sum_residual(sum_poly: &[EF], exp_poly: &[EF]) -> Vec<EF> {
    assert!(sum_poly.len().is_power_of_two());
    assert_eq!(exp_poly.len() % sum_poly.len(), 0);
    exp_poly
        .chunks(exp_poly.len() / sum_poly.len())
        .zip(sum_poly)
        .map(|(row, &sum)| sum - row.iter().copied().fold(EF::ZERO, |acc, x| acc + x))
        .collect()
}

fn softmax_sum_residual_at(sum_poly: &[EF], exp_poly: &[EF], point: &[EF]) -> Option<EF> {
    if sum_poly.len() != 1 << point.len()
        || exp_poly.len() % sum_poly.len() != 0
        || !(exp_poly.len() / sum_poly.len()).is_power_of_two()
    {
        return None;
    }
    let row_sums = exp_poly
        .chunks(exp_poly.len() / sum_poly.len())
        .map(|row| row.iter().copied().fold(EF::ZERO, |acc, x| acc + x))
        .collect::<Vec<_>>();
    Some(eval_sumcheck_order(sum_poly.to_vec(), point) - eval_sumcheck_order(row_sums, point))
}

fn encoded_softmax_max_sources(config: &Config, witness: &Witness) -> (Vec<EF>, Vec<EF>) {
    let row_len = (config.n_layer * config.n_head * config.n_seq)
        .max(1)
        .next_power_of_two();
    let seq_len = config.n_seq.max(1).next_power_of_two();
    let mut x_max = Vec::with_capacity(row_len);
    let mut scores = Vec::with_capacity(row_len * seq_len);
    for block in &witness.blocks {
        for h in 0..block.attention.scores.len() {
            let head_scores = &block.attention.scores[h];
            for i in 0..config.n_seq {
                x_max.push(ef_from_i64(block.attention.x_max[h].get(i, 0)));
                for j in 0..config.n_seq {
                    scores.push(ef_from_i64(head_scores.get(0, i, j)));
                }
                scores.resize(scores.len() + seq_len - config.n_seq, EF::ZERO);
            }
        }
    }
    x_max.resize(row_len, EF::ZERO);
    scores.resize(row_len * seq_len, EF::ZERO);
    (x_max, scores)
}

fn softmax_max_factor(config: &Config, x_max: &[EF], scores: &[EF], j: usize) -> Vec<EF> {
    let seq_len = scores.len() / x_max.len();
    assert!(seq_len.is_power_of_two());
    x_max
        .iter()
        .enumerate()
        .map(|(row, &max)| {
            if row >= config.n_layer * config.n_head * config.n_seq || row % config.n_seq < j {
                return EF::ONE;
            }
            max - scores[row * seq_len + j]
        })
        .collect()
}

fn softmax_max_valid_rows(config: &Config, row_len: usize) -> Vec<EF> {
    let active_rows = config.n_layer * config.n_head * config.n_seq;
    assert!(active_rows <= row_len);
    (0..row_len)
        .map(|row| if row < active_rows { EF::ONE } else { EF::ZERO })
        .collect()
}

fn softmax_max_factor_at(
    config: &Config,
    x_max: &[EF],
    scores: &[EF],
    j: usize,
    point: &[EF],
) -> EF {
    eval_sumcheck_order(softmax_max_factor(config, x_max, scores, j), point)
}

fn encoded_wiring_sources(
    config: &Config,
    weights: &ModelWeights,
    witness: &Witness,
) -> (Vec<EF>, Vec<EF>) {
    let mut lhs = Vec::new();
    let mut rhs = Vec::new();
    let mut residual = &witness.x0;
    for (layer, block) in witness.blocks.iter().enumerate() {
        let bw = &weights.blocks[layer];
        push_layernorm_wiring(&mut lhs, &mut rhs, config, &block.ln1, &bw.ln_1_g, &bw.ln_1_b);
        push_row_bias_wiring(
            &mut lhs, &mut rhs,
            &block.attention.qkv,
            &block.attention.q_qkv,
            &bw.attn_b,
        );
        push_attention_head_merge_wiring(&mut lhs, &mut rhs, config, block);
        push_residual_output_wiring(
            &mut lhs, &mut rhs,
            &block.attention.x_out,
            residual,
            &block.attention.q_apr,
            &bw.attn_proj_b,
        );

        push_layernorm_wiring(&mut lhs, &mut rhs, config, &block.ln2, &bw.ln_2_g, &bw.ln_2_b);
        push_row_bias_wiring(&mut lhs, &mut rhs, &block.mlp.fc, &block.mlp.q_fc, &bw.fc_b);
        push_residual_output_wiring(
            &mut lhs, &mut rhs,
            &block.mlp.x_out,
            &block.attention.x_out,
            &block.mlp.q_fpr,
            &bw.fproj_b,
        );
        residual = &block.mlp.x_out;
    }
    push_layernorm_wiring(
        &mut lhs, &mut rhs,
        config,
        &witness.lnf,
        &weights.ln_f_g,
        &weights.ln_f_b,
    );
    let len = pcs_poly_len(lhs.len());
    lhs.resize(len, EF::ZERO);
    rhs.resize(len, EF::ZERO);
    (lhs, rhs)
}

fn pcs_poly_len(actual_len: usize) -> usize {
    actual_len.max(1).next_power_of_two()
}

#[cfg(any())]
fn linear_num_vars(len: usize) -> usize {
    len.trailing_zeros() as usize
}

#[cfg(any())]
fn softmax_residual_len(config: &Config) -> usize {
    pcs_poly_len(config.n_layer * config.n_head * config.n_seq)
}

#[cfg(any())]
fn wiring_residual_len(config: &Config) -> usize {
    let per_block = config.n_seq * (10 * config.d_model + config.mlp_hidden);
    let final_layer_norm = 2 * config.n_seq * config.d_model;
    pcs_poly_len(config.n_layer * per_block + final_layer_norm)
}

fn push_layernorm_wiring(
    lhs: &mut Vec<EF>,
    rhs: &mut Vec<EF>,
    config: &Config,
    ln: &crate::witness::LayerNormWitness,
    gain: &[i64],
    bias: &[i64],
) {
    for r in 0..ln.input.rows() {
        let sx = ln.sum_x[r] as i128;
        for c in 0..ln.input.cols() {
            let expected_dividend =
                gain[c] as i128 * (config.d_model as i128 * ln.input.get(r, c) as i128 - sx);
            lhs.push(ef_from_i64(ln.dividend.get(r, c)));
            rhs.push(ef_from_i64(expected_dividend as i64));
            lhs.push(ef_from_i64(ln.output.get(r, c)));
            rhs.push(ef_from_i64(ln.quotient.get(r, c) + bias[c]));
        }
    }
}

fn push_row_bias_wiring(
    lhs: &mut Vec<EF>,
    rhs: &mut Vec<EF>,
    with_bias: &crate::tensor::Matrix,
    quotient: &crate::tensor::Matrix,
    bias: &[i64],
) {
    for r in 0..with_bias.rows() {
        for (c, &b) in bias.iter().enumerate().take(with_bias.cols()) {
            lhs.push(ef_from_i64(with_bias.get(r, c)));
            rhs.push(ef_from_i64(quotient.get(r, c) + b));
        }
    }
}

fn push_attention_head_merge_wiring(
    lhs: &mut Vec<EF>,
    rhs: &mut Vec<EF>,
    config: &Config,
    block: &crate::witness::BlockWitness,
) {
    for h in 0..config.n_head {
        for i in 0..config.n_seq {
            for d in 0..config.d_head {
                let merged = block.attention.q_ao.get(i, h * config.d_head + d);
                let head = block.attention.q_ao_heads[h].get(0, i, d);
                lhs.push(ef_from_i64(merged));
                rhs.push(ef_from_i64(head));
            }
        }
    }
}

fn push_residual_output_wiring(
    lhs: &mut Vec<EF>,
    rhs: &mut Vec<EF>,
    output: &crate::tensor::Matrix,
    residual: &crate::tensor::Matrix,
    quotient: &crate::tensor::Matrix,
    bias: &[i64],
) {
    for r in 0..output.rows() {
        for (c, &b) in bias.iter().enumerate().take(output.cols()) {
            lhs.push(ef_from_i64(output.get(r, c)));
            rhs.push(ef_from_i64(residual.get(r, c) + quotient.get(r, c) + b));
        }
    }
}

fn push_layernorm_input_poly(
    out: &mut Vec<EF>,
    feat_pow2: usize,
    ln: &crate::witness::LayerNormWitness,
) {
    for r in 0..ln.input.rows() {
        for c in 0..ln.input.cols() {
            out.push(EF::from(encode_i64(ln.input.get(r, c))));
        }
        out.resize(out.len() + feat_pow2 - ln.input.cols(), EF::ZERO);
    }
}

fn pointwise_product(left: &[EF], right: &[EF]) -> Vec<EF> {
    left.iter().zip(right).map(|(&l, &r)| l * r).collect()
}

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

#[cfg(any())]
fn prove_unified_lut_lookup(
    config: &Config,
    weights: &ModelWeights,
    witness: &Witness,
    offline: &OfflineProverData,
    oracle: &mut RandomOracle<EF>,
) -> UnifiedLookupProof {
    let row_challenge = oracle.next_field();
    let table = build_lut_table(config, weights, row_challenge);
    let query_build = build_virtual_lut_queries(config, weights, witness, row_challenge);

    let mut multiplicities = vec![0u64; table.rows.len()];
    for &idx in &query_build.table_indices {
        multiplicities[idx] += 1;
    }

    let mut table_poly: Vec<EF> = table.rows.iter().map(|r| r.folded).collect();
    let num_table_rows = table_poly.len();
    while !table_poly.len().is_power_of_two() {
        let idx = table_poly.len() as u64;
        table_poly.push(fold_row(row_challenge, idx, 0, TYPE_DUMMY));
        multiplicities.push(0);
    }

    let multiplicity_poly: Vec<EF> = multiplicities.into_iter().map(ef_from_u64).collect();
    let (multiplicity_state, multiplicity_commitment) =
        Basefold::<Goldilocks, EF>::commit_ext(MlPoly(multiplicity_poly.clone()), 1);
    let (lookup_proof, query_point, table_point) = lookup::prove_with_opening_points(
        &query_build.prover_query,
        &table_poly,
        &multiplicity_poly,
        oracle,
    );
    let query = prove_virtual_query_opening(query_build.layout, &query_point, oracle);
    let input_eval = offline.table.input.eval(&table_point);
    let output_eval = offline.table.output.eval(&table_point);
    let ty_eval = offline.table.ty.eval(&table_point);
    let multiplicity_eval = multiplicity_state.eval(&table_point);
    let table_pcs = TablePcsProof {
        input_eval,
        output_eval,
        ty_eval,
        multiplicity_commitment: multiplicity_commitment.0,
        multiplicity_eval,
        proof: Basefold::<Goldilocks, EF>::batch_prove(
            &[
                (&offline.table.input, table_point.clone()),
                (&offline.table.output, table_point.clone()),
                (&offline.table.ty, table_point.clone()),
            ],
            &[(&multiplicity_state, table_point.clone())],
            oracle,
        ),
    };

    UnifiedLookupProof {
        table_pcs,
        proof: lookup_proof,
        query,
        num_real_queries: query_build.num_real_queries,
        num_padded_queries: query_build.prover_query.len(),
        num_table_rows,
    }
}

#[cfg(any())]
fn verify_unified_lut_lookup(
    proof: &UnifiedLookupProof,
    offline: &OfflineVerifierKey,
    oracle: &mut RandomOracle<EF>,
) -> bool {
    let row_challenge = oracle.next_field();
    if !proof.query.padded_len.is_power_of_two()
        || proof.num_real_queries > proof.query.padded_len
        || proof.num_padded_queries != proof.query.padded_len
        || proof.num_table_rows > offline.table.padded_rows
    {
        return false;
    }
    let claims = match lookup::verify(&proof.proof, oracle) {
        Some(claims) => claims,
        None => return false,
    };
    let query_claim = claims
        .iter()
        .find(|c| c.poly == Poly::Query)
        .expect("lookup query claim");
    let mult_claim = claims
        .iter()
        .find(|c| c.poly == Poly::Mult)
        .expect("lookup mult claim");
    let table_claim = claims
        .iter()
        .find(|c| c.poly == Poly::Table)
        .expect("lookup table claim");
    if !verify_virtual_query_opening(
        &proof.query,
        row_challenge,
        &query_claim.point,
        query_claim.value,
        oracle,
    ) || mult_claim.value != proof.table_pcs.multiplicity_eval
        || mult_claim.point != table_claim.point
        || table_claim.point.len() != offline.table.padded_rows.trailing_zeros() as usize
    {
        return false;
    }
    if proof.table_pcs.input_eval
        + row_challenge * proof.table_pcs.output_eval
        + row_challenge.square() * proof.table_pcs.ty_eval
        != table_claim.value
    {
        return false;
    }
    let commits = crate::pcs::table_commitments(offline);
    let refs = [
        (
            &commits[0],
            table_claim.point.clone(),
            proof.table_pcs.input_eval,
            6,
        ),
        (
            &commits[1],
            table_claim.point.clone(),
            proof.table_pcs.output_eval,
            6,
        ),
        (
            &commits[2],
            table_claim.point.clone(),
            proof.table_pcs.ty_eval,
            6,
        ),
    ];
    let mult_commit = pcs::basefold::BasefoldCommit(proof.table_pcs.multiplicity_commitment);
    let mult_refs = [(
        &mult_commit,
        table_claim.point.clone(),
        proof.table_pcs.multiplicity_eval,
        6,
    )];
    Basefold::<Goldilocks, EF>::batch_verify(&refs, &mult_refs, &proof.table_pcs.proof, oracle)
}

fn prove_unified_lut_lookup(
    config: &Config,
    weights: &ModelWeights,
    witness: &Witness,
    oracle: &mut RandomOracle<EF>,
) -> UnifiedLookupProof {
    let row_challenge = oracle.next_field();
    let table = build_lut_table(config, weights, row_challenge);
    let query_build = build_virtual_lut_queries(config, weights, witness, row_challenge);
    let mut multiplicities = vec![0u64; table.rows.len()];
    for &idx in &query_build.table_indices {
        multiplicities[idx] += 1;
    }
    let mut table_poly = table.rows.iter().map(|row| row.folded).collect::<Vec<_>>();
    let num_table_rows = table_poly.len();
    while !table_poly.len().is_power_of_two() {
        let idx = table_poly.len() as u64;
        table_poly.push(fold_row(row_challenge, idx, 0, TYPE_DUMMY));
        multiplicities.push(0);
    }
    let multiplicity_poly = multiplicities
        .into_iter()
        .map(ef_from_u64)
        .collect::<Vec<_>>();
    let (proof, query_point) = lookup::prove_with_query_point(
        &query_build.prover_query,
        &table_poly,
        &multiplicity_poly,
        oracle,
    );
    let query = prove_virtual_query_opening(query_build.layout, &query_point, oracle);
    UnifiedLookupProof {
        table_poly,
        multiplicity_poly,
        proof,
        query,
        num_real_queries: query_build.num_real_queries,
        num_padded_queries: query_build.prover_query.len(),
        num_table_rows,
    }
}

fn verify_unified_lut_lookup(proof: &UnifiedLookupProof, oracle: &mut RandomOracle<EF>) -> bool {
    let row_challenge = oracle.next_field();
    if !proof.table_poly.len().is_power_of_two()
        || proof.table_poly.len() != proof.multiplicity_poly.len()
        || !proof.query.padded_len.is_power_of_two()
        || proof.num_real_queries > proof.query.padded_len
        || proof.num_padded_queries != proof.query.padded_len
        || proof.num_table_rows > proof.table_poly.len()
    {
        return false;
    }
    let claims = match lookup::verify(&proof.proof, oracle) {
        Some(claims) => claims,
        None => return false,
    };
    let query_claim = claims
        .iter()
        .find(|claim| claim.poly == Poly::Query)
        .expect("query claim");
    let mult_claim = claims
        .iter()
        .find(|claim| claim.poly == Poly::Mult)
        .expect("multiplicity claim");
    let table_claim = claims
        .iter()
        .find(|claim| claim.poly == Poly::Table)
        .expect("table claim");
    verify_virtual_query_opening(
        &proof.query,
        row_challenge,
        &query_claim.point,
        query_claim.value,
        oracle,
    ) && MlPoly(proof.multiplicity_poly.clone()).eval(&mult_claim.point) == mult_claim.value
        && MlPoly(proof.table_poly.clone()).eval(&table_claim.point) == table_claim.value
}

struct VirtualQueryBuild {
    prover_query: Vec<EF>,
    table_indices: Vec<usize>,
    layout: VirtualQueryLayout,
    num_real_queries: usize,
}

struct VirtualQueryLayout {
    segments: Vec<QuerySegmentLayout>,
    range_checks: Vec<RangeValueCheckLayout>,
    padded_len: usize,
}

enum QuerySegmentLayout {
    Direct {
        start: usize,
        input_poly: Vec<EF>,
        output_poly: Vec<EF>,
        ty: u64,
    },
    TypeA {
        start: usize,
        q_poly: Vec<EF>,
        divisor: u64,
        ty: u64,
        matmul: PendingMatmul,
    },
    Padding {
        start: usize,
        len: usize,
        value: EF,
    },
}

impl QuerySegmentLayout {
    fn len(&self) -> usize {
        match self {
            QuerySegmentLayout::Direct { input_poly, .. } => input_poly.len(),
            QuerySegmentLayout::TypeA { q_poly, .. } => q_poly.len(),
            QuerySegmentLayout::Padding { len, .. } => *len,
        }
    }

    fn set_start(&mut self, start: usize) {
        match self {
            QuerySegmentLayout::Direct { start: s, .. }
            | QuerySegmentLayout::TypeA { start: s, .. }
            | QuerySegmentLayout::Padding { start: s, .. } => *s = start,
        }
    }
}

enum PendingMatmul {
    Regular {
        a: crate::tensor::Matrix,
        b: crate::tensor::Matrix,
        row_vars: usize,
        col_vars: usize,
    },
    Transposed {
        a: crate::tensor::Matrix,
        b: crate::tensor::Matrix,
        row_vars: usize,
        col_vars: usize,
    },
}

struct RangeValueCheckLayout {
    start: usize,
    len: usize,
    kind: RangeValueCheckKind,
    source: RangeValueLayout,
}

enum RangeValueLayout {
    QuotientBound {
        quotient_poly: Vec<EF>,
        offset: u64,
    },
    Sqrt {
        input_poly: Vec<EF>,
        feat_vars: usize,
        d_model: u64,
        std_poly: Vec<EF>,
        upper: bool,
    },
    TypeB {
        dividend_poly: Vec<EF>,
        quotient_poly: Vec<EF>,
        divisor_poly: Vec<EF>,
        upper: bool,
    },
}

fn prove_virtual_query_opening(
    layout: VirtualQueryLayout,
    query_point: &[EF],
    oracle: &mut RandomOracle<EF>,
) -> VirtualQueryProof {
    let mut segments = Vec::with_capacity(layout.segments.len());
    for segment in layout.segments {
        match segment {
            QuerySegmentLayout::Direct {
                start,
                input_poly,
                output_poly,
                ty,
            } => segments.push(QuerySegmentProof::Direct {
                start,
                input_poly,
                output_poly,
                ty,
            }),
            QuerySegmentLayout::TypeA {
                start,
                q_poly,
                divisor,
                ty,
                matmul,
            } => {
                let local_point = segment_local_point(query_point, start, q_poly.len());
                let matmul = prove_pending_matmul_at(matmul, &local_point, oracle);
                segments.push(QuerySegmentProof::TypeA {
                    start,
                    q_poly,
                    divisor,
                    ty,
                    matmul,
                });
            }
            QuerySegmentLayout::Padding { start, len, value } => {
                segments.push(QuerySegmentProof::Padding { start, len, value });
            }
        }
    }
    let range_checks = layout
        .range_checks
        .into_iter()
        .map(|check| {
            let value_len = range_check_value_len(check.len, &check.kind);
            let value_vars = value_len.trailing_zeros() as usize;
            let point = oracle.next_n_fields(value_vars);
            let source = prove_range_value_source(check.source, point.clone(), oracle);
            RangeValueCheckProof {
                start: check.start,
                len: check.len,
                point,
                kind: check.kind,
                source,
            }
        })
        .collect();
    VirtualQueryProof {
        segments,
        range_checks,
        padded_len: layout.padded_len,
    }
}

fn verify_virtual_query_opening(
    proof: &VirtualQueryProof,
    row_challenge: EF,
    query_point: &[EF],
    claim: EF,
    oracle: &mut RandomOracle<EF>,
) -> bool {
    if proof.padded_len != 1 << query_point.len() {
        return false;
    }
    let mut acc = EF::ZERO;
    for segment in &proof.segments {
        let len = segment.len();
        if !len.is_power_of_two()
            || segment.start() % len != 0
            || segment.start() + len > proof.padded_len
        {
            return false;
        }
        let selector = segment_selector_eval(query_point, segment.start(), len);
        let value = match segment {
            QuerySegmentProof::Direct {
                input_poly,
                output_poly,
                ty,
                ..
            } => {
                if input_poly.len() != len || output_poly.len() != len {
                    return false;
                }
                let local_point = &query_point[..len.trailing_zeros() as usize];
                let input = MlPoly(input_poly.clone()).eval(local_point);
                let output = MlPoly(output_poly.clone()).eval(local_point);
                input + row_challenge * output + row_challenge.square() * ef_from_u64(*ty)
            }
            QuerySegmentProof::TypeA {
                q_poly,
                divisor,
                ty,
                matmul,
                ..
            } => {
                let local_point = &query_point[..len.trailing_zeros() as usize];
                let q = MlPoly(q_poly.clone()).eval(local_point);
                let c = matmul.claimed_eval - ef_from_u64(*divisor) * q;
                if !verify_matmul_eval_at_local(local_point, matmul, oracle) {
                    return false;
                }
                c + row_challenge.square() * ef_from_u64(*ty)
            }
            QuerySegmentProof::Padding { value, .. } => *value,
        };
        acc += selector * value;
    }
    if acc != claim {
        return false;
    }
    for check in &proof.range_checks {
        if !verify_range_value_check(check, &proof.segments, oracle) {
            return false;
        }
    }
    true
}

fn prove_pending_matmul_at(
    pending: PendingMatmul,
    local_point: &[EF],
    oracle: &mut RandomOracle<EF>,
) -> MatmulEvalProof {
    match pending {
        PendingMatmul::Regular {
            a,
            b,
            row_vars,
            col_vars,
        } => {
            assert_eq!(local_point.len(), row_vars + col_vars);
            let col_point = local_point[..col_vars].to_vec();
            let row_point = local_point[col_vars..].to_vec();
            prove_regular_matmul_eval_at(&a, &b, row_point, col_point, oracle)
        }
        PendingMatmul::Transposed {
            a,
            b,
            row_vars,
            col_vars,
        } => {
            assert_eq!(local_point.len(), row_vars + col_vars);
            let col_point = local_point[..col_vars].to_vec();
            let row_point = local_point[col_vars..].to_vec();
            prove_transposed_matmul_eval_at(&a, &b, row_point, col_point, oracle)
        }
    }
}

fn verify_matmul_eval_at_local(
    local_point: &[EF],
    proof: &MatmulEvalProof,
    oracle: &mut RandomOracle<EF>,
) -> bool {
    let col_vars = proof.col_point.len();
    let row_vars = proof.row_point.len();
    if local_point.len() != row_vars + col_vars {
        return false;
    }
    let expected_col = &local_point[..col_vars];
    let expected_row = &local_point[col_vars..];
    verify_matmul_eval_at(expected_row, expected_col, proof, oracle)
}

fn prove_regular_matmul_eval_at(
    a: &crate::tensor::Matrix,
    b: &crate::tensor::Matrix,
    row_point: Vec<EF>,
    col_point: Vec<EF>,
    oracle: &mut RandomOracle<EF>,
) -> MatmulEvalProof {
    let (a_vec, b_vec) = regular_matmul_contraction_vectors(a, b, &row_point, &col_point);
    let claimed_eval = dot(&a_vec, &b_vec);
    let (proof, _) = sumcheck::prove(vec![a_vec.clone(), b_vec.clone()], oracle);
    MatmulEvalProof {
        row_point,
        col_point,
        claimed_eval,
        left_poly: a_vec,
        right_poly: b_vec,
        proof,
    }
}

fn prove_transposed_matmul_eval_at(
    a: &crate::tensor::Matrix,
    b_t_source: &crate::tensor::Matrix,
    row_point: Vec<EF>,
    col_point: Vec<EF>,
    oracle: &mut RandomOracle<EF>,
) -> MatmulEvalProof {
    let (a_vec, b_vec) =
        transposed_matmul_contraction_vectors(a, b_t_source, &row_point, &col_point);
    let claimed_eval = dot(&a_vec, &b_vec);
    let (proof, _) = sumcheck::prove(vec![a_vec.clone(), b_vec.clone()], oracle);
    MatmulEvalProof {
        row_point,
        col_point,
        claimed_eval,
        left_poly: a_vec,
        right_poly: b_vec,
        proof,
    }
}

fn verify_matmul_eval_at(
    expected_row: &[EF],
    expected_col: &[EF],
    proof: &MatmulEvalProof,
    oracle: &mut RandomOracle<EF>,
) -> bool {
    if proof.left_poly.len() != proof.right_poly.len()
        || !proof.left_poly.len().is_power_of_two()
        || proof.row_point != expected_row
        || proof.col_point != expected_col
    {
        return false;
    }
    let sc_point = match sumcheck::verify(proof.claimed_eval, &proof.proof, oracle) {
        Some(point) => point,
        None => return false,
    };
    if proof.proof.final_evals.len() != 2 {
        return false;
    }
    proof.proof.final_evals[0] == eval_sumcheck_order(proof.left_poly.clone(), &sc_point)
        && proof.proof.final_evals[1] == eval_sumcheck_order(proof.right_poly.clone(), &sc_point)
}

#[cfg(any())]
fn prove_source_regular_matmul(
    a_source: &str,
    b_source: &str,
    a: &crate::tensor::Matrix,
    b: &crate::tensor::Matrix,
    row_point: Vec<EF>,
    col_point: Vec<EF>,
    oracle: &mut RandomOracle<EF>,
) -> (SourceMatmulProof, [OracleClaim; 2]) {
    let (a_vec, b_vec) = regular_matmul_contraction_vectors(a, b, &row_point, &col_point);
    let claimed_eval = dot(&a_vec, &b_vec);
    let (proof, sumcheck_point) = sumcheck::prove(vec![a_vec, b_vec], oracle);
    assert_eq!(proof.final_evals.len(), 2);
    let (a_terminal, b_terminal) = (proof.final_evals[0], proof.final_evals[1]);
    let k_point = sumcheck_opening_point(&sumcheck_point);
    let mut a_point = k_point.clone();
    a_point.extend_from_slice(&row_point);
    let mut b_point = col_point.clone();
    b_point.extend_from_slice(&k_point);
    (
        SourceMatmulProof {
            row_point,
            col_point,
            claimed_eval,
            proof,
        },
        [
            OracleClaim {
                oracle: a_source.to_owned(),
                point: a_point,
                value: a_terminal,
            },
            OracleClaim {
                oracle: b_source.to_owned(),
                point: b_point,
                value: b_terminal,
            },
        ],
    )
}

#[cfg(any())]
fn verify_source_regular_matmul(
    a_source: &str,
    b_source: &str,
    expected_row: &[EF],
    expected_col: &[EF],
    proof: &SourceMatmulProof,
    oracle: &mut RandomOracle<EF>,
) -> Option<[OracleClaim; 2]> {
    if proof.row_point != expected_row || proof.col_point != expected_col {
        return None;
    }
    let sumcheck_point = sumcheck::verify(proof.claimed_eval, &proof.proof, oracle)?;
    if proof.proof.final_evals.len() != 2 {
        return None;
    }
    let k_point = sumcheck_opening_point(&sumcheck_point);
    let mut a_point = k_point.clone();
    a_point.extend_from_slice(expected_row);
    let mut b_point = expected_col.to_vec();
    b_point.extend_from_slice(&k_point);
    Some([
        OracleClaim {
            oracle: a_source.to_owned(),
            point: a_point,
            value: proof.proof.final_evals[0],
        },
        OracleClaim {
            oracle: b_source.to_owned(),
            point: b_point,
            value: proof.proof.final_evals[1],
        },
    ])
}

fn segment_local_point(point: &[EF], start: usize, len: usize) -> Vec<EF> {
    assert!(len.is_power_of_two());
    assert_eq!(start % len, 0);
    point[..len.trailing_zeros() as usize].to_vec()
}

fn segment_selector_eval(point: &[EF], start: usize, len: usize) -> EF {
    debug_assert!(len.is_power_of_two());
    debug_assert_eq!(start % len, 0);
    let local_vars = len.trailing_zeros() as usize;
    let fixed = start >> local_vars;
    let mut eq = EF::ONE;
    for (bit, &r) in point[local_vars..].iter().enumerate() {
        eq *= if (fixed >> bit) & 1 == 1 {
            r
        } else {
            EF::ONE - r
        };
    }
    eq
}

fn verify_range_value_check(
    check: &RangeValueCheckProof,
    segments: &[QuerySegmentProof],
    oracle: &mut RandomOracle<EF>,
) -> bool {
    let Some(input_poly) = find_direct_segment_input(segments, check.start, check.len) else {
        return false;
    };
    let value_len = range_check_value_len(check.len, &check.kind);
    if !value_len.is_power_of_two() {
        return false;
    }
    let value_vars = value_len.trailing_zeros() as usize;
    let expected = oracle.next_n_fields(value_vars);
    if check.point != expected {
        return false;
    }
    let source = match verify_range_value_source(&check.source, &check.point, oracle) {
        Some(value) => value,
        None => return false,
    };
    let value = match check.kind {
        RangeValueCheckKind::Direct => MlPoly(input_poly.to_vec()).eval(&check.point),
        RangeValueCheckKind::Limbs { num_limbs } => {
            eval_limb_value(input_poly, num_limbs, &check.point)
        }
    };
    value == source
}

fn find_direct_segment_input(
    segments: &[QuerySegmentProof],
    start: usize,
    len: usize,
) -> Option<&[EF]> {
    segments.iter().find_map(|segment| match segment {
        QuerySegmentProof::Direct {
            start: s,
            input_poly,
            ..
        } if *s == start && input_poly.len() == len => Some(input_poly.as_slice()),
        _ => None,
    })
}

fn range_check_value_len(len: usize, kind: &RangeValueCheckKind) -> usize {
    match kind {
        RangeValueCheckKind::Direct => len,
        RangeValueCheckKind::Limbs { num_limbs } => {
            assert!(num_limbs.is_power_of_two());
            assert_eq!(len % num_limbs, 0);
            len / num_limbs
        }
    }
}

fn eval_limb_value(limb_poly: &[EF], num_limbs: usize, point: &[EF]) -> EF {
    assert!(num_limbs.is_power_of_two());
    assert!(num_limbs <= 4);
    assert_eq!(limb_poly.len(), num_limbs * (1 << point.len()));
    let limb_vars = num_limbs.trailing_zeros() as usize;
    let mut acc = EF::ZERO;
    for limb in 0..num_limbs {
        let mut p = Vec::with_capacity(limb_vars + point.len());
        for bit in 0..limb_vars {
            p.push(if (limb >> bit) & 1 == 1 {
                EF::ONE
            } else {
                EF::ZERO
            });
        }
        p.extend_from_slice(point);
        acc += ef_from_u64(1u64 << (LIMB_BITS * limb)) * MlPoly(limb_poly.to_vec()).eval(&p);
    }
    acc
}

fn prove_range_value_source(
    source: RangeValueLayout,
    point: Vec<EF>,
    oracle: &mut RandomOracle<EF>,
) -> RangeValueProof {
    match source {
        RangeValueLayout::QuotientBound {
            quotient_poly,
            offset,
        } => RangeValueProof::QuotientBound {
            quotient_poly,
            offset,
        },
        RangeValueLayout::Sqrt {
            input_poly,
            feat_vars,
            d_model,
            std_poly,
            upper,
        } => {
            let half = (EF::ONE + EF::ONE).inverse();
            let mut input_point = vec![half; feat_vars];
            input_point.extend_from_slice(&point);
            let input_square = prove_product_eval_at(
                input_poly.clone(),
                input_poly.clone(),
                input_point,
                feat_vars,
                oracle,
            );
            let sum_x_square =
                prove_virtual_sum_x_square(input_poly.clone(), feat_vars, point.clone(), oracle);
            let std_square =
                prove_product_eval_at(std_poly.clone(), std_poly.clone(), point, 0, oracle);
            RangeValueProof::Sqrt {
                input_poly,
                feat_vars,
                d_model,
                std_poly,
                upper,
                std_square,
                input_square,
                sum_x_square,
            }
        }
        RangeValueLayout::TypeB {
            dividend_poly,
            quotient_poly,
            divisor_poly,
            upper,
        } => {
            let product = prove_product_eval_at(
                quotient_poly.clone(),
                divisor_poly.clone(),
                point,
                0,
                oracle,
            );
            RangeValueProof::TypeB {
                dividend_poly,
                quotient_poly,
                divisor_poly,
                upper,
                product,
            }
        }
    }
}

fn verify_range_value_source(
    source: &RangeValueProof,
    point: &[EF],
    oracle: &mut RandomOracle<EF>,
) -> Option<EF> {
    match source {
        RangeValueProof::QuotientBound {
            quotient_poly,
            offset,
        } => {
            if quotient_poly.len() != 1 << point.len() {
                return None;
            }
            Some(MlPoly(quotient_poly.clone()).eval(point) + ef_from_u64(*offset))
        }
        RangeValueProof::Sqrt {
            input_poly,
            feat_vars,
            d_model,
            std_poly,
            upper,
            std_square,
            input_square,
            sum_x_square,
        } => {
            let half = (EF::ONE + EF::ONE).inverse();
            let mut input_point = vec![half; *feat_vars];
            input_point.extend_from_slice(point);
            if !verify_product_eval_at(input_point.clone(), input_square, oracle) {
                return None;
            }
            if !verify_virtual_sum_x_square(point, sum_x_square, input_poly, *feat_vars, oracle) {
                return None;
            }
            if !verify_product_eval_at(point.to_vec(), std_square, oracle) {
                return None;
            }
            let feat_pow2 = ef_from_u64(1u64 << *feat_vars);
            let sum_x2 = feat_pow2 * input_square.claimed_eval;
            let var_sum = ef_from_u64(*d_model) * sum_x2 - sum_x_square.claimed_eval;
            let x = var_sum + EF::ONE;
            let y = MlPoly(std_poly.clone()).eval(point);
            Some(if *upper {
                std_square.claimed_eval + y - x
            } else {
                x - (std_square.claimed_eval - y)
            })
        }
        RangeValueProof::TypeB {
            dividend_poly,
            divisor_poly,
            upper,
            product,
            ..
        } => {
            if !verify_product_eval_at(point.to_vec(), product, oracle) {
                return None;
            }
            let a = MlPoly(dividend_poly.clone()).eval(point);
            let b = MlPoly(divisor_poly.clone()).eval(point);
            let rem = a - product.claimed_eval;
            Some(if *upper { b - EF::ONE - rem } else { rem })
        }
    }
}

fn prove_virtual_sum_x_square(
    input_poly: Vec<EF>,
    feat_vars: usize,
    point: Vec<EF>,
    oracle: &mut RandomOracle<EF>,
) -> VirtualSquareProof {
    let sum_x_poly = derived_sum_x_poly(&input_poly, feat_vars);
    assert_eq!(sum_x_poly.len(), 1 << point.len());
    let product_poly = pointwise_product(&sum_x_poly, &sum_x_poly);
    let claimed_eval = MlPoly(product_poly).eval(&point);
    let eq_poly = MlPoly::new_eq(&point).0;
    let (proof, _) = sumcheck::prove(vec![eq_poly, sum_x_poly.clone(), sum_x_poly], oracle);
    VirtualSquareProof {
        claimed_eval,
        proof,
    }
}

fn verify_virtual_sum_x_square(
    point: &[EF],
    proof: &VirtualSquareProof,
    input_poly: &[EF],
    feat_vars: usize,
    oracle: &mut RandomOracle<EF>,
) -> bool {
    let feat_pow2 = 1usize << feat_vars;
    if input_poly.len() % feat_pow2 != 0 || input_poly.len() / feat_pow2 != 1 << point.len() {
        return false;
    }
    let sc_point = match sumcheck::verify(proof.claimed_eval, &proof.proof, oracle) {
        Some(point) => point,
        None => return false,
    };
    if proof.proof.final_evals.len() != 3 {
        return false;
    }
    let eq_poly = MlPoly::new_eq(&point.to_vec()).0;
    let expected_sum_x = eval_sumcheck_order(derived_sum_x_poly(input_poly, feat_vars), &sc_point);
    proof.proof.final_evals[0] == eval_sumcheck_order(eq_poly, &sc_point)
        && proof.proof.final_evals[1] == expected_sum_x
        && proof.proof.final_evals[2] == expected_sum_x
}

fn derived_sum_x_poly(input_poly: &[EF], feat_vars: usize) -> Vec<EF> {
    let feat_pow2 = 1usize << feat_vars;
    assert_eq!(input_poly.len() % feat_pow2, 0);
    input_poly
        .chunks(feat_pow2)
        .map(|row| row.iter().copied().fold(EF::ZERO, |acc, v| acc + v))
        .collect()
}

struct QuerySegmentBuild {
    layout: QuerySegmentLayout,
    values: Vec<EF>,
    table_indices: Vec<usize>,
    real_len: usize,
    range_check: Option<RangeValueCheckLayout>,
}

fn build_virtual_lut_queries(
    config: &Config,
    weights: &ModelWeights,
    witness: &Witness,
    row_challenge: EF,
) -> VirtualQueryBuild {
    let table = RowOffsets::new(config);
    let mut segments = Vec::new();

    push_function_segments(
        config,
        weights,
        witness,
        row_challenge,
        &table,
        &mut segments,
    );
    push_q_prob_segment(witness, row_challenge, &table, &mut segments);
    let mut range_checks = Vec::new();
    push_limb_segments(config, witness, row_challenge, &table, &mut segments);
    push_type_a_segments(
        config,
        weights,
        witness,
        row_challenge,
        &table,
        &mut segments,
    );

    segments.sort_by_key(|s| core::cmp::Reverse(s.layout.len()));

    let mut prover_query = Vec::new();
    let mut table_indices = Vec::new();
    let mut proof_segments = Vec::new();
    let mut num_real_queries = 0usize;
    let mut start = 0usize;
    for mut segment in segments {
        let len = segment.layout.len();
        assert!(len.is_power_of_two());
        assert_eq!(start % len, 0);
        segment.layout.set_start(start);
        if let Some(mut check) = segment.range_check {
            check.start = start;
            check.len = len;
            range_checks.push(check);
        }
        prover_query.extend(segment.values);
        table_indices.extend(segment.table_indices);
        proof_segments.push(segment.layout);
        num_real_queries += segment.real_len;
        start += len;
    }

    let padded_len = prover_query.len().max(1).next_power_of_two();
    let default = fold_row(row_challenge, 0, weights.exp_lut[0], TYPE_EXP);
    append_query_padding(
        &mut prover_query,
        &mut table_indices,
        &mut proof_segments,
        padded_len,
        default,
    );

    let layout = VirtualQueryLayout {
        segments: proof_segments,
        range_checks,
        padded_len,
    };
    VirtualQueryBuild {
        prover_query,
        table_indices,
        layout,
        num_real_queries,
    }
}

fn append_query_padding(
    prover_query: &mut Vec<EF>,
    table_indices: &mut Vec<usize>,
    segments: &mut Vec<QuerySegmentLayout>,
    padded_len: usize,
    value: EF,
) {
    while prover_query.len() < padded_len {
        let start = prover_query.len();
        let remaining = padded_len - start;
        let align = if start == 0 {
            remaining
        } else {
            start & start.wrapping_neg()
        };
        let len = largest_power_of_two_at_most(remaining.min(align));
        segments.push(QuerySegmentLayout::Padding { start, len, value });
        prover_query.resize(start + len, value);
        table_indices.resize(start + len, 0);
    }
}

fn largest_power_of_two_at_most(x: usize) -> usize {
    1usize << (usize::BITS as usize - 1 - x.leading_zeros() as usize)
}

fn push_direct_segment(
    segments: &mut Vec<QuerySegmentBuild>,
    row_challenge: EF,
    table_offset: usize,
    ty: u64,
    inputs: Vec<u64>,
    outputs: Vec<i64>,
    default_input: u64,
    default_output: i64,
) {
    let real_len = inputs.len();
    segments.push(direct_segment_build(
        row_challenge,
        table_offset,
        ty,
        inputs,
        outputs,
        default_input,
        default_output,
        None,
        real_len,
    ));
}

fn push_function_segments(
    config: &Config,
    weights: &ModelWeights,
    witness: &Witness,
    row_challenge: EF,
    table: &RowOffsets,
    segments: &mut Vec<QuerySegmentBuild>,
) {
    let mut exp_inputs = Vec::new();
    let mut exp_outputs = Vec::new();
    let mut gelu_inputs = Vec::new();
    let mut gelu_outputs = Vec::new();
    for block in &witness.blocks {
        for (h, scores) in block.attention.scores.iter().enumerate() {
            for i in 0..config.n_seq {
                let x_max = block.attention.x_max[h].get(i, 0);
                for j in 0..config.n_seq {
                    let idx = if j <= i {
                        scores.get(0, i, j) - x_max + config.max_v * config.scale
                    } else {
                        0
                    };
                    exp_inputs.push(idx as u64);
                    exp_outputs.push(block.attention.exp[h].get(0, i, j));
                }
            }
        }
        for (flat_idx, &fc) in block.mlp.fc.data().iter().enumerate() {
            gelu_inputs.push((fc + config.max_v * config.scale) as u64);
            gelu_outputs.push(block.mlp.act.data()[flat_idx]);
        }
    }
    push_direct_segment(
        segments,
        row_challenge,
        table.exp,
        TYPE_EXP,
        exp_inputs,
        exp_outputs,
        0,
        weights.exp_lut[0],
    );
    push_direct_segment(
        segments,
        row_challenge,
        table.gelu,
        TYPE_GELU,
        gelu_inputs,
        gelu_outputs,
        0,
        weights.gelu_lut[0],
    );
}

fn push_q_prob_segment(
    witness: &Witness,
    row_challenge: EF,
    table: &RowOffsets,
    segments: &mut Vec<QuerySegmentBuild>,
) {
    let mut values = Vec::new();
    for block in &witness.blocks {
        for probs in &block.attention.q_prob {
            for &q in probs.data() {
                assert!(
                    (0..8192).contains(&(q as i128)),
                    "q_prob out of RANGE13: {q}"
                );
                values.push(q as i128);
            }
        }
    }
    let quotient_poly = encoded_i128_poly_padded(&values);
    let real_len = values.len();
    push_quotient_bound_segment(
        segments,
        row_challenge,
        table.range13,
        TYPE_RANGE13,
        8192,
        values,
        real_len,
        quotient_poly,
        0,
    );
}

fn push_limb_segments(
    config: &Config,
    witness: &Witness,
    row_challenge: EF,
    table: &RowOffsets,
    segments: &mut Vec<QuerySegmentBuild>,
) {
    let (sqrt_input, sqrt_feat_vars, sqrt_d_model, sqrt_std, sqrt_lower, sqrt_upper) =
        sqrt_limb_sources(config, witness);
    push_limb_segment_with_check(
        segments,
        row_challenge,
        table,
        sqrt_lower,
        1,
        2,
        RangeValueLayout::Sqrt {
            input_poly: sqrt_input.clone(),
            feat_vars: sqrt_feat_vars,
            d_model: sqrt_d_model,
            std_poly: sqrt_std.clone(),
            upper: false,
        },
    );
    push_limb_segment_with_check(
        segments,
        row_challenge,
        table,
        sqrt_upper,
        1,
        2,
        RangeValueLayout::Sqrt {
            input_poly: sqrt_input,
            feat_vars: sqrt_feat_vars,
            d_model: sqrt_d_model,
            std_poly: sqrt_std,
            upper: true,
        },
    );

    let (ln_a, ln_q, ln_std, ln_rem, ln_upper) = layernorm_division_limb_sources(witness);
    let ln_q_values = layernorm_quotient_values(witness);
    let ln_q_real_len = ln_q_values.len();
    push_signed_quotient_bound_segment(
        segments,
        row_challenge,
        table,
        ln_q_values,
        ln_q_real_len,
        ln_q.clone(),
        quotient_bound_offset(config),
    );
    push_limb_segment_with_check(
        segments,
        row_challenge,
        table,
        ln_rem,
        0,
        2,
        RangeValueLayout::TypeB {
            dividend_poly: ln_a.clone(),
            quotient_poly: ln_q.clone(),
            divisor_poly: ln_std.clone(),
            upper: false,
        },
    );
    push_limb_segment_with_check(
        segments,
        row_challenge,
        table,
        ln_upper,
        0,
        2,
        RangeValueLayout::TypeB {
            dividend_poly: ln_a,
            quotient_poly: ln_q,
            divisor_poly: ln_std,
            upper: true,
        },
    );

    let (sm_a, sm_q, sm_sum, sm_rem, sm_upper) = softmax_division_limb_sources(config, witness);
    push_range20_segment_with_check(
        segments,
        row_challenge,
        table,
        sm_rem,
        0,
        RangeValueLayout::TypeB {
            dividend_poly: sm_a.clone(),
            quotient_poly: sm_q.clone(),
            divisor_poly: sm_sum.clone(),
            upper: false,
        },
    );
    push_range20_segment_with_check(
        segments,
        row_challenge,
        table,
        sm_upper,
        0,
        RangeValueLayout::TypeB {
            dividend_poly: sm_a,
            quotient_poly: sm_q,
            divisor_poly: sm_sum,
            upper: true,
        },
    );
}

fn push_limb_segment_with_check(
    segments: &mut Vec<QuerySegmentBuild>,
    row_challenge: EF,
    table: &RowOffsets,
    values: Vec<i128>,
    pad_value: i128,
    num_limbs: usize,
    source: RangeValueLayout,
) {
    let inputs = limb_inputs_from_values(&values, pad_value, num_limbs);
    let zeros = vec![0; inputs.len()];
    let real_len = inputs.len();
    segments.push(direct_segment_build(
        row_challenge,
        table.limb16,
        TYPE_LIMB16,
        inputs,
        zeros,
        0,
        0,
        Some(RangeValueCheckLayout {
            start: 0,
            len: 0,
            kind: RangeValueCheckKind::Limbs { num_limbs },
            source,
        }),
        real_len,
    ));
}

fn push_range20_segment_with_check(
    segments: &mut Vec<QuerySegmentBuild>,
    row_challenge: EF,
    table: &RowOffsets,
    values: Vec<i128>,
    pad_value: i128,
    source: RangeValueLayout,
) {
    let mut inputs = Vec::with_capacity(values.len().max(1).next_power_of_two());
    for value in values {
        assert!(
            (0..RANGE20_SIZE as i128).contains(&value),
            "RANGE20 value out of bounds: {value}"
        );
        inputs.push(value as u64);
    }
    assert!(
        (0..RANGE20_SIZE as i128).contains(&pad_value),
        "RANGE20 padding out of bounds: {pad_value}"
    );
    let real_len = inputs.len();
    let zeros = vec![0; real_len];
    segments.push(direct_segment_build(
        row_challenge,
        table.range20,
        TYPE_RANGE20,
        inputs,
        zeros,
        pad_value as u64,
        0,
        Some(RangeValueCheckLayout {
            start: 0,
            len: 0,
            kind: RangeValueCheckKind::Direct,
            source,
        }),
        real_len,
    ));
}

fn push_quotient_bound_segment(
    segments: &mut Vec<QuerySegmentBuild>,
    row_challenge: EF,
    table_offset: usize,
    ty: u64,
    range_size: usize,
    mut values: Vec<i128>,
    real_len: usize,
    quotient_poly: Vec<EF>,
    offset: u64,
) {
    assert!(real_len <= values.len());
    if values.len() < quotient_poly.len() {
        values.resize(quotient_poly.len(), 0);
    }
    assert_eq!(values.len(), quotient_poly.len());
    assert!(values.len().is_power_of_two());
    let mut inputs = Vec::with_capacity(values.len());
    for value in values {
        let shifted = value + offset as i128;
        assert!(
            (0..range_size as i128).contains(&shifted),
            "quotient bound value out of range: q={value}, offset={offset}, range_size={range_size}"
        );
        inputs.push(shifted as u64);
    }
    let zeros = vec![0; inputs.len()];
    segments.push(direct_segment_build(
        row_challenge,
        table_offset,
        ty,
        inputs,
        zeros,
        offset,
        0,
        Some(RangeValueCheckLayout {
            start: 0,
            len: 0,
            kind: RangeValueCheckKind::Direct,
            source: RangeValueLayout::QuotientBound {
                quotient_poly,
                offset,
            },
        }),
        real_len,
    ));
}

fn push_signed_quotient_bound_segment(
    segments: &mut Vec<QuerySegmentBuild>,
    row_challenge: EF,
    table: &RowOffsets,
    values: Vec<i128>,
    real_len: usize,
    quotient_poly: Vec<EF>,
    range_offset: u64,
) {
    if values.iter().all(|&value| {
        let shifted = value + range_offset as i128;
        (0..RANGE20_SIZE as i128).contains(&shifted)
    }) {
        push_quotient_bound_segment(
            segments,
            row_challenge,
            table.range20,
            TYPE_RANGE20,
            RANGE20_SIZE,
            values,
            real_len,
            quotient_poly,
            range_offset,
        );
    } else {
        let shifted_values: Vec<i128> = values
            .into_iter()
            .map(|value| value + QUOTIENT_LIMB_OFFSET as i128)
            .collect();
        push_limb_segment_with_check(
            segments,
            row_challenge,
            table,
            shifted_values,
            QUOTIENT_LIMB_OFFSET as i128,
            2,
            RangeValueLayout::QuotientBound {
                quotient_poly,
                offset: QUOTIENT_LIMB_OFFSET,
            },
        );
    }
}

fn direct_segment_build(
    row_challenge: EF,
    table_offset: usize,
    ty: u64,
    mut inputs: Vec<u64>,
    mut outputs: Vec<i64>,
    default_input: u64,
    default_output: i64,
    range_check: Option<RangeValueCheckLayout>,
    real_len: usize,
) -> QuerySegmentBuild {
    assert_eq!(inputs.len(), outputs.len());
    assert!(real_len <= inputs.len());
    let len = if inputs.len() == real_len {
        real_len.max(1).next_power_of_two()
    } else {
        assert!(inputs.len().is_power_of_two());
        inputs.len()
    };
    inputs.resize(len, default_input);
    outputs.resize(len, default_output);

    let mut values = Vec::with_capacity(len);
    let mut table_indices = Vec::with_capacity(len);
    let mut input_poly = Vec::with_capacity(len);
    let mut output_poly = Vec::with_capacity(len);
    for (&input, &output) in inputs.iter().zip(&outputs) {
        values.push(fold_row(row_challenge, input, output, ty));
        table_indices.push(table_offset + input as usize);
        input_poly.push(ef_from_u64(input));
        output_poly.push(ef_from_i64(output));
    }
    QuerySegmentBuild {
        layout: QuerySegmentLayout::Direct {
            start: 0,
            input_poly,
            output_poly,
            ty,
        },
        values,
        table_indices,
        real_len,
        range_check,
    }
}

fn limb_inputs_from_values(values: &[i128], pad_value: i128, num_limbs: usize) -> Vec<u64> {
    assert!(num_limbs.is_power_of_two());
    assert!(num_limbs <= 4);
    let max = 1i128 << (LIMB_BITS * num_limbs);
    let value_len = values.len().max(1).next_power_of_two();
    let mut inputs = Vec::with_capacity(value_len * num_limbs);
    for &value in values {
        assert!(
            (0..max).contains(&value),
            "{num_limbs}-limb value out of bounds: {value}"
        );
        let mut v = value as u128;
        for _ in 0..num_limbs {
            let limb = (v & 0xffff) as u64;
            inputs.push(limb);
            v >>= LIMB_BITS;
        }
        assert_eq!(v, 0, "{num_limbs}-limb value decomposition failed");
    }
    assert!(
        (0..max).contains(&pad_value),
        "{num_limbs}-limb padding out of bounds: {pad_value}"
    );
    let mut pad = pad_value as u128;
    let mut pad_limbs = Vec::with_capacity(num_limbs);
    for _ in 0..num_limbs {
        pad_limbs.push((pad & 0xffff) as u64);
        pad >>= LIMB_BITS;
    }
    assert_eq!(pad, 0, "{num_limbs}-limb padding decomposition failed");
    while inputs.len() < value_len * num_limbs {
        for &limb in &pad_limbs {
            inputs.push(limb);
        }
    }
    inputs
}

fn sqrt_limb_sources(
    config: &Config,
    witness: &Witness,
) -> (Vec<EF>, usize, u64, Vec<EF>, Vec<i128>, Vec<i128>) {
    let mut std = Vec::new();
    let mut lower = Vec::new();
    let mut upper = Vec::new();
    for ln in all_layernorms(witness) {
        for (&var_sum, &s) in ln.var_sum.iter().zip(&ln.std) {
            let x = var_sum as i128 + 1;
            let y = s as i128;
            std.push(ef_from_i64(s));
            lower.push(x - (y * y - y));
            upper.push(y * y + y - x);
        }
    }
    let input = encoded_layernorm_input_poly(config, witness);
    let feat_vars = config.d_model.next_power_of_two().trailing_zeros() as usize;
    let len = std.len().max(1).next_power_of_two();
    std.resize(len, EF::ONE);
    (input, feat_vars, config.d_model as u64, std, lower, upper)
}

fn layernorm_division_limb_sources(
    witness: &Witness,
) -> (Vec<EF>, Vec<EF>, Vec<EF>, Vec<i128>, Vec<i128>) {
    let mut a = Vec::new();
    let mut q = Vec::new();
    let mut std = Vec::new();
    let mut rems = Vec::new();
    let mut uppers = Vec::new();
    for ln in all_layernorms(witness) {
        for r in 0..ln.quotient.rows() {
            let s = ln.std[r] as i128;
            for c in 0..ln.quotient.cols() {
                let av = ln.dividend.get(r, c) as i128;
                let qv = ln.quotient.get(r, c) as i128;
                let rem = av - qv * s;
                a.push(ef_from_i64(ln.dividend.get(r, c)));
                q.push(ef_from_i64(ln.quotient.get(r, c)));
                std.push(ef_from_i64(ln.std[r]));
                rems.push(rem);
                uppers.push(s - 1 - rem);
            }
        }
    }
    let len = a.len().max(1).next_power_of_two();
    a.resize(len, EF::ZERO);
    q.resize(len, EF::ZERO);
    std.resize(len, EF::ONE);
    (a, q, std, rems, uppers)
}

fn softmax_division_limb_sources(
    config: &Config,
    witness: &Witness,
) -> (Vec<EF>, Vec<EF>, Vec<EF>, Vec<i128>, Vec<i128>) {
    let mut a = Vec::new();
    let mut q = Vec::new();
    let mut sum = Vec::new();
    let mut rems = Vec::new();
    let mut uppers = Vec::new();
    for block in &witness.blocks {
        for h in 0..block.attention.q_prob.len() {
            let probs = &block.attention.q_prob[h];
            let exp = &block.attention.exp[h];
            let sums = &block.attention.sum_exp[h];
            for i in 0..probs.d1() {
                let s = sums.get(i, 0) as i128;
                for j in 0..probs.d2() {
                    let dividend = exp.get(0, i, j) as i128 * config.scale as i128;
                    let qv = probs.get(0, i, j) as i128;
                    let rem = dividend - qv * s;
                    a.push(
                        EF::from(encode_i64(exp.get(0, i, j))) * ef_from_u64(config.scale as u64),
                    );
                    q.push(ef_from_i64(probs.get(0, i, j)));
                    sum.push(ef_from_i64(sums.get(i, 0)));
                    rems.push(rem);
                    uppers.push(s - 1 - rem);
                }
            }
        }
    }
    let len = a.len().max(1).next_power_of_two();
    a.resize(len, EF::ZERO);
    q.resize(len, EF::ZERO);
    sum.resize(len, EF::ONE);
    (a, q, sum, rems, uppers)
}

fn all_layernorms(witness: &Witness) -> Vec<&crate::witness::LayerNormWitness> {
    let mut out = Vec::with_capacity(witness.blocks.len() * 2 + 1);
    for block in &witness.blocks {
        out.push(&block.ln1);
        out.push(&block.ln2);
    }
    out.push(&witness.lnf);
    out
}

fn push_type_a_segments(
    config: &Config,
    weights: &ModelWeights,
    witness: &Witness,
    row_challenge: EF,
    table: &RowOffsets,
    segments: &mut Vec<QuerySegmentBuild>,
) {
    let q_bound_offset = quotient_bound_offset(config);
    for (layer, block) in witness.blocks.iter().enumerate() {
        let bw = &weights.blocks[layer];
        push_type_a_regular_segment(
            row_challenge,
            table,
            segments,
            &block.ln1.output,
            &bw.attn_w,
            &block.attention.q_qkv,
            config.scale as u64,
            q_bound_offset,
        );
        for h in 0..config.n_head {
            let q = qkv_head_matrix(config, block, h, 0);
            let k = qkv_head_matrix(config, block, h, 1);
            push_type_a_transposed_segment(
                row_challenge,
                table,
                segments,
                &q,
                &k,
                &block.attention.scores[h].head_matrix(0),
                config.sqrt_d_scale() as u64,
                q_bound_offset,
            );
            let prob = block.attention.q_prob[h].head_matrix(0);
            let v = qkv_head_matrix(config, block, h, 2);
            push_type_a_regular_segment(
                row_challenge,
                table,
                segments,
                &prob,
                &v,
                &block.attention.q_ao_heads[h].head_matrix(0),
                config.scale as u64,
                q_bound_offset,
            );
        }
        push_type_a_regular_segment(
            row_challenge,
            table,
            segments,
            &block.attention.q_ao,
            &bw.attn_proj_w,
            &block.attention.q_apr,
            config.scale as u64,
            q_bound_offset,
        );
        push_type_a_regular_segment(
            row_challenge,
            table,
            segments,
            &block.ln2.output,
            &bw.fc_w,
            &block.mlp.q_fc,
            config.scale as u64,
            q_bound_offset,
        );
        push_type_a_regular_segment(
            row_challenge,
            table,
            segments,
            &block.mlp.act,
            &bw.fproj_w,
            &block.mlp.q_fpr,
            config.scale as u64,
            q_bound_offset,
        );
    }
    push_type_a_transposed_segment(
        row_challenge,
        table,
        segments,
        &witness.lnf.output,
        &weights.wte,
        &witness.q_log,
        config.scale as u64,
        q_bound_offset,
    );
}

fn push_type_a_regular_segment(
    row_challenge: EF,
    table: &RowOffsets,
    segments: &mut Vec<QuerySegmentBuild>,
    a: &crate::tensor::Matrix,
    b: &crate::tensor::Matrix,
    q: &crate::tensor::Matrix,
    divisor: u64,
    quotient_bound_offset: u64,
) {
    assert_eq!(a.cols(), b.rows());
    assert_eq!(a.rows(), q.rows());
    assert_eq!(b.cols(), q.cols());
    let row_vars = q.rows().next_power_of_two().trailing_zeros() as usize;
    let col_vars = q.cols().next_power_of_two().trailing_zeros() as usize;
    let row_pow = 1usize << row_vars;
    let col_pow = 1usize << col_vars;
    let (ty, offset) = const_div_range(divisor, table);
    let mut values = Vec::with_capacity(row_pow * col_pow);
    let mut table_indices = Vec::with_capacity(row_pow * col_pow);
    for i in 0..row_pow {
        for j in 0..col_pow {
            let rem = if i < q.rows() && j < q.cols() {
                let mut product = 0i128;
                for k in 0..a.cols() {
                    product += a.get(i, k) as i128 * b.get(k, j) as i128;
                }
                product - q.get(i, j) as i128 * divisor as i128
            } else {
                0
            };
            assert!(rem >= 0 && rem < divisor as i128);
            values.push(fold_row(row_challenge, rem as u64, 0, ty));
            table_indices.push(offset + rem as usize);
        }
    }
    let q_poly = padded_matrix_poly(q, row_pow, col_pow);
    segments.push(QuerySegmentBuild {
        layout: QuerySegmentLayout::TypeA {
            start: 0,
            q_poly: q_poly.clone(),
            divisor,
            ty,
            matmul: PendingMatmul::Regular {
                a: a.clone(),
                b: b.clone(),
                row_vars,
                col_vars,
            },
        },
        values,
        table_indices,
        real_len: q.rows() * q.cols(),
        range_check: None,
    });
    push_signed_quotient_bound_segment(
        segments,
        row_challenge,
        table,
        padded_matrix_i128_values(q, row_pow, col_pow),
        q.rows() * q.cols(),
        q_poly,
        quotient_bound_offset,
    );
}

fn push_type_a_transposed_segment(
    row_challenge: EF,
    table: &RowOffsets,
    segments: &mut Vec<QuerySegmentBuild>,
    a: &crate::tensor::Matrix,
    b_t_source: &crate::tensor::Matrix,
    q: &crate::tensor::Matrix,
    divisor: u64,
    quotient_bound_offset: u64,
) {
    assert_eq!(a.cols(), b_t_source.cols());
    assert_eq!(a.rows(), q.rows());
    assert_eq!(b_t_source.rows(), q.cols());
    let row_vars = q.rows().next_power_of_two().trailing_zeros() as usize;
    let col_vars = q.cols().next_power_of_two().trailing_zeros() as usize;
    let row_pow = 1usize << row_vars;
    let col_pow = 1usize << col_vars;
    let (ty, offset) = const_div_range(divisor, table);
    let mut values = Vec::with_capacity(row_pow * col_pow);
    let mut table_indices = Vec::with_capacity(row_pow * col_pow);
    for i in 0..row_pow {
        for j in 0..col_pow {
            let rem = if i < q.rows() && j < q.cols() {
                let mut product = 0i128;
                for k in 0..a.cols() {
                    product += a.get(i, k) as i128 * b_t_source.get(j, k) as i128;
                }
                product - q.get(i, j) as i128 * divisor as i128
            } else {
                0
            };
            assert!(rem >= 0 && rem < divisor as i128);
            values.push(fold_row(row_challenge, rem as u64, 0, ty));
            table_indices.push(offset + rem as usize);
        }
    }
    let q_poly = padded_matrix_poly(q, row_pow, col_pow);
    segments.push(QuerySegmentBuild {
        layout: QuerySegmentLayout::TypeA {
            start: 0,
            q_poly: q_poly.clone(),
            divisor,
            ty,
            matmul: PendingMatmul::Transposed {
                a: a.clone(),
                b: b_t_source.clone(),
                row_vars,
                col_vars,
            },
        },
        values,
        table_indices,
        real_len: q.rows() * q.cols(),
        range_check: None,
    });
    push_signed_quotient_bound_segment(
        segments,
        row_challenge,
        table,
        padded_matrix_i128_values(q, row_pow, col_pow),
        q.rows() * q.cols(),
        q_poly,
        quotient_bound_offset,
    );
}

fn padded_matrix_poly(m: &crate::tensor::Matrix, row_pow: usize, col_pow: usize) -> Vec<EF> {
    let mut out = Vec::with_capacity(row_pow * col_pow);
    for r in 0..row_pow {
        for c in 0..col_pow {
            out.push(if r < m.rows() && c < m.cols() {
                ef_from_i64(m.get(r, c))
            } else {
                EF::ZERO
            });
        }
    }
    out
}

fn padded_matrix_i128_values(
    m: &crate::tensor::Matrix,
    row_pow: usize,
    col_pow: usize,
) -> Vec<i128> {
    let mut out = Vec::with_capacity(row_pow * col_pow);
    for r in 0..row_pow {
        for c in 0..col_pow {
            out.push(if r < m.rows() && c < m.cols() {
                m.get(r, c) as i128
            } else {
                0
            });
        }
    }
    out
}

fn quotient_bound_offset(config: &Config) -> u64 {
    (config.max_v * config.scale) as u64
}

fn layernorm_quotient_values(witness: &Witness) -> Vec<i128> {
    let mut out = Vec::new();
    for ln in all_layernorms(witness) {
        out.extend(ln.quotient.data().iter().map(|&v| v as i128));
    }
    out
}

fn encoded_i128_poly_padded(values: &[i128]) -> Vec<EF> {
    let len = values.len().max(1).next_power_of_two();
    let mut out = Vec::with_capacity(len);
    for &value in values {
        assert!(
            value >= i64::MIN as i128 && value <= i64::MAX as i128,
            "value outside i64 range: {value}"
        );
        out.push(ef_from_i64(value as i64));
    }
    out.resize(len, EF::ZERO);
    out
}

fn const_div_range(divisor: u64, table: &RowOffsets) -> (u64, usize) {
    match divisor {
        0..=4096 => (TYPE_RANGE12, table.range12),
        4097..=32768 => (TYPE_RANGE15, table.range15),
        _ => panic!("unsupported constant divisor {divisor}"),
    }
}

struct Table {
    rows: Vec<Row>,
}

#[derive(Clone)]
struct Row {
    folded: EF,
}

fn build_lut_table(config: &Config, weights: &ModelWeights, row_challenge: EF) -> Table {
    let mut rows = Vec::with_capacity(
        weights.exp_lut.len()
            + weights.gelu_lut.len()
            + LIMB_SIZE
            + 4096
            + 8192
            + 32768
            + RANGE20_SIZE,
    );
    for (idx, &out) in weights.exp_lut.iter().enumerate() {
        rows.push(Row {
            folded: fold_row(row_challenge, idx as u64, out, TYPE_EXP),
        });
    }
    for (idx, &out) in weights.gelu_lut.iter().enumerate() {
        rows.push(Row {
            folded: fold_row(row_challenge, idx as u64, out, TYPE_GELU),
        });
    }
    for limb in 0..LIMB_SIZE {
        rows.push(Row {
            folded: fold_row(row_challenge, limb as u64, 0, TYPE_LIMB16),
        });
    }
    for v in 0..4096 {
        rows.push(Row {
            folded: fold_row(row_challenge, v, 0, TYPE_RANGE12),
        });
    }
    for v in 0..8192 {
        rows.push(Row {
            folded: fold_row(row_challenge, v, 0, TYPE_RANGE13),
        });
    }
    for v in 0..32768 {
        rows.push(Row {
            folded: fold_row(row_challenge, v, 0, TYPE_RANGE15),
        });
    }
    for v in 0..RANGE20_SIZE {
        rows.push(Row {
            folded: fold_row(row_challenge, v as u64, 0, TYPE_RANGE20),
        });
    }
    assert_eq!(
        weights.exp_lut.len(),
        (config.max_v * config.scale + 1) as usize
    );
    assert_eq!(
        weights.gelu_lut.len(),
        (2 * config.max_v * config.scale + 1) as usize
    );
    Table { rows }
}

struct RowOffsets {
    exp: usize,
    gelu: usize,
    limb16: usize,
    range12: usize,
    range13: usize,
    range15: usize,
    range20: usize,
}

impl RowOffsets {
    fn new(config: &Config) -> Self {
        let exp = 0;
        let gelu = exp + (config.max_v * config.scale + 1) as usize;
        let limb16 = gelu + (2 * config.max_v * config.scale + 1) as usize;
        let range12 = limb16 + LIMB_SIZE;
        let range13 = range12 + 4096;
        let range15 = range13 + 8192;
        let range20 = range15 + 32768;
        Self {
            exp,
            gelu,
            limb16,
            range12,
            range13,
            range15,
            range20,
        }
    }
}

fn fold_row(row_challenge: EF, input: u64, output: i64, ty: u64) -> EF {
    ef_from_u64(input)
        + row_challenge * ef_from_i64(output)
        + row_challenge.square() * ef_from_u64(ty)
}

fn ef_from_i64(x: i64) -> EF {
    EF::from(encode_i64(x))
}

fn ef_from_u64(x: u64) -> EF {
    EF::from(Goldilocks::new(x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{model::BlockWeights, piop::prove, tensor::Matrix, witness::Config};

    #[test]
    fn unified_lut_logup_accepts_and_rejects_tamper() {
        let cfg = tiny_config();
        let weights = tiny_weights(&cfg);
        let piop = prove(&cfg, &weights, Matrix::new(2, 2, vec![1, -2, 3, 1]));

        let mut rng = rand::rng();
        let mut oracle = RandomOracle::<EF>::new(&mut rng);
        let mut protocol = prove_protocol(&cfg, &weights, &piop.witness, &mut oracle);

        oracle.restart();
        assert!(verify_protocol(&cfg, &protocol, &mut oracle));

        protocol.softmax_sum.source_polys[0][0] += EF::ONE;
        oracle.restart();
        assert!(!verify_protocol(&cfg, &protocol, &mut oracle));
        protocol.softmax_sum.source_polys[0][0] -= EF::ONE;

        protocol.softmax_max.source_polys[0][0] += EF::ONE;
        oracle.restart();
        assert!(!verify_protocol(&cfg, &protocol, &mut oracle));
        protocol.softmax_max.source_polys[0][0] -= EF::ONE;

        protocol.wiring.source_polys[0][0] += EF::ONE;
        oracle.restart();
        assert!(!verify_protocol(&cfg, &protocol, &mut oracle));
        protocol.wiring.source_polys[0][0] -= EF::ONE;

        match &mut protocol.lookup.query.segments[0] {
            QuerySegmentProof::Direct { input_poly, .. } => input_poly[0] += EF::ONE,
            QuerySegmentProof::TypeA { q_poly, .. } => q_poly[0] += EF::ONE,
            QuerySegmentProof::Padding { value, .. } => *value += EF::ONE,
        }
        oracle.restart();
        assert!(!verify_protocol(&cfg, &protocol, &mut oracle));
    }

    #[test]
    fn softmax_max_accepts_non_power_of_two_rows() {
        let mut cfg = tiny_config();
        cfg.n_seq = 3;
        let weights = tiny_weights(&cfg);
        let witness = prove(&cfg, &weights, Matrix::new(3, 2, vec![1, -2, 3, 1, 0, 2])).witness;

        let mut rng = rand::rng();
        let mut oracle = RandomOracle::<EF>::new(&mut rng);
        let proof = prove_softmax_max_check(&cfg, &witness, &mut oracle);

        oracle.restart();
        assert!(verify_softmax_max_check(&cfg, &proof, &mut oracle));
    }

    #[test]
    fn product_terminal_is_bound_by_sumcheck() {
        let mut rng = rand::rng();
        let mut oracle = RandomOracle::<EF>::new(&mut rng);
        let point = oracle.next_n_fields(6);
        let left = (0..64).map(|i| EF::from(Goldilocks::new(i + 3))).collect();
        let right = (0..64)
            .map(|i| EF::from(Goldilocks::new(5 * i + 7)))
            .collect();
        let proof = prove_product_eval_at(left, right, point.clone(), 0, &mut oracle);

        oracle.restart();
        let expected_point = oracle.next_n_fields(6);
        assert_eq!(expected_point, point);
        assert!(verify_product_eval_at(expected_point, &proof, &mut oracle));
    }

    #[test]
    fn matmul_terminal_is_bound_by_sumcheck() {
        let a = Matrix::new(8, 64, (0..512).map(|i| (i % 17) as i64 - 8).collect());
        let b = Matrix::new(64, 8, (0..512).map(|i| (i % 13) as i64 - 6).collect());
        let mut rng = rand::rng();
        let mut oracle = RandomOracle::<EF>::new(&mut rng);
        let row_point = oracle.next_n_fields(3);
        let col_point = oracle.next_n_fields(3);
        let proof =
            prove_regular_matmul_eval_at(&a, &b, row_point.clone(), col_point.clone(), &mut oracle);

        oracle.restart();
        let expected_row = oracle.next_n_fields(3);
        let expected_col = oracle.next_n_fields(3);
        assert!(verify_matmul_eval_at(
            &expected_row,
            &expected_col,
            &proof,
            &mut oracle
        ));
    }

    #[cfg(any())]
    #[test]
    fn source_matmul_reduces_to_real_oracle_claims() {
        let a = Matrix::new(8, 64, (0..512).map(|i| (i % 17) as i64 - 8).collect());
        let b = Matrix::new(64, 8, (0..512).map(|i| (i % 13) as i64 - 6).collect());
        let sources = OnlineProverCommitments::commit(BTreeMap::from([
            (
                "a".into(),
                a.data().iter().copied().map(encode_i64).collect::<Vec<_>>(),
            ),
            (
                "b".into(),
                b.data().iter().copied().map(encode_i64).collect::<Vec<_>>(),
            ),
        ]));
        let mut rng = rand::rng();
        let mut oracle = RandomOracle::<EF>::new(&mut rng);
        let row_point = oracle.next_n_fields(3);
        let col_point = oracle.next_n_fields(3);
        let (proof, claims) = prove_source_regular_matmul(
            "a",
            "b",
            &a,
            &b,
            row_point.clone(),
            col_point.clone(),
            &mut oracle,
        );
        let opening = sources.open(&claims, &mut oracle);

        oracle.restart();
        let expected_row = oracle.next_n_fields(3);
        let expected_col = oracle.next_n_fields(3);
        let claims = verify_source_regular_matmul(
            "a",
            "b",
            &expected_row,
            &expected_col,
            &proof,
            &mut oracle,
        )
        .expect("matmul sumcheck should verify");
        assert!(opening.verify(&sources.verifier, &claims, &mut oracle));
    }

    fn tiny_config() -> Config {
        Config {
            n_layer: 1,
            n_seq: 2,
            n_head: 1,
            d_head: 2,
            d_model: 2,
            mlp_hidden: 3,
            vocab: 4,
            scale: 4,
            max_v: 2,
        }
    }

    fn tiny_weights(cfg: &Config) -> ModelWeights {
        let exp_lut = (0..=(cfg.max_v * cfg.scale))
            .map(|i| if i == 0 { 0 } else { 1 })
            .collect();
        let gelu_lut = (0..=(2 * cfg.max_v * cfg.scale))
            .map(|i| i - cfg.max_v * cfg.scale)
            .collect();
        ModelWeights {
            wte: Matrix::new(4, 2, vec![1, 0, 0, 1, 1, 1, -1, 2]),
            wpe: Matrix::zeros(1024, 2),
            has_wpe: true,
            ln_f_g: vec![4, 4],
            ln_f_b: vec![0, 0],
            exp_lut,
            gelu_lut,
            blocks: vec![BlockWeights {
                ln_1_g: vec![4, 4],
                ln_1_b: vec![0, 0],
                attn_w: Matrix::new(2, 6, vec![1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]),
                attn_b: vec![0; 6],
                attn_proj_w: Matrix::new(2, 2, vec![1, 0, 0, 1]),
                attn_proj_b: vec![0, 0],
                ln_2_g: vec![4, 4],
                ln_2_b: vec![0, 0],
                fc_w: Matrix::new(2, 3, vec![1, 0, 1, 0, 1, -1]),
                fc_b: vec![0, 0, 0],
                fproj_w: Matrix::new(3, 2, vec![1, 0, 0, 1, 1, -1]),
                fproj_b: vec![0, 0],
            }],
        }
    }
}
