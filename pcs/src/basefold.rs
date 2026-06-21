use core::marker::PhantomData;
use std::collections::BTreeMap;

use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::{ExtensionField, Field, TwoAdicField};
use utils::{
    merkle::{MerkleTreeProver, MerkleTreeVerifier, Serialize, hash_leaf},
    oracle::RandomOracle,
    poly::MlPoly,
};

/// Number of interleaved RS codewords the polynomial is split into at commit time.
/// The first `LOG_INTERLEAVE = log2(INTERLEAVE)` variables select the chunk, and are
/// handled by the eq-compression step rather than by folding rounds.
pub const INTERLEAVE: usize = 64;
pub const LOG_INTERLEAVE: usize = 6;

/// Column-tile width for the commit hashing pass and the `prove` compression; a tile
/// (`TILE * INTERLEAVE` field elements) fits in cache, so the column gather's reads stay
/// contiguous and its working set resident.
const TILE: usize = 64;

#[derive(Debug, Clone, Default)]
pub struct Basefold<BF, EF>
where
    BF: Field,
    EF: ExtensionField<BF>,
{
    _marker: PhantomData<(BF, EF)>,
}

/// `LF` is the leaf field the codewords live in (the base field `BF` for `commit_base`, the
/// extension field `EF` for `commit_ext`); `EF` is the folding/challenge field. Storing
/// base-field codewords as `BF` rather than upcasting to `EF` halves the committed and revealed
/// data for `commit_base`.
pub struct BasefoldProverState<LF: Field, EF: Field> {
    /// Interleaved RS codewords, codeword-major: `codes[j]` is the `j`-th codeword (length
    /// `L = code_length`), `j in 0..INTERLEAVE`. The committed Merkle leaves are the columns
    /// `codes[*][i]`; no transposed copy is stored — leaves are hashed tile-by-tile at commit,
    /// and the handful of queried columns are gathered lazily in `prove`.
    codes: Vec<Vec<LF>>,
    mt_prover: MerkleTreeProver,
    /// The full multilinear polynomial as hypercube evaluations (upcast to `EF`), used to
    /// drive the evaluation-reduction rounds.
    poly: MlPoly<EF>,
    num_vars: usize,
    code_rate: usize,
    /// `log2` of the interleave factor: the polynomial is split into `2^log_interleave`
    /// interleaved chunks, so `code_length = 2^{num_vars - log_interleave} << code_rate` and the
    /// FRI fold runs `num_vars - log_interleave` rounds. The first `log_interleave` variables are
    /// handled by eq-compression. `commit_base`/`commit_ext` use `LOG_INTERLEAVE`; the
    /// `*_on_domain` variants vary it so polynomials of different sizes share one RS domain.
    log_interleave: usize,
}

impl<LF: Field, EF: Field> BasefoldProverState<LF, EF> {
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn log_interleave(&self) -> usize {
        self.log_interleave
    }

    pub fn code_rate(&self) -> usize {
        self.code_rate
    }

    pub fn eval(&self, point: &[EF]) -> EF {
        self.poly.clone().eval(point)
    }
}

pub struct BasefoldCommit(pub [u8; 32]);

/// A Merkle opening at one oracle level: the proof bytes plus the revealed leaves
/// (keyed by leaf index). Level 0 leaves hold the `INTERLEAVE`-element interleaved column;
/// folded levels hold a single field element.
pub struct RoundOpening<F: Field> {
    proof_bytes: Vec<u8>,
    values: BTreeMap<usize, Vec<F>>,
}

pub struct BasefoldProof<LF: Field, EF: Field> {
    /// The two round-polynomial evals `(l_i(0), l_i(1))` per evaluation-reduction round
    /// (`num_vars - LOG_INTERLEAVE` of them). Sending both (rather than only `l_i(0)` and
    /// reconstructing `l_i(1)` from the claim) lets the opening point be **any** point,
    /// including ones with `0`/`1` coordinates (e.g. structured reshape/limb points).
    reduction_msgs: Vec<(EF, EF)>,
    /// Merkle roots of the intermediate folded oracles (levels `1..R`, the final level is
    /// sent in the clear as `final_codeword`).
    fold_roots: Vec<[u8; 32]>,
    /// The fully folded codeword (length `2^code_rate`), a constant RS codeword.
    final_codeword: Vec<EF>,
    /// Level-0 opening: the interleaved column leaves, in the leaf field `LF`.
    base_opening: RoundOpening<LF>,
    /// Openings for the folded oracle levels `1..R` (always in `EF`).
    fold_openings: Vec<RoundOpening<EF>>,
    claimed_eval: EF,
    num_vars: usize,
    code_rate: usize,
    log_interleave: usize,
}

/// Number of FRI queries: `ceil(100 / code_rate)` since the soundness error per query is
/// `2^{-code_rate}` (RS rate `1/2^code_rate`).
fn query_num(code_rate: usize) -> usize {
    (100 + code_rate - 1) / code_rate
}

impl<LF: Field, EF: Field> BasefoldProof<LF, EF> {
    /// The claimed evaluation `f(point)` this proof opens (`y` in `f(point) = y`).
    pub fn claimed_eval(&self) -> EF {
        self.claimed_eval
    }

    /// Serialized proof size in bytes: reduction messages + fold roots + final codeword +
    /// query openings (Merkle paths and revealed leaves) + the claimed evaluation.
    pub fn size_bytes(&self) -> usize {
        let lf = core::mem::size_of::<LF>();
        let ef = core::mem::size_of::<EF>();
        let mut s = self.reduction_msgs.len() * 2 * ef
            + self.fold_roots.len() * 32
            + self.final_codeword.len() * ef
            + ef;
        s += self.base_opening.proof_bytes.len();
        for v in self.base_opening.values.values() {
            s += v.len() * lf;
        }
        for op in &self.fold_openings {
            s += op.proof_bytes.len();
            for v in op.values.values() {
                s += v.len() * ef;
            }
        }
        s
    }
}

/// One commitment's level-0 Merkle opening inside a batched proof: the path bytes plus the
/// revealed interleaved-column leaves (each `2^{log_interleave}` values in the leaf field `LF`).
pub struct BaseOpening<LF: Field> {
    proof_bytes: Vec<u8>,
    values: BTreeMap<usize, Vec<LF>>,
}

/// A batched opening of several commitments — all sharing one RS domain `D = 2^{log_len}` — each
/// at a single point. The FRI fold chain (`fold_roots`, `final_codeword`, `fold_openings`) and
/// the query set are **shared**; only the per-channel `reduction_msgs` and the per-commitment
/// base openings are individual. Claimed values are inputs to `batch_verify`, not stored.
pub struct BatchProof<BF: Field, EF: Field> {
    /// `reduction_msgs[ch]` is the length-`R` evaluation-reduction chain of channel `ch`
    /// (channels ordered: all base-field commitments first, then extension-field ones).
    reduction_msgs: Vec<Vec<EF>>,
    fold_roots: Vec<[u8; 32]>,
    final_codeword: Vec<EF>,
    fold_openings: Vec<RoundOpening<EF>>,
    base_bf: Vec<BaseOpening<BF>>,
    base_ef: Vec<BaseOpening<EF>>,
    code_rate: usize,
    log_len: usize,
}

impl<BF: Field, EF: Field> BatchProof<BF, EF> {
    /// Serialized proof size in bytes: per-channel reduction chains + shared fold roots + final
    /// codeword + shared folded-level openings + per-commitment base openings (base-field leaves
    /// counted at `size_of::<BF>()`, extension-field at `size_of::<EF>()`).
    pub fn size_bytes(&self) -> usize {
        let bf = core::mem::size_of::<BF>();
        let ef = core::mem::size_of::<EF>();
        let mut s = self.reduction_msgs.iter().map(|c| c.len() * ef).sum::<usize>()
            + self.fold_roots.len() * 32
            + self.final_codeword.len() * ef;
        for op in &self.fold_openings {
            s += op.proof_bytes.len();
            for v in op.values.values() {
                s += v.len() * ef;
            }
        }
        for op in &self.base_bf {
            s += op.proof_bytes.len();
            for v in op.values.values() {
                s += v.len() * bf;
            }
        }
        for op in &self.base_ef {
            s += op.proof_bytes.len();
            for v in op.values.values() {
                s += v.len() * ef;
            }
        }
        s
    }
}

/// Eq-variant FRI fold of a codeword. Treating `values` as `P` evaluated on the size-`len`
/// subgroup `{g^i}` (natural order), the pair `(P(g^i), P(g^{i+h}))` recovers
/// `P_e(g^{2i}) = (x+nx)/2` and `P_o(g^{2i}) = (x-nx)·g^{-i}/2`. The eq binding of the LSB
/// variable is `(1-r)·P_e + r·P_o`, so the folded codeword RS-encodes the eq-folded
/// evaluation vector over the squared domain.
fn eq_fold_codeword<F: TwoAdicField>(values: &[F], r: F) -> Vec<F> {
    let len = values.len();
    let h = len / 2;
    let g = F::two_adic_generator(len.trailing_zeros() as usize);
    let g_inv = g.inverse();
    let inv2 = F::TWO.inverse();
    let mut twiddle = F::ONE;
    let mut out = Vec::with_capacity(h);
    for i in 0..h {
        let x = values[i];
        let nx = values[i + h];
        let pe = (x + nx) * inv2;
        let po = (x - nx) * twiddle * inv2;
        out.push((F::ONE - r) * pe + r * po);
        twiddle *= g_inv;
    }
    out
}

/// Single-position eq-fold (verifier side): folds the pair at `(pos, pos+len/2)` of a
/// size-`len` codeword, producing the value at position `pos` of the squared domain.
fn eq_fold_at<F: TwoAdicField>(a: F, b: F, r: F, len: usize, pos: usize) -> F {
    let g = F::two_adic_generator(len.trailing_zeros() as usize);
    let twiddle = g.inverse().exp_u64(pos as u64);
    let inv2 = F::TWO.inverse();
    let pe = (a + b) * inv2;
    let po = (a - b) * twiddle * inv2;
    (F::ONE - r) * pe + r * po
}

/// The opened leaf positions at a given level: for each base query index, both the fold
/// position `idx % half` and its sibling `+ half`. Deterministic from `base`, so prover
/// and verifier agree.
fn level_positions(base: &[usize], half: usize) -> Vec<usize> {
    let mut positions = Vec::with_capacity(base.len() * 2);
    for &idx in base {
        let pos = idx % half;
        positions.push(pos);
        positions.push(pos + half);
    }
    positions.sort_unstable();
    positions.dedup();
    positions
}

impl<BF, EF> Basefold<BF, EF>
where
    BF: TwoAdicField,
    EF: ExtensionField<BF> + TwoAdicField,
{
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }

    fn to_commit<LF: Field>(
        codes: Vec<Vec<LF>>,
        poly: MlPoly<EF>,
        code_rate: usize,
        log_interleave: usize,
    ) -> (BasefoldProverState<LF, EF>, BasefoldCommit) {
        let code_length = codes[0].len();
        // `codes.len() == 2^log_interleave` interleaved codewords; each Merkle leaf is one
        // column of `interleave` values. Hash the leaves tile-by-tile, never materializing the
        // full transpose: each tile gathers `TILE` columns into a small cache-resident buffer
        // using *contiguous* reads from every codeword (`cw[start..end]`), then serializes+hashes
        // each column. Avoids both a big transpose allocation and strided per-column reads.
        let interleave = codes.len();
        let mut tile = vec![LF::ZERO; TILE * interleave];
        let mut buf = Vec::with_capacity(interleave * core::mem::size_of::<LF>());
        let mut leaf_hashes = Vec::with_capacity(code_length);
        let mut start = 0;
        while start < code_length {
            let end = (start + TILE).min(code_length);
            let width = end - start;
            for j in 0..interleave {
                let cw = &codes[j];
                for local in 0..width {
                    tile[local * interleave + j] = cw[start + local];
                }
            }
            for local in 0..width {
                buf.clear();
                Serialize::serialize_fields_into(
                    &tile[local * interleave..(local + 1) * interleave],
                    &mut buf,
                );
                leaf_hashes.push(hash_leaf(&buf));
            }
            start = end;
        }
        let mt_prover = MerkleTreeProver::from_leaf_hashes(leaf_hashes);
        let commit = mt_prover.commit();
        let num_vars = poly.0.len().trailing_zeros() as usize;
        (
            BasefoldProverState {
                codes,
                mt_prover,
                poly,
                num_vars,
                code_rate,
                log_interleave,
            },
            BasefoldCommit(commit),
        )
    }

    pub fn commit_base(
        poly: MlPoly<BF>,
        code_rate: usize,
    ) -> (BasefoldProverState<BF, EF>, BasefoldCommit) {
        Self::commit_base_on_domain(poly, code_rate, LOG_INTERLEAVE)
    }

    /// Commit a same-domain family of base-field MLEs. Each polynomial retains its own Merkle
    /// root and prover state; batching here shares the caller-visible setup and avoids forcing
    /// users to duplicate the domain parameters before the later `batch_prove` call.
    pub fn batch_commit_base_on_domain(
        polies: Vec<MlPoly<BF>>,
        code_rate: usize,
        log_interleave: usize,
    ) -> Vec<(BasefoldProverState<BF, EF>, BasefoldCommit)> {
        polies
            .into_iter()
            .map(|poly| Self::commit_base_on_domain(poly, code_rate, log_interleave))
            .collect()
    }

    pub fn commit_ext(
        poly: MlPoly<EF>,
        code_rate: usize,
    ) -> (BasefoldProverState<EF, EF>, BasefoldCommit) {
        Self::commit_ext_on_domain(poly, code_rate, LOG_INTERLEAVE)
    }

    /// Commit splitting into `2^log_interleave` interleaved chunks, so the RS codeword has length
    /// `2^{num_vars - log_interleave} << code_rate`. Choosing `log_interleave = num_vars - R` for
    /// a shared `R` makes polynomials of different sizes land on one identical RS domain, which is
    /// what lets `batch_prove` fold them together.
    pub fn commit_base_on_domain(
        poly: MlPoly<BF>,
        code_rate: usize,
        log_interleave: usize,
    ) -> (BasefoldProverState<BF, EF>, BasefoldCommit) {
        let polies = poly.split(poly.0.len() >> log_interleave);
        let code_length = polies[0].0.len() << code_rate;
        let dft = Radix2Dit::<BF>::default();
        // Keep the codewords in the base field (8 bytes each) instead of upcasting to `EF`
        // (16 bytes); this halves the committed leaf data and the data revealed per query.
        let codes = polies
            .iter()
            .map(|poly| {
                let mut coeffs = poly.0.clone();
                coeffs.resize(code_length, BF::ZERO);
                dft.dft(coeffs)
            })
            .collect::<Vec<_>>();

        let poly_ef = MlPoly(poly.0.into_iter().map(EF::from).collect());
        Self::to_commit(codes, poly_ef, code_rate, log_interleave)
    }

    pub fn commit_ext_on_domain(
        poly: MlPoly<EF>,
        code_rate: usize,
        log_interleave: usize,
    ) -> (BasefoldProverState<EF, EF>, BasefoldCommit) {
        let polies = poly.split(poly.0.len() >> log_interleave);
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

        Self::to_commit(codes, poly, code_rate, log_interleave)
    }

    pub fn prove<LF: Field>(
        state: &BasefoldProverState<LF, EF>,
        point: Vec<EF>,
        oracle: &mut RandomOracle<EF>,
    ) -> BasefoldProof<LF, EF>
    where
        EF: ExtensionField<LF>,
    {
        let n = state.num_vars;
        let li = state.log_interleave;
        assert_eq!(point.len(), n);
        assert!(n >= li);
        let rounds = n - li;
        let l = state.codes[0].len(); // = code_length L

        let claimed_eval = state.poly.clone().eval(&point);

        // Compress the `2^li` interleaved codewords with eq of the first `li` variables:
        // `oracle0 = sum_j eq4[j] * codes[j]`. As an AXPY over codeword-major `codes`, each
        // codeword is read contiguously (a column-by-column dot would read strided).
        let eq4 = MlPoly::new_eq(&point[0..li].to_vec()).0;
        let mut oracle0 = vec![EF::ZERO; l];
        for j in 0..state.codes.len() {
            let w = eq4[j];
            let cw = &state.codes[j];
            for i in 0..l {
                oracle0[i] += w * EF::from(cw[i]);
            }
        }

        // The (n-li)-variate multilinear m = f(z[0..li], .) as hypercube evaluations.
        let mut m = state.poly.clone();
        m.fold(&point[0..li]);
        let zp = &point[li..];

        // Evaluation-reduction rounds interleaved with codeword folding.
        let mut reduction_msgs = Vec::with_capacity(rounds);
        let mut levels: Vec<Vec<EF>> = vec![oracle0];
        let mut y = claimed_eval;
        let mut challenges = Vec::with_capacity(rounds);
        for round in 0..rounds {
            let cur = m.0.len();
            let half = cur / 2;
            let m0 = (0..half).map(|i| m.0[2 * i]).collect::<Vec<_>>();
            let m1 = (0..half).map(|i| m.0[2 * i + 1]).collect::<Vec<_>>();
            let rest = &zp[round + 1..];
            let l0 = MlPoly(m0).eval(rest);
            let l1 = MlPoly(m1).eval(rest);
            debug_assert_eq!((EF::ONE - zp[round]) * l0 + zp[round] * l1, y);
            reduction_msgs.push((l0, l1));

            let r = oracle.next_field();
            challenges.push(r);
            y = (EF::ONE - r) * l0 + r * l1;
            m.fold(&[r]);

            let next = eq_fold_codeword(levels.last().unwrap(), r);
            levels.push(next);
        }

        let final_codeword = levels[rounds].clone();

        // Commit the intermediate folded oracles (levels 1..rounds); the final level is
        // revealed in the clear.
        let mut fold_roots = Vec::new();
        let mut folded_mts = Vec::new();
        for k in 1..rounds {
            let mt = MerkleTreeProver::new(
                &levels[k]
                    .iter()
                    .map(|&v| Serialize::serialize_fields(&[v]))
                    .collect(),
            );
            fold_roots.push(mt.commit());
            folded_mts.push(mt);
        }

        // Query phase.
        let q = query_num(state.code_rate);
        let mut base = oracle
            .next_n_ints(q)
            .into_iter()
            .map(|v| v % (l / 2))
            .collect::<Vec<_>>();
        base.sort_unstable();
        base.dedup();

        // Level 0: open the interleaved column leaves (in the leaf field `LF`).
        let base_positions = level_positions(&base, l / 2);
        let base_opening = RoundOpening {
            proof_bytes: state.mt_prover.open(&base_positions),
            values: base_positions
                .iter()
                .map(|&p| {
                    (
                        p,
                        (0..state.codes.len())
                            .map(|j| state.codes[j][p])
                            .collect::<Vec<LF>>(),
                    )
                })
                .collect::<BTreeMap<_, _>>(),
        };

        // Folded levels 1..rounds (in `EF`).
        let mut fold_openings = Vec::with_capacity(rounds.saturating_sub(1));
        for k in 1..rounds {
            let cur_len = l >> k;
            let positions = level_positions(&base, cur_len / 2);
            let mt = &folded_mts[k - 1];
            let proof_bytes = mt.open(&positions);
            let values = positions
                .iter()
                .map(|&p| (p, vec![levels[k][p]]))
                .collect::<BTreeMap<_, _>>();
            fold_openings.push(RoundOpening {
                proof_bytes,
                values,
            });
        }

        BasefoldProof {
            reduction_msgs,
            fold_roots,
            final_codeword,
            base_opening,
            fold_openings,
            claimed_eval,
            num_vars: n,
            code_rate: state.code_rate,
            log_interleave: li,
        }
    }

    pub fn verify<LF: Field>(
        commit: &BasefoldCommit,
        point: Vec<EF>,
        proof: &BasefoldProof<LF, EF>,
        oracle: &mut RandomOracle<EF>,
    ) -> bool
    where
        EF: ExtensionField<LF>,
    {
        let n = proof.num_vars;
        let li = proof.log_interleave;
        if point.len() != n || n < li {
            return false;
        }
        let rounds = n - li;
        let l = (1usize << (n - li)) << proof.code_rate;

        let eq4 = MlPoly::new_eq(&point[0..li].to_vec()).0;
        let zp = &point[li..];

        // Evaluation-reduction claim chain: rebuild each l_i(1) from the running claim and
        // the sent l_i(0), then advance with the round challenge.
        let mut y = proof.claimed_eval;
        let mut challenges = Vec::with_capacity(rounds);
        for round in 0..rounds {
            let (l0, l1) = proof.reduction_msgs[round];
            let zi = zp[round];
            // The round polynomial l_i(X) = (1−X)l0 + X·l1 must agree with the running claim at
            // z_i; valid at any z_i (including 0/1), since both evals are sent.
            if (EF::ONE - zi) * l0 + zi * l1 != y {
                return false;
            }
            let r = oracle.next_field();
            challenges.push(r);
            y = (EF::ONE - r) * l0 + r * l1;
        }

        // Recompute the query indices (same oracle stream as the prover).
        let q = query_num(proof.code_rate);
        let mut base = oracle
            .next_n_ints(q)
            .into_iter()
            .map(|v| v % (l / 2))
            .collect::<Vec<_>>();
        base.sort_unstable();
        base.dedup();

        // Verify the level-0 Merkle opening (leaves serialized in the leaf field `LF`).
        {
            let opening = &proof.base_opening;
            let leaf_indices = opening.values.keys().cloned().collect::<Vec<_>>();
            let leaves = leaf_indices
                .iter()
                .map(|p| Serialize::serialize_fields(&opening.values[p]))
                .collect::<Vec<_>>();
            let verifier = MerkleTreeVerifier::new(l, &commit.0);
            if !verifier.verify(opening.proof_bytes.clone(), &leaf_indices, &leaves) {
                return false;
            }
        }

        // Verify the folded-level Merkle openings (levels 1..rounds, in `EF`).
        for k in 1..rounds {
            let cur_len = l >> k;
            let opening = &proof.fold_openings[k - 1];
            let leaf_indices = opening.values.keys().cloned().collect::<Vec<_>>();
            let leaves = leaf_indices
                .iter()
                .map(|p| Serialize::serialize_fields(&opening.values[p]))
                .collect::<Vec<_>>();
            let verifier = MerkleTreeVerifier::new(cur_len, &proof.fold_roots[k - 1]);
            if !verifier.verify(opening.proof_bytes.clone(), &leaf_indices, &leaves) {
                return false;
            }
        }

        // Fold-consistency chain for each query index. Level 0 recomputes the compressed value
        // from the `LF` column (upcast to `EF`); folded levels read the single `EF` value.
        let val_at = |k: usize, p: usize| -> EF {
            if k == 0 {
                let leaf = &proof.base_opening.values[&p];
                let mut acc = EF::ZERO;
                for (j, &v) in leaf.iter().enumerate() {
                    acc += eq4[j] * EF::from(v);
                }
                acc
            } else {
                proof.fold_openings[k - 1].values[&p][0]
            }
        };
        let final_len = l >> rounds;
        for &idx in &base {
            let mut derived: Option<EF> = None;
            for k in 0..rounds {
                let cur_len = l >> k;
                let half = cur_len / 2;
                let pos = idx % half;
                let sib = pos + half;
                let a = val_at(k, pos);
                let b = val_at(k, sib);
                if let Some(d) = derived {
                    let prev_pos = idx % cur_len;
                    let expected = if prev_pos == pos { a } else { b };
                    if d != expected {
                        return false;
                    }
                }
                derived = Some(eq_fold_at(a, b, challenges[k], cur_len, pos));
            }
            let fpos = idx % final_len;
            if derived.unwrap() != proof.final_codeword[fpos] {
                return false;
            }
        }

        // The final codeword is a constant RS codeword equal to the reduced claim.
        let c0 = proof.final_codeword[0];
        if !proof.final_codeword.iter().all(|&v| v == c0) {
            return false;
        }
        if y != c0 {
            return false;
        }
        true
    }

    /// Build one batch channel from a commitment's `codes` (leaf field `LF`), its full polynomial
    /// (already upcast to `EF`), and a single evaluation point: the eq-compressed level-0 codeword
    /// `oracle0` (length `D`), the reduction driver `m = poly.fold(point[0..li])`, the residual
    /// point `zp = point[li..]` (length `R`), and the claimed evaluation.
    fn make_channel<LF: Field>(
        codes: &[Vec<LF>],
        poly: &MlPoly<EF>,
        point: &[EF],
        log_interleave: usize,
        d: usize,
    ) -> (Vec<EF>, MlPoly<EF>, Vec<EF>, EF)
    where
        EF: ExtensionField<LF>,
    {
        let eq4 = MlPoly::new_eq(&point[0..log_interleave].to_vec()).0;
        let mut oracle0 = vec![EF::ZERO; d];
        for j in 0..codes.len() {
            let w = eq4[j];
            let cw = &codes[j];
            for i in 0..d {
                oracle0[i] += w * EF::from(cw[i]);
            }
        }
        let mut m = poly.clone();
        m.fold(&point[0..log_interleave]);
        let y = poly.clone().eval(point);
        (oracle0, m, point[log_interleave..].to_vec(), y)
    }

    /// Batched opening: prove each `(state, point)` task at its single point, where **all** tasks
    /// share one RS domain `D` (commit them with matching `log_interleave = num_vars - R`). The
    /// codewords are random-linear-combined into one and folded `R` rounds with shared challenges,
    /// so the fold chain and query phase are shared across all polynomials. Base-field-leaf
    /// (`commit_base*`) and extension-field-leaf (`commit_ext*`) commitments are passed separately;
    /// channels are ordered base-field first, then extension-field.
    pub fn batch_prove(
        bf_tasks: &[(&BasefoldProverState<BF, EF>, Vec<EF>)],
        ef_tasks: &[(&BasefoldProverState<EF, EF>, Vec<EF>)],
        oracle: &mut RandomOracle<EF>,
    ) -> BatchProof<BF, EF> {
        let d = bf_tasks
            .first()
            .map(|(s, _)| s.codes[0].len())
            .or_else(|| ef_tasks.first().map(|(s, _)| s.codes[0].len()))
            .expect("batch_prove needs at least one task");
        let code_rate = bf_tasks
            .first()
            .map(|(s, _)| s.code_rate)
            .or_else(|| ef_tasks.first().map(|(s, _)| s.code_rate))
            .unwrap();
        for (s, _) in bf_tasks {
            assert_eq!(s.codes[0].len(), d, "all batched commitments must share one RS domain");
            assert_eq!(s.code_rate, code_rate);
        }
        for (s, _) in ef_tasks {
            assert_eq!(s.codes[0].len(), d, "all batched commitments must share one RS domain");
            assert_eq!(s.code_rate, code_rate);
        }
        let log_len = d.trailing_zeros() as usize;
        let rounds = log_len - code_rate; // shared R
        let num_channels = bf_tasks.len() + ef_tasks.len();

        // Build channels (base-field commitments first, then extension-field).
        let mut oracle0s = Vec::with_capacity(num_channels);
        let mut ms = Vec::with_capacity(num_channels);
        let mut zps = Vec::with_capacity(num_channels);
        let mut ys = Vec::with_capacity(num_channels);
        for (state, point) in bf_tasks {
            assert_eq!(point.len(), state.num_vars);
            let (o, m, zp, y) =
                Self::make_channel::<BF>(&state.codes, &state.poly, point, state.log_interleave, d);
            debug_assert_eq!(zp.len(), rounds);
            oracle0s.push(o);
            ms.push(m);
            zps.push(zp);
            ys.push(y);
        }
        for (state, point) in ef_tasks {
            assert_eq!(point.len(), state.num_vars);
            let (o, m, zp, y) =
                Self::make_channel::<EF>(&state.codes, &state.poly, point, state.log_interleave, d);
            debug_assert_eq!(zp.len(), rounds);
            oracle0s.push(o);
            ms.push(m);
            zps.push(zp);
            ys.push(y);
        }

        let coeffs = oracle.next_n_fields(num_channels);

        // Level-0 combined codeword: random linear combination of all channels' oracle0.
        let mut combined = vec![EF::ZERO; d];
        for ch in 0..num_channels {
            let w = coeffs[ch];
            for i in 0..d {
                combined[i] += w * oracle0s[ch][i];
            }
        }

        // Fold loop: every channel runs the same `R` reduction rounds, sharing the fold challenge.
        let mut reduction_msgs = vec![Vec::with_capacity(rounds); num_channels];
        let mut levels: Vec<Vec<EF>> = vec![combined];
        for round in 0..rounds {
            let mut l01 = Vec::with_capacity(num_channels);
            for ch in 0..num_channels {
                let m = &ms[ch];
                let cur = m.0.len();
                let half = cur / 2;
                let m0 = (0..half).map(|i| m.0[2 * i]).collect::<Vec<_>>();
                let m1 = (0..half).map(|i| m.0[2 * i + 1]).collect::<Vec<_>>();
                let rest = &zps[ch][round + 1..];
                let l0 = MlPoly(m0).eval(rest);
                let l1 = MlPoly(m1).eval(rest);
                debug_assert_eq!((EF::ONE - zps[ch][round]) * l0 + zps[ch][round] * l1, ys[ch]);
                reduction_msgs[ch].push(l0);
                l01.push((l0, l1));
            }
            let r = oracle.next_field();
            for ch in 0..num_channels {
                let (l0, l1) = l01[ch];
                ys[ch] = (EF::ONE - r) * l0 + r * l1;
                ms[ch].fold(&[r]);
            }
            let next = eq_fold_codeword(levels.last().unwrap(), r);
            levels.push(next);
        }
        let final_codeword = levels[rounds].clone();

        // Commit intermediate folded oracles (levels 1..rounds).
        let mut fold_roots = Vec::new();
        let mut folded_mts = Vec::new();
        for k in 1..rounds {
            let mt = MerkleTreeProver::new(
                &levels[k]
                    .iter()
                    .map(|&v| Serialize::serialize_fields(&[v]))
                    .collect(),
            );
            fold_roots.push(mt.commit());
            folded_mts.push(mt);
        }

        // Shared query set (one domain ⇒ identical positions for every commitment).
        let q = query_num(code_rate);
        let mut base = oracle
            .next_n_ints(q)
            .into_iter()
            .map(|v| v % (d / 2))
            .collect::<Vec<_>>();
        base.sort_unstable();
        base.dedup();
        let base_positions = level_positions(&base, d / 2);

        let base_bf = bf_tasks
            .iter()
            .map(|(state, _)| BaseOpening {
                proof_bytes: state.mt_prover.open(&base_positions),
                values: base_positions
                    .iter()
                    .map(|&p| {
                        (
                            p,
                            (0..state.codes.len())
                                .map(|j| state.codes[j][p])
                                .collect::<Vec<BF>>(),
                        )
                    })
                    .collect::<BTreeMap<_, _>>(),
            })
            .collect::<Vec<_>>();
        let base_ef = ef_tasks
            .iter()
            .map(|(state, _)| BaseOpening {
                proof_bytes: state.mt_prover.open(&base_positions),
                values: base_positions
                    .iter()
                    .map(|&p| {
                        (
                            p,
                            (0..state.codes.len())
                                .map(|j| state.codes[j][p])
                                .collect::<Vec<EF>>(),
                        )
                    })
                    .collect::<BTreeMap<_, _>>(),
            })
            .collect::<Vec<_>>();

        // Shared folded-level openings.
        let mut fold_openings = Vec::with_capacity(rounds.saturating_sub(1));
        for k in 1..rounds {
            let cur_len = d >> k;
            let positions = level_positions(&base, cur_len / 2);
            let mt = &folded_mts[k - 1];
            let proof_bytes = mt.open(&positions);
            let values = positions
                .iter()
                .map(|&p| (p, vec![levels[k][p]]))
                .collect::<BTreeMap<_, _>>();
            fold_openings.push(RoundOpening { proof_bytes, values });
        }

        BatchProof {
            reduction_msgs,
            fold_roots,
            final_codeword,
            fold_openings,
            base_bf,
            base_ef,
            code_rate,
            log_len,
        }
    }

    /// Verify a [`BatchProof`]. Each claim is `(commit, point, claimed_value, log_interleave)`;
    /// base-field-leaf commitments first, then extension-field, matching `batch_prove`'s ordering.
    pub fn batch_verify(
        bf_claims: &[(&BasefoldCommit, Vec<EF>, EF, usize)],
        ef_claims: &[(&BasefoldCommit, Vec<EF>, EF, usize)],
        proof: &BatchProof<BF, EF>,
        oracle: &mut RandomOracle<EF>,
    ) -> bool {
        let d = 1usize << proof.log_len;
        let code_rate = proof.code_rate;
        let rounds = proof.log_len - code_rate;
        let num_bf = bf_claims.len();
        let num_channels = num_bf + ef_claims.len();
        if proof.reduction_msgs.len() != num_channels
            || proof.base_bf.len() != num_bf
            || proof.base_ef.len() != ef_claims.len()
            || proof.fold_roots.len() != rounds.saturating_sub(1)
            || proof.fold_openings.len() != rounds.saturating_sub(1)
        {
            return false;
        }

        // Per-channel metadata (ordered bf, then ef): point, log_interleave, eq-compress weights.
        let mut points: Vec<&Vec<EF>> = Vec::with_capacity(num_channels);
        let mut lis: Vec<usize> = Vec::with_capacity(num_channels);
        let mut eq4s: Vec<Vec<EF>> = Vec::with_capacity(num_channels);
        let mut ys: Vec<EF> = Vec::with_capacity(num_channels);
        for (_, point, claim, li) in bf_claims.iter().chain(ef_claims.iter()) {
            if point.len() < *li || point.len() - li != rounds || proof.reduction_msgs[points.len()].len() != rounds {
                return false;
            }
            eq4s.push(MlPoly::new_eq(&point[0..*li].to_vec()).0);
            points.push(point);
            lis.push(*li);
            ys.push(*claim);
        }

        let coeffs = oracle.next_n_fields(num_channels);

        // Evaluation-reduction chains (shared challenge per round).
        let mut challenges = Vec::with_capacity(rounds);
        for round in 0..rounds {
            let r = oracle.next_field();
            challenges.push(r);
            for ch in 0..num_channels {
                let l0 = proof.reduction_msgs[ch][round];
                let zi = points[ch][lis[ch] + round];
                if zi == EF::ZERO {
                    return false;
                }
                let l1 = (ys[ch] - (EF::ONE - zi) * l0) * zi.inverse();
                ys[ch] = (EF::ONE - r) * l0 + r * l1;
            }
        }

        // Recompute the shared query set.
        let q = query_num(code_rate);
        let mut base = oracle
            .next_n_ints(q)
            .into_iter()
            .map(|v| v % (d / 2))
            .collect::<Vec<_>>();
        base.sort_unstable();
        base.dedup();
        let base_positions = level_positions(&base, d / 2);

        // Verify each commitment's base Merkle opening.
        for (i, (commit, _, _, _)) in bf_claims.iter().enumerate() {
            let op = &proof.base_bf[i];
            let indices = op.values.keys().cloned().collect::<Vec<_>>();
            if indices != base_positions {
                return false;
            }
            let leaves = indices
                .iter()
                .map(|p| Serialize::serialize_fields(&op.values[p]))
                .collect::<Vec<_>>();
            if !MerkleTreeVerifier::new(d, &commit.0).verify(op.proof_bytes.clone(), &indices, &leaves) {
                return false;
            }
        }
        for (i, (commit, _, _, _)) in ef_claims.iter().enumerate() {
            let op = &proof.base_ef[i];
            let indices = op.values.keys().cloned().collect::<Vec<_>>();
            if indices != base_positions {
                return false;
            }
            let leaves = indices
                .iter()
                .map(|p| Serialize::serialize_fields(&op.values[p]))
                .collect::<Vec<_>>();
            if !MerkleTreeVerifier::new(d, &commit.0).verify(op.proof_bytes.clone(), &indices, &leaves) {
                return false;
            }
        }

        // Verify the shared folded-level Merkle openings.
        for k in 1..rounds {
            let cur_len = d >> k;
            let op = &proof.fold_openings[k - 1];
            let indices = op.values.keys().cloned().collect::<Vec<_>>();
            let leaves = indices
                .iter()
                .map(|p| Serialize::serialize_fields(&op.values[p]))
                .collect::<Vec<_>>();
            if !MerkleTreeVerifier::new(cur_len, &proof.fold_roots[k - 1]).verify(
                op.proof_bytes.clone(),
                &indices,
                &leaves,
            ) {
                return false;
            }
        }

        // Fold-consistency on the combined codeword. Level 0 is the coeff-weighted sum of every
        // channel's eq-compressed leaf; folded levels read the committed combined value.
        let val_at = |k: usize, p: usize| -> EF {
            if k == 0 {
                let mut acc = EF::ZERO;
                for ch in 0..num_channels {
                    let leaf_val = if ch < num_bf {
                        let leaf = &proof.base_bf[ch].values[&p];
                        let mut a = EF::ZERO;
                        for (j, &v) in leaf.iter().enumerate() {
                            a += eq4s[ch][j] * EF::from(v);
                        }
                        a
                    } else {
                        let leaf = &proof.base_ef[ch - num_bf].values[&p];
                        let mut a = EF::ZERO;
                        for (j, &v) in leaf.iter().enumerate() {
                            a += eq4s[ch][j] * v;
                        }
                        a
                    };
                    acc += coeffs[ch] * leaf_val;
                }
                acc
            } else {
                proof.fold_openings[k - 1].values[&p][0]
            }
        };
        let final_len = d >> rounds;
        for &idx in &base {
            let mut derived: Option<EF> = None;
            for k in 0..rounds {
                let cur_len = d >> k;
                let half = cur_len / 2;
                let pos = idx % half;
                let sib = pos + half;
                let a = val_at(k, pos);
                let b = val_at(k, sib);
                if let Some(dd) = derived {
                    let prev_pos = idx % cur_len;
                    let expected = if prev_pos == pos { a } else { b };
                    if dd != expected {
                        return false;
                    }
                }
                derived = Some(eq_fold_at(a, b, challenges[k], cur_len, pos));
            }
            let fpos = idx % final_len;
            if derived.unwrap() != proof.final_codeword[fpos] {
                return false;
            }
        }

        // Final constant must equal the coeff-weighted sum of the channels' reduced claims.
        let c0 = proof.final_codeword[0];
        if !proof.final_codeword.iter().all(|&v| v == c0) {
            return false;
        }
        let combined_claim: EF = (0..num_channels).map(|ch| coeffs[ch] * ys[ch]).sum();
        if combined_claim != c0 {
            return false;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_goldilocks::Goldilocks;
    use rand::RngExt;

    type GoldilocksExt2 = BinomialExtensionField<Goldilocks, 2>;

    #[test]
    fn prover_can_be_parameterized_by_base_and_extension_fields() {
        let _prover = Basefold::<Goldilocks, GoldilocksExt2>::new();
    }

    #[test]
    fn commit_prove_verify_round_trip() {
        let mut rng = rand::rng();
        let n = 12usize;
        let code_rate = 2usize;

        let coeffs = (0..(1 << n))
            .map(|_| rng.random::<Goldilocks>())
            .collect::<Vec<_>>();
        let poly = MlPoly(coeffs.clone());
        let point = (0..n)
            .map(|_| rng.random::<GoldilocksExt2>())
            .collect::<Vec<_>>();

        let (state, commit) = Basefold::<Goldilocks, GoldilocksExt2>::commit_base(poly, code_rate);

        let mut oracle = RandomOracle::<GoldilocksExt2>::new(&mut rng);
        let proof =
            Basefold::<Goldilocks, GoldilocksExt2>::prove(&state, point.clone(), &mut oracle);

        // claimed_eval matches an independent multilinear evaluation.
        let expected =
            MlPoly(coeffs.iter().map(|&x| GoldilocksExt2::from(x)).collect()).eval(&point);
        assert_eq!(proof.claimed_eval, expected);

        oracle.restart();
        assert!(Basefold::<Goldilocks, GoldilocksExt2>::verify(
            &commit,
            point.clone(),
            &proof,
            &mut oracle
        ));
    }

    #[test]
    fn commit_ext_round_trip() {
        let mut rng = rand::rng();
        let n = 12usize;
        let code_rate = 2usize;

        let coeffs = (0..(1 << n))
            .map(|_| rng.random::<GoldilocksExt2>())
            .collect::<Vec<_>>();
        let poly = MlPoly(coeffs.clone());
        let point = (0..n)
            .map(|_| rng.random::<GoldilocksExt2>())
            .collect::<Vec<_>>();

        let (state, commit) = Basefold::<Goldilocks, GoldilocksExt2>::commit_ext(poly, code_rate);

        let mut oracle = RandomOracle::<GoldilocksExt2>::new(&mut rng);
        let proof =
            Basefold::<Goldilocks, GoldilocksExt2>::prove(&state, point.clone(), &mut oracle);
        assert_eq!(proof.claimed_eval, MlPoly(coeffs).eval(&point));

        oracle.restart();
        assert!(Basefold::<Goldilocks, GoldilocksExt2>::verify(
            &commit,
            point.clone(),
            &proof,
            &mut oracle
        ));
    }

    #[test]
    fn tampered_proof_is_rejected() {
        let mut rng = rand::rng();
        let n = 12usize;
        let code_rate = 2usize;

        let coeffs = (0..(1 << n))
            .map(|_| rng.random::<Goldilocks>())
            .collect::<Vec<_>>();
        let poly = MlPoly(coeffs);
        let point = (0..n)
            .map(|_| rng.random::<GoldilocksExt2>())
            .collect::<Vec<_>>();

        let (state, commit) = Basefold::<Goldilocks, GoldilocksExt2>::commit_base(poly, code_rate);
        let mut oracle = RandomOracle::<GoldilocksExt2>::new(&mut rng);
        let mut proof =
            Basefold::<Goldilocks, GoldilocksExt2>::prove(&state, point.clone(), &mut oracle);

        proof.reduction_msgs[0].0 += GoldilocksExt2::ONE;

        oracle.restart();
        assert!(!Basefold::<Goldilocks, GoldilocksExt2>::verify(
            &commit,
            point.clone(),
            &proof,
            &mut oracle
        ));
    }

    type LS = Basefold<Goldilocks, GoldilocksExt2>;

    /// Commit `bf` (base-field) and `ef` (extension-field) polynomials of the given sizes onto
    /// one shared RS domain (`log_interleave = n - r`), open all at fresh single points, and
    /// return the proof, commitments, points, and expected claims.
    fn batch_setup(
        bf_sizes: &[usize],
        ef_sizes: &[usize],
        r: usize,
        code_rate: usize,
        rng: &mut impl rand::Rng,
    ) -> (
        Vec<(BasefoldProverState<Goldilocks, GoldilocksExt2>, BasefoldCommit, Vec<GoldilocksExt2>, GoldilocksExt2)>,
        Vec<(BasefoldProverState<GoldilocksExt2, GoldilocksExt2>, BasefoldCommit, Vec<GoldilocksExt2>, GoldilocksExt2)>,
    ) {
        let bf: Vec<_> = bf_sizes
            .iter()
            .map(|&n| {
                let coeffs: Vec<Goldilocks> = (0..(1 << n)).map(|_| rng.random()).collect();
                let point: Vec<GoldilocksExt2> = (0..n).map(|_| rng.random()).collect();
                let claim =
                    MlPoly(coeffs.iter().map(|&x| GoldilocksExt2::from(x)).collect()).eval(&point);
                let (state, commit) = LS::commit_base_on_domain(MlPoly(coeffs), code_rate, n - r);
                (state, commit, point, claim)
            })
            .collect();
        let ef: Vec<_> = ef_sizes
            .iter()
            .map(|&n| {
                let coeffs: Vec<GoldilocksExt2> = (0..(1 << n)).map(|_| rng.random()).collect();
                let point: Vec<GoldilocksExt2> = (0..n).map(|_| rng.random()).collect();
                let claim = MlPoly(coeffs.clone()).eval(&point);
                let (state, commit) = LS::commit_ext_on_domain(MlPoly(coeffs), code_rate, n - r);
                (state, commit, point, claim)
            })
            .collect();
        (bf, ef)
    }

    #[test]
    fn batch_open_mixed_sizes_and_fields_round_trip() {
        let mut rng = rand::rng();
        let code_rate = 1usize;
        let r = 6usize; // largest (2^12) uses 64 layers; shared domain D = 2^(r+cr)
        let (bf, ef) = batch_setup(&[12, 10, 9], &[11], r, code_rate, &mut rng);

        let bf_tasks: Vec<_> = bf.iter().map(|(s, _, p, _)| (s, p.clone())).collect();
        let ef_tasks: Vec<_> = ef.iter().map(|(s, _, p, _)| (s, p.clone())).collect();

        let mut oracle = RandomOracle::<GoldilocksExt2>::new(&mut rng);
        let proof = LS::batch_prove(&bf_tasks, &ef_tasks, &mut oracle);

        let bf_claims: Vec<_> = bf
            .iter()
            .map(|(_, c, p, v)| (c, p.clone(), *v, p.len() - r))
            .collect();
        let ef_claims: Vec<_> = ef
            .iter()
            .map(|(_, c, p, v)| (c, p.clone(), *v, p.len() - r))
            .collect();

        oracle.restart();
        assert!(LS::batch_verify(&bf_claims, &ef_claims, &proof, &mut oracle));
    }

    #[test]
    fn batch_open_tampering_is_rejected() {
        let mut rng = rand::rng();
        let code_rate = 1usize;
        let r = 6usize;
        let (bf, ef) = batch_setup(&[12, 10], &[11], r, code_rate, &mut rng);

        let bf_tasks: Vec<_> = bf.iter().map(|(s, _, p, _)| (s, p.clone())).collect();
        let ef_tasks: Vec<_> = ef.iter().map(|(s, _, p, _)| (s, p.clone())).collect();
        let mut oracle = RandomOracle::<GoldilocksExt2>::new(&mut rng);
        let proof = LS::batch_prove(&bf_tasks, &ef_tasks, &mut oracle);

        let mk_claims = || {
            let bf_claims: Vec<_> = bf
                .iter()
                .map(|(_, c, p, v)| (c, p.clone(), *v, p.len() - r))
                .collect::<Vec<_>>();
            let ef_claims: Vec<_> = ef
                .iter()
                .map(|(_, c, p, v)| (c, p.clone(), *v, p.len() - r))
                .collect::<Vec<_>>();
            (bf_claims, ef_claims)
        };

        // Sanity: untampered verifies.
        {
            let (bc, ec) = mk_claims();
            oracle.restart();
            assert!(LS::batch_verify(&bc, &ec, &proof, &mut oracle));
        }
        // Tampered reduction message.
        {
            let mut p = clone_batch_proof(&proof);
            p.reduction_msgs[0][0] += GoldilocksExt2::ONE;
            let (bc, ec) = mk_claims();
            oracle.restart();
            assert!(!LS::batch_verify(&bc, &ec, &p, &mut oracle));
        }
        // Wrong input claim.
        {
            let (mut bc, ec) = mk_claims();
            bc[0].2 += GoldilocksExt2::ONE;
            oracle.restart();
            assert!(!LS::batch_verify(&bc, &ec, &proof, &mut oracle));
        }
        // Tampered base leaf.
        {
            let mut p = clone_batch_proof(&proof);
            let first_key = *p.base_bf[0].values.keys().next().unwrap();
            p.base_bf[0].values.get_mut(&first_key).unwrap()[0] += Goldilocks::ONE;
            let (bc, ec) = mk_claims();
            oracle.restart();
            assert!(!LS::batch_verify(&bc, &ec, &p, &mut oracle));
        }
    }

    /// Deep-ish clone of a `BatchProof` for tamper tests (the struct isn't `Clone`).
    fn clone_batch_proof(
        p: &BatchProof<Goldilocks, GoldilocksExt2>,
    ) -> BatchProof<Goldilocks, GoldilocksExt2> {
        let clone_round = |o: &RoundOpening<GoldilocksExt2>| RoundOpening {
            proof_bytes: o.proof_bytes.clone(),
            values: o.values.clone(),
        };
        BatchProof {
            reduction_msgs: p.reduction_msgs.clone(),
            fold_roots: p.fold_roots.clone(),
            final_codeword: p.final_codeword.clone(),
            fold_openings: p.fold_openings.iter().map(clone_round).collect(),
            base_bf: p
                .base_bf
                .iter()
                .map(|o| BaseOpening {
                    proof_bytes: o.proof_bytes.clone(),
                    values: o.values.clone(),
                })
                .collect(),
            base_ef: p
                .base_ef
                .iter()
                .map(|o| BaseOpening {
                    proof_bytes: o.proof_bytes.clone(),
                    values: o.values.clone(),
                })
                .collect(),
            code_rate: p.code_rate,
            log_len: p.log_len,
        }
    }
}
