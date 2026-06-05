//! LigeSIS: a Ligero-style RS-code multilinear PCS whose column digests are committed with a
//! subset-sum (binary-SIS) hash and a secondary FRI-style PCS (here `Basefold` at coderate
//! ρ=1/2). This is the single-machine Protocol 1 of the LigeSIS paper.
//!
//! # Overview
//! `Commit` reshapes the `2^μ` evaluations of `f` into `F ∈ F^{m×n}`, RS-encodes each row at
//! rate 1/2 to `F' ∈ F^{m×2n}`, bit-decomposes `F'` into a binary `B ∈ {0,1}^{ηm×2n}`, hashes
//! its columns `H = A·B ∈ F^{C×2n}`, and commits `H` with `Basefold`. That `Basefold`
//! commitment *is* the LigeSIS commitment to `f`.
//!
//! `Eval` proves `f̄(z)=y`. Writing `z=(z1,z2)` (`|z1|=log m`, `|z2|=log n`), the prover sends
//! `ā = ēq_{z1}^T F` (so `y = ā(z2)`), the verifier samples a random column set `I ⊆ [2n]`,
//! and the prover commits the opened columns `B_I`. Soundness then rests on three checks over
//! `B_I`, all reduced to `Basefold` openings:
//!   1. **Binary check** — `B_I ∈ {0,1}` (sumcheck of `B_I·(B_I−1)·ēq_{α1} = 0`).
//!   2. **Hash consistency** — the subset-sum hash of the opened columns matches the committed
//!      digest `H` at the queried positions. Folding `H`'s `C` rows by a random `ēq_{α2}` and
//!      the `s` columns by `ēq_{r2}`, this is a tiny lookup tying a bilinear form on `(A,B_I)`
//!      to a single-point evaluation of `H`.
//!   3. **Reconstruction consistency** — `v⃗^T B_I = RS(ā)` at the queried positions, where
//!      `v⃗ = ēq_{z1} ⊗ (2⁰…2^{η−1})` (so `v⃗^T B = ēq_{z1}^T F' = RS(ēq_{z1}^T F) = RS(ā)`).
//!      `RS(ā)` is not separately committed; its needed evaluation is linked back to the
//!      commitment of `ā` by one extra sumcheck against the public encoding matrix.
//!
//! # Conventions
//! Matrices are flattened row-major (`flat[row*cols + col]`), so the *column* dimension takes
//! the low multilinear variables and the *row* dimension the high variables; an evaluation
//! "fold rows by `p_row`, cols by `p_col`" is `MlPoly(flat).eval([p_col, p_row].concat())`.
//! `sumcheck` binds the high index first, so a sumcheck challenge vector is reversed (`rev`)
//! before being used as an `MlPoly`/`Basefold` evaluation point. Every reduction is guarded by
//! a `debug_assert` against a directly-computed value.

use core::marker::PhantomData;

use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::{ExtensionField, Field, PrimeField64, TwoAdicField};
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};
use utils::{
    oracle::RandomOracle,
    poly::MlPoly,
    sumcheck::{self, SumcheckProof},
};

use utils::merkle::{MerkleTreeProver, MerkleTreeVerifier, Serialize};

use crate::basefold::{Basefold, BasefoldCommit, BatchProof, LOG_INTERLEAVE};
use crate::basefold::BasefoldProverState;
use crate::subset_sum::{C, SubsetSumHash};

/// Bits per base-field element used in the bit decomposition (Goldilocks canonical values are
/// `< 2^64`). `v⃗`'s weights are `2^0 … 2^{ETA-1}`.
pub const ETA: usize = 64;

/// Secondary-PCS code rate exponent: code length = `len << CODE_RATE`, i.e. ρ = 1/2.
const CODE_RATE: usize = 1;

/// Number of opened columns `|I| = s` (a power of two `< 2n`).
const NUM_COLS_OPENED: usize = 128;

pub struct Ligesis<BF, EF>(PhantomData<(BF, EF)>)
where
    BF: Field,
    EF: ExtensionField<BF>;

/// Public/proving key: the subset-sum hash and the secondary commitment to `Ā`.
pub struct ProvingKey<BF: Field, EF: Field> {
    pub hash: SubsetSumHash<BF>,
    /// `Basefold` prover state for the flattened hash matrix `Ā` (used to open it during `Eval`).
    pub a_state: BasefoldProverState<BF, EF>,
    pub com_a: [u8; 32],
    pub log_m: usize,
    pub log_n: usize,
}

/// Verifying key: the commitment to `Ā` and the matrix shape.
#[derive(Clone)]
pub struct VerifyingKey {
    pub com_a: [u8; 32],
    pub log_m: usize,
    pub log_n: usize,
    pub width: usize,
}

pub struct LigesisCommit(pub [u8; 32]);

/// Per-commitment prover data retained between `Commit` and `Eval`.
pub struct CommitData<BF: Field, EF: Field> {
    f: Vec<BF>,
    /// `b_cols[j]` is the bit column `B[:,j] ∈ {0,1}^{ηm}`, `j ∈ [2n]`.
    b_cols: Vec<Vec<bool>>,
    /// Digest matrix `H ∈ F^{C×2n}`, row-major.
    h_matrix: Vec<Vec<BF>>,
    /// Column-wise Merkle tree over `H` (leaf `j` = `H[:,j]`); its root is the commitment.
    h_merkle: MerkleTreeProver,
    _marker: PhantomData<EF>,
}

pub struct LigesisProof<BF: Field, EF: Field> {
    pub y: EF,
    com_a_vec: [u8; 32],
    com_i: [u8; 32],
    y4: EF,
    y5: EF,
    /// Opened digest columns `H[:,j]` for `j` in the (sorted) queried set, with their Merkle proof.
    h_cols: Vec<Vec<BF>>,
    h_open: Vec<u8>,
    sc_binary: SumcheckProof<EF>,
    sc_hash: SumcheckProof<EF>,
    sc_recon: SumcheckProof<EF>,
    sc_table: SumcheckProof<EF>,
    sc_rslink: SumcheckProof<EF>,
    /// Stage A: product-sumchecks reducing `B_I`'s 3 opening points and `ā`'s 2 points each to a
    /// single point, before the shared-domain batched FRI.
    sc_bi_batch: SumcheckProof<EF>,
    sc_a_batch: SumcheckProof<EF>,
    /// Stage B: one shared-RS-domain batched opening of `H`, `Ā`, `B_I`, `ā`.
    batch: BatchProof<BF, EF>,
}

impl<BF: Field, EF: Field> LigesisProof<BF, EF> {
    /// Serialized proof size in bytes: the claimed value, the four reduction scalars, the two
    /// auxiliary commitments, the seven sumcheck proofs, and the single batched opening.
    pub fn size_bytes(&self) -> usize {
        let ef = core::mem::size_of::<EF>();
        let bf = core::mem::size_of::<BF>();
        let mut s = ef * 3 + 64; // y, y4, y5, com_a_vec, com_i
        s += self.h_open.len() + self.h_cols.iter().map(|c| c.len() * bf).sum::<usize>();
        s += self.sc_binary.size_bytes()
            + self.sc_hash.size_bytes()
            + self.sc_recon.size_bytes()
            + self.sc_table.size_bytes()
            + self.sc_rslink.size_bytes()
            + self.sc_bi_batch.size_bytes()
            + self.sc_a_batch.size_bytes();
        s += self.batch.size_bytes();
        s
    }
}

/// Reverse a slice (sumcheck challenge order → `MlPoly` evaluation-point order).
fn rev<T: Clone>(v: &[T]) -> Vec<T> {
    v.iter().rev().cloned().collect()
}

/// Concatenate `low` (faster-varying variables) then `high` into one evaluation point.
fn cat<F: Clone>(low: &[F], high: &[F]) -> Vec<F> {
    low.iter().chain(high.iter()).cloned().collect()
}

/// `MlPoly(vec).eval(point)` without disturbing `vec`.
fn mle_eval<F: Field>(vec: &[F], point: &[F]) -> F {
    MlPoly(vec.to_vec()).eval(point)
}

/// The multilinear `g = Σ_j γ^j ēq_{points[j]}` as hypercube evaluations. Used to batch several
/// opening points of one polynomial into a single product-sumcheck `Σ_b p(b)·g(b)` whose claim is
/// `Σ_j γ^j p(points[j])`, reducing to a single evaluation point.
fn combine_eq<F: Field>(points: &[Vec<F>], gamma: F) -> Vec<F> {
    let len = 1usize << points[0].len();
    let mut out = vec![F::ZERO; len];
    let mut g = F::ONE;
    for p in points {
        let eqp = MlPoly::new_eq(p).0;
        for i in 0..len {
            out[i] += g * eqp[i];
        }
        g *= gamma;
    }
    out
}

/// Variable counts of the batched committed polys for an `m×n` shape, over {`Ā`, `B_I`, `ā`}.
/// (`H` is committed by a plain column-wise Merkle tree, not batched, so it's excluded.)
fn committed_vars(log_m: usize, log_n: usize) -> [usize; 3] {
    let log_c = C.trailing_zeros() as usize;
    let log_eta = ETA.trailing_zeros() as usize;
    let log_s = NUM_COLS_OPENED.trailing_zeros() as usize;
    let na = log_c + log_eta + log_m; // Ā : C·η·m
    let nbi = log_eta + log_m + log_s; // B_I : η·m·s
    let naa = log_n; // ā : n
    [na, nbi, naa]
}

fn max_committed_vars(log_m: usize, log_n: usize) -> usize {
    committed_vars(log_m, log_n).into_iter().max().unwrap()
}

fn min_committed_vars(log_m: usize, log_n: usize) -> usize {
    committed_vars(log_m, log_n).into_iter().min().unwrap()
}

/// Shared FRI fold-round count `R` for the batched open: the largest poly keeps the standard
/// `2^LOG_INTERLEAVE` layers, smaller polys use fewer so all share one RS domain `D = 2^{R+cr}`.
fn batch_rounds(log_m: usize, log_n: usize) -> usize {
    max_committed_vars(log_m, log_n) - LOG_INTERLEAVE
}

/// Interleave-layer exponent that puts a committed poly of `num_vars` variables onto the shared
/// domain. Requires `num_vars ≥ R` (see the shared-domain precondition, checked in `setup`).
fn shared_log_interleave(num_vars: usize, log_m: usize, log_n: usize) -> usize {
    num_vars - batch_rounds(log_m, log_n)
}

impl<BF, EF> Ligesis<BF, EF>
where
    BF: TwoAdicField + PrimeField64,
    EF: ExtensionField<BF> + ExtensionField<EF> + TwoAdicField,
    StandardUniform: Distribution<BF>,
{
    /// Powers of two `2^0 … 2^{ETA-1}` in the base field, by repeated doubling.
    fn pow2() -> Vec<BF> {
        let mut p = Vec::with_capacity(ETA);
        let mut cur = BF::ONE;
        for _ in 0..ETA {
            p.push(cur);
            cur = cur.double();
        }
        p
    }

    /// RS-encode a length-`len` vector (zero-padded to `2*len`) via the size-`2*len` DFT,
    /// matching `Basefold`'s encoding (`output[j] = Σ_k coeff_k · g^{jk}`, `g` the `2 len`-th
    /// two-adic generator).
    fn rs_encode_ext(coeffs: &[EF], len: usize) -> Vec<EF> {
        let mut c = coeffs.to_vec();
        c.resize(2 * len, EF::ZERO);
        Radix2Dit::<BF>::default().dft_algebra(c)
    }

    fn rs_encode_base(coeffs: &[BF], len: usize) -> Vec<BF> {
        let mut c = coeffs.to_vec();
        c.resize(2 * len, BF::ZERO);
        Radix2Dit::<BF>::default().dft(c)
    }

    /// `Setup`: sample the subset-sum hash for an `m×n` layout and commit its flattened matrix
    /// `Ā` with the secondary PCS.
    pub fn setup<R: Rng + ?Sized>(
        log_m: usize,
        log_n: usize,
        rng: &mut R,
    ) -> (ProvingKey<BF, EF>, VerifyingKey) {
        let m = 1usize << log_m;
        let width = ETA * m;
        // Shared-domain precondition: every batched poly (`Ā`, `B_I`, `ā`) must be within
        // `LOG_INTERLEAVE` vars of the largest, so they share one RS domain.
        assert!(
            max_committed_vars(log_m, log_n) - min_committed_vars(log_m, log_n) <= LOG_INTERLEAVE,
            "batched LigeSIS needs the {{Ā, B_I, ā}} var counts within LOG_INTERLEAVE of each other"
        );
        let hash = SubsetSumHash::<BF>::new(width, rng);
        let a_vars = (C * width).trailing_zeros() as usize;
        let (a_state, com) = Basefold::<BF, EF>::commit_base_on_domain(
            MlPoly(hash.a_mle_flat()),
            CODE_RATE,
            shared_log_interleave(a_vars, log_m, log_n),
        );
        let vk = VerifyingKey {
            com_a: com.0,
            log_m,
            log_n,
            width,
        };
        let pk = ProvingKey {
            hash,
            a_state,
            com_a: com.0,
            log_m,
            log_n,
        };
        (pk, vk)
    }

    /// `Commit`: encode rows, bit-decompose, subset-sum hash the columns, and `Basefold`-commit
    /// the digest matrix `H`. The returned commitment is the LigeSIS commitment to `f`.
    pub fn commit(
        pk: &ProvingKey<BF, EF>,
        f: MlPoly<BF>,
    ) -> (LigesisCommit, CommitData<BF, EF>) {
        let m = 1usize << pk.log_m;
        let n = 1usize << pk.log_n;
        let two_n = 2 * n;
        assert_eq!(f.0.len(), m * n);
        let f = f.0;

        // F[i][:] -> F'[i][:] = RS(row), length 2n; then bit-decompose into column-major bits.
        let mut b_cols = vec![vec![false; ETA * m]; two_n];
        for i in 0..m {
            let row = &f[i * n..(i + 1) * n];
            let enc = Self::rs_encode_base(row, n);
            for j in 0..two_n {
                let val = enc[j].as_canonical_u64();
                for k in 0..ETA {
                    if (val >> k) & 1 == 1 {
                        b_cols[j][ETA * i + k] = true;
                    }
                }
            }
        }

        let h_matrix = pk.hash.hash_columns(&b_cols); // C x 2n
        // Commit H column-wise with a plain Merkle tree: leaf j = serialize(H[:,j]) (the C
        // digest entries). H is raw digest data — it only needs binding, not a low-degree test —
        // so a vector commitment suffices, and its queried columns are opened directly in `Eval`.
        let leaves: Vec<Vec<u8>> = (0..two_n)
            .map(|j| {
                let col = (0..C).map(|r| h_matrix[r][j]).collect::<Vec<_>>();
                Serialize::serialize_fields(&col)
            })
            .collect();
        let h_merkle = MerkleTreeProver::new(&leaves);
        let com = h_merkle.commit();

        (
            LigesisCommit(com),
            CommitData {
                f,
                b_cols,
                h_matrix,
                h_merkle,
                _marker: PhantomData,
            },
        )
    }

    /// Deterministically sample `s` distinct column indices in `[2n]` from the oracle. Prover
    /// and verifier run the identical procedure on the same oracle stream.
    fn sample_columns(oracle: &mut RandomOracle<EF>, two_n: usize, s: usize) -> Vec<usize> {
        let raw = oracle.next_n_ints(s * 4);
        let mut seen = vec![false; two_n];
        let mut idx = Vec::with_capacity(s);
        for v in raw {
            let j = v % two_n;
            if !seen[j] {
                seen[j] = true;
                idx.push(j);
                if idx.len() == s {
                    break;
                }
            }
        }
        assert_eq!(idx.len(), s, "not enough distinct columns sampled from oracle");
        idx
    }

    pub fn prove(
        pk: &ProvingKey<BF, EF>,
        data: &CommitData<BF, EF>,
        z: Vec<EF>,
        oracle: &mut RandomOracle<EF>,
    ) -> LigesisProof<BF, EF> {
        let log_m = pk.log_m;
        let log_n = pk.log_n;
        let m = 1usize << log_m;
        let n = 1usize << log_n;
        let two_n = 2 * n;
        let width = ETA * m; // W
        let s = NUM_COLS_OPENED;
        let log_s = s.trailing_zeros() as usize;
        let log_w = width.trailing_zeros() as usize;
        assert_eq!(z.len(), log_m + log_n);
        let pow2 = Self::pow2();

        let z2 = z[0..log_n].to_vec();
        let z1 = z[log_n..].to_vec();
        let eqz1 = MlPoly::new_eq(&z1).0; // length m, EF

        // ā = ēq_{z1}^T F  (length n, in EF); y = ā(z2). Accumulate row-major: F is stored
        // row-major (`F[i][j] = f[i*n+j]`), so iterating rows outer / columns inner reads each
        // row contiguously, instead of striding by `n` (a cache/TLB miss per access for large n).
        let mut a_vec = vec![EF::ZERO; n];
        for i in 0..m {
            let w = eqz1[i];
            let row = &data.f[i * n..(i + 1) * n];
            for j in 0..n {
                a_vec[j] += w * EF::from(row[j]);
            }
        }        let y = mle_eval(&a_vec, &z2);
        let (a_vec_state, com_a_vec) = Basefold::<BF, EF>::commit_ext_on_domain(
            MlPoly(a_vec.clone()),
            CODE_RATE,
            shared_log_interleave(log_n, log_m, log_n),
        );

        // RS(ā), length 2n (used as the reconstruction table; not separately committed).
        let ra = Self::rs_encode_ext(&a_vec, n);

        // --- (a) sample opened columns I ---
        let idx = Self::sample_columns(oracle, two_n, s);

        // Open the queried digest columns H[:,I] directly from the Merkle tree (sorted indices
        // for rs_merkle); the verifier recomputes the hash-channel claim y2 from these.
        let mut sorted_idx = idx.clone();
        sorted_idx.sort_unstable();
        let h_open = data.h_merkle.open(&sorted_idx);
        let h_cols: Vec<Vec<BF>> = sorted_idx
            .iter()
            .map(|&j| (0..C).map(|r| data.h_matrix[r][j]).collect())
            .collect();

        // B_I commitment: flat[i*s + k] = B[i][I[k]] (column k low, row i high).
        let mut bi_flat = vec![BF::ZERO; width * s];
        for i in 0..width {
            for k in 0..s {
                if data.b_cols[idx[k]][i] {
                    bi_flat[i * s + k] = BF::ONE;
                }
            }
        }
        let bi_vars = (width * s).trailing_zeros() as usize;
        let (bi_state, com_i) = Basefold::<BF, EF>::commit_base_on_domain(
            MlPoly(bi_flat.clone()),
            CODE_RATE,
            shared_log_interleave(bi_vars, log_m, log_n),
        );

        // --- (b,c) challenges α1 (binary check), α2 (hash-row fold) ---
        let alpha1 = oracle.next_n_fields(log_w + log_s);
        let alpha2 = oracle.next_n_fields(C.trailing_zeros() as usize);
        let eq_alpha2 = MlPoly::new_eq(&alpha2).0; // length C

        // --- (d) binary check: Σ B_I(B_I-1) ēq_{α1} = 0 ---
        let bi_ef: Vec<EF> = bi_flat.iter().map(|&b| EF::from(b)).collect();
        let bi_minus1: Vec<EF> = bi_ef.iter().map(|&b| b - EF::ONE).collect();
        let eq_alpha1 = MlPoly::new_eq(&alpha1).0;
        let (sc_binary, ch_bin) =
            sumcheck::prove(vec![bi_ef, bi_minus1, eq_alpha1], oracle);
        let r1 = rev(&ch_bin); // B_I eval point (length log_w + log_s)

        // --- (e) r2: column-combination of the opened set ---
        let r2 = oracle.next_n_fields(log_s);
        let eq_r2 = MlPoly::new_eq(&r2).0; // length s

        // g1 = ēq_{α2}^T A (length W); g2 = B̄_I(·, r2) (length W).
        let g1: Vec<EF> = (0..width)
            .map(|i| {
                let mut acc = EF::ZERO;
                for r in 0..C {
                    acc += eq_alpha2[r] * EF::from(pk.hash.a[r][i]);
                }
                acc
            })
            .collect();
        let g2: Vec<EF> = (0..width)
            .map(|i| {
                let mut acc = EF::ZERO;
                for k in 0..s {
                    if data.b_cols[idx[k]][i] {
                        acc += eq_r2[k];
                    }
                }
                acc
            })
            .collect();

        // y4 (recon channel): subset sum of RS(ā) over the queried columns.
        let y4: EF = (0..s).map(|k| eq_r2[k] * ra[idx[k]]).sum();
        // Sanity: the hash-channel subset sum — which the verifier recomputes from the opened H
        // columns — equals the bilinear form the `sc_hash` sumcheck proves.
        debug_assert_eq!(
            (0..s)
                .map(|k| {
                    eq_r2[k]
                        * (0..C)
                            .map(|r| eq_alpha2[r] * EF::from(data.h_matrix[r][idx[k]]))
                            .sum::<EF>()
                })
                .sum::<EF>(),
            (0..width).map(|i| g1[i] * g2[i]).sum::<EF>()
        );

        // v⃗ = ēq_{z1} ⊗ (2^0..2^{η-1}) (length W).
        let v_vec: Vec<EF> = (0..width)
            .map(|i| eqz1[i / ETA] * EF::from(pow2[i % ETA]))
            .collect();
        debug_assert_eq!(y4, (0..width).map(|i| v_vec[i] * g2[i]).sum::<EF>());

        // --- (f) hash bilinear sumcheck: Σ_b̄ Ā(α2,b̄) B̄_I(b̄,r2) = y2 ---
        let (sc_hash, ch_hash) = sumcheck::prove(vec![g1.clone(), g2.clone()], oracle);
        let r4 = rev(&ch_hash); // length log_w

        // --- (g) recon bilinear sumcheck: Σ_b v⃗(b) B̄_I(b,r2) = y4 ---
        let (sc_recon, ch_recon) = sumcheck::prove(vec![v_vec.clone(), g2.clone()], oracle);
        let r5 = rev(&ch_recon); // length log_w

        // --- (h) recon table sumcheck over [2n]: Σ_j sel(j)·ra(j) = y4 → sel(r3)·ra(r3). ---
        let mut sel = vec![EF::ZERO; two_n];
        for k in 0..s {
            sel[idx[k]] = eq_r2[k];
        }
        debug_assert_eq!(y4, (0..two_n).map(|j| sel[j] * ra[j]).sum::<EF>());
        let (sc_table, ch_table) = sumcheck::prove(vec![sel, ra.clone()], oracle);
        let r3 = rev(&ch_table); // length log(2n)

        // y5 = RS(ā)(r3).
        let y5 = mle_eval(&ra, &r3);

        // --- (k) RS-link sumcheck: ⟨a⃗, w⟩ = y5, w[k] = Σ_j ēq_{r3}(j) g^{jk} ---
        let w_vec = Self::rs_weights(&r3, n, two_n);
        debug_assert_eq!(y5, (0..n).map(|k| a_vec[k] * w_vec[k]).sum::<EF>());
        let (sc_rslink, ch_rs) = sumcheck::prove(vec![a_vec.clone(), w_vec], oracle);

        // The single points each committed poly is opened at, after Stage A.
        let p_a = cat(&r4, &alpha2);
        let p_bi_hash = cat(&r2, &r4);
        let p_bi_recon = cat(&r2, &r5);
        let r6 = rev(&ch_rs);

        // --- Stage A: reduce the multi-point polys to a single point each via product-sumcheck.
        // B_I is opened at {r1, (r2,r4), (r2,r5)} with claims {B̄_I(r1), B̄_I(r4,r2), B̄_I(r5,r2)}
        // = {sc_binary.fe0, sc_hash.fe1, sc_recon.fe1}.
        let gamma_bi = oracle.next_field();
        let g_bi = combine_eq(&[r1.clone(), p_bi_hash, p_bi_recon], gamma_bi);
        let bi_ef_full: Vec<EF> = bi_flat.iter().map(|&b| EF::from(b)).collect();
        let (sc_bi_batch, ch_bi_batch) = sumcheck::prove(vec![bi_ef_full, g_bi], oracle);
        let r_bi = rev(&ch_bi_batch);
        // ā is opened at {z2, r6} with claims {ā(z2)=y, ā(r6)=sc_rslink.fe0}.
        let gamma_a = oracle.next_field();
        let g_a = combine_eq(&[z2.clone(), r6], gamma_a);
        let (sc_a_batch, ch_a_batch) = sumcheck::prove(vec![a_vec.clone(), g_a], oracle);
        let r_a = rev(&ch_a_batch);

        // --- Stage B: one shared-RS-domain batched opening of H, Ā, B_I (base-field leaves) and
        // ā (extension-field leaves).
        let batch = Basefold::<BF, EF>::batch_prove(
            &[(&pk.a_state, p_a), (&bi_state, r_bi)],
            &[(&a_vec_state, r_a)],
            oracle,
        );

        LigesisProof {
            y,
            com_a_vec: com_a_vec.0,
            com_i: com_i.0,
            y4,
            y5,
            h_cols,
            h_open,
            sc_binary,
            sc_hash,
            sc_recon,
            sc_table,
            sc_rslink,
            sc_bi_batch,
            sc_a_batch,
            batch,
        }
    }

    /// `w[k] = Σ_{j<2n} ēq_{r3}(j) · g^{jk}` for `k ∈ [n]`, `g` the `2n`-th two-adic generator.
    /// `= Π_t (1 − r3[t] + r3[t]·g^{2^t·k})` since `j = Σ_t j_t 2^t`.
    fn rs_weights(r3: &[EF], n: usize, two_n: usize) -> Vec<EF> {
        let log_2n = two_n.trailing_zeros() as usize;
        let g = BF::two_adic_generator(log_2n);
        (0..n)
            .map(|k| {
                let mut prod = EF::ONE;
                for t in 0..log_2n {
                    let e = (((1u64 << t) * k as u64) % two_n as u64) as u64;
                    let gp = EF::from(g.exp_u64(e));
                    prod *= EF::ONE - r3[t] + r3[t] * gp;
                }
                prod
            })
            .collect()
    }

    pub fn verify(
        vk: &VerifyingKey,
        commit: &LigesisCommit,
        z: Vec<EF>,
        proof: &LigesisProof<BF, EF>,
        oracle: &mut RandomOracle<EF>,
    ) -> bool {
        let log_m = vk.log_m;
        let log_n = vk.log_n;
        let m = 1usize << log_m;
        let n = 1usize << log_n;
        let two_n = 2 * n;
        let width = ETA * m;
        let s = NUM_COLS_OPENED;
        let log_s = s.trailing_zeros() as usize;
        if z.len() != log_m + log_n {
            return false;
        }
        let pow2 = Self::pow2();
        let z2 = z[0..log_n].to_vec();
        let z1 = z[log_n..].to_vec();
        let eqz1 = MlPoly::new_eq(&z1).0;

        // Mirror the prover's oracle consumption exactly.
        let idx = Self::sample_columns(oracle, two_n, s);

        // Verify the opened digest columns H[:,sorted_idx] against the commitment (Merkle root).
        let mut sorted_idx = idx.clone();
        sorted_idx.sort_unstable();
        if proof.h_cols.len() != sorted_idx.len() || proof.h_cols.iter().any(|c| c.len() != C) {
            return false;
        }
        let h_leaves: Vec<Vec<u8>> = proof
            .h_cols
            .iter()
            .map(|c| Serialize::serialize_fields(c))
            .collect();
        if !MerkleTreeVerifier::new(two_n, &commit.0).verify(
            proof.h_open.clone(),
            &sorted_idx,
            &h_leaves,
        ) {
            return false;
        }

        let alpha1 = oracle.next_n_fields(width.trailing_zeros() as usize + log_s);
        let alpha2 = oracle.next_n_fields(C.trailing_zeros() as usize);

        // (d) binary check.
        let ch_bin = match sumcheck::verify(EF::ZERO, &proof.sc_binary, oracle) {
            Some(c) => c,
            None => return false,
        };
        let r1 = rev(&ch_bin);
        let fe = &proof.sc_binary.final_evals;
        // fe[0] = B̄_I(r1) is carried into the Stage-A `B_I` batch; here only the
        // verifier-computable relations are checked.
        if fe.len() != 3 || fe[1] != fe[0] - EF::ONE || fe[2] != MlPoly::<EF>::eval_eq(&alpha1, &r1) {
            return false;
        }

        // (e) r2.
        let r2 = oracle.next_n_fields(log_s);
        let eq_r2 = MlPoly::new_eq(&r2).0; // length s

        // Hash-channel claim, recomputed from the opened H columns:
        // y2 = Σ_k ēq_{r2}(k)·(ēq_{α2}^T H[:,I[k]]).
        let eq_alpha2 = MlPoly::new_eq(&alpha2).0; // length C
        let y2: EF = (0..s)
            .map(|k| {
                let col = &proof.h_cols[sorted_idx.binary_search(&idx[k]).unwrap()];
                eq_r2[k] * (0..C).map(|r| eq_alpha2[r] * EF::from(col[r])).sum::<EF>()
            })
            .sum();

        // (f) hash bilinear sumcheck (claim y2).
        let ch_hash = match sumcheck::verify(y2, &proof.sc_hash, oracle) {
            Some(c) => c,
            None => return false,
        };
        let r4 = rev(&ch_hash);
        // fe[0] = Ā(α2,r4) becomes the `Ā` batch claim; fe[1] = B̄_I(r4,r2) feeds Stage-A `B_I`.
        if proof.sc_hash.final_evals.len() != 2 {
            return false;
        }

        // (g) recon bilinear sumcheck (claim y4).
        let ch_recon = match sumcheck::verify(proof.y4, &proof.sc_recon, oracle) {
            Some(c) => c,
            None => return false,
        };
        let r5 = rev(&ch_recon);
        let v_vec: Vec<EF> = (0..width)
            .map(|i| eqz1[i / ETA] * EF::from(pow2[i % ETA]))
            .collect();
        let fe = &proof.sc_recon.final_evals;
        // fe[1] = B̄_I(r5,r2) feeds Stage-A `B_I`; fe[0] = v⃗(r5) is verifier-computable.
        if fe.len() != 2 || fe[0] != mle_eval(&v_vec, &r5) {
            return false;
        }

        // (h) recon table sumcheck (claim y4): Σ_j sel(j)·ra(j) = y4 → sel(r3)·ra(r3)=y5.
        let ch_table = match sumcheck::verify(proof.y4, &proof.sc_table, oracle) {
            Some(c) => c,
            None => return false,
        };
        let r3 = rev(&ch_table);
        let mut sel = vec![EF::ZERO; two_n];
        for k in 0..s {
            sel[idx[k]] = eq_r2[k];
        }
        let fe = &proof.sc_table.final_evals;
        if fe.len() != 2 || fe[0] != mle_eval(&sel, &r3) || fe[1] != proof.y5 {
            return false;
        }

        // (k) RS-link sumcheck (claim y5).
        let ch_rs = match sumcheck::verify(proof.y5, &proof.sc_rslink, oracle) {
            Some(c) => c,
            None => return false,
        };
        let r6 = rev(&ch_rs);
        let w_vec = Self::rs_weights(&r3, n, two_n);
        let fe = &proof.sc_rslink.final_evals;
        // fe[0] = ā(r6) feeds Stage-A `ā`; fe[1] = w(r6) is verifier-computable.
        if fe.len() != 2 || fe[1] != mle_eval(&w_vec, &r6) {
            return false;
        }

        // The single points each committed poly is opened at, after Stage A.
        let p_a = cat(&r4, &alpha2);
        let p_bi_hash = cat(&r2, &r4);
        let p_bi_recon = cat(&r2, &r5);

        // --- Stage A verification: rebuild each multi-point batching sumcheck's claim from the
        // already-bound `final_evals`, verify it, and read off the single reduced point + value.
        let gamma_bi = oracle.next_field();
        let bi_pts = [r1.clone(), p_bi_hash.clone(), p_bi_recon.clone()];
        let bi_vals = [
            proof.sc_binary.final_evals[0],
            proof.sc_hash.final_evals[1],
            proof.sc_recon.final_evals[1],
        ];
        let claim_bi = horner(&bi_vals, gamma_bi);
        let ch_bi = match sumcheck::verify(claim_bi, &proof.sc_bi_batch, oracle) {
            Some(c) => c,
            None => return false,
        };
        let r_bi = rev(&ch_bi);
        if proof.sc_bi_batch.final_evals.len() != 2
            || proof.sc_bi_batch.final_evals[1] != combine_eq_eval(&bi_pts, gamma_bi, &r_bi)
        {
            return false;
        }
        let v_bi = proof.sc_bi_batch.final_evals[0];

        let gamma_a = oracle.next_field();
        let a_pts = [z2.clone(), r6.clone()];
        let a_vals = [proof.y, proof.sc_rslink.final_evals[0]];
        let claim_a = horner(&a_vals, gamma_a);
        let ch_a = match sumcheck::verify(claim_a, &proof.sc_a_batch, oracle) {
            Some(c) => c,
            None => return false,
        };
        let r_a = rev(&ch_a);
        if proof.sc_a_batch.final_evals.len() != 2
            || proof.sc_a_batch.final_evals[1] != combine_eq_eval(&a_pts, gamma_a, &r_a)
        {
            return false;
        }
        let v_a = proof.sc_a_batch.final_evals[0];

        // --- Stage B: one shared-RS-domain batched opening. Claims (base-field first):
        //   Ā @ p_a = sc_hash.fe0,  B_I @ r_bi = v_bi,  ā @ r_a = v_a.  (H is Merkle-committed,
        //   opened directly above, so it is not batched here.)
        let nv_a = (C * width).trailing_zeros() as usize;
        let nv_bi = (width * s).trailing_zeros() as usize;
        let li = |nv: usize| shared_log_interleave(nv, log_m, log_n);
        let com_aa = BasefoldCommit(vk.com_a);
        let com_bi = BasefoldCommit(proof.com_i);
        let com_av = BasefoldCommit(proof.com_a_vec);
        let bf_refs = vec![
            (&com_aa, p_a, proof.sc_hash.final_evals[0], li(nv_a)),
            (&com_bi, r_bi, v_bi, li(nv_bi)),
        ];
        let ef_refs = vec![(&com_av, r_a, v_a, li(log_n))];
        Basefold::<BF, EF>::batch_verify(&bf_refs, &ef_refs, &proof.batch, oracle)
    }
}

/// `Σ_j vals[j]·γ^j` (Horner) — the combined claim of a Stage-A multi-point batching sumcheck.
fn horner<F: Field>(vals: &[F], gamma: F) -> F {
    let mut acc = F::ZERO;
    for &v in vals.iter().rev() {
        acc = acc * gamma + v;
    }
    acc
}

/// `Σ_j γ^j ēq_{points[j]}(r)` — the verifier-side value of `combine_eq`'s MLE at `r`.
fn combine_eq_eval<F: Field>(points: &[Vec<F>], gamma: F, r: &[F]) -> F {
    let mut acc = F::ZERO;
    let mut g = F::ONE;
    for p in points {
        acc += g * MlPoly::<F>::eval_eq(p, &r.to_vec());
        g *= gamma;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_goldilocks::Goldilocks;
    use rand::RngExt;

    type EF = BinomialExtensionField<Goldilocks, 2>;
    type LS = Ligesis<Goldilocks, EF>;

    fn setup_commit(
        log_m: usize,
        log_n: usize,
        rng: &mut impl Rng,
    ) -> (
        ProvingKey<Goldilocks, EF>,
        VerifyingKey,
        Vec<Goldilocks>,
        Vec<EF>,
    ) {
        let mu = log_m + log_n;
        let f: Vec<Goldilocks> = (0..(1 << mu)).map(|_| rng.random()).collect();
        let z: Vec<EF> = (0..mu).map(|_| rng.random()).collect();
        let (pk, vk) = LS::setup(log_m, log_n, rng);
        (pk, vk, f, z)
    }

    #[test]
    fn round_trip() {
        let mut rng = rand::rng();
        let (log_m, log_n) = (3, 10); // shared-domain batch needs log_n >= log_m + log_s(=7)
        let (pk, vk, f, z) = setup_commit(log_m, log_n, &mut rng);

        let expected = MlPoly(f.iter().map(|&x| EF::from(x)).collect()).eval(&z);
        let (commit, data) = LS::commit(&pk, MlPoly(f));

        let mut oracle = RandomOracle::<EF>::new(&mut rng);
        let proof = LS::prove(&pk, &data, z.clone(), &mut oracle);
        assert_eq!(proof.y, expected);

        oracle.restart();
        assert!(LS::verify(&vk, &commit, z, &proof, &mut oracle));
    }

    #[test]
    fn tampered_eval_is_rejected() {
        let mut rng = rand::rng();
        let (log_m, log_n) = (3, 10);
        let (pk, vk, f, z) = setup_commit(log_m, log_n, &mut rng);
        let (commit, data) = LS::commit(&pk, MlPoly(f));

        let mut oracle = RandomOracle::<EF>::new(&mut rng);
        let mut proof = LS::prove(&pk, &data, z.clone(), &mut oracle);
        proof.y += EF::ONE;

        oracle.restart();
        assert!(!LS::verify(&vk, &commit, z, &proof, &mut oracle));
    }

    #[test]
    fn tampered_sumcheck_is_rejected() {
        let mut rng = rand::rng();
        let (log_m, log_n) = (3, 10);
        let (pk, vk, f, z) = setup_commit(log_m, log_n, &mut rng);
        let (commit, data) = LS::commit(&pk, MlPoly(f));

        let mut oracle = RandomOracle::<EF>::new(&mut rng);
        let mut proof = LS::prove(&pk, &data, z.clone(), &mut oracle);
        proof.sc_hash.round_polys[0][0] += EF::ONE;

        oracle.restart();
        assert!(!LS::verify(&vk, &commit, z, &proof, &mut oracle));
    }

    #[test]
    fn tampered_h_column_is_rejected() {
        let mut rng = rand::rng();
        let (log_m, log_n) = (3, 10);
        let (pk, vk, f, z) = setup_commit(log_m, log_n, &mut rng);
        let (commit, data) = LS::commit(&pk, MlPoly(f));

        let mut oracle = RandomOracle::<EF>::new(&mut rng);
        let mut proof = LS::prove(&pk, &data, z.clone(), &mut oracle);
        // Corrupt one opened digest entry: the Merkle check against the commitment must fail.
        proof.h_cols[0][0] += Goldilocks::ONE;

        oracle.restart();
        assert!(!LS::verify(&vk, &commit, z, &proof, &mut oracle));
    }
}
