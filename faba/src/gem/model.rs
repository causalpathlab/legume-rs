//! Feature embedding by weighted pooling across modalities, plus
//! per-level pb heads + per-cell head.
//!
//! An input feature row has identity (gene_id, modality_id, region_id)
//! and embeds via the exp log-deviation gate:
//!
//!     AGG row  ({g}/AGG/total):       e_f = β_g
//!     comp row ({g}/{m}/{detail}):    e_f = β_g ⊙ exp(Σ_k z_{g,k}·δ_{k,m,:} + γ_{m,r,:})
//!
//! Biases are per-(gene, AGG) or per-(gene, modality). The RHS of the
//! bilinear `e_f · e_axis + b_f + b_axis` is one of:
//!
//!     Axis::Cell     — e_cell [N_cells, H]   (per-cell head)
//!     Axis::Pb(ℓ)    — e_pb_per_level[ℓ]     (per-level pb head)
//!
//! Composite-sum training (matches senna bge): each step sums the NCE
//! loss across the cell axis and every pb level; a single AdamW
//! `backward_step` updates every Var (β, z, δ, γ, b_agg, b_comp, e_cell,
//! b_cell, e_pb_per_level, b_pb_per_level). The shared feature side
//! gets gradient from every axis; each per-axis head accumulates only
//! from its own draws.
//!
//! Pb-level heads are training scaffolding that feeds gradient into
//! β/z/δ/γ at coarser (lower-variance) resolution. They're **not**
//! written to disk — only `e_cell` is a deliverable, alongside
//! `cell_to_pb.parquet` for downstream pb-level views.
//!
//! All public `embed_*` / `bias_*` / `rhs_*` methods take **plain
//! `&[u32]`** index slices — the sampler stays tensor-free.

use super::common::{candle_core, candle_nn};
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Init, VarMap};

/// Initialiser stdev for β, z, δ, γ, E_p, E_cell. Match the order of
/// magnitude used in graph-embedding-util's `JointEmbedModel`. Also the σ
/// of the per-feature prior null `e_f ~ N(0, σ²I)` that the feature
/// prior-score QC tests against (see `manifest::write_feature_prior_score`).
pub const PARAM_INIT_STD: f64 = 0.05;

/// Which right-hand-side embedding table the bilinear scores against.
/// The shared feature side (β, z, δ, γ) is reused across all axes; only
/// the pb/cell head varies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    /// Per-cell head. Bilinear `e_f · e_cell[c] + b_f + b_cell[c]`.
    Cell,
    /// Per-pb head at curriculum level `ℓ` (0 = coarsest).
    Pb(usize),
}

pub struct GemModel {
    pub n_modalities: usize,
    pub n_programs: usize,
    pub n_regions: usize,
    pub embedding_dim: usize,
    pub n_cells: usize,
    pub dev: Device,

    /// Owns every Var registered below. The optimizer pulls from here.
    pub varmap: VarMap,

    ////////////////////////////////////////
    // Feature params (shared across axes)
    ////////////////////////////////////////
    pub beta: Tensor, // [G, H] — base gene embedding (was `rho`)
    pub z: Tensor,    // [G, K] — gene's K-program "mode" loadings
    /// `[K, M, H]` — program × modality deviation **direction** (replaces
    /// the old scalar `q [K, M]`). `δ_{k,m,:}` is a full H-vector, so a
    /// program can push the satellite embedding in a new direction, not
    /// just rescale β_g. Note the name collides with senna-ETM / GWAS /
    /// APA β — distinct object, documented here only.
    pub delta: Tensor,
    /// `[M, R, H]` — additive per-(modality, region) log-space offset.
    /// Region = transcript-position bin; lets two same-gene/same-modality
    /// components in different regions diverge even at equal z.
    pub gamma: Tensor,
    pub b_agg: Tensor,  // [G]
    pub b_comp: Tensor, // [G, M]

    ////////////////////////////////////////
    // Cell-axis head (stage 2)
    ////////////////////////////////////////
    pub e_cell: Tensor, // [N_cells, H]
    pub b_cell: Tensor, // [N_cells]

    ////////////////////////////////////////
    // Per-level pb heads (stage 1 scaffolding, coarsest-first)
    ////////////////////////////////////////
    pub e_pb_per_level: Vec<Tensor>, // each [N_pb_ℓ, H]
    pub b_pb_per_level: Vec<Tensor>, // each [N_pb_ℓ]
}

impl GemModel {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_genes: usize,
        n_modalities: usize,
        n_programs: usize,
        n_regions: usize,
        embedding_dim: usize,
        n_cells: usize,
        n_pbs_per_level: &[usize],
        dev: &Device,
    ) -> Result<Self> {
        let n_regions = n_regions.max(1);
        let varmap = VarMap::new();
        let init_rand = Init::Randn {
            mean: 0.0,
            stdev: PARAM_INIT_STD,
        };
        let init_zero = Init::Const(0.0);

        let beta = varmap.get((n_genes, embedding_dim), "beta", init_rand, DType::F32, dev)?;
        let z = varmap.get((n_genes, n_programs), "z", init_rand, DType::F32, dev)?;
        // δ is the program×modality deviation *direction*: δ[k, m, :] is
        // a full H-vector so program k can move the satellite embedding
        // anywhere in H, not merely rescale β_g (the old scalar q). The
        // exp gate (see `embed_and_bias_rows`) matches the simulator's
        // generative `r_{g,m} ∝ exp(Σ_k z_{g,k} · A_{m,k,:})`.
        let delta = varmap.get(
            (n_programs, n_modalities, embedding_dim),
            "delta",
            init_rand,
            DType::F32,
            dev,
        )?;
        // γ[m, r, :] is the additive log-space region offset per modality.
        let gamma = varmap.get(
            (n_modalities, n_regions, embedding_dim),
            "gamma",
            init_rand,
            DType::F32,
            dev,
        )?;
        let b_agg = varmap.get(n_genes, "b_agg", init_zero, DType::F32, dev)?;
        let b_comp = varmap.get(
            (n_genes, n_modalities),
            "b_comp",
            init_zero,
            DType::F32,
            dev,
        )?;
        let e_cell = varmap.get(
            (n_cells.max(1), embedding_dim),
            "e_cell",
            init_rand,
            DType::F32,
            dev,
        )?;
        let b_cell = varmap.get(n_cells.max(1), "b_cell", init_zero, DType::F32, dev)?;

        let mut e_pb_per_level = Vec::with_capacity(n_pbs_per_level.len());
        let mut b_pb_per_level = Vec::with_capacity(n_pbs_per_level.len());
        for (l, &n_pb) in n_pbs_per_level.iter().enumerate() {
            let n_pb = n_pb.max(1);
            let e = varmap.get(
                (n_pb, embedding_dim),
                &format!("e_pb_l{l}"),
                init_rand,
                DType::F32,
                dev,
            )?;
            let b = varmap.get(n_pb, &format!("b_pb_l{l}"), init_zero, DType::F32, dev)?;
            e_pb_per_level.push(e);
            b_pb_per_level.push(b);
        }

        Ok(Self {
            n_modalities,
            n_programs,
            n_regions,
            embedding_dim,
            n_cells,
            dev: dev.clone(),
            varmap,
            beta,
            z,
            delta,
            gamma,
            b_agg,
            b_comp,
            e_cell,
            b_cell,
            e_pb_per_level,
            b_pb_per_level,
        })
    }

    /// Deterministically re-initialise the random-init Vars from a seeded
    /// host RNG. candle's CPU `Init::Randn` cannot be seeded via
    /// `Device::set_seed`, so without this the trained β / pre-L2 cell norms
    /// — and therefore the refine-pass QC cutoffs — wobble run-to-run.
    /// Biases (names starting `b`) stay zero; every other Var gets `N(0, σ²)`
    /// with `σ = PARAM_INIT_STD`. Vars are visited in sorted-name order so the
    /// draw sequence is independent of map iteration order.
    pub fn seed_init(&self, seed: u64) -> Result<()> {
        use rand::{rngs::StdRng, SeedableRng};
        use rand_distr::{Distribution, StandardNormal};
        let mut rng = StdRng::seed_from_u64(seed);
        let data = self.varmap.data().lock().unwrap();
        let mut names: Vec<&String> = data.keys().collect();
        names.sort();
        for name in names {
            let var = &data[name];
            let dims = var.shape().dims().to_vec();
            let n: usize = dims.iter().product();
            let values: Vec<f32> = if name.starts_with('b') {
                vec![0.0; n]
            } else {
                (0..n)
                    .map(|_| {
                        let x: f32 = StandardNormal.sample(&mut rng);
                        PARAM_INIT_STD as f32 * x
                    })
                    .collect()
            };
            let t = Tensor::from_vec(values, dims, &self.dev)?;
            var.set(&t)?;
        }
        Ok(())
    }

    fn idx_u32(&self, ids: &[u32]) -> Result<Tensor> {
        Ok(Tensor::from_slice(ids, ids.len(), &self.dev)?)
    }

    /// Build the `(agg_mask, not_agg_mask)` f32 tensors for a slate.
    /// One host→device upload; `not_agg` is computed on-device via affine.
    pub fn agg_masks_f32(&self, is_agg: &[bool]) -> Result<(Tensor, Tensor)> {
        let n = is_agg.len();
        let agg_buf: Vec<f32> = is_agg
            .iter()
            .map(|&a| if a { 1.0_f32 } else { 0.0 })
            .collect();
        let agg = Tensor::from_vec(agg_buf, n, &self.dev)?;
        let not_agg = agg.affine(-1.0, 1.0)?; // 1.0 - agg, computed on device
        Ok((agg, not_agg))
    }

    /// Feature-side embedding for a batch of rows. `gene_for_rho`
    /// indexes β (base gene embedding); `gene_for_z` indexes z (program
    /// loadings); `modality_for_q` selects the δ / γ modality slice;
    /// `region_for_delta` selects the γ region row. Positives and random
    /// in-class negatives coincide on all of these; swap-gene-mode swaps
    /// `gene_for_z`; swap-modality swaps `(modality_for_q,
    /// region_for_delta)`. `is_agg` zeros the log-deviation gate (AGG
    /// rows use β_g unmodified).
    ///
    /// **Exp log-deviation gate.** A satellite row deviates β_g by a
    /// per-row H-vector built from (z, δ, γ):
    ///
    ///     AGG row  ({g}/AGG/total):     e_f = β_g
    ///     comp row ({g}/{m}/{detail}):  e_f = β_g ⊙ exp(logdev_{g,m,r})
    ///     logdev_{g,m,r} = Σ_k z_{g,k} · δ_{k,m,:} + γ_{m,r,:}
    ///
    /// `exp(0) = 1` makes the gate the identity at z = 0, γ = 0, so a
    /// fresh model behaves like β_g on every row and gradient still
    /// flows through β. Unlike the old scalar gate, δ_{k,m,:} is a full
    /// H-vector, so a program can move the satellite in a **new**
    /// H-direction (not just rescale β_g's magnitude). exp(·) guarantees
    /// positivity and matches the simulator's generative
    /// `r ∝ exp(Σ z A)`. The additive γ_{m,r,:} resolves two same-gene
    /// components sitting in different transcript regions.
    ///
    /// Fused embed + bias for one slate. Builds the per-row `agg` /
    /// `not_agg` masks and the flat `b_comp` index tensor **once**, then
    /// reuses them for both the gate computation and the bias lookup.
    #[allow(clippy::too_many_arguments)]
    pub fn embed_and_bias_rows(
        &self,
        gene_for_rho: &[u32],
        gene_for_z: &[u32],
        modality_for_q: &[u32],
        region_for_delta: &[u32],
        gene_for_bias: &[u32],
        modality_for_bias: &[u32],
        is_agg: &[bool],
    ) -> Result<(Tensor, Tensor)> {
        let b = gene_for_rho.len();
        debug_assert_eq!(gene_for_z.len(), b);
        debug_assert_eq!(modality_for_q.len(), b);
        debug_assert_eq!(region_for_delta.len(), b);
        debug_assert_eq!(gene_for_bias.len(), b);
        debug_assert_eq!(modality_for_bias.len(), b);
        debug_assert_eq!(is_agg.len(), b);

        // Pack all six u32 index arrays plus pre-computed flat indices into a
        // single [6 × B] tensor — one host→device transfer instead of six
        // separate `from_slice`/`from_vec` calls.  Layout is [6, B] so each
        // `narrow(0, row, 1).squeeze(0)` yields a contiguous [B] view (no
        // copy on CUDA).  Rows:
        //   0  gene_for_rho
        //   1  gene_for_z
        //   2  modality_for_q
        //   3  gamma_flat = m * R + r          (pre-computed on CPU)
        //   4  gene_for_bias
        //   5  flat_b_comp = g * M + m         (pre-computed on CPU)
        let r_cols = self.n_regions as u32;
        let m_cols = self.n_modalities as u32;
        let mut idx_buf = Vec::<u32>::with_capacity(b * 6);
        idx_buf.extend_from_slice(gene_for_rho);
        idx_buf.extend_from_slice(gene_for_z);
        idx_buf.extend_from_slice(modality_for_q);
        idx_buf.extend(
            modality_for_q
                .iter()
                .zip(region_for_delta)
                .map(|(&m, &r)| m * r_cols + r),
        );
        idx_buf.extend_from_slice(gene_for_bias);
        idx_buf.extend(
            gene_for_bias
                .iter()
                .zip(modality_for_bias)
                .map(|(&g, &m)| g * m_cols + m),
        );
        let idx_t = Tensor::from_vec(idx_buf, (6, b), &self.dev)?;
        let g_rho = idx_t.narrow(0, 0, 1)?.squeeze(0)?; // [B]
        let g_z = idx_t.narrow(0, 1, 1)?.squeeze(0)?;
        let m_q = idx_t.narrow(0, 2, 1)?.squeeze(0)?;
        let gamma_idx = idx_t.narrow(0, 3, 1)?.squeeze(0)?;
        let g_bias_idx = idx_t.narrow(0, 4, 1)?.squeeze(0)?;
        let flat_idx = idx_t.narrow(0, 5, 1)?.squeeze(0)?;

        // Agg mask: one upload, not_agg computed on-device via affine.
        let agg_buf: Vec<f32> = is_agg
            .iter()
            .map(|&a| if a { 1.0_f32 } else { 0.0 })
            .collect();
        let agg = Tensor::from_vec(agg_buf, b, &self.dev)?;
        let not_agg = agg.affine(-1.0, 1.0)?; // 1.0 - agg

        // ── embedding side ──
        let h = self.embedding_dim;

        let beta_b = self.beta.index_select(&g_rho, 0)?; // [B, H]
        let z_b = self.z.index_select(&g_z, 0)?; // [B, K]

        // Σ_k z_{g,k} · δ_{k,m,:}  →  [B, H].
        // δ is [K, M, H]; gather the per-row modality slice δ[:, m, :]
        // (index_select on dim 1 → [K, B, H]) then contract over K via a
        // batched matmul z_b[B,1,K] · δ_m[B,K,H] → [B,1,H] (gemm; folds
        // away the explicit [B,K,H] product). `transpose` makes the slice
        // non-contiguous, so realise it before matmul.
        let delta_m = self
            .delta
            .index_select(&m_q, 1)?
            .transpose(0, 1)?
            .contiguous()?; // [B, K, H]
        let z_delta = z_b.unsqueeze(1)?.matmul(&delta_m)?.squeeze(1)?; // [B, H]

        // γ_{m,r,:}  →  [B, H].  γ is [M*R, H] after reshape.
        let gamma_b = self
            .gamma
            .reshape((self.n_modalities * self.n_regions, h))?
            .index_select(&gamma_idx, 0)?; // [B, H]

        let logdev = (z_delta + gamma_b)?; // [B, H]
                                           // Zero the deviation for AGG rows → e_f = β_g exactly.
        let logdev_masked = logdev.broadcast_mul(&not_agg.unsqueeze(1)?)?; // [B, H]
        let e = (beta_b * logdev_masked.exp()?)?;

        // ── bias side (reuses g_bias_idx, flat_idx, agg, not_agg) ──
        let b_agg_b = self.b_agg.index_select(&g_bias_idx, 0)?;
        let b_comp_b = self.b_comp.flatten_all()?.index_select(&flat_idx, 0)?;
        let bias = (((b_agg_b * &agg)?) + (b_comp_b * &not_agg)?)?;

        Ok((e, bias))
    }

    /// RHS embedding + bias for a batch of axis-ids. `Axis::Cell` → e_cell;
    /// `Axis::Pb(ℓ)` → e_pb_per_level[ℓ]. Returns `(E [B, H], b [B])`.
    pub fn rhs_rows(&self, axis: Axis, ids: &[u32]) -> Result<(Tensor, Tensor)> {
        let idx = self.idx_u32(ids)?;
        let (e_src, b_src) = match axis {
            Axis::Cell => (&self.e_cell, &self.b_cell),
            Axis::Pb(level) => (&self.e_pb_per_level[level], &self.b_pb_per_level[level]),
        };
        Ok((e_src.index_select(&idx, 0)?, b_src.index_select(&idx, 0)?))
    }

    /// Bilinear diagonal score: `Σ_h e_f[h] · e_rhs[h] + b_f + b_rhs`.
    pub fn score_diag(
        e_f: &Tensor,
        e_rhs: &Tensor,
        b_f: &Tensor,
        b_rhs: &Tensor,
    ) -> Result<Tensor> {
        let dot = (e_f * e_rhs)?.sum(1)?;
        Ok(((dot + b_f)? + b_rhs)?)
    }
}
