//! Feature embedding by weighted pooling across modalities, plus
//! per-level pb heads + per-cell head.
//!
//! Per the plan, an input feature row has identity (gene_id, modality_id)
//! and embeds as:
//!
//!     AGG row  ({g}/AGG/total):       e_f = ρ_g
//!     comp row ({g}/{m}/{detail}):    e_f = ρ_g + Σ_k z_{g,k} · Q_{k,m,:}
//!
//! Biases are per-(gene, AGG) or per-(gene, modality). The RHS of the
//! bilinear `e_f · e_axis + b_f + b_axis` is one of:
//!
//!     Axis::Cell     — e_cell [N_cells, H]   (per-cell head)
//!     Axis::Pb(ℓ)    — e_pb_per_level[ℓ]     (per-level pb head)
//!
//! Composite-sum training (matches senna bge): each step sums the NCE
//! loss across the cell axis and every pb level; a single AdamW
//! `backward_step` updates every Var (ρ, z, Q, b_agg, b_comp, e_cell,
//! b_cell, e_pb_per_level, b_pb_per_level). The shared feature side
//! gets gradient from every axis; each per-axis head accumulates only
//! from its own draws.
//!
//! Pb-level heads are training scaffolding that feeds gradient into
//! ρ/z/Q at coarser (lower-variance) resolution. They're **not**
//! written to disk — only `e_cell` is a deliverable, alongside
//! `cell_to_pb.parquet` for downstream pb-level views.
//!
//! All public `embed_*` / `bias_*` / `rhs_*` methods take **plain
//! `&[u32]`** index slices — the sampler stays tensor-free.

use super::common::{candle_core, candle_nn};
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Init, VarMap};

/// Initialiser stdev for ρ, z, Q, E_p, E_cell. Match the order of
/// magnitude used in graph-embedding-util's `JointEmbedModel`.
const PARAM_INIT_STD: f64 = 0.05;

/// Which right-hand-side embedding table the bilinear scores against.
/// The shared feature side (ρ, z, Q) is reused across all axes; only
/// the pb/cell head varies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    /// Per-cell head. Bilinear `e_f · e_cell[c] + b_f + b_cell[c]`.
    Cell,
    /// Per-pb head at curriculum level `ℓ` (0 = coarsest).
    Pb(usize),
}

pub struct RnaModEmbedModel {
    pub n_modalities: usize,
    pub n_programs: usize,
    pub embedding_dim: usize,
    pub n_cells: usize,
    pub dev: Device,

    /// Owns every Var registered below. The optimizer pulls from here.
    pub varmap: VarMap,

    ////////////////////////////////////////
    // Feature params (shared across axes)
    ////////////////////////////////////////
    pub rho: Tensor,    // [G, H] — gene's cell-space direction
    pub z: Tensor,      // [G, K] — gene's K-program "mode" loadings
    pub q: Tensor,      // [K, M] — program × modality scalar response (sim's A_{m,k})
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

impl RnaModEmbedModel {
    pub fn new(
        n_genes: usize,
        n_modalities: usize,
        n_programs: usize,
        embedding_dim: usize,
        n_cells: usize,
        n_pbs_per_level: &[usize],
        dev: &Device,
    ) -> Result<Self> {
        let varmap = VarMap::new();
        let init_rand = Init::Randn {
            mean: 0.0,
            stdev: PARAM_INIT_STD,
        };
        let init_zero = Init::Const(0.0);

        let rho = varmap.get((n_genes, embedding_dim), "rho", init_rand, DType::F32, dev)?;
        let z = varmap.get((n_genes, n_programs), "z", init_rand, DType::F32, dev)?;
        // Q is the program×modality scalar response: Q[k, m] tells how
        // much program k drives modality m's gate. Shape (K, M) only —
        // no H axis, because the gate is a scalar that uniformly scales
        // ρ_g for the (g, m) row. See the "scalar gate" comment on
        // `embed_rows` below.
        let q = varmap.get((n_programs, n_modalities), "q", init_rand, DType::F32, dev)?;
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
            embedding_dim,
            n_cells,
            dev: dev.clone(),
            varmap,
            rho,
            z,
            q,
            b_agg,
            b_comp,
            e_cell,
            b_cell,
            e_pb_per_level,
            b_pb_per_level,
        })
    }

    fn idx_u32(&self, ids: &[u32]) -> Result<Tensor> {
        Ok(Tensor::from_slice(ids, ids.len(), &self.dev)?)
    }

    /// Build the `(agg_mask, not_agg_mask)` tensors for a slate in a single
    /// pass — saves the `Tensor::ones - agg` round-trip that `embed_rows`
    /// and `bias_rows` would otherwise each do separately on every call.
    pub fn agg_masks_f32(&self, is_agg: &[bool]) -> Result<(Tensor, Tensor)> {
        let n = is_agg.len();
        let mut agg = Vec::with_capacity(n);
        let mut not_agg = Vec::with_capacity(n);
        for &b in is_agg {
            if b {
                agg.push(1.0_f32);
                not_agg.push(0.0_f32);
            } else {
                agg.push(0.0_f32);
                not_agg.push(1.0_f32);
            }
        }
        let agg_t = Tensor::from_vec(agg, n, &self.dev)?;
        let not_agg_t = Tensor::from_vec(not_agg, n, &self.dev)?;
        Ok((agg_t, not_agg_t))
    }

    /// Feature-side embedding for a batch of rows. `gene_for_rho`
    /// indexes ρ; `gene_for_z` indexes z (gene-mode loading);
    /// `modality_for_q` selects the Q column. Positives and random
    /// in-class negatives coincide on all three; swap-gene-mode swaps
    /// `gene_for_z`. `is_agg` zeros the z·Q gate (AGG rows use ρ_g
    /// unmodified).
    ///
    /// **Scalar multiplicative gate.** Component rows scale ρ_g by a
    /// per-(g, m) scalar built from the (z, Q) factors:
    ///
    ///     AGG row  ({g}/AGG/total):       e_f = ρ_g
    ///     comp row ({g}/{m}/{detail}):    e_f = ρ_g · (1 + Σ_k z_{g,k} · Q_{k,m})
    ///
    /// The "+1" makes the gate the identity at z = 0, so a fresh model
    /// behaves like ρ_g on every row and modifier-row gradient flows
    /// through ρ as well as through (z, Q). The gate is a *scalar*: it
    /// uniformly scales every H-dim of ρ_g for that (g, m) row, so the
    /// modifier embedding always points in ρ_g's direction but with
    /// modality-specific magnitude. This is exactly the simulator's
    /// generative form: `r_{g,m} ∝ exp(Σ_k z_{g,k} · A_{m,k})` with
    /// scalar `A_{m,k}` — our `Q[k, m]` plays the role of `A`.
    /// Fused embed + bias for one slate. Builds the per-row `agg` /
    /// `not_agg` masks and the flat `b_comp` index tensor **once**, then
    /// reuses them for both the gate computation and the bias lookup.
    /// Saves ~10 tensor ops per sub-batch vs calling `embed_rows` then
    /// `bias_rows` separately (which each rebuild the masks and a
    /// `Tensor::ones` then subtract).
    #[allow(clippy::too_many_arguments)]
    pub fn embed_and_bias_rows(
        &self,
        gene_for_rho: &[u32],
        gene_for_z: &[u32],
        modality_for_q: &[u32],
        gene_for_bias: &[u32],
        modality_for_bias: &[u32],
        is_agg: &[bool],
    ) -> Result<(Tensor, Tensor)> {
        let b = gene_for_rho.len();
        debug_assert_eq!(gene_for_z.len(), b);
        debug_assert_eq!(modality_for_q.len(), b);
        debug_assert_eq!(gene_for_bias.len(), b);
        debug_assert_eq!(modality_for_bias.len(), b);
        debug_assert_eq!(is_agg.len(), b);

        let (agg, not_agg) = self.agg_masks_f32(is_agg)?;

        // ── embedding side ──
        let g_rho = self.idx_u32(gene_for_rho)?;
        let g_z = self.idx_u32(gene_for_z)?;
        let m_q = self.idx_u32(modality_for_q)?;

        let rho_b = self.rho.index_select(&g_rho, 0)?; // [B, H]
        let z_b = self.z.index_select(&g_z, 0)?; // [B, K]
        let q_b = self.q.index_select(&m_q, 1)?.transpose(0, 1)?; // [B, K]

        let zq = (z_b * q_b)?.sum(1)?; // [B]
        let zq_masked = (zq * &not_agg)?; // [B] — scalar zero for AGG
        let gate = (zq_masked + 1.0_f64)?; // [B]
        let gate_h = gate.unsqueeze(1)?; // [B, 1]
        let e = rho_b.broadcast_mul(&gate_h)?;

        // ── bias side (shares `agg` / `not_agg`) ──
        let m_cols = self.n_modalities as u32;
        let flat_idx_vec: Vec<u32> = gene_for_bias
            .iter()
            .zip(modality_for_bias.iter())
            .map(|(&g, &m)| g * m_cols + m)
            .collect();
        let g_bias_idx = self.idx_u32(gene_for_bias)?;
        let flat_idx = Tensor::from_vec(flat_idx_vec, b, &self.dev)?;

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
