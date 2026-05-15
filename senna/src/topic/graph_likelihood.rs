//! Generative feature-graph likelihood (Ball-Karrer-Newman 2011, degree-
//! corrected mixed-membership Poisson SBM) for the ETM-factorized
//! `senna indexed-topic` decoder.
//!
//! ## Decoupled embedding
//!
//! Historically the BKN graph likelihood operated on the **same** ρ that
//! the encoder/decoder shared. That coupling is degenerate: the bilinear
//! `λ_uv = d̂_u · d̂_v · ⟨ρ̃_u, ρ̃_v⟩` is maximized by colinear ρ̃ rows, so
//! gradient descent collapses ρ toward rank-1 (especially on dense networks
//! without clean community structure — PPI, generic co-expression graphs).
//! Because the encoder pools via ρ, that collapse propagates straight into
//! θ even when β = softmax(α·ρᵀ) still carries multi-program structure on
//! α's free dimensions.
//!
//! We fix this by giving the graph likelihood its **own** embedding
//! `ρ_graph [D, H]`, fully separate from the encoder's `ρ_enc`. No
//! cross-talk in this revision — gradients from the graph NLL land on
//! `ρ_graph` only; gradients from the data ELBO land on `ρ_enc` only:
//!
//! ```text
//! L_graph  = λ_G · NLL_BKN(ρ_graph)                  # graph side, on ρ_graph
//! L_topic  = E[KL − log p(x|θ,β)] + anchor           # data side, on ρ_enc
//! ```
//!
//! `ρ_graph` trains as a side embedding, available for downstream use or
//! for future coupling back to the encoder (e.g. via a Gaussian tether
//! `λ_T·mean((ρ_enc−ρ_graph)²)`).
//!
//! ## BKN rate model (operating on ρ_graph)
//!
//! ```text
//! ρ̃ = softplus(ρ_graph)                  # non-negative module strengths [D, H]
//! λ_uv  =  d̂_u · d̂_v · ⟨ρ̃_u, ρ̃_v⟩       # bilinear Poisson rate, BKN-DCSBM
//! A_uv  ~  Poisson(λ_uv)
//! ```
//!
//! ## Closed-form partition
//!
//! Naive `Σ_{u<v} λ_uv` is O(D²). The bilinear form lets it factorize:
//!
//! ```text
//! 2 · Σ_{u<v} λ_uv  =  ‖S‖²  −  ‖d̂ ⊙ ρ̃‖²_F
//! where S_h  =  Σ_u d̂_u · ρ̃_{u,h}
//! ```
//!
//! → O(D·H) per evaluation; no negative sampling needed.

use crate::embed_common::*;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use matrix_util::pair_graph::FeaturePairGraph;

/// Precomputed device-side tensors for the BKN graph-likelihood term,
/// plus the decoupled `ρ_graph` Var the term operates on.
pub(crate) struct PoissonGraphConfig {
    /// `[|E|]` u32 row indices.
    pub edges_u: Tensor,
    /// `[|E|]` u32 row indices (paired with `edges_u`).
    pub edges_v: Tensor,
    /// `[D, 1]` per-node empirical degree (DCSBM correction). Stored on
    /// device, broadcast-multiplied with ρ̃.
    pub degrees: Tensor,
    /// λ_G — relative weight of the graph Poisson NLL.
    pub loss_weight: f32,
    /// Decoupled graph-side embedding `[D, H]`, registered as a Var in the
    /// shared VarMap under `feature.embeddings.graph`. Graph NLL gradients
    /// land here only; the encoder's ρ is not touched.
    pub rho_graph: Tensor,
}

impl PoissonGraphConfig {
    /// Build from a CPU-resident `FeaturePairGraph`. Computes empirical
    /// degree from the edge list (each undirected edge contributes 1 to
    /// each endpoint). Edges are uploaded to `dev` once and reused.
    ///
    /// Registers `ρ_graph` as a `[D, H]` Var via the supplied `VarBuilder`,
    /// initialized to zeros so softplus(ρ_graph) starts at log(2) and the
    /// graph NLL has a well-conditioned starting point (no Kaiming-noise
    /// asymmetry on a parameter that's about to be shaped by edges alone).
    pub fn build(
        graph: &FeaturePairGraph,
        n_features: usize,
        embedding_dim: usize,
        loss_weight: f32,
        vb: &VarBuilder,
        dev: &Device,
    ) -> anyhow::Result<Self> {
        let n_edges = graph.feature_edges.len();
        let mut u_vec: Vec<u32> = Vec::with_capacity(n_edges);
        let mut v_vec: Vec<u32> = Vec::with_capacity(n_edges);
        for &(u, v) in &graph.feature_edges {
            u_vec.push(u as u32);
            v_vec.push(v as u32);
        }
        // Normalize degrees by sqrt(2|E|) so that Σ_u d̂_u ≈ 1 (DCSBM
        // identifiability anchor — see Karrer & Newman 2011 eq. 4).
        let two_m = 2.0 * n_edges as f32;
        let scale = if two_m > 0.0 { two_m.sqrt() } else { 1.0 };
        let mut deg: Vec<f32> = graph
            .feature_degrees()
            .iter()
            .map(|&d| d as f32 / scale)
            .collect();
        deg.resize(n_features, 0.0);
        let degrees_mat: Mat = Mat::from_iterator(n_features, 1, deg);

        let edges_u = Tensor::from_vec(u_vec, n_edges, dev)?;
        let edges_v = Tensor::from_vec(v_vec, n_edges, dev)?;
        let degrees = degrees_mat
            .to_tensor(dev)?
            .to_dtype(candle_core::DType::F32)?;

        let rho_graph = vb.get_with_hints(
            (n_features, embedding_dim),
            "feature.embeddings.graph",
            candle_nn::Init::Const(0.0),
        )?;

        Ok(Self {
            edges_u,
            edges_v,
            degrees,
            loss_weight,
            rho_graph,
        })
    }
}

/// Evaluate the (weighted, sign-flipped) Poisson graph NLL at the current
/// `ρ_graph`. Returns `λ_G · NLL` ready to feed to AdamW. Gradients land
/// on `cfg.rho_graph` only — the encoder's ρ is **not** touched.
pub(crate) fn graph_loss(cfg: &PoissonGraphConfig) -> anyhow::Result<Tensor> {
    let eps = 1e-8f64;
    let rho = &cfg.rho_graph;

    // ρ̃ = softplus(ρ_graph) ≥ 0, with the numerically-stable formulation
    // `max(x, 0) + log(1 + exp(−|x|))` so values past ~20 don't overflow.
    let zero_or_x = rho.relu()?;
    let abs_x = rho.abs()?;
    let smooth = ((abs_x.neg()?.exp()? + 1.0)?).log()?;
    let rho_pos = (zero_or_x + smooth)?; // [D, H]

    // Degree-weighted version, reused for both S and Q.
    let d_rho = rho_pos.broadcast_mul(&cfg.degrees)?; // [D, H]

    // Positive edges: log λ_uv = log(d̂_u·d̂_v · ρ̃_u·ρ̃_v); the log d̂
    // terms are constants w.r.t. ρ and drop from the optimization loss,
    // leaving the bilinear-product log only.
    let rho_u = rho_pos.index_select(&cfg.edges_u, 0)?; // [|E|, H]
    let rho_v = rho_pos.index_select(&cfg.edges_v, 0)?; // [|E|, H]
    let dot_e = (rho_u * rho_v)?.sum(1)?; // [|E|]
    let neg_log_lik = (dot_e + eps)?.log()?.sum_all()?.neg()?;

    // Closed-form non-edge partition (over unordered u<v):
    //   2 · Σ_{u<v} λ_uv = ‖S‖² − ‖d̂ ⊙ ρ̃‖²_F
    //   where S_h = Σ_u d̂_u · ρ̃_{u,h}.
    let s_h = d_rho.sum(0)?; // [H]
    let s_norm_sq = (&s_h * &s_h)?.sum_all()?;
    let diag = (&d_rho * &d_rho)?.sum_all()?;
    let partition = ((s_norm_sq - diag)? * 0.5)?;

    // Poisson NLL: -Σ_E log λ_uv  +  Σ_{u<v} λ_uv (constants dropped).
    let nll = (neg_log_lik + partition)?;
    Ok(nll.affine(f64::from(cfg.loss_weight), 0.0)?)
}
