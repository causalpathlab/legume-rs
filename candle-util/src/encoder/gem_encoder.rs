//! Splice-aware masked encoder for `faba gem-encoder`.
//!
//! Reads a cell's top-K **genes** with BOTH tracks attached — nascent
//! (unspliced) and mature (spliced) counts for the same gene — and pools each
//! track over the cell's own top-K context, in the `H`-dimensional embedding
//! space and never over the full `D`-dimensional gene space.
//!
//! ```text
//! a^u, a^s = anscombe_lite(observed, residual, gene_mean)       [N,K] each
//! t^u = a^u ⊙ ρ_g                                                 [N,K,H]  nascent (base)
//! t^s = a^s ⊙ (ρ_g + δ_g)                                         [N,K,H]  mature
//!
//! p^u = Σ_k visible^u ⊙ t^u                               [N,H]
//! p^s = Σ_k visible^s ⊙ t^s                               [N,H]
//! pool = [p^s ‖ p^u]                                      [N,2H]
//!
//! z = clamp(z_mean(BN(FC(pool))), ±8)                     [N,K_latent]
//! ```
//!
//! Zeroing the hidden positions BEFORE the sum is what makes the mask exact:
//! there is no softmax to renormalize, so a hidden gene contributes literally
//! nothing and the visible ones keep their magnitudes. Both tracks reach the
//! trunk as separate blocks, so the FC learns the cross-track relation and
//! **either track alone still varies per cell** — which is what the masked
//! objective requires, since half its modes hide one track outright.
//!
//! The parameterization runs `u + δ → s`: nascent is the base and mature is
//! `ρ + δ`, so only the MATURE likelihood can move `δ` (see
//! [`crate::decoder::gem_etm`] for why that placement matters).
//!
//! # Why there is no attention here
//!
//! An earlier version pooled through `M` learned latent slots with a cross-track
//! stage `X = Attn(Q=S_slots, K/V=U_slots) + U_slots`. It was **removed**, not
//! disabled, for a structural reason and a measured one.
//!
//! Structural: `s_slots` entered only as the attention QUERY, never as a value,
//! so the mature track's *content* never reached the output — only its say in
//! how nascent slots were weighted. The mature track is the better-measured of
//! the two, so that discards the stronger signal by construction. And because
//! both terms were functions of `u_slots`, a minibatch with nascent hidden
//! (`MatureToNascent`, 30 % of the default schedule) drove `pool` to exactly
//! zero and handed **every cell in the batch one identical latent** — the
//! decoder then spent that share of each epoch fitting a cell-independent θ.
//!
//! Measured: on the one clean single-variable comparison, between-cell variance
//! share was 0.039 with attention against 0.137 with this sum — 3.5× worse, a
//! near-uniform θ and a UMAP of fragmented speckle. Every result that survived
//! scrutiny in this project used sum pooling.
//!
//! Repairing it (adding `+ s_slots`) would have made it a sum with extra steps
//! plus `M·H` slot parameters, so the flag and the machinery went instead.

use crate::nn::batch_norm;
use crate::nn::layers::*;
use crate::nn::soft_clamp::{soft_clamp, MASKED_LOGIT_CLAMP as LOGIT_CLAMP};
use crate::value_transform::anscombe_lite;
use candle_core::{Result, Tensor};
use candle_nn::{Linear, ModuleT, VarBuilder, VarMap};

/// Borrowed per-minibatch encoder input: the cell's top-K genes with both
/// tracks, plus each track's visibility mask.
pub struct GemEncoderInput<'a> {
    /// `[N, K]` u32 — per-cell top-K **gene** ids.
    pub gene_indices: &'a Tensor,
    /// `[N, K]` f32 — nascent (unspliced) counts at `gene_indices`.
    pub nascent_observed: &'a Tensor,
    /// `[N, K]` f32 — mature (spliced) counts at `gene_indices`.
    pub mature_observed: &'a Tensor,
    /// `[N, K]` f32 — batch RESIDUAL (`μ_residual`) at the genes' NASCENT rows.
    /// Per track: the residual is fitted per row, and applying one track's to
    /// both would leave an `r^u/r^s` remainder that is itself a splice-ratio
    /// distortion — i.e. it would contaminate `δ`.
    pub nascent_residual: Option<&'a Tensor>,
    /// `[N, K]` f32 — batch RESIDUAL (`μ_residual`) at the genes' MATURE rows.
    pub mature_residual: Option<&'a Tensor>,
    /// `[N, K]` f32 — per-gene mean nascent rate at `gene_indices`.
    pub nascent_mean: Option<&'a Tensor>,
    /// `[N, K]` f32 — per-gene mean mature rate at `gene_indices`.
    pub mature_mean: Option<&'a Tensor>,
    /// `[N, K]` 1 = this gene's nascent count is visible to the encoder.
    pub nascent_visible: &'a Tensor,
    /// `[N, K]` 1 = this gene's mature count is visible to the encoder.
    pub mature_visible: &'a Tensor,
}

pub struct GemIndexedEncoderArgs<'a> {
    /// Number of genes `G` (the gene-keyed axis, NOT the 2G row axis).
    pub n_genes: usize,
    /// Latent width `K_latent` — the number of factors.
    pub n_latent: usize,
    /// Embedding width `H`.
    pub embedding_dim: usize,
    /// Inject reparameterized Gaussian noise on the latent logits during
    /// training — a VAE's sampling WITHOUT its KL. Off by default.
    pub latent_noise: bool,
    /// FC stack widths after pooling; last entry is the trunk output width.
    pub layers: &'a [usize],
}

/// Splice-aware masked encoder. Owns the gene-keyed `ρ` and splice-ratio `δ`
/// that the decoder ties to.
pub struct GemIndexedEncoder {
    embedding_dim: usize,
    /// `ρ [G, H]` NASCENT gene embeddings (learnable, the base).
    feature_embeddings: Tensor,
    /// `δ [G, H]` splice-ratio offset; the mature embedding is `ρ + δ`.
    /// Zero-init, so training starts from "mature composition equals nascent
    /// composition" — the null the ridge shrinks toward — and moves off it only
    /// as the `u→s` likelihood demands.
    delta_embeddings: Tensor,
    fc: StackLayers<Linear>,
    bn_z: batch_norm::BatchNorm,
    z_mean: Linear,
    /// Log-variance head for [`GemIndexedEncoderArgs::latent_noise`]. `None`
    /// means no sampling at all (the latent is a pure point estimate).
    z_lnvar: Option<Linear>,
}

impl GemIndexedEncoder {
    pub fn new(args: GemIndexedEncoderArgs, varmap: &VarMap, vb: VarBuilder) -> Result<Self> {
        debug_assert!(!args.layers.is_empty());
        let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
        let h = args.embedding_dim;

        let feature_embeddings =
            vb.get_with_hints((args.n_genes, h), "feature.embeddings", init_ws)?;
        // δ starts at exactly zero: β^s ≡ β^u at init.
        let delta_embeddings =
            vb.get_with_hints((args.n_genes, h), "delta.embeddings", candle_nn::init::ZERO)?;
        let fc_dims = args.layers[..args.layers.len() - 1].to_vec();
        let out_dim = *args.layers.last().unwrap();
        let fc = stack_relu_linear(
            2 * h, // [p^s ‖ p^u]
            out_dim,
            &fc_dims,
            vb.pp("nn.enc.fc"),
        )?;

        let bn_z = batch_norm::batch_norm(
            out_dim,
            batch_norm::BatchNormConfig {
                eps: 1e-4,
                remove_mean: true,
                affine: true,
                momentum: 0.1,
            },
            varmap,
            vb.pp("nn.enc.bn_z"),
        )?;

        let z_mean = candle_nn::linear(out_dim, args.n_latent, vb.pp("nn.enc.z.mean"))?;

        let z_lnvar = if args.latent_noise {
            Some(candle_nn::linear(
                out_dim,
                args.n_latent,
                vb.pp("nn.enc.z.lnvar"),
            )?)
        } else {
            None
        };

        Ok(Self {
            embedding_dim: h,
            feature_embeddings,
            delta_embeddings,
            fc,
            bn_z,
            z_mean,
            z_lnvar,
        })
    }

    /// The gene-keyed nascent embedding `ρ [G, H]` — tie the decoder to this.
    #[must_use]
    pub fn feature_embeddings(&self) -> &Tensor {
        &self.feature_embeddings
    }
    /// The splice-ratio offset `δ [G, H]` — tie the decoder to this.
    #[must_use]
    pub fn delta_embeddings(&self) -> &Tensor {
        &self.delta_embeddings
    }

    /// Value-weighted tokens for one track: `a ⊙ ρ^τ`, `[N, K, H]`.
    ///
    /// Visibility is applied by the CALLER, by zeroing the value before the
    /// sum — see [`Self::pool`]. These tokens carry the raw value at every
    /// position.
    fn tokens(
        &self,
        gene_indices: &Tensor,
        values: &Tensor,
        residual: Option<&Tensor>,
        values_mean: Option<&Tensor>,
        add_delta: bool,
    ) -> Result<Tensor> {
        let (n, k) = gene_indices.dims2()?;
        let h = self.embedding_dim;
        let flat = gene_indices.flatten_all()?;

        let base = self.feature_embeddings.index_select(&flat, 0)?;
        let emb = if add_delta {
            (base + self.delta_embeddings.index_select(&flat, 0)?)?
        } else {
            base
        }
        .reshape((n, k, h))?;

        let a_nk = anscombe_lite(values, residual, values_mean)?; // [N, K]
        emb.broadcast_mul(&a_nk.unsqueeze(2)?)
    }

    /// Collapse the cell's gene tokens into the trunk input, `[N, 2H]`.
    ///
    /// Per-track masked value-weighted sum, concatenated. Because the tracks
    /// are pooled INDEPENDENTLY, hiding one entirely leaves the other's half of
    /// the vector intact — the property the masked objective depends on.
    fn pool(&self, input: &GemEncoderInput<'_>) -> Result<Tensor> {
        let t_u = self.tokens(
            input.gene_indices,
            input.nascent_observed,
            input.nascent_residual,
            input.nascent_mean,
            false, // nascent = base
        )?;
        let t_s = self.tokens(
            input.gene_indices,
            input.mature_observed,
            input.mature_residual,
            input.mature_mean,
            true, // mature = base + delta
        )?;

        // Zeroing the hidden positions BEFORE the sum is what makes the mask
        // exact here: there is no softmax to renormalize, so a hidden gene
        // contributes literally nothing and the visible ones keep their
        // magnitudes.
        let p_u = t_u
            .broadcast_mul(&input.nascent_visible.unsqueeze(2)?)?
            .sum(1)?; // [N, H]
        let p_s = t_s
            .broadcast_mul(&input.mature_visible.unsqueeze(2)?)?
            .sum(1)?; // [N, H]
                      // Mature first: it is the better-measured track, so the trunk's
                      // leading H inputs are the ones carrying the stronger signal.
        Tensor::cat(&[&p_s, &p_u], 1) // [N, 2H]
    }

    /// Shared trunk: pool → FC → BN → `[N, L]`.
    fn hidden(&self, input: &GemEncoderInput<'_>, train: bool) -> Result<Tensor> {
        let pooled = self.pool(input)?;
        let fc_nl = self.fc.forward_t(&pooled, train)?;
        self.bn_z.forward_t(&fc_nl, train)
    }

    /// Clamped pre-simplex logits `z [N, K_latent]` — the unconstrained
    /// pre-activation `θ = softmax(z)` maps from, and the space the diagnostic
    /// velocity increment is differenced in (see
    /// [`crate::vae::masked_gem::theta_log_simplex`]). Softmax logits are
    /// identified only up to an additive constant per row, so a difference of
    /// two of these is gauge-dependent — which is one reason the headline
    /// velocity is the dictionary operator on `θ`, not this difference.
    pub fn logits(&self, input: &GemEncoderInput<'_>, train: bool) -> Result<Tensor> {
        let bn_nl = self.hidden(input, train)?;
        soft_clamp(&self.z_mean.forward_t(&bn_nl, train)?, LOGIT_CLAMP)
    }

    /// Forward to the raw latent logits `z [N, K_latent]`.
    ///
    /// Deterministic by default. A full Gaussian/KL head existed and was
    /// removed: at `kl = 1.0` the latent collapsed to effective rank 1.03 (a
    /// literal one-dimensional curve), at `kl = 0.1` it reached 2.14 against
    /// stick-breaking's 4.9–6.7, and the splice-ratio check degraded
    /// monotonically with KL weight (r = 0.53 → 0.44 → 0.37). The masked
    /// objective is already the regularizer; the KL only removed the per-cell
    /// spread the model exists to measure.
    ///
    /// [`GemIndexedEncoderArgs::latent_noise`] keeps the reparameterized SAMPLE
    /// without the KL — a noisy autoencoder rather than a variational one. That
    /// isolates the two halves of what the old head did: this is the test of
    /// whether the sampling alone regularizes usefully once the prior-pulling is
    /// gone. Sampling is training-only; inference stays deterministic, so the
    /// three velocity passes remain comparable.
    pub fn forward_latent(&self, input: &GemEncoderInput<'_>, train: bool) -> Result<Tensor> {
        let bn_nl = self.hidden(input, train)?;
        let z_mean = soft_clamp(&self.z_mean.forward_t(&bn_nl, train)?, LOGIT_CLAMP)?;
        match (&self.z_lnvar, train) {
            (Some(lv), true) => {
                let lnvar = soft_clamp(&lv.forward_t(&bn_nl, train)?, LOGIT_CLAMP)?;
                let sigma = lnvar.affine(0.5, 0.0)?.exp()?;
                let eps = Tensor::randn(0f32, 1f32, z_mean.shape(), z_mean.device())?;
                z_mean + sigma.mul(&eps)?
            }
            _ => Ok(z_mean),
        }
    }
}

#[cfg(test)]
#[path = "gem_encoder_tests.rs"]
mod tests;
