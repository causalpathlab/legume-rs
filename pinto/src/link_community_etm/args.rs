//! CLI arguments for `pinto lc-etm` — link community via embedded topic model.

use clap::{Parser, ValueEnum};

/// Per-gene likelihood for the masked imputation loss (`--train-mode masked`).
/// CLI mirror of [`candle_util::vae::masked_topic::MaskedLikelihood`] (that
/// crate can't derive clap's `ValueEnum` without a clap dependency).
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
#[clap(rename_all = "lowercase")]
pub enum MaskedLikelihood {
    /// Negative binomial — over-dispersed per-gene counts. Default.
    Nb,
    /// Multinomial / categorical — depth-invariant; the same likelihood the
    /// ELBO path uses, so `elbo` vs `masked --masked-likelihood multinomial`
    /// isolates the training objective from the likelihood.
    Multinomial,
}

impl MaskedLikelihood {
    /// Map to the candle-util training enum.
    pub fn to_lib(self) -> candle_util::vae::masked_topic::MaskedLikelihood {
        use candle_util::vae::masked_topic::MaskedLikelihood as L;
        match self {
            MaskedLikelihood::Nb => L::Nb,
            MaskedLikelihood::Multinomial => L::Multinomial,
        }
    }
}

/// Training objective for the edge-document ETM.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
#[clap(rename_all = "lowercase")]
pub enum TrainMode {
    /// Amortized VAE / ELBO: reconstruct the full pooled document `y_e`
    /// via the topic mixture, with a KL term on the per-edge posterior.
    /// Gives a calibrated posterior (meaningful entropy) but can
    /// posterior-collapse (topics go unused).
    Elbo,
    /// Masked NB imputation (BERT/MLM-style): hold out a fraction of each
    /// edge's genes, encode the visible remainder, predict the held-out
    /// counts. No KL → structurally collapse-proof; the objective rewards
    /// *predictive* communities. Point-estimate π (no posterior).
    Masked,
}

#[derive(Parser, Debug, Clone)]
pub struct SrtLinkCommunityEtmArgs {
    #[command(flatten)]
    pub common: crate::util::input::SrtInputArgs,

    #[arg(
        long,
        value_enum,
        default_value_t = TrainMode::Masked,
        help = "Training objective: masked (BERT-like imputation, default) or elbo (generative VAE)",
        long_help = "Training objective for the edge-document ETM.\n\
                       masked (default) — hold out a fraction of each edge's genes\n\
                       and predict them from the rest (no KL). Structurally\n\
                       collapse-proof; strongly preferred on real data (ELBO\n\
                       posterior-collapses — one community swallows most edges).\n\
                       elbo — generative VAE with a KL'd per-edge posterior; use\n\
                       only when you need calibrated posterior entropy."
    )]
    pub train_mode: TrainMode,

    #[arg(
        long,
        value_enum,
        default_value_t = MaskedLikelihood::Nb,
        help = "Masked mode: per-gene likelihood (nb = over-dispersed counts; multinomial = compositional)"
    )]
    pub masked_likelihood: MaskedLikelihood,

    #[arg(
        long,
        default_value_t = 0.5,
        help = "Masked mode: fraction of each edge's genes held out as prediction targets",
        long_help = "Only used with --train-mode masked. Each minibatch holds out\n\
                       this fraction of an edge's observed (nonzero) genes as NB\n\
                       imputation targets; the encoder sees only the visible\n\
                       remainder. Typical 0.3–0.6. Ignored under --train-mode elbo."
    )]
    pub mask_fraction: f64,

    #[arg(
        long,
        default_value_t = 50,
        help = "Number of spatial link communities (topics K)",
        long_help = "Number of link communities / topics (K). Each cell-cell edge\n\
                       is soft-assigned to a categorical distribution over K\n\
                       communities via the encoder; β = softmax(α · ρᵀ) gives the\n\
                       per-community gene rates."
    )]
    pub n_communities: usize,

    #[arg(
        long,
        default_value_t = 256,
        help = "Top-K genes per edge for encoder context window",
        long_help = "Number of top-K genes (by count) selected per edge for the\n\
                       indexed encoder's input. The encoder looks up ρ at these K\n\
                       genes and aggregates them into the per-edge hidden state.\n\
                       Larger context = more capacity, slower training."
    )]
    pub context_size: usize,

    #[arg(
        long,
        default_value_t = 64,
        help = "Embedding dimension H (ρ, α, encoder hidden)",
        long_help = "Embedding dimension H. ρ ∈ ℝ^{G×H} (gene embedding, shared\n\
                       between encoder and decoder), α ∈ ℝ^{K×H} (community\n\
                       embedding), encoder hidden state ∈ ℝ^H per edge."
    )]
    pub embedding_dim: usize,

    #[arg(long, default_value_t = 100, help = "Number of training epochs")]
    pub num_epochs: usize,

    #[arg(
        long,
        default_value_t = 4096,
        help = "Edge minibatch size",
        long_help = "Number of edges per minibatch. Smaller = more noise per step\n\
                       but lower memory; larger = more stable gradients but more\n\
                       memory per step. With E ≈ 2M edges, 4096 gives ~500\n\
                       steps/epoch at the finest V-cycle level."
    )]
    pub batch_edges: usize,

    #[arg(long, default_value_t = 1e-3, help = "Adam learning rate")]
    pub lr: f32,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "Topic smoothing strength α ∈ [0, 1) (0 disables)",
        long_help = "Per-edge topic smoothing strength: mix the encoder output\n\
                       with the uniform distribution in log-space:\n\
                         log_z ← log((1-α)·exp(log_z) + α/K).\n\
                       Stabilises early training; 0 disables."
    )]
    pub topic_smoothing: f64,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Global L2 gradient norm clip per minibatch (0 = off)"
    )]
    pub grad_clip: f32,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "L2 penalty on ρ (gene embedding); 0 disables"
    )]
    pub feature_embedding_l2: f32,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "AdamW decoupled weight decay (post-step)"
    )]
    pub weight_decay: f32,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Min total count to include a gene in shortlist weighting",
        long_help = "Per-gene shortlist weights are min-clamped to this floor\n\
                       before scoring top-K candidates. Effectively removes genes\n\
                       below this total count from being selected into the\n\
                       encoder/decoder context window."
    )]
    pub min_gene_count: f32,
}
