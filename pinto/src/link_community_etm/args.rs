//! CLI arguments for `pinto lc-etm` — link community via embedded topic model.

use crate::util::device::ComputeDevice;
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

    #[arg(long, default_value_t = ComputeDevice::Cpu, value_enum, help = "Compute device")]
    pub device: ComputeDevice,

    #[arg(long, default_value_t = 0, help = "Device ordinal (for cuda/metal)")]
    pub device_no: usize,

    #[arg(
        long,
        value_enum,
        default_value_t = TrainMode::Masked,
        help = "Training objective:\n\
                masked (BERT-like imputation, default) or elbo (generative VAE)",
        long_help = "Training objective for the edge-document ETM.\n\
                     \n\
                     masked is the default, and carries no KL term.\n\
                     It holds out a fraction of each edge's genes,\n\
                     then predicts them from the rest. That is structurally collapse-proof.\n\
                     Strongly preferred on real data.\n\
                     \n\
                     elbo is a generative VAE with a KL'd per-edge posterior.\n\
                     On real data it posterior-collapses.\n\
                     One community then swallows most edges.\n\
                     Use it only when you need calibrated posterior entropy."
    )]
    pub train_mode: TrainMode,

    #[arg(
        long,
        value_enum,
        default_value_t = MaskedLikelihood::Nb,
        help = "Masked mode:\n\
                per-gene likelihood (nb = over-dispersed counts; multinomial = compositional)",
        hide = true,
    )]
    pub masked_likelihood: MaskedLikelihood,

    #[arg(
        long,
        default_value_t = 0.5,
        help = "Masked mode: fraction of each edge's genes held out as prediction targets",
        long_help = "Used only with --train-mode masked.\n\
                     Each minibatch holds out this share of an edge's genes.\n\
                     They become the NB imputation targets.\n\
                     The encoder sees only the visible remainder.\n\
                     Typical values run 0.3 to 0.6.\n\
                     --train-mode elbo ignores this flag.",
        hide = true
    )]
    pub mask_fraction: f64,

    #[arg(
        long,
        default_value_t = 50,
        help = "Number of spatial link communities (topics K)",
        long_help = "Number of link communities, or topics (K).\n\
                     The encoder soft-assigns every cell-cell edge.\n\
                     Each gets a categorical over the K communities.\n\
                     β = softmax(α · ρᵀ) then gives the per-community gene rates."
    )]
    pub n_communities: usize,

    #[arg(
        long,
        default_value_t = 256,
        help = "Top-K genes per edge for encoder context window",
        long_help = "Top-K genes, by count, taken per edge as encoder input.\n\
                     The encoder looks up ρ at those genes.\n\
                     It aggregates them into the per-edge hidden state.\n\
                     A larger context buys capacity and costs training time.",
        hide = true
    )]
    pub context_size: usize,

    #[arg(
        long,
        default_value_t = 64,
        help = "Embedding dimension H (ρ, α, encoder hidden)",
        long_help = "Embedding dimension H. ρ ∈ ℝ^{G×H} is the gene embedding.\n\
                     Encoder and decoder share it. α ∈ ℝ^{K×H} is the community embedding.\n\
                     The encoder hidden state is ℝ^H per edge."
    )]
    pub embedding_dim: usize,

    #[arg(long, default_value_t = 100, help = "Number of training epochs")]
    pub num_epochs: usize,

    #[arg(
        long,
        default_value_t = 4096,
        help = "Edge minibatch size",
        long_help = "Edges per minibatch. Smaller batches add noise per step and save memory.\n\
                     Larger ones steady the gradient and cost memory. At E ≈ 2M edges,\n\
                     4096 gives ~500 steps per epoch, measured at the finest V-cycle level.",
        hide = true
    )]
    pub batch_edges: usize,

    #[arg(long, default_value_t = 1e-3, help = "Adam learning rate")]
    pub lr: f32,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "Topic smoothing strength α ∈ [0, 1) (0 disables)",
        long_help = "Per-edge topic smoothing strength.\n\
                     It mixes encoder output with the uniform, in log space:\n\
                     log_z ← log((1-α)·exp(log_z) + α/K). This stabilises early training;\n\
                     0 disables it.",
        hide = true
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
        help = "L2 penalty on ρ (gene embedding); 0 disables",
        hide = true
    )]
    pub feature_embedding_l2: f32,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "AdamW decoupled weight decay (post-step)",
        hide = true
    )]
    pub weight_decay: f32,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Min total count to include a gene in shortlist weighting",
        long_help = "Floor that per-gene shortlist weights are clamped to.\n\
                     Clamping happens before top-K candidates are scored.\n\
                     Genes under this total count never enter it."
    )]
    pub min_gene_count: f32,
}
