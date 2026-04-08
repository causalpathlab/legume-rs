use candle_util::candle_core::{Result, Tensor};
use candle_util::candle_loss_functions::nb_log_likelihood;
use candle_util::candle_nn::{self, ops, VarBuilder};

pub struct DecoderArgs {
    pub n_features_atac: usize,
    pub n_features_rna: usize,
    pub n_topics: usize,
}

/// Chickpea decoder: ATAC (NB) + RNA (NB) likelihood module.
///
/// ATAC: learnable log_beta_atac[P, K].
/// RNA: gated mixture in log-space between linked dictionary (from SuSiE M × β)
/// and independent log_beta_rna[G, K].
///
///   log W[g,k] = α[g] · log W_linked[g,k] + (1-α[g]) · log β_rna[g,k]
///
/// where α[g] = sigmoid(gate_logit[g]) controls per-gene mix.
pub struct ChickpeaDecoder {
    pub log_beta_atac: Tensor, // [d_peaks_l, K]
    log_beta_rna: Tensor,      // [d_genes_l, K]
    gate_logit: Tensor,        // [d_genes_l, 1]
    bias_k: Tensor,            // [1, K]
    log_phi_atac: Tensor,      // [1, d_peaks_l]
    log_phi_rna: Tensor,       // [1, d_genes_l]
}

impl ChickpeaDecoder {
    pub fn new(args: DecoderArgs, vs: VarBuilder) -> Result<Self> {
        let DecoderArgs {
            n_features_atac,
            n_features_rna,
            n_topics,
        } = args;

        let beta_init = candle_nn::Init::Randn {
            mean: 0.0,
            stdev: 0.1,
        };
        let log_beta_atac =
            vs.get_with_hints((n_features_atac, n_topics), "log_beta_atac", beta_init)?;
        let log_beta_rna =
            vs.get_with_hints((n_features_rna, n_topics), "log_beta_rna", beta_init)?;
        let gate_logit = vs.get_with_hints(
            (n_features_rna, 1),
            "gate_logit",
            candle_nn::Init::Const(0.0),
        )?;

        let bias_k = vs.get_with_hints((1, n_topics), "bias_k", candle_nn::Init::Const(0.0))?;
        let log_phi_atac = vs.get_with_hints(
            (1, n_features_atac),
            "log_phi_atac",
            candle_nn::Init::Const(std::f64::consts::LN_2),
        )?;
        let log_phi_rna = vs.get_with_hints(
            (1, n_features_rna),
            "log_phi_rna",
            candle_nn::Init::Const(std::f64::consts::LN_2),
        )?;

        Ok(Self {
            log_beta_atac,
            log_beta_rna,
            gate_logit,
            bias_k,
            log_phi_atac,
            log_phi_rna,
        })
    }

    fn biased_z(&self, log_z_nk: &Tensor) -> Result<Tensor> {
        log_z_nk.broadcast_add(&self.bias_k)?.exp()
    }

    /// ATAC NB log-likelihood. Returns [N] per-sample.
    pub fn forward_atac(&self, log_z_nk: &Tensor, x_atac: &Tensor) -> Result<Tensor> {
        let beta = self.log_beta_atac.exp()?;
        let z_nk = self.biased_z(log_z_nk)?;
        let recon = z_nk.matmul(&beta.t()?)?;
        let lib = x_atac.sum(1)?.unsqueeze(1)?;
        let recon_sum = recon.sum(1)?.unsqueeze(1)?;
        let mu = recon
            .broadcast_mul(&lib)?
            .broadcast_div(&(&recon_sum + 1e-8)?)?;
        nb_log_likelihood(x_atac, &mu, &self.log_phi_atac)
    }

    /// RNA NB log-likelihood with gated log-space mixture.
    ///
    /// log W[g,k] = α[g] · log W_linked[g,k] + (1-α[g]) · log β_rna[g,k]
    /// where W_linked is provided externally (from SuSiE M × ATAC β).
    pub fn forward_rna(
        &self,
        log_z_nk: &Tensor,
        x_rna: &Tensor,
        log_w_linked: &Tensor,
    ) -> Result<Tensor> {
        let w = self.gated_log_rna_dictionary(log_w_linked)?.exp()?;
        let z_nk = self.biased_z(log_z_nk)?;
        let recon = z_nk.matmul(&w.t()?)?;
        let lib = x_rna.sum(1)?.unsqueeze(1)?;
        let recon_sum = recon.sum(1)?.unsqueeze(1)?;
        let mu = recon
            .broadcast_mul(&lib)?
            .broadcast_div(&(&recon_sum + 1e-8)?)?;
        nb_log_likelihood(x_rna, &mu, &self.log_phi_rna)
    }

    /// Per-gene gate values α[G] = sigmoid(gate_logit). For diagnostics.
    pub fn gate_alpha(&self) -> Result<Tensor> {
        ops::sigmoid(&self.gate_logit)?.squeeze(1)
    }

    /// Gated RNA dictionary in log-space:
    ///   log W[g,k] = α[g] · log_w_linked[g,k] + (1-α[g]) · log_beta_rna[g,k]
    pub fn gated_log_rna_dictionary(&self, log_w_linked: &Tensor) -> Result<Tensor> {
        let alpha = ops::sigmoid(&self.gate_logit)?;
        let one_minus_alpha = (1.0 - &alpha)?;
        alpha.broadcast_mul(log_w_linked)? + one_minus_alpha.broadcast_mul(&self.log_beta_rna)?
    }
}
