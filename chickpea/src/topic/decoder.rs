use candle_util::candle_core::Result;
use candle_util::candle_core::Tensor;
use candle_util::candle_loss_functions::nb_log_likelihood;
use candle_util::candle_nn::{self, VarBuilder};

pub struct DecoderArgs {
    pub n_features_atac: usize,
    pub n_features_rna: usize,
    pub n_topics: usize,
}

/// Chickpea decoder: ATAC (NB) + RNA (NB) likelihood module.
///
/// ATAC has its own learnable dictionary log_beta_atac[P, K].
/// RNA dictionary W[G, K] is provided externally (from SuSiE linkage M × β).
pub struct ChickpeaDecoder {
    pub log_beta_atac: Tensor, // [d_peaks_l, K]
    bias_k: Tensor,            // [1, K]
    intercept_g: Tensor,       // [1, d_genes_l]
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

        let bias_k = vs.get_with_hints((1, n_topics), "bias_k", candle_nn::Init::Const(0.0))?;
        let intercept_g = vs.get_with_hints(
            (1, n_features_rna),
            "intercept_g",
            candle_nn::Init::Const(0.0),
        )?;
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
            bias_k,
            intercept_g,
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

    /// RNA NB log-likelihood with externally provided dictionary W[G,K].
    pub fn forward_rna(&self, log_z_nk: &Tensor, x_rna: &Tensor, w_gk: &Tensor) -> Result<Tensor> {
        let z_nk = self.biased_z(log_z_nk)?;
        let recon = z_nk.matmul(&w_gk.t()?)?;
        let recon = recon.broadcast_add(&self.intercept_g.exp()?)?;
        let lib = x_rna.sum(1)?.unsqueeze(1)?;
        let recon_sum = recon.sum(1)?.unsqueeze(1)?;
        let mu = recon
            .broadcast_mul(&lib)?
            .broadcast_div(&(&recon_sum + 1e-8)?)?;
        nb_log_likelihood(x_rna, &mu, &self.log_phi_rna)
    }
}
