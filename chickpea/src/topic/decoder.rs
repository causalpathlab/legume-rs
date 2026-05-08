use candle_util::candle_core::{Device, Result, Tensor};
use candle_util::candle_nn::{self, ops, VarBuilder};

pub struct DecoderArgs {
    pub n_features_atac: usize,
    pub n_features_rna: usize,
    pub n_topics: usize,
}

/// ATAC + RNA Fisher-weighted multinomial decoder.
///
/// ATAC: learnable `log_beta_atac[P, K]`.
/// RNA: log-space gated mixture between linked dictionary (from SuSiE M × β)
/// and independent `log_beta_rna[G, K]`:
///
///   log W[g,k] = α_g · log W_linked[g,k] + (1-α_g) · log β_rna[g,k]
///
/// where `α_g = sigmoid(gate_logit)`. NB-Fisher feature weights, when
/// attached, downweight high-mean / high-dispersion housekeeping features.
pub struct ChickpeaDecoder {
    pub log_beta_atac: Tensor, // [d_peaks_l, K]
    log_beta_rna: Tensor,      // [d_genes_l, K]
    gate_logit: Tensor,        // [d_genes_l, 1]
    bias_k: Tensor,            // [1, K]
    feature_weights_atac: Option<Tensor>, // [1, d_peaks_l]
    feature_weights_rna: Option<Tensor>,  // [1, d_genes_l]
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

        Ok(Self {
            log_beta_atac,
            log_beta_rna,
            gate_logit,
            bias_k,
            feature_weights_atac: None,
            feature_weights_rna: None,
        })
    }

    pub fn attach_feature_weights_atac(&mut self, weights: &[f32], dev: &Device) -> Result<()> {
        self.feature_weights_atac =
            Some(Tensor::from_slice(weights, (1, weights.len()), dev)?);
        Ok(())
    }

    pub fn attach_feature_weights_rna(&mut self, weights: &[f32], dev: &Device) -> Result<()> {
        self.feature_weights_rna =
            Some(Tensor::from_slice(weights, (1, weights.len()), dev)?);
        Ok(())
    }

    fn biased_z(&self, log_z_nk: &Tensor) -> Result<Tensor> {
        log_z_nk.broadcast_add(&self.bias_k)?.exp()
    }

    /// ATAC Fisher-weighted multinomial log-likelihood. Returns `[N]`.
    pub fn forward_atac(&self, log_z_nk: &Tensor, x_atac: &Tensor) -> Result<Tensor> {
        let beta = self.log_beta_atac.exp()?;
        self.forward_modality(log_z_nk, x_atac, &beta, self.feature_weights_atac.as_ref())
    }

    /// RNA Fisher-weighted multinomial log-likelihood with gated dictionary.
    pub fn forward_rna(
        &self,
        log_z_nk: &Tensor,
        x_rna: &Tensor,
        log_w_linked: &Tensor,
    ) -> Result<Tensor> {
        let w = self.gated_log_rna_dictionary(log_w_linked)?.exp()?;
        self.forward_modality(log_z_nk, x_rna, &w, self.feature_weights_rna.as_ref())
    }

    fn forward_modality(
        &self,
        log_z_nk: &Tensor,
        x: &Tensor,
        dictionary: &Tensor,
        feature_weights: Option<&Tensor>,
    ) -> Result<Tensor> {
        let z_nk = self.biased_z(log_z_nk)?;
        let recon = z_nk.matmul(&dictionary.t()?)?;
        let recon_sum = recon.sum(1)?.unsqueeze(1)?;
        let p = recon.broadcast_div(&recon_sum)?;
        let log_p = (p + 1e-8)?.log()?;
        let weighted_x = match feature_weights {
            Some(w) => x.broadcast_mul(w)?,
            None => x.clone(),
        };
        (weighted_x * log_p)?.sum(1)
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
