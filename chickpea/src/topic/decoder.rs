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
/// and independent log_beta_rna[G, K], optionally wrapped with ambient mixture.
///
///   log W[g,k] = α_g · log W_linked[g,k] + (1-α_g) · log β_rna[g,k]
///   π_ng       = (1-ρ_n) · (θ_n · W)_g + ρ_n · α_amb[g]        (if ambient on)
///
/// where α_g = sigmoid(gate_logit) and ρ_n = sigmoid(a · log L_n + b).
pub struct ChickpeaDecoder {
    pub log_beta_atac: Tensor, // [d_peaks_l, K]
    log_beta_rna: Tensor,      // [d_genes_l, K]
    gate_logit: Tensor,        // [d_genes_l, 1]
    bias_k: Tensor,            // [1, K]
    log_phi_atac: Tensor,      // [1, d_peaks_l]
    log_phi_rna: Tensor,       // [1, d_genes_l]
    // Ambient RNA mixture
    log_alpha_rna: Tensor, // [1, d_genes_l] ambient gene logits
    rho_a: Tensor,         // [1, 1] slope in ρ = sigmoid(a · log L + b)
    rho_b: Tensor,         // [1, 1] bias
    ambient_enabled: bool,
    rho_prior_weight: f32,
    rho_prior_alpha: f32,
    rho_prior_beta: f32,
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

        // ρ init a=-0.5, b=0: larger libraries get smaller ρ
        // (L=1000 → ρ≈0.03, L=100 → ρ≈0.09).
        let log_alpha_rna = vs.get_with_hints(
            (1, n_features_rna),
            "log_alpha_rna",
            candle_nn::Init::Const(0.0),
        )?;
        let rho_a = vs.get_with_hints((1, 1), "rho_a", candle_nn::Init::Const(-0.5))?;
        let rho_b = vs.get_with_hints((1, 1), "rho_b", candle_nn::Init::Const(0.0))?;

        Ok(Self {
            log_beta_atac,
            log_beta_rna,
            gate_logit,
            bias_k,
            log_phi_atac,
            log_phi_rna,
            log_alpha_rna,
            rho_a,
            rho_b,
            ambient_enabled: true,
            rho_prior_weight: 0.0,
            rho_prior_alpha: 2.0,
            rho_prior_beta: 18.0,
        })
    }

    pub fn set_ambient(&mut self, enabled: bool) {
        self.ambient_enabled = enabled;
    }

    /// Configure the Beta(α, β) prior over ρ. `weight=0` disables the prior.
    pub fn set_rho_prior(&mut self, weight: f32, alpha: f32, beta: f32) {
        self.rho_prior_weight = weight;
        self.rho_prior_alpha = alpha;
        self.rho_prior_beta = beta;
    }

    pub fn ambient_enabled(&self) -> bool {
        self.ambient_enabled
    }

    /// Ambient RNA profile α_amb [1, G] as a simplex.
    pub fn alpha_amb(&self) -> Result<Tensor> {
        let r = self.log_alpha_rna.rank();
        ops::log_softmax(&self.log_alpha_rna, r - 1)?.exp()
    }

    /// Per-sample ambient fraction ρ_n from library sizes `lib_n1` [N, 1].
    pub fn rho_from_lib(&self, lib_n1: &Tensor) -> Result<Tensor> {
        let log_lib = (lib_n1 + 1e-8)?.log()?;
        let z = log_lib
            .broadcast_mul(&self.rho_a)?
            .broadcast_add(&self.rho_b)?;
        ops::sigmoid(&z)
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

    /// RNA NB log-likelihood with gated log-space mixture and optional ambient.
    ///
    ///   topic_π = normalize_rows(θ · W)
    ///   π       = (1-ρ) · topic_π + ρ · α_amb          (ambient on)
    ///   μ       = L · π
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
        let topic_pi = recon.broadcast_div(&(&recon_sum + 1e-8)?)?;

        if !self.ambient_enabled {
            let mu = topic_pi.broadcast_mul(&lib)?;
            return nb_log_likelihood(x_rna, &mu, &self.log_phi_rna);
        }

        let alpha = self.alpha_amb()?;
        let rho_n1 = self.rho_from_lib(&lib)?;
        let one_minus_rho = rho_n1.affine(-1.0, 1.0)?;
        let pi = topic_pi
            .broadcast_mul(&one_minus_rho)?
            .broadcast_add(&alpha.broadcast_mul(&rho_n1)?)?;
        let mu = pi.broadcast_mul(&lib)?;
        let data_llik = nb_log_likelihood(x_rna, &mu, &self.log_phi_rna)?;

        if self.rho_prior_weight <= 0.0 {
            return Ok(data_llik);
        }

        // Added to llik so training's `loss = kl - llik` subtracts it,
        // pulling ρ toward the Beta mean.
        let eps = 1e-6f64;
        let log_rho = (&rho_n1 + eps)?.log()?;
        let log_1m_rho = (&one_minus_rho + eps)?.log()?;
        let coeff_a = (self.rho_prior_alpha - 1.0) as f64;
        let coeff_b = (self.rho_prior_beta - 1.0) as f64;
        let term = ((log_rho * coeff_a)? + (log_1m_rho * coeff_b)?)?;
        let log_prior_n = term.squeeze(1)?;
        let scale = self.rho_prior_weight as f64;
        data_llik + (log_prior_n * scale)?
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
