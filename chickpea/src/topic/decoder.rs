use super::susie::SuSiE;
use candle_util::candle_core::{IndexOp, Result, Tensor};
use candle_util::candle_loss_functions::nb_log_likelihood;
use candle_util::candle_nn::{self, VarBuilder};

pub struct DecoderArgs {
    pub n_genes: usize,
    pub n_peaks: usize,
    pub n_topics: usize,
    pub n_ser_components: usize,
    pub cis_indices: Tensor,
    pub cis_mask: Tensor,
}

/// Chickpea decoder: ATAC (NB) + RNA (NB with SuSiE linkage).
///
/// β[P,K]: direct learnable log-dictionary.
/// M[G,C_max]: SuSiE — sum of L single-effect regressions per gene.
/// RNA dictionary: W[g,k] = Σ_p exp(M_gp) · β[p,k].
pub struct ChickpeaDecoder {
    pub log_beta: Tensor,         // [P, K]
    pub susie: SuSiE,             // M[G, C_max]
    bias_k: Tensor,               // [1, K]
    intercept_g: Tensor,          // [1, G]
    log_phi_atac: Tensor,         // [1, P]
    log_phi_rna: Tensor,          // [1, G]
    pub flat_cis_indices: Tensor, // [G*C_max] u32, pre-flattened
    n_topics: usize,
}

impl ChickpeaDecoder {
    pub fn new(args: DecoderArgs, vs: VarBuilder) -> Result<Self> {
        let DecoderArgs {
            n_genes,
            n_peaks,
            n_topics,
            n_ser_components,
            cis_indices,
            cis_mask,
        } = args;

        let beta_init = candle_nn::Init::Randn {
            mean: 0.0,
            stdev: 0.1,
        };
        let log_beta = vs.get_with_hints((n_peaks, n_topics), "log_beta", beta_init)?;

        let susie = SuSiE::new(
            n_genes,
            cis_indices.dim(1)?,
            n_ser_components,
            Some(cis_mask),
            vs.pp("susie"),
        )?;

        let ln2 = (2.0_f64).ln();
        let bias_k = vs.get_with_hints((1, n_topics), "bias_k", candle_nn::Init::Const(0.0))?;
        let intercept_g =
            vs.get_with_hints((1, n_genes), "intercept_g", candle_nn::Init::Const(0.0))?;
        let log_phi_atac =
            vs.get_with_hints((1, n_peaks), "log_phi_atac", candle_nn::Init::Const(ln2))?;
        let log_phi_rna =
            vs.get_with_hints((1, n_genes), "log_phi_rna", candle_nn::Init::Const(ln2))?;

        let flat_cis_indices = cis_indices.flatten_all()?;

        Ok(Self {
            log_beta,
            susie,
            bias_k,
            intercept_g,
            log_phi_atac,
            log_phi_rna,
            flat_cis_indices,
            n_topics,
        })
    }

    fn biased_z(&self, log_z_nk: &Tensor) -> Result<Tensor> {
        log_z_nk.broadcast_add(&self.bias_k)?.exp()
    }

    /// ATAC NB log-likelihood. Returns [N] per-sample.
    pub fn forward_atac(&self, log_z_nk: &Tensor, x_atac: &Tensor) -> Result<Tensor> {
        let beta_pk = self.log_beta.exp()?;
        let z_nk = self.biased_z(log_z_nk)?;
        let recon_np = z_nk.matmul(&beta_pk.t()?)?;
        let lib = x_atac.sum(1)?.unsqueeze(1)?;
        let recon_sum = recon_np.sum(1)?.unsqueeze(1)?;
        let mu_np = recon_np
            .broadcast_mul(&lib)?
            .broadcast_div(&(&recon_sum + 1e-8)?)?;
        nb_log_likelihood(x_atac, &mu_np, &self.log_phi_atac)
    }

    /// RNA NB log-likelihood with pre-computed SuSiE M. Returns [N] per-sample.
    pub fn forward_rna(&self, log_z_nk: &Tensor, x_rna: &Tensor, m_gc: &Tensor) -> Result<Tensor> {
        let w_gk = self.rna_dictionary_from_m(m_gc)?;
        let z_nk = self.biased_z(log_z_nk)?;
        let recon_ng = z_nk.matmul(&w_gk.t()?)?;
        let recon_ng = recon_ng.broadcast_add(&self.intercept_g.exp()?)?;
        let lib = x_rna.sum(1)?.unsqueeze(1)?;
        let recon_sum = recon_ng.sum(1)?.unsqueeze(1)?;
        let mu_ng = recon_ng
            .broadcast_mul(&lib)?
            .broadcast_div(&(&recon_sum + 1e-8)?)?;
        nb_log_likelihood(x_rna, &mu_ng, &self.log_phi_rna)
    }

    /// Derived RNA dictionary from pre-computed M.
    fn rna_dictionary_from_m(&self, m_gc: &Tensor) -> Result<Tensor> {
        let beta_pk = self.log_beta.exp()?;
        let beta_gathered = beta_pk.i(&self.flat_cis_indices)?;
        let beta_gathered =
            beta_gathered.reshape((self.susie.n_genes, self.susie.c_max, self.n_topics))?;
        let m_weights = m_gc.exp()?.unsqueeze(2)?;
        m_weights.broadcast_mul(&beta_gathered)?.sum(1)
    }

    /// RNA dictionary using posterior mean (eval mode).
    pub fn rna_dictionary(&self, train: bool) -> Result<Tensor> {
        let m_gc = self.susie.forward(train)?;
        self.rna_dictionary_from_m(&m_gc)
    }
}
