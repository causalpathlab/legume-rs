use candle_util::candle_aux_layers::{stack_relu_linear, StackLayers};
use candle_util::candle_core::{Result, Tensor};
use candle_util::candle_loss_functions::gaussian_kl_loss;
use candle_util::candle_nn::{self, ops, BatchNorm, Linear, ModuleT, VarBuilder};

/// Shared MLP head: FC → BN → z_mean, z_lnvar.
struct MlpHead {
    fc: StackLayers<Linear>,
    bn_z: BatchNorm,
    z_mean: Linear,
    z_lnvar: Linear,
}

impl MlpHead {
    fn new(
        in_dim: usize,
        n_topics: usize,
        hidden_layers: &[usize],
        vs: VarBuilder,
    ) -> Result<Self> {
        let fc_dims = hidden_layers[..hidden_layers.len() - 1].to_vec();
        let out_dim = *hidden_layers.last().unwrap();
        let fc = stack_relu_linear(in_dim, out_dim, &fc_dims, vs.pp("fc"))?;
        let bn_config = candle_nn::BatchNormConfig {
            eps: 1e-4,
            remove_mean: true,
            affine: true,
            momentum: 0.1,
        };
        let bn_z = candle_nn::batch_norm(out_dim, bn_config, vs.pp("bn_z"))?;
        let z_mean = candle_nn::linear(out_dim, n_topics, vs.pp("z_mean"))?;
        let z_lnvar = candle_nn::linear(out_dim, n_topics, vs.pp("z_lnvar"))?;
        Ok(Self {
            fc,
            bn_z,
            z_mean,
            z_lnvar,
        })
    }

    fn forward(&self, x: &Tensor, train: bool) -> Result<(Tensor, Tensor)> {
        let fc_out = self.fc.forward_t(x, train)?;
        let bn_out = self.bn_z.forward_t(&fc_out, train)?;
        let z_mean = self.z_mean.forward_t(&bn_out, train)?;
        let z_lnvar = self.z_lnvar.forward_t(&bn_out, train)?.clamp(-10.0, 10.0)?;
        Ok((z_mean, z_lnvar))
    }
}

fn log1p_normalize(x: &Tensor, n_features: usize) -> Result<Tensor> {
    let lx = (x + 1.0)?.log()?;
    let denom = lx.sum_keepdim(1)?;
    lx.broadcast_div(&(&denom + 1e-8)?)? * (n_features as f64)
}

fn reparameterize(z_mean: &Tensor, z_lnvar: &Tensor, train: bool) -> Result<Tensor> {
    if train {
        let eps = Tensor::randn_like(z_mean, 0., 1.)?;
        z_mean + (z_lnvar * 0.5)?.exp()? * eps
    } else {
        Ok(z_mean.clone())
    }
}

/// Gene-centric fusion encoder expert.
///
/// Fuses RNA expression with linked ATAC peaks weighted by pre-computed M weights.
struct GeneFusedEncoder {
    head: MlpHead,
    n_genes: usize,
}

/// ATAC encoder expert.
struct AtacEncoder {
    head: MlpHead,
    n_peaks: usize,
}

/// MoE encoder combining gene-fused and ATAC experts.
pub struct ChickpeaEncoder {
    gene_expert: GeneFusedEncoder,
    atac_expert: AtacEncoder,
    gate_logit: Tensor,
}

impl GeneFusedEncoder {
    fn new(
        n_genes: usize,
        n_topics: usize,
        hidden_layers: &[usize],
        vs: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            head: MlpHead::new(n_genes, n_topics, hidden_layers, vs)?,
            n_genes,
        })
    }

    /// `m_weights`: pre-computed exp(SuSiE M) [G, C_max].
    fn forward(
        &self,
        x_rna: &Tensor,
        x_atac: &Tensor,
        m_weights: &Tensor,
        flat_cis_indices: &Tensor,
        c_max: usize,
        train: bool,
    ) -> Result<(Tensor, Tensor)> {
        let n = x_rna.dim(0)?;

        // Gather ATAC for cis-candidates, weighted sum by M
        let atac_gathered = x_atac.index_select(flat_cis_indices, 1)?;
        let atac_gathered = atac_gathered.reshape((n, self.n_genes, c_max))?;
        let agg = atac_gathered
            .broadcast_mul(&m_weights.unsqueeze(0)?)?
            .sum(2)?;

        // Fuse + normalize
        let fused = ((x_rna + 1.0)?.log()? + (agg + 1.0)?.log()?)?;
        let normalized = log1p_normalize(&fused, self.n_genes)?;

        self.head.forward(&normalized, train)
    }
}

impl AtacEncoder {
    fn new(
        n_peaks: usize,
        n_topics: usize,
        hidden_layers: &[usize],
        vs: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            head: MlpHead::new(n_peaks, n_topics, hidden_layers, vs)?,
            n_peaks,
        })
    }

    fn forward(&self, x_atac: &Tensor, train: bool) -> Result<(Tensor, Tensor)> {
        let normalized = log1p_normalize(x_atac, self.n_peaks)?;
        self.head.forward(&normalized, train)
    }
}

impl ChickpeaEncoder {
    pub fn new(
        n_genes: usize,
        n_peaks: usize,
        n_topics: usize,
        hidden_layers: &[usize],
        vs: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            gene_expert: GeneFusedEncoder::new(
                n_genes,
                n_topics,
                hidden_layers,
                vs.pp("gene_expert"),
            )?,
            atac_expert: AtacEncoder::new(n_peaks, n_topics, hidden_layers, vs.pp("atac_expert"))?,
            gate_logit: vs.get_with_hints(
                (1, n_topics),
                "gate_logit",
                candle_nn::Init::Const(0.0),
            )?,
        })
    }

    /// Forward pass. `m_weights` = exp(SuSiE M), pre-computed once per minibatch.
    pub fn forward(
        &self,
        x_rna: &Tensor,
        x_atac: &Tensor,
        m_weights: &Tensor,
        flat_cis_indices: &Tensor,
        c_max: usize,
        train: bool,
    ) -> Result<(Tensor, Tensor)> {
        let (z_gene_mean, z_gene_lnvar) =
            self.gene_expert
                .forward(x_rna, x_atac, m_weights, flat_cis_indices, c_max, train)?;
        let (z_atac_mean, z_atac_lnvar) = self.atac_expert.forward(x_atac, train)?;

        // MoE gating
        let gate = ops::sigmoid(&self.gate_logit)?;
        let gate_comp = (1.0 - &gate)?;

        let z_mean =
            (gate.broadcast_mul(&z_gene_mean)? + gate_comp.broadcast_mul(&z_atac_mean)?)?;

        let var_gene = z_gene_lnvar.exp()?;
        let var_atac = z_atac_lnvar.exp()?;
        let g2 = (&gate * &gate)?;
        let gc2 = (&gate_comp * &gate_comp)?;
        let combined_var = (g2.broadcast_mul(&var_gene)? + gc2.broadcast_mul(&var_atac)?)?;
        let z_lnvar = (combined_var + 1e-8)?.log()?;

        let z = reparameterize(&z_mean, &z_lnvar, train)?;
        let log_z = ops::log_softmax(&z, 1)?;
        let kl = gaussian_kl_loss(&z_mean, &z_lnvar)?;

        Ok((log_z, kl))
    }
}
