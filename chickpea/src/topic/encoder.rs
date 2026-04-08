use candle_util::candle_aux_layers::{stack_relu_linear, StackLayers};
use candle_util::candle_core::{Result, Tensor};
use candle_util::candle_loss_functions::gaussian_kl_loss;
use candle_util::candle_nn::{self, ops, BatchNorm, Linear, ModuleT, VarBuilder};

/// Inputs for the encoder forward pass.
pub struct EncoderInput<'a> {
    pub x_rna: &'a Tensor,
    pub x_atac: &'a Tensor,
    pub batch_rna: Option<&'a Tensor>,
    pub batch_atac: Option<&'a Tensor>,
    pub m_weights: &'a Tensor,
    pub flat_cis_indices: &'a Tensor,
    pub c_max: usize,
}

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

/// log1p normalize, then subtract batch correction in the same normalized space.
fn log1p_normalize_corrected(
    x: &Tensor,
    batch: Option<&Tensor>,
    n_features: usize,
) -> Result<Tensor> {
    let h = log1p_normalize(x, n_features)?;
    match batch {
        Some(b) => h - log1p_normalize(b, n_features)?,
        None => Ok(h),
    }
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

    fn forward(&self, inp: &EncoderInput, train: bool) -> Result<(Tensor, Tensor)> {
        let n = inp.x_rna.dim(0)?;

        let atac_gathered = inp.x_atac.index_select(inp.flat_cis_indices, 1)?;
        let atac_gathered = atac_gathered.reshape((n, self.n_genes, inp.c_max))?;
        let agg = atac_gathered
            .broadcast_mul(&inp.m_weights.unsqueeze(0)?)?
            .sum(2)?;

        let log_agg = (&agg + 1.0)?.log()?;
        let fused = ((inp.x_rna + 1.0)?.log()? + &log_agg)?;
        let normalized = match inp.batch_rna {
            Some(b) => {
                let fused_b = ((b + 1.0)?.log()? + &log_agg)?;
                (log1p_normalize(&fused, self.n_genes)? - log1p_normalize(&fused_b, self.n_genes)?)?
            }
            None => log1p_normalize(&fused, self.n_genes)?,
        };

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

    fn forward(
        &self,
        x_atac: &Tensor,
        batch_atac: Option<&Tensor>,
        train: bool,
    ) -> Result<(Tensor, Tensor)> {
        let normalized = log1p_normalize_corrected(x_atac, batch_atac, self.n_peaks)?;
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

    pub fn forward(&self, inp: &EncoderInput, train: bool) -> Result<(Tensor, Tensor)> {
        let (z_gene_mean, z_gene_lnvar) = self.gene_expert.forward(inp, train)?;
        let (z_atac_mean, z_atac_lnvar) =
            self.atac_expert
                .forward(inp.x_atac, inp.batch_atac, train)?;

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
