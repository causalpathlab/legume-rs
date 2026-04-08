use candle_util::candle_core::{Result, Tensor};
use candle_util::candle_encoder_indexed::{IndexedEmbeddingEncoder, IndexedEmbeddingEncoderArgs};
use candle_util::candle_loss_functions::gaussian_kl_loss;
use candle_util::candle_nn::{ops, VarBuilder};

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

/// Chickpea encoder: averages gene and ATAC indexed experts.
///
/// Gene selection drives both experts:
/// 1. Top-K genes per sample selected by RNA expression
/// 2. Gene expert: indexed gene embeddings weighted by fused RNA+ATAC signal
/// 3. ATAC expert: indexed peak embeddings for cis-window peaks of selected genes
pub struct ChickpeaEncoder {
    gene_expert: IndexedEmbeddingEncoder,
    atac_expert: IndexedEmbeddingEncoder,
    context_size: usize,
}

impl ChickpeaEncoder {
    pub fn new(
        n_genes: usize,
        n_peaks: usize,
        n_topics: usize,
        embedding_dim: usize,
        context_size: usize,
        hidden_layers: &[usize],
        vs: VarBuilder,
    ) -> Result<Self> {
        let gene_expert = IndexedEmbeddingEncoder::new(
            IndexedEmbeddingEncoderArgs {
                n_features: n_genes,
                n_topics,
                embedding_dim,
                layers: hidden_layers,
            },
            vs.pp("gene_expert"),
        )?;

        let atac_expert = IndexedEmbeddingEncoder::new(
            IndexedEmbeddingEncoderArgs {
                n_features: n_peaks,
                n_topics,
                embedding_dim,
                layers: hidden_layers,
            },
            vs.pp("atac_expert"),
        )?;

        Ok(Self {
            gene_expert,
            atac_expert,
            context_size,
        })
    }

    pub fn forward(&self, inp: &EncoderInput, train: bool) -> Result<(Tensor, Tensor)> {
        // Select top-K genes and derive their cis-peak union
        let sel = select_genes_and_peaks(inp, self.context_size)?;

        // Gene expert: fused RNA+ATAC for selected genes only
        let (gene_values, gene_null) = self.prepare_gene_values(inp, &sel)?;
        let (z_gene_mean, z_gene_lnvar) = self.gene_expert.latent_gaussian_params_indexed(
            &sel.gene_idx,
            &gene_values,
            gene_null.as_ref(),
            train,
        )?;

        // ATAC expert: cis-window peaks of selected genes
        let atac_values = inp.x_atac.index_select(&sel.peak_idx, 1)?;
        let atac_null = inp
            .batch_atac
            .map(|b| b.index_select(&sel.peak_idx, 1))
            .transpose()?;
        let (z_atac_mean, z_atac_lnvar) = self.atac_expert.latent_gaussian_params_indexed(
            &sel.peak_idx,
            &atac_values,
            atac_null.as_ref(),
            train,
        )?;

        // Average experts
        let z_mean = ((&z_gene_mean + &z_atac_mean)? * 0.5)?;
        let combined_var = ((z_gene_lnvar.exp()? + z_atac_lnvar.exp()?)? * 0.25)?;
        let z_lnvar = (combined_var + 1e-8)?.log()?;

        let z = if train {
            let eps = Tensor::randn_like(&z_mean, 0., 1.)?;
            (&z_mean + (&z_lnvar * 0.5)?.exp()?.broadcast_mul(&eps)?)?
        } else {
            z_mean.clone()
        };

        Ok((
            ops::log_softmax(&z, 1)?,
            gaussian_kl_loss(&z_mean, &z_lnvar)?,
        ))
    }

    /// Fuse RNA + SuSiE-weighted ATAC for union-selected genes only.
    fn prepare_gene_values(
        &self,
        inp: &EncoderInput,
        sel: &GeneSelection,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let n = inp.x_rna.dim(0)?;
        let s = sel.gene_union.len();

        // Gather cis-peak ATAC only for union genes → [N, S, C_max]
        let union_cis: Vec<u32> = sel
            .gene_union
            .iter()
            .flat_map(|&g| {
                let base = g * inp.c_max;
                (0..inp.c_max).map(move |c| sel.cis_raw[base + c])
            })
            .collect();
        let union_cis_t = Tensor::from_vec(union_cis, s * inp.c_max, inp.x_atac.device())?;
        let atac_gathered = inp
            .x_atac
            .index_select(&union_cis_t, 1)?
            .reshape((n, s, inp.c_max))?;

        let m_data: Vec<f32> = inp.m_weights.flatten_all()?.to_vec1()?;
        let m_ref = &m_data;
        let union_m: Vec<f32> = sel
            .gene_union
            .iter()
            .flat_map(|&g| {
                let base = g * inp.c_max;
                (0..inp.c_max).map(move |c| m_ref[base + c])
            })
            .collect();
        let m_union =
            Tensor::from_vec(union_m, (s, inp.c_max), inp.x_atac.device())?.unsqueeze(0)?;

        let agg = atac_gathered.broadcast_mul(&m_union)?.sum(2)?; // [N, S]

        // Fuse in log-space: log(rna+1) + log(atac_agg+1)
        let rna_sel = inp.x_rna.index_select(&sel.gene_idx, 1)?; // [N, S]
        let fused = ((rna_sel + 1.0)?.log()? + (&agg + 1.0)?.log()?)?;

        let null = match inp.batch_rna {
            Some(b) => {
                let b_sel = b.index_select(&sel.gene_idx, 1)?;
                Some(((b_sel + 1.0)?.log()? + (&agg + 1.0)?.log()?)?)
            }
            None => None,
        };

        Ok((fused, null))
    }
}

/// Precomputed gene and peak selections for the encoder.
struct GeneSelection {
    gene_idx: Tensor,       // [S_genes] u32 union of selected genes
    gene_union: Vec<usize>, // same as gene_idx but as Vec for indexing
    peak_idx: Tensor,       // [S_peaks] u32 union of cis-peaks of selected genes
    cis_raw: Vec<u32>,      // flat cis indices (cached to avoid repeated to_vec1)
}

/// Select top-K genes per sample (partial sort), compute gene union and peak union.
fn select_genes_and_peaks(inp: &EncoderInput, k: usize) -> Result<GeneSelection> {
    let (n, d) = (inp.x_rna.dim(0)?, inp.x_rna.dim(1)?);
    let k = k.min(d);
    let data: Vec<f32> = inp.x_rna.flatten_all()?.to_vec1()?;
    let cis_raw: Vec<u32> = inp.flat_cis_indices.to_vec1()?;

    let mut gene_set = rustc_hash::FxHashSet::default();
    let mut peak_set = rustc_hash::FxHashSet::default();

    for row in 0..n {
        let row_data = &data[row * d..(row + 1) * d];
        let mut idx_val: Vec<(usize, f32)> = row_data.iter().copied().enumerate().collect();

        // Partial sort: O(d) average to find top-K
        if k < idx_val.len() {
            idx_val.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
            idx_val.truncate(k);
        }

        for &(g, _) in &idx_val {
            if gene_set.insert(g) {
                // New gene — add its cis-peaks to peak union
                let base = g * inp.c_max;
                for c in 0..inp.c_max {
                    peak_set.insert(cis_raw[base + c] as usize);
                }
            }
        }
    }

    let mut gene_union: Vec<usize> = gene_set.into_iter().collect();
    gene_union.sort_unstable();
    let gene_u32: Vec<u32> = gene_union.iter().map(|&i| i as u32).collect();
    let gene_idx = Tensor::from_vec(gene_u32, gene_union.len(), inp.x_rna.device())?;

    let mut peak_union: Vec<usize> = peak_set.into_iter().collect();
    peak_union.sort_unstable();
    let peak_u32: Vec<u32> = peak_union.iter().map(|&i| i as u32).collect();
    let peak_idx = Tensor::from_vec(peak_u32, peak_union.len(), inp.x_atac.device())?;

    Ok(GeneSelection {
        gene_idx,
        gene_union,
        peak_idx,
        cis_raw,
    })
}
