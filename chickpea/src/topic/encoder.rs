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
    n_genes: usize,
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
            n_genes,
            context_size,
        })
    }

    pub fn forward(&self, inp: &EncoderInput, train: bool) -> Result<(Tensor, Tensor)> {
        let n = inp.x_rna.dim(0)?;

        // Step 1: Select top-K genes per sample by RNA expression
        let (gene_union_idx, selected_genes_per_row) =
            top_k_gene_indices(inp.x_rna, self.context_size)?;

        // Step 2: Gene expert — fused RNA+ATAC values for selected genes
        let (gene_values, gene_null) =
            self.prepare_gene_values(inp, n, &gene_union_idx, &selected_genes_per_row)?;
        let (z_gene_mean, z_gene_lnvar) = self.gene_expert.latent_gaussian_params_indexed(
            &gene_union_idx,
            &gene_values,
            gene_null.as_ref(),
            train,
        )?;

        // Step 3: ATAC expert — cis-window peaks of selected genes
        let (atac_peak_idx, atac_values, atac_null) =
            self.prepare_atac_from_genes(inp, n, &selected_genes_per_row)?;
        let (z_atac_mean, z_atac_lnvar) = self.atac_expert.latent_gaussian_params_indexed(
            &atac_peak_idx,
            &atac_values,
            atac_null.as_ref(),
            train,
        )?;

        // Simple average of experts
        let z_mean = ((&z_gene_mean + &z_atac_mean)? * 0.5)?;
        let combined_var = ((z_gene_lnvar.exp()? + z_atac_lnvar.exp()?)? * 0.25)?;
        let z_lnvar = (combined_var + 1e-8)?.log()?;

        let z = if train {
            let eps = Tensor::randn_like(&z_mean, 0., 1.)?;
            (&z_mean + (&z_lnvar * 0.5)?.exp()?.broadcast_mul(&eps)?)?
        } else {
            z_mean.clone()
        };
        let log_z = ops::log_softmax(&z, 1)?;
        let kl = gaussian_kl_loss(&z_mean, &z_lnvar)?;

        Ok((log_z, kl))
    }

    /// Prepare gene expert values: fuse RNA with SuSiE-weighted ATAC for selected genes.
    fn prepare_gene_values(
        &self,
        inp: &EncoderInput,
        n: usize,
        gene_union_idx: &Tensor,
        _selected_per_row: &[Vec<usize>],
    ) -> Result<(Tensor, Option<Tensor>)> {
        // Gather ATAC for cis-candidates of ALL genes, weighted sum by M → [N, G]
        let atac_gathered = inp.x_atac.index_select(inp.flat_cis_indices, 1)?;
        let atac_gathered = atac_gathered.reshape((n, self.n_genes, inp.c_max))?;
        let agg = atac_gathered
            .broadcast_mul(&inp.m_weights.unsqueeze(0)?)?
            .sum(2)?; // [N, G]

        // Fuse in log-space
        let fused = ((inp.x_rna + 1.0)?.log()? + (&agg + 1.0)?.log()?)?; // [N, G]

        // Gather only the union-selected gene columns
        let values = fused.index_select(gene_union_idx, 1)?; // [N, S_genes]

        let null = match inp.batch_rna {
            Some(b) => {
                let fused_b = ((b + 1.0)?.log()? + (&agg + 1.0)?.log()?)?;
                Some(fused_b.index_select(gene_union_idx, 1)?)
            }
            None => None,
        };

        Ok((values, null))
    }

    /// Prepare ATAC expert input: gather cis-window peaks for the selected genes.
    /// For each sample's top-K genes, collect their cis-candidate peak indices,
    /// take union across samples, and gather ATAC counts + optional batch.
    fn prepare_atac_from_genes(
        &self,
        inp: &EncoderInput,
        _n: usize,
        selected_per_row: &[Vec<usize>],
    ) -> Result<(Tensor, Tensor, Option<Tensor>)> {
        let cis_raw: Vec<u32> = inp.flat_cis_indices.to_vec1()?;

        // Union of peak indices across all samples' selected genes' cis-windows
        let mut peak_union = rustc_hash::FxHashSet::default();
        for genes in selected_per_row {
            for &g in genes {
                let base = g * inp.c_max;
                for c in 0..inp.c_max {
                    peak_union.insert(cis_raw[base + c] as usize);
                }
            }
        }
        let mut peak_indices: Vec<usize> = peak_union.into_iter().collect();
        peak_indices.sort_unstable();

        let idx_u32: Vec<u32> = peak_indices.iter().map(|&i| i as u32).collect();
        let peak_idx_tensor = Tensor::from_vec(idx_u32, peak_indices.len(), inp.x_atac.device())?;

        // Gather ATAC counts for union peaks
        let values = inp.x_atac.index_select(&peak_idx_tensor, 1)?;

        let null = match inp.batch_atac {
            Some(b) => Some(b.index_select(&peak_idx_tensor, 1)?),
            None => None,
        };

        Ok((peak_idx_tensor, values, null))
    }
}

/// Select top-K genes per sample by value. Returns:
/// - union_indices: [S] u32 tensor of union gene indices across all samples
/// - per_row: Vec<Vec<usize>> of selected gene indices per sample
fn top_k_gene_indices(x_rna: &Tensor, k: usize) -> Result<(Tensor, Vec<Vec<usize>>)> {
    let (n, d) = (x_rna.dim(0)?, x_rna.dim(1)?);
    let k = k.min(d);
    let data: Vec<f32> = x_rna.flatten_all()?.to_vec1()?;

    let mut union_set = rustc_hash::FxHashSet::default();
    let mut per_row = Vec::with_capacity(n);

    for row in 0..n {
        let row_data = &data[row * d..(row + 1) * d];
        let mut idx_val: Vec<(usize, f32)> = row_data.iter().copied().enumerate().collect();
        idx_val.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let selected: Vec<usize> = idx_val.iter().take(k).map(|&(i, _)| i).collect();
        for &i in &selected {
            union_set.insert(i);
        }
        per_row.push(selected);
    }

    let mut union_indices: Vec<usize> = union_set.into_iter().collect();
    union_indices.sort_unstable();

    let idx_u32: Vec<u32> = union_indices.iter().map(|&i| i as u32).collect();
    let tensor = Tensor::from_vec(idx_u32, union_indices.len(), x_rna.device())?;

    Ok((tensor, per_row))
}
