use crate::common::*;
use crate::linkage::{rna_dictionary_from_m, save_linkage_results, FeatureAnnotations};
use crate::topic::encoder::EncoderInput;
use crate::topic::training::TrainedModel;
use candle_util::candle_core::{Device, Tensor};
use genomic_data::coordinates::PeakCoord;

pub struct EvalContext<'a> {
    pub rna_coarsenings: &'a [Option<FeatureCoarsening>],
    pub atac_coarsenings: &'a [Option<FeatureCoarsening>],
    pub cis_indices: &'a Tensor,
    pub flat_cis_indices: &'a Tensor,
    pub gene_names: &'a [Box<str>],
    pub peak_names: &'a [Box<str>],
    pub peak_coords: &'a [Option<PeakCoord>],
    pub data_stack: &'a SparseIoStack,
    pub c_max: usize,
    pub dev: &'a Device,
}

pub fn save_outputs(
    model: &TrainedModel,
    ctx: &EvalContext,
    out_prefix: &str,
) -> anyhow::Result<()> {
    save_parameters(model, ctx, out_prefix)?;
    save_latent(model, ctx, out_prefix)?;
    info!("Done.");
    Ok(())
}

/// Save model parameters: dictionaries and linkage results.
fn save_parameters(
    model: &TrainedModel,
    ctx: &EvalContext,
    out_prefix: &str,
) -> anyhow::Result<()> {
    let num_levels = model.level_dims.len();
    let finest_dec = model.decoders.last().unwrap();
    let finest_susie = model.susies.last().unwrap();

    info!("Saving parameters to {}.*", out_prefix);

    finest_dec
        .log_beta_atac
        .exp()?
        .to_parquet(&format!("{}.atac_dict.parquet", out_prefix))?;

    let m_gc = finest_susie.forward()?;
    let is_coarsened = ctx.rna_coarsenings.last().unwrap().is_some()
        || ctx.atac_coarsenings.last().unwrap().is_some();
    let log_w_linked = if is_coarsened {
        m_gc.exp()?
            .matmul(&finest_dec.log_beta_atac.exp()?)?
            .log()?
    } else {
        rna_dictionary_from_m(&m_gc, &finest_dec.log_beta_atac, ctx.flat_cis_indices)?.log()?
    };
    finest_dec
        .gated_log_rna_dictionary(&log_w_linked)?
        .exp()?
        .to_parquet(&format!("{}.rna_dict.parquet", out_prefix))?;

    finest_dec
        .gate_alpha()?
        .unsqueeze(1)?
        .to_parquet(&format!("{}.gate_alpha.parquet", out_prefix))?;

    if finest_dec.ambient_enabled() {
        finest_dec
            .alpha_amb()?
            .t()?
            .to_parquet(&format!("{}.alpha_amb.parquet", out_prefix))?;
    }

    finest_dec
        .log_beta_atac
        .to_parquet(&format!("{}.log_beta.parquet", out_prefix))?;

    let linkage_indices = if is_coarsened {
        let (dg, dp) = model.level_dims[num_levels - 1];
        let seq: Vec<u32> = (0..dg).flat_map(|_| (0..dp).map(|p| p as u32)).collect();
        Tensor::from_vec(seq, (dg, dp), ctx.dev)?
    } else {
        ctx.cis_indices.clone()
    };
    save_linkage_results(
        &finest_susie.pip()?,
        &finest_susie.posterior_mean()?,
        &finest_susie.posterior_var()?,
        &linkage_indices,
        &FeatureAnnotations {
            gene_names: ctx.gene_names,
            peak_names: ctx.peak_names,
            peak_coords: ctx.peak_coords,
        },
        &format!("{}.results.bed.gz", out_prefix),
    )?;

    Ok(())
}

/// Evaluate encoder on original cells and save per-cell topic proportions.
fn save_latent(model: &TrainedModel, ctx: &EvalContext, out_prefix: &str) -> anyhow::Result<()> {
    let finest_susie = model.susies.last().unwrap();
    let finest_dec = model.decoders.last().unwrap();
    let m_gc = finest_susie.forward()?;

    let enc_weights = match &model.enc_expand_indices {
        Some(idx) => m_gc
            .flatten_all()?
            .index_select(idx, 0)?
            .reshape((ctx.gene_names.len(), ctx.c_max))?
            .exp()?,
        None => m_gc.exp()?,
    };

    let ncells = ctx.data_stack.num_columns()?;
    let block_size = 4096usize;
    let n_blocks = ncells.div_ceil(block_size);
    let mut all_prop: Vec<Tensor> = Vec::with_capacity(n_blocks);
    let collect_rho = finest_dec.ambient_enabled();
    let mut all_rho: Vec<Tensor> = if collect_rho {
        Vec::with_capacity(n_blocks)
    } else {
        Vec::new()
    };

    info!("Evaluating {} cells in {} blocks...", ncells, n_blocks);

    for b in 0..n_blocks {
        let start = b * block_size;
        let end = ((b + 1) * block_size).min(ncells);

        let rna_block = ctx.data_stack.stack[0]
            .read_columns_tensor(start..end)?
            .t()?
            .contiguous()?;
        let atac_block = ctx.data_stack.stack[1]
            .read_columns_tensor(start..end)?
            .t()?
            .contiguous()?;

        if collect_rho {
            let lib_n1 = rna_block.sum(1)?.unsqueeze(1)?;
            all_rho.push(finest_dec.rho_from_lib(&lib_n1)?);
        }

        let enc_inp = EncoderInput {
            x_rna: &rna_block,
            x_atac: &atac_block,
            batch_rna: None,
            batch_atac: None,
            m_weights: &enc_weights,
            flat_cis_indices: ctx.flat_cis_indices,
            c_max: ctx.c_max,
        };
        let (log_z, _) = model.encoder.forward(&enc_inp, false)?;
        all_prop.push(log_z.exp()?);
    }

    Tensor::cat(&all_prop, 0)?.to_parquet(&format!("{}.prop.parquet", out_prefix))?;
    if collect_rho {
        Tensor::cat(&all_rho, 0)?.to_parquet(&format!("{}.rho.parquet", out_prefix))?;
    }
    Ok(())
}
