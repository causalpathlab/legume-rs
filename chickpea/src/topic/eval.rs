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

    // `T{c}` topic-column names for the three K-dim outputs below;
    // single closure keeps them in sync if K changes.
    let topic_col_names = |t: &Tensor| -> Vec<Box<str>> {
        let k = t.dims().last().copied().unwrap_or(0);
        (0..k).map(|i| format!("T{i}").into_boxed_str()).collect()
    };

    let atac_dict_t = finest_dec.log_beta_atac.exp()?;
    let atac_dict_cols = topic_col_names(&atac_dict_t);
    atac_dict_t.to_parquet_with_names(
        &format!("{}.atac_dict.parquet", out_prefix),
        (None, None),
        Some(&atac_dict_cols),
    )?;

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
    let rna_dict_t = finest_dec.gated_log_rna_dictionary(&log_w_linked)?.exp()?;
    let rna_dict_cols = topic_col_names(&rna_dict_t);
    rna_dict_t.to_parquet_with_names(
        &format!("{}.rna_dict.parquet", out_prefix),
        (None, None),
        Some(&rna_dict_cols),
    )?;

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

    let log_beta_cols = topic_col_names(&finest_dec.log_beta_atac);
    finest_dec.log_beta_atac.to_parquet_with_names(
        &format!("{}.log_beta.parquet", out_prefix),
        (None, None),
        Some(&log_beta_cols),
    )?;

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
    let num_features: usize = ctx.data_stack.stack.iter().map(|v| v.num_rows()).sum();
    let block_size = matrix_util::utils::default_block_size(num_features);
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

    let prop = Tensor::cat(&all_prop, 0)?;
    let k_topics = prop.dims().last().copied().unwrap_or(0);
    let prop_topic_cols: Vec<Box<str>> = (0..k_topics)
        .map(|i| format!("T{i}").into_boxed_str())
        .collect();
    prop.to_parquet_with_names(
        &format!("{}.prop.parquet", out_prefix),
        (None, None),
        Some(&prop_topic_cols),
    )?;
    if collect_rho {
        Tensor::cat(&all_rho, 0)?.to_parquet(&format!("{}.rho.parquet", out_prefix))?;
    }
    Ok(())
}
