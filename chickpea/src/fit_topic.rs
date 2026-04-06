use crate::chickpea_input::load_paired_data;
use crate::common::*;
use crate::topic::decoder::DecoderArgs;
use crate::topic::{ChickpeaDecoder, ChickpeaEncoder};
use candle_util::candle_core::{DType, Device, Tensor};
use candle_util::candle_nn::{self, AdamW, Optimizer, ParamsAdamW, VarMap};
use data_beans_alg::collapse_data::{CollapsedOut, MultilevelParams};

#[derive(Args, Debug)]
pub struct FitTopicArgs {
    // ---- Input files ----
    #[arg(
        long,
        required = true,
        value_delimiter = ',',
        help = "RNA count matrix files",
        long_help = "Comma-separated paths to RNA count matrices (sparse zarr/h5).\n\
                     Multiple files are merged on shared row names (genes).\n\
                     Example: --rna-files sample1.rna.zarr,sample2.rna.zarr"
    )]
    rna_files: Vec<Box<str>>,

    #[arg(
        long,
        required = true,
        value_delimiter = ',',
        help = "ATAC count matrix files",
        long_help = "Comma-separated paths to ATAC count matrices (sparse zarr/h5).\n\
                     Multiple files are merged on shared row names (peaks).\n\
                     Cell barcodes must match the RNA files exactly."
    )]
    atac_files: Vec<Box<str>>,

    #[arg(
        long,
        value_delimiter = ',',
        help = "Batch membership files",
        long_help = "One file per data file, in RNA-then-ATAC order.\n\
                     Each file has one batch label per cell (one per line).\n\
                     If omitted, each data file is treated as its own batch."
    )]
    batch_files: Option<Vec<Box<str>>>,

    // ---- Model ----
    #[arg(
        long,
        default_value_t = 10,
        help = "Number of latent topics",
        long_help = "Number of latent topics K, shared between RNA and ATAC.\n\
                     Controls the rank of the topic proportion matrix theta[K,N]\n\
                     and the ATAC dictionary beta[P,K]."
    )]
    n_topics: usize,

    #[arg(
        long,
        default_value_t = 3,
        help = "SER components per gene",
        long_help = "Number of Single Effect Regression (SER) components\n\
                     in the SuSiE model for gene-peak linkage.\n\
                     Each component selects one peak per gene via softmax.\n\
                     Sets the maximum number of causal peaks per gene."
    )]
    n_ser_components: usize,

    #[arg(
        long,
        default_value_t = 50,
        help = "Max cis-candidates per gene",
        long_help = "Number of top correlated peaks to consider per gene.\n\
                     Peaks are ranked by absolute Pearson correlation\n\
                     with gene expression across pseudobulk samples.\n\
                     Smaller values improve sparsity but may miss true links."
    )]
    max_cis: usize,

    // ---- Collapsing ----
    #[arg(
        long,
        default_value_t = 64,
        help = "Projection dimension",
        long_help = "Dimension of the shared random projection for\n\
                     cell grouping across both modalities.\n\
                     Higher values preserve more cell-cell distance structure."
    )]
    proj_dim: usize,

    #[arg(
        long,
        default_value_t = 14,
        help = "Sort dimension",
        long_help = "Binary partitioning dimension for pseudobulk grouping.\n\
                     Produces up to 2^sort_dim super-cell groups.\n\
                     Higher values give more samples but noisier estimates."
    )]
    sort_dim: usize,

    #[arg(
        long,
        default_value_t = 2,
        help = "Coarsening levels",
        long_help = "Number of hierarchical coarsening levels.\n\
                     Level 0 is coarsest, last level is finest.\n\
                     Training uses the finest level."
    )]
    num_levels: usize,

    // ---- Training ----
    #[arg(
        long,
        default_value_t = 100,
        help = "Training epochs",
        long_help = "Total training epochs, split equally between:\n\
                     Stage 1 (ATAC-only, learns dictionary beta) and\n\
                     Stage 2 (joint RNA+ATAC, learns linkage M)."
    )]
    epochs: usize,

    #[arg(
        long,
        default_value_t = 0.001,
        help = "Learning rate",
        long_help = "AdamW learning rate for all parameters."
    )]
    learning_rate: f64,

    #[arg(
        long,
        default_value_t = 256,
        help = "Minibatch size",
        long_help = "Number of pseudobulk samples per minibatch.\n\
                     Clamped to total sample count if larger."
    )]
    minibatch_size: usize,

    // ---- Output ----
    #[arg(
        long,
        short,
        required = true,
        help = "Output prefix",
        long_help = "Output prefix for all result files.\n\
                     Produces: {out}.atac_dict.parquet, {out}.rna_dict.parquet,\n\
                     {out}.linkage.parquet, {out}.log_beta.parquet,\n\
                     {out}.prop.parquet"
    )]
    out: Box<str>,
}

pub fn fit_topic_model(args: &FitTopicArgs) -> anyhow::Result<()> {
    // ---- 1. Load paired RNA + ATAC data ----
    let mut paired = load_paired_data(
        &args.rna_files,
        &args.atac_files,
        args.batch_files.as_deref(),
    )?;

    let ncells = paired.data_stack.num_columns()?;
    let block_size = Some(ncells.min(4096));

    // ---- 2. Shared random projection ----
    info!(
        "Random projection (dim={}, {} cells)...",
        args.proj_dim, ncells
    );

    let proj = paired.data_stack.project_columns_with_batch_correction(
        args.proj_dim,
        block_size,
        Some(&paired.batch_membership),
    )?;

    // ---- 3. Multi-level collapsing ----
    let collapsed: Vec<Vec<CollapsedOut>> = paired
        .data_stack
        .collapse_columns_multilevel_vec(
            &proj.proj,
            &paired.batch_membership,
            &MultilevelParams {
                knn_super_cells: DEFAULT_KNN,
                num_levels: args.num_levels,
                sort_dim: args.sort_dim,
                num_opt_iter: DEFAULT_OPT_ITER,
            },
        )?
        .into_iter()
        .rev()
        .collect();

    for (level, ld) in collapsed.iter().enumerate() {
        info!(
            "Level {}: RNA {}x{}, ATAC {}x{}",
            level,
            ld[0].mu_observed.nrows(),
            ld[0].mu_observed.ncols(),
            ld[1].mu_observed.nrows(),
            ld[1].mu_observed.ncols(),
        );
    }

    let n_genes = collapsed[0][0].mu_observed.nrows();
    let n_peaks = collapsed[0][1].mu_observed.nrows();

    // ---- 4. Build cis-mask by correlation ----
    let dev = Device::Cpu;
    let rna_mat = collapsed.last().unwrap()[0].mu_observed.posterior_mean();
    let atac_mat = collapsed.last().unwrap()[1].mu_observed.posterior_mean();
    let (cis_indices, cis_mask) =
        build_cis_mask_by_correlation(rna_mat, atac_mat, args.max_cis, &dev)?;

    // ---- 5. Model setup ----
    let varmap = VarMap::new();
    let vs = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &dev);

    // Per-level decoders (independent params, keyed by level).
    // Currently all levels have the same feature dims; when feature
    // coarsening is added, each decoder will have its own D_l.
    let num_levels = collapsed.len();
    let decoders: Vec<ChickpeaDecoder> = (0..num_levels)
        .map(|i| {
            ChickpeaDecoder::new(
                DecoderArgs {
                    n_genes,
                    n_peaks,
                    n_topics: args.n_topics,
                    n_ser_components: args.n_ser_components,
                    cis_indices: cis_indices.clone(),
                    cis_mask: cis_mask.clone(),
                },
                vs.pp(format!("dec_{i}")),
            )
            .expect("decoder creation")
        })
        .collect();

    // Single shared encoder (full features across all levels)
    let encoder = ChickpeaEncoder::new(n_genes, n_peaks, args.n_topics, &[128], vs.pp("encoder"))?;

    let mut adam = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: args.learning_rate,
            ..Default::default()
        },
    )?;

    let total_samples: usize = collapsed.iter().map(|ld| ld[0].mu_observed.ncols()).sum();
    info!(
        "Model: {} topics, {} decoders, {} params, {} total samples",
        args.n_topics,
        num_levels,
        varmap.all_vars().len(),
        total_samples,
    );

    let finest_dec = decoders.last().unwrap();
    let c_max = finest_dec.susie.c_max;

    // Unified training: Stage 1 = ATAC-only, Stage 2 = joint RNA+ATAC
    let stage1_epochs = args.epochs / 2;
    let stage2_epochs = args.epochs - stage1_epochs;

    for (stage, stage_epochs, include_rna) in [
        ("S1 ATAC-only", stage1_epochs, false),
        ("S2 Joint", stage2_epochs, true),
    ] {
        info!(
            "=== {} ({} epochs x {} levels) ===",
            stage, stage_epochs, num_levels
        );

        for epoch in 0..stage_epochs {
            let (mut la_tot, mut lr_tot, mut kl_tot, mut n_tot) = (0f32, 0f32, 0f32, 0usize);

            for (level_data, dec) in collapsed.iter().zip(decoders.iter()) {
                let x_rna = sample_to_tensor(&level_data[0].mu_observed, &dev)?;
                let x_atac = sample_to_tensor(&level_data[1].mu_observed, &dev)?;
                let ns = x_rna.dim(0)?;
                let mb = args.minibatch_size.min(ns);

                for b in 0..n_batches(ns, mb) {
                    let (start, len) = mb_range(b, mb, ns);
                    let mb_rna = x_rna.narrow(0, start, len)?;
                    let mb_atac = x_atac.narrow(0, start, len)?;

                    // Compute SuSiE M once per minibatch (shared by encoder + decoder)
                    let m_gc = finest_dec.susie.forward(true)?;
                    let m_weights = m_gc.exp()?;

                    let (log_z, kl_enc) = encoder.forward(
                        &mb_rna,
                        &mb_atac,
                        &m_weights,
                        &finest_dec.flat_cis_indices,
                        c_max,
                        true,
                    )?;

                    let llik_atac = dec.forward_atac(&log_z, &mb_atac)?;
                    let mut loss = (&kl_enc - &llik_atac)?.mean_all()?;

                    if include_rna {
                        let llik_rna = dec.forward_rna(&log_z, &mb_rna, &m_gc)?;
                        loss = (&loss - llik_rna.mean_all()?)?;
                        lr_tot += llik_rna.sum_all()?.to_scalar::<f32>()?;
                    }

                    adam.backward_step(&loss)?;
                    la_tot += llik_atac.sum_all()?.to_scalar::<f32>()?;
                    kl_tot += kl_enc.sum_all()?.to_scalar::<f32>()?;
                }
                n_tot += ns;
            }

            if epoch % 20 == 0 || epoch == stage_epochs - 1 {
                let ns = n_tot as f32;
                if include_rna {
                    info!(
                        "  S2 {}/{}: llik_rna={:.1}, llik_atac={:.1}, kl={:.1}",
                        epoch + 1,
                        stage_epochs,
                        lr_tot / ns,
                        la_tot / ns,
                        kl_tot / ns
                    );
                } else {
                    info!(
                        "  S1 {}/{}: llik_atac={:.1}, kl={:.1}",
                        epoch + 1,
                        stage_epochs,
                        la_tot / ns,
                        kl_tot / ns
                    );
                }
            }
        }
    }

    // ---- 7. Save outputs (from finest decoder) ----
    info!("Saving to {}.*", args.out);

    finest_dec
        .log_beta
        .exp()?
        .to_parquet(&format!("{}.atac_dict.parquet", args.out))?;
    finest_dec
        .rna_dictionary(false)?
        .to_parquet(&format!("{}.rna_dict.parquet", args.out))?;
    finest_dec
        .log_beta
        .to_parquet(&format!("{}.log_beta.parquet", args.out))?;

    save_linkage_parquet(
        &finest_dec.susie.pip()?,
        &finest_dec.susie.posterior_mean()?,
        &finest_dec.susie.posterior_var()?,
        &cis_indices,
        &format!("{}.linkage.parquet", args.out),
    )?;

    // Latent states from finest level (eval mode)
    {
        let finest = collapsed.last().unwrap();
        let x_rna = mean_to_tensor(&finest[0].mu_observed, &dev)?;
        let x_atac = mean_to_tensor(&finest[1].mu_observed, &dev)?;
        let m_weights = finest_dec.susie.forward(false)?.exp()?;
        let (log_z, _) = encoder.forward(
            &x_rna,
            &x_atac,
            &m_weights,
            &finest_dec.flat_cis_indices,
            c_max,
            false,
        )?;
        log_z
            .exp()?
            .to_parquet(&format!("{}.prop.parquet", args.out))?;
    }

    info!("Done.");
    Ok(())
}

// ---- Helpers ----

fn n_batches(n: usize, mb: usize) -> usize {
    n.div_ceil(mb)
}

fn mb_range(b: usize, mb: usize, n: usize) -> (usize, usize) {
    let start = b * mb;
    (start, (start + mb).min(n) - start)
}

fn sample_to_tensor(
    gamma: &matrix_param::dmatrix_gamma::GammaMatrix,
    dev: &Device,
) -> anyhow::Result<Tensor> {
    Ok(gamma
        .posterior_sample()?
        .transpose()
        .to_tensor(dev)?
        .contiguous()?)
}

fn mean_to_tensor(
    gamma: &matrix_param::dmatrix_gamma::GammaMatrix,
    dev: &Device,
) -> anyhow::Result<Tensor> {
    Ok(gamma
        .posterior_mean()
        .transpose()
        .to_tensor(dev)?
        .contiguous()?)
}

fn build_cis_mask_by_correlation(
    rna_mat: &nalgebra::DMatrix<f32>,
    atac_mat: &nalgebra::DMatrix<f32>,
    max_c: usize,
    dev: &Device,
) -> anyhow::Result<(Tensor, Tensor)> {
    let n_genes = rna_mat.nrows();
    let n_peaks = atac_mat.nrows();
    let n_samples = rna_mat.ncols();
    let c = max_c.min(n_peaks);

    info!(
        "Gene-peak correlations ({} x {} x {} samples)...",
        n_genes, n_peaks, n_samples
    );

    // Precompute centered+normalized peak vectors
    let mut peak_centered: Vec<Vec<f32>> = Vec::with_capacity(n_peaks);
    let mut peak_norms: Vec<f32> = Vec::with_capacity(n_peaks);
    for p in 0..n_peaks {
        let row: Vec<f32> = atac_mat.row(p).iter().copied().collect();
        let mean: f32 = row.iter().sum::<f32>() / n_samples as f32;
        let centered: Vec<f32> = row.iter().map(|&v| v - mean).collect();
        let norm: f32 = centered.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        peak_centered.push(centered);
        peak_norms.push(norm);
    }

    let mut all_indices: Vec<u32> = Vec::with_capacity(n_genes * c);

    for g in 0..n_genes {
        let row: Vec<f32> = rna_mat.row(g).iter().copied().collect();
        let mean: f32 = row.iter().sum::<f32>() / n_samples as f32;
        let centered: Vec<f32> = row.iter().map(|&v| v - mean).collect();
        let norm: f32 = centered.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);

        let mut corrs: Vec<(u32, f32)> = (0..n_peaks)
            .map(|p| {
                let dot: f32 = centered
                    .iter()
                    .zip(&peak_centered[p])
                    .map(|(a, b)| a * b)
                    .sum();
                (p as u32, (dot / (norm * peak_norms[p])).abs())
            })
            .collect();

        corrs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        all_indices.extend(corrs[..c].iter().map(|&(idx, _)| idx));
    }

    let cis_indices = Tensor::from_vec(all_indices, (n_genes, c), dev)?;
    let mask = Tensor::ones((n_genes, c), DType::F32, dev)?;
    info!("Cis-mask: {} genes x {} candidates", n_genes, c);
    Ok((cis_indices, mask))
}

fn save_linkage_parquet(
    pip: &Tensor,
    mean: &Tensor,
    var: &Tensor,
    cis_indices: &Tensor,
    path: &str,
) -> anyhow::Result<()> {
    let (n_genes, c_max) = (pip.dim(0)?, pip.dim(1)?);
    let pip_data: Vec<f32> = pip.flatten_all()?.to_vec1()?;
    let mean_data: Vec<f32> = mean.flatten_all()?.to_vec1()?;
    let var_data: Vec<f32> = var.flatten_all()?.to_vec1()?;
    let idx_data: Vec<u32> = cis_indices.flatten_all()?.to_vec1()?;

    let n_entries = n_genes * c_max;
    let mut mat = nalgebra::DMatrix::<f32>::zeros(n_entries, 5);
    for g in 0..n_genes {
        for c in 0..c_max {
            let i = g * c_max + c;
            mat[(i, 0)] = g as f32;
            mat[(i, 1)] = idx_data[i] as f32;
            mat[(i, 2)] = pip_data[i];
            mat[(i, 3)] = mean_data[i];
            mat[(i, 4)] = var_data[i];
        }
    }
    mat.to_parquet(path)?;
    info!("Wrote {} ({} entries)", path, n_entries);
    Ok(())
}
