use crate::chickpea_input::load_paired_data;
use crate::cis_mask::*;
use crate::coarsening::{coarsen_tensor, log_spaced_coarsenings};
use crate::common::*;
use crate::linkage::{precompute_expand_indices, rna_dictionary_from_m, save_linkage_parquet};
use crate::topic::decoder::DecoderArgs;
use crate::topic::{ChickpeaDecoder, ChickpeaEncoder, SuSiE};
use candle_util::candle_core::{DType, Device, Tensor};
use candle_util::candle_nn::{self, AdamW, Optimizer, ParamsAdamW, VarMap};
use data_beans_alg::collapse_data::{CollapsedOut, MultilevelParams};
use std::sync::atomic::{AtomicBool, Ordering};

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
        default_value_t = 1,
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
        long_help = "Maximum number of cis-candidate peaks per gene.\n\
                     When using distance-based filtering (--cis-window),\n\
                     peaks within the window are ranked by distance and\n\
                     the closest max_cis are selected.\n\
                     When using correlation-based filtering (--cis-window 0),\n\
                     peaks are ranked by absolute Pearson correlation."
    )]
    max_cis: usize,

    #[arg(
        long,
        default_value_t = 500000,
        help = "Cis-window size in bp",
        long_help = "Genomic distance window (in base pairs) around each gene's\n\
                     TSS for selecting candidate peaks.\n\
                     Only peaks on the same chromosome within ±cis_window bp\n\
                     of the gene TSS are considered.\n\
                     Set to 0 to fall back to correlation-based selection."
    )]
    cis_window: i64,

    #[arg(
        long,
        help = "Gene coordinates file (TSV: gene, chr, tss)",
        long_help = "Path to a TSV file with gene coordinates.\n\
                     Expected columns: gene, chr, tss (tab-separated, with header).\n\
                     Gene names must match the RNA matrix row names.\n\
                     Produced by sim-link as {out}.gene_coords.tsv.gz.\n\
                     Mutually exclusive with --gff-file."
    )]
    gene_coords: Option<Box<str>>,

    #[arg(
        long,
        help = "GFF/GTF file for gene TSS positions",
        long_help = "Path to a GFF3 or GTF annotation file.\n\
                     Gene TSS positions are extracted and matched to\n\
                     RNA matrix row names by gene_name or gene_id.\n\
                     Mutually exclusive with --gene-coords."
    )]
    gff_file: Option<Box<str>>,

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
        long_help = "Total training epochs."
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

    // ---- Feature coarsening ----
    #[arg(
        long,
        default_value_t = 0,
        help = "Max coarsened gene features (0 = disabled)",
        long_help = "Maximum number of coarsened gene modules at the finest level.\n\
                     Coarser levels use log-spaced smaller targets.\n\
                     Gene features are grouped by co-expression in pseudobulk.\n\
                     Set to 0 to disable feature coarsening (full resolution)."
    )]
    max_coarse_genes: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "Max coarsened peak features (0 = disabled)",
        long_help = "Maximum number of coarsened peak modules at the finest level.\n\
                     Coarser levels use log-spaced smaller targets.\n\
                     Peak features are grouped by co-accessibility in pseudobulk.\n\
                     Set to 0 to disable feature coarsening (full resolution)."
    )]
    max_coarse_peaks: usize,

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

    // ---- 4. Build cis-mask ----
    let dev = Device::Cpu;

    let gene_names = paired.data_stack.stack[0].row_names()?;
    let peak_names = paired.data_stack.stack[1].row_names()?;

    let (cis_indices, cis_mask) = if args.cis_window > 0 {
        // Distance-based: load gene coordinates from --gene-coords or --gff-file
        let gene_tss = if let Some(ref coords_path) = args.gene_coords {
            load_gene_coords_tsv(coords_path, &gene_names)?
        } else if let Some(ref gff_path) = args.gff_file {
            load_gene_tss(gff_path, &gene_names)?
        } else {
            anyhow::bail!(
                "--cis-window > 0 requires either --gene-coords or --gff-file \
                 to provide gene TSS positions"
            );
        };

        build_cis_mask_by_distance(&peak_names, &gene_tss, args.cis_window, args.max_cis, &dev)?
    } else {
        // Correlation-based fallback
        let rna_mat = collapsed.last().unwrap()[0].mu_observed.posterior_mean();
        let atac_mat = collapsed.last().unwrap()[1].mu_observed.posterior_mean();
        build_cis_mask_by_correlation(rna_mat, atac_mat, args.max_cis, &dev)?
    };

    // ---- 5. Feature coarsening (computed once before training) ----
    let num_levels = collapsed.len();
    let c_max = cis_indices.dim(1)?;
    let flat_cis_indices = cis_indices.flatten_all()?;

    let rna_coarsenings = log_spaced_coarsenings(
        collapsed.last().unwrap()[0].mu_observed.posterior_mean(),
        num_levels,
        args.max_coarse_genes,
    )?;
    let atac_coarsenings = log_spaced_coarsenings(
        collapsed.last().unwrap()[1].mu_observed.posterior_mean(),
        num_levels,
        args.max_coarse_peaks,
    )?;

    let level_dims: Vec<(usize, usize)> = (0..num_levels)
        .map(|i| {
            let dg = rna_coarsenings[i]
                .as_ref()
                .map(|c| c.num_coarse)
                .unwrap_or(n_genes);
            let dp = atac_coarsenings[i]
                .as_ref()
                .map(|c| c.num_coarse)
                .unwrap_or(n_peaks);
            (dg, dp)
        })
        .collect();

    for (i, &(dg, dp)) in level_dims.iter().enumerate() {
        info!("Level {} dims: {} genes, {} peaks", i, dg, dp);
    }

    // ---- 6. Model setup ----
    let varmap = VarMap::new();
    let vs = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &dev);

    // Per-level decoders with coarsened dims
    let decoders: Vec<ChickpeaDecoder> = level_dims
        .iter()
        .enumerate()
        .map(|(i, &(dg, dp))| {
            ChickpeaDecoder::new(
                DecoderArgs {
                    n_features_atac: dp,
                    n_features_rna: dg,
                    n_topics: args.n_topics,
                },
                vs.pp(format!("dec_{i}")),
            )
            .expect("decoder creation")
        })
        .collect();

    // Per-level SuSiE with coarsened dims (no cis window for coarsened levels)
    let susies: Vec<SuSiE> = level_dims
        .iter()
        .enumerate()
        .map(|(i, &(dg, dp))| {
            // Finest level uses cis_mask if no feature coarsening, else all-to-all
            let (n_cands, mask) = if rna_coarsenings[i].is_none() && atac_coarsenings[i].is_none() {
                (c_max, Some(cis_mask.clone()))
            } else {
                (dp, None)
            };
            SuSiE::new(
                dg,
                n_cands,
                args.n_ser_components,
                mask,
                vs.pp(format!("susie_{i}")),
            )
            .expect("SuSiE creation")
        })
        .collect();

    // Membership lookups for encoder weights (finest level)
    let finest_gene_members: Option<Vec<usize>> = rna_coarsenings
        .last()
        .and_then(|c| c.as_ref())
        .map(|c| c.fine_to_coarse.clone());
    let finest_peak_members: Option<Vec<usize>> = atac_coarsenings
        .last()
        .and_then(|c| c.as_ref())
        .map(|c| c.fine_to_coarse.clone());

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
        "Model: {} topics, {} levels (dims {:?}), {} params, {} total samples",
        args.n_topics,
        num_levels,
        level_dims,
        varmap.all_vars().len(),
        total_samples,
    );

    // ---- 7. Training ----
    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = Arc::clone(&stop);
        ctrlc::set_handler(move || {
            info!("Interrupt received — stopping training early and saving results...");
            stop.store(true, Ordering::SeqCst);
        })
        .expect("failed to set signal handler");
    }

    // Precompute encoder membership indices (constant across epochs)
    let enc_expand_indices: Option<Tensor> = match (&finest_gene_members, &finest_peak_members) {
        (Some(gm), Some(pm)) => Some(precompute_expand_indices(
            gm,
            pm,
            &flat_cis_indices,
            n_genes,
            c_max,
            &dev,
        )?),
        _ => None,
    };

    let n_epochs = args.epochs;
    info!("Training ({} epochs x {} levels)", n_epochs, num_levels);

    for epoch in 0..n_epochs {
        let (mut la_tot, mut lr_tot, mut kl_tot, mut n_tot) = (0f32, 0f32, 0f32, 0usize);

        // Finest SuSiE M for encoder (shared across levels)
        let finest_m = susies.last().unwrap().forward(true)?;
        let enc_m_weights = match &enc_expand_indices {
            Some(idx) => finest_m
                .flatten_all()?
                .index_select(idx, 0)?
                .reshape((n_genes, c_max))?
                .exp()?,
            None => finest_m.exp()?,
        };

        for (i, (level_data, dec)) in collapsed.iter().zip(decoders.iter()).enumerate() {
            let susie = &susies[i];
            let rna_fc = rna_coarsenings[i].as_ref();
            let atac_fc = atac_coarsenings[i].as_ref();

            let x_rna_full = sample_to_tensor(&level_data[0].mu_observed, &dev)?;
            let x_atac_full = sample_to_tensor(&level_data[1].mu_observed, &dev)?;
            let ns = x_rna_full.dim(0)?;
            let mb = args.minibatch_size.min(ns);

            let x_rna = coarsen_tensor(&x_rna_full, rna_fc)?;
            let x_atac = coarsen_tensor(&x_atac_full, atac_fc)?;

            for b in 0..n_batches(ns, mb) {
                let (start, len) = mb_range(b, mb, ns);
                let mb_rna_full = x_rna_full.narrow(0, start, len)?;
                let mb_atac_full = x_atac_full.narrow(0, start, len)?;
                let mb_rna = x_rna.narrow(0, start, len)?;
                let mb_atac = x_atac.narrow(0, start, len)?;

                let (log_z, kl_enc) = encoder.forward(
                    &mb_rna_full,
                    &mb_atac_full,
                    &enc_m_weights,
                    &flat_cis_indices,
                    c_max,
                    true,
                )?;

                let llik_atac = dec.forward_atac(&log_z, &mb_atac)?;

                // RNA dictionary from per-level SuSiE
                let m_gc = susie.forward(true)?;
                let w_gk = if rna_fc.is_some() || atac_fc.is_some() {
                    m_gc.exp()?.matmul(&dec.log_beta_atac.exp()?)?
                } else {
                    rna_dictionary_from_m(&m_gc, &dec.log_beta_atac, &flat_cis_indices)?
                };
                let llik_rna = dec.forward_rna(&log_z, &mb_rna, &w_gk)?;

                let loss = (&kl_enc - &llik_atac - &llik_rna)?.mean_all()?;
                adam.backward_step(&loss)?;

                la_tot += llik_atac.sum_all()?.to_scalar::<f32>()?;
                lr_tot += llik_rna.sum_all()?.to_scalar::<f32>()?;
                kl_tot += kl_enc.sum_all()?.to_scalar::<f32>()?;
            }
            n_tot += ns;
        }

        if epoch % 20 == 0 || epoch == n_epochs - 1 {
            let ns = n_tot as f32;
            info!(
                "  {}/{}: llik_rna={:.1}, llik_atac={:.1}, kl={:.1}",
                epoch + 1,
                n_epochs,
                lr_tot / ns,
                la_tot / ns,
                kl_tot / ns
            );
        }

        if stop.load(Ordering::SeqCst) {
            info!("Stopping early at epoch {}", epoch + 1);
            break;
        }
    }

    // ---- 8. Save outputs (from finest level) ----
    let finest_dec = decoders.last().unwrap();
    let finest_susie = susies.last().unwrap();

    info!("Saving to {}.*", args.out);

    finest_dec
        .log_beta_atac
        .exp()?
        .to_parquet(&format!("{}.atac_dict.parquet", args.out))?;

    let m_gc_eval = finest_susie.forward(false)?;
    let is_coarsened =
        rna_coarsenings.last().unwrap().is_some() || atac_coarsenings.last().unwrap().is_some();
    let w_gk_eval = if is_coarsened {
        m_gc_eval.exp()?.matmul(&finest_dec.log_beta_atac.exp()?)?
    } else {
        rna_dictionary_from_m(&m_gc_eval, &finest_dec.log_beta_atac, &flat_cis_indices)?
    };
    w_gk_eval.to_parquet(&format!("{}.rna_dict.parquet", args.out))?;

    finest_dec
        .log_beta_atac
        .to_parquet(&format!("{}.log_beta.parquet", args.out))?;

    let linkage_indices = if is_coarsened {
        let (dg, dp) = level_dims[num_levels - 1];
        let seq: Vec<u32> = (0..dg).flat_map(|_| (0..dp).map(|p| p as u32)).collect();
        Tensor::from_vec(seq, (dg, dp), &dev)?
    } else {
        cis_indices.clone()
    };
    save_linkage_parquet(
        &finest_susie.pip()?,
        &finest_susie.posterior_mean()?,
        &finest_susie.posterior_var()?,
        &linkage_indices,
        &format!("{}.linkage.parquet", args.out),
    )?;

    // Latent states from finest level (eval mode)
    {
        let finest = collapsed.last().unwrap();
        let x_rna = mean_to_tensor(&finest[0].mu_observed, &dev)?;
        let x_atac = mean_to_tensor(&finest[1].mu_observed, &dev)?;

        let enc_weights = match &enc_expand_indices {
            Some(idx) => m_gc_eval
                .flatten_all()?
                .index_select(idx, 0)?
                .reshape((n_genes, c_max))?
                .exp()?,
            None => m_gc_eval.exp()?,
        };

        let (log_z, _) = encoder.forward(
            &x_rna,
            &x_atac,
            &enc_weights,
            &flat_cis_indices,
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
