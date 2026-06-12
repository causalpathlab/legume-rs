//! `peak-to-gene` CLI args and orchestrator.

use crate::common::*;
use crate::p2g::embed::{build_atac_embedding, cis_link_stats, project_gene, pve_adjust};
use crate::p2g::finemap::{finemap_gene, FinemapParams};
use crate::p2g::input::{load_gene_coords_tsv, load_paired_data};
use crate::p2g::knockoff::{knockoff_threshold, knockoff_w, KnockoffParams};
use crate::p2g::output::{write_bed, LinkRecord};
use crate::p2g::tmle::{centered_log1p, cis_link_stats_tmle, LocoConfounders, ModalityBlock};
use data_beans_alg::collapse_data::MultilevelParams;
use data_beans_alg::refine_multilevel::RefineParams;
use genomic_data::coordinates::{find_cis_peaks, load_gene_tss, parse_peak_coordinates};
use rayon::prelude::*;

#[derive(Args, Debug)]
pub struct PeakToGeneArgs {
    /* Input */
    #[arg(
        long,
        required = true,
        value_delimiter = ',',
        help = "RNA sparse matrices (zarr/h5), comma-separated"
    )]
    rna_files: Vec<Box<str>>,

    #[arg(
        long,
        required = true,
        value_delimiter = ',',
        help = "ATAC sparse matrices (zarr/h5), comma-separated"
    )]
    atac_files: Vec<Box<str>>,

    #[arg(
        long,
        value_delimiter = ',',
        help = "Batch label files, one per data file in RNA-then-ATAC order"
    )]
    batch_files: Option<Vec<Box<str>>>,

    /// Shared cell QC (on by default; `--no-qc` to disable). MAD outliers +
    /// near-empty floor, with optional `--qc-auto-cutoff` 2-means cell calling
    /// (`--qc-histogram` to show the nnz distribution). Dropped cells are
    /// removed from BOTH modalities up front, before projection / collapse.
    #[command(flatten)]
    qc: data_beans::qc_lib::QcArgs,

    /* Cis enumeration */
    #[arg(
        long,
        default_value_t = 500000,
        help = "Cis-window in bp around each gene TSS (peak midpoint distance)"
    )]
    cis_window: i64,

    #[arg(
        long,
        help = "Gene coordinates TSV (gene<TAB>chr<TAB>tss). From sim-link gene_coords.tsv.gz"
    )]
    gene_coords: Option<Box<str>>,

    #[arg(
        long,
        help = "GFF/GTF annotation for gene TSS. Alternative to --gene-coords"
    )]
    gff_file: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 200,
        help = "Max cis-candidate peaks per gene (nearest by distance)"
    )]
    max_cis: usize,

    /* Pseudobulk / collapse */
    #[arg(
        long,
        default_value_t = 64,
        help = "Random projection dimension for cell grouping"
    )]
    proj_dim: usize,

    #[arg(
        long,
        default_value_t = 14,
        help = "Binary sort dimension. Yields ~2^sort_dim pseudobulk samples"
    )]
    sort_dim: usize,

    #[arg(
        long,
        default_value_t = false,
        help = "Use batch-adjusted pseudobulk (mu_adjusted) when available"
    )]
    use_adjusted: bool,

    #[arg(
        long,
        default_value_t = 1,
        help = "Hierarchical refinement levels; refined finest level is used (1 = single level)"
    )]
    num_levels: usize,

    /* Embedding */
    #[arg(
        long,
        default_value_t = 50,
        help = "ATAC embedding rank d (rSVD of standardized log1p ATAC pseudobulk)"
    )]
    embedding_dim: usize,

    #[arg(
        long,
        default_value_t = false,
        help = "Disable PVE (winner's-curse) z-score shrinkage"
    )]
    no_pve_adjust: bool,

    /* TMLE deconfounding (LOCO topic adjustment) */
    #[arg(
        long,
        default_value_t = false,
        help = "Deconfound peak→gene z by LOCO topic adjustment\n\
                (residual-on-residual DML/TMLE)"
    )]
    tmle: bool,

    #[arg(
        long,
        default_value_t = 20,
        help = "Number of off-chromosome topic confounder factors m for --tmle"
    )]
    tmle_rank: usize,

    #[arg(
        long,
        default_value_t = false,
        help = "Use ATAC-only --tmle confounder instead of joint RNA+ATAC\n\
                co-embedding (tighter FDP, lower power)"
    )]
    tmle_atac_only: bool,

    /* Fine-mapping (SuSiE-RSS) */
    #[arg(
        long,
        default_value_t = 10,
        help = "Number of single-effect components L (max causal peaks per gene)"
    )]
    num_components: usize,

    #[arg(
        long,
        default_value_t = 5.0,
        help = "SuSiE prior effect-size variance (z-score scale)"
    )]
    prior_var: f64,

    /* Knockoff FDR (optional) */
    #[arg(
        long,
        default_value_t = 0.0,
        help = "Target FDR for knockoff-selected links (0 = off; PIPs only)"
    )]
    fdr: f64,

    #[arg(
        long,
        default_value_t = 0.05,
        help = "Knockoff LD ridge λ in R_λ = (1-λ)R + λI"
    )]
    ko_ridge: f64,

    #[arg(
        long,
        value_enum,
        default_value = "equi",
        help = "Knockoff diagonal s: equi (default), mvr, or me",
        long_help = "Knockoff diagonal s method: equi (equicorrelated, default), mvr\n\
                     (min-variance reconstructability), me (max entropy). For dense cis\n\
                     sets reduce --max-cis / raise --ko-ridge — the s-method does not\n\
                     rescue a rank-deficient LD matrix."
    )]
    ko_s: KoSMethod,

    #[arg(long, default_value_t = 42, help = "Random seed for knockoff sampling")]
    seed: u64,

    /* Output */
    #[arg(
        long,
        default_value_t = 0.0,
        help = "PIP threshold for the summary log line (all links are written)"
    )]
    pip_threshold: f32,

    #[arg(
        long,
        short,
        required = true,
        help = "Output prefix (produces {out}.results.bed.gz)"
    )]
    out: Box<str>,
}

/// CLI surface for [`matrix_util::knockoff::KnockoffS`].
#[derive(Clone, Copy, Debug, ValueEnum)]
enum KoSMethod {
    Equi,
    Mvr,
    Me,
}

impl From<KoSMethod> for matrix_util::knockoff::KnockoffS {
    fn from(m: KoSMethod) -> Self {
        match m {
            KoSMethod::Equi => Self::Equicorrelated,
            KoSMethod::Mvr => Self::Mvr,
            KoSMethod::Me => Self::Me,
        }
    }
}

pub fn run_peak_to_gene(args: &PeakToGeneArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;

    /* 1. Load paired RNA + ATAC */
    let mut paired = load_paired_data(
        &args.rna_files,
        &args.atac_files,
        args.batch_files.as_deref(),
    )?;

    /* 1b. Cell QC (on by default): drop ambient / outlier cells from BOTH
    modalities up front — before projection / collapse — so they never shape
    the pseudobulk or the peak-to-gene links. `--qc-auto-cutoff` adds 2-means
    cell calling; `--qc-histogram` prints the nnz distribution. */
    if let Some(cfg) = args.qc.to_config() {
        let report = data_beans::qc_lib::compute_qc_stack(&paired.data_stack, &cfg, None)?;
        info!(
            "cell QC: dropping {} / {} cells before peak-to-gene linkage",
            report.n_cells_dropped,
            report.train_keep.len(),
        );
        if report.n_cells_dropped > 0 {
            paired.data_stack.mask_columns_all(&report.train_keep)?;
            paired.batch_membership =
                data_beans::qc_lib::filter_by_keep(&paired.batch_membership, &report.train_keep);
        }
    }

    let gene_names = paired.data_stack.stack[0].row_names()?;
    let peak_names = paired.data_stack.stack[1].row_names()?;
    let n_genes = gene_names.len();

    /* 2. Project + collapse to a single pseudobulk level */
    let block_size: Option<usize> = None;
    info!(
        "Random projection (dim={}, {} cells)...",
        args.proj_dim,
        paired.data_stack.num_columns()?
    );
    let proj = paired.data_stack.project_columns_with_batch_correction(
        args.proj_dim,
        block_size,
        Some(&paired.batch_membership),
    )?;

    let levels = paired.data_stack.collapse_columns_multilevel_vec(
        &proj.proj,
        &paired.batch_membership,
        &MultilevelParams {
            knn_pb_samples: DEFAULT_KNN,
            num_levels: args.num_levels.max(1),
            sort_dim: args.sort_dim,
            num_opt_iter: DEFAULT_OPT_ITER,
            refine: Some(RefineParams::default()),
            output_calibration: matrix_param::traits::CalibrateTarget::All,
        },
    )?;
    if levels.is_empty() {
        anyhow::bail!("collapse produced no levels");
    }
    // Use the hierarchically-refined finest level (most pb samples). With
    // --num-levels > 1 the collapse refines pb assignments using the level
    // hierarchy (bottom-up coarsening + sibling-constrained refinement);
    // pooling levels as extra columns adds redundancy, not signal, so we take
    // the refined finest level only.
    let finest = levels
        .iter()
        .max_by_key(|lvl| pick_pseudobulk(&lvl[0], args.use_adjusted).ncols())
        .expect("levels is non-empty");
    let rna_pb = pick_pseudobulk(&finest[0], args.use_adjusted);
    let atac_pb = pick_pseudobulk(&finest[1], args.use_adjusted);
    let s = rna_pb.ncols();
    info!(
        "Pseudobulk: RNA {}x{}, ATAC {}x{} ({} refinement level(s), {} samples)",
        rna_pb.nrows(),
        s,
        atac_pb.nrows(),
        atac_pb.ncols(),
        levels.len(),
        s
    );
    if s < 50 {
        info!(
            "warning: only {} pseudobulk samples; correlations may be unstable",
            s
        );
    }

    /* 3. Coordinates + gene TSS */
    let peak_coords = parse_peak_coordinates(&peak_names);
    let gene_tss = if args.cis_window > 0 {
        if let Some(path) = &args.gene_coords {
            load_gene_coords_tsv(path, &gene_names)?
        } else if let Some(path) = &args.gff_file {
            load_gene_tss(path, &gene_names)?
        } else {
            anyhow::bail!("--cis-window > 0 requires either --gene-coords or --gff-file");
        }
    } else {
        anyhow::bail!("--cis-window must be > 0");
    };

    /* 4. Association model: shared ATAC embedding (default) or LOCO+TMLE
    deconfounded residual statistics. The embedding reads peak→gene off a
    low-rank topic space (sees only topic co-variation); TMLE residualizes
    gene + peak against an off-chromosome topic confounder so confounded
    bystanders collapse and identifiable cis links survive. */
    let emb;
    let tmle_state;
    let n_eff;
    if args.tmle {
        let peak_chr: Vec<Option<Box<str>>> = peak_coords
            .iter()
            .map(|c| c.as_ref().map(|co| co.chr.clone()))
            .collect();
        let gene_chr: Vec<Option<Box<str>>> = gene_tss
            .iter()
            .map(|t| t.as_ref().map(|x| x.chr.clone()))
            .collect();
        let mut blocks = vec![ModalityBlock {
            pb: atac_pb,
            chr: &peak_chr,
        }];
        if !args.tmle_atac_only {
            // Default joint RNA+ATAC confounder: a two-modality cell-state
            // estimate (more power). LOCO drops chr-c features in BOTH blocks
            // (off-chr genes too) so a chr-c gene's cis signal can't leak in.
            blocks.push(ModalityBlock {
                pb: rna_pb,
                chr: &gene_chr,
            });
        }
        info!(
            "Building LOCO {} confounders (rank {})...",
            if args.tmle_atac_only {
                "ATAC"
            } else {
                "joint RNA+ATAC"
            },
            args.tmle_rank
        );
        let loco = LocoConfounders::build(&blocks, args.tmle_rank)?;
        let m = loco.m;
        let peak_clog: Vec<DVec> = (0..peak_names.len())
            .into_par_iter()
            .map(|p| centered_log1p(&atac_pb.row(p).iter().copied().collect::<Vec<_>>()))
            .collect();
        info!("LOCO confounders ready (m={}, n_eff={})", m, s - m);
        emb = None;
        tmle_state = Some((loco, peak_clog));
        n_eff = s.saturating_sub(m);
    } else {
        info!("Building ATAC embedding (rank {})...", args.embedding_dim);
        let e = build_atac_embedding(atac_pb, args.embedding_dim)?;
        info!("ATAC embedding: {} peaks x {} dims", e.w.nrows(), e.d);
        emb = Some(e);
        tmle_state = None;
        n_eff = s;
    }

    let ko_params = KnockoffParams {
        ridge: args.ko_ridge,
        seed: args.seed,
        s_method: args.ko_s.into(),
    };
    let use_fdr = args.fdr > 0.0;

    /* 5. Per-gene fine-mapping (parallel over genes) */
    let per_gene: Vec<Vec<LinkRecord>> = (0..n_genes)
        .into_par_iter()
        .map(|g| {
            let Some(tss) = gene_tss[g].as_ref() else {
                return Vec::new();
            };
            let mut cis = find_cis_peaks(tss, &peak_coords, args.cis_window);
            if cis.is_empty() {
                return Vec::new();
            }
            let peak_mid = |p: usize| -> i64 {
                peak_coords[p]
                    .as_ref()
                    .map(|c| (c.start + c.end) / 2)
                    .unwrap_or(0)
            };
            if cis.len() > args.max_cis {
                cis.sort_by_key(|&p| (peak_mid(p) - tss.tss).abs());
                cis.truncate(args.max_cis);
            }

            // Marginal z + peak–peak LD R, via whichever estimator is active.
            let (z_raw, r) = if let Some((loco, peak_clog)) = tmle_state.as_ref() {
                let gene_clog = centered_log1p(&rna_pb.row(g).iter().copied().collect::<Vec<_>>());
                let w = loco.get(&tss.chr);
                let (z, r, _n_eff) = cis_link_stats_tmle(&gene_clog, peak_clog, &cis, w);
                (z, r)
            } else {
                let emb = emb.as_ref().expect("embedding built when !tmle");
                let gene_rate: Vec<f32> = rna_pb.row(g).iter().copied().collect();
                let proj = project_gene(emb, &gene_rate);
                cis_link_stats(&proj, emb, &cis, n_eff as f64)
            };
            let mut z = z_raw.clone();
            if !args.no_pve_adjust {
                z.iter_mut().for_each(|zc| *zc = pve_adjust(*zc, n_eff));
            }

            let params = FinemapParams {
                num_components: args.num_components,
                prior_var: args.prior_var,
            };
            let (pip, eff_mean, eff_std) = finemap_gene(&r, &z, &params);

            // Knockoff importance W from the raw (pre-PVE) z, keeping the N(0,R)
            // null. The pooled FDR threshold is applied after all genes score.
            let w_stat = if use_fdr {
                knockoff_w(&z_raw, &r, &ko_params, g)
            } else {
                vec![f32::NAN; cis.len()]
            };

            cis.iter()
                .enumerate()
                .map(|(j, &p)| {
                    let (chr, start, end) = match peak_coords[p].as_ref() {
                        Some(co) => (co.chr.clone(), co.start, co.end),
                        None => (".".into(), 0i64, 0i64),
                    };
                    let mid = (start + end) / 2;
                    LinkRecord {
                        chr,
                        start,
                        end,
                        peak_id: peak_names[p].clone(),
                        gene_id: gene_names[g].clone(),
                        pip: pip[j],
                        effect_mean: eff_mean[j],
                        effect_std: eff_std[j],
                        z: z[j],
                        distance: (mid - tss.tss).abs(),
                        w_stat: w_stat[j],
                        selected: false,
                    }
                })
                .collect()
        })
        .collect();

    let n_skipped = per_gene.iter().filter(|v| v.is_empty()).count();
    let mut records: Vec<LinkRecord> = per_gene.into_iter().flatten().collect();
    info!(
        "Fine-mapped {} genes ({} skipped: no TSS / no cis peaks)",
        n_genes - n_skipped,
        n_skipped
    );

    /* 6. Pooled knockoff filter (genome-wide FDR over links) */
    if use_fdr {
        let all_w: Vec<f32> = records.iter().map(|r| r.w_stat).collect();
        let tau = knockoff_threshold(&all_w, args.fdr);
        let mut n_sel = 0usize;
        for rec in records.iter_mut() {
            if rec.w_stat.is_finite() && rec.w_stat >= tau {
                rec.selected = true;
                n_sel += 1;
            }
        }
        info!(
            "Knockoff filter (FDR={:.3}): threshold W>={:.4}, {} links selected",
            args.fdr, tau, n_sel
        );
    }

    /* 7. Write */
    let path = format!("{}.results.bed.gz", args.out);
    write_bed(&mut records, args.pip_threshold, use_fdr, &path)?;
    Ok(())
}

/// Pick `mu_adjusted` (batch-corrected) when requested and available, else
/// `mu_observed`. Returns the [features, samples] posterior-mean intensities.
fn pick_pseudobulk(co: &data_beans_alg::collapse_data::CollapsedOut, use_adjusted: bool) -> &Mat {
    if use_adjusted {
        if let Some(adj) = co.mu_adjusted.as_ref() {
            return adj.posterior_mean();
        }
    }
    co.mu_observed.posterior_mean()
}
