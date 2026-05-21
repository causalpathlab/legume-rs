//! `peak-to-gene` CLI args and orchestrator.

use crate::common::*;
use crate::p2g::embed::{build_atac_embedding, cis_link_stats, project_gene, pve_adjust};
use crate::p2g::finemap::{finemap_gene, FinemapParams};
use crate::p2g::output::{write_bed, LinkRecord};
use crate::topic::cis_mask::load_gene_coords_tsv;
use crate::topic::input::load_paired_data;
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

    #[arg(long, help = "GFF/GTF annotation for gene TSS. Alternative to --gene-coords")]
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

pub fn run_peak_to_gene(args: &PeakToGeneArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;

    /* 1. Load paired RNA + ATAC */
    let mut paired = load_paired_data(&args.rna_files, &args.atac_files, args.batch_files.as_deref())?;
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
            num_levels: 1,
            sort_dim: args.sort_dim,
            num_opt_iter: DEFAULT_OPT_ITER,
            refine: Some(RefineParams::default()),
        },
    )?;
    let level = levels
        .last()
        .ok_or_else(|| anyhow::anyhow!("collapse produced no levels"))?;

    let rna_pb = pick_pseudobulk(&level[0], args.use_adjusted);
    let atac_pb = pick_pseudobulk(&level[1], args.use_adjusted);
    let s = rna_pb.ncols();
    info!(
        "Pseudobulk: RNA {}x{}, ATAC {}x{} ({} samples)",
        rna_pb.nrows(),
        s,
        atac_pb.nrows(),
        atac_pb.ncols(),
        s
    );
    if s < 50 {
        info!("warning: only {} pseudobulk samples; correlations may be unstable", s);
    }

    /* 3. Global ATAC embedding (peaks = W, samples = V) */
    info!("Building ATAC embedding (rank {})...", args.embedding_dim);
    let emb = build_atac_embedding(atac_pb, args.embedding_dim)?;
    info!("ATAC embedding: {} peaks x {} dims", emb.w.nrows(), emb.d);

    /* 4. Coordinates + gene TSS */
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

            // Project the gene into the ATAC embedding; marginal z + LD share norms.
            let gene_rate: Vec<f32> = rna_pb.row(g).iter().copied().collect();
            let proj = project_gene(&emb, &gene_rate);
            let (mut z, r) = cis_link_stats(&proj, &emb, &cis);
            if !args.no_pve_adjust {
                z.iter_mut().for_each(|zc| *zc = pve_adjust(*zc, s));
            }

            let params = FinemapParams {
                num_components: args.num_components,
                prior_var: args.prior_var,
            };
            let (pip, eff_mean, eff_std) = finemap_gene(&r, &z, &params);

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

    /* 6. Write */
    let path = format!("{}.results.bed.gz", args.out);
    write_bed(&mut records, args.pip_threshold, &path)?;
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
