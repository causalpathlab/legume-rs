mod sample;

use crate::common::*;
use data_beans::sparse_io::{create_sparse_from_triplets, SparseIoBackend};
use rand::prelude::*;

#[derive(Args, Debug)]
pub struct SimLinkArgs {
    #[arg(
        long,
        short,
        required = true,
        help = "Output prefix",
        long_help = "Output prefix for all generated files.\n\
                     Produces: {out}.atac.{backend}, {out}.rna.{backend},\n\
                     {out}.dict.parquet, {out}.prop.parquet,\n\
                     {out}.derived_dict.parquet, {out}.ground_truth.tsv.gz,\n\
                     {out}.gene_names.txt, {out}.peak_names.txt, {out}.barcodes.txt"
    )]
    out: Box<str>,

    #[arg(long, default_value_t = 2000, help = "Number of genes")]
    n_genes: usize,

    #[arg(long, default_value_t = 10000, help = "Number of ATAC peaks")]
    n_peaks: usize,

    #[arg(long, default_value_t = 5000, help = "Number of cells")]
    n_cells: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "Number of latent topics",
        long_help = "Number of latent topics K.\n\
                     Shared between ATAC and RNA modalities.\n\
                     Controls the rank of the topic proportion matrix theta[K,N]."
    )]
    n_topics: usize,

    #[arg(
        long,
        default_value_t = 3,
        help = "Causal peaks per linked gene",
        long_help = "Number of ATAC peaks linked to each causal gene\n\
                     in the indicator matrix M[G,P].\n\
                     Peaks are sampled uniformly without replacement."
    )]
    n_causal_per_gene: usize,

    #[arg(
        long,
        default_value_t = 0.3,
        help = "Fraction of genes with causal peak links",
        long_help = "Fraction of genes that have non-zero rows in M.\n\
                     Remaining genes have W[g,:] = 0 and produce no RNA counts."
    )]
    linked_gene_fraction: f32,

    #[arg(
        long,
        default_value_t = 5000,
        help = "RNA sequencing depth per cell",
        long_help = "Baseline RNA depth d_X per cell.\n\
                     Actual per-cell depth is tau_i = d_X * exp(N(0, sigma_tau^2))."
    )]
    depth_rna: usize,

    #[arg(
        long,
        default_value_t = 2000,
        help = "ATAC sequencing depth per cell",
        long_help = "Baseline ATAC depth d_A per cell.\n\
                     Actual per-cell depth is rho_i = d_A * exp(N(0, sigma_rho^2))."
    )]
    depth_atac: usize,

    #[arg(
        long,
        default_value_t = 0.5,
        help = "Per-cell ATAC depth noise SD",
        long_help = "Standard deviation of per-cell log-normal ATAC depth noise.\n\
                     Controls sigma_rho in: ln(rho_i) ~ N(ln(d_A), sigma_rho^2).\n\
                     Larger values increase cell-to-cell depth variability."
    )]
    cell_sd_log_depth_atac: f32,

    #[arg(
        long,
        default_value_t = 0.5,
        help = "Per-cell RNA depth noise SD",
        long_help = "Standard deviation of per-cell log-normal RNA depth noise.\n\
                     Controls sigma_tau in: ln(tau_i) ~ N(ln(d_X), sigma_tau^2).\n\
                     Larger values increase cell-to-cell depth variability."
    )]
    cell_sd_log_depth_rna: f32,

    #[arg(
        long,
        default_value_t = 0.8,
        help = "PVE for topic structure",
        long_help = "Proportion of variance explained by topic assignments.\n\
                     Each cell is assigned a dominant topic; theta is then\n\
                     mixed as: pve * one_hot + (1-pve)/(K-1) * background.\n\
                     Higher PVE gives sharper, more separable cell clusters."
    )]
    pve_topic: f32,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "Gene-topic effect SD (0 = disabled)",
        long_help = "Standard deviation of log-normal gene-topic effects gamma[G,K].\n\
                     When > 0, gamma(g,t) ~ LogNormal(0, sd^2) modulates\n\
                     how much each topic contributes to each gene's expression,\n\
                     independent of the peak-gene mapping.\n\
                     Set to 0 to disable (gamma = 1 for all g,t)."
    )]
    gene_topic_sd: f32,

    #[arg(long, default_value_t = 42, help = "Random seed")]
    rseed: u64,

    #[arg(
        long,
        default_value = "zarr",
        help = "Sparse backend format",
        long_help = "Output format for sparse count matrices.\n\
                     Options: zarr (default), h5/hdf5."
    )]
    backend: String,
}

pub fn run_sim_link(args: &SimLinkArgs) -> anyhow::Result<()> {
    let g = args.n_genes;
    let p = args.n_peaks;
    let n = args.n_cells;
    let k = args.n_topics;
    let mut rng = StdRng::seed_from_u64(args.rseed);

    let backend = match args.backend.as_str() {
        "h5" | "hdf5" => SparseIoBackend::HDF5,
        _ => SparseIoBackend::Zarr,
    };

    info!(
        "Simulating: {} genes, {} peaks, {} cells, {} topics (pve={}, gene_topic_sd={})",
        g, p, n, k, args.pve_topic, args.gene_topic_sd
    );

    // ---- Shared parameters ----
    let theta_kn = sample::sample_topic_proportions(k, n, args.pve_topic, &mut rng);
    let beta_atac = sample::sample_dictionary(p, k, &mut rng);

    // ---- Names (needed for zarr metadata) ----
    let peak_names = generate_peak_names(p);
    let gene_names = generate_indexed_names(g, "gene");
    let cell_names = generate_indexed_names(n, "cell");

    // ---- ATAC counts ----
    let rho = sample::sample_cell_depths(n, args.depth_atac, args.cell_sd_log_depth_atac, &mut rng);
    info!("Sampling ATAC counts: {} peaks x {} cells", p, n);
    let atac_seed: u64 = rng.next_u64();
    let atac_triplets = sample::sample_poisson_counts(&beta_atac, &theta_kn, &rho, None, atac_seed);
    info!("ATAC: {} non-zeros", atac_triplets.len());

    let atac_path = format!("{}.atac.{}", args.out, args.backend);
    let mut atac_data = create_sparse_from_triplets(
        &atac_triplets,
        (p, n, atac_triplets.len()),
        Some(&atac_path),
        Some(&backend),
    )?;
    atac_data.register_row_names_vec(&peak_names);
    atac_data.register_column_names_vec(&cell_names);
    info!("Wrote ATAC to {}", atac_path);

    // ---- Indicator matrix M[G × P] ----
    let n_linked = (g as f32 * args.linked_gene_fraction) as usize;
    let (indicator_genes, indicator_peaks) =
        sample::sample_indicator_matrix(g, p, n_linked, args.n_causal_per_gene, &mut rng);
    info!(
        "{} linked genes, {} total entries in M",
        n_linked,
        indicator_genes.len()
    );

    // ---- RNA counts ----
    // W[G × K] = M × β_atac  (derived dictionary)
    let w_gk = sample::build_derived_dictionary(&indicator_genes, &indicator_peaks, &beta_atac, g);

    // Optional gene-topic effects γ[G × K]
    let gamma_gk = if args.gene_topic_sd > 0.0 {
        Some(sample::sample_gene_topic_effects(
            g,
            k,
            args.gene_topic_sd,
            &mut rng,
        ))
    } else {
        None
    };

    let tau = sample::sample_cell_depths(n, args.depth_rna, args.cell_sd_log_depth_rna, &mut rng);
    info!("Sampling RNA counts: {} genes x {} cells", g, n);
    let rna_seed: u64 = rng.next_u64();
    let rna_triplets =
        sample::sample_poisson_counts(&w_gk, &theta_kn, &tau, gamma_gk.as_ref(), rna_seed);
    info!("RNA: {} non-zeros", rna_triplets.len());

    let rna_path = format!("{}.rna.{}", args.out, args.backend);
    let mut rna_data = create_sparse_from_triplets(
        &rna_triplets,
        (g, n, rna_triplets.len()),
        Some(&rna_path),
        Some(&backend),
    )?;
    rna_data.register_row_names_vec(&gene_names);
    rna_data.register_column_names_vec(&cell_names);
    info!("Wrote RNA to {}", rna_path);

    // ---- Write parameters to parquet ----

    let dict_file = format!("{}.dict.parquet", args.out);
    beta_atac.to_parquet_with_names(&dict_file, (Some(&peak_names), Some("peak")), None)?;
    info!("Wrote dictionary to {}", dict_file);

    let prop_file = format!("{}.prop.parquet", args.out);
    theta_kn.transpose().to_parquet_with_names(
        &prop_file,
        (Some(&cell_names), Some("cell")),
        None,
    )?;
    info!("Wrote proportions to {}", prop_file);

    let derived_file = format!("{}.derived_dict.parquet", args.out);
    w_gk.to_parquet_with_names(&derived_file, (Some(&gene_names), Some("gene")), None)?;
    info!("Wrote derived dictionary to {}", derived_file);

    if let Some(ref gamma) = gamma_gk {
        let gamma_file = format!("{}.gamma.parquet", args.out);
        gamma.to_parquet_with_names(&gamma_file, (Some(&gene_names), Some("gene")), None)?;
        info!("Wrote gene-topic effects to {}", gamma_file);
    }

    // ---- Ground truth & names ----
    write_ground_truth(
        &indicator_genes,
        &indicator_peaks,
        &peak_names,
        &gene_names,
        &args.out,
    )?;
    write_names(&args.out, &peak_names, &gene_names, &cell_names)?;

    info!(
        "Done. Outputs at {}.{{rna,atac}}.{}",
        args.out, args.backend
    );

    Ok(())
}

const N_CHROMOSOMES: usize = 22;
const PEAK_BIN_WIDTH: usize = 500;
const PEAK_GAP: usize = 500;

/// Generate peak names in faba-style genomic coordinate format: chr{N}:{start}-{end}
fn generate_peak_names(n_peaks: usize) -> Vec<Box<str>> {
    (0..n_peaks)
        .map(|i| {
            let chr = (i % N_CHROMOSOMES) + 1;
            let start = (i / N_CHROMOSOMES) * (PEAK_BIN_WIDTH + PEAK_GAP);
            let end = start + PEAK_BIN_WIDTH;
            format!("chr{}:{}-{}", chr, start, end).into_boxed_str()
        })
        .collect()
}

fn generate_indexed_names(n: usize, prefix: &str) -> Vec<Box<str>> {
    (0..n)
        .map(|i| format!("{}_{}", prefix, i).into_boxed_str())
        .collect()
}

fn write_ground_truth(
    indicator_genes: &[usize],
    indicator_peaks: &[usize],
    peak_names: &[Box<str>],
    gene_names: &[Box<str>],
    out_prefix: &str,
) -> anyhow::Result<()> {
    use matrix_util::common_io::open_buf_writer;
    use std::io::Write;

    let path = format!("{}.ground_truth.tsv.gz", out_prefix);
    let mut writer = open_buf_writer(&path)?;

    writeln!(writer, "gene\tpeak")?;
    for i in 0..indicator_genes.len() {
        writeln!(
            writer,
            "{}\t{}",
            gene_names[indicator_genes[i]], peak_names[indicator_peaks[i]]
        )?;
    }

    info!("Wrote ground truth to {}", path);
    Ok(())
}

fn write_names(
    out_prefix: &str,
    peak_names: &Vec<Box<str>>,
    gene_names: &Vec<Box<str>>,
    cell_names: &Vec<Box<str>>,
) -> anyhow::Result<()> {
    use matrix_util::common_io::write_lines;

    write_lines(gene_names, &format!("{}.gene_names.txt", out_prefix))?;
    write_lines(peak_names, &format!("{}.peak_names.txt", out_prefix))?;
    write_lines(cell_names, &format!("{}.barcodes.txt", out_prefix))?;

    Ok(())
}
