mod sample;

use crate::common::*;
use data_beans::sparse_io::{create_sparse_from_triplets, SparseIoBackend};
use genomic_data::coordinates::GeneTss;
use rand::prelude::*;

#[derive(Args, Debug)]
pub struct SimLinkArgs {
    #[arg(long, short, required = true, help = "Output prefix for all files")]
    out: Box<str>,

    #[arg(long, default_value_t = 2000, help = "Number of genes (G)")]
    n_genes: usize,

    #[arg(long, default_value_t = 10000, help = "Number of ATAC peaks (P)")]
    n_peaks: usize,

    #[arg(long, default_value_t = 5000, help = "Number of cells (N)")]
    n_cells: usize,

    #[arg(
        long,
        default_value_t = 10,
        help = "Coarse topics (K), shared by ATAC and RNA"
    )]
    n_topics: usize,

    #[arg(
        long,
        default_value_t = 1,
        help = "RNA subtypes per coarse topic; K_total = K × K_sub. Set 1 for flat"
    )]
    n_sub_topics: usize,

    #[arg(long, default_value_t = 3, help = "Causal peaks per linked gene")]
    n_causal_per_gene: usize,

    #[arg(
        long,
        default_value_t = 0.3,
        help = "Fraction of genes with causal peak links"
    )]
    linked_gene_fraction: f32,

    #[arg(long, default_value_t = 5000, help = "Baseline RNA depth per cell")]
    depth_rna: usize,

    #[arg(long, default_value_t = 2000, help = "Baseline ATAC depth per cell")]
    depth_atac: usize,

    #[arg(
        long,
        default_value_t = 0.5,
        help = "SD of log-normal per-cell depth noise (ATAC)"
    )]
    cell_sd_log_depth_atac: f32,

    #[arg(
        long,
        default_value_t = 0.5,
        help = "SD of log-normal per-cell depth noise (RNA)"
    )]
    cell_sd_log_depth_rna: f32,

    #[arg(
        long,
        default_value_t = 0.8,
        help = "PVE for coarse topic assignment. Higher = sharper clusters"
    )]
    pve_topic: f32,

    #[arg(
        long,
        default_value_t = 0.8,
        help = "PVE for subtype within coarse topic. Only used when K_sub > 1"
    )]
    pve_sub_topic: f32,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "Gene-topic effect SD; LogNormal modulation. 0 = disabled"
    )]
    gene_topic_sd: f32,

    #[arg(long, default_value_t = 42, help = "Random seed for reproducibility")]
    rseed: u64,

    #[arg(
        long,
        default_value = "zarr",
        help = "Sparse matrix backend: zarr or h5"
    )]
    backend: Box<str>,
}

pub fn run_sim_link(args: &SimLinkArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;

    let g = args.n_genes;
    let p = args.n_peaks;
    let n = args.n_cells;
    let k = args.n_topics;
    let k_sub = args.n_sub_topics;
    let k_total = k * k_sub;
    let mut rng = StdRng::seed_from_u64(args.rseed);

    let backend = match &*args.backend {
        "h5" | "hdf5" => SparseIoBackend::HDF5,
        _ => SparseIoBackend::Zarr,
    };

    info!(
        "Simulating: {} genes, {} peaks, {} cells, {} topics x {} subtypes = {} total (pve_coarse={}, pve_sub={}, gene_topic_sd={})",
        g, p, n, k, k_sub, k_total, args.pve_topic, args.pve_sub_topic, args.gene_topic_sd
    );

    // ---- Topic proportions (nested) ----
    let theta_seed: u64 = rng.next_u64();
    let (theta_full, theta_coarse) = sample::sample_nested_topic_proportions(
        k,
        k_sub,
        n,
        args.pve_topic,
        args.pve_sub_topic,
        theta_seed,
    );

    // ---- Dictionaries ----
    // β_ext[P, K_total] — full fine-grained dictionary
    let beta_ext = sample::sample_dictionary(p, k_total, &mut rng);
    // β_atac[P, K] — marginalized for ATAC (sums over subtypes)
    let beta_atac = sample::marginalize_dictionary(&beta_ext, k, k_sub);

    // ---- Names ----
    let peak_names = generate_peak_names(p);
    let gene_names = generate_indexed_names(g, "gene");
    let gene_coords = generate_gene_coords(g);
    let cell_names = generate_indexed_names(n, "cell");

    // ---- ATAC counts: β_atac × θ_coarse ----
    let rho = sample::sample_cell_depths(n, args.depth_atac, args.cell_sd_log_depth_atac, &mut rng);
    info!("Sampling ATAC counts: {} peaks x {} cells", p, n);
    let atac_seed: u64 = rng.next_u64();
    let atac_triplets =
        sample::sample_poisson_counts(&beta_atac, &theta_coarse, &rho, None, atac_seed);
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
    let (indicator_genes, indicator_peaks) = sample::sample_indicator_matrix(
        g,
        p,
        n_linked,
        args.n_causal_per_gene,
        N_CHROMOSOMES,
        &mut rng,
    );
    info!(
        "{} linked genes, {} total entries in M",
        n_linked,
        indicator_genes.len()
    );

    // ---- RNA counts: W × θ_full where W = M × β_ext ----
    let w_gk = sample::build_derived_dictionary(&indicator_genes, &indicator_peaks, &beta_ext, g);

    let gamma_gk = if args.gene_topic_sd > 0.0 {
        Some(sample::sample_gene_topic_effects(
            g,
            k_total,
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
        sample::sample_poisson_counts(&w_gk, &theta_full, &tau, gamma_gk.as_ref(), rna_seed);
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
    info!("Wrote ATAC dictionary (marginalized) to {}", dict_file);

    let prop_file = format!("{}.prop.parquet", args.out);
    theta_coarse.transpose().to_parquet_with_names(
        &prop_file,
        (Some(&cell_names), Some("cell")),
        None,
    )?;
    info!("Wrote coarse proportions to {}", prop_file);

    let derived_file = format!("{}.derived_dict.parquet", args.out);
    w_gk.to_parquet_with_names(&derived_file, (Some(&gene_names), Some("gene")), None)?;
    info!("Wrote derived dictionary to {}", derived_file);

    if k_sub > 1 {
        let ext_file = format!("{}.beta_ext.parquet", args.out);
        beta_ext.to_parquet_with_names(&ext_file, (Some(&peak_names), Some("peak")), None)?;
        info!("Wrote extended dictionary to {}", ext_file);

        let full_prop_file = format!("{}.theta_full.parquet", args.out);
        theta_full.transpose().to_parquet_with_names(
            &full_prop_file,
            (Some(&cell_names), Some("cell")),
            None,
        )?;
        info!("Wrote full (nested) proportions to {}", full_prop_file);
    }

    if let Some(ref gamma) = gamma_gk {
        let gamma_file = format!("{}.gamma.parquet", args.out);
        gamma.to_parquet_with_names(&gamma_file, (Some(&gene_names), Some("gene")), None)?;
        info!("Wrote gene-topic effects to {}", gamma_file);
    }

    // ---- Ground truth, names & gene annotations ----
    write_ground_truth(
        &indicator_genes,
        &indicator_peaks,
        &peak_names,
        &gene_names,
        &args.out,
    )?;
    write_names(&args.out, &peak_names, &gene_names, &cell_names)?;
    write_gene_coords(&gene_names, &gene_coords, &args.out)?;

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

/// Generate genomic coordinates for simulated genes.
///
/// Genes are placed on chromosomes (round-robin) at positions interleaved
/// with peak bins so that each gene has nearby peaks for a valid cis-window.
fn generate_gene_coords(n_genes: usize) -> Vec<GeneTss> {
    (0..n_genes)
        .map(|i| {
            let chr = (i % N_CHROMOSOMES) + 1;
            let gene_on_chr = i / N_CHROMOSOMES;
            let tss =
                gene_on_chr as i64 * (PEAK_BIN_WIDTH + PEAK_GAP) as i64 + PEAK_BIN_WIDTH as i64 / 2;
            GeneTss {
                chr: format!("chr{}", chr).into(),
                tss,
            }
        })
        .collect()
}

/// Write gene annotations: gene_name \t chr \t tss
fn write_gene_coords(
    gene_names: &[Box<str>],
    coords: &[GeneTss],
    out_prefix: &str,
) -> anyhow::Result<()> {
    use matrix_util::common_io::open_buf_writer;
    use std::io::Write;

    let path = format!("{}.gene_coords.tsv.gz", out_prefix);
    let mut writer = open_buf_writer(&path)?;

    writeln!(writer, "gene\tchr\ttss")?;
    for (name, coord) in gene_names.iter().zip(coords.iter()) {
        writeln!(writer, "{}\t{}\t{}", name, coord.chr, coord.tss)?;
    }

    info!("Wrote gene coordinates to {}", path);
    Ok(())
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
