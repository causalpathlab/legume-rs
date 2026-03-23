use crate::common::*;
use data_beans::convert::try_open_or_convert;
use data_beans::sparse_io::{create_sparse_from_triplets, SparseIoBackend};
use rand::prelude::*;
use rand_distr::Poisson;

#[derive(Args, Debug)]
pub struct SimLinkArgs {
    #[arg(long, short, required = true, help = "Output prefix")]
    out: Box<str>,

    #[arg(long, default_value_t = 2000, help = "Number of genes")]
    n_genes: usize,

    #[arg(
        long,
        help = "Observed ATAC file (zarr/h5). When provided, uses real ATAC \
                data instead of de novo simulation. Peaks and cells are \
                determined by the file; --n-peaks and --n-cells are ignored."
    )]
    atac_file: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 10000,
        help = "Number of ATAC peaks (de novo mode)"
    )]
    n_peaks: usize,

    #[arg(long, default_value_t = 5000, help = "Number of cells (de novo mode)")]
    n_cells: usize,

    #[arg(long, default_value_t = 10, help = "Number of latent topics")]
    n_topics: usize,

    #[arg(long, default_value_t = 3, help = "Causal peaks per gene")]
    n_causal_per_gene: usize,

    #[arg(
        long,
        default_value_t = 0.3,
        help = "Fraction of genes with causal links"
    )]
    linked_gene_fraction: f32,

    #[arg(long, default_value_t = 0.5, help = "Causal effect size scale")]
    effect_scale: f32,

    #[arg(long, default_value_t = 5000, help = "RNA sequencing depth per cell")]
    depth_rna: usize,

    #[arg(
        long,
        default_value_t = 2000,
        help = "ATAC sequencing depth per cell (de novo mode)"
    )]
    depth_atac: usize,

    #[arg(long, default_value_t = 42, help = "Random seed")]
    rseed: u64,

    #[arg(long, default_value = "zarr", help = "Output backend: zarr or h5")]
    backend: String,
}

pub fn run_sim_link(args: &SimLinkArgs) -> anyhow::Result<()> {
    let g = args.n_genes;
    let k = args.n_topics;
    let mut rng = StdRng::seed_from_u64(args.rseed);

    let backend = match args.backend.as_str() {
        "h5" | "hdf5" => SparseIoBackend::HDF5,
        _ => SparseIoBackend::Zarr,
    };

    // ---- ATAC: observed or de novo ----
    let (atac_dense, p, n, observed_atac) = if let Some(ref atac_file) = args.atac_file {
        let (dense, p, n) = load_observed_atac(atac_file)?;
        (dense, p, n, true)
    } else {
        let p = args.n_peaks;
        let n = args.n_cells;
        (Mat::zeros(0, 0), p, n, false) // placeholder, filled below
    };

    info!(
        "Simulating: {} genes, {} peaks, {} cells, {} topics",
        g, p, n, k
    );

    // Shared topic proportions for both modalities (de novo)
    let theta_kn = sample_topic_proportions(k, n, &mut rng)?;

    let atac_dense = if observed_atac {
        atac_dense
    } else {
        info!("De novo ATAC: {} peaks, {} cells", p, k);
        let beta_atac = sample_dictionary(p, k, &mut rng);
        let atac_triplets = sample_poisson_counts(&beta_atac, &theta_kn, args.depth_atac, &mut rng);
        info!("ATAC: {} non-zeros", atac_triplets.len());

        let atac_path = format!("{}.atac.{}", args.out, args.backend);
        create_sparse_from_triplets(
            &atac_triplets,
            (p, n, atac_triplets.len()),
            Some(&atac_path),
            Some(&backend),
        )?;
        info!("Wrote ATAC to {}", atac_path);

        triplets_to_dense(&atac_triplets, p, n)
    };

    // ---- Causal links C[G × P] ----
    let n_linked = (g as f32 * args.linked_gene_fraction) as usize;
    let (causal_genes, causal_peaks, causal_effects) = sample_causal_links(
        g,
        p,
        n_linked,
        args.n_causal_per_gene,
        args.effect_scale,
        &mut rng,
    );
    info!(
        "{} linked genes, {} total causal links",
        n_linked,
        causal_effects.len()
    );

    // ---- RNA with causal modulation ----
    let beta_rna = sample_dictionary(g, k, &mut rng);

    info!("Sampling RNA counts with causal modulation...");
    let rna_triplets = sample_rna_with_links(
        &beta_rna,
        &theta_kn,
        args.depth_rna,
        &causal_genes,
        &causal_peaks,
        &causal_effects,
        &atac_dense,
        &mut rng,
    );
    info!("RNA: {} non-zeros", rna_triplets.len());

    let rna_path = format!("{}.rna.{}", args.out, args.backend);
    create_sparse_from_triplets(
        &rna_triplets,
        (g, n, rna_triplets.len()),
        Some(&rna_path),
        Some(&backend),
    )?;
    info!("Wrote RNA to {}", rna_path);

    // ---- Outputs ----
    write_ground_truth(&causal_genes, &causal_peaks, &causal_effects, &args.out)?;
    write_names(&args.out, g, p, n)?;

    if observed_atac {
        info!("Done. RNA at {}.rna.{}", args.out, args.backend);
    } else {
        info!(
            "Done. Outputs at {}.{{rna,atac}}.{}",
            args.out, args.backend
        );
    }

    Ok(())
}

/// Load observed ATAC data from a sparse backend into a dense matrix.
fn load_observed_atac(atac_file: &str) -> anyhow::Result<(Mat, usize, usize)> {
    info!("Loading observed ATAC from {}", atac_file);
    let data = try_open_or_convert(atac_file)?;

    let p = data.num_rows().unwrap_or(0);
    let n = data.num_columns().unwrap_or(0);
    anyhow::ensure!(p > 0 && n > 0, "Empty ATAC file: {} rows × {} cols", p, n);

    info!("Observed ATAC: {} peaks × {} cells", p, n);

    let mut data_vec = SparseIoVec::new();
    data_vec.push(Arc::from(data), None)?;

    let block_size = 100;
    let mut atac_dense = Mat::zeros(p, n);
    for chunk_start in (0..n).step_by(block_size) {
        let chunk_end = (chunk_start + block_size).min(n);
        let cols = (chunk_start..chunk_end).collect::<Vec<_>>().into_iter();
        let block = data_vec.read_columns_dmatrix(cols)?;
        for (local_j, global_j) in (chunk_start..chunk_end).enumerate() {
            atac_dense
                .column_mut(global_j)
                .copy_from(&block.column(local_j));
        }
    }

    let nnz = atac_dense.iter().filter(|&&x| x > 0.0).count();
    info!("Loaded observed ATAC: {} nnz", nnz);

    Ok((atac_dense, p, n))
}

fn sample_topic_proportions(k: usize, n: usize, rng: &mut impl Rng) -> anyhow::Result<Mat> {
    use rand_distr::Gamma;

    // Manual Dirichlet: sample K independent Gamma(alpha, 1), normalize
    let alpha = 0.5f64;
    let gamma_dist = Gamma::new(alpha, 1.0)?;

    let mut theta = Mat::zeros(k, n);
    for j in 0..n {
        let mut sum = 0.0f64;
        for i in 0..k {
            let g: f64 = gamma_dist.sample(rng);
            theta[(i, j)] = g as f32;
            sum += g;
        }
        // Normalize to sum to 1
        for i in 0..k {
            theta[(i, j)] /= sum as f32;
        }
    }

    Ok(theta)
}

fn sample_dictionary(d: usize, k: usize, rng: &mut impl Rng) -> Mat {
    let normal = rand_distr::Normal::new(0.0f32, 1.0).unwrap();

    // Logits [D × K]
    let mut logits = Mat::zeros(d, k);
    for i in 0..d {
        for j in 0..k {
            logits[(i, j)] = normal.sample(rng);
        }
    }

    // Row-wise softmax → probability dictionary
    let mut beta = Mat::zeros(d, k);
    for i in 0..d {
        let max_val: f32 = (0..k)
            .map(|j| logits[(i, j)])
            .fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = (0..k).map(|j| (logits[(i, j)] - max_val).exp()).sum();
        for j in 0..k {
            beta[(i, j)] = (logits[(i, j)] - max_val).exp() / exp_sum;
        }
    }

    beta
}

/// Sample sparse causal links.
///
/// Returns (gene_indices, peak_indices, effect_sizes) where each triple
/// represents one causal link: gene[i] is affected by peak[j] with effect[k].
fn sample_causal_links(
    n_genes: usize,
    n_peaks: usize,
    n_linked_genes: usize,
    n_causal_per_gene: usize,
    effect_scale: f32,
    rng: &mut impl Rng,
) -> (Vec<usize>, Vec<usize>, Vec<f32>) {
    let mut genes = Vec::new();
    let mut peaks = Vec::new();
    let mut effects = Vec::new();

    // Choose which genes have links
    let mut gene_indices: Vec<usize> = (0..n_genes).collect();
    gene_indices.shuffle(rng);
    let linked_genes = &gene_indices[..n_linked_genes.min(n_genes)];

    let normal = rand_distr::Normal::new(0.0f32, effect_scale).unwrap();

    for &gi in linked_genes {
        let n_causal = n_causal_per_gene.min(n_peaks);
        let sampled = rand::seq::index::sample(rng, n_peaks, n_causal);

        for pi in sampled.into_iter() {
            let effect = normal.sample(rng).abs(); // positive effects only
            genes.push(gi);
            peaks.push(pi);
            effects.push(effect);
        }
    }

    (genes, peaks, effects)
}

/// Sample from Poisson with overflow guard.
fn sample_poisson_safe(rate: f32, rng: &mut impl Rng) -> f32 {
    if rate > 700.0 {
        rate.round()
    } else {
        Poisson::new(rate as f64)
            .map(|d| d.sample(rng) as f32)
            .unwrap_or(0.0)
    }
}

/// Compute topic model rate: depth * β[d,:] · θ[:,n].
fn topic_rate(beta: &Mat, theta: &Mat, row: usize, col: usize, k: usize, depth: f32) -> f32 {
    let mut rate = 0.0f32;
    for kk in 0..k {
        rate += beta[(row, kk)] * theta[(kk, col)];
    }
    rate * depth
}

/// Sample Poisson counts from a topic model: rate_dn = depth * β[d,:] · θ[:,n].
fn sample_poisson_counts(
    beta_dk: &Mat,
    theta_kn: &Mat,
    depth: usize,
    rng: &mut impl Rng,
) -> Vec<(u64, u64, f32)> {
    let d = beta_dk.nrows();
    let n = theta_kn.ncols();
    let k = beta_dk.ncols();
    let depth_f = depth as f32;

    let mut triplets = Vec::new();

    for j in 0..n {
        for i in 0..d {
            let rate = topic_rate(beta_dk, theta_kn, i, j, k, depth_f);
            if rate > 1e-6 {
                let count = sample_poisson_safe(rate, rng);
                if count > 0.0 {
                    triplets.push((i as u64, j as u64, count));
                }
            }
        }
    }

    triplets
}

/// Convert triplets to a dense matrix [D × N].
fn triplets_to_dense(triplets: &[(u64, u64, f32)], d: usize, n: usize) -> Mat {
    let mut mat = Mat::zeros(d, n);
    for &(row, col, val) in triplets {
        mat[(row as usize, col as usize)] += val;
    }
    mat
}

/// Sample RNA counts with causal modulation from ATAC.
#[allow(clippy::too_many_arguments)]
fn sample_rna_with_links(
    beta_rna: &Mat,
    theta_kn: &Mat,
    depth: usize,
    causal_genes: &[usize],
    causal_peaks: &[usize],
    causal_effects: &[f32],
    atac_dense: &Mat,
    rng: &mut impl Rng,
) -> Vec<(u64, u64, f32)> {
    let g = beta_rna.nrows();
    let n = theta_kn.ncols();
    let k = beta_rna.ncols();
    let depth_f = depth as f32;

    // Build per-gene causal link lookup
    let mut gene_links: Vec<Vec<(usize, f32)>> = vec![Vec::new(); g];
    for (idx, &gi) in causal_genes.iter().enumerate() {
        gene_links[gi].push((causal_peaks[idx], causal_effects[idx]));
    }

    let mut triplets = Vec::new();

    for j in 0..n {
        for (i, links) in gene_links.iter().enumerate() {
            let base_rate = topic_rate(beta_rna, theta_kn, i, j, k, depth_f);
            let mut log_rate = base_rate.max(1e-8).ln();

            for &(pi, effect) in links {
                log_rate += effect * (1.0 + atac_dense[(pi, j)]).ln();
            }

            let rate = log_rate.exp();
            if rate > 1e-6 {
                let count = sample_poisson_safe(rate, rng);
                if count > 0.0 {
                    triplets.push((i as u64, j as u64, count));
                }
            }
        }
    }

    triplets
}

fn write_ground_truth(
    causal_genes: &[usize],
    causal_peaks: &[usize],
    causal_effects: &[f32],
    out_prefix: &str,
) -> anyhow::Result<()> {
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;

    let path = format!("{}.ground_truth.tsv.gz", out_prefix);
    let file = std::fs::File::create(&path)?;
    let mut gz = GzEncoder::new(file, Compression::default());

    writeln!(gz, "gene\tpeak\teffect")?;
    for i in 0..causal_genes.len() {
        writeln!(
            gz,
            "gene_{}\tpeak_{}\t{:.6}",
            causal_genes[i], causal_peaks[i], causal_effects[i]
        )?;
    }
    gz.finish()?;

    info!("Wrote ground truth to {}", path);
    Ok(())
}

fn write_names(
    out_prefix: &str,
    n_genes: usize,
    n_peaks: usize,
    n_cells: usize,
) -> anyhow::Result<()> {
    use std::io::Write;

    // Gene names
    let path = format!("{}.gene_names.txt", out_prefix);
    let mut f = std::fs::File::create(&path)?;
    for i in 0..n_genes {
        writeln!(f, "gene_{}", i)?;
    }

    // Peak names
    let path = format!("{}.peak_names.txt", out_prefix);
    let mut f = std::fs::File::create(&path)?;
    for i in 0..n_peaks {
        writeln!(f, "peak_{}", i)?;
    }

    // Cell barcodes (shared between RNA and ATAC)
    let path = format!("{}.barcodes.txt", out_prefix);
    let mut f = std::fs::File::create(&path)?;
    for i in 0..n_cells {
        writeln!(f, "cell_{}", i)?;
    }

    Ok(())
}
