use std::sync::Arc;

use anyhow::Result;
use clap::Args;
use log::info;
use nalgebra::DMatrix;
use rayon::prelude::*;
use rust_htslib::tpool::ThreadPool;

use data_beans::convert::try_open_or_convert;
use data_beans::sparse_io_vector::SparseIoVec;
use fagioli::genotype::{BedReader, GenomicRegion, GenotypeReader};
use fagioli::io::cell_annotations::{
    build_onehot_membership, infer_cell_annotations, read_cell_annotations,
    read_membership_proportions,
};
use fagioli::io::covariates::load_covariate_files;
use fagioli::io::gene_annotations::{load_bed_annotations, load_gtf};
use fagioli::io::results::{
    write_gene_summary, write_parameters, write_variant_results, VariantRow,
};
use fagioli::mapping::map_qtl_helpers::*;
use fagioli::mapping::pseudobulk::{collapse_pseudobulk, Membership};
use fagioli::sgvb::{fit_block_weighted, FitConfig, ModelType};
use matrix_util::common_io::basename;

#[derive(Args, Debug, Clone)]
pub struct MapQtlArgs {
    // ── SC input ──────────────────────────────────────────────────────────
    /// Single-cell count matrices (Zarr, HDF5, or mtx paths; multiple files supported)
    #[arg(long, num_args = 1..)]
    pub sc_backend_files: Vec<Box<str>>,

    /// Cell annotations file (TSV or TSV.GZ): cell_id, individual_id[, cell_type]
    #[arg(long)]
    pub cell_annotations: Option<Box<str>>,

    /// Soft membership proportions from parquet (alternative to hard cell-type column)
    #[arg(long)]
    pub membership_parquet: Option<Box<str>>,

    // ── Genotype ──────────────────────────────────────────────────────────
    /// PLINK BED file prefix (without .bed)
    #[arg(long)]
    pub bed_prefix: String,

    /// Chromosome
    #[arg(long)]
    pub chromosome: String,

    /// Left position bound (optional)
    #[arg(long)]
    pub left_bound: Option<u64>,

    /// Right position bound (optional)
    #[arg(long)]
    pub right_bound: Option<u64>,

    /// Max individuals to use from genotype file
    #[arg(long)]
    pub max_individuals: Option<usize>,

    // ── Gene annotations ──────────────────────────────────────────────────
    /// GTF/GFF file for gene annotations (cis-eQTL mode)
    #[arg(long)]
    pub gtf_file: Option<String>,

    /// BED file for gene annotations: chr, start, end, gene_id[, gene_name[, strand]]
    #[arg(long)]
    pub gene_bed_file: Option<String>,

    /// Cis window size in base pairs (default: 1Mb)
    #[arg(long, default_value = "1000000")]
    pub cis_window: u64,

    // ── Pseudobulk parameters ─────────────────────────────────────────────
    /// Gamma prior shape parameter
    #[arg(long, default_value = "1.0")]
    pub gamma_a0: f32,

    /// Gamma prior rate parameter
    #[arg(long, default_value = "1.0")]
    pub gamma_b0: f32,

    /// Minimum effective cell weight to include an individual-celltype pair
    #[arg(long, default_value = "1.0")]
    pub min_cell_weight: f32,

    // ── Model parameters ──────────────────────────────────────────────────
    /// Fine-mapping model: susie, bisusie, multilevel-susie
    #[arg(long, default_value = "susie")]
    pub model: String,

    /// Number of SuSiE/BiSuSiE components (L)
    #[arg(long, default_value = "10")]
    pub num_components: usize,

    /// Comma-separated prior variances for coordinate search
    #[arg(long, default_value = "0.01,0.05,0.1,0.2,0.5,1.0")]
    pub prior_var: String,

    // ── SGVB parameters ───────────────────────────────────────────────────
    /// Number of SGVB Monte Carlo samples
    #[arg(long, default_value = "20")]
    pub num_sgvb_samples: usize,

    /// AdamW learning rate
    #[arg(long, default_value = "0.01")]
    pub learning_rate: f64,

    /// Number of training iterations per gene
    #[arg(long, default_value = "500")]
    pub num_iterations: usize,

    /// Minibatch size (use minibatch when N > batch_size)
    #[arg(long, default_value = "1000")]
    pub batch_size: usize,

    /// Number of ELBO values to average for model selection
    #[arg(long, default_value = "50")]
    pub elbo_window: usize,

    /// Block size for MultiLevelSusieVar tree
    #[arg(long, default_value = "50")]
    pub ml_block_size: usize,

    // ── Empirical Bayes ───────────────────────────────────────────────────
    /// Enable cross-gene empirical Bayes for prior variance
    #[arg(long)]
    pub empirical_bayes: bool,

    // ── Covariates ──────────────────────────────────────────────────────
    /// Include cell-type composition covariates (default: true)
    #[arg(long, default_value = "true")]
    pub composition_covariates: bool,

    /// Additional covariate file(s) — TSV/CSV with individuals as rows.
    /// First column = individual ID, remaining columns = covariate values.
    /// Multiple files can be specified. Covariates are concatenated with
    /// composition covariates and centered before fitting.
    #[arg(long)]
    pub covariate_files: Vec<String>,

    // ── Misc ──────────────────────────────────────────────────────────────
    /// Random seed
    #[arg(long, default_value = "42")]
    pub seed: u64,

    /// Output prefix
    #[arg(short, long)]
    pub output: String,
}

pub fn map_qtl(args: &MapQtlArgs) -> Result<()> {
    info!("Starting map-qtl");

    let num_threads = rayon::current_num_threads().max(1) as u32;
    let tpool = ThreadPool::new(num_threads)?;
    info!("Using {} threads", num_threads);

    // ── Step 1: Open SC backend + cell annotations + membership ──────────
    let attach_data_name = args.sc_backend_files.len() > 1;
    let mut data_vec = SparseIoVec::new();
    for data_file in &args.sc_backend_files {
        info!("Importing data file: {}", data_file);
        let data = try_open_or_convert(data_file)?;
        let data_name = attach_data_name.then(|| basename(data_file)).transpose()?;
        data_vec.push(Arc::from(data), data_name)?;
    }

    let column_names = data_vec.column_names()?;

    let annotations = if let Some(path) = &args.cell_annotations {
        read_cell_annotations(path)?
    } else {
        infer_cell_annotations(&column_names)
    };

    // ── Step 2: Build membership ─────────────────────────────────────────
    let membership = if let Some(parquet_path) = &args.membership_parquet {
        let mut m = read_membership_proportions(parquet_path, &column_names)?;
        let has_negative = m.matrix.iter().any(|&v| v < 0.0);
        if has_negative {
            info!("Membership has negative values; applying softmax (logit → probability)");
            for i in 0..m.matrix.nrows() {
                let max_val = m
                    .matrix
                    .row(i)
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0f32;
                for j in 0..m.matrix.ncols() {
                    let e = (m.matrix[(i, j)] - max_val).exp();
                    m.matrix[(i, j)] = e;
                    sum_exp += e;
                }
                for j in 0..m.matrix.ncols() {
                    m.matrix[(i, j)] /= sum_exp;
                }
            }
        }
        m
    } else if let Some(ann_path) = &args.cell_annotations {
        build_onehot_membership(ann_path, &column_names)?
    } else {
        Membership {
            matrix: DMatrix::from_element(column_names.len(), 1, 1.0),
            cell_type_names: vec![Box::from("all")],
        }
    };

    // ── Step 3: Collapse pseudobulk ──────────────────────────────────────
    info!(
        "Collapsing pseudobulk with Poisson-Gamma (a0={}, b0={})...",
        args.gamma_a0, args.gamma_b0
    );
    let collapsed = collapse_pseudobulk(
        data_vec,
        &annotations,
        &membership,
        args.gamma_a0,
        args.gamma_b0,
    )?;

    let n_genes = collapsed.gene_names.len();
    let n_ct = collapsed.cell_type_names.len();
    let n_individuals = collapsed.individual_ids.len();

    info!(
        "Pseudobulk: {} genes, {} cell types, {} individuals",
        n_genes, n_ct, n_individuals
    );

    // ── Step 4: Read genotypes ───────────────────────────────────────────
    let region = GenomicRegion::new(
        Some(args.chromosome.clone()),
        args.left_bound,
        args.right_bound,
    );

    let mut reader = BedReader::new(&args.bed_prefix)?;
    let geno = reader.read(args.max_individuals, Some(region))?;
    let m_snps = geno.num_snps();

    info!(
        "Loaded genotypes: {} individuals x {} SNPs",
        geno.num_individuals(),
        m_snps
    );

    // ── Step 5: Match individuals ────────────────────────────────────────
    let matched = match_individuals(&collapsed.individual_ids, &geno.individual_ids);
    let n_matched = matched.pb_indices.len();
    info!(
        "Matched {}/{} individuals between pseudobulk and genotypes",
        n_matched, n_individuals
    );

    if n_matched < 10 {
        anyhow::bail!(
            "Too few matched individuals ({}) for fine-mapping",
            n_matched
        );
    }

    // ── Step 6: Covariates (composition + user-supplied) ─────────────────
    let composition = if args.composition_covariates {
        compute_composition_covariates(&collapsed, &matched.pb_indices, n_ct)
    } else {
        None
    };

    let covariates = if !args.covariate_files.is_empty() {
        let matched_ids: Vec<&str> = matched
            .pb_indices
            .iter()
            .map(|&i| collapsed.individual_ids[i].as_ref())
            .collect();
        let extra = load_covariate_files(&args.covariate_files, &matched_ids, n_matched)?;
        merge_covariates(composition, &extra, n_matched)
    } else {
        composition
    };

    // ── Step 7: Load gene annotations ────────────────────────────────────
    let gene_annot = if let Some(ref gtf_path) = args.gtf_file {
        Some(load_gtf(
            gtf_path,
            Some(&args.chromosome),
            args.left_bound,
            args.right_bound,
            args.cis_window,
        )?)
    } else if let Some(ref bed_path) = args.gene_bed_file {
        Some(load_bed_annotations(
            bed_path,
            Some(&args.chromosome),
            args.cis_window,
        )?)
    } else {
        info!("No gene annotations provided → trans mode: all SNPs for all genes");
        None
    };

    // ── Step 8: Build gene specs ─────────────────────────────────────────
    let gene_specs = build_gene_specs(gene_annot.as_ref(), &collapsed, &geno, m_snps, n_genes);
    let n_testable = gene_specs.len();
    info!(
        "Testing {} genes ({} mode)",
        n_testable,
        if gene_annot.is_some() { "cis" } else { "trans" }
    );

    if n_testable == 0 {
        anyhow::bail!("No testable genes found (check gene annotations vs pseudobulk gene names)");
    }

    // ── Build fit config ─────────────────────────────────────────────────
    let model_type: ModelType = args.model.parse()?;
    let prior_vars: Vec<f32> = args
        .prior_var
        .split(',')
        .map(|s| s.trim().parse::<f32>())
        .collect::<std::result::Result<Vec<_>, _>>()?;

    let fit_config = FitConfig {
        model_type,
        num_components: args.num_components,
        num_sgvb_samples: args.num_sgvb_samples,
        learning_rate: args.learning_rate,
        num_iterations: args.num_iterations,
        batch_size: args.batch_size,
        prior_vars,
        elbo_window: args.elbo_window,
        seed: args.seed,
        ml_block_size: args.ml_block_size,
    };

    info!(
        "Model: {:?}, L={}, prior_vars={:?}",
        model_type, args.num_components, &fit_config.prior_vars
    );

    // ── Step 9: Per-gene parallel fine-mapping ───────────────────────────
    info!("Starting per-gene fine-mapping ({} genes)", n_testable);

    let gene_results: Vec<GeneResult> = gene_specs
        .par_iter()
        .enumerate()
        .filter_map(|(spec_idx, spec)| {
            if spec.cis_indices.len() < 2 {
                return None;
            }

            let (y_g, v_g) = build_gene_phenotype(
                &collapsed,
                spec.gene_idx,
                &matched.pb_indices,
                n_ct,
                args.min_cell_weight,
            );

            let (x_g, valid_cis_indices) =
                build_cis_genotypes(&geno, &matched.geno_indices, &spec.cis_indices)?;

            let mut gene_config = fit_config.clone();
            gene_config.seed = fit_config.seed.wrapping_add(spec_idx as u64);

            let detailed =
                match fit_block_weighted(&x_g, &y_g, &v_g, covariates.as_ref(), &gene_config) {
                    Ok(d) => d,
                    Err(e) => {
                        log::warn!("Gene {} failed: {}", spec.gene_id, e);
                        return None;
                    }
                };

            let z_marginal = compute_marginal_z(&x_g, &y_g, &v_g);

            info!(
                "Gene {}/{}: {} ({} cis SNPs, avg_elbo={:.2})",
                spec_idx + 1,
                n_testable,
                spec.gene_id,
                x_g.ncols(),
                detailed.result.avg_elbo,
            );

            Some(GeneResult {
                gene_id: spec.gene_id.clone(),
                cis_snp_indices: valid_cis_indices,
                detailed,
                z_marginal,
            })
        })
        .collect();

    let n_fitted = gene_results.len();
    info!("Successfully fitted {} / {} genes", n_fitted, n_testable);

    // ── Step 10: Optional empirical Bayes ─────────────────────────────────
    let eb_weights = if args.empirical_bayes {
        compute_eb_weights(&gene_results)
    } else {
        None
    };

    // ── Step 11: Write output ────────────────────────────────────────────
    let variant_rows = build_qtl_variant_rows(
        &gene_results,
        &collapsed.cell_type_names,
        eb_weights.as_deref(),
    );
    write_variant_results(
        &format!("{}.results.bed.gz", args.output),
        &["gene_id", "cell_type"],
        &variant_rows,
        &geno,
        &tpool,
    )?;

    write_gene_summary(
        &format!("{}.gene_summary.tsv.gz", args.output),
        &gene_results,
        n_ct,
        &tpool,
    )?;

    let params = serde_json::json!({
        "command": "map-qtl",
        "sc_backend_files": args.sc_backend_files,
        "bed_prefix": args.bed_prefix,
        "chromosome": args.chromosome,
        "num_individuals_genotype": geno.num_individuals(),
        "num_snps": m_snps,
        "num_individuals_matched": n_matched,
        "num_genes_total": n_genes,
        "num_genes_tested": n_testable,
        "num_genes_fitted": n_fitted,
        "num_cell_types": n_ct,
        "cell_type_names": collapsed.cell_type_names,
        "gene_annotation_mode": if gene_annot.is_some() { "cis" } else { "trans" },
        "cis_window": args.cis_window,
        "model": args.model,
        "num_components": args.num_components,
        "prior_vars": &fit_config.prior_vars,
        "num_sgvb_samples": args.num_sgvb_samples,
        "learning_rate": args.learning_rate,
        "num_iterations": args.num_iterations,
        "batch_size": args.batch_size,
        "elbo_window": args.elbo_window,
        "gamma_a0": args.gamma_a0,
        "gamma_b0": args.gamma_b0,
        "min_cell_weight": args.min_cell_weight,
        "empirical_bayes": args.empirical_bayes,
        "composition_covariates": args.composition_covariates,
        "covariate_files": args.covariate_files,
        "seed": args.seed,
    });
    write_parameters(&format!("{}.parameters.json", args.output), &params)?;

    info!("map-qtl completed successfully");
    Ok(())
}

// ── Statistical inference helpers ────────────────────────────────────────────

/// Compute weighted OLS marginal z-scores per (SNP, cell_type).
///
/// For each (SNP j, cell type k):
///   `β̂ = (x_j' W_k x_j)⁻¹ x_j' W_k y_k`, with `W_k = diag(1/V_ik)`
///   `SE = 1 / √(x_j' W_k x_j)`
///   `z_jk = β̂_jk / SE_jk`
fn compute_marginal_z(x: &DMatrix<f32>, y: &DMatrix<f32>, var: &DMatrix<f32>) -> DMatrix<f32> {
    let p = x.ncols();
    let k = y.ncols();
    let n = x.nrows();

    let mut z = DMatrix::<f32>::zeros(p, k);

    for j in 0..p {
        for ct in 0..k {
            let mut xwx = 0.0f32;
            let mut xwy = 0.0f32;

            for i in 0..n {
                let w = 1.0 / var[(i, ct)];
                let xv = x[(i, j)];
                xwx += xv * xv * w;
                xwy += xv * w * y[(i, ct)];
            }

            if xwx > 1e-10 {
                let beta = xwy / xwx;
                let se = 1.0 / xwx.sqrt();
                z[(j, ct)] = beta / se;
            }
        }
    }

    z
}

/// Compute empirical Bayes weights across genes for prior variance selection.
fn compute_eb_weights(gene_results: &[GeneResult]) -> Option<Vec<f32>> {
    if gene_results.len() < 2 {
        return None;
    }

    let n_priors = gene_results[0].detailed.per_prior_elbos.len();
    info!(
        "Computing empirical Bayes weights across {} genes",
        gene_results.len()
    );

    let mut sum_elbos = vec![0.0f64; n_priors];
    for gr in gene_results {
        for (v_idx, &elbo) in gr.detailed.per_prior_elbos.iter().enumerate() {
            sum_elbos[v_idx] += elbo as f64;
        }
    }

    let max_sum = sum_elbos.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = sum_elbos.iter().map(|e| (e - max_sum).exp()).collect();
    let sum_w: f64 = weights.iter().sum();
    let eb_w: Vec<f32> = weights.iter().map(|w| (*w / sum_w) as f32).collect();

    info!("EB weights: {:?}", eb_w);
    Some(eb_w)
}

/// Build variant result rows from gene-level fine-mapping output,
/// optionally applying empirical Bayes reweighting.
fn build_qtl_variant_rows(
    gene_results: &[GeneResult],
    cell_type_names: &[Box<str>],
    eb_weights: Option<&[f32]>,
) -> Vec<VariantRow> {
    let n_ct = cell_type_names.len();
    let mut rows = Vec::new();

    for gr in gene_results {
        let (pip, eff_mean, eff_std) = if let Some(eb_w) = eb_weights {
            eb_reweight(gr, eb_w, n_ct)
        } else {
            (
                gr.detailed.result.pip.clone(),
                gr.detailed.result.effect_mean.clone(),
                gr.detailed.result.effect_std.clone(),
            )
        };

        for (snp_j, &global_snp) in gr.cis_snp_indices.iter().enumerate() {
            for ct_idx in 0..n_ct {
                rows.push(VariantRow {
                    snp_idx: global_snp,
                    labels: vec![
                        Box::from(gr.gene_id.as_str()),
                        cell_type_names[ct_idx].clone(),
                    ],
                    pip: pip[(snp_j, ct_idx)],
                    effect_mean: eff_mean[(snp_j, ct_idx)],
                    effect_std: eff_std[(snp_j, ct_idx)],
                    z_marginal: gr.z_marginal[(snp_j, ct_idx)],
                });
            }
        }
    }

    rows
}
