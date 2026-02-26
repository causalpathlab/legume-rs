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
use fagioli::sgvb::{fit_block_weighted, ComputeDevice, FitConfig, ModelType};
use matrix_util::common_io::basename;

#[derive(Args, Debug, Clone)]
pub struct MapQtlArgs {
    // ── Single-cell input ────────────────────────────────────────────────
    #[arg(long, num_args = 1.., help = "Single-cell count matrices (Zarr, HDF5, or mtx; multiple supported)")]
    pub sc_backend_files: Vec<Box<str>>,

    #[arg(
        long,
        help = "Cell annotations TSV: cell_id, individual_id[, cell_type]",
        long_help = "Cell annotations file (TSV or TSV.GZ).\n\
            Columns: cell_id, individual_id, and optionally cell_type.\n\
            If cell_type column is present, hard cell-type assignments are used.\n\
            Use --membership-parquet for soft assignments instead."
    )]
    pub cell_annotations: Option<Box<str>>,

    #[arg(
        long,
        help = "Soft membership proportions (parquet, alternative to cell_type column)"
    )]
    pub membership_parquet: Option<Box<str>>,

    // ── Genotype ─────────────────────────────────────────────────────────
    #[arg(long, help = "PLINK BED file prefix (without .bed/.bim/.fam)")]
    pub bed_prefix: String,

    #[arg(long, help = "Chromosome to analyze")]
    pub chromosome: String,

    #[arg(long, help = "Left genomic position bound (bp)")]
    pub left_bound: Option<u64>,

    #[arg(long, help = "Right genomic position bound (bp)")]
    pub right_bound: Option<u64>,

    #[arg(long, help = "Max individuals to use from genotype file")]
    pub max_individuals: Option<usize>,

    // ── Gene annotations ─────────────────────────────────────────────────
    #[arg(long, help = "GTF/GFF gene annotation file for cis-eQTL windows")]
    pub gtf_file: Option<String>,

    #[arg(
        long,
        help = "BED gene annotation file: chr, start, end, gene_id[, name[, strand]]"
    )]
    pub gene_bed_file: Option<String>,

    #[arg(
        long,
        default_value = "1000000",
        help = "Cis window size in bp (default: 1Mb)"
    )]
    pub cis_window: u64,

    // ── Pseudobulk parameters ────────────────────────────────────────────
    #[arg(
        long,
        default_value = "1.0",
        help = "Gamma prior shape (a0) for pseudobulk"
    )]
    pub gamma_a0: f32,

    #[arg(
        long,
        default_value = "1.0",
        help = "Gamma prior rate (b0) for pseudobulk"
    )]
    pub gamma_b0: f32,

    #[arg(
        long,
        default_value = "1.0",
        help = "Min effective cell weight per individual-celltype pair"
    )]
    pub min_cell_weight: f32,

    // ── Model parameters ─────────────────────────────────────────────────
    #[arg(
        long,
        default_value = "susie",
        help = "Fine-mapping model: susie, bisusie, multilevel-susie"
    )]
    pub model: String,

    #[arg(
        long,
        default_value = "10",
        help = "Number of SuSiE/BiSuSiE components (L)"
    )]
    pub num_components: usize,

    #[arg(
        long,
        default_value = "0.05,0.1,0.12,0.15,0.18,0.2,0.25,0.3,0.5",
        help = "Comma-separated prior variances for coordinate search"
    )]
    pub prior_var: String,

    // ── SGVB training ────────────────────────────────────────────────────
    #[arg(
        long,
        default_value = "20",
        help = "SGVB Monte Carlo samples per iteration"
    )]
    pub num_sgvb_samples: usize,

    #[arg(long, default_value = "0.01", help = "AdamW learning rate")]
    pub learning_rate: f64,

    #[arg(long, default_value = "500", help = "Training iterations per gene")]
    pub num_iterations: usize,

    #[arg(
        long,
        default_value = "1000",
        help = "Minibatch size (full batch if N <= batch_size)"
    )]
    pub batch_size: usize,

    #[arg(
        long,
        default_value = "50",
        help = "ELBO values to average for convergence"
    )]
    pub elbo_window: usize,

    #[arg(
        long,
        default_value = "50",
        help = "Block size for MultiLevelSusieVar tree"
    )]
    pub ml_block_size: usize,

    // ── Empirical Bayes ──────────────────────────────────────────────────
    #[arg(long, help = "Enable cross-gene empirical Bayes for prior variance")]
    pub empirical_bayes: bool,

    // ── Covariates ───────────────────────────────────────────────────────
    #[arg(
        long,
        default_value = "true",
        help = "Include cell-type composition as covariates"
    )]
    pub composition_covariates: bool,

    #[arg(
        long,
        help = "Additional covariate TSV/CSV files (individual_id + values)",
        long_help = "Additional covariate file(s) in TSV/CSV format.\n\
            First column = individual ID, remaining columns = covariate values.\n\
            Multiple files can be specified. Covariates are concatenated with\n\
            composition covariates and centered before fitting."
    )]
    pub covariate_files: Vec<String>,

    // ── Device ───────────────────────────────────────────────────────────
    #[arg(
        long,
        value_enum,
        default_value = "cpu",
        help = "Compute device: cpu, cuda, metal"
    )]
    pub device: ComputeDevice,

    #[arg(long, default_value_t = 0, help = "Device number for cuda or metal")]
    pub device_no: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "Parallel jobs (0 = auto: all CPUs for cpu, 1 for gpu)"
    )]
    pub jobs: usize,

    // ── Misc ─────────────────────────────────────────────────────────────
    #[arg(long, default_value = "42", help = "Random seed")]
    pub seed: u64,

    #[arg(short, long, help = "Output prefix for results and parameters")]
    pub output: String,
}

pub fn map_qtl(args: &MapQtlArgs) -> Result<()> {
    info!("Starting map-qtl");

    let device = args.device.to_device(args.device_no)?;
    let use_gpu = args.device != ComputeDevice::Cpu;
    let num_jobs = if args.jobs == 0 {
        if use_gpu {
            1
        } else {
            rayon::current_num_threads()
        }
    } else {
        args.jobs
    };
    info!("Compute device: {:?}, jobs: {}", args.device, num_jobs);

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
        sigma2_inf: 0.0,
        prior_alpha: 1.0,
    };

    info!(
        "Model: {:?}, L={}, prior_vars={:?}",
        model_type, args.num_components, &fit_config.prior_vars
    );

    // ── Step 9: Per-gene fine-mapping ──────────────────────────────────
    info!(
        "Starting per-gene fine-mapping ({} genes, {} jobs)",
        n_testable, num_jobs
    );

    let fit_gene = |(spec_idx, spec): (usize, &GeneSpec)| -> Option<GeneResult> {
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

        let detailed = match fit_block_weighted(
            &x_g,
            &y_g,
            &v_g,
            covariates.as_ref(),
            &gene_config,
            &device,
        ) {
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
            detailed.best_result().avg_elbo,
        );

        Some(GeneResult {
            gene_id: spec.gene_id.clone(),
            cis_snp_indices: valid_cis_indices,
            detailed,
            z_marginal,
        })
    };

    let gene_results: Vec<GeneResult> = if num_jobs <= 1 {
        gene_specs.iter().enumerate().filter_map(fit_gene).collect()
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_jobs)
            .build()?;
        pool.install(|| {
            gene_specs
                .par_iter()
                .enumerate()
                .filter_map(fit_gene)
                .collect()
        })
    };

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
                gr.detailed.best_result().pip.clone(),
                gr.detailed.best_result().effect_mean.clone(),
                gr.detailed.best_result().effect_std.clone(),
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
