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
use fagioli::io::results::{write_gene_summary, write_parameters, write_variant_results};
use fagioli::mapping::fit_qtl_helpers::*;
use fagioli::mapping::pseudobulk::{collapse_pseudobulk, Membership};
use fagioli::sgvb::{fit_block_weighted, ComputeDevice, FitConfig, ModelType, PriorType};
use matrix_util::common_io::basename;

#[derive(Args, Debug, Clone)]
pub struct FitQtlSgvbArgs {
    // ── Single-cell input ────────────────────────────────────────────────
    #[arg(
        long, num_args = 1..,
        help = "Single-cell count matrix files (Zarr, HDF5, or mtx)",
        long_help = "One or more single-cell count matrix files. Supported formats:\n\
            Zarr (.zarr), HDF5 (.h5), or Matrix Market (.mtx).\n\
            Multiple files are merged by cell ID. Genes (rows) must match\n\
            across files. Cells (columns) are the union of all files."
    )]
    pub sc_backend_files: Vec<Box<str>>,

    #[arg(
        long,
        help = "Cell annotations file mapping cells to individuals",
        long_help = "Cell annotations file (TSV or TSV.GZ) with columns:\n\
            - cell_id: must match column names in --sc-backend-files\n\
            - individual_id: donor/sample ID, must match IIDs in --bed-prefix .fam\n\
            - cell_type (optional): hard cell-type label for stratified pseudobulk\n\n\
            If cell_type column is present, pseudobulk is computed per (individual, cell_type).\n\
            For soft/probabilistic cell-type assignments, use --membership-parquet instead."
    )]
    pub cell_annotations: Option<Box<str>>,

    #[arg(
        long,
        help = "Soft cell-type membership proportions (parquet)",
        long_help = "Parquet file with soft/probabilistic cell-type membership proportions.\n\
            Rows = cells (indexed by cell_id), columns = cell types.\n\
            Each row sums to 1.0. Alternative to hard cell_type column in\n\
            --cell-annotations. Enables fractional pseudobulk aggregation."
    )]
    pub membership_parquet: Option<Box<str>>,

    // ── Genotype ─────────────────────────────────────────────────────────
    #[arg(
        long,
        help = "PLINK BED file prefix (without .bed/.bim/.fam)",
        long_help = "Path prefix for PLINK binary genotype files. The tool reads\n\
            {prefix}.bed, {prefix}.bim, and {prefix}.fam. Individual IDs (IID\n\
            column in .fam) are matched against individual_id in cell annotations."
    )]
    pub bed_prefix: Box<str>,

    #[arg(long, help = "Chromosome to analyze (must match chr column in .bim)")]
    pub chromosome: Box<str>,

    #[arg(
        long,
        help = "Left genomic position bound in bp (inclusive, filters SNPs)"
    )]
    pub left_bound: Option<u64>,

    #[arg(
        long,
        help = "Right genomic position bound in bp (inclusive, filters SNPs)"
    )]
    pub right_bound: Option<u64>,

    #[arg(
        long,
        help = "Subsample to at most N individuals from the genotype file"
    )]
    pub max_individuals: Option<usize>,

    // ── Gene annotations ─────────────────────────────────────────────────
    #[arg(
        long,
        help = "GTF/GFF gene annotation file for defining cis-eQTL windows",
        long_help = "GTF or GFF3 gene annotation file. Gene TSS (transcription start site)\n\
            is extracted and a cis-window of --cis-window bp is placed around it.\n\
            Only SNPs within the cis-window are tested for each gene.\n\
            Provide either --gtf-file or --gene-bed-file (not both)."
    )]
    pub gtf_file: Option<Box<str>>,

    #[arg(
        long,
        help = "BED gene annotation file (chr, start, end, gene_id[, name[, strand]])",
        long_help = "BED-format gene annotation file with columns:\n\
            chr, start, end, gene_id, [name], [strand].\n\
            The cis-window (--cis-window) is placed around gene start (or TSS\n\
            if strand is provided). Alternative to --gtf-file."
    )]
    pub gene_bed_file: Option<Box<str>>,

    #[arg(
        long,
        default_value = "1000000",
        help = "Cis-window size in bp around each gene TSS",
        long_help = "Size of the cis-window in base pairs, placed symmetrically around\n\
            each gene's TSS. Only SNPs within [TSS - window, TSS + window] are\n\
            included in the per-gene fine-mapping model. Default: 1000000 (1 Mb)."
    )]
    pub cis_window: u64,

    // ── Pseudobulk parameters ────────────────────────────────────────────
    #[arg(
        long,
        default_value = "1.0",
        help = "Gamma prior shape parameter (a0) for Poisson-Gamma pseudobulk",
        long_help = "Shape parameter (a0) of the Gamma prior on per-individual expression\n\
            rates in the Poisson-Gamma pseudobulk model. Larger values = stronger\n\
            shrinkage toward the prior mean. Default: 1.0."
    )]
    pub gamma_a0: f32,

    #[arg(
        long,
        default_value = "1.0",
        help = "Gamma prior rate parameter (b0) for Poisson-Gamma pseudobulk",
        long_help = "Rate parameter (b0) of the Gamma prior on per-individual expression\n\
            rates. The prior mean is a0/b0. Default: 1.0."
    )]
    pub gamma_b0: f32,

    #[arg(
        long,
        default_value = "1.0",
        help = "Min effective cells per individual-celltype pair to include",
        long_help = "Minimum effective cell weight (sum of membership proportions) per\n\
            (individual, cell_type) pair. Pairs below this threshold are excluded\n\
            from the pseudobulk. Prevents noisy estimates from individuals with\n\
            very few cells of a given type. Default: 1.0."
    )]
    pub min_cell_weight: f32,

    // ── Model parameters ─────────────────────────────────────────────────
    #[arg(
        long,
        default_value = "susie",
        help = "Fine-mapping model: 'susie', 'bisusie', or 'spike-slab'",
        long_help = "Fine-mapping model to use:\n\n\
            - susie: Sum of Single Effects with null absorber. Each of L components\n\
              selects one causal SNP via softmax over p+1 positions — p real SNPs\n\
              plus a null position that absorbs mass when there is no signal.\n\n\
            - bisusie: Bivariate SuSiE with separate predictor/outcome softmaxes.\n\n\
            - spike-slab: Independent per-SNP Bernoulli inclusion gates with\n\
              Gaussian slab. No component structure.\n\n\
            Default: susie."
    )]
    pub model: Box<str>,

    #[arg(
        long,
        default_value = "single",
        help = "Prior type: 'single' (grid search) or 'ash' (mixture-of-Gaussians)",
        long_help = "Prior type for effect sizes:\n\n\
            - single: Fixed single-Gaussian prior. The model is fit once for each\n\
              value in --prior-var, and the best is selected by ELBO. Default.\n\n\
            - ash: Mixture-of-Gaussians (adaptive shrinkage) prior. The --prior-var\n\
              grid becomes mixture components with learnable weights, plus a near-zero\n\
              spike component. Single fit, no grid search."
    )]
    pub prior_type: Box<str>,

    #[arg(
        long,
        default_value = "10",
        help = "Number of SuSiE components L (max causal SNPs per gene)",
        long_help = "Number of single-effect components (L) in the SuSiE/BiSuSiE model.\n\
            Each component can select one causal SNP, so L is the maximum number\n\
            of causal variants the model can identify per gene. Higher L increases\n\
            model capacity but slows optimization.\n\
            Ignored for spike-slab. Default: 10."
    )]
    pub num_components: usize,

    #[arg(
        long,
        default_value = "0.05,0.1,0.12,0.15,0.18,0.2,0.25,0.3,0.5",
        help = "Prior variance grid for effect sizes (comma-separated)",
        long_help = "Comma-separated list of prior variances for the effect size distribution.\n\
            The model is fit once for each value, and the best is selected by ELBO.\n\
            Prior variance controls expected effect size magnitude:\n\
            - smaller values (0.01-0.05): small effects, conservative\n\
            - larger values (0.3-1.0): large effects, liberal\n\
            Default: 0.05,0.1,0.12,0.15,0.18,0.2,0.25,0.3,0.5"
    )]
    pub prior_var: Box<str>,

    // ── SGVB training ────────────────────────────────────────────────────
    #[arg(
        long,
        default_value = "20",
        help = "Monte Carlo samples per SGVB gradient step",
        long_help = "Number of Monte Carlo samples (S) drawn per gradient step in\n\
            Stochastic Gradient Variational Bayes. More samples reduce gradient\n\
            variance but increase per-step cost. Default: 20."
    )]
    pub num_sgvb_samples: usize,

    #[arg(long, default_value = "0.01", help = "AdamW optimizer learning rate")]
    pub learning_rate: f64,

    #[arg(
        long,
        default_value = "500",
        help = "Max gradient steps per gene per prior variance"
    )]
    pub num_iterations: usize,

    #[arg(
        long,
        help = "Row minibatch size (full batch when N <= this value); \
                omit to auto-scale by variant count",
        long_help = "Number of individuals sampled per gradient step. When the total\n\
            number of individuals N exceeds this value, random minibatches of\n\
            this size are drawn each iteration. When N <= batch_size, all\n\
            individuals are used (full batch). Disabled for multilevel models.\n\
            Omit for auto-scaling by variant count."
    )]
    pub batch_size: Option<usize>,

    #[arg(
        long,
        default_value = "50",
        help = "Trailing window size for averaging ELBO (convergence diagnostic)",
        long_help = "Number of recent ELBO (evidence lower bound) values to average\n\
            for the reported convergence diagnostic. The average ELBO over the\n\
            last elbo_window iterations is reported per gene. This value is used\n\
            for prior variance selection (best ELBO wins). Default: 50."
    )]
    pub elbo_window: usize,

    // ── Empirical Bayes ──────────────────────────────────────────────────
    #[arg(
        long,
        help = "Re-weight prior variances across genes via empirical Bayes",
        long_help = "Enable cross-gene empirical Bayes for prior variance selection.\n\
            Instead of selecting the best prior variance per gene independently,\n\
            learns a shared prior variance distribution across all genes and\n\
            re-weights per-gene results accordingly. Useful when many genes are\n\
            tested and effect sizes are expected to be similar across genes."
    )]
    pub empirical_bayes: bool,

    // ── Covariates ───────────────────────────────────────────────────────
    #[arg(
        long,
        default_value = "true",
        help = "Include cell-type composition fractions as covariates",
        long_help = "When true, per-individual cell-type composition fractions are\n\
            included as covariates in the linear model. This adjusts for\n\
            compositional confounding (individuals with different cell-type\n\
            proportions). Default: true."
    )]
    pub composition_covariates: bool,

    #[arg(
        long,
        help = "Additional covariate files (TSV/CSV: individual_id + values)",
        long_help = "Additional covariate file(s) in TSV or CSV format.\n\
            First column = individual ID (must match .fam IIDs),\n\
            remaining columns = numeric covariate values.\n\
            Multiple files can be specified (repeated --covariate-files).\n\
            All covariates are concatenated with composition covariates\n\
            and column-centered before fitting."
    )]
    pub covariate_files: Vec<Box<str>>,

    // ── Device ───────────────────────────────────────────────────────────
    #[arg(
        long,
        value_enum,
        default_value = "cpu",
        help = "Hardware device for tensor computation: cpu, cuda, or metal"
    )]
    pub device: ComputeDevice,

    #[arg(
        long,
        default_value_t = 0,
        help = "GPU device index (for cuda or metal, 0-indexed)"
    )]
    pub device_no: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "Number of parallel gene-fitting jobs (0 = auto)",
        long_help = "Number of genes to fit in parallel.\n\
            0 = automatic: uses all CPU cores for --device cpu, or 1 for GPU.\n\
            Set to 1 for sequential execution (useful for debugging)."
    )]
    pub jobs: usize,

    // ── Misc ─────────────────────────────────────────────────────────────
    #[arg(long, default_value = "42", help = "Random seed for reproducibility")]
    pub seed: u64,

    #[arg(
        short,
        long,
        help = "Output file prefix (produces {prefix}.results.bed.gz and {prefix}.parameters.json)"
    )]
    pub output: Box<str>,
}

pub fn fit_qtl_sgvb(args: &FitQtlSgvbArgs) -> Result<()> {
    info!("Starting fit-qtl-sgvb");

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
    let prior_type: PriorType = args.prior_type.parse()?;
    let prior_vars: Vec<f32> = args
        .prior_var
        .split(',')
        .map(|s| s.trim().parse::<f32>())
        .collect::<std::result::Result<Vec<_>, _>>()?;

    let fit_config = FitConfig {
        model_type,
        prior_type,
        num_components: args.num_components,
        num_sgvb_samples: args.num_sgvb_samples,
        learning_rate: args.learning_rate,
        num_iterations: args.num_iterations,
        batch_size: args
            .batch_size
            .unwrap_or_else(|| matrix_util::utils::default_block_size(m_snps)),
        prior_vars,
        elbo_window: args.elbo_window,
        seed: args.seed,
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
        "command": "fit-qtl-sgvb",
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
        "batch_size": fit_config.batch_size,
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

    info!("fit-qtl-sgvb completed successfully");
    Ok(())
}
