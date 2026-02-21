use std::sync::Arc;

use anyhow::Result;
use clap::Args;
use log::info;
use matrix_util::traits::MatOps;
use nalgebra::DMatrix;
use rayon::prelude::*;
use rust_htslib::bgzf;
use rust_htslib::tpool::ThreadPool;
use std::io::Write;

use data_beans::convert::try_open_or_convert;
use data_beans::sparse_io_vector::SparseIoVec;
use fagioli::genotype::{BedReader, GenomicRegion, GenotypeReader};
use fagioli::mapping::pseudobulk::{
    build_onehot_membership, collapse_pseudobulk, infer_cell_annotations, read_cell_annotations,
    read_membership_proportions, Membership,
};
use fagioli::sgvb::{fit_block_weighted, BlockFitResultDetailed, FitConfig, ModelType};
use fagioli::simulation::{load_bed_annotations, load_gtf, GeneAnnotations};
use matrix_param::traits::Inference;
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

    // ── Composition covariates ────────────────────────────────────────────
    /// Include cell-type composition covariates (default: true)
    #[arg(long, default_value = "true")]
    pub composition_covariates: bool,

    // ── Misc ──────────────────────────────────────────────────────────────
    /// Random seed
    #[arg(long, default_value = "42")]
    pub seed: u64,

    /// Output prefix
    #[arg(short, long)]
    pub output: String,
}

/// Per-gene result from fine-mapping.
struct GeneResult {
    gene_id: String,
    cis_snp_indices: Vec<usize>,
    detailed: BlockFitResultDetailed,
    z_marginal: DMatrix<f32>,
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
        // If membership has negatives → softmax per row (logit → probability)
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
    let n_geno = geno.num_individuals();
    let m_snps = geno.num_snps();

    info!("Loaded genotypes: {} individuals x {} SNPs", n_geno, m_snps);

    // ── Step 5: Match individuals between pseudobulk and genotypes ───────
    let geno_id_lookup: std::collections::HashMap<&str, usize> = geno
        .individual_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (id.as_ref(), i))
        .collect();

    let mut matched_pb_indices: Vec<usize> = Vec::new();
    let mut matched_geno_indices: Vec<usize> = Vec::new();

    for (pb_idx, pb_id) in collapsed.individual_ids.iter().enumerate() {
        if let Some(&geno_idx) = geno_id_lookup.get(pb_id.as_ref()) {
            matched_pb_indices.push(pb_idx);
            matched_geno_indices.push(geno_idx);
        }
    }

    let n_matched = matched_pb_indices.len();
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

    // ── Step 6: Compute composition covariates ───────────────────────────
    let composition = if args.composition_covariates && n_ct > 1 {
        info!("Computing cell-type composition covariates");
        let mut comp = DMatrix::<f32>::zeros(n_matched, n_ct);
        for (row, &pb_idx) in matched_pb_indices.iter().enumerate() {
            let mut total_weight = 0.0f32;
            for ct_idx in 0..n_ct {
                let w = collapsed.cell_weights[ct_idx][pb_idx];
                comp[(row, ct_idx)] = w;
                total_weight += w;
            }
            if total_weight > 0.0 {
                for ct_idx in 0..n_ct {
                    comp[(row, ct_idx)] /= total_weight;
                }
            }
        }
        // Center columns
        for j in 0..n_ct {
            let mean = comp.column(j).sum() / n_matched as f32;
            for i in 0..n_matched {
                comp[(i, j)] -= mean;
            }
        }
        Some(comp)
    } else {
        None
    };

    // ── Step 7: Load gene annotations ────────────────────────────────────
    let gene_annot: Option<GeneAnnotations> = if let Some(ref gtf_path) = args.gtf_file {
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

    // ── Build per-gene phenotype data ────────────────────────────────────
    // Y_g: (N_matched, K_cell_types) — log expression
    // V_g: (N_matched, K_cell_types) — variance of log expression

    // Determine which genes to process and their cis SNP indices
    struct GeneSpec {
        gene_idx: usize,
        gene_id: String,
        cis_indices: Vec<usize>,
    }

    let gene_specs: Vec<GeneSpec> = if let Some(ref annot) = gene_annot {
        // Cis mode: match gene names to pseudobulk gene names via substring matching
        annot
            .genes
            .iter()
            .enumerate()
            .filter_map(|(ann_idx, gene)| {
                let gene_id_str = match &gene.gene_id {
                    genomic_data::gff::GeneId::Ensembl(s) => s.as_ref(),
                    genomic_data::gff::GeneId::Missing => return None,
                };

                // Flexible name match: gene_id or gene_name against pseudobulk names
                use data_beans::utilities::name_matching::flexible_name_match;
                let pb_gene_idx = collapsed
                    .gene_names
                    .iter()
                    .position(|pb_name| {
                        flexible_name_match(gene_id_str, pb_name)
                            || gene
                                .gene_name
                                .as_ref()
                                .is_some_and(|gn| flexible_name_match(gn, pb_name))
                    })?;

                // Get cis SNP indices
                let cis_indices =
                    annot.cis_snp_indices(ann_idx, &geno.positions, &geno.chromosomes);

                if cis_indices.is_empty() {
                    return None;
                }

                Some(GeneSpec {
                    gene_idx: pb_gene_idx,
                    gene_id: gene_id_str.to_string(),
                    cis_indices,
                })
            })
            .collect()
    } else {
        // Trans mode: all genes × all SNPs
        let all_snp_indices: Vec<usize> = (0..m_snps).collect();
        (0..n_genes)
            .map(|g| GeneSpec {
                gene_idx: g,
                gene_id: collapsed.gene_names[g].to_string(),
                cis_indices: all_snp_indices.clone(),
            })
            .collect()
    };

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

    // ── Step 8: Per-gene parallel loop ───────────────────────────────────
    info!("Starting per-gene fine-mapping ({} genes)", n_testable);

    let gene_results: Vec<GeneResult> = gene_specs
        .par_iter()
        .enumerate()
        .filter_map(|(spec_idx, spec)| {
            let gene_idx = spec.gene_idx;
            let p_cis = spec.cis_indices.len();

            if p_cis < 2 {
                return None;
            }

            // (a) Build Y_g: (N_matched, K) — posterior log mean across cell types
            let mut y_g = DMatrix::<f32>::zeros(n_matched, n_ct);
            let mut v_g = DMatrix::<f32>::zeros(n_matched, n_ct);

            for ct_idx in 0..n_ct {
                let log_mean = collapsed.gamma_params[ct_idx].posterior_log_mean();
                let log_sd = collapsed.gamma_params[ct_idx].posterior_log_sd();

                for (row, &pb_idx) in matched_pb_indices.iter().enumerate() {
                    let lm = log_mean[(gene_idx, pb_idx)];
                    let ls = log_sd[(gene_idx, pb_idx)];

                    if ls > 0.0 && collapsed.cell_weights[ct_idx][pb_idx] >= args.min_cell_weight {
                        y_g[(row, ct_idx)] = lm;
                        v_g[(row, ct_idx)] = ls * ls;
                    } else {
                        // Missing entry: huge variance, zero mean
                        y_g[(row, ct_idx)] = 0.0;
                        v_g[(row, ct_idx)] = 1e6;
                    }
                }
            }

            // (c) Build X_g: cis genotypes for matched individuals, standardized
            let mut x_g = DMatrix::<f32>::zeros(n_matched, p_cis);
            for (col, &snp_idx) in spec.cis_indices.iter().enumerate() {
                for (row, &geno_idx) in matched_geno_indices.iter().enumerate() {
                    x_g[(row, col)] = geno.genotypes[(geno_idx, snp_idx)];
                }
            }
            x_g.scale_columns_inplace();

            // Check for degenerate columns (zero variance → remove)
            let valid_cols: Vec<usize> = (0..p_cis)
                .filter(|&j| {
                    let col = x_g.column(j);
                    col.iter().any(|&v| v.abs() > 1e-8)
                })
                .collect();

            if valid_cols.len() < 2 {
                return None;
            }

            let x_g = if valid_cols.len() < p_cis {
                DMatrix::from_fn(n_matched, valid_cols.len(), |i, j| x_g[(i, valid_cols[j])])
            } else {
                x_g
            };
            let p_valid = x_g.ncols();

            // Map valid column indices back to original cis indices
            let valid_cis_indices: Vec<usize> =
                valid_cols.iter().map(|&j| spec.cis_indices[j]).collect();

            // (d) Fit weighted model
            let mut gene_config = fit_config.clone();
            gene_config.seed = fit_config.seed.wrapping_add(spec_idx as u64);

            let detailed =
                match fit_block_weighted(&x_g, &y_g, &v_g, composition.as_ref(), &gene_config) {
                    Ok(d) => d,
                    Err(e) => {
                        log::warn!("Gene {} failed: {}", spec.gene_id, e);
                        return None;
                    }
                };

            // (e) Compute marginal z-scores: weighted OLS per (SNP, cell_type)
            let z_marginal = compute_marginal_z(&x_g, &y_g, &v_g);

            info!(
                "Gene {}/{}: {} ({} cis SNPs, avg_elbo={:.2})",
                spec_idx + 1,
                n_testable,
                spec.gene_id,
                p_valid,
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

    // ── Step 9: Optional empirical Bayes ─────────────────────────────────
    let eb_weights: Option<Vec<f32>> = if args.empirical_bayes && n_fitted > 1 {
        info!(
            "Computing empirical Bayes weights across {} genes",
            n_fitted
        );
        let n_priors = fit_config.prior_vars.len();

        // For each prior_var, sum log-likelihoods across genes
        let mut sum_elbos = vec![0.0f64; n_priors];
        for gr in &gene_results {
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
    } else {
        None
    };

    // ── Step 10: Write output ────────────────────────────────────────────
    let out_file = format!("{}.results.bed.gz", args.output);
    info!("Writing results to {}", out_file);

    let mut writer = bgzf::Writer::from_path(&out_file)?;
    writer.set_thread_pool(&tpool)?;

    // Header
    writeln!(
        writer,
        "#chr\tstart\tend\tsnp_id\tgene_id\tcell_type\tpip\teffect_mean\teffect_std\tz_marginal"
    )?;

    for gr in &gene_results {
        let p_cis = gr.cis_snp_indices.len();

        // Re-weight if EB
        let (pip, eff_mean, eff_std) = if let Some(ref eb_w) = eb_weights {
            let mut pip_avg = DMatrix::<f32>::zeros(p_cis, n_ct);
            let mut eff_avg = DMatrix::<f32>::zeros(p_cis, n_ct);
            let mut std_avg = DMatrix::<f32>::zeros(p_cis, n_ct);

            // Combine EB weights with per-gene ELBOs
            let max_elbo = gr
                .detailed
                .per_prior_elbos
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let gene_weights: Vec<f32> = gr
                .detailed
                .per_prior_elbos
                .iter()
                .enumerate()
                .map(|(v_idx, &e)| eb_w[v_idx] * (e - max_elbo).exp())
                .collect();
            let sum_gw: f32 = gene_weights.iter().sum();

            for (v_idx, &gw) in gene_weights.iter().enumerate() {
                let w = gw / sum_gw;
                pip_avg += &gr.detailed.per_prior_pips[v_idx] * w;
                eff_avg += &gr.detailed.per_prior_effects[v_idx] * w;
                std_avg += &gr.detailed.per_prior_stds[v_idx] * w;
            }
            (pip_avg, eff_avg, std_avg)
        } else {
            (
                gr.detailed.result.pip.clone(),
                gr.detailed.result.effect_mean.clone(),
                gr.detailed.result.effect_std.clone(),
            )
        };

        for (snp_j, &global_snp) in gr.cis_snp_indices.iter().enumerate() {
            let chr = &geno.chromosomes[global_snp];
            let pos = geno.positions[global_snp];
            let snp_id = &geno.snp_ids[global_snp];

            for ct_idx in 0..n_ct {
                let ct_name = &collapsed.cell_type_names[ct_idx];
                writeln!(
                    writer,
                    "{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.4}",
                    chr,
                    pos,
                    pos + 1,
                    snp_id,
                    gr.gene_id,
                    ct_name,
                    pip[(snp_j, ct_idx)],
                    eff_mean[(snp_j, ct_idx)],
                    eff_std[(snp_j, ct_idx)],
                    gr.z_marginal[(snp_j, ct_idx)],
                )?;
            }
        }
    }

    writer.flush()?;
    info!("Results written: {}", out_file);

    // ── Gene summary ─────────────────────────────────────────────────────
    let summary_file = format!("{}.gene_summary.tsv.gz", args.output);
    info!("Writing gene summary to {}", summary_file);

    let mut summary_writer = bgzf::Writer::from_path(&summary_file)?;
    summary_writer.set_thread_pool(&tpool)?;

    writeln!(
        summary_writer,
        "gene_id\tnum_cis_snps\tmax_pip\tavg_elbo\tnum_significant_cell_types"
    )?;

    for gr in &gene_results {
        let max_pip = gr
            .detailed
            .result
            .pip
            .iter()
            .cloned()
            .fold(0.0f32, f32::max);

        // Count cell types with at least one SNP at PIP >= 0.5
        let mut sig_ct = 0usize;
        for ct_idx in 0..n_ct {
            let has_sig =
                (0..gr.cis_snp_indices.len()).any(|j| gr.detailed.result.pip[(j, ct_idx)] >= 0.5);
            if has_sig {
                sig_ct += 1;
            }
        }

        writeln!(
            summary_writer,
            "{}\t{}\t{:.6}\t{:.2}\t{}",
            gr.gene_id,
            gr.cis_snp_indices.len(),
            max_pip,
            gr.detailed.result.avg_elbo,
            sig_ct,
        )?;
    }

    summary_writer.flush()?;
    info!("Gene summary written: {}", summary_file);

    // ── Parameters JSON ──────────────────────────────────────────────────
    let param_file = format!("{}.parameters.json", args.output);
    let params = serde_json::json!({
        "command": "map-qtl",
        "sc_backend_files": args.sc_backend_files,
        "bed_prefix": args.bed_prefix,
        "chromosome": args.chromosome,
        "num_individuals_genotype": n_geno,
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
        "seed": args.seed,
    });
    std::fs::write(&param_file, serde_json::to_string_pretty(&params)?)?;
    info!("Wrote parameters: {}", param_file);

    info!("map-qtl completed successfully");
    Ok(())
}

/// Compute weighted OLS marginal z-scores per (SNP, cell_type).
///
/// For each (SNP j, cell type k):
///   β̂ = (x_j' W_k x_j)⁻¹ x_j' W_k y_k, with W_k = diag(1/V_ik)
///   SE = 1 / √(x_j' W_k x_j)
///   z_jk = β̂_jk / SE_jk
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
