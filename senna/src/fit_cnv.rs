use clap::Args;
use log::info;
use nalgebra::DMatrix;

use auxiliary_data::data_loading::{read_data_on_shared_rows, ReadSharedRowsArgs};
use cnv::genome_order::{GenePosition, GenomeOrder};
use cnv::gibbs_hmm::{fit_gibbs_hmm_multilevel, GibbsHmmConfig};
use data_beans_alg::collapse_data::MultilevelParams;
use data_beans_alg::random_projection::RandProjOps;
use genomic_data::gff::{build_gene_map, read_gff_record_vec, FeatureType, GeneId};
use matrix_param::traits::Inference;
use matrix_util::traits::IoOps;

use crate::embed_common::*;

/// Standalone CNV subcommand args.
#[derive(Args, Debug)]
pub struct CnvStandaloneArgs {
    #[arg(required = true, value_delimiter = ',', help = "Sparse data files")]
    data_files: Vec<Box<str>>,

    #[arg(long, short, required = true, help = "Output prefix")]
    out: Box<str>,

    #[arg(long, required = true, help = "GFF/GTF annotation file")]
    gff: Box<str>,

    #[arg(long, short, value_delimiter(','), help = "Batch membership files")]
    batch_files: Option<Vec<Box<str>>>,

    #[arg(long, short = 'p', default_value_t = 50, help = "Projection dimension")]
    proj_dim: usize,

    #[arg(long, short = 'd', default_value_t = 10, help = "Sort dimension")]
    sort_dim: usize,

    #[arg(long, default_value_t = 10, help = "KNN for cross-batch matching")]
    knn_cells: usize,

    #[arg(long, default_value_t = 100, help = "Collapsing EM iterations")]
    iter_opt: usize,

    #[arg(
        long,
        default_value_t = 3,
        help = "Number of coarsening levels (thresholds from 0.8 down)"
    )]
    num_levels: usize,

    #[arg(
        long,
        default_value_t = 0.4,
        help = "Finest correlation threshold for gene coarsening"
    )]
    corr_threshold: f32,

    #[arg(long, default_value_t = 1, help = "Number of mixture components")]
    n_components: usize,

    #[arg(long, default_value_t = 3, help = "Number of CN states")]
    n_states: usize,

    #[arg(long, default_value_t = 500, help = "Gibbs iterations")]
    n_gibbs: usize,

    #[arg(long, default_value_t = 200, help = "Gibbs warmup iterations")]
    warmup: usize,
}

/// Standalone CNV calling pipeline:
///   1. Load sparse data
///   2. Collapse to pseudobulk with batch adjustment → log(mu_residual)
///   3. Genome-order genes using GFF
///   4. Greedy coarsening of adjacent genes by correlation
///   5. Mixture Gibbs HMM on coarsened blocks
///   6. Expand to gene-level and write output
pub fn run_cnv_standalone(args: &CnvStandaloneArgs) -> anyhow::Result<()> {
    // --- 1. Load data ---
    info!("Loading data ...");
    let mut data_vec = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: args.data_files.clone(),
        batch_files: args.batch_files.clone(),
        preload: false,
    })?;

    let gene_names = data_vec.data.row_names()?;
    let num_genes = data_vec.data.num_rows();
    let num_cells = data_vec.data.num_columns();
    info!("{} genes, {} cells", num_genes, num_cells);

    // --- 2. Collapse to pseudobulk ---
    info!("Random projection (dim={}) ...", args.proj_dim);
    let proj_out = data_vec.data.project_columns_with_batch_correction(
        args.proj_dim,
        None,
        Some(&data_vec.batch),
    )?;
    let proj_kn = proj_out.proj;

    info!("Multi-level collapsing ...");
    let collapsed_levels: Vec<CollapsedOut> = data_vec.data.collapse_columns_multilevel_vec(
        &proj_kn,
        &data_vec.batch,
        &MultilevelParams {
            knn_super_cells: args.knn_cells,
            num_levels: 1,
            sort_dim: args.sort_dim,
            num_opt_iter: args.iter_opt,
        },
    )?;

    let finest = collapsed_levels
        .last()
        .ok_or_else(|| anyhow::anyhow!("no collapsed levels"))?;

    let mu_residual = finest
        .mu_residual
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("no mu_residual — need >=2 batches for CNV calling"))?;

    let log_resid = mu_residual.posterior_log_mean().clone();
    let n_samples = log_resid.ncols();
    info!(
        "log(mu_residual): {} genes × {} pseudobulk samples",
        log_resid.nrows(),
        n_samples,
    );

    // Export raw mu_residual
    mu_residual.posterior_mean().to_parquet_with_names(
        &format!("{}.mu_residual.parquet", &args.out),
        (Some(&gene_names), Some("gene")),
        None,
    )?;
    info!("Wrote {}.mu_residual.parquet", &args.out);

    // --- 3. Genome ordering ---
    info!("Reading GFF {} ...", &args.gff);
    let gff_records = read_gff_record_vec(&args.gff)?;
    let gene_map = build_gene_map(&gff_records, Some(&FeatureType::Gene))?;

    let gene_positions: Vec<GenePosition> = gene_names
        .iter()
        .enumerate()
        .filter_map(|(idx, name)| {
            let gene_id = GeneId::Ensembl(name.clone());
            let rec = gene_map.get(&gene_id)?;
            let rec = rec.value();
            Some(GenePosition {
                gene_idx: idx,
                chromosome: rec.seqname.clone(),
                position: rec.start as u64,
            })
        })
        .collect();

    let genome_order = GenomeOrder::from_positions(&gene_positions);

    let match_rate = genome_order.len() as f32 / num_genes as f32;
    info!(
        "CNV: {}/{} genes on canonical chromosomes ({:.0}%), {} chromosomes",
        genome_order.len(),
        num_genes,
        match_rate * 100.0,
        genome_order.chr_boundaries.len()
    );

    if genome_order.is_empty() {
        anyhow::bail!(
            "No genes mapped to canonical chromosomes. Check GFF gene IDs vs data row names."
        );
    }

    // Reorder log_resid to genome order
    let ordered_log_resid = genome_order.reorder_rows(&log_resid)?;
    let n_ordered = ordered_log_resid.nrows();
    info!("Genome-ordered: {} genes", n_ordered);

    // --- 4 + 5. Multi-level coarsening + Gibbs HMM ---
    let chr_bounds = &genome_order.chr_boundaries;

    // Build correlation thresholds: linearly spaced from 0.9 down to corr_threshold
    let corr_thresholds: Vec<f32> = if args.num_levels <= 1 {
        vec![args.corr_threshold]
    } else {
        let hi = 0.9f32;
        let lo = args.corr_threshold;
        (0..args.num_levels)
            .map(|i| hi - (hi - lo) * i as f32 / (args.num_levels - 1) as f32)
            .collect()
    };
    info!("Correlation thresholds: {:?}", corr_thresholds);

    let config = GibbsHmmConfig {
        n_components: args.n_components,
        n_states: args.n_states,
        n_iter: args.n_gibbs,
        warmup: args.warmup,
        seed: 42,
        ..Default::default()
    };

    let (result, coarsening) =
        fit_gibbs_hmm_multilevel(&ordered_log_resid, chr_bounds, &corr_thresholds, &config);

    // --- 6. Expand to gene-level and write ---
    let m = result.n_components;
    let k = args.n_states;

    // Viterbi states: expand [B × M] → [G_ordered × M]
    let mut viterbi_gene = DMatrix::<f32>::zeros(n_ordered, m);
    for c in 0..m {
        let gene_path = coarsening.expand_vec_to_genes(&result.viterbi_paths[c], n_ordered);
        for g in 0..n_ordered {
            viterbi_gene[(g, c)] = gene_path[g] as f32;
        }
    }
    viterbi_gene.to_parquet(&format!("{}.cnv_viterbi.parquet", &args.out))?;
    info!("Wrote {}.cnv_viterbi.parquet", &args.out);

    // Posteriors: expand [B × K] per component → [G_ordered × (M*K)]
    let mut posteriors_gene = DMatrix::<f32>::zeros(n_ordered, m * k);
    for c in 0..m {
        let expanded = coarsening.expand_to_genes(&result.posteriors_mean[c], n_ordered);
        for g in 0..n_ordered {
            for j in 0..k {
                posteriors_gene[(g, c * k + j)] = expanded[(g, j)];
            }
        }
    }
    posteriors_gene.to_parquet(&format!("{}.cnv_posteriors.parquet", &args.out))?;
    info!("Wrote {}.cnv_posteriors.parquet", &args.out);

    // Responsibilities [S × M]
    result
        .responsibilities
        .to_parquet(&format!("{}.cnv_responsibilities.parquet", &args.out))?;
    info!("Wrote {}.cnv_responsibilities.parquet", &args.out);

    // Block annotation TSV
    {
        use matrix_util::common_io::write_lines;
        let mut lines: Vec<Box<str>> =
            vec!["block_idx\tchromosome\tstart_gene\tend_gene\tnum_genes".into()];
        for (b, block) in coarsening.blocks.iter().enumerate() {
            lines.push(
                format!(
                    "{}\t{}\t{}\t{}\t{}",
                    b,
                    block.chromosome,
                    block.start,
                    block.end,
                    block.len()
                )
                .into(),
            );
        }
        let outfile = format!("{}.cnv_blocks.tsv", &args.out);
        write_lines(&lines, &outfile)?;
        info!("Wrote {}", outfile);
    }

    // Gene-to-genome mapping
    {
        use matrix_util::common_io::write_lines;
        let mut lines: Vec<Box<str>> = vec!["gene_idx\tchromosome\tposition".into()];
        for &idx in &genome_order.ordered_indices {
            if let Some(gp) = gene_positions.iter().find(|gp| gp.gene_idx == idx) {
                lines.push(format!("{}\t{}\t{}", idx, gp.chromosome, gp.position).into());
            }
        }
        let outfile = format!("{}.cnv_genes.tsv", &args.out);
        write_lines(&lines, &outfile)?;
        info!("Wrote {}", outfile);
    }

    // Summary
    info!(
        "CNV done: {} components, {} blocks, emission means: {:?}",
        m,
        coarsening.num_blocks(),
        result
            .emission_means
            .iter()
            .map(|v| v.as_slice().to_vec())
            .collect::<Vec<_>>()
    );

    Ok(())
}
