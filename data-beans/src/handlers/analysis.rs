use crate::hdf5_io::*;
use crate::qc::*;
use crate::simulations::core as simulate;
use crate::simulations::multimodal as simulate_multimodal;
use crate::sparse_io::*;
use crate::sparse_io_vector::*;
use data_beans::zarr_io::{apply_zip_flag, finalize_zarr_output};

use clap::{Args, ValueEnum};
use log::info;
use matrix_util::common_io::*;
use matrix_util::membership::Membership;
use matrix_util::mtx_io;
use matrix_util::traits::IoOps;
use rayon::prelude::*;
use regex::Regex;
use std::sync::Arc;

#[derive(ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
pub enum StatDim {
    Row,
    Column,
}

#[derive(Args, Debug)]
pub struct RunStatArgs {
    #[arg(
        required = true,
        value_delimiter = ',',
        help = "Input data files in '.zarr' or '.h5' format",
        long_help = "Provide data files in either '.zarr' or '.h5' format. \n\
		     You can convert '.mtx' files to '.zarr' or '.h5' using\n\
		     the 'data-beans from-mtx' command."
    )]
    pub data_files: Vec<Box<str>>,

    #[arg(
        short,
        long,
        value_enum,
        help = "Statistics dimension (row or column)",
        long_help = "Choose whether to compute statistics over rows or columns."
    )]
    pub stat_dim: StatDim,

    #[arg(
        short,
        long,
        help = "Row name regex pattern for column statistics",
        long_help = "Specify a regex pattern to select row names \n\
		     when accumulating statistics over columns.\n\
		     Only rows matching this pattern will be included.\n\
		     Examples: '^MT-' (starts with MT-), 'GAPDH$' (ends with GAPDH),\n\
		     '^(MT|RPL|RPS)-' (mitochondrial or ribosomal genes).\n\
		     Matching is case-insensitive."
    )]
    pub row_name_pattern: Option<Box<str>>,

    #[arg(
        short = 'g',
        long,
        help = "Column group membership file for row statistics",
        long_help = "Provide a file that defines column group membership \n\
		     when accumulating statistics over rows. \n\
		     This provides statistics computed for group-wise analysis."
    )]
    pub column_group_file: Option<Box<str>>,

    #[arg(
        short = 'd',
        long,
        default_value = "@",
        help = "Delimiter for extracting base barcode from column names",
        long_help = "Delimiter character used to extract base barcode for matching. \n\
		     For example, with delimiter '@', column 'ACGT-1@batch1' matches \n\
		     membership key 'ACGT-1@batch2' via base key 'ACGT-1'."
    )]
    pub delimiter: char,

    #[arg(
        long,
        alias = "preload-data",
        default_value_t = false,
        help = "Preload data into memory for faster processing",
        long_help = "Preload all column data into memory before computing statistics. \n\
		     This can significantly speed up processing but requires more memory."
    )]
    pub preload: bool,

    #[arg(
        long,
        value_enum,
        default_value = "100",
        help = "Block size for processing",
        long_help = "Set the block size for processing data. \n\
		     Adjust this value to optimize performance for your hardware."
    )]
    pub block_size: usize,

    #[arg(
        short,
        long,
        default_value = "stdout",
        help = "Output statistics file",
        long_help = "Specify the output file for statistics. \n\
		     You can provide a '.parquet' file for efficient storage, \n\
		     or use 'stdout' to print results to the console."
    )]
    pub output: Box<str>,
}

#[derive(clap::Args, Debug)]
pub struct RunSimulateArgs {
    #[arg(short, long, help = "Number of rows, genes, or features")]
    pub rows: usize,

    #[arg(short, long, help = "Number of columns or cells")]
    pub cols: usize,

    #[arg(
        long,
        default_value_t = 1000,
        help = "Depth per column (expected non-zero genes per cell)"
    )]
    pub depth: usize,

    #[arg(
        short,
        long,
        default_value_t = 1,
        help = "Number of factors (cell types, topics, states, etc.)"
    )]
    pub factors: usize,

    #[arg(short, long, default_value_t = 1, help = "Number of batches")]
    pub batches: usize,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Proportion of variance explained by topic membership"
    )]
    pub pve_topic: f32,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Proportion of variance explained by batch effects"
    )]
    pub pve_batch: f32,

    #[arg(short, long, help = "Output file header")]
    pub output: Box<str>,

    #[arg(
        long,
        default_value_t = 10.0,
        help = "Overdispersion parameter for Gamma dictionary"
    )]
    pub overdisp: f32,

    #[arg(long, default_value_t = 42, help = "Random seed")]
    pub rseed: u64,

    #[arg(long, help = "Hierarchical tree depth for binary tree dictionary")]
    pub hierarchical_depth: Option<usize>,

    #[arg(long, default_value_t = 0, help = "Number of housekeeping genes")]
    pub n_housekeeping: usize,

    #[arg(long, default_value_t = 10.0, help = "Housekeeping fold change")]
    pub housekeeping_fold: f32,

    #[arg(
        long,
        default_value_t = 0,
        help = "Number of chromosomes for CNV simulation (0 = disabled)"
    )]
    pub n_chromosomes: usize,

    #[arg(
        long,
        default_value_t = 0.5,
        help = "Expected CNV events per chromosome"
    )]
    pub cnv_events_per_chr: f32,

    #[arg(
        long,
        default_value_t = 0.15,
        help = "CNV block size as fraction of genes per chromosome"
    )]
    pub cnv_block_frac: f32,

    #[arg(long, default_value_t = 2.0, help = "Fold-change for CNV gain events")]
    pub cnv_gain_fold: f32,

    #[arg(long, default_value_t = 0.5, help = "Fold-change for CNV loss events")]
    pub cnv_loss_fold: f32,

    #[arg(long, default_value_t = false, help = "Save output in MTX format")]
    pub save_mtx: bool,

    #[arg(
        long,
        value_enum,
        default_value = "zarr",
        help = "Backend format for output"
    )]
    pub backend: SparseIoBackend,

    /// produce a `.zarr.zip` archive instead of a `.zarr` directory
    #[arg(long, default_value_t = false)]
    pub zip: bool,
}

#[derive(Args, Debug)]
pub struct RunSimulateMultimodalArgs {
    #[arg(short, long, help = "Number of features (shared across modalities)")]
    pub rows: usize,

    #[arg(short, long, help = "Number of cells")]
    pub cols: usize,

    #[arg(
        long,
        value_delimiter = ',',
        help = "Depth per modality (comma-separated, e.g., 1000,500)"
    )]
    pub depth: Vec<usize>,

    #[arg(short, long, default_value_t = 5, help = "Number of topics")]
    pub factors: usize,

    #[arg(short, long, default_value_t = 1, help = "Number of batches")]
    pub batches: usize,

    #[arg(long, default_value_t = 1.0, help = "Scale of base dictionary logits")]
    pub base_scale: f32,

    #[arg(long, default_value_t = 1.0, help = "Scale of non-zero delta entries")]
    pub delta_scale: f32,

    #[arg(
        long,
        default_value_t = 5,
        help = "Number of non-zero delta genes per topic"
    )]
    pub n_delta_features: usize,

    #[arg(long, default_value_t = 1.0, help = "PVE by topic membership")]
    pub pve_topic: f32,

    #[arg(long, default_value_t = 1.0, help = "PVE by batch effects")]
    pub pve_batch: f32,

    #[arg(long, default_value_t = 42, help = "Random seed")]
    pub rseed: u64,

    #[arg(
        long,
        default_value_t = false,
        help = "Share batch effects across modalities"
    )]
    pub shared_batch_effects: bool,

    #[arg(long, help = "Hierarchical tree depth for base dictionary")]
    pub hierarchical_depth: Option<usize>,

    #[arg(
        long,
        default_value_t = 10.0,
        help = "Overdispersion for hierarchical dictionary"
    )]
    pub overdisp: f32,

    #[arg(long, default_value_t = 0, help = "Number of housekeeping genes")]
    pub n_housekeeping: usize,

    #[arg(long, default_value_t = 10.0, help = "Housekeeping fold change")]
    pub housekeeping_fold: f32,

    #[arg(short, long, help = "Output file header")]
    pub output: Box<str>,

    #[arg(long, default_value_t = false, help = "Save output in MTX format")]
    pub save_mtx: bool,

    #[arg(long, value_enum, default_value = "zarr", help = "Backend format")]
    pub backend: SparseIoBackend,

    /// produce a `.zarr.zip` archive instead of a `.zarr` directory
    #[arg(long, default_value_t = false)]
    pub zip: bool,
}

/// Compute statistics across sparse matrix data
///
/// This function computes row or column statistics across one or more sparse matrix
/// data files. Statistics can be stratified by column groups if a group membership
/// file is provided.
pub fn run_stat(cmd_args: &RunStatArgs) -> anyhow::Result<()> {
    let output = cmd_args.output.clone();
    dirname(&output).as_deref().map(mkdir).transpose()?;

    // to avoid duplicate barcodes in the column names
    let attach_data_name = cmd_args.data_files.len() > 1;

    let mut data = SparseIoVec::new();
    for data_file in cmd_args.data_files.iter() {
        let (backend, data_file) = resolve_backend_file(data_file, None)?;

        let mut this_data = open_sparse_matrix(&data_file, &backend)?;
        if cmd_args.preload {
            info!("Preloading data from {} ...", data_file);
            this_data.preload_columns()?;
        }
        let data_name = attach_data_name.then(|| basename(&data_file)).transpose()?;
        data.push(Arc::from(this_data), data_name)?;
    }

    match cmd_args.stat_dim {
        StatDim::Row => {
            if let Some(column_group_file) = &cmd_args.column_group_file {
                let cols = data.column_names()?;

                // Load membership and match to data columns
                // Use delimiter to extract base barcode for matching
                let membership = Membership::from_file(column_group_file, 0, 1, true)?
                    .with_delimiter(cmd_args.delimiter);
                let (column_membership, stats) = membership.match_keys(&cols);

                info!(
                    "Column matching: {} exact + {} base_key + {} prefix = {}/{} matched",
                    stats.exact,
                    stats.base_key,
                    stats.prefix,
                    stats.total_matched(),
                    stats.total()
                );

                if column_membership.is_empty() {
                    let data_sample: Vec<_> = cols.iter().take(3).collect();
                    let memb_sample = membership.sample_keys(3);
                    info!("Data columns sample: {:?}", data_sample);
                    info!("Membership keys sample: {:?}", memb_sample);
                }

                let unique_groups = membership.unique_groups();
                info!(
                    "Will collect stats for {} groups: {:?}",
                    unique_groups.len(),
                    unique_groups
                );

                let (group_names, group_stats) = collect_stratified_row_stat_across_vec(
                    &data,
                    &column_membership,
                    cmd_args.block_size,
                )?;

                info!(
                    "Collected {} group stats: {:?}",
                    group_names.len(),
                    group_names
                );

                if cmd_args.output.eq_ignore_ascii_case("stdout") {
                    for (g, row_stat) in group_names.iter().zip(group_stats.iter()) {
                        let out: Vec<Box<str>> = row_stat
                            .to_string_vec(&data.row_names()?, "\t")?
                            .into_iter()
                            .map(|s| format!("{}\t{}", g, s).into_boxed_str())
                            .collect();
                        write_lines(&out, &cmd_args.output)?;
                    }
                } else {
                    use matrix_util::sparse_stat::save_grouped_stats_parquet;

                    info!("writing out: {}", cmd_args.output);
                    save_grouped_stats_parquet(
                        &cmd_args.output,
                        &data.row_names()?,
                        &group_names,
                        &group_stats,
                    )?;
                }
            } else {
                let row_stat = collect_row_stat_across_vec(&data, cmd_args.block_size)?;
                row_stat.save(&cmd_args.output, &data.row_names()?, "\t")?;
            }
        }
        StatDim::Column => {
            let select_rows = cmd_args.row_name_pattern.as_ref().map(|pattern| {
                let re = Regex::new(&format!("(?i){}", pattern))
                    .expect("Invalid regex pattern for --row-name-pattern");
                let row_names = data.row_names().expect("couldn't get the row names");
                let selected: Vec<_> = row_names
                    .iter()
                    .enumerate()
                    .filter_map(|(i, name)| if re.is_match(name) { Some(i) } else { None })
                    .collect();
                info!(
                    "Row pattern '{}' matched {}/{} rows",
                    pattern,
                    selected.len(),
                    row_names.len()
                );
                selected
            });

            let col_stat =
                collect_column_stat_across_vec(&data, select_rows.as_deref(), cmd_args.block_size)?;

            col_stat.save(&cmd_args.output, &data.column_names()?, "\t")?;
        }
    };

    Ok(())
}

/// Simulate factored Poisson-Gamma data
///
/// This function generates simulated single-cell RNA-seq data using a factored
/// Poisson-Gamma model. The simulated data includes batch effects and topic
/// (cell type) structure. Output is saved in the specified backend format
/// (.zarr or .h5) along with parameter files.
pub fn run_simulate(cmd_args: &RunSimulateArgs) -> anyhow::Result<()> {
    let effective_output = apply_zip_flag(&cmd_args.output, cmd_args.zip);
    let output: Box<str> = strip_backend_suffix(&effective_output).into();

    dirname(&output).as_deref().map(mkdir).transpose()?;

    let backend = cmd_args.backend.clone();
    let (_, backend_file) = resolve_backend_file(&effective_output, Some(backend.clone()))?;

    let mtx_file = output.to_string() + ".mtx.gz";
    let row_file = output.to_string() + ".rows.gz";
    let col_file = output.to_string() + ".cols.gz";

    let dict_file = mtx_file.replace(".mtx.gz", ".dict.parquet");
    let prop_file = mtx_file.replace(".mtx.gz", ".prop.parquet");
    let batch_memb_file = mtx_file.replace(".mtx.gz", ".batch.gz");
    let ln_batch_file = mtx_file.replace(".mtx.gz", ".ln_batch.parquet");

    remove_all_files(&vec![
        backend_file.clone(),
        mtx_file.clone().into_boxed_str(),
        dict_file.clone().into_boxed_str(),
        prop_file.clone().into_boxed_str(),
        batch_memb_file.clone().into_boxed_str(),
        ln_batch_file.clone().into_boxed_str(),
    ])
    .expect("failed to clean up existing output files");

    let sim_args = simulate::SimArgs {
        rows: cmd_args.rows,
        cols: cmd_args.cols,
        depth: cmd_args.depth,
        factors: cmd_args.factors,
        batches: cmd_args.batches,
        overdisp: cmd_args.overdisp,
        pve_topic: cmd_args.pve_topic,
        pve_batch: cmd_args.pve_batch,
        rseed: cmd_args.rseed,
        hierarchical_depth: cmd_args.hierarchical_depth,
        n_housekeeping: cmd_args.n_housekeeping,
        housekeeping_fold: cmd_args.housekeeping_fold,
        n_chromosomes: cmd_args.n_chromosomes,
        cnv_events_per_chr: cmd_args.cnv_events_per_chr,
        cnv_block_frac: cmd_args.cnv_block_frac,
        cnv_gain_fold: cmd_args.cnv_gain_fold,
        cnv_loss_fold: cmd_args.cnv_loss_fold,
    };

    let sim = simulate::generate_factored_poisson_gamma_data(&sim_args)?;
    info!("successfully generated factored Poisson-Gamma data");

    let batch_out: Vec<Box<str>> = sim
        .batch_membership
        .iter()
        .map(|&x| Box::from(x.to_string()))
        .collect();

    write_lines(&batch_out, &batch_memb_file)?;
    info!("batch membership: {:?}", &batch_memb_file);

    let mtx_shape = (sim_args.rows, sim_args.cols, sim.triplets.len());

    let rows: Vec<Box<str>> = (0..cmd_args.rows)
        .map(|i| i.to_string().into_boxed_str())
        .collect();

    let cols: Vec<Box<str>> = (0..cmd_args.cols)
        .map(|i| i.to_string().into_boxed_str())
        .collect();

    sim.ln_delta_db
        .to_parquet_with_names(&ln_batch_file, (Some(&rows), Some("feature")), None)?;
    sim.theta_kn.transpose().to_parquet_with_names(
        &prop_file,
        (Some(&cols), Some("cell")),
        None,
    )?;
    sim.beta_dk
        .to_parquet_with_names(&dict_file, (Some(&rows), Some("feature")), None)?;

    if let Some(ref node_probs) = sim.hierarchy_node_probs {
        let hierarchy_file = mtx_file.replace(".mtx.gz", ".hierarchy.parquet");
        node_probs.to_parquet_with_names(&hierarchy_file, (Some(&rows), Some("feature")), None)?;
        info!("wrote hierarchy node probabilities: {:?}", &hierarchy_file);
    }

    if let (Some(ref chromosomes), Some(ref positions), Some(ref states)) =
        (&sim.gene_chromosomes, &sim.gene_positions, &sim.cnv_states)
    {
        let cnv_file = mtx_file.replace(".mtx.gz", ".cnv_ground_truth.tsv.gz");
        let state_labels = ["loss", "neutral", "gain"];
        let cnv_lines: Vec<Box<str>> = std::iter::once("gene\tchromosome\tposition\tstate".into())
            .chain(
                rows.iter()
                    .zip(chromosomes.iter())
                    .zip(positions.iter())
                    .zip(states.iter())
                    .map(|(((g, chr), pos), &st)| {
                        format!("{}\t{}\t{}\t{}", g, chr, pos, state_labels[st as usize]).into()
                    }),
            )
            .collect();
        write_lines(&cnv_lines, &cnv_file)?;
        info!("wrote CNV ground truth: {:?}", &cnv_file);
    }

    info!(
        "wrote parameter files:\n{:?},\n{:?},\n{:?}",
        &ln_batch_file, &dict_file, &prop_file
    );

    if cmd_args.save_mtx {
        let mut triplets = sim.triplets.clone();
        triplets.par_sort_by_key(|&(row, _, _)| row);
        triplets.par_sort_by_key(|&(_, col, _)| col);

        mtx_io::write_mtx_triplets(&triplets, sim_args.rows, sim_args.cols, &mtx_file)?;
        write_lines(&rows, &row_file)?;
        write_lines(&cols, &col_file)?;

        info!(
            "save mtx, row, and column files:\n{}\n{}\n{}",
            mtx_file, row_file, col_file
        );
    }

    info!("registering triplets ...");

    let mut data = create_sparse_from_triplets(
        &sim.triplets,
        mtx_shape,
        Some(&backend_file),
        Some(&backend),
    )?;

    info!("created sparse matrix: {}", backend_file);

    data.register_row_names_vec(&rows);
    data.register_column_names_vec(&cols);

    finalize_zarr_output(&backend_file, &effective_output)?;
    info!("done");
    Ok(())
}

/// Run multimodal simulation with shared base + delta dictionaries.
pub fn run_simulate_multimodal(cmd_args: &RunSimulateMultimodalArgs) -> anyhow::Result<()> {
    let output = cmd_args.output.clone();
    dirname(&output).as_deref().map(mkdir).transpose()?;

    let sim_args = simulate_multimodal::MultimodalSimArgs {
        rows: cmd_args.rows,
        cols: cmd_args.cols,
        depth_per_modality: cmd_args.depth.clone(),
        factors: cmd_args.factors,
        batches: cmd_args.batches,
        base_scale: cmd_args.base_scale,
        delta_scale: cmd_args.delta_scale,
        n_delta_features: cmd_args.n_delta_features,
        pve_topic: cmd_args.pve_topic,
        pve_batch: cmd_args.pve_batch,
        rseed: cmd_args.rseed,
        shared_batch_effects: cmd_args.shared_batch_effects,
        hierarchical_depth: cmd_args.hierarchical_depth,
        overdisp: cmd_args.overdisp,
        n_housekeeping: cmd_args.n_housekeeping,
        housekeeping_fold: cmd_args.housekeeping_fold,
    };

    let mm = sim_args.depth_per_modality.len();
    let sim = simulate_multimodal::generate_multimodal_data(&sim_args)?;
    info!("generated multimodal data: {} modalities", mm);

    let rows: Vec<Box<str>> = (0..cmd_args.rows)
        .map(|i| i.to_string().into_boxed_str())
        .collect();
    let cols: Vec<Box<str>> = (0..cmd_args.cols)
        .map(|i| i.to_string().into_boxed_str())
        .collect();

    // Shared outputs
    let prop_file = format!("{}.prop.parquet", output);
    sim.theta_kn.transpose().to_parquet_with_names(
        &prop_file,
        (Some(&cols), Some("cell")),
        None,
    )?;

    let batch_file = format!("{}.batch.gz", output);
    let batch_out: Vec<Box<str>> = sim
        .batch_membership
        .iter()
        .map(|&x| Box::from(x.to_string()))
        .collect();
    write_lines(&batch_out, &batch_file)?;

    // W_base
    let base_file = format!("{}.w_base.parquet", output);
    sim.w_base_kd
        .to_parquet_with_names(&base_file, (None, None), None)?;

    // Per-modality outputs
    let backend = cmd_args.backend.clone();
    for m in 0..mm {
        let suffix = format!(".m{}", m);
        let modality_output = apply_zip_flag(&format!("{}{}", output, suffix), cmd_args.zip);
        let (_, backend_file) = resolve_backend_file(&modality_output, Some(backend.clone()))?;

        let mtx_shape = (cmd_args.rows, cmd_args.cols, sim.triplets[m].len());

        // Dictionary
        let dict_file = format!("{}{}.dict.parquet", output, suffix);
        sim.beta_dk[m].to_parquet_with_names(&dict_file, (Some(&rows), Some("feature")), None)?;

        // Batch effects
        let ln_batch_file = format!("{}{}.ln_batch.parquet", output, suffix);
        sim.ln_delta_db[m].to_parquet_with_names(
            &ln_batch_file,
            (Some(&rows), Some("feature")),
            None,
        )?;

        // Delta (non-reference only)
        if m > 0 {
            let delta_file = format!("{}.w_delta{}.parquet", output, suffix);
            sim.w_delta_kd[m - 1].to_parquet_with_names(&delta_file, (None, None), None)?;

            let mask_file = format!("{}.spike_mask{}.parquet", output, suffix);
            sim.spike_mask_kd[m - 1].to_parquet_with_names(&mask_file, (None, None), None)?;
        }

        // MTX
        if cmd_args.save_mtx {
            let mtx_file = format!("{}{}.mtx.gz", output, suffix);
            let mut triplets = sim.triplets[m].clone();
            triplets.par_sort_by_key(|&(row, col, _)| (col, row));
            mtx_io::write_mtx_triplets(&triplets, cmd_args.rows, cmd_args.cols, &mtx_file)?;
        }

        // Sparse backend
        let mut data = create_sparse_from_triplets(
            &sim.triplets[m],
            mtx_shape,
            Some(&backend_file),
            Some(&backend),
        )?;

        data.register_row_names_vec(&rows);
        data.register_column_names_vec(&cols);

        finalize_zarr_output(&backend_file, &modality_output)?;
        info!("modality {}: {}", m, backend_file);
    }

    info!("done");
    Ok(())
}
