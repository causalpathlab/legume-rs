use crate::common::*;
use crate::input::*;

use auxiliary_data::cell_annotations::{CellAnnotations, CellTypeMembership};
use clap::Parser;
use data_beans_alg::gene_weighting::compute_nb_fisher_weights;
use data_beans_alg::pseudobulk::collapse_pseudobulk_weighted;
use matrix_param::io::ParamIo;
use matrix_util::common_io::write_lines;
use rustc_hash::FxHashMap as HashMap;

#[derive(Parser, Debug, Clone)]
pub struct CollapseArgs {
    #[arg(
        required = true,
        help = "Data files of either `.zarr` `.h` format",
        long_help = "Data files of either `.zarr` or `.h5` format. \n\
		     All the formats in the given list should be identical. \n\
		     You can convert `.mtx` to `.zarr` or `.h5` using the `data-beans`"
    )]
    data_files: Vec<Box<str>>,

    #[arg(
        short = 'i',
        long,
        value_delimiter = ',',
        help = "Individual membership file names (comma-separated).",
        long_help = "Individual membership files (comma-separated file names). \n\
		     Each line in each file can specify: \n\
		     * just  individual ID or\n\
		     * (1) Cell and (2) individual ID pair."
    )]
    indv_files: Option<Vec<Box<str>>>,

    #[arg(
        short = 't',
        long = "topic-assignment-files",
        value_delimiter = ',',
        help = "Latent topic assignment file names (comma-separated).",
        long_help = "Latent topic assignment files (comma-separated file names). \n\
		     Each line in each file can specify:\n\
		     * just topic name or \n\
		     * (1) cell and (2) topic name pair."
    )]
    topic_assignment_files: Option<Vec<Box<str>>>,

    #[arg(
        short = 'r',
        long = "topic-proportion-files",
        value_delimiter = ',',
        help = "Latent topic proportion file names (comma-separated).",
        long_help = "Latent topic proportion files (comma-separated file names). \n\
		     Each file contains a full `cell x topic` matrix."
    )]
    topic_proportion_files: Option<Vec<Box<str>>>,

    #[arg(
        long = "topic-proportion-value",
        default_value = "logit",
        help = "Is topic proportion matrix of probability?",
        long_help = "Specify if the topic proportion matrix is of probability type. \n\
		     Default is `logit`-valued."
    )]
    topic_proportion_value: TopicValue,

    #[arg(
        long = "a0",
        default_value_t = 1.0,
        help = "Hyperparameter a0 in Gamma(a0, b0)."
    )]
    a0: f32,

    #[arg(
        long = "b0",
        default_value_t = 1.0,
        help = "Hyperparameter b0 in Gamma(a0, b0)."
    )]
    b0: f32,

    #[arg(short, long = "out", required = true, help = "Output file name.")]
    out: Box<str>,

    #[arg(
        long = "preload-data",
        default_value_t = false,
        help = "Preload all the columns data."
    )]
    preload_data: bool,

    #[arg(
        long = "no-adjust-housekeeping",
        default_value_t = false,
        help = "Disable NB-Fisher housekeeping gene adjustment.",
        long_help = "By default, per-(individual, cell_type) count sums are row-scaled by\n\
                     NB-Fisher weights w_g = 1 / (1 + π_g · s̄ · φ(μ_g)) before the\n\
                     Gamma posterior update. High-mean / high-dispersion housekeeping\n\
                     genes get attenuated, matching pinto's gene-topic adjustment.\n\
                     Use this flag to disable for raw rates."
    )]
    no_adjust_housekeeping: bool,

    #[arg(
        long = "block-size",
        default_value_t = 1000,
        help = "Block size for reading cells when fitting the NB trend."
    )]
    block_size: usize,
}

pub fn run_collapse(args: CollapseArgs) -> anyhow::Result<()> {
    let data = read_input_data(InputDataArgs {
        data_files: args.data_files,
        indv_files: args.indv_files,
        topic_assignment_files: args.topic_assignment_files,
        topic_proportion_files: args.topic_proportion_files,
        exposure_assignment_file: None,
        preload_data: args.preload_data,
        topic_value: args.topic_proportion_value,
    })?;

    info!("Read the full data");

    let column_names = data.sparse_data.column_names()?;

    // Convert cocoa's cell_to_indv into CellAnnotations
    let annotations = build_cell_annotations(&data.cell_to_indv, &column_names);

    // Convert cocoa's cell_topic matrix into CellTypeMembership
    let membership = CellTypeMembership {
        matrix: data.cell_topic,
        cell_type_names: data.sorted_topic_names,
    };

    // NB-Fisher housekeeping weights (default ON, matches pinto)
    let gene_weights: Option<Vec<f32>> = if args.no_adjust_housekeeping {
        None
    } else {
        info!("Computing NB-Fisher housekeeping weights (--no-adjust-housekeeping to disable)");
        let w = compute_nb_fisher_weights(&data.sparse_data, Some(args.block_size))?;
        let wmin = w.iter().cloned().fold(f32::INFINITY, f32::min);
        let wmax = w.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        info!(
            "NB-Fisher weights: {} genes, min={:.4}, max={:.4}",
            w.len(),
            wmin,
            wmax
        );
        let gw_path = format!("{}.gene_weights.tsv.gz", args.out);
        let gene_names_w = data.sparse_data.row_names()?;
        let gw_lines: Vec<Box<str>> = std::iter::once("gene\tweight".into())
            .chain(
                gene_names_w
                    .iter()
                    .zip(w.iter())
                    .map(|(g, &wg)| format!("{}\t{}", g, wg).into()),
            )
            .collect();
        write_lines(&gw_lines, &gw_path)?;
        info!("Wrote gene weights to {}", gw_path);
        Some(w)
    };

    // Delegate to data-beans-alg's collapse_pseudobulk_weighted
    let collapsed = collapse_pseudobulk_weighted(
        data.sparse_data,
        &annotations,
        &membership,
        args.a0,
        args.b0,
        gene_weights.as_deref(),
    )?;

    // Write per cell type
    for (ct_idx, ct_name) in collapsed.cell_type_names.iter().enumerate() {
        let out_path = format!("{}.{}.parquet", args.out, ct_name);
        info!("Writing {} to {}", ct_name, out_path);

        collapsed.gamma_params[ct_idx].to_melted_parquet(
            &out_path,
            (Some(&collapsed.gene_names), Some("gene")),
            (Some(&collapsed.individual_ids), Some("individual")),
        )?;
    }

    info!(
        "Collapse complete: {} cell types, {} individuals, {} genes",
        collapsed.cell_type_names.len(),
        collapsed.individual_ids.len(),
        collapsed.gene_names.len(),
    );

    Ok(())
}

/// Convert cocoa's per-cell individual labels into CellAnnotations.
fn build_cell_annotations(cell_to_indv: &[Box<str>], column_names: &[Box<str>]) -> CellAnnotations {
    // Collect unique individual names in sorted order
    let mut indv_set: Vec<Box<str>> = cell_to_indv
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    indv_set.retain(|s| !s.is_empty() && s.as_ref() != "NA");

    let indv_to_idx: HashMap<Box<str>, usize> = indv_set
        .iter()
        .enumerate()
        .map(|(i, name)| (name.clone(), i))
        .collect();

    // Map each cell (by column name) to its individual index
    let cell_to_individual: HashMap<Box<str>, usize> = column_names
        .iter()
        .zip(cell_to_indv.iter())
        .filter_map(|(cell_name, indv_name)| {
            indv_to_idx
                .get(indv_name)
                .map(|&idx| (cell_name.clone(), idx))
        })
        .collect();

    CellAnnotations {
        cell_to_individual,
        individual_ids: indv_set,
    }
}
