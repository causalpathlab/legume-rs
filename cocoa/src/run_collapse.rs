use crate::common::*;
use crate::input::*;

use auxiliary_data::cell_annotations::{CellAnnotations, CellTypeMembership};
use clap::Parser;
use data_beans_alg::pseudobulk::collapse_pseudobulk;
use matrix_param::io::ParamIo;
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

    // Delegate to data-beans-alg's collapse_pseudobulk
    let collapsed = collapse_pseudobulk(
        data.sparse_data,
        &annotations,
        &membership,
        args.a0,
        args.b0,
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
