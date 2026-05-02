use crate::hdf5_io::*;
use crate::qc::*;
use crate::sparse_io::*;
use crate::sparse_io_vector::*;

use clap::{Args, ValueEnum};
use log::info;
use matrix_util::common_io::*;
use matrix_util::membership::Membership;
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
        help = "Cells per rayon job (omit for auto-scaling by feature count)"
    )]
    pub block_size: Option<usize>,

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
