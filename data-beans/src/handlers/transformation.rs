use crate::handlers::merging::{find_aligned_rows, generate_unique_batch_names, run_merge_backend};
use crate::interactive::{confirm, prompt_user_action, UserAction};
use crate::misc::*;
use crate::qc::*;
use crate::sparse_io::*;
use crate::utilities::io_helpers::{
    read_col_names, read_row_names, MAX_COLUMN_NAME_IDX, MAX_ROW_NAME_IDX,
};
use crate::MergeBackendArgs;

use log::info;
use matrix_util::common_io::*;
use matrix_util::membership::Membership;
use matrix_util::traits::RunningStatOps;

// Import the argument structs from main.rs
// These will need to be public in main.rs
use crate::{ReorderRowsArgs, RunSqueezeArgs, SubsetColumnsArgs, SubsetRowsArgs};

/// Subset columns from a sparse matrix data file
///
/// This function takes a subset of columns from the input data file and writes
/// them to a new output file. Columns can be specified either by indices or by
/// a file containing column names. Optionally, the output can be squeezed to
/// remove rows/columns with too few non-zero entries.
pub fn subset_columns(args: &SubsetColumnsArgs) -> anyhow::Result<()> {
    let columns_indices = args.column_indices.clone();
    let column_name_file = args.name_file.clone();

    let (backend, data_file) = resolve_backend_file(&args.data_file, None)?;

    let (_, output_file) = resolve_backend_file(&args.output, Some(backend.clone()))?;

    if let Some(out_dir) = dirname(&output_file) {
        mkdir(&out_dir)?;
    }

    recursive_copy(&data_file, &output_file)?;
    info!("copied the existing data file {}", data_file);

    let mut data = open_sparse_matrix(&output_file, &backend)?;

    let original_ncol = data.num_columns().unwrap_or(0);
    info!("original data: {} columns", original_ncol);

    let selected_columns = if let Some(idx) = columns_indices {
        info!("using {} column indices directly", idx.len());
        idx
    } else if let Some(column_file) = column_name_file {
        let col_names = read_col_names(column_file, MAX_COLUMN_NAME_IDX)?;
        if col_names.is_empty() {
            return Err(anyhow::anyhow!("Empty column file"));
        }
        info!("read {} column names from file", col_names.len());

        let data_col_names = data
            .column_names()
            .expect("column names not found in data file");

        // Build membership from column names in file (these are the ones we want to keep)
        // The keys are the column names from the file, values are their indices
        let membership = Membership::from_pairs(
            col_names
                .iter()
                .enumerate()
                .map(|(i, name)| (name.clone(), i.to_string().into_boxed_str())),
            args.allow_prefix,
        )
        .with_delimiter(args.delimiter);

        // Match data columns against the membership to find which ones to keep
        let (matched, stats) = membership.match_keys(&data_col_names);

        info!(
            "Column matching: {} exact + {} base_key + {} prefix = {}/{} matched",
            stats.exact,
            stats.base_key,
            stats.prefix,
            stats.total_matched(),
            data_col_names.len()
        );

        if matched.is_empty() {
            let file_sample: Vec<_> = col_names.iter().take(3).collect();
            let data_sample: Vec<_> = data_col_names.iter().take(3).collect();
            info!("column names from file (sample): {:?}", file_sample);
            info!("column names in data (sample): {:?}", data_sample);
            return Err(anyhow::anyhow!(
                "Found empty columns - no matching column names"
            ));
        }

        // Build index list: for each data column that matched, include its index
        let idx: Vec<usize> = data_col_names
            .iter()
            .enumerate()
            .filter_map(|(i, name)| matched.get(name).map(|_| i))
            .collect();

        info!("subsetting to {} columns", idx.len());
        idx
    } else {
        return Err(anyhow::anyhow!(
            "either `column-indices` or `name-file` must be provided"
        ));
    };

    data.subset_columns_rows(Some(&selected_columns), None)?;

    // Verify by re-opening the file
    drop(data);
    let data = open_sparse_matrix(&output_file, &backend)?;
    let new_ncol = data.num_columns().unwrap_or(0);
    info!("after subset: {} columns (verified)", new_ncol);

    if args.do_squeeze {
        info!("Squeeze the backend data {}", &output_file);
        let squeeze_args = RunSqueezeArgs {
            data_files: vec![output_file],
            row_nnz_cutoff: args.row_nnz_cutoff,
            column_nnz_cutoff: args.column_nnz_cutoff,
            block_size: 100,
            preload: true,
            show_histogram: false,
            save_histogram: None,
            dry_run: false,
            interactive: false,
            output: None,
            row_align: crate::RowAlignMode::Common,
        };
        run_squeeze(&squeeze_args)?;
    }

    info!("done");
    Ok(())
}

/// Subset rows from a sparse matrix data file
///
/// This function takes a subset of rows from the input data file and writes
/// them to a new output file. Rows can be specified either by indices or by
/// a file containing row names. Optionally, the output can be squeezed to
/// remove rows/columns with too few non-zero entries.
pub fn subset_rows(args: &SubsetRowsArgs) -> anyhow::Result<()> {
    let row_indices = args.row_indices.clone();
    let row_name_file = args.name_file.clone();

    let (backend, data_file) = resolve_backend_file(&args.data_file, None)?;

    let (_, output_file) = resolve_backend_file(&args.output, Some(backend.clone()))?;

    if let Some(out_dir) = dirname(&output_file) {
        mkdir(&out_dir)?;
    }

    recursive_copy(&data_file, &output_file)?;
    info!("copied the existing data file {}", data_file);

    let mut data = open_sparse_matrix(&output_file, &backend)?;

    let original_nrow = data.num_rows().unwrap_or(0);
    info!("original data: {} rows", original_nrow);

    let selected_rows = if let Some(idx) = row_indices {
        info!("using {} row indices directly", idx.len());
        idx
    } else if let Some(row_file) = row_name_file {
        let row_names = read_row_names(row_file, MAX_ROW_NAME_IDX)?;
        if row_names.is_empty() {
            return Err(anyhow::anyhow!("Empty row file"));
        }
        info!("read {} row names from file", row_names.len());

        let data_row_names = data.row_names().expect("row names not found in data file");

        // Build membership from row names in file (these are the ones we want to keep)
        let membership = Membership::from_pairs(
            row_names
                .iter()
                .enumerate()
                .map(|(i, name)| (name.clone(), i.to_string().into_boxed_str())),
            args.allow_prefix,
        )
        .with_delimiter(args.delimiter);

        // Match data rows against the membership to find which ones to keep
        let (matched, stats) = membership.match_keys(&data_row_names);

        info!(
            "Row matching: {} exact + {} base_key + {} prefix = {}/{} matched",
            stats.exact,
            stats.base_key,
            stats.prefix,
            stats.total_matched(),
            data_row_names.len()
        );

        if matched.is_empty() {
            let file_sample: Vec<_> = row_names.iter().take(3).collect();
            let data_sample: Vec<_> = data_row_names.iter().take(3).collect();
            info!("row names from file (sample): {:?}", file_sample);
            info!("row names in data (sample): {:?}", data_sample);
            return Err(anyhow::anyhow!("Found empty rows - no matching row names"));
        }

        // Build index list: for each data row that matched, include its index
        let idx: Vec<usize> = data_row_names
            .iter()
            .enumerate()
            .filter_map(|(i, name)| matched.get(name).map(|_| i))
            .collect();

        info!("subsetting to {} rows", idx.len());
        idx
    } else {
        return Err(anyhow::anyhow!(
            "either `row-indices` or `name-file` must be provided"
        ));
    };

    info!("subsetting to {} rows", selected_rows.len());
    data.subset_columns_rows(None, Some(&selected_rows))?;

    // Verify by re-opening the file
    drop(data);
    let data = open_sparse_matrix(&output_file, &backend)?;
    let new_nrow = data.num_rows().unwrap_or(0);
    info!("after subset: {} rows (verified)", new_nrow);

    if args.do_squeeze {
        info!("Squeeze the backend data {}", &output_file);
        let squeeze_args = RunSqueezeArgs {
            data_files: vec![output_file],
            row_nnz_cutoff: args.row_nnz_cutoff,
            column_nnz_cutoff: args.column_nnz_cutoff,
            block_size: 100,
            preload: true,
            show_histogram: false,
            save_histogram: None,
            dry_run: false,
            interactive: false,
            output: None,
            row_align: crate::RowAlignMode::Common,
        };
        run_squeeze(&squeeze_args)?;
    }

    info!("done");
    Ok(())
}

/// Reorder rows in a sparse matrix data file
///
/// This function reorders the rows of the input data file according to the
/// order specified in a row name file, and writes the result to a new output file.
pub fn reorder_rows(args: &ReorderRowsArgs) -> anyhow::Result<()> {
    let row_names_order: Vec<Box<str>> = read_row_names(args.row_file.clone(), MAX_ROW_NAME_IDX)?;

    let (backend, data_file) = resolve_backend_file(&args.data_file, None)?;
    let (_, output_file) = resolve_backend_file(&args.output, Some(backend.clone()))?;

    if let Some(out_dir) = dirname(&output_file) {
        mkdir(&out_dir)?;
    }

    recursive_copy(&data_file, &output_file)?;
    info!("copied the existing data file {}", data_file);

    let mut data = open_sparse_matrix(&output_file, &backend)?;
    data.reorder_rows(&row_names_order)?;

    info!("done");
    Ok(())
}

/// Squeeze a sparse matrix by removing rows and columns with too few non-zero entries
///
/// If --output is specified: Squeezes all files and merges into single output file.
/// Otherwise, modifies files in-place (with confirmation in interactive mode).
pub fn run_squeeze(cmd_args: &RunSqueezeArgs) -> anyhow::Result<()> {
    let mut row_nnz_cutoff = cmd_args.row_nnz_cutoff;
    let mut col_nnz_cutoff = cmd_args.column_nnz_cutoff;

    // If output specified with multiple files, squeeze to temp and merge
    if cmd_args.output.is_some() && cmd_args.data_files.len() > 1 {
        return run_squeeze_and_merge(cmd_args, row_nnz_cutoff, col_nnz_cutoff);
    }

    for data_file_arg in &cmd_args.data_files {
        info!("Processing file: {}", data_file_arg);
        let (backend, data_file) = resolve_backend_file(data_file_arg, None)?;

        // Determine target file
        let target_file = if let Some(output_prefix) = &cmd_args.output {
            let (_, output_file) = resolve_backend_file(output_prefix, Some(backend.clone()))?;
            // Copy to output location first
            if std::path::Path::new(output_file.as_ref()).exists() {
                return Err(anyhow::anyhow!(
                    "Output file already exists: {}. Please remove it first or choose a different name.",
                    output_file
                ));
            }
            info!("Copying {} to {}", data_file, output_file);
            recursive_copy(&data_file, &output_file)?;
            output_file
        } else {
            data_file.clone()
        };

        let data = open_sparse_matrix(&target_file, &backend)?;

        let nrow = data.num_rows().unwrap();
        let ncol = data.num_columns().unwrap();

        info!("before squeeze -- data: {} rows x {} columns", nrow, ncol);

        // Collect statistics for histogram
        let col_stat = collect_column_stat(data.as_ref(), cmd_args.block_size)?;
        let row_stat = collect_row_stat(data.as_ref(), cmd_args.block_size)?;

        let row_nnz_vec = row_stat.count_positives();
        let col_nnz_vec = col_stat.count_positives();

        // Show/save histogram if requested or in interactive mode
        if cmd_args.show_histogram || cmd_args.save_histogram.is_some() || cmd_args.interactive {
            display_nnz_histogram(
                &target_file,
                &row_nnz_vec,
                &col_nnz_vec,
                row_nnz_cutoff,
                col_nnz_cutoff,
                cmd_args.show_histogram || cmd_args.interactive,
                cmd_args.save_histogram.as_deref(),
            )?;
        }

        // Interactive mode: prompt user for action
        if cmd_args.interactive {
            match prompt_user_action(&row_nnz_vec, &col_nnz_vec, row_nnz_cutoff, col_nnz_cutoff)? {
                UserAction::Proceed => {
                    info!("Proceeding with squeeze operation...");
                }
                UserAction::AdjustCutoffs(new_row, new_col) => {
                    row_nnz_cutoff = new_row;
                    col_nnz_cutoff = new_col;
                    info!(
                        "Updated cutoffs: row={}, column={}",
                        row_nnz_cutoff, col_nnz_cutoff
                    );

                    // Show updated histogram with new cutoffs
                    display_nnz_histogram(
                        &target_file,
                        &row_nnz_vec,
                        &col_nnz_vec,
                        row_nnz_cutoff,
                        col_nnz_cutoff,
                        true,
                        None,
                    )?;

                    // Ask again with new cutoffs
                    match prompt_user_action(
                        &row_nnz_vec,
                        &col_nnz_vec,
                        row_nnz_cutoff,
                        col_nnz_cutoff,
                    )? {
                        UserAction::Proceed => {
                            info!("Proceeding with squeeze operation...");
                        }
                        _ => {
                            info!("Cancelled squeeze operation");
                            continue;
                        }
                    }
                }
                UserAction::Cancel => {
                    info!("Cancelled squeeze operation");
                    continue;
                }
            }

            // Confirm in-place modification if no output
            if cmd_args.output.is_none() {
                let msg = format!(
                    "Modify {} in-place? This will permanently alter the file",
                    data_file_arg
                );
                if !confirm(&msg)? {
                    info!("Skipping in-place modification of {}", data_file_arg);
                    continue;
                }
            }
        }

        // Skip actual squeeze if dry run
        if cmd_args.dry_run {
            info!("Dry run mode - skipping squeeze operation");
            continue;
        }

        // Perform squeeze
        squeeze_by_nnz(
            data.as_ref(),
            SqueezeCutoffs {
                row: row_nnz_cutoff,
                column: col_nnz_cutoff,
            },
            cmd_args.block_size,
            cmd_args.preload,
        )?;

        let data = open_sparse_matrix(&target_file, &backend)?;

        info!(
            "after squeeze -- data: {} rows x {} columns",
            data.num_rows().unwrap(),
            data.num_columns().unwrap()
        );
    }

    Ok(())
}

/// Squeeze multiple files and merge into single output
///
/// For Common mode: Squeeze each file first, then find common rows, subset, and merge.
/// For Union mode: Merge first with union of all rows, then squeeze the merged result.
fn run_squeeze_and_merge(
    cmd_args: &RunSqueezeArgs,
    mut row_nnz_cutoff: usize,
    mut col_nnz_cutoff: usize,
) -> anyhow::Result<()> {
    let output_prefix = cmd_args.output.as_ref().unwrap();
    info!(
        "Squeeze and merge mode: {} files -> {}",
        cmd_args.data_files.len(),
        output_prefix
    );

    // Handle interactive mode for first file to get cutoffs
    if cmd_args.interactive || cmd_args.show_histogram {
        let (backend, data_file) = resolve_backend_file(&cmd_args.data_files[0], None)?;
        let data = open_sparse_matrix(&data_file, &backend)?;

        let col_stat = collect_column_stat(data.as_ref(), cmd_args.block_size)?;
        let row_stat = collect_row_stat(data.as_ref(), cmd_args.block_size)?;
        let row_nnz_vec = row_stat.count_positives();
        let col_nnz_vec = col_stat.count_positives();

        display_nnz_histogram(
            &data_file,
            &row_nnz_vec,
            &col_nnz_vec,
            row_nnz_cutoff,
            col_nnz_cutoff,
            true,
            cmd_args.save_histogram.as_deref(),
        )?;

        if cmd_args.interactive {
            match prompt_user_action(&row_nnz_vec, &col_nnz_vec, row_nnz_cutoff, col_nnz_cutoff)? {
                UserAction::Proceed => {
                    info!("Proceeding with squeeze and merge...");
                }
                UserAction::AdjustCutoffs(new_row, new_col) => {
                    row_nnz_cutoff = new_row;
                    col_nnz_cutoff = new_col;
                    info!(
                        "Updated cutoffs: row={}, column={}",
                        row_nnz_cutoff, col_nnz_cutoff
                    );
                }
                UserAction::Cancel => {
                    info!("Operation cancelled");
                    return Ok(());
                }
            }
        }
    }

    if cmd_args.dry_run {
        info!("Dry run complete");
        return Ok(());
    }

    match cmd_args.row_align {
        crate::RowAlignMode::Union => {
            run_merge_then_squeeze(cmd_args, row_nnz_cutoff, col_nnz_cutoff)
        }
        crate::RowAlignMode::Common => {
            run_squeeze_then_merge(cmd_args, row_nnz_cutoff, col_nnz_cutoff)
        }
    }
}

/// Union mode: Merge first with union of all rows, then squeeze the merged result.
fn run_merge_then_squeeze(
    cmd_args: &RunSqueezeArgs,
    row_nnz_cutoff: usize,
    col_nnz_cutoff: usize,
) -> anyhow::Result<()> {
    use data_beans::sparse_data_visitors::create_jobs;
    use fnv::FnvHashMap as HashMap;
    use rayon::prelude::*;

    let output_prefix = cmd_args.output.as_ref().unwrap();
    info!("Union mode: merge first, then squeeze");

    // Generate unique batch names
    let batch_names = generate_unique_batch_names(&cmd_args.data_files)?;
    info!("Batch names: {:?}", batch_names);

    // Step 1: Build union of all row names across all files
    info!("Building union of row names...");
    let mut all_row_names: Vec<Box<str>> = Vec::new();
    let mut row_name_set: std::collections::HashSet<Box<str>> = std::collections::HashSet::new();

    for data_file_arg in &cmd_args.data_files {
        let (backend, data_file) = resolve_backend_file(data_file_arg, None)?;
        let data = open_sparse_matrix(&data_file, &backend)?;
        for row_name in data.row_names()? {
            if row_name_set.insert(row_name.clone()) {
                all_row_names.push(row_name);
            }
        }
    }
    all_row_names.sort();
    info!("Union contains {} unique rows", all_row_names.len());

    // Create row name to union index mapping
    let row_to_union_idx: HashMap<Box<str>, u64> = all_row_names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.clone(), i as u64))
        .collect();

    // Step 2: Read triplets from each file and remap row indices
    info!("Reading and remapping triplets...");
    let mut all_triplets: Vec<(u64, u64, f32)> = Vec::new();
    let mut column_names: Vec<Box<str>> = Vec::new();
    let mut column_batch_names: Vec<Box<str>> = Vec::new();
    let mut col_offset: u64 = 0;

    for (batch_idx, data_file_arg) in cmd_args.data_files.iter().enumerate() {
        info!(
            "Processing file {}/{}: {}",
            batch_idx + 1,
            cmd_args.data_files.len(),
            data_file_arg
        );

        let (backend, data_file) = resolve_backend_file(data_file_arg, None)?;
        let mut data = open_sparse_matrix(&data_file, &backend)?;
        data.preload_columns()?;

        let file_row_names = data.row_names()?;
        let ncols = data.num_columns().unwrap_or(0);

        // Build mapping from file's row index to union row index
        let file_row_to_union: Vec<u64> = file_row_names
            .iter()
            .map(|name| *row_to_union_idx.get(name).unwrap())
            .collect();

        // Read triplets and remap
        let jobs = create_jobs(ncols, Some(cmd_args.block_size));
        let triplets_curr: Vec<(u64, u64, f32)> = jobs
            .par_iter()
            .filter_map(|(lb, ub)| {
                if let Ok((_, _, triplets)) = data.read_triplets_by_columns((*lb..*ub).collect()) {
                    Some(
                        triplets
                            .iter()
                            .map(|&(i, j, x_ij)| {
                                let union_row = file_row_to_union[i as usize];
                                (union_row, j + col_offset, x_ij)
                            })
                            .collect::<Vec<_>>(),
                    )
                } else {
                    None
                }
            })
            .flatten()
            .collect();

        all_triplets.extend(triplets_curr);

        // Add column names with batch suffix
        let batch_name = &batch_names[batch_idx];
        let file_col_names = data.column_names()?;
        column_names.extend(
            file_col_names
                .into_iter()
                .map(|x| format!("{}{}{}", x, COLUMN_SEP, batch_name).into_boxed_str()),
        );
        column_batch_names.extend(vec![batch_name.clone(); ncols]);

        col_offset += ncols as u64;
        info!("  Added {} columns, {} triplets", ncols, all_triplets.len());
    }

    // Step 3: Create merged sparse matrix
    info!(
        "Creating merged matrix: {} rows x {} columns, {} triplets",
        all_row_names.len(),
        column_names.len(),
        all_triplets.len()
    );

    let (backend, backend_file) = resolve_backend_file(output_prefix, None)?;

    if std::path::Path::new(backend_file.as_ref()).exists() {
        info!("Removing existing output file: {}", &backend_file);
        remove_file(&backend_file)?;
    }

    let mut merged_data = create_sparse_from_triplets(
        &all_triplets,
        (all_row_names.len(), column_names.len(), all_triplets.len()),
        Some(&backend_file),
        Some(&backend),
    )?;

    merged_data.register_row_names_vec(&all_row_names);
    merged_data.register_column_names_vec(&column_names);

    info!("Created merged file: {}", &backend_file);

    // Step 4: Squeeze the merged result
    info!(
        "Squeezing merged data with cutoffs: row={}, column={}",
        row_nnz_cutoff, col_nnz_cutoff
    );

    drop(merged_data); // Close before squeeze
    let merged_data = open_sparse_matrix(&backend_file, &backend)?;

    squeeze_by_nnz(
        merged_data.as_ref(),
        SqueezeCutoffs {
            row: row_nnz_cutoff,
            column: col_nnz_cutoff,
        },
        cmd_args.block_size,
        cmd_args.preload,
    )?;

    // Verify and report
    let final_data = open_sparse_matrix(&backend_file, &backend)?;
    info!(
        "After squeeze: {} rows x {} columns",
        final_data.num_rows().unwrap(),
        final_data.num_columns().unwrap()
    );

    // Write batch membership file
    let batch_memb_file = format!("{}.batch.gz", output_prefix);
    let batch_map: HashMap<Box<str>, Box<str>> =
        column_names.into_iter().zip(column_batch_names).collect();
    let default_batch = basename(output_prefix)?;
    let final_col_names = final_data.column_names()?;
    let final_batch_names: Vec<Box<str>> = final_col_names
        .iter()
        .map(|k| batch_map.get(k).unwrap_or(&default_batch).clone())
        .collect();
    write_lines(&final_batch_names, &batch_memb_file)?;

    info!("Squeeze and merge (union) complete!");
    Ok(())
}

/// Common mode: Squeeze each file first, then find common rows, subset, and merge.
fn run_squeeze_then_merge(
    cmd_args: &RunSqueezeArgs,
    row_nnz_cutoff: usize,
    col_nnz_cutoff: usize,
) -> anyhow::Result<()> {
    let output_prefix = cmd_args.output.as_ref().unwrap();
    info!("Common mode: squeeze first, then merge common rows");

    // Create temp directory
    let output_dir = dirname(output_prefix).unwrap_or_else(|| ".".into());
    mkdir(&output_dir)?;

    let temp_dir = tempfile::Builder::new()
        .prefix(".squeeze_temp_")
        .tempdir_in(&*output_dir)?;
    let temp_dir_path = temp_dir
        .path()
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Invalid temp dir path"))?;

    info!("Created temp directory: {}", temp_dir_path);

    let batch_names = generate_unique_batch_names(&cmd_args.data_files)?;
    info!("Batch names: {:?}", batch_names);

    let mut temp_files: Vec<Box<str>> = Vec::new();

    // Step 1: Squeeze each file to temp location
    for (idx, data_file_arg) in cmd_args.data_files.iter().enumerate() {
        info!(
            "Processing file {}/{}: {}",
            idx + 1,
            cmd_args.data_files.len(),
            data_file_arg
        );

        let (backend, data_file) = resolve_backend_file(data_file_arg, None)?;
        let data = open_sparse_matrix(&data_file, &backend)?;

        info!(
            "before squeeze -- data: {} rows x {} columns",
            data.num_rows().unwrap(),
            data.num_columns().unwrap()
        );

        let backend_ext = match backend {
            SparseIoBackend::Zarr => "zarr",
            SparseIoBackend::HDF5 => "h5",
        };
        let temp_file = format!("{}/{}.{}", temp_dir_path, batch_names[idx], backend_ext);

        info!("Copying to temp: {}", temp_file);
        recursive_copy(&data_file, &temp_file)?;

        let temp_data = open_sparse_matrix(&temp_file, &backend)?;
        squeeze_by_nnz(
            temp_data.as_ref(),
            SqueezeCutoffs {
                row: row_nnz_cutoff,
                column: col_nnz_cutoff,
            },
            cmd_args.block_size,
            cmd_args.preload,
        )?;

        let squeezed = open_sparse_matrix(&temp_file, &backend)?;
        info!(
            "after squeeze -- data: {} rows x {} columns",
            squeezed.num_rows().unwrap(),
            squeezed.num_columns().unwrap()
        );

        temp_files.push(temp_file.into_boxed_str());
    }

    // Step 2: Find common rows across squeezed files
    info!("Finding common rows...");

    let squeezed_data: Vec<_> = temp_files
        .iter()
        .map(|f| {
            let (backend, _) = resolve_backend_file(f, None)?;
            open_sparse_matrix(f, &backend)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let data_refs: Vec<&dyn SparseIo<IndexIter = Vec<usize>>> =
        squeezed_data.iter().map(|d| d.as_ref()).collect();
    let aligned_rows = find_aligned_rows(&data_refs, crate::RowAlignMode::Common)?;

    info!(
        "Found {} common rows across {} files",
        aligned_rows.len(),
        temp_files.len()
    );

    // Step 3: Subset each file to common rows
    for (idx, temp_file) in temp_files.iter().enumerate() {
        let (backend, _) = resolve_backend_file(temp_file, None)?;
        let mut data = open_sparse_matrix(temp_file, &backend)?;

        let current_rows = data.row_names()?;
        let row_indices: Vec<usize> = aligned_rows
            .iter()
            .filter_map(|target_row| current_rows.iter().position(|r| r == target_row))
            .collect();

        info!(
            "  File {}: subsetting from {} to {} rows",
            idx,
            current_rows.len(),
            row_indices.len()
        );

        data.subset_columns_rows(None, Some(&row_indices))?;
    }

    // Step 4: Merge aligned files
    info!("Merging {} aligned files...", temp_files.len());

    let (backend, _) = resolve_backend_file(&temp_files[0], None)?;
    let merge_args = MergeBackendArgs {
        data_files: temp_files.clone(),
        backend,
        output: cmd_args.output.clone().unwrap(),
        do_squeeze: false,
        row_nnz_cutoff: 0,
        column_nnz_cutoff: 0,
        block_size: cmd_args.block_size,
    };

    run_merge_backend(&merge_args)?;

    drop(temp_dir);
    info!("Cleaned up temporary directory");

    info!("Squeeze and merge (common) complete!");
    Ok(())
}

/// Display and/or save nnz histogram with cutoff markers
fn display_nnz_histogram(
    data_file: &str,
    row_nnz: &[f32],
    col_nnz: &[f32],
    row_cutoff: usize,
    col_cutoff: usize,
    show: bool,
    save_prefix: Option<&str>,
) -> anyhow::Result<()> {
    use matrix_util::common_io::write_types;

    // Save raw nnz data if requested
    if let Some(prefix) = save_prefix {
        let row_file = format!("{}.row_nnz.txt", prefix);
        let col_file = format!("{}.col_nnz.txt", prefix);

        let row_nnz_usize: Vec<usize> = row_nnz.iter().map(|&x| x as usize).collect();
        let col_nnz_usize: Vec<usize> = col_nnz.iter().map(|&x| x as usize).collect();

        write_types(&row_nnz_usize, &row_file)?;
        write_types(&col_nnz_usize, &col_file)?;

        info!("Saved row nnz data to: {}", row_file);
        info!("Saved column nnz data to: {}", col_file);
    }

    if show {
        println!("\n========================================");
        println!("NNZ Distribution for: {}", data_file);
        println!("========================================\n");

        print_nnz_summary("Rows", row_nnz, row_cutoff);
        println!();
        print_nnz_summary("Columns", col_nnz, col_cutoff);

        println!("\n========================================\n");
    }

    Ok(())
}

/// Print summary statistics and histogram for nnz distribution
fn print_nnz_summary(label: &str, nnz: &[f32], cutoff: usize) {
    const MAX_BAR_WIDTH: usize = 50; // Maximum width for histogram bars

    let total = nnz.len();
    let below_cutoff = nnz.iter().filter(|&&x| (x as usize) < cutoff).count();
    let pct_removed = if total > 0 {
        100.0 * below_cutoff as f64 / total as f64
    } else {
        0.0
    };

    // Calculate basic statistics
    let min = nnz.iter().copied().fold(f32::INFINITY, f32::min) as usize;
    let max = nnz.iter().copied().fold(f32::NEG_INFINITY, f32::max) as usize;
    let sum: f32 = nnz.iter().sum();
    let mean = if total > 0 { sum / total as f32 } else { 0.0 };

    // Calculate median
    let mut sorted = nnz.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if total > 0 {
        if total.is_multiple_of(2) {
            (sorted[total / 2 - 1] + sorted[total / 2]) / 2.0
        } else {
            sorted[total / 2]
        }
    } else {
        0.0
    };

    println!("{} NNZ Distribution:", label);
    println!("  Total: {}", total);
    println!(
        "  Min: {}, Max: {}, Mean: {:.2}, Median: {:.2}",
        min, max, mean, median
    );
    println!(
        "  Cutoff: {} (removes {} / {} = {:.2}%)",
        cutoff, below_cutoff, total, pct_removed
    );

    // Create histogram with log10(nnz+1) bins
    let hist = create_log_histogram(nnz, cutoff);

    // Find max count for proportional scaling
    let max_count = hist.iter().map(|(_, count, _)| *count).max().unwrap_or(1);

    println!("  Histogram (log10(nnz+1) scale):");
    for (bin_label, count, is_cutoff_bin) in hist {
        let marker = if is_cutoff_bin { " <-- CUTOFF" } else { "" };
        // Scale bar width proportionally to fit within MAX_BAR_WIDTH
        let bar_width = ((count as f64 / max_count as f64) * MAX_BAR_WIDTH as f64).round() as usize;
        let bar_width = bar_width.max(1); // Ensure at least 1 char for non-zero counts
        let bar = "█".repeat(bar_width);
        println!("    {:>8}: {:>6} {}{}", bin_label, count, bar, marker);
    }
}

/// Create histogram with log10(nnz+1) transformation
fn create_log_histogram(nnz: &[f32], cutoff: usize) -> Vec<(String, usize, bool)> {
    let cutoff_log = ((cutoff as f64 + 1.0).log10() * 10.0).round() as i32;

    // Create bins: bin value represents log10(nnz+1)*10 as integer
    let mut bins: std::collections::BTreeMap<i32, usize> = std::collections::BTreeMap::new();

    for &val in nnz {
        let log_val = ((val as f64 + 1.0).log10() * 10.0).round() as i32;
        *bins.entry(log_val).or_insert(0) += 1;
    }

    // Convert to output format with labels
    bins.into_iter()
        .map(|(bin, count)| {
            let bin_float = bin as f64 / 10.0;
            let nnz_approx = (10.0_f64.powf(bin_float) - 1.0).round() as usize;
            let label = format!("~{}", nnz_approx);
            let is_cutoff = bin == cutoff_log;
            (label, count, is_cutoff)
        })
        .collect()
}
