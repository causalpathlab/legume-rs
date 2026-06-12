use super::{RowAlignMode, RunSqueezeArgs};
use crate::handlers::merging::{
    find_aligned_rows, generate_unique_batch_names, run_merge_backend, MergeBackendArgs,
};
use crate::hdf5_io::*;
use crate::interactive::{confirm, prompt_user_action, UserAction};
use crate::qc::*;
use crate::sparse_io::*;
use crate::zarr_io::{finalize_zarr_output, materialize_writable_backend};

use log::info;
use matrix_util::common_io::*;
use matrix_util::traits::RunningStatOps;

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

        // Resolve target path (do not stage yet — defer until after user confirm)
        let target_file = if let Some(output_prefix) = &cmd_args.output {
            let (_, output_file) = resolve_backend_file(output_prefix, Some(backend.clone()))?;
            if std::path::Path::new(output_file.as_ref()).exists() {
                return Err(anyhow::anyhow!(
                    "Output file already exists: {}. Please remove it first or choose a different name.",
                    output_file
                ));
            }
            output_file
        } else {
            if data_file.ends_with(".zarr.zip") {
                return Err(anyhow::anyhow!(
                    "In-place squeeze is not supported for `.zarr.zip` archives (write store is read-only). Pass `--output <prefix>` to produce a new backend."
                ));
            }
            data_file.clone()
        };

        // Open source for stats — staging hasn't happened, so read from input directly
        let data = open_sparse_matrix(&data_file, &backend)?;

        let nrow = data.num_rows().unwrap();
        let ncol = data.num_columns().unwrap();

        info!("before squeeze -- data: {} rows x {} columns", nrow, ncol);

        // Collect statistics for histogram
        let col_stat = collect_column_stat(data.as_ref(), cmd_args.block_size)?;
        let row_stat = collect_row_stat(data.as_ref(), cmd_args.block_size)?;

        let row_nnz_vec = row_stat.count_positives();
        let col_nnz_vec = col_stat.count_positives();

        // Suggest cutoffs by 2-means clustering of log(1+nnz)
        let want_hist =
            cmd_args.show_histogram || cmd_args.save_histogram.is_some() || cmd_args.interactive;
        let want_suggest = want_hist || cmd_args.auto_cutoff;
        let row_suggest = if want_suggest {
            suggest_nnz_cutoff(&row_nnz_vec)
        } else {
            None
        };
        let col_suggest = if want_suggest {
            suggest_nnz_cutoff(&col_nnz_vec)
        } else {
            None
        };

        // Interactive or auto: derive the cutoff from the suggestion when the
        // user left it unset (explicit non-zero values always win, per dimension)
        if cmd_args.interactive || cmd_args.auto_cutoff {
            row_nnz_cutoff = resolve_cutoff(cmd_args.row_nnz_cutoff, row_suggest, "row");
            col_nnz_cutoff = resolve_cutoff(cmd_args.column_nnz_cutoff, col_suggest, "column");
        }

        // Headless auto mode (no histogram shown): report what was applied
        if cmd_args.auto_cutoff && !want_hist {
            report_resolved_cutoff("row", &row_nnz_vec, row_nnz_cutoff);
            report_resolved_cutoff("column", &col_nnz_vec, col_nnz_cutoff);
        }

        // Show/save histogram if requested or in interactive mode
        if want_hist {
            display_nnz_histogram(
                &data_file,
                NnzAxis {
                    nnz: &row_nnz_vec,
                    cutoff: row_nnz_cutoff,
                    suggest: row_suggest,
                },
                NnzAxis {
                    nnz: &col_nnz_vec,
                    cutoff: col_nnz_cutoff,
                    suggest: col_suggest,
                },
                cmd_args.show_histogram || cmd_args.interactive,
                cmd_args.save_histogram.as_deref(),
            )?;
        }

        // Interactive mode: prompt user for action
        if cmd_args.interactive {
            let proceed = loop {
                match prompt_user_action(
                    &row_nnz_vec,
                    &col_nnz_vec,
                    row_nnz_cutoff,
                    col_nnz_cutoff,
                )? {
                    UserAction::Proceed => {
                        info!("Proceeding with squeeze operation...");
                        break true;
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
                            &data_file,
                            NnzAxis {
                                nnz: &row_nnz_vec,
                                cutoff: row_nnz_cutoff,
                                suggest: row_suggest,
                            },
                            NnzAxis {
                                nnz: &col_nnz_vec,
                                cutoff: col_nnz_cutoff,
                                suggest: col_suggest,
                            },
                            true,
                            None,
                        )?;
                    }
                    UserAction::Cancel => {
                        info!("Cancelled squeeze operation");
                        break false;
                    }
                }
            };

            if !proceed {
                continue;
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

        // Stage now (only when we'll actually perform the squeeze)
        drop(data);
        if cmd_args.output.is_some() {
            info!("Staging {} -> {}", data_file, target_file);
            materialize_writable_backend(&data_file, &target_file)?;
        }
        let data = open_sparse_matrix(&target_file, &backend)?;

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
        drop(data);

        if let Some(output_prefix) = &cmd_args.output {
            finalize_zarr_output(&target_file, output_prefix)?;
        }
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

    // Handle interactive / histogram / auto-cutoff for the first file to get cutoffs
    if cmd_args.interactive || cmd_args.show_histogram || cmd_args.auto_cutoff {
        let (backend, data_file) = resolve_backend_file(&cmd_args.data_files[0], None)?;
        let data = open_sparse_matrix(&data_file, &backend)?;

        let col_stat = collect_column_stat(data.as_ref(), cmd_args.block_size)?;
        let row_stat = collect_row_stat(data.as_ref(), cmd_args.block_size)?;
        let row_nnz_vec = row_stat.count_positives();
        let col_nnz_vec = col_stat.count_positives();

        // Suggest cutoffs by 2-means clustering of log(1+nnz)
        let row_suggest = suggest_nnz_cutoff(&row_nnz_vec);
        let col_suggest = suggest_nnz_cutoff(&col_nnz_vec);

        // Interactive or auto: derive the cutoff from the suggestion when the
        // user left it unset (explicit non-zero values always win, per dimension)
        if cmd_args.interactive || cmd_args.auto_cutoff {
            row_nnz_cutoff = resolve_cutoff(cmd_args.row_nnz_cutoff, row_suggest, "row");
            col_nnz_cutoff = resolve_cutoff(cmd_args.column_nnz_cutoff, col_suggest, "column");
        }

        let show_hist = cmd_args.show_histogram || cmd_args.interactive;
        if show_hist || cmd_args.save_histogram.is_some() {
            display_nnz_histogram(
                &data_file,
                NnzAxis {
                    nnz: &row_nnz_vec,
                    cutoff: row_nnz_cutoff,
                    suggest: row_suggest,
                },
                NnzAxis {
                    nnz: &col_nnz_vec,
                    cutoff: col_nnz_cutoff,
                    suggest: col_suggest,
                },
                show_hist,
                cmd_args.save_histogram.as_deref(),
            )?;
        }

        // Headless auto mode (no histogram shown): report what was applied
        if cmd_args.auto_cutoff && !show_hist {
            report_resolved_cutoff("row", &row_nnz_vec, row_nnz_cutoff);
            report_resolved_cutoff("column", &col_nnz_vec, col_nnz_cutoff);
        }

        if cmd_args.interactive {
            loop {
                match prompt_user_action(
                    &row_nnz_vec,
                    &col_nnz_vec,
                    row_nnz_cutoff,
                    col_nnz_cutoff,
                )? {
                    UserAction::Proceed => {
                        info!("Proceeding with squeeze and merge...");
                        break;
                    }
                    UserAction::AdjustCutoffs(new_row, new_col) => {
                        row_nnz_cutoff = new_row;
                        col_nnz_cutoff = new_col;
                        info!(
                            "Updated cutoffs: row={}, column={}",
                            row_nnz_cutoff, col_nnz_cutoff
                        );

                        display_nnz_histogram(
                            &data_file,
                            NnzAxis {
                                nnz: &row_nnz_vec,
                                cutoff: row_nnz_cutoff,
                                suggest: row_suggest,
                            },
                            NnzAxis {
                                nnz: &col_nnz_vec,
                                cutoff: col_nnz_cutoff,
                                suggest: col_suggest,
                            },
                            true,
                            None,
                        )?;
                    }
                    UserAction::Cancel => {
                        info!("Operation cancelled");
                        return Ok(());
                    }
                }
            }
        }
    }

    if cmd_args.dry_run {
        info!("Dry run complete");
        return Ok(());
    }

    match cmd_args.row_align {
        RowAlignMode::Union => run_merge_then_squeeze(cmd_args, row_nnz_cutoff, col_nnz_cutoff),
        RowAlignMode::Common => run_squeeze_then_merge(cmd_args, row_nnz_cutoff, col_nnz_cutoff),
    }
}

/// Union mode: Merge first with union of all rows, then squeeze the merged result.
fn run_merge_then_squeeze(
    cmd_args: &RunSqueezeArgs,
    row_nnz_cutoff: usize,
    col_nnz_cutoff: usize,
) -> anyhow::Result<()> {
    use data_beans::sparse_data_visitors::create_jobs;
    use rayon::prelude::*;
    use rustc_hash::FxHashMap as HashMap;

    let output_prefix = cmd_args.output.as_ref().unwrap();
    info!("Union mode: merge first, then squeeze");

    // Generate unique batch names
    let batch_names = generate_unique_batch_names(&cmd_args.data_files)?;
    info!("Batch names: {:?}", batch_names);

    // Step 1: Build union of all row names across all files
    info!("Building union of row names...");
    let mut all_row_names: Vec<Box<str>> = Vec::new();
    let mut row_name_set: rustc_hash::FxHashSet<Box<str>> = rustc_hash::FxHashSet::default();

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
        let jobs = create_jobs(ncols, data.num_rows().unwrap_or(0), cmd_args.block_size);
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

    let shape = (all_row_names.len(), column_names.len(), all_triplets.len());
    let mut merged_data = create_sparse_from_triplets_owned(
        all_triplets,
        shape,
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

        info!("Staging to temp: {}", temp_file);
        materialize_writable_backend(&data_file, &temp_file)?;

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
    let aligned_rows = find_aligned_rows(&data_refs, RowAlignMode::Common)?;

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
        zip: false,
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

/// Per-axis nnz state passed to [`display_nnz_histogram`].
struct NnzAxis<'a> {
    nnz: &'a [f32],
    cutoff: usize,
    suggest: Option<usize>,
}

/// Display and/or save nnz histogram with cutoff markers
fn display_nnz_histogram(
    data_file: &str,
    row: NnzAxis<'_>,
    col: NnzAxis<'_>,
    show: bool,
    save_prefix: Option<&str>,
) -> anyhow::Result<()> {
    use matrix_util::common_io::write_types;

    // Save raw nnz data if requested
    if let Some(prefix) = save_prefix {
        let row_file = format!("{}.row_nnz.txt", prefix);
        let col_file = format!("{}.col_nnz.txt", prefix);

        let row_nnz_usize: Vec<usize> = row.nnz.iter().map(|&x| x as usize).collect();
        let col_nnz_usize: Vec<usize> = col.nnz.iter().map(|&x| x as usize).collect();

        write_types(&row_nnz_usize, &row_file)?;
        write_types(&col_nnz_usize, &col_file)?;

        info!("Saved row nnz data to: {}", row_file);
        info!("Saved column nnz data to: {}", col_file);
    }

    if show {
        println!("\n========================================");
        println!("NNZ Distribution for: {}", data_file);
        println!("========================================\n");

        print_nnz_summary("Rows", row.nnz, row.cutoff, row.suggest);
        println!();
        print_nnz_summary("Columns", col.nnz, col.cutoff, col.suggest);

        println!("\n========================================\n");
    }

    Ok(())
}

/// Resolve the effective cutoff for one dimension. An explicit user value
/// (non-zero) always wins; otherwise fall back to the k-means suggestion, or 0
/// (keep everything) when no suggestion is available.
fn resolve_cutoff(explicit: usize, suggested: Option<usize>, label: &str) -> usize {
    if explicit != 0 {
        return explicit;
    }
    match suggested {
        Some(c) => c,
        None => {
            info!("no {label} cutoff suggestion (degenerate nnz); keeping all");
            0
        }
    }
}

/// Print the resolved cutoff and how much it drops (headless `--auto-cutoff` mode)
fn report_resolved_cutoff(label: &str, nnz: &[f32], cutoff: usize) {
    let total = nnz.len();
    let removed = nnz.iter().filter(|&&x| (x as usize) < cutoff).count();
    let pct = if total > 0 {
        100.0 * removed as f64 / total as f64
    } else {
        0.0
    };
    println!("auto {label} cutoff: {cutoff} (removes {removed} / {total} = {pct:.2}%)");
}
