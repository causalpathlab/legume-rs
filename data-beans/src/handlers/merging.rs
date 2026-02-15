use crate::misc::*;
use crate::sparse_io::*;
use crate::utilities::io_helpers::{read_col_names, read_row_names};

use data_beans::sparse_data_visitors::create_jobs;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressDrawTarget};
use log::info;
use matrix_util::common_io::*;
use matrix_util::mtx_io;
use rayon::prelude::*;

use fnv::FnvHashMap as HashMap;

// Import the argument structs and run_squeeze function from main
use crate::run_squeeze;
use crate::{AlignDataArgs, MergeBackendArgs, MergeMtxArgs, RunSqueezeArgs};

pub fn run_merge_backend(args: &MergeBackendArgs) -> anyhow::Result<()> {
    if args.data_files.len() <= 1 {
        info!("no need to merge one file");
        return Ok(());
    }

    if args.verbose > 0 {
        std::env::set_var("RUST_LOG", "info");
    }

    let num_batches = args.data_files.len();
    info!("merging over {} batches ...", num_batches);

    // Generate unique batch names
    let batch_names = generate_unique_batch_names(&args.data_files)?;
    info!("Batch names: {:?}", batch_names);

    let mut row_names = vec![];
    let mut column_names = vec![];
    let mut column_batch_names = vec![];
    let mut triplets = vec![];

    let mut ntot = 0;
    for (batch_idx, data_file) in args.data_files.iter().enumerate() {
        info!("Importing data file: {}", data_file);

        let backend = match file_ext(data_file)?.to_string().as_str() {
            "h5" => SparseIoBackend::HDF5,
            "zarr" => SparseIoBackend::Zarr,
            _ => SparseIoBackend::Zarr,
        };

        let mut data = open_sparse_matrix(data_file, &backend)?;
        data.preload_columns()?;

        if row_names.is_empty() {
            row_names = data.row_names()?;
        } else {
            info!("checking if the row names are consistent");
            assert_eq!(row_names, data.row_names()?);
        }

        if let Some(ntot_curr) = data.num_columns() {
            // 1. read triplets
            let jobs = create_jobs(ntot_curr, Some(args.block_size));
            let triplets_curr = jobs
                .par_iter()
                .progress_count(jobs.len() as u64)
                .filter_map(|(lb, ub)| {
                    if let Ok((_, _, triplets)) =
                        data.read_triplets_by_columns((*lb..*ub).collect())
                    {
                        let offset = (*lb + ntot) as u64;
                        Some(
                            triplets
                                .iter()
                                .map(|&(i, j, x_ij)| (i, j + offset, x_ij))
                                .collect::<Vec<_>>(),
                        )
                    } else {
                        None
                    }
                })
                .flatten()
                .collect::<Vec<_>>();

            let _names = data.column_names()?;
            let batch_name = &batch_names[batch_idx];
            column_names.extend(
                _names
                    .into_iter()
                    .map(|x| format!("{}{}{}", x, COLUMN_SEP, batch_name).into_boxed_str())
                    .collect::<Vec<_>>(),
            );

            triplets.extend(triplets_curr);
            column_batch_names.extend(vec![batch_name.clone(); ntot_curr]);
            ntot += ntot_curr;
        }
    }

    info!("Found {} columns/barcodes ...", column_names.len());

    let (backend, backend_file) = resolve_backend_file(&args.output, Some(args.backend.clone()))?;

    if std::path::Path::new(backend_file.as_ref()).exists() {
        info!(
            "This existing backend file '{}' will be deleted",
            &backend_file
        );
        remove_file(&backend_file)?;
    }

    let mut data = create_sparse_from_triplets(
        &triplets,
        (row_names.len(), column_names.len(), triplets.len()),
        Some(&backend_file),
        Some(&backend),
    )?;

    data.register_row_names_vec(&row_names);
    data.register_column_names_vec(&column_names);

    info!(
        "Successfully created a sparse backend file: {}",
        &backend_file
    );

    let batch_map = column_names
        .into_iter()
        .zip(column_batch_names)
        .collect::<HashMap<_, _>>();

    if args.do_squeeze {
        info!("Squeeze the backend data {}", &backend_file);
        let squeeze_args = RunSqueezeArgs {
            data_files: vec![backend_file.clone()],
            row_nnz_cutoff: args.row_nnz_cutoff,
            column_nnz_cutoff: args.column_nnz_cutoff,
            block_size: args.block_size,
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

    // do the batch mapping at the end
    let batch_memb_file = format!("{}.batch.gz", args.output);
    let data = open_sparse_matrix(&backend_file, &backend)?;
    let default_batch = basename(&args.output)?;
    let column_batch_names = data
        .column_names()?
        .iter()
        .map(|k| batch_map.get(k).unwrap_or(&default_batch).clone())
        .collect::<Vec<_>>();

    write_lines(&column_batch_names, &batch_memb_file)?;

    info!("done");
    Ok(())
}

pub fn run_merge_mtx(args: &MergeMtxArgs) -> anyhow::Result<()> {
    if args.verbose > 0 {
        std::env::set_var("RUST_LOG", "info");
    }

    let directories = args.data_directories.clone();

    let mut mtx_files = vec![];
    let mut row_files = vec![];
    let mut col_files = vec![];
    let mut batch_names = vec![];

    for dir in directories.iter() {
        let dir = dir.clone().into_string();

        if let Some(base) = std::path::Path::new(&dir).file_stem() {
            let base = base.to_str().expect("invalid base name").to_string();
            info!("Searching relevant files within: {}", &base);
            let batch_name = Some(base);

            if let Ok(this_dir) = std::fs::read_dir(&dir) {
                let mut mtx: Option<Box<str>> = None;
                let mut row: Option<Box<str>> = None;
                let mut col: Option<Box<str>> = None;

                for x in this_dir {
                    if let Some(_path) = x?.path().to_str() {
                        let _path = _path.to_string();

                        if _path.ends_with(args.mtx_file_name.as_ref()) {
                            mtx = Some(_path.into_boxed_str());
                        } else if _path.ends_with(args.feature_file_name.as_ref()) {
                            row = Some(_path.into_boxed_str());
                        } else if _path.ends_with(args.barcode_file_name.as_ref()) {
                            col = Some(_path.into_boxed_str());
                        }
                    }
                }

                if let (Some(m), Some(r), Some(c), Some(b)) = (mtx, row, col, batch_name) {
                    info!("Build {} from {}, {}, {} ", &b, &m, &r, &c);
                    mtx_files.push(m);
                    row_files.push(r);
                    col_files.push(c);
                    batch_names.push(b);
                }
            }
        }

        info!("Searching subdir within: {}", &dir);

        let mut sub_dir_vec = std::fs::read_dir(&dir)?
            .filter_map(Result::ok)
            .collect::<Vec<_>>();

        sub_dir_vec.sort_by_key(|entry| entry.file_name());

        for sub in sub_dir_vec {
            if let Some(sub_dir) = sub.path().to_str() {
                let mut mtx: Option<Box<str>> = None;
                let mut row: Option<Box<str>> = None;
                let mut col: Option<Box<str>> = None;

                if let Some(base) = std::path::Path::new(&sub_dir).file_stem() {
                    let base = base.to_str().expect("invalid base name").to_string();
                    info!("searching {} ...", &base);

                    let batch_name = Some(base);

                    if let Ok(sub_dir) = std::fs::read_dir(sub_dir) {
                        for x in sub_dir {
                            if let Some(_path) = x?.path().to_str() {
                                let _path = _path.to_string();

                                info!("Found: {}", &_path);

                                if _path.ends_with(args.mtx_file_name.as_ref()) {
                                    mtx = Some(_path.into_boxed_str());
                                } else if _path.ends_with(args.feature_file_name.as_ref()) {
                                    row = Some(_path.into_boxed_str());
                                } else if _path.ends_with(args.barcode_file_name.as_ref()) {
                                    col = Some(_path.into_boxed_str());
                                }
                            }
                        }
                    }

                    if let (Some(m), Some(r), Some(c), Some(b)) = (mtx, row, col, batch_name) {
                        info!("Build {} from {}, {}, {} ", &b, &m, &r, &c);
                        mtx_files.push(m);
                        row_files.push(r);
                        col_files.push(c);
                        batch_names.push(b);
                    }
                }
            }
        }
    }

    let num_batches = batch_names.len();

    info!("merging over {} batches ...", num_batches);

    debug_assert_eq!(num_batches, mtx_files.len());
    debug_assert_eq!(num_batches, row_files.len());
    debug_assert_eq!(num_batches, col_files.len());

    if num_batches == 0 {
        return Err(anyhow::anyhow!("No relevant files found"));
    }

    info!("Finding common rows/features ...");

    let mut row_hash: HashMap<Box<str>, usize> = HashMap::default();

    for row_file in row_files.iter() {
        let row_names = read_row_names(row_file.clone(), args.num_feature_name_words)?;
        for name in row_names.iter() {
            let n = row_hash.entry(name.clone()).or_insert(0);
            *n += 1;
        }
    }

    let mut common_rows: Vec<Box<str>> = row_hash
        .into_iter()
        .filter_map(|(k, v)| {
            if v == num_batches {
                Some(k.clone())
            } else {
                None
            }
        })
        .collect();

    common_rows.sort_by_key(|x| x.to_string());

    let row_pos: HashMap<Box<str>, usize> = common_rows
        .iter()
        .enumerate()
        .map(|(i, v)| (v.clone(), i))
        .collect();

    info!(
        "Found {} common row/feature names across {} file sets",
        row_pos.len(),
        num_batches
    );

    info!("Elongating column/barcode names ...");

    let mut column_names = vec![];
    let mut column_batch_names = vec![];

    for (col_file, batch_name) in col_files.iter().zip(batch_names.iter()) {
        let _names = read_col_names(col_file.clone(), args.num_barcode_name_words)?;
        let nn = _names.len();
        column_names.extend(
            _names
                .into_iter()
                .map(|x| format!("{}{}{}", x, COLUMN_SEP, batch_name).into_boxed_str())
                .collect::<Vec<_>>(),
        );

        column_batch_names.extend(vec![batch_name.clone().into_boxed_str(); nn]);
    }

    info!("Found {} columns/barcodes ...", column_names.len());

    info!("Renaming triplets...");

    let mut renamed_triplets = vec![];
    let mut offset = 0;
    let mut nnz_tot = 0;

    let pb = ProgressBar::new(num_batches as u64);

    if args.verbose > 0 {
        pb.set_draw_target(ProgressDrawTarget::hidden());
    }

    for b in 0..num_batches {
        let row_names = read_row_names(row_files[b].clone(), args.num_feature_name_words)?;

        let (triplets, shape) = mtx_io::read_mtx_triplets(&mtx_files[b].clone())?;

        if let Some((nrow, ncol, nnz)) = shape {
            info!(
                "{}: {} rows, {} columns, {} non-zeros",
                &mtx_files[b], nrow, ncol, nnz
            );

            let triplets = triplets
                .into_iter()
                .filter_map(|(batch_i, batch_j, x_ij)| {
                    if let Some(i) = row_pos.get(&row_names[batch_i as usize]) {
                        let i = *i as u64;
                        let j = batch_j + offset;
                        Some((i, j, x_ij))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();

            nnz_tot += triplets.len();
            renamed_triplets.extend(triplets);

            let batch_col_names =
                read_col_names(col_files[b].clone(), args.num_barcode_name_words)?;
            let batch_name = batch_names[b].clone();

            (0..ncol).for_each(|batch_j| {
                let loc = format!("{}{}{}", batch_col_names[batch_j], COLUMN_SEP, batch_name)
                    .into_boxed_str();

                let glob_j = offset as usize + batch_j;
                let glob = column_names[glob_j].clone();
                debug_assert_eq!(glob, loc);
            });

            offset += ncol as u64;
            pb.inc(1);
        }
    }
    pb.finish_and_clear();

    info!("Done with creating triplets from {} mtx files", num_batches);

    let backend = args.backend.clone();
    let output = args.output.clone();
    let batch_memb_file = (output.to_string() + ".batch.gz").into_boxed_str();

    let backend_file = match backend {
        SparseIoBackend::HDF5 => format!("{}.h5", &output),
        SparseIoBackend::Zarr => format!("{}.zarr", &output),
    };

    if std::path::Path::new(&backend_file).exists() {
        info!(
            "This existing backend file '{}' will be deleted",
            &backend_file
        );
        remove_file(&backend_file)?;
    }

    let mut data = create_sparse_from_triplets(
        &renamed_triplets,
        (row_pos.len(), offset as usize, nnz_tot),
        Some(&backend_file),
        Some(&backend),
    )?;

    data.register_row_names_vec(&common_rows);
    data.register_column_names_vec(&column_names);

    info!(
        "Successfully created a sparse backend file: {}",
        &backend_file
    );

    let batch_map = column_names
        .into_iter()
        .zip(column_batch_names)
        .collect::<HashMap<_, _>>();

    if args.do_squeeze {
        info!("Squeeze the backend data {}", &backend_file);
        let squeeze_args = RunSqueezeArgs {
            data_files: vec![backend_file.clone().into_boxed_str()],
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

    // do the batch mapping at the end
    let data = open_sparse_matrix(&backend_file, &backend)?;
    let default_batch = basename(&args.output)?;
    let column_batch_names = data
        .column_names()?
        .iter()
        .map(|k| batch_map.get(k).unwrap_or(&default_batch).clone())
        .collect::<Vec<_>>();

    write_lines(&column_batch_names, &batch_memb_file)?;

    info!("done");
    Ok(())
}

/// Generate unique batch names from file paths
/// If basenames are unique, use them as-is
/// If duplicates exist, add numeric suffixes
pub fn generate_unique_batch_names(files: &[Box<str>]) -> anyhow::Result<Vec<Box<str>>> {
    use std::collections::HashMap;

    // Extract basenames
    let basenames: Vec<_> = files
        .iter()
        .map(|f| basename(f))
        .collect::<anyhow::Result<Vec<_>>>()?;

    // Count occurrences of each basename
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for name in &basenames {
        *counts.entry(name.as_ref()).or_insert(0) += 1;
    }

    // Generate unique names
    let mut name_counters: HashMap<&str, usize> = HashMap::new();
    let unique_names: Vec<Box<str>> = basenames
        .iter()
        .map(|name| {
            let count = counts.get(name.as_ref()).unwrap();
            if *count == 1 {
                // Unique basename, use as-is
                name.clone()
            } else {
                // Duplicate basename, add suffix
                let counter = name_counters.entry(name.as_ref()).or_insert(0);
                let unique_name = format!("{}_{}", name, counter).into_boxed_str();
                *counter += 1;
                unique_name
            }
        })
        .collect();

    Ok(unique_names)
}

/// Find common or union rows across multiple sparse matrices
pub fn find_aligned_rows(
    data_vec: &[&dyn SparseIo<IndexIter = Vec<usize>>],
    mode: crate::RowAlignMode,
) -> anyhow::Result<Vec<Box<str>>> {
    use fnv::FnvHashMap as HashMap;
    use rayon::prelude::*;

    let fully_shared = data_vec.len();
    let mut row_counts: HashMap<Box<str>, usize> = HashMap::default();

    // Count occurrences of each row across all files
    for data in data_vec {
        for row_name in data.row_names()? {
            *row_counts.entry(row_name).or_default() += 1;
        }
    }

    // Filter based on mode
    let mut aligned_rows: Vec<Box<str>> = match mode {
        crate::RowAlignMode::Common => {
            // Intersection: only rows present in ALL files
            row_counts
                .into_iter()
                .filter_map(|(row, count)| {
                    if count == fully_shared {
                        Some(row)
                    } else {
                        None
                    }
                })
                .collect()
        }
        crate::RowAlignMode::Union => {
            // Union: rows present in ANY file
            row_counts.into_keys().collect()
        }
    };

    if aligned_rows.is_empty() {
        return Err(anyhow::anyhow!("No rows found for alignment"));
    }

    aligned_rows.par_sort();
    Ok(aligned_rows)
}

pub fn align_backends(args: &AlignDataArgs) -> anyhow::Result<()> {
    let n_data = args.data_files.len();
    let n_data_columns = n_data.div_ceil(args.num_data_types);
    let n_expected = n_data_columns * args.num_data_types;

    if n_expected != n_data {
        return Err(anyhow::anyhow!(format!(
            "Should be multiple of {}: actual data files {} < expected {}",
            args.num_data_types, n_data, n_expected
        )));
    }

    let n_data_rows = n_expected.div_ceil(n_data_columns);

    let mut full_data_vec = args
        .data_files
        .iter()
        .enumerate()
        .map(|(i, a_file)| -> anyhow::Result<_> {
            let ext = file_ext(a_file)?;
            let base = basename(a_file)?;

            let data_col_id = i % n_data_columns + 1;
            let data_row_id = i / n_data_columns + 1;
            let dst_path = format!(
                "{}/{}_{}_{}.{}",
                args.output_directory, data_row_id, data_col_id, base, ext
            );
            info!("renaming files for easier sorting: {}", dst_path);
            recursive_copy(a_file, &dst_path)?;
            let (backend, a_copied_file) = resolve_backend_file(&dst_path, None)?;
            open_sparse_matrix(&a_copied_file, &backend)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    // identify common rows
    let mut shared_rows = vec![];
    for r in 0..n_data_rows {
        let data_set_idx: Vec<usize> = (r * n_data_columns..(r + 1) * n_data_columns).collect();

        let fully_shared = data_set_idx.len();
        let mut shared: HashMap<Box<str>, usize> = HashMap::default();
        for &di in data_set_idx.iter() {
            for v in full_data_vec[di].row_names()? {
                let count = shared.entry(v).or_default();
                *count += 1;
            }
        }

        let mut shared: Vec<Box<str>> = shared
            .into_iter()
            .filter_map(|(v, n)| if n == fully_shared { Some(v) } else { None })
            .collect();

        info!(
            "found {} shared rows/features on the data type {}",
            shared.len(),
            r
        );

        if shared.is_empty() {
            return Err(anyhow::anyhow!("no features are shared"));
        }

        shared.par_sort();
        shared_rows.push(shared);
    }

    for data_col in 0..n_data_columns {
        info!("aligning on the data column {}", data_col);

        // 0. subset of data sets
        let data_set_idx: Vec<usize> = (data_col..n_data).step_by(n_data_columns).collect();

        let subset_data_vec = data_set_idx
            .iter()
            .map(|&di| &full_data_vec[di])
            .collect::<Vec<_>>();

        // 1. figure out shared names for each column
        let fully_shared = data_set_idx.len();
        let mut shared: HashMap<Box<str>, usize> = HashMap::default();
        for &d in subset_data_vec.iter() {
            for v in d.column_names()? {
                let count = shared.entry(v).or_default();
                *count += 1;
            }
        }

        let mut shared: Vec<Box<str>> = shared
            .into_iter()
            .filter_map(|(v, n)| if n == fully_shared { Some(v) } else { None })
            .collect();

        info!("found {} shared columns", shared.len());

        if shared.is_empty() {
            for &di in data_set_idx.iter() {
                let data = &mut full_data_vec[di];
                info!(
                    "let's remove this unaligned backend: {}",
                    data.get_backend_file_name()
                );
                data.remove_backend_file()?;
            }
            continue;
        }

        shared.par_sort();

        // 2. subset columns
        for (r, &di) in data_set_idx.iter().enumerate() {
            let data = &mut full_data_vec[di];

            let pos: HashMap<Box<str>, usize> = data
                .column_names()?
                .into_iter()
                .enumerate()
                .map(|(i, x)| (x, i))
                .collect();
            let columns: Vec<usize> = shared.iter().map(|x| pos[x]).collect();

            let pos: HashMap<Box<str>, usize> = data
                .row_names()?
                .into_iter()
                .enumerate()
                .map(|(i, x)| (x, i))
                .collect();
            let rows: Vec<usize> = shared_rows[r].iter().map(|x| pos[x]).collect();
            data.subset_columns_rows(Some(&columns), Some(&rows))?;
        }

        info!(
            "done on the data column {}/{}",
            data_col + 1,
            n_data_columns
        );
    }

    Ok(())
}
