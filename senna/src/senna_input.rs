use crate::embed_common::*;
use dashmap::DashSet as HashSet;
use matrix_util::common_io::{self, basename, file_ext, read_lines};

//////////////////////////////////////////
// read data files and batch membership //
//////////////////////////////////////////
pub struct ReadSharedRowsArgs {
    pub data_files: Vec<Box<str>>,
    pub batch_files: Option<Vec<Box<str>>>,
    pub preload: bool,
}

pub struct SparseDataWithBatch {
    pub data: SparseIoVec,
    pub batch: Vec<Box<str>>,
    pub nbatch: usize,
}

pub fn read_data_on_shared_rows(args: ReadSharedRowsArgs) -> anyhow::Result<SparseDataWithBatch> {
    // to avoid duplicate barcodes in the column names
    let attach_data_name = args.data_files.len() > 1;

    let mut data_vec = SparseIoVec::new();
    for data_file in args.data_files.iter() {
        info!("Importing data file: {}", data_file);

        let backend = match file_ext(data_file)?.to_string().as_str() {
            "h5" => SparseIoBackend::HDF5,
            "zarr" => SparseIoBackend::Zarr,
            _ => return Err(anyhow::anyhow!("unknown backend file {}", data_file)),
        };

        let mut data = open_sparse_matrix(data_file, &backend)?;

        if args.preload {
            data.preload_columns()?;
        }

        let data_name = attach_data_name.then(|| basename(data_file)).transpose()?;
        data_vec.push(Arc::from(data), data_name)?;
    }

    // check if row names are the same
    let row_names = data_vec[0].row_names()?;

    for j in 1..data_vec.len() {
        let row_names_j = data_vec[j].row_names()?;
        if row_names != row_names_j {
            return Err(anyhow::anyhow!(
                "Row names are not the same. Consider using `data-beans sort-rows` to make sure that the row names are consistent across data sets."
            ));
        }
    }

    // check batch membership
    let mut batch_membership = Vec::with_capacity(data_vec.len());

    if let Some(batch_files) = &args.batch_files {
        if batch_files.len() != args.data_files.len() {
            return Err(anyhow::anyhow!("# batch files != # of data files"));
        }

        for batch_file in batch_files.iter() {
            info!("Reading batch file: {}", batch_file);
            for s in read_lines(batch_file)? {
                batch_membership.push(s.to_string().into_boxed_str());
            }
        }
    } else {
        // Extract batch info from column names and/or file names
        let num_files = args.data_files.len();
        let column_counts = data_vec.num_columns_by_data()?;

        for (file_idx, &ncols) in column_counts.iter().enumerate() {
            let data_file = args.data_files[file_idx].clone();
            let (_dir, file_base, _ext) = common_io::dir_base_ext(&data_file)?;

            // Get column names for this file's range
            let col_start: usize = column_counts[..file_idx].iter().copied().sum();
            let col_end = col_start + ncols;
            let column_names = data_vec.column_names()?;
            let file_columns = &column_names[col_start..col_end];

            // Check if column names contain '@' (embedded batch info)
            let has_embedded_batch = file_columns
                .first()
                .map_or(false, |name| name.contains('@'));

            if has_embedded_batch {
                // Parse batch from column names
                for col_name in file_columns {
                    let embedded_batch = col_name
                        .rsplit('@')
                        .next()
                        .unwrap_or(col_name.as_ref());

                    // If multiple files, combine embedded batch with file name
                    let batch = if num_files > 1 {
                        format!("{}@{}", embedded_batch, file_base).into_boxed_str()
                    } else {
                        embedded_batch.to_string().into_boxed_str()
                    };
                    batch_membership.push(batch);
                }
                if num_files > 1 {
                    info!("File {}: combining embedded batch with file name '{}'", file_idx, file_base);
                } else {
                    info!("Extracting batch from column names (detected '@' separator)");
                }
            } else {
                // Use file name as batch
                info!("File {}: using file name '{}' as batch", file_idx, file_base);
                batch_membership.extend(vec![file_base; ncols]);
            }
        }
    }

    if batch_membership.len() != data_vec.num_columns() {
        return Err(anyhow::anyhow!(
            "# batch membership {} != # of columns {}",
            batch_membership.len(),
            data_vec.num_columns()
        ));
    }

    let batch_hash: HashSet<Box<str>> = batch_membership.iter().cloned().collect();
    let nbatch = batch_hash.len();

    Ok(SparseDataWithBatch {
        data: data_vec,
        batch: batch_membership,
        nbatch,
    })
}

///////////////////////////////////////////////
// read data stack (vector of `SparseIoVec`) //
///////////////////////////////////////////////

pub struct ReadSharedColumnsArgs {
    pub data_files: Vec<Box<str>>,
    pub batch_files: Option<Vec<Box<str>>>,
    pub num_types: usize,
    pub preload: bool,
}

pub struct SparseStackWithBatch {
    pub data_stack: SparseIoStack,
    pub batch_stack: Vec<Vec<Box<str>>>,
    pub nbatch_stack: Vec<usize>,
}

pub fn read_data_on_shared_columns(
    args: ReadSharedColumnsArgs,
) -> anyhow::Result<SparseStackWithBatch> {
    let nfiles = args.data_files.len();

    let nfiles_per_type = nfiles.div_ceil(args.num_types);

    let mut data_stack = SparseIoStack::new();

    if nfiles_per_type * args.num_types != nfiles {
        return Err(anyhow::anyhow!(
            "Found fewer data sets: {} vs. {}",
            nfiles_per_type * args.num_types,
            nfiles
        ));
    }

    if let Some(batch_files) = &args.batch_files {
        if batch_files.len() != nfiles {
            return Err(anyhow::anyhow!(
                "data files {} vs. batch files {}",
                nfiles,
                batch_files.len()
            ));
        }
    }

    use matrix_util::common_io::{file_ext, read_lines};

    for files in args.data_files.chunks(nfiles_per_type) {
        let mut data_vec = SparseIoVec::new();
        let attach_data_name = files.len() > 1;

        for (data_idx, data_file) in files.iter().enumerate() {
            info!("Importing data file: {}", data_file);
            let backend = match file_ext(data_file)?.to_string().as_str() {
                "h5" => SparseIoBackend::HDF5,
                "zarr" => SparseIoBackend::Zarr,
                _ => return Err(anyhow::anyhow!("unknown backend file {}", data_file)),
            };

            let mut data = open_sparse_matrix(data_file, &backend)?;

            if args.preload {
                data.preload_columns()?;
            }
            let data_name = attach_data_name.then(|| data_idx.to_string().into_boxed_str());
            data_vec.push(Arc::from(data), data_name)?;
        }

        data_stack.push(data_vec)?;
    }

    let mut batch_stack = vec![];
    let mut nbatch_stack = vec![];

    if let Some(batch_files) = &args.batch_files {
        for (data_vec, batch_file) in data_stack.stack.iter_mut().zip(batch_files) {
            let ntot = data_vec.num_columns();
            let mut batch_membership = Vec::with_capacity(ntot);

            info!("Reading batch file: {}", batch_file);
            for s in read_lines(batch_file)? {
                batch_membership.push(s.to_string().into_boxed_str());
            }
            let batch_hash: HashSet<Box<str>> = batch_membership.iter().cloned().collect();
            let nbatch = batch_hash.len();

            nbatch_stack.push(nbatch);
            batch_stack.push(batch_membership);
        }
    } else {
        for data_vec in data_stack.stack.iter_mut() {
            let ntot = data_vec.num_columns();
            let mut batch_membership = Vec::with_capacity(ntot);
            for (id, &nn) in data_vec.num_columns_by_data()?.iter().enumerate() {
                batch_membership.extend(vec![id.to_string().into_boxed_str(); nn]);
            }

            let batch_hash: HashSet<Box<str>> = batch_membership.iter().cloned().collect();
            let nbatch = batch_hash.len();

            nbatch_stack.push(nbatch);
            batch_stack.push(batch_membership);
        }
    }

    Ok(SparseStackWithBatch {
        data_stack,
        batch_stack,
        nbatch_stack,
    })
}

// /// Build an affine transformation matrix that will help reduce
// /// dimensions in training
// ///
// /// * `collapsed`: data matrices derived from collapsing operations
// /// * `target_size`: targeting size
// pub fn build_row_aggregator(collapsed: &CollapsedOut, target_size: usize) -> anyhow::Result<Mat> {
//     if collapsed.mu_observed.nrows() > target_size {
//         let log_x_nd = collapsed.mu_adjusted.as_ref().map_or_else(
//             || {
//                 collapsed
//                     .mu_observed
//                     .posterior_log_mean()
//                     .transpose()
//                     .clone()
//             },
//             |x| x.posterior_log_mean().transpose().clone(),
//         );

//         let kk = target_size.ilog2() as usize;
//         info!(
//             "reduce data features: {} -> {}",
//             log_x_nd.ncols(),
//             target_size,
//         );

//         let membership = row_membership_matrix(binary_sort_columns(&log_x_nd, kk)?)?;

//         if membership.ncols() != target_size {
//             let d_available = membership.ncols().min(target_size);
//             let mut ret = Mat::zeros(membership.nrows(), target_size);
//             ret.columns_range_mut(0..d_available)
//                 .copy_from(&membership.columns_range(0..d_available));
//             Ok(ret)
//         } else {
//             Ok(membership)
//         }
//     } else {
//         Ok(Mat::identity(collapsed.mu_observed.nrows(), target_size))
//     }
// }
