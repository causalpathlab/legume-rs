use crate::embed_common::*;
use data_beans::convert::try_open_or_convert;

pub use auxiliary_data::data_loading::{
    read_data_on_shared_rows, ReadSharedRowsArgs, SparseDataWithBatch,
};

/// Collapse a batch-membership slice into a single batch ("all"), neutralizing
/// any downstream per-batch correction. Use when `--ignore-batch` is set.
pub fn collapse_to_single_batch(membership: &mut [Box<str>]) {
    let label: Box<str> = "all".into();
    for tag in membership.iter_mut() {
        *tag = label.clone();
    }
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

    use matrix_util::common_io::read_lines;

    for files in args.data_files.chunks(nfiles_per_type) {
        let mut data_vec = SparseIoVec::new();
        let attach_data_name = files.len() > 1;

        for (data_idx, data_file) in files.iter().enumerate() {
            info!("Importing data file: {data_file}");

            let mut data = try_open_or_convert(data_file)?;

            if args.preload {
                data.preload_columns()?;
            }
            let data_name = attach_data_name.then(|| data_idx.to_string().into_boxed_str());
            data_vec.push(Arc::from(data), data_name)?;
        }

        data_stack.push(data_vec)?;
    }

    let mut batch_stack = vec![];

    if let Some(batch_files) = &args.batch_files {
        for (data_vec, batch_file) in data_stack.stack.iter_mut().zip(batch_files) {
            let ntot = data_vec.num_columns();
            let mut batch_membership = Vec::with_capacity(ntot);

            info!("Reading batch file: {batch_file}");
            for s in read_lines(batch_file)? {
                batch_membership.push(s.to_string().into_boxed_str());
            }
            batch_stack.push(batch_membership);
        }
    } else {
        for data_vec in &mut data_stack.stack {
            let mut batch_membership = Vec::with_capacity(data_vec.num_columns());
            for (id, &nn) in data_vec.num_columns_by_data()?.iter().enumerate() {
                batch_membership.extend(vec![id.to_string().into_boxed_str(); nn]);
            }
            batch_stack.push(batch_membership);
        }
    }

    Ok(SparseStackWithBatch {
        data_stack,
        batch_stack,
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
