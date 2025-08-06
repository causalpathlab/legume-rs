use crate::embed_common::*;
use matrix_util::common_io::{self, basename, extension, read_lines};

//////////////////////////////////////////
// read data files and batch membership //
//////////////////////////////////////////
pub struct ReadArgs {
    pub data_files: Vec<Box<str>>,
    pub batch_files: Option<Vec<Box<str>>>,
}

pub struct SparseDataWithBatch {
    pub data: SparseIoVec,
    pub batch: Vec<Box<str>>,
}

pub fn read_sparse_data_with_membership(args: ReadArgs) -> anyhow::Result<SparseDataWithBatch> {
    // push data files and collect batch membership
    let file = args.data_files[0].as_ref();
    let backend = match extension(file)?.to_string().as_str() {
        "h5" => SparseIoBackend::HDF5,
        "zarr" => SparseIoBackend::Zarr,
        _ => SparseIoBackend::Zarr,
    };

    // to avoid duplicate barcodes in the column names
    let attach_data_name = args.data_files.len() > 1;

    let mut data_vec = SparseIoVec::new();
    for data_file in args.data_files.iter() {
        info!("Importing data file: {}", data_file);

        match extension(data_file)?.as_ref() {
            "zarr" => {
                assert_eq!(backend, SparseIoBackend::Zarr);
            }
            "h5" => {
                assert_eq!(backend, SparseIoBackend::HDF5);
            }
            _ => return Err(anyhow::anyhow!("Unknown file format: {}", data_file)),
        };

        let data = open_sparse_matrix(data_file, &backend)?;
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
        for (id, &nn) in data_vec.num_columns_by_data()?.iter().enumerate() {
            let data_file = args.data_files[id].clone();
            let (_dir, base, _ext) = common_io::dir_base_ext(&data_file)?;
            batch_membership.extend(vec![base; nn]);
        }
    }

    if batch_membership.len() != data_vec.num_columns()? {
        return Err(anyhow::anyhow!(
            "# batch membership {} != # of columns {}",
            batch_membership.len(),
            data_vec.num_columns()?
        ));
    }

    Ok(SparseDataWithBatch {
        data: data_vec,
        batch: batch_membership,
    })
}

/// Build an affine transformation matrix that will help reduce
/// dimensions in training
///
/// * `collapsed`: data matrices derived from collapsing operations
/// * `target_size`: targeting size
pub fn build_row_aggregator(collapsed: &CollapsedOut, target_size: usize) -> anyhow::Result<Mat> {
    if collapsed.mu_observed.nrows() > target_size {
        let log_x_nd = collapsed.mu_adjusted.as_ref().map_or_else(
            || {
                collapsed
                    .mu_observed
                    .posterior_log_mean()
                    .transpose()
                    .clone()
            },
            |x| x.posterior_log_mean().transpose().clone(),
        );

        let kk = target_size.ilog2() as usize;
        info!(
            "reduce data features: {} -> {}",
            log_x_nd.ncols(),
            target_size,
        );

        let membership = row_membership_matrix(binary_sort_columns(&log_x_nd, kk)?)?;

        if membership.ncols() != target_size {
            let d_available = membership.ncols().min(target_size);
            let mut ret = Mat::zeros(membership.nrows(), target_size);
            ret.columns_range_mut(0..d_available)
                .copy_from(&membership.columns_range(0..d_available));
            Ok(ret)
        } else {
            Ok(membership)
        }
    } else {
        Ok(Mat::identity(collapsed.mu_observed.nrows(), target_size))
    }
}
