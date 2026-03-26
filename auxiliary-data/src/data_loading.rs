use std::sync::Arc;

use log::info;

use data_beans::convert::try_open_or_convert;
use data_beans::sparse_io_vector::SparseIoVec;
use matrix_util::common_io::{self, basename, read_lines};

/// Arguments for loading multiple sparse data files with shared row names.
pub struct ReadSharedRowsArgs {
    pub data_files: Vec<Box<str>>,
    pub batch_files: Option<Vec<Box<str>>>,
    pub preload: bool,
}

/// Sparse data with per-cell batch labels.
pub struct SparseDataWithBatch {
    pub data: SparseIoVec,
    pub batch: Vec<Box<str>>,
}

/// Load multiple sparse data files, verify shared row names, and auto-detect batch.
///
/// Batch assignment priority:
/// 1. Explicit batch files (one label per cell per file)
/// 2. Embedded `@`-separated batch info in column names (e.g., `barcode@donor`)
/// 3. File name as batch label (one batch per input file)
pub fn read_data_on_shared_rows(args: ReadSharedRowsArgs) -> anyhow::Result<SparseDataWithBatch> {
    // to avoid duplicate barcodes in the column names
    let attach_data_name = args.data_files.len() > 1;

    let mut data_vec = SparseIoVec::new();
    for data_file in args.data_files.iter() {
        info!("Importing data file: {}", data_file);

        let mut data = try_open_or_convert(data_file)?;

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
        let column_names = data_vec.column_names()?;
        let mut col_start = 0usize;

        for (file_idx, &ncols) in column_counts.iter().enumerate() {
            let data_file = args.data_files[file_idx].clone();
            let (_dir, file_base, _ext) = common_io::dir_base_ext(&data_file)?;

            let col_end = col_start + ncols;
            let file_columns = &column_names[col_start..col_end];

            // Check if column names contain '@' (embedded batch info)
            let has_embedded_batch = file_columns.first().is_some_and(|name| name.contains('@'));

            if has_embedded_batch {
                // Parse batch from column names
                for col_name in file_columns {
                    let embedded_batch = col_name.rsplit('@').next().unwrap_or(col_name.as_ref());

                    // If multiple files, combine embedded batch with file name
                    let batch = if num_files > 1 {
                        format!("{}@{}", embedded_batch, file_base).into_boxed_str()
                    } else {
                        embedded_batch.to_string().into_boxed_str()
                    };
                    batch_membership.push(batch);
                }
                if num_files > 1 {
                    info!(
                        "File {}: combining embedded batch with file name '{}'",
                        file_idx, file_base
                    );
                } else {
                    info!("Extracting batch from column names (detected '@' separator)");
                }
            } else {
                // Use file name as batch
                info!(
                    "File {}: using file name '{}' as batch",
                    file_idx, file_base
                );
                batch_membership.extend(vec![file_base; ncols]);
            }
            col_start = col_end;
        }
    }

    if batch_membership.len() != data_vec.num_columns() {
        return Err(anyhow::anyhow!(
            "# batch membership {} != # of columns {}",
            batch_membership.len(),
            data_vec.num_columns()
        ));
    }

    Ok(SparseDataWithBatch {
        data: data_vec,
        batch: batch_membership,
    })
}
