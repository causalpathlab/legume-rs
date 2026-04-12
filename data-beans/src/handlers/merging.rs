use crate::handlers::transformation::{run_squeeze, RowAlignMode, RunSqueezeArgs};
use crate::hdf5_io::*;
use crate::sparse_io::*;
use crate::utilities::io_helpers::{read_col_names, read_row_names};

use clap::Args;
use data_beans::sparse_data_visitors::create_jobs;
use data_beans::zarr_io::{apply_zip_flag, finalize_zarr_output, materialize_writable_backend};
use indicatif::ProgressBar;
use log::info;
use matrix_util::common_io::*;
use matrix_util::mtx_io;
use rayon::prelude::*;

use rustc_hash::FxHashMap as HashMap;

#[derive(Args, Debug)]
pub struct AlignDataArgs {
    /// data file -- either `.zarr` or `.h5`
    #[arg(required = true, value_delimiter = ',')]
    pub data_files: Vec<Box<str>>,

    /// Data types (treating them as different rows)
    #[arg(short = 'r', long, required = true)]
    pub num_data_types: usize,

    /// output directory
    #[arg(short, long, required = true)]
    pub output_directory: Box<str>,
}

#[derive(Args, Debug)]
pub struct MergeBackendArgs {
    #[arg(
        value_delimiter = ',',
        help = "Input data files",
        long_help = "Data files to be merged into a single backend. \n\
		     Provide one or more files in supported formats."
    )]
    pub data_files: Vec<Box<str>>,

    #[arg(
        long,
        value_enum,
        default_value = "zarr",
        help = "Backend format",
        long_help = "Specify the backend format to use for the merged data. \n\
		     Supported formats include 'zarr', 'h5', etc."
    )]
    pub backend: SparseIoBackend,

    #[arg(
        short,
        long,
        required = true,
        help = "Output file header",
        long_help = "Output file header: {output}.{backend} and {output}.batch.gz. \n\
		     With --zip, zarr backend produces {output}.zarr.zip instead of {output}.zarr.\n\
		     The backend will contain everything. \n\
		     Batch assignment information will be saved in a separate file \n\
		     and is needed for embedding steps later."
    )]
    pub output: Box<str>,

    /// produce a `.zarr.zip` archive instead of a `.zarr` directory
    #[arg(long, default_value_t = false)]
    pub zip: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Squeeze sparse rows/columns",
        long_help = "Enable squeezing to remove rows and columns with too few non-zeros. \n\
		     This can help reduce file size and improve performance."
    )]
    pub do_squeeze: bool,

    #[arg(
        long,
        default_value_t = 1,
        help = "Row non-zero cutoff",
        long_help = "Minimum number of non-zero elements required for rows. \n\
		     Rows with fewer non-zeros will be removed if squeezing is enabled."
    )]
    pub row_nnz_cutoff: usize,

    #[arg(
        long,
        default_value_t = 1,
        help = "Column non-zero cutoff",
        long_help = "Minimum number of non-zero elements required for columns. \n\
		     Columns with fewer non-zeros will be removed if squeezing is enabled."
    )]
    pub column_nnz_cutoff: usize,

    #[arg(
        long,
        default_value = "100",
        help = "Block size for parallel processing",
        long_help = "Block size for parallel processing. \n\
		     Adjust this value to optimize performance for your hardware."
    )]
    pub block_size: usize,
}

#[derive(clap::Args, Debug)]
pub struct MergeMtxArgs {
    #[arg(
        value_delimiter = ',',
        required = true,
        help = "Input data directories",
        long_help = "Within each directory and its sub-directories, \n\
		     the program will search for files named as specified by \n\
                     (1) `mtx_file_name`, \n\
		     (2) `feature_file_name`, \n\
		     and (3) `barcode_file_name` \n\
		     to merge into one backend file."
    )]
    pub data_directories: Vec<Box<str>>,

    #[arg(
        long,
        value_enum,
        default_value = "zarr",
        help = "Backend format",
        long_help = "Specify the backend format for the merged data. \n\
                     Supported formats include 'zarr', 'h5', etc."
    )]
    pub backend: SparseIoBackend,

    #[arg(
        short,
        long,
        required = true,
        help = "Output file header",
        long_help = "Output file header: {output}.{backend} and {output}.batch.gz. \n\
                     With --zip, zarr backend produces {output}.zarr.zip instead of {output}.zarr.\n\
                     The backend will contain all merged data. \n\
                     Batch assignment information will be saved in a separate file and \n\
		     is needed for embedding steps later."
    )]
    pub output: Box<str>,

    /// produce a `.zarr.zip` archive instead of a `.zarr` directory
    #[arg(long, default_value_t = false)]
    pub zip: bool,

    #[arg(
        short,
        long,
        default_value = "matrix.mtx",
        help = "Matrix file name",
        long_help = "Name of the matrix file to search for in each directory. \n\
                     The default for 10x data is 'matrix.mtx'."
    )]
    pub mtx_file_name: Box<str>,

    #[arg(
        short,
        long,
        default_value = "genes.tsv.gz",
        help = "Feature/row file name",
        long_help = "Name of the feature (row) file to search for in each directory. \n\
                     The default is 'genes.tsv.gz'."
    )]
    pub feature_file_name: Box<str>,

    #[arg(
        long,
        default_value_t = 2,
        help = "Number of words for feature names",
        long_help = "Number of words to use when parsing feature names from the feature file. \n\
                     Adjust this to match your data format."
    )]
    pub num_feature_name_words: usize,

    #[arg(
        short,
        long,
        default_value = "barcodes.tsv.gz",
        help = "Barcode/column file name",
        long_help = "Name of the barcode (column) file to search for in each directory. \n\
                     The default is 'barcodes.tsv.gz'."
    )]
    pub barcode_file_name: Box<str>,

    #[arg(
        long,
        default_value_t = 5,
        help = "Number of words for barcode names",
        long_help = "Number of words to use when parsing barcode names from the barcode file. \n\
                     Adjust this to match your data format."
    )]
    pub num_barcode_name_words: usize,

    #[arg(
        long,
        default_value_t = false,
        help = "Squeeze sparse rows/columns",
        long_help = "Enable squeezing to remove rows and columns with too few non-zeros. \n\
                     This can help reduce file size and improve performance."
    )]
    pub do_squeeze: bool,

    #[arg(
        long,
        default_value_t = 1,
        help = "Row non-zero cutoff",
        long_help = "Minimum number of non-zero elements required for rows. \n\
                     Rows with fewer non-zeros will be removed if squeezing is enabled."
    )]
    pub row_nnz_cutoff: usize,

    #[arg(
        long,
        default_value_t = 1,
        help = "Column non-zero cutoff",
        long_help = "Minimum number of non-zero elements required for columns. \n\
                     Columns with fewer non-zeros will be removed if squeezing is enabled."
    )]
    pub column_nnz_cutoff: usize,

    #[arg(
        long,
        default_value_t = 100,
        help = "Block size for parallel processing and squeeze pass",
        long_help = "Number of columns handled per rayon job during triplet read \n\
                     and the post-merge squeeze pass. \n\
                     Smaller values bound peak memory at the cost of extra scheduling overhead."
    )]
    pub block_size: usize,
}

/// Sort triplets into CSC layout and split into (colptr, row indices, values).
///
/// `colptr[j]` is the local offset of column `j`'s first entry; the trailing
/// sentinel is written by `finalize_streaming_csc`.
fn build_csc_slab(
    mut triplets: Vec<(u64, u64, f32)>,
    ncol: usize,
) -> (Vec<u64>, Vec<u64>, Vec<f32>) {
    triplets.par_sort_by_key(|&(row, col, _)| (col, row));

    let mut local_colptr = vec![0u64; ncol];
    let mut col_counts = vec![0u64; ncol];
    for &(_, col_local, _) in &triplets {
        col_counts[col_local as usize] += 1;
    }
    let mut acc: u64 = 0;
    for j in 0..ncol {
        local_colptr[j] = acc;
        acc += col_counts[j];
    }
    debug_assert_eq!(acc as usize, triplets.len());

    let row_indices: Vec<u64> = triplets.iter().map(|&(r, _, _)| r).collect();
    let values: Vec<f32> = triplets.iter().map(|&(_, _, v)| v).collect();
    (local_colptr, row_indices, values)
}

pub fn run_merge_backend(args: &MergeBackendArgs) -> anyhow::Result<()> {
    if args.data_files.len() <= 1 {
        info!("no need to merge one file");
        return Ok(());
    }

    let num_batches = args.data_files.len();
    info!("merging over {} batches ...", num_batches);

    let batch_names = generate_unique_batch_names(&args.data_files)?;
    info!("Batch names: {:?}", batch_names);

    struct BatchHandle {
        path: Box<str>,
        backend: SparseIoBackend,
        ncol: usize,
        col_offset: u64,
        nnz_offset: u64,
    }

    let mut batches: Vec<BatchHandle> = Vec::with_capacity(num_batches);
    let mut row_names: Vec<Box<str>> = vec![];
    let mut column_names: Vec<Box<str>> = vec![];
    let mut column_batch_names: Vec<Box<str>> = vec![];
    let mut col_offset: u64 = 0;
    let mut nnz_offset: u64 = 0;

    for (batch_idx, data_file) in args.data_files.iter().enumerate() {
        info!("inventorying data file: {}", data_file);

        let backend = match file_ext(data_file)?.to_string().as_str() {
            "h5" => SparseIoBackend::HDF5,
            "zarr" => SparseIoBackend::Zarr,
            _ => SparseIoBackend::Zarr,
        };

        let data = open_sparse_matrix(data_file, &backend)?;
        let ncol = data
            .num_columns()
            .ok_or_else(|| anyhow::anyhow!("missing ncol in {}", data_file))?;
        let nnz = data
            .num_non_zeros()
            .ok_or_else(|| anyhow::anyhow!("missing nnz in {}", data_file))?;

        if row_names.is_empty() {
            row_names = data.row_names()?;
        } else {
            info!("checking if the row names are consistent");
            assert_eq!(row_names, data.row_names()?);
        }

        let _names = data.column_names()?;
        let batch_name = &batch_names[batch_idx];
        column_names.extend(
            _names
                .into_iter()
                .map(|x| format!("{}{}{}", x, COLUMN_SEP, batch_name).into_boxed_str()),
        );
        column_batch_names.extend(vec![batch_name.clone(); ncol]);

        batches.push(BatchHandle {
            path: data_file.clone(),
            backend,
            ncol,
            col_offset,
            nnz_offset,
        });

        col_offset += ncol as u64;
        nnz_offset += nnz as u64;
    }

    let total_ncol = col_offset as usize;
    let total_nnz = nnz_offset as usize;
    let total_nrow = row_names.len();
    info!(
        "Found {} columns/barcodes across {} batches (total nnz = {})",
        total_ncol, num_batches, total_nnz
    );

    let effective_output = apply_zip_flag(&args.output, args.zip);
    let (backend, backend_file) =
        resolve_backend_file(&effective_output, Some(args.backend.clone()))?;

    if std::path::Path::new(backend_file.as_ref()).exists() {
        info!(
            "This existing backend file '{}' will be deleted",
            &backend_file
        );
        remove_file(&backend_file)?;
    }

    let mut out = create_sparse_streaming_empty(Some(&backend_file), Some(&backend))?;
    out.begin_streaming_csc((total_nrow, total_ncol, total_nnz))?;

    let pb = ProgressBar::new(num_batches as u64);
    for h in &batches {
        let src = open_sparse_matrix(&h.path, &h.backend)?;
        let jobs = create_jobs(h.ncol, Some(args.block_size));
        let triplets_batch: Vec<(u64, u64, f32)> = jobs
            .par_iter()
            .filter_map(|(lb, ub)| {
                src.read_triplets_by_columns((*lb..*ub).collect())
                    .ok()
                    .map(|(_, _, t)| {
                        let lb_u64 = *lb as u64;
                        t.into_iter()
                            .map(move |(i, j_local, x)| (i, j_local + lb_u64, x))
                            .collect::<Vec<_>>()
                    })
            })
            .flatten()
            .collect();

        let (local_colptr, row_indices, values) = build_csc_slab(triplets_batch, h.ncol);
        out.append_csc_slab(
            h.col_offset,
            h.nnz_offset,
            &local_colptr,
            &row_indices,
            &values,
        )?;

        pb.inc(1);
    }
    pb.finish_and_clear();

    out.finalize_streaming_csc()?;
    info!("transposing CSC → CSR on disk");
    out.build_csr_from_csc_streaming()?;

    out.register_row_names_vec(&row_names);
    out.register_column_names_vec(&column_names);
    drop(out);

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
            row_align: RowAlignMode::Common,
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

    finalize_zarr_output(&backend_file, &args.output)?;
    info!("done");
    Ok(())
}

pub fn run_merge_mtx(args: &MergeMtxArgs) -> anyhow::Result<()> {
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
    let mut batch_ncols: Vec<usize> = Vec::with_capacity(num_batches);

    for (col_file, batch_name) in col_files.iter().zip(batch_names.iter()) {
        let _names = read_col_names(col_file.clone(), args.num_barcode_name_words)?;
        let nn = _names.len();
        batch_ncols.push(nn);
        column_names.extend(
            _names
                .into_iter()
                .map(|x| format!("{}{}{}", x, COLUMN_SEP, batch_name).into_boxed_str())
                .collect::<Vec<_>>(),
        );

        column_batch_names.extend(vec![batch_name.clone().into_boxed_str(); nn]);
    }

    info!("Found {} columns/barcodes ...", column_names.len());

    info!("Sizing per-batch nnz ...");
    let mut batch_nnz: Vec<usize> = Vec::with_capacity(num_batches);
    let pb1 = ProgressBar::new(num_batches as u64);
    for b in 0..num_batches {
        let row_names = read_row_names(row_files[b].clone(), args.num_feature_name_words)?;
        let (_nrow, _ncol, header_nnz) = mtx_io::read_mtx_header(&mtx_files[b])?;

        // Skip the parse if every row in this batch is kept — header nnz is exact.
        let no_filter = row_names.iter().all(|n| row_pos.contains_key(n));
        let count = if no_filter {
            header_nnz
        } else {
            let (triplets, _shape) = mtx_io::read_mtx_triplets(&mtx_files[b].clone())?;
            triplets
                .iter()
                .filter(|(batch_i, _, _)| row_pos.contains_key(&row_names[*batch_i as usize]))
                .count()
        };
        batch_nnz.push(count);
        pb1.inc(1);
    }
    pb1.finish_and_clear();

    let total_nnz: usize = batch_nnz.iter().sum();
    let total_ncol: usize = batch_ncols.iter().sum();
    let total_nrow = common_rows.len();

    info!(
        "Total filtered nnz = {} across {} batches",
        total_nnz, num_batches
    );

    let backend = args.backend.clone();
    let effective_output = apply_zip_flag(&args.output, args.zip);
    let output = args.output.clone();
    let batch_memb_file = (output.to_string() + ".batch.gz").into_boxed_str();

    let (_, backend_file) = resolve_backend_file(&effective_output, Some(backend.clone()))?;
    let backend_file: String = backend_file.to_string();

    if std::path::Path::new(&backend_file).exists() {
        info!(
            "This existing backend file '{}' will be deleted",
            &backend_file
        );
        remove_file(&backend_file)?;
    }

    let mut out = create_sparse_streaming_empty(Some(&backend_file), Some(&backend))?;
    out.begin_streaming_csc((total_nrow, total_ncol, total_nnz))?;

    let mut col_offset: u64 = 0;
    let mut nnz_offset: u64 = 0;
    let pb2 = ProgressBar::new(num_batches as u64);
    for b in 0..num_batches {
        let row_names = read_row_names(row_files[b].clone(), args.num_feature_name_words)?;
        let (triplets, _shape) = mtx_io::read_mtx_triplets(&mtx_files[b].clone())?;

        let triplets_batch: Vec<(u64, u64, f32)> = triplets
            .into_iter()
            .filter_map(|(batch_i, batch_j, x_ij)| {
                row_pos
                    .get(&row_names[batch_i as usize])
                    .map(|i| (*i as u64, batch_j, x_ij))
            })
            .collect();

        debug_assert_eq!(triplets_batch.len(), batch_nnz[b]);

        let ncol_b = batch_ncols[b];
        let (local_colptr, row_indices, values) = build_csc_slab(triplets_batch, ncol_b);
        out.append_csc_slab(col_offset, nnz_offset, &local_colptr, &row_indices, &values)?;

        col_offset += ncol_b as u64;
        nnz_offset += batch_nnz[b] as u64;
        pb2.inc(1);
    }
    pb2.finish_and_clear();

    out.finalize_streaming_csc()?;
    info!("transposing CSC → CSR on disk");
    out.build_csr_from_csc_streaming()?;

    out.register_row_names_vec(&common_rows);
    out.register_column_names_vec(&column_names);
    drop(out);

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
            block_size: args.block_size,
            preload: true,
            show_histogram: false,
            save_histogram: None,
            dry_run: false,
            interactive: false,
            output: None,
            row_align: RowAlignMode::Common,
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

    finalize_zarr_output(&backend_file, &effective_output)?;
    info!("done");
    Ok(())
}

/// Generate unique batch names from file paths
/// If basenames are unique, use them as-is
/// If duplicates exist, add numeric suffixes
pub fn generate_unique_batch_names(files: &[Box<str>]) -> anyhow::Result<Vec<Box<str>>> {
    use rustc_hash::FxHashMap as HashMap;

    // Extract basenames
    let basenames: Vec<_> = files
        .iter()
        .map(|f| basename(f))
        .collect::<anyhow::Result<Vec<_>>>()?;

    // Count occurrences of each basename
    let mut counts: HashMap<&str, usize> = Default::default();
    for name in &basenames {
        *counts.entry(name.as_ref()).or_insert(0) += 1;
    }

    // Generate unique names
    let mut name_counters: HashMap<&str, usize> = Default::default();
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
    mode: RowAlignMode,
) -> anyhow::Result<Vec<Box<str>>> {
    use rayon::prelude::*;
    use rustc_hash::FxHashMap as HashMap;

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
        RowAlignMode::Common => {
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
        RowAlignMode::Union => {
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
            // Use a writable extension (`.zarr`/`.h5`) regardless of whether
            // the input is `.zarr.zip`, since `subset_columns_rows` below
            // mutates the backend and read-only zip stores cannot absorb writes.
            let dst_ext = if a_file.ends_with(".zarr.zip") || a_file.ends_with(".zarr") {
                "zarr"
            } else {
                "h5"
            };
            let base = basename(a_file)?;

            let data_col_id = i % n_data_columns + 1;
            let data_row_id = i / n_data_columns + 1;
            let dst_path = format!(
                "{}/{}_{}_{}.{}",
                args.output_directory, data_row_id, data_col_id, base, dst_ext
            );
            info!("staging aligned copy: {}", dst_path);
            materialize_writable_backend(a_file, &dst_path)?;
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
