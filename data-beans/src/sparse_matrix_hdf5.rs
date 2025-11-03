// #![allow(dead_code)]

use crate::sparse_io::*;
use hdf5::filters::blosc_set_nthreads;
use log::info;
use matrix_util::common_io::*;
use std::ops::Range;
use std::sync::Arc;

use anyhow::anyhow;

const NUM_CHUNKS: usize = 1000;
const MIN_CHUNK_SIZE: usize = 8192;
const COMPRESSION_LEVEL: u8 = 5;

/// 10x-like cell-feature matrix with hdf5 (feature x cell)
///
/// ```text
/// (root)
///     ├── nrow
///     ├── ncell
///     ├── by_column
///     │   ├── data
///     │   ├── indices (row indices)
///     │   └── indptr (column pointers)
///     └── by_row
///         ├── data
///         ├── indices (column indices)
///         └── indptr (row pointers)
/// ```
///
#[derive(Debug, Clone)]
pub struct SparseMtxData {
    backend: Arc<hdf5::File>,
    file_name: String,
    max_row_name_idx: usize,
    max_column_name_idx: usize,
    by_column_indptr: Vec<u64>,
    by_row_indptr: Vec<u64>,
    by_column_indicies: Option<Vec<u64>>,
    by_column_data: Option<Vec<f32>>,
}

impl SparseMtxData {
    /// Create an empty new `SparseMtxData` with HDF5 file. If no
    /// `backend_file` is provided, a temporary file will be created.
    ///
    /// * `backend_file`: HDF5 file to be associated with
    pub fn new(backend_file: Option<&str>) -> anyhow::Result<Self> {
        let ret = match backend_file {
            Some(backend_file) => Self::register_backend_file(backend_file)?,
            None => {
                let backend_file = create_temp_dir_file(".h5")?;
                let backend_file = backend_file.to_str().expect("to_str failed");
                Self::register_backend_file(backend_file)?
            }
        };
        Ok(ret)
    }

    /// Open existing `SparseMtxData` from a backend HDF5 file
    pub fn open(backend_file: &str) -> anyhow::Result<Self> {
        let hdf5_backend = hdf5::File::open(backend_file)?;

        if let (Some(nrow), Some(ncol), Some(nnz)) = (
            Self::_num_rows(&hdf5_backend),
            Self::_num_columns(&hdf5_backend),
            Self::_num_nnz(&hdf5_backend),
        ) {
            info!("#rows: {}, #columns: {}, #non-zeros: {}", nrow, ncol, nnz);
        } else {
            anyhow::bail!("Couldn't figure out the size of this sparse matrix data");
        }

        let mut ret = Self {
            backend: hdf5_backend.into(),
            file_name: backend_file.to_string(),
            max_row_name_idx: MAX_ROW_NAME_IDX,
            max_column_name_idx: MAX_COLUMN_NAME_IDX,
            by_column_indptr: vec![],
            by_row_indptr: vec![],
            by_column_indicies: None,
            by_column_data: None,
        };

        ret.read_column_indptr()?;
        ret.read_row_indptr()?;

        Ok(ret)
    }

    /// Create `SparseMtxData` from mtx file with `backend_file` as
    /// the backend file.  If no `backend_file` is provided, it will
    /// be the same as `mtx_file` with `.h5` extension.
    /// * `mtx_file`: mtx file to be read into HDF5 backend
    /// * `backend_file`: HDF5 file to be associated with
    /// * `index_by_row`: if true, the matrix will be indexed by row
    pub fn from_mtx_file(
        mtx_file: &str,
        backend_file: Option<&str>,
        index_by_row: Option<bool>,
    ) -> anyhow::Result<Self> {
        // create an object
        let mut ret = match backend_file {
            Some(backend_file) => {
                info!("backend file : {}", backend_file);
                Self::register_backend_file(backend_file)?
            }
            None => {
                let backend_file = mtx_file.to_string() + ".h5";
                info!("backend file : {}", backend_file);
                Self::register_backend_file(backend_file.as_ref())?
            }
        };

        // populate data from mtx file
        info!("importing mtx file by column");
        ret.import_mtx_file_by_col(mtx_file)?;
        ret.read_column_indptr()?;

        if Some(true) == index_by_row {
            info!("importing mtx file by row");
            ret.import_mtx_file_by_row(mtx_file)?;
            ret.read_row_indptr()?;
        }

        info!("created sparse backend from {}", mtx_file);
        Ok(ret)
    }

    /// Create a new `SparseMtxData` from ndarray with its backend
    /// HDF5 file. If no `backend_file` is provided, a temporary file
    /// will be created.
    /// * `array`: 2D array to be written into HDF5 backend
    /// * `backend_file`: HDF5 file to be associated with
    /// * `index_by_row`: if true, the matrix will be indexed by row
    ///
    pub fn from_ndarray(
        array: &Array2<f32>,
        backend_file: Option<&str>,
        index_by_row: Option<bool>,
    ) -> anyhow::Result<Self> {
        let mut ret = match backend_file {
            Some(backend_file) => Self::register_backend_file(backend_file)?,
            None => {
                let backend_file = create_temp_dir_file(".h5")?;
                let backend_file = backend_file.to_str().expect("to_str failed");
                Self::register_backend_file(backend_file)?
            }
        };

        ret.import_ndarray_by_col(array)?; // populate data from mtx file
        ret.read_column_indptr()?; // reload column indptr

        if Some(true) == index_by_row {
            ret.import_ndarray_by_row(array)?; //
            ret.read_row_indptr()?; //
        }

        Ok(ret)
    }

    /// Create a new `SparseMtxData` from DMatrix with its backend
    /// HDF5 file. If no `backend_file` is provided, a temporary file
    /// will be created.
    /// * `array`: 2D array to be written into HDF5 backend
    /// * `backend_file`: HDF5 file to be associated with
    /// * `index_by_row`: if true, the matrix will be indexed by row
    ///
    pub fn from_dmatrix(
        matrix: &DMatrix<f32>,
        backend_file: Option<&str>,
        index_by_row: Option<bool>,
    ) -> anyhow::Result<Self> {
        let mut ret = match backend_file {
            Some(backend_file) => Self::register_backend_file(backend_file)?,
            None => {
                let backend_file = create_temp_dir_file(".h5")?;
                let backend_file = backend_file.to_str().expect("to_str failed");
                Self::register_backend_file(backend_file)?
            }
        };

        ret.import_dmatrix_by_col(matrix)?; // populate data from mtx file
        ret.read_column_indptr()?; // reload column indptr

        if Some(true) == index_by_row {
            ret.import_dmatrix_by_row(matrix)?; //
            ret.read_row_indptr()?; //
        }

        Ok(ret)
    }

    /////////////////////////////
    // purely helper functions //
    /////////////////////////////

    fn _num_rows(file: &hdf5::File) -> Option<usize> {
        file.attr("nrow").ok()?.read_scalar().ok()
    }

    fn _num_columns(file: &hdf5::File) -> Option<usize> {
        file.attr("ncol").ok()?.read_scalar().ok()
    }

    fn _num_nnz(file: &hdf5::File) -> Option<usize> {
        file.attr("nnz").ok()?.read_scalar().ok()
    }

    fn set_attrs(&mut self, attr_name: &str, value: usize) -> anyhow::Result<()> {
        if self.backend.attr(attr_name).is_err() {
            self.backend
                .new_attr::<usize>()
                .create(attr_name)?
                .write_scalar(&value)?;
        } else if self.backend.attr(attr_name)?.read_scalar::<usize>()? != value {
            return Err(anyhow!(format!("{} mismatch", attr_name)));
        }
        Ok(())
    }

    //////////////////////
    // backend related  //
    //////////////////////

    /// Associate sparse matrix data with a HDF5 file
    /// * `hdf5_file`: HDF5 file to be associated with
    fn register_backend_file(hdf5_file: &str) -> anyhow::Result<Self> {
        let hdf5_backend = hdf5::File::create(hdf5_file)?;

        Ok(Self {
            backend: hdf5_backend.into(),
            file_name: hdf5_file.to_string(),
            max_row_name_idx: MAX_ROW_NAME_IDX,
            max_column_name_idx: MAX_COLUMN_NAME_IDX,
            by_column_indptr: vec![],
            by_row_indptr: vec![],
            by_column_indicies: None,
            by_column_data: None,
        })
    }
}

impl SparseIo for SparseMtxData {
    type IndexIter = Vec<usize>;

    /// Helper function to create a new backend file
    fn initialize_backend(&mut self) -> anyhow::Result<()> {
        self.remove_backend_file()?;
        self.backend = hdf5::File::create(&self.file_name)?.into();
        self.max_column_name_idx = MAX_COLUMN_NAME_IDX;
        self.max_row_name_idx = MAX_ROW_NAME_IDX;
        self.by_column_indptr = vec![];
        self.by_row_indptr = vec![];

        Ok(())
    }

    /// Record the shape of the mtx file into the HDF5 backend
    fn record_mtx_shape(&mut self, mtx_shape: Option<(usize, usize, usize)>) -> anyhow::Result<()> {
        if let Some((nrow, ncol, nnz)) = mtx_shape {
            self.set_attrs("nrow", nrow)?;
            self.set_attrs("ncol", ncol)?;
            self.set_attrs("nnz", nnz)?;
            self.backend.flush()?;
        }

        Ok(())
    }

    /// Read column index pointers
    fn read_column_indptr(&mut self) -> anyhow::Result<()> {
        if let Ok(by_column) = self.backend.group("/by_column") {
            let indptr = by_column.dataset("indptr")?.read_1d::<u64>()?;
            self.by_column_indptr.clear();
            self.by_column_indptr.extend(indptr);
        }
        Ok(())
    }

    /// Read row index pointers
    fn read_row_indptr(&mut self) -> anyhow::Result<()> {
        if let Ok(by_row) = self.backend.group("/by_row") {
            let indptr = by_row.dataset("indptr")?.read_1d::<u64>()?;
            self.by_row_indptr.clear();
            self.by_row_indptr.extend(indptr);
        }
        Ok(())
    }

    fn preload_columns(&mut self) -> anyhow::Result<()> {
        let by_column = self.backend.group("/by_column")?;
        let data = by_column.dataset("data")?.read_1d::<f32>()?.to_vec();
        let indices = by_column.dataset("indices")?.read_1d::<u64>()?.to_vec();

        self.by_column_data = Some(data);
        self.by_column_indicies = Some(indices);
        Ok(())
    }

    fn clean_preloaded_columns(&mut self) {
        self.by_column_data = None;
        self.by_column_indicies = None;
    }

    /// Remove backend file to free up disk space
    fn remove_backend_file(&self) -> anyhow::Result<()> {
        let backend = std::path::Path::new(&self.file_name);
        if backend.exists() {
            std::fs::remove_file(backend)?;
        }
        Ok(())
    }

    /// Access file name of the hdf5 backend file
    fn get_backend_file_name(&self) -> &str {
        &self.file_name
    }

    fn backend_type(&self) -> SparseIoBackend {
        SparseIoBackend::HDF5
    }

    /// Export the data to a mtx file. This will take time.
    /// * `mtx_file`: mtx file to be written
    fn to_mtx_file(&self, mtx_file: &str) -> anyhow::Result<()> {
        let by_column = self.backend.group("/by_column")?;

        let indptr = by_column.dataset("indptr")?.read_1d::<u64>()?;
        let data = by_column.dataset("data")?;
        let indices = by_column.dataset("indices")?;

        if let (Some(ncol), Some(nrow), Some(nnz)) =
            (self.num_columns(), self.num_rows(), self.num_non_zeros())
        {
            let mut buf = open_buf_writer(mtx_file)?;
            writeln!(buf, "%%MatrixMarket matrix coordinate real general")?;
            writeln!(buf, "{}\t{}\t{}", nrow, ncol, nnz)?;

            for jj in 0..ncol {
                let start = indptr[jj] as usize;
                let end = indptr[jj + 1] as usize;

                let data_slice = data.read_slice_1d::<f32, _>(start..end)?;
                let indices_slice = indices.read_slice_1d::<u64, _>(start..end)?;
                // write them with 1-based indices
                for k in 0..(end - start) {
                    let val = data_slice[k];
                    let ii = indices_slice[k] as usize;
                    writeln!(buf, "{}\t{}\t{}", ii + 1, jj + 1, val)?;
                }
            }
            buf.flush()?;
            info!(
                "{}: {} rows, {} columns, {} non-zeros",
                mtx_file, nrow, ncol, nnz
            );

            Ok(())
        } else {
            Err(anyhow!("Unable to figure out the size of the backend data"))
        }
    }

    /// Set row names for the matrix
    /// * `row_name_file`: a file each line contains row name words
    fn register_row_names_file(&mut self, row_name_file: &str) {
        self.register_names_file(
            "/row_names",
            row_name_file,
            0..self.max_row_name_idx,
            ROW_SEP,
        )
        .expect("failed to add row names");
    }

    /// Set row names for the matrix
    /// * `rows`: a vector of row names
    fn register_row_names_vec(&mut self, rows: &[Box<str>]) {
        self.register_names_vec("/row_names", rows)
            .expect("failed to add row names");
    }

    /// Set column names for the matrix
    /// * `column_name_file`: a file each line contains column name words
    fn register_column_names_file(&mut self, column_name_file: &str) {
        self.register_names_file(
            "/column_names",
            column_name_file,
            0..self.max_column_name_idx,
            COLUMN_SEP,
        )
        .expect("failed to add column names");
    }

    /// Set column names for the matrix
    /// * `columns`: a vector of column names
    fn register_column_names_vec(&mut self, columns: &[Box<str>]) {
        self.register_names_vec("/column_names", columns)
            .expect("failed to add column names");
    }

    /// Number of rows in the underlying data matrix
    fn num_rows(&self) -> Option<usize> {
        Self::_num_rows(&self.backend)
    }

    /// Number of columns in the underlying data matrix
    fn num_columns(&self) -> Option<usize> {
        Self::_num_columns(&self.backend)
    }

    /// Number of non-zero elements
    fn num_non_zeros(&self) -> Option<usize> {
        Self::_num_nnz(&self.backend)
    }

    /// Add arbitrary names (a vector of strings)
    /// * `group_name`: group name
    /// * `name_file`: a file each line contains name words
    /// * `name_columns`: range of columns to be used for name
    /// * `name_sep`: separator for name columns
    fn register_names_file(
        &mut self,
        key: &str,
        name_file: &str,
        name_columns: Range<usize>,
        name_sep: &str,
    ) -> anyhow::Result<()> {
        use hdf5::types::VarLenUnicode;

        let _names = read_lines_of_words(name_file, -1)?.lines;
        let name_columns = name_columns.clone().collect::<Vec<_>>();
        let _names: Vec<VarLenUnicode> = _names
            .iter()
            .map(|x| {
                name_columns
                    .iter()
                    .filter_map(|&i| x.get(i))
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(name_sep)
                    .parse()
                    .expect("invalid name")
            })
            .collect();

        let root = self.backend.group("/")?;
        root.new_dataset::<VarLenUnicode>()
            .shape(_names.len())
            .chunk([_names.len()])
            .create(key)?
            .write(&_names)?;

        Ok(())
    }

    /// Add arbitrary names (a vector of strings)
    /// * `group_name`: group name
    /// * `names`: a file each line contains name words
    fn register_names_vec(&mut self, key: &str, names: &[Box<str>]) -> anyhow::Result<()> {
        use hdf5::types::VarLenUnicode;

        let _names: Vec<VarLenUnicode> = names
            .iter()
            .map(|x| x.to_string().parse().expect("invalid name"))
            .collect::<Vec<_>>();

        let root = self.backend.group("/")?;
        root.new_dataset::<VarLenUnicode>()
            .shape(_names.len())
            .chunk([_names.len()])
            .create(key)?
            .write(&_names)?;

        Ok(())
    }

    fn row_names(&self) -> anyhow::Result<Vec<Box<str>>> {
        self.retrieve_registered_names("/row_names")
    }

    fn column_names(&self) -> anyhow::Result<Vec<Box<str>>> {
        self.retrieve_registered_names("/column_names")
    }

    /// Get back the registered names
    /// * `key`: key for the registered names
    fn retrieve_registered_names(&self, key: &str) -> anyhow::Result<Vec<Box<str>>> {
        use hdf5::types::VarLenUnicode;

        let root = self.backend.group("/")?;

        let ret = root.dataset(key)?.read_1d::<VarLenUnicode>()?;

        Ok(ret.iter().map(|x| x.to_string().into_boxed_str()).collect())
    }

    /// Read columns within the range and return a vector of triplets (row, col, value)
    /// * `col` : usize
    ///
    fn read_triplets_by_single_column(
        &self,
        j_data: usize,
    ) -> anyhow::Result<(usize, usize, Vec<(u64, u64, f32)>)> {
        let by_column = self.backend.group("/by_column")?;
        debug_assert!(!self.by_column_indptr.is_empty());

        let indptr = &self.by_column_indptr;
        debug_assert!((j_data + 1) < indptr.len());

        let nrow = self
            .num_rows()
            .ok_or(anyhow!("can't figure out the number of rows"))?;

        if let (Some(data), Some(indices)) = (&self.by_column_data, &self.by_column_indicies) {
            let ncol_out = 1;
            let jj = 0;

            // [start, end)
            let start = indptr[j_data] as usize;
            let end = indptr[j_data + 1] as usize;
            let ret: Vec<(u64, u64, f32)> = indices[start..end]
                .iter()
                .zip(data[start..end].iter())
                .map(|(&ii, &x_ij)| (ii, jj, x_ij))
                .collect();
            Ok((nrow, ncol_out, ret))
        } else {
            let data = by_column.dataset("data")?;
            let indices = by_column.dataset("indices")?;

            let mut ret = Vec::new();
            let ncol_out = 1;
            let jj = 0;

            debug_assert!((j_data + 1) < indptr.len());

            // [start, end)
            let start = indptr[j_data] as usize;
            let end = indptr[j_data + 1] as usize;

            if start < end {
                let data_slice = data.read_slice_1d::<f32, _>(start..end)?;
                let indices_slice = indices.read_slice_1d::<u64, _>(start..end)?;

                for k in 0..(end - start) {
                    let x_ij = data_slice[k];
                    let ii = indices_slice[k];
                    debug_assert!((ii as usize) < nrow);
                    ret.push((ii, jj, x_ij));
                }
            }

            Ok((nrow, ncol_out, ret))
        }
    }

    /// Read columns within the range and return a vector of triplets (row, col, value)
    /// * `columns` : range e.g., 0..3 -> [0, 1, 2] or vec![0, 1, 2]
    ///
    fn read_triplets_by_columns(
        &self,
        columns: Self::IndexIter,
    ) -> anyhow::Result<(usize, usize, Vec<(u64, u64, f32)>)> {
        // need to open backend again?
        // let backend = hdf5::File::open(&self.file_name)?;
        let by_column = self.backend.group("/by_column")?;

        debug_assert!(!self.by_column_indptr.is_empty());

        let indptr = &self.by_column_indptr;

        let columns_vec = columns.into_iter().collect::<Vec<usize>>();

        let nrow = self
            .num_rows()
            .ok_or(anyhow!("can't figure out the number of rows"))?;

        let ncol = self
            .num_columns()
            .ok_or(anyhow!("can't figure out the number of columns"))?;

        let min_start = columns_vec
            .iter()
            .map(|&j_data| indptr[j_data])
            .min()
            .unwrap_or(0);

        let max_end = columns_vec
            .iter()
            .map(|&j_data| indptr[j_data + 1])
            .max()
            .unwrap_or(0);

        if let (Some(data), Some(indices)) = (&self.by_column_data, &self.by_column_indicies) {
            let ncol_out = columns_vec.len();

            let mut ret: Vec<(u64, u64, f32)> = Vec::with_capacity((max_end - min_start) as usize);

            for (jj, &j_data) in columns_vec.iter().enumerate() {
                let jj = jj as u64;
                if j_data < ncol {
                    let start = indptr[j_data] as usize;
                    let end = indptr[j_data + 1] as usize;
                    for (&ii, &x_ij) in indices[start..end].iter().zip(data[start..end].iter()) {
                        ret.push((ii, jj, x_ij));
                    }
                }
            }

            Ok((nrow, ncol_out, ret))
        } else {
            let data = by_column.dataset("data")?;
            let indices = by_column.dataset("indices")?;

            let ncol_out = columns_vec.len();

            let mut ret: Vec<(u64, u64, f32)> = Vec::with_capacity((max_end - min_start) as usize);

            for (jj, &j_data) in columns_vec.iter().enumerate() {
                let jj = jj as u64;
                if j_data < ncol {
                    // [start, end)
                    let start = indptr[j_data] as usize;
                    let end = indptr[j_data + 1] as usize;

                    if start < end {
                        let data_slice = data.read_slice_1d::<f32, _>(start..end)?;
                        let indices_slice = indices.read_slice_1d::<u64, _>(start..end)?;

                        for k in 0..(end - start) {
                            let x_ij = data_slice[k];
                            let ii = indices_slice[k];
                            debug_assert!((ii as usize) < nrow);
                            ret.push((ii, jj, x_ij));
                        }
                    }
                }
            }
            Ok((nrow, ncol_out, ret))
        }
    }

    /// Read rows within the range and return a vector of triplets (row, column, value)
    /// * `rows` : range e.g., 0..3 -> [0, 1, 2] or vec![0, 1, 2]
    ///
    fn read_triplets_by_rows(
        &self,
        rows: Self::IndexIter,
    ) -> anyhow::Result<(usize, usize, Vec<(u64, u64, f32)>)> {
        // need to open backend again?
        // let backend = hdf5::File::open(&self.file_name)?;
        let by_row = self.backend.group("/by_row")?;
        debug_assert!(!self.by_row_indptr.is_empty());
        let indptr = &self.by_row_indptr;
        let data = by_row.dataset("data")?;
        let indices = by_row.dataset("indices")?;

        let rows_vec = rows.into_iter().collect::<Vec<usize>>();

        if let (Some(ncol), Some(nrow)) = (self.num_columns(), self.num_rows()) {
            let mut ret = Vec::new();
            let nrow_out = rows_vec.len();
            for (ii, &i_data) in rows_vec.iter().enumerate() {
                let ii = ii as u64;
                if i_data < nrow {
                    debug_assert!((i_data + 1) < indptr.len());

                    let start = indptr[i_data] as usize;
                    let end = indptr[i_data + 1] as usize;

                    if start < end {
                        let data_slice = data.read_slice_1d::<f32, _>(start..end)?;
                        let indices_slice = indices.read_slice_1d::<u64, _>(start..end)?;

                        for k in 0..(end - start) {
                            let x_ij = data_slice[k];
                            let jj = indices_slice[k];
                            debug_assert!((jj as usize) < ncol);
                            ret.push((ii, jj, x_ij));
                        }
                    }
                }
            }
            Ok((nrow_out, ncol, ret))
        } else {
            Err(anyhow!("Unable to figure out the size of the backend data"))
        }
    }

    /// Helper function to add CSR dataset to HDF5 backend
    ///
    /// ```text
    ///     └── by_row
    ///         ├── data
    ///         ├── indices (column indices)
    ///         └── isndptr (row pointers)
    /// ```
    fn record_csr_dataset_backend(
        &mut self,
        csr_cols: &[u64],
        csr_vals: &[f32],
        csr_rowptr: &[u64],
    ) -> anyhow::Result<()> {
        // Populate them into HDF5
        if self.backend.group("/by_row").is_err() {
            let _root = self.backend.create_group("/by_row")?;
            // info!("Group: {:?} created", root);
        }

        {
            let num_threads = num_cpus::get(); // Gets the number of logical CPUs
            blosc_set_nthreads(num_threads as u8); // Set the number of threads for Blosc
        }

        let csr = self.backend.group("/by_row")?;

        let nelem = csr_vals.len();
        let nchunks = NUM_CHUNKS;
        let chunk_size = (nelem / nchunks).max(MIN_CHUNK_SIZE).min(nelem);

        csr.new_dataset::<f32>()
            .shape(csr_vals.len())
            .chunk([chunk_size])
            .blosc_blosclz(COMPRESSION_LEVEL, true)
            .create("data")?
            .write(&csr_vals)?;

        let nelem = csr_rowptr.len();
        let nchunks = NUM_CHUNKS;
        let chunk_size = (nelem / nchunks).max(MIN_CHUNK_SIZE).min(nelem);

        csr.new_dataset::<u64>()
            .shape(csr_rowptr.len())
            .chunk([chunk_size])
            .blosc_blosclz(COMPRESSION_LEVEL, true)
            .create("indptr")?
            .write(&csr_rowptr)?;

        let nelem = csr_cols.len();
        let nchunks = NUM_CHUNKS;
        let chunk_size = (nelem / nchunks).max(MIN_CHUNK_SIZE).min(nelem);

        csr.new_dataset::<u64>()
            .shape(csr_cols.len())
            .chunk([chunk_size])
            .blosc_blosclz(COMPRESSION_LEVEL, true)
            .create("indices")?
            .write(&csr_cols)?;

        self.backend.flush()?;

        Ok(())
    }

    /// Helper function to add CSC dataset to HDF5 backend
    ///
    /// ```text
    /// Helper function to record the CSC dataset
    ///     ├── by_column
    ///     │   ├── data
    ///     │   ├── indices (row indices)
    ///     │   └── indptr (column pointers)
    /// ```
    fn record_csc_dataset_backend(
        &mut self,
        csc_rows: &[u64],
        csc_vals: &[f32],
        csc_colptr: &[u64],
    ) -> anyhow::Result<()> {
        // Populate them into HDF5
        if self.backend.group("/by_column").is_err() {
            let _root = self.backend.create_group("/by_column")?;
            // info!("Group: {:?} created", root);
        }

        {
            let num_threads = num_cpus::get(); // Gets the number of logical CPUs
            blosc_set_nthreads(num_threads as u8); // Set the number of threads for Blosc
        }

        let csc = self.backend.group("/by_column")?;

        let nelem = csc_vals.len();
        let nchunks = NUM_CHUNKS;
        let chunk_size = (nelem / nchunks).max(MIN_CHUNK_SIZE).min(nelem);

        csc.new_dataset::<f32>()
            .shape(csc_vals.len())
            .chunk([chunk_size])
            .blosc_blosclz(COMPRESSION_LEVEL, true)
            .create("data")?
            .write(&csc_vals)?;

        let nelem = csc_colptr.len();
        let nchunks = NUM_CHUNKS;
        let chunk_size = (nelem / nchunks).max(MIN_CHUNK_SIZE).min(nelem);

        csc.new_dataset::<u64>()
            .shape(csc_colptr.len())
            .chunk([chunk_size])
            .blosc_blosclz(COMPRESSION_LEVEL, true)
            .create("indptr")?
            .write(&csc_colptr)?;

        let nelem = csc_rows.len();
        let nchunks = NUM_CHUNKS;
        let chunk_size = (nelem / nchunks).max(MIN_CHUNK_SIZE).min(nelem);

        csc.new_dataset::<u64>()
            .shape(csc_rows.len())
            .chunk([chunk_size])
            .blosc_blosclz(COMPRESSION_LEVEL, true)
            .create("indices")?
            .write(&csc_rows)?;

        self.backend.flush()?;

        Ok(())
    }
}
