use crate::common_io::*;
use crate::mtx_io::*;
use crate::sparse_io::*;
use hdf5::filters::blosc_set_nthreads;
use num_cpus;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

const CHUNK_SIZE: usize = 1000;
const MAX_ROW_NAME_IDX: usize = 3;
const MAX_COLUMN_NAME_IDX: usize = 10;

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
#[derive(Debug)]
pub struct SparseMtxData {
    backend: Arc<hdf5::File>,
    file_name: String,
    chunk_size: usize,
    max_row_name_idx: usize,
    max_column_name_idx: usize,
    by_column_indptr: Vec<u64>,
    by_row_indptr: Vec<u64>,
    remapped_rows: Option<HashMap<usize, usize>>,
}

#[allow(dead_code)]
impl SparseMtxData {
    /// Open existing `SparseMtxData` from a backend HDF5 file
    pub fn open(backend_file: &str) -> anyhow::Result<Self> {
        let hdf5_backend = hdf5::File::open(backend_file)?;

        if let (Some(nrow), Some(ncol), Some(nnz)) = (
            Self::_num_rows(&hdf5_backend),
            Self::_num_columns(&hdf5_backend),
            Self::_num_nnz(&hdf5_backend),
        ) {
            dbg!(format!(
                "#rows: {}, #columns: {}, #non-zeros: {}",
                nrow, ncol, nnz
            ));
        } else {
            anyhow::bail!("Couldn't figure out the size of this sparse matrix data");
        }

        let mut ret = Self {
            backend: hdf5_backend.into(),
            file_name: backend_file.to_string(),
            chunk_size: CHUNK_SIZE,
            max_row_name_idx: MAX_ROW_NAME_IDX,
            max_column_name_idx: MAX_COLUMN_NAME_IDX,
            by_column_indptr: vec![],
            by_row_indptr: vec![],
            remapped_rows: None,
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
            Some(backend_file) => Self::create_backend_file(backend_file)?,
            None => {
                let backend_file = mtx_file.to_string() + ".h5";
                Self::create_backend_file(backend_file.as_ref())?
            }
        };

        // populate data from mtx file
        ret.import_mtx_by_col(mtx_file)?;
        ret.read_column_indptr()?;

        if Some(true) == index_by_row {
            ret.import_mtx_by_row(mtx_file)?;
            ret.read_row_indptr()?;
        }

        Ok(ret)
    }

    /// Read mtx file and populate the data into HDF5 for faster row-by-row access
    /// * `mtx_file`: mtx file to be read into HDF5 backend
    fn import_mtx_by_row(self: &mut Self, mtx_file: &str) -> anyhow::Result<()> {
        let (mut mtx_triplets, mtx_shape) = read_mtx_triplets(mtx_file)?;

        self.record_mtx_shape(mtx_shape)?;

        if mtx_triplets.len() == 0 {
            return Err(anyhow::anyhow!("No data in mtx file"));
        }
        self.record_triplets_by_row(&mut mtx_triplets)
    }

    /// Read mtx file and populate the data into HDF5 for faster column-by-column access
    /// * `mtx_file`: mtx file to be read into HDF5 backend
    fn import_mtx_by_col(self: &mut Self, mtx_file: &str) -> anyhow::Result<()> {
        let (mut mtx_triplets, mtx_shape) = read_mtx_triplets(mtx_file)?;

        if mtx_triplets.len() == 0 {
            return Err(anyhow::anyhow!("No data in mtx file"));
        }
        self.record_mtx_shape(mtx_shape)?;
        self.record_triplets_by_col(&mut mtx_triplets)
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
            Some(backend_file) => Self::create_backend_file(backend_file)?,
            None => {
                let backend_file = create_temp_dir_file(".h5")?;
                let backend_file = backend_file.to_str().expect("to_str failed");
                Self::create_backend_file(&backend_file)?
            }
        };

        ret.import_ndarray_by_col(&array)?; // populate data from mtx file
        ret.read_column_indptr()?; // reload column indptr

        if Some(true) == index_by_row {
            ret.import_ndarray_by_row(&array)?; //
            ret.read_row_indptr()?; //
        }

        Ok(ret)
    }

    /// Read column index pointers
    pub fn read_column_indptr(self: &mut Self) -> anyhow::Result<()> {
        if let Ok(by_column) = self.backend.group("/by_column") {
            let indptr = by_column.dataset("indptr")?.read_1d::<u64>()?;
            self.by_column_indptr.clear();
            self.by_column_indptr.extend(indptr);
        }
        Ok(())
    }

    /// Read row index pointers
    pub fn read_row_indptr(self: &mut Self) -> anyhow::Result<()> {
        if let Ok(by_row) = self.backend.group("/by_row") {
            let indptr = by_row.dataset("indptr")?.read_1d::<u64>()?;
            self.by_row_indptr.clear();
            self.by_row_indptr.extend(indptr);
        }
        Ok(())
    }

    /// Associate sparse matrix data with a HDF5 file
    /// * `hdf5_file`: HDF5 file to be associated with
    fn create_backend_file(hdf5_file: &str) -> anyhow::Result<Self> {
        // dbg!(&hdf5_file);

        let hdf5_backend = hdf5::File::create(hdf5_file)?;

        Ok(Self {
            backend: hdf5_backend.into(),
            file_name: hdf5_file.to_string(),
            chunk_size: CHUNK_SIZE,
            max_row_name_idx: MAX_ROW_NAME_IDX,
            max_column_name_idx: MAX_COLUMN_NAME_IDX,
            by_column_indptr: vec![],
            by_row_indptr: vec![],
            remapped_rows: None,
        })
    }

    /// Remove backend file to free up disk space
    pub fn remove_backend_file(self: &Self) -> anyhow::Result<()> {
        let backend = std::path::Path::new(&self.file_name);
        if backend.exists() {
            std::fs::remove_file(&backend)?;
        }
        Ok(())
    }

    /// Access file name of the hdf5 backend file
    pub fn get_backend_file_name(self: &Self) -> &str {
        &self.file_name
    }

    /// Export the data to a mtx file. This will take time.
    /// * `mtx_file`: mtx file to be written
    pub fn to_mtx_file(&self, mtx_file: &str) -> anyhow::Result<()> {
        let by_column = self.backend.group("/by_column")?;

        let indptr = by_column.dataset("indptr")?.read_1d::<u64>()?;
        let data = by_column.dataset("data")?;
        let indices = by_column.dataset("indices")?;

        let mut buf = open_buf_writer(mtx_file)?;

        if let (Some(ncol), Some(nrow), Some(nnz)) =
            (self.num_columns(), self.num_rows(), self.num_non_zeros())
        {
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
            println!(
                "{}: {} rows, {} columns, {} non-zeros",
                mtx_file, nrow, ncol, nnz
            );

            Ok(())
        } else {
            return Err(anyhow::anyhow!(
                "Unable to figure out the size of the backend data"
            ));
        }
    }

    /////////////////////////////////
    // `ndarray` related functions //
    /////////////////////////////////

    /// Add ndarray to HDF5 backend by row (CSR format)
    /// * `array`: ndarray to be added
    fn import_ndarray_by_row(&mut self, array: &Array2<f32>) -> anyhow::Result<()> {
        let nrow = array.shape()[0];
        let ncol = array.shape()[1];
        let eps = 1e-6;

        let mut mtx_triplets = array
            .indexed_iter()
            .filter(|(_, &elem)| elem.abs() > eps)
            .map(|((row, col), &value)| (row as u64, col as u64, value))
            .collect::<Vec<(u64, u64, f32)>>();

        let nnz = mtx_triplets.len();

        let mtx_shape = (nrow, ncol, nnz);
        self.record_mtx_shape(Some(mtx_shape))?;
        self.record_triplets_by_row(&mut mtx_triplets)
    }

    /// Add ndarray to HDF5 backend by column (CSC format)
    /// * `array`: ndarray to be added
    fn import_ndarray_by_col(&mut self, array: &Array2<f32>) -> anyhow::Result<()> {
        let nrow = array.shape()[0];
        let ncol = array.shape()[1];
        let eps = 1e-6;

        let mut mtx_triplets = array
            .indexed_iter()
            .filter(|(_, &elem)| elem.abs() > eps)
            .map(|((row, col), &value)| (row as u64, col as u64, value))
            .collect::<Vec<(u64, u64, f32)>>();

        let nnz = mtx_triplets.len();

        let mtx_shape = (nrow, ncol, nnz);
        self.record_mtx_shape(Some(mtx_shape))?;
        self.record_triplets_by_col(&mut mtx_triplets)
    }

    //////////////
    // triplets //
    //////////////

    /// Helper function to add triplets to HDF5 backend by row (CSR format)
    fn record_triplets_by_row(
        &mut self,
        row_col_val_triplets: &mut Vec<(u64, u64, f32)>,
    ) -> anyhow::Result<()> {
        debug_assert!(row_col_val_triplets.len() > 0);

        row_col_val_triplets.sort_by_key(|&(_, col, _)| col);
        row_col_val_triplets.sort_by_key(|&(row, _, _)| row);
        // dbg!(&row_col_val_triplets);

        let mut csr_rowptr = vec![];
        let mut csr_cols = vec![];
        let mut csr_vals = vec![];

        let nrow = self.num_rows().expect("should have `nrow`");
        let nnz = row_col_val_triplets.len();

        // fill in rowptr 0 to the first row index
        let first = row_col_val_triplets[0].0;
        for _ in 0..first {
            csr_rowptr.push(0);
        }

        // for the first row/triplet
        csr_rowptr.push(0);
        csr_cols.push(row_col_val_triplets[0].1);
        csr_vals.push(row_col_val_triplets[0].2);

        for i in 1..nnz {
            let lb = row_col_val_triplets[i - 1].0;
            let ub = row_col_val_triplets[i].0;
            for _ in lb..ub {
                csr_rowptr.push(i);
            }
            csr_cols.push(row_col_val_triplets[i].1);
            csr_vals.push(row_col_val_triplets[i].2);
        }

        // fill in the rest of the rowptr
        let last = row_col_val_triplets[nnz - 1].0 as usize;
        for _ in last..nrow {
            csr_rowptr.push(nnz);
        }

        self.record_csr_dataset_backend(&csr_cols, &csr_vals, &csr_rowptr)
    }

    /// Helper function to add triplets to HDF5 backend by column (CSC format)
    fn record_triplets_by_col(
        &mut self,
        row_col_val_triplets: &mut Vec<(u64, u64, f32)>,
    ) -> anyhow::Result<()> {
        debug_assert!(row_col_val_triplets.len() > 0);

        row_col_val_triplets.sort_by_key(|&(row, _, _)| row);
        row_col_val_triplets.sort_by_key(|&(_, col, _)| col);
        // dbg!(&row_col_val_triplets);

        let mut csc_colptr = vec![];
        let mut csc_rows = vec![];
        let mut csc_vals = vec![];

        let ncol = self.num_columns().expect("should have `ncol`");
        let nnz = row_col_val_triplets.len();

        // fill in colptr 0 to the first column index
        let first = row_col_val_triplets[0].1;
        for _ in 0..first {
            csc_colptr.push(0);
        }

        // for the first column/triplet
        csc_colptr.push(0);
        csc_rows.push(row_col_val_triplets[0].0);
        csc_vals.push(row_col_val_triplets[0].2);

        for i in 1..nnz {
            let lb = row_col_val_triplets[i - 1].1;
            let ub = row_col_val_triplets[i].1;
            for _ in lb..ub {
                csc_colptr.push(i);
            }
            csc_rows.push(row_col_val_triplets[i].0 as u64);
            csc_vals.push(row_col_val_triplets[i].2);
        }

        // fill in the rest of the colptr
        let last = row_col_val_triplets[nnz - 1].1 as usize;
        for _ in last..ncol {
            csc_colptr.push(nnz);
        }
        // dbg!(&csc_colptr);

        self.record_csc_dataset_backend(&csc_rows, &csc_vals, &csc_colptr)
    }

    //////////////////////
    // backend related  //
    //////////////////////

    /// Helper function to add CSR dataset to HDF5 backend
    ///
    /// ```text
    ///     └── by_row
    ///         ├── data
    ///         ├── indices (column indices)
    ///         └── isndptr (row pointers)
    /// ```
    fn record_csr_dataset_backend(
        self: &mut Self,
        csr_cols: &Vec<u64>,
        csr_vals: &Vec<f32>,
        csr_rowptr: &Vec<usize>,
    ) -> anyhow::Result<()> {
        // Populate them into HDF5
        if self.backend.group("/by_row").is_err() {
            let root = self.backend.create_group("/by_row")?;
            eprintln!("Group: {:?} created", root);
        }

        {
            let num_threads = num_cpus::get(); // Gets the number of logical CPUs
            blosc_set_nthreads(num_threads as u8); // Set the number of threads for Blosc
        }

        let csr = self.backend.group("/by_row")?;

        csr.new_dataset::<f32>()
            .shape(csr_vals.len())
            .chunk([self.chunk_size.min(csr_vals.len())])
            .blosc_zstd(9, true)
            .create("data")?
            .write(&csr_vals)?;

        csr.new_dataset::<u64>()
            .shape(csr_rowptr.len())
            .chunk([self.chunk_size.min(csr_rowptr.len())])
            .blosc_zstd(9, true)
            .create("indptr")?
            .write(&csr_rowptr)?;

        csr.new_dataset::<u64>()
            .shape(csr_cols.len())
            .chunk([self.chunk_size.min(csr_cols.len())])
            .blosc_zstd(9, true)
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
        self: &mut Self,
        csc_rows: &Vec<u64>,
        csc_vals: &Vec<f32>,
        csc_colptr: &Vec<usize>,
    ) -> anyhow::Result<()> {
        // Populate them into HDF5
        if self.backend.group("/by_column").is_err() {
            let root = self.backend.create_group("/by_column")?;
            eprintln!("Group: {:?} created", root);
        }

        {
            let num_threads = num_cpus::get(); // Gets the number of logical CPUs
            blosc_set_nthreads(num_threads as u8); // Set the number of threads for Blosc
        }

        let csc = self.backend.group("/by_column")?;

        csc.new_dataset::<f32>()
            .shape(csc_vals.len())
            .chunk([self.chunk_size.min(csc_vals.len())])
            .blosc_zstd(9, true)
            .create("data")?
            .write(&csc_vals)?;

        csc.new_dataset::<u64>()
            .shape(csc_colptr.len())
            .chunk([self.chunk_size.min(csc_colptr.len())])
            .blosc_zstd(9, true)
            .create("indptr")?
            .write(&csc_colptr)?;

        csc.new_dataset::<u64>()
            .shape(csc_rows.len())
            .chunk([self.chunk_size.min(csc_rows.len())])
            .blosc_zstd(9, true)
            .create("indices")?
            .write(&csc_rows)?;

        self.backend.flush()?;

        Ok(())
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

    fn set_attrs(self: &mut Self, attr_name: &str, value: usize) -> anyhow::Result<()> {
        if self.backend.attr(attr_name).is_err() {
            self.backend
                .new_attr::<usize>()
                .create(attr_name)?
                .write_scalar(&value)?;
        } else if self.backend.attr(attr_name)?.read_scalar::<usize>()? != value {
            return Err(anyhow::anyhow!(format!("{} mismatch", attr_name)));
        }
        Ok(())
    }

    /// Record the shape of the mtx file into the HDF5 backend
    fn record_mtx_shape(
        self: &mut Self,
        mtx_shape: Option<(usize, usize, usize)>,
    ) -> anyhow::Result<()> {
        if let Some((nrow, ncol, nnz)) = mtx_shape {
            self.set_attrs("nrow", nrow)?;
            self.set_attrs("ncol", ncol)?;
            self.set_attrs("nnz", nnz)?;
            self.backend.flush()?;
        }

        Ok(())
    }
}

impl SparseIo for SparseMtxData {
    /// Set row names for the matrix
    /// * `row_name_file`: a file each line contains row name words
    fn register_row_names_file(self: &mut Self, row_name_file: &str) {
        self.register_names_file("/row_names", row_name_file, 0..self.max_row_name_idx, "_")
            .expect("failed to add row names");
    }

    /// Set row names for the matrix
    /// * `rows`: a vector of row names
    fn register_row_names_vec(&mut self, rows: &Vec<Box<str>>) {
        self.register_names_vec("/row_names", rows)
            .expect("failed to add row names");
    }

    /// Set column names for the matrix
    /// * `column_name_file`: a file each line contains column name words
    fn register_column_names_file(self: &mut Self, column_name_file: &str) {
        self.register_names_file(
            "/column_names",
            column_name_file,
            0..self.max_column_name_idx,
            "@",
        )
        .expect("failed to add column names");
    }

    /// Set column names for the matrix
    /// * `columns`: a vector of column names
    fn register_column_names_vec(&mut self, columns: &Vec<Box<str>>) {
        self.register_names_vec("/column_names", columns)
            .expect("failed to add column names");
    }

    /// Number of rows in the underlying data matrix
    fn num_rows(self: &Self) -> Option<usize> {
        Self::_num_rows(&self.backend)
    }

    /// Number of columns in the underlying data matrix
    fn num_columns(self: &Self) -> Option<usize> {
        Self::_num_columns(&self.backend)
    }

    /// Number of non-zero elements
    fn num_non_zeros(self: &Self) -> Option<usize> {
        Self::_num_nnz(&self.backend)
    }

    /// Add arbitrary names (a vector of strings)
    /// * `group_name`: group name
    /// * `name_file`: a file each line contains name words
    /// * `name_columns`: range of columns to be used for name
    /// * `name_sep`: separator for name columns
    fn register_names_file(
        self: &mut Self,
        key: &str,
        name_file: &str,
        name_columns: Range<usize>,
        name_sep: &str,
    ) -> anyhow::Result<()> {
        use hdf5::types::VarLenUnicode;

        let (_names, _) = read_lines_of_words(name_file, -1)?;

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
            .chunk([self.chunk_size.min(_names.len())])
            .create(key)?
            .write(&_names)?;

        Ok(())
    }

    /// Add arbitrary names (a vector of strings)
    /// * `group_name`: group name
    /// * `names`: a file each line contains name words
    fn register_names_vec(&mut self, key: &str, names: &Vec<Box<str>>) -> anyhow::Result<()> {
        use hdf5::types::VarLenUnicode;

        let _names: Vec<VarLenUnicode> = names
            .iter()
            .map(|x| x.to_string().parse().expect("invalid name"))
            .collect::<Vec<_>>();

        let root = self.backend.group("/")?;
        root.new_dataset::<VarLenUnicode>()
            .shape(_names.len())
            .chunk([self.chunk_size.min(_names.len())])
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

    type IndexIter = Vec<usize>;

    /// Read columns within the range and return dense `ndarray::Array2`
    /// * `columns` : range e.g., 0..3 -> [0, 1, 2] or vec![0, 1, 2]
    ///
    fn read_columns(self: &Self, columns: Self::IndexIter) -> anyhow::Result<Array2<f32>> {
        // need to open backend again?
        // let backend = hdf5::File::open(&self.file_name)?;
        let by_column = self.backend.group("/by_column")?;

        debug_assert!(self.by_column_indptr.len() > 0);
        // let indptr = by_column.dataset("indptr")?.read_1d::<u64>()?;

        let indptr = &self.by_column_indptr;
        let data = by_column.dataset("data")?;
        let indices = by_column.dataset("indices")?;

        let columns_vec = columns.into_iter().collect::<Vec<usize>>();
        let ncol_out = columns_vec.len();

        if let (Some(ncol), Some(nrow)) = (self.num_columns(), self.num_rows()) {
            let mut ret: Array2<f32> = Array2::zeros((nrow, ncol_out));

            for (jj, &j_data) in columns_vec.iter().enumerate() {
                if j_data < ncol {
                    debug_assert!((j_data + 1) < indptr.len());

                    // [start, end)
                    let start = indptr[j_data] as usize;
                    let end = indptr[j_data + 1] as usize;

                    if start < end {
                        let data_slice = data.read_slice_1d::<f32, _>(start..end)?;
                        let indices_slice = indices.read_slice_1d::<u64, _>(start..end)?;
                        for k in 0..(end - start) {
                            let x_ij = data_slice[k];
                            let old_ii = indices_slice[k] as usize;
                            let row_idx = self
                                .remapped_rows
                                .as_ref()
                                .and_then(|remap| remap.get(&old_ii))
                                .or(Some(&old_ii));

                            if let Some(ii) = row_idx {
                                ret[(*ii, jj)] = x_ij;
                            }
                        }
                    }
                }
            }
            Ok(ret)
        } else {
            return Err(anyhow::anyhow!(
                "Unable to figure out the size of the backend data"
            ));
        }
    }

    /// Read rows within the range and return dense `ndarray::Array2`
    /// * `rows` : range e.g., 0..3 -> [0, 1, 2] or vec![0, 1, 2]
    ///
    fn read_rows(self: &Self, rows: Self::IndexIter) -> anyhow::Result<Array2<f32>> {
        // need to open backend again?
        // let backend = hdf5::File::open(&self.file_name)?;
        let by_row = self.backend.group("/by_row")?;

        debug_assert!(self.by_row_indptr.len() > 0);
        // let indptr = by_row.dataset("indptr")?.read_1d::<u64>()?;

        let indptr = &self.by_row_indptr;
        let data = by_row.dataset("data")?;
        let indices = by_row.dataset("indices")?;

        let rows_vec = rows.into_iter().collect::<Vec<usize>>();
        let nrow_out = rows_vec.len();

        if let (Some(ncol), Some(nrow)) = (self.num_columns(), self.num_rows()) {
            let mut ret: Array2<f32> = Array2::zeros((nrow_out, ncol));

            for (ii, &i_data) in rows_vec.iter().enumerate() {
                if i_data < nrow {
                    debug_assert!((i_data + 1) < indptr.len());

                    let start = indptr[i_data] as usize;
                    let end = indptr[i_data + 1] as usize;

                    if start < end {
                        let data_slice = data.read_slice_1d::<f32, _>(start..end)?;
                        let indices_slice = indices.read_slice_1d::<u64, _>(start..end)?;

                        for k in 0..(end - start) {
                            let x_ij = data_slice[k];
                            let jj = indices_slice[k] as usize;
                            ret[(ii, jj)] = x_ij;
                        }
                    }
                }
            }
            Ok(ret)
        } else {
            return Err(anyhow::anyhow!(
                "Unable to figure out the size of the backend data"
            ));
        }
    }

    /// Reposition rows in a new order specified by `remap`
    /// * `remap` - a hashmap of old row index to new row index
    fn remap_rows(&mut self, remap: HashMap<usize, usize>) -> anyhow::Result<()> {
        if self.remapped_rows.is_some() {
            eprintln!("remap_rows: overwriting on the existing remapped_rows\n");
        }

        // update maximum number of rows
        let new_nrow = remap
            .values()
            .max()
            .expect("failed to figure out the new maximum number of rows")
            .clone()
            + 1;
        self.set_attrs("nrow", new_nrow)?;
        self.backend.flush()?;

        // copy down the remap hashmap
        self.remapped_rows = Some(HashMap::new());
        for (old_idx, new_idx) in remap.iter() {
            if *new_idx < new_nrow {
                self.remapped_rows
                    .as_mut()
                    .unwrap()
                    .insert(*old_idx, *new_idx);
            }
        }

        Ok(())
    }
}
