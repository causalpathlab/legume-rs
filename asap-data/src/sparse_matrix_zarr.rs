use crate::common_io::*;
use crate::mtx_io::*;
use ndarray::prelude::*;
use std::ops::Range;
use std::sync::Arc;
use zarrs::array::DataType;
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::ReadableWritableListableStorageTraits as ZStorageTraits;

const CHUNK_SIZE: usize = 1000;
const MAX_ROW_NAME_IDX: usize = 3;
const MAX_COLUMN_NAME_IDX: usize = 10;

/// 10x-like cell-feature matrix with `zarr` backend (feature x cell)
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
#[allow(dead_code)]
pub struct SparseMtxData {
    pub store: Arc<dyn ZStorageTraits>,
    file_name: String,
    chunk_size: usize,
    max_row_name_idx: usize,
    max_column_name_idx: usize,
    by_column_indptr: Vec<u64>,
    by_row_indptr: Vec<u64>,
}

#[allow(dead_code)]
impl SparseMtxData {
    /// Create `SparseMtxData` instance from an existing zarr backend file
    /// * `zarr_file` - zarr backend file
    pub fn open(backend_file: &str) -> anyhow::Result<Self> {
        let store = Arc::new(FilesystemStore::new(backend_file)?);

        let mut ret = Self {
            store: store.clone(),
            file_name: backend_file.to_string(),
            chunk_size: CHUNK_SIZE,
            max_row_name_idx: MAX_ROW_NAME_IDX,
            max_column_name_idx: MAX_COLUMN_NAME_IDX,
            by_column_indptr: vec![],
            by_row_indptr: vec![],
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
        let mut ret = match backend_file {
            Some(backend_file) => Self::create_backend_file(backend_file)?,
            None => {
                let backend_file = mtx_file.to_string() + ".zarr";
                Self::create_backend_file(&backend_file)?
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

    /// Read mtx file and populate the data into zarr for faster row-by-row access
    /// * `mtx_file`: mtx file to be read into zarr backend
    pub fn import_mtx_by_row(self: &mut Self, mtx_file: &str) -> anyhow::Result<()> {
        let (mut mtx_triplets, mtx_shape) = read_mtx_triplets(mtx_file)?;

        self.record_mtx_shape(mtx_shape)?;

        if mtx_triplets.len() == 0 {
            return Err(anyhow::anyhow!("No data in mtx file"));
        }
        self.record_triplets_by_row(&mut mtx_triplets)
    }

    /// Read mtx file and populate the data for faster column-by-column access
    /// * `mtx_file`: mtx file to be read into zarr backend
    pub fn import_mtx_by_col(self: &mut Self, mtx_file: &str) -> anyhow::Result<()> {
        let (mut mtx_triplets, mtx_shape) = read_mtx_triplets(mtx_file)?;

        if mtx_triplets.len() == 0 {
            return Err(anyhow::anyhow!("No data in mtx file"));
        }
        self.record_mtx_shape(mtx_shape)?;
        self.record_triplets_by_col(&mut mtx_triplets)
    }

    /// Create a new `SparseMtxData` instance from an `ndarray` array
    /// * `array` - 2D array to be added to the backend
    /// * `backend_file` - Optional zarr backend file
    /// * `index_by_row` - Optional flag to index by row (CSR format)
    pub fn from_ndarray(
        array: &Array2<f32>,
        zarr_file: Option<&str>,
        index_by_row: Option<bool>,
    ) -> anyhow::Result<Self> {
        let mut ret = match zarr_file {
            Some(backend_file) => Self::create_backend_file(backend_file)?,
            None => {
                let backend_file = create_temp_dir_file(".zarr")?;
                let backend_file = backend_file.to_str().expect("to_str failed");
                Self::create_backend_file(&backend_file)?
            }
        };

        ret.import_ndarray_by_col(&array)?; // for column-wise
        ret.read_column_indptr()?; // pointers

        if Some(true) == index_by_row {
            ret.import_ndarray_by_row(&array)?;
        }
        Ok(ret)
    }

    // pub fn from_hdf5_file(
    //     hdf5_file: &str,
    //     backend_file: Option<&str>,
    //     index_by_row: Option<bool>,
    // ) -> anyhow::Result<Self> {
    //     todo!("need to implement")
    // }

    // pub fn to_hdf5_file(&self, hdf5_file: &str) -> anyhow::Result<()> {
    //     todo!("export to hdf5");
    // }

    /// Export the data to a mtx file. This will take time.
    /// * `mtx_file`: mtx file to be written
    pub fn to_mtx_file(&self, mtx_file: &str) -> anyhow::Result<()> {
        use zarrs::array::Array as ZArray;
        use zarrs::array_subset::ArraySubset;

        let key = "/by_column/indptr";
        let indptr = ZArray::open(self.store.clone(), key)?;
        let indptr = indptr.retrieve_array_subset_ndarray::<u64>(&indptr.subset_all())?;

        let key = "/by_column/data";
        let data = ZArray::open(self.store.clone(), key)?;
        let key = "/by_column/indices";
        let indices = ZArray::open(self.store.clone(), key)?;

        let mut buf = open_buf_writer(mtx_file)?;

        if let (Some(ncol), Some(nrow), Some(nnz)) =
            (self.num_columns(), self.num_rows(), self.num_non_zeros())
        {
            let nrow = nrow as usize;
            let ncol = ncol as usize;
            let nnz = nnz as usize;

            writeln!(buf, "%%MatrixMarket matrix coordinate real general")?;
            writeln!(buf, "{}\t{}\t{}", nrow, ncol, nnz)?;
            debug_assert!(indptr.len() == ncol + 1);

            for jj in 0..ncol {
                let start = indptr[jj];
                let end = indptr[jj + 1];

                let subset = ArraySubset::new_with_ranges(&[start..end]);
                let data_slice = data.retrieve_array_subset_ndarray::<f32>(&subset)?;
                let indices_slice = indices.retrieve_array_subset_ndarray::<u64>(&subset)?;

                // write them with 1-based indices
                for k in 0..(end - start) {
                    let val = data_slice[k as usize];
                    let ii = indices_slice[k as usize] as usize;
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

    /// Show the hierarchy of the zarr store
    pub fn print_hierarchy(self: &Self) -> anyhow::Result<()> {
        let node = zarrs::node::Node::open(&self.store, "/")?;
        let tree = node.hierarchy_tree();
        println!("hierarchy_tree:\n{}", tree);
        Ok(())
    }

    /// Access file name of the zarr backend
    pub fn get_backend_file_name(self: &Self) -> &str {
        &self.file_name
    }

    /// Read row index pointers
    pub fn read_row_indptr(self: &mut Self) -> anyhow::Result<()> {
        use zarrs::array::Array;
        let key = "/by_row/indptr";
        if let Ok(indptr) = Array::open(self.store.clone(), key) {
            let indptr_vec = indptr.retrieve_array_subset_ndarray::<u64>(&indptr.subset_all())?;
            self.by_row_indptr.clear();
            self.by_row_indptr.extend(indptr_vec);
        }
        Ok(())
    }

    /// Read column index pointers
    pub fn read_column_indptr(self: &mut Self) -> anyhow::Result<()> {
        use zarrs::array::Array as ZArray;
        let key = "/by_column/indptr";
        if let Ok(indptr) = ZArray::open(self.store.clone(), key) {
            let indptr_vec = indptr.retrieve_array_subset_ndarray::<u64>(&indptr.subset_all())?;
            self.by_column_indptr.clear();
            self.by_column_indptr.extend(indptr_vec);
        }
        Ok(())
    }

    /// Read rows within the range and return dense `ndarray::Array2`
    /// * `rows` : range e.g., 0..3 -> [0, 1, 2] or vec![0, 1, 2]
    ///
    pub fn read_rows<I>(self: &Self, rows: I) -> anyhow::Result<Array2<f32>>
    where
        I: IntoIterator<Item = usize>,
    {
        use zarrs::array::Array as ZArray;
        use zarrs::array_subset::ArraySubset;

        debug_assert!(self.by_row_indptr.len() > 0);
        let indptr = &self.by_row_indptr;

        let rows_vec = rows.into_iter().collect::<Vec<usize>>();
        let nrow_out = rows_vec.len();

        let key = "/by_row/data";
        let data = ZArray::open(self.store.clone(), key)?;
        let key = "/by_row/indices";
        let indices = ZArray::open(self.store.clone(), key)?;

        if let (Some(nrow), Some(ncol)) = (self.num_rows(), self.num_columns()) {
            let nrow = nrow as usize;
            let ncol = ncol as usize;

            debug_assert!(indptr.len() > nrow);

            let mut ret: Array2<f32> = Array2::zeros((nrow_out, ncol));

            for (ii, &i_data) in rows_vec.iter().enumerate() {
                if i_data < nrow {
                    debug_assert!((i_data + 1) < indptr.len());

                    // [start, end)
                    let start = indptr[i_data];
                    let end = indptr[i_data + 1];

                    if start < end {
                        let subset = ArraySubset::new_with_ranges(&[start..end]);
                        let data_slice = data.retrieve_array_subset_ndarray::<f32>(&subset)?;
                        let indices_slice =
                            indices.retrieve_array_subset_ndarray::<u64>(&subset)?;

                        for k in 0..(end - start) {
                            let x_ij = data_slice[k as usize];
                            let jj = indices_slice[k as usize] as usize;
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

    /// Read columns within the range and return dense `ndarray::Array2`
    /// * `columns` : range e.g., 0..3 -> [0, 1, 2] or vec![0, 1, 2]
    ///
    pub fn read_columns<I>(self: &Self, columns: I) -> anyhow::Result<Array2<f32>>
    where
        I: IntoIterator<Item = usize>,
    {
        use zarrs::array::Array as ZArray;
        use zarrs::array_subset::ArraySubset;

        debug_assert!(self.by_column_indptr.len() > 0);
        let indptr = &self.by_column_indptr;

        let columns_vec = columns.into_iter().collect::<Vec<usize>>();
        let ncol_out = columns_vec.len();

        let key = "/by_column/data";
        let data = ZArray::open(self.store.clone(), key)?;
        let key = "/by_column/indices";
        let indices = ZArray::open(self.store.clone(), key)?;

        if let (Some(ncol), Some(nrow)) = (self.num_columns(), self.num_rows()) {
            let nrow = nrow as usize;
            let ncol = ncol as usize;

            debug_assert!(indptr.len() > ncol);

            let mut ret: Array2<f32> = Array2::zeros((nrow, ncol_out));

            // dbg!(&columns_vec);

            for (jj, &j_data) in columns_vec.iter().enumerate() {
                if j_data < ncol {
                    debug_assert!((j_data + 1) < indptr.len());

                    // [start, end)
                    let start = indptr[j_data];
                    let end = indptr[j_data + 1];

                    if start < end {
                        let subset = ArraySubset::new_with_ranges(&[start..end]);
                        let data_slice = data.retrieve_array_subset_ndarray::<f32>(&subset)?;
                        let indices_slice =
                            indices.retrieve_array_subset_ndarray::<u64>(&subset)?;

                        for k in 0..(end - start) {
                            let x_ij = data_slice[k as usize];
                            let ii = indices_slice[k as usize] as usize;
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

    /// Helper function to create a new zarr backend file
    fn create_backend_file(zarr_file: &str) -> anyhow::Result<Self> {
        // dbg!(zarr_file);
        use zarrs::group::GroupBuilder;
        let store = Arc::new(FilesystemStore::new(zarr_file)?);
        let root = GroupBuilder::new().build(store.clone(), "/")?;
        root.store_metadata()?;

        Ok(Self {
            store: store.clone(),
            file_name: zarr_file.to_string(),
            chunk_size: CHUNK_SIZE,
            max_row_name_idx: MAX_ROW_NAME_IDX,
            max_column_name_idx: MAX_COLUMN_NAME_IDX,
            by_column_indptr: vec![],
            by_row_indptr: vec![],
        })
    }

    /// Clean up the backend file
    pub fn remove_backend_file(&self) -> anyhow::Result<()> {
        let backend = std::path::Path::new(&self.file_name);
        if backend.exists() {
            std::fs::remove_dir_all(backend)?;
        }
        Ok(())
    }

    /////////////////////////////////
    // `ndarray` related functions //
    /////////////////////////////////

    /// Add ndarray to zarr backend by row (CSR format)
    /// * `array` - 2D array to be added to the backend
    pub fn import_ndarray_by_row(&mut self, array: &Array2<f32>) -> anyhow::Result<()> {
        let nrow = array.shape()[0];
        let ncol = array.shape()[1];
        let eps = 1e-6;

        // dbg!("importing ndarray by row...");
        let mut mtx_triplets = array
            .indexed_iter()
            .filter(|(_, &elem)| elem.abs() > eps)
            .map(|((row, col), &value)| (row as u64, col as u64, value))
            .collect::<Vec<(u64, u64, f32)>>();

        let nnz = mtx_triplets.len();
        let mtx_shape = (nrow, ncol, nnz);
        self.record_mtx_shape(Some(mtx_shape))?;

        // dbg!(format!("populated: {} elements", mtx_triplets.len()));

        self.record_triplets_by_row(&mut mtx_triplets)
    }

    /// Add ndarray to zarr backend by column (CSC format)
    /// * `array` - 2D array to be added to the backend
    pub fn import_ndarray_by_col(&mut self, array: &Array2<f32>) -> anyhow::Result<()> {
        let nrow = array.shape()[0];
        let ncol = array.shape()[1];
        let eps = 1e-6;

        // dbg!("importing ndarray by column...");
        let mut mtx_triplets = array
            .indexed_iter()
            .filter(|(_, &elem)| elem.abs() > eps)
            .map(|((row, col), &value)| (row as u64, col as u64, value))
            .collect::<Vec<(u64, u64, f32)>>();

        let nnz = mtx_triplets.len();
        let mtx_shape = (nrow, ncol, nnz);
        self.record_mtx_shape(Some(mtx_shape))?;

        // dbg!(format!("populated: {} elements", mtx_triplets.len()));

        self.record_triplets_by_col(&mut mtx_triplets)
    }

    //////////////
    // triplets //
    //////////////

    /// Helper function to add triplets to zarr backend by row (CSR format)
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
                csr_rowptr.push(i as u64);
            }
	    csr_cols.push(row_col_val_triplets[i].1);
	    csr_vals.push(row_col_val_triplets[i].2);
        }

        // fill in the rest of the rowptr
        let last = row_col_val_triplets[nnz - 1].0;
        let m = nnz as u64;
        for _ in last..nrow {
            csr_rowptr.push(m);
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
                csc_colptr.push(i as u64);
            }
            csc_rows.push(row_col_val_triplets[i].0 as u64);
            csc_vals.push(row_col_val_triplets[i].2);
        }

        // fill in the rest of the colptr
        let last = row_col_val_triplets[nnz - 1].1;
        let m = nnz as u64;
        for _ in last..ncol {
            csc_colptr.push(m);
        }
        // dbg!(&csc_colptr);

        self.record_csc_dataset_backend(&csc_rows, &csc_vals, &csc_colptr)
    }

    //////////////////////
    // backend related  //
    //////////////////////

    /// CSR data structure in Zarr backend
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
        csr_rowptr: &Vec<u64>,
    ) -> anyhow::Result<()> {
        // open or create the group "/by_row"
        let key = "/by_row";
        self._add_group(key)?;

        let key = "/by_row/data";
        self.new_filled_vector(key, DataType::Float32, csr_vals.clone())?;
        let key = "/by_row/indices";
        self.new_filled_vector(key, DataType::UInt64, csr_cols.clone())?;
        let key = "/by_row/indptr";
        self.new_filled_vector(key, DataType::UInt64, csr_rowptr.clone())?;

        Ok(())
    }

    /// CSC data structure in Zarr backend
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
        csc_colptr: &Vec<u64>,
    ) -> anyhow::Result<()> {
        // open or create the group "/by_column"
        let key = "/by_column";
        self._add_group(key)?;

        let key = "/by_column/data";
        self.new_filled_vector(key, DataType::Float32, csc_vals.clone())?;
        let key = "/by_column/indices";
        // dbg!(key);
        self.new_filled_vector(key, DataType::UInt64, csc_rows.clone())?;

        let key = "/by_column/indptr";
        // dbg!(key);
        self.new_filled_vector(key, DataType::UInt64, csc_colptr.clone())?;

        Ok(())
    }

    /// Helper function to create a filled 1D array with the given
    /// data type and fill value. This is the most useful function to
    /// create a vector like data.
    ///
    /// * `key` - the key name
    /// * `dt` - the data type among `DataType`
    /// * `vec` - the vector to be stored
    ///
    fn new_filled_vector<V: zarrs::array::Element>(
        self: &mut Self,
        key: &str,
        dt: DataType,
        vec: Vec<V>,
    ) -> anyhow::Result<()> {
        use zarrs::array::codec::GzipCodec;
        use zarrs::array::ArrayBuilder;
        use zarrs::array::DataType;
        use zarrs::array::FillValue;
        // use zarrs::array::ZARR_NAN_F32;

        let chunk_size = self.chunk_size.min(vec.len());

        let fill = match dt {
            DataType::Float32 => FillValue::from(zarrs::array::ZARR_NAN_F32),
            DataType::UInt64 => FillValue::from(0u64),
            DataType::String => FillValue::from(""),
            _ => FillValue::from(0),
        };

        let array = ArrayBuilder::new(
            vec![vec.len() as u64],              // array shape
            dt.into(),                           // data type
            vec![chunk_size as u64].try_into()?, // chunk shape
            fill,                                //
        )
        .bytes_to_bytes_codecs(vec![Arc::new(GzipCodec::new(5)?)])
        .build(self.store.clone(), key)?;

        array.store_array_subset_ndarray(&[0], Array1::from(vec))?;
        array.store_metadata()?;
        Ok(())
    }

    /// Helper function to keep the matrix shape
    fn record_mtx_shape(
        self: &mut Self,
        mtx_shape: Option<(usize, usize, usize)>,
    ) -> anyhow::Result<()> {
        let set_root_attr = |attr_name: &str, value: usize| -> anyhow::Result<()> {
            let mut root = zarrs::group::Group::open(self.store.clone(), "/")?;

            let new_value = serde_json::to_value(value)?;
            if let Some(old_value) = root.attributes().get(attr_name) {
                if *(old_value) != new_value {
                    return Err(anyhow::anyhow!("{} mismatch", attr_name));
                }
            } else {
                root.attributes_mut()
                    .insert((*attr_name).to_string(), new_value);
                root.store_metadata()?;
            }
            Ok(())
        };

        if let Some((nrow, ncol, nnz)) = mtx_shape {
            set_root_attr("nrow", nrow)?;
            set_root_attr("ncol", ncol)?;
            set_root_attr("nnz", nnz)?;
        }
        Ok(())
    }

    /// Set row names for the matrix
    /// * `row_name_file`: a file each line contains row name words
    pub fn register_row_names(self: &mut Self, row_name_file: &str) {
        self.add_names("row_names", row_name_file, 0..self.max_row_name_idx, "_")
            .expect("failed to add row names");
    }

    /// Set column names for the matrix
    /// * `column_name_file`: a file each line contains column name words
    pub fn register_column_names(self: &mut Self, column_name_file: &str) {
        self.add_names(
            "column_names",
            column_name_file,
            0..self.max_column_name_idx,
            "@",
        )
        .expect("failed to add column names");
    }

    fn add_names(
        self: &mut Self,
        group_name: &str,
        name_file: &str,
        name_columns: Range<usize>,
        name_sep: &str,
    ) -> anyhow::Result<()> {
        let (_names, _) = read_lines_of_words(name_file, -1)?;

        let name_columns = name_columns.clone().collect::<Vec<_>>();

        let _names: Vec<String> = _names
            .iter()
            .map(|x| {
                name_columns
                    .iter()
                    .filter_map(|&i| x.get(i))
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(name_sep)
                    .parse()
                    .expect("invalidrow name")
            })
            .collect();

        self.new_filled_vector(group_name, DataType::String, _names)?;
        Ok(())
    }

    /// Number of rows in the matrix
    pub fn num_rows(self: &Self) -> Option<u64> {
        Self::_num_rows(self.store.clone())
    }

    /// Number of columns in the matrix
    pub fn num_columns(self: &Self) -> Option<u64> {
        Self::_num_columns(self.store.clone())
    }

    /// Number of non-zero elements in the matrix
    pub fn num_non_zeros(self: &Self) -> Option<u64> {
        Self::_num_nnz(self.store.clone())
    }

    /////////////////////////////
    // purely helper functions //
    /////////////////////////////

    fn _num_nnz(store: Arc<dyn ZStorageTraits>) -> Option<u64> {
        Self::_get_group_attr::<u64>(store.clone(), "/", "nnz")
    }

    fn _num_rows(store: Arc<dyn ZStorageTraits>) -> Option<u64> {
        Self::_get_group_attr::<u64>(store.clone(), "/", "nrow")
    }

    fn _num_columns(store: Arc<dyn ZStorageTraits>) -> Option<u64> {
        Self::_get_group_attr::<u64>(store.clone(), "/", "ncol")
    }

    /// Helper function to get an attribute from a group named `group_name`
    fn _get_group_attr<V>(
        store: Arc<dyn ZStorageTraits>,
        group_name: &str,
        attr_name: &str,
    ) -> Option<V>
    where
        V: serde::de::DeserializeOwned,
    {
        zarrs::group::Group::open(store, group_name)
            .ok()
            .and_then(|grp| grp.attributes().get(attr_name).cloned())
            .and_then(|attr| serde_json::from_value(attr.clone()).ok())
    }

    /// Helper function to add a group in `self.store`
    fn _add_group(self: &mut Self, group_name: &str) -> anyhow::Result<()> {
        use zarrs::group::Group;

        if Group::open(self.store.clone(), group_name).is_err() {
            let new_group =
                zarrs::group::GroupBuilder::new().build(self.store.clone(), group_name)?;
            new_group.store_metadata()?;
        } else {
            dbg!("group already exists");
        }

        Ok(())
    }
}
