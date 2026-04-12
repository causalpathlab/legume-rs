#![allow(dead_code)]

use crate::sparse_io::*;
use log::info;
use matrix_util::common_io::*;
use std::ops::Range;
use std::sync::Arc;
use zarrs::array::{data_type, ArraySubset, DataType};
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::ReadableListableStorageTraits as ZReadStorageTraits;

use anyhow::anyhow;

use crate::utilities::io_helpers::chunk_elems;

const COMPRESSION_LEVEL: i32 = 5;

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
#[derive(Clone)]
pub struct SparseMtxData {
    read_store: Arc<dyn ZReadStorageTraits>,
    write_store: Option<Arc<FilesystemStore>>,
    file_name: String,
    max_row_name_idx: usize,
    max_column_name_idx: usize,
    by_column_indptr: Vec<u64>,
    by_row_indptr: Vec<u64>,
    by_column_indices: Option<Vec<u64>>,
    by_column_data: Option<Vec<f32>>,
}

impl SparseMtxData {
    /// Get the writable store, or error if this is a read-only backend (e.g. zip archive).
    fn write_store(&self) -> anyhow::Result<&Arc<FilesystemStore>> {
        self.write_store
            .as_ref()
            .ok_or_else(|| anyhow!("store is read-only (zip archive)"))
    }
}

impl SparseMtxData {
    /// Create an empty new `SparseMtxData` instance with a zarr
    /// backend file If no `backend_file` is provided, a temporary
    /// file will be created.
    ///
    /// * `backend_file` - Optional zarr backend file
    pub fn new(zarr_file: Option<&str>) -> anyhow::Result<Self> {
        Self::create_backend(zarr_file)
    }

    /// Helper to create a backend file (from provided path or temp file)
    fn create_backend(zarr_file: Option<&str>) -> anyhow::Result<Self> {
        match zarr_file {
            Some(backend_file) => Self::register_backend_file(backend_file),
            None => {
                let backend_file = create_temp_dir_file(".zarr")?;
                let backend_file = backend_file
                    .to_str()
                    .ok_or_else(|| anyhow::anyhow!("Failed to convert path to string"))?;
                Self::register_backend_file(backend_file)
            }
        }
    }

    /// Create `SparseMtxData` instance from an existing zarr backend file
    /// * `zarr_file` - zarr backend file (directory or `.zarr.zip`)
    pub fn open(backend_file: &str) -> anyhow::Result<Self> {
        let (read_store, write_store) = crate::zarr_io::open_zarr_store_rw(backend_file)?;

        if (
            Self::_num_rows(read_store.clone()),
            Self::_num_columns(read_store.clone()),
            Self::_num_nnz(read_store.clone()),
        ) == (None, None, None)
        {
            anyhow::bail!("Couldn't figure out the size of this sparse matrix data");
        }

        let mut ret = Self {
            read_store,
            write_store,
            file_name: backend_file.to_string(),
            max_row_name_idx: MAX_ROW_NAME_IDX,
            max_column_name_idx: MAX_COLUMN_NAME_IDX,
            by_column_indptr: vec![],
            by_row_indptr: vec![],
            by_column_indices: None,
            by_column_data: None,
        };

        ret.read_column_indptr()?;
        ret.read_row_indptr()?;

        Ok(ret)
    }

    /// Create `SparseMtxData` from mtx file with `backend_file` as
    /// the backend file.  If no `backend_file` is provided, it will
    /// be the same as `mtx_file` with `.zarr` extension.
    /// * `mtx_file`: mtx file to be read into zarr backend
    /// * `backend_file`: zarr file to be associated with
    /// * `index_by_row`: if true, the matrix will be indexed by row
    pub fn from_mtx_file(
        mtx_file: &str,
        backend_file: Option<&str>,
        index_by_row: Option<bool>,
    ) -> anyhow::Result<Self> {
        let zarr_file = backend_file
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("{}.zarr", mtx_file));

        info!("backend file: {}", zarr_file);
        let mut ret = Self::register_backend_file(&zarr_file)?;

        info!("importing mtx file by column");
        ret.import_mtx_file_by_col(mtx_file)?;
        ret.read_column_indptr()?;

        if index_by_row == Some(true) {
            info!("importing mtx file by row");
            ret.import_mtx_file_by_row(mtx_file)?;
            ret.read_row_indptr()?;
        }

        info!("created sparse backend from {}", mtx_file);
        Ok(ret)
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
        let mut ret = Self::create_backend(zarr_file)?;

        ret.import_ndarray_by_col(array)?;
        ret.read_column_indptr()?;

        if index_by_row == Some(true) {
            ret.import_ndarray_by_row(array)?;
            ret.read_row_indptr()?;
        }
        Ok(ret)
    }

    /// Create a new `SparseMtxData` instance from an `DMatrix` array
    /// * `array` - 2D array to be added to the backend
    /// * `backend_file` - Optional zarr backend file
    /// * `index_by_row` - Optional flag to index by row (CSR format)
    pub fn from_dmatrix(
        matrix: &DMatrix<f32>,
        zarr_file: Option<&str>,
        index_by_row: Option<bool>,
    ) -> anyhow::Result<Self> {
        let mut ret = Self::create_backend(zarr_file)?;

        ret.import_dmatrix_by_col(matrix)?;
        ret.read_column_indptr()?;

        if index_by_row == Some(true) {
            ret.import_dmatrix_by_row(matrix)?;
            ret.read_row_indptr()?;
        }
        Ok(ret)
    }

    /// Show the hierarchy of the zarr store
    pub fn print_hierarchy(&self) -> anyhow::Result<()> {
        use zarrs::config::MetadataRetrieveVersion;
        let node =
            zarrs::node::Node::open_opt(&self.read_store, "/", &MetadataRetrieveVersion::Default)?;
        let tree = node.hierarchy_tree();
        info!("hierarchy_tree:\n{}", tree);
        Ok(())
    }

    /// Helper function to create a new zarr backend file
    fn register_backend_file(zarr_file: &str) -> anyhow::Result<Self> {
        use zarrs::group::GroupBuilder;
        let store = Arc::new(FilesystemStore::new(zarr_file)?);
        let root = GroupBuilder::new().build(store.clone(), "/")?;
        root.store_metadata()?;

        Ok(Self {
            read_store: store.clone(),
            write_store: Some(store),
            file_name: zarr_file.to_string(),
            max_row_name_idx: MAX_ROW_NAME_IDX,
            max_column_name_idx: MAX_COLUMN_NAME_IDX,
            by_column_indptr: vec![],
            by_row_indptr: vec![],
            by_column_indices: None,
            by_column_data: None,
        })
    }

    //////////////////////
    // backend related  //
    //////////////////////

    /// Helper function to create a filled 1D array with the given
    /// data type and fill value. This is the most useful function to
    /// create a vector like data.
    ///
    /// * `key` - the key name
    /// * `dt` - the data type among `DataType`
    /// * `vec` - the vector to be stored
    ///
    fn new_filled_vector<V>(&mut self, key: &str, dt: DataType, vec: &[V]) -> anyhow::Result<()>
    where
        V: zarrs::array::Element,
    {
        use zarrs::array::codec::ZstdCodec;
        use zarrs::array::ArrayBuilder;
        use zarrs::array::FillValue;

        let ws = self.write_store()?;

        let nelem = vec.len();
        let chunk_size = chunk_elems(nelem, std::mem::size_of::<V>());

        let fill = if dt == data_type::float32() {
            FillValue::from(zarrs::array::ZARR_NAN_F32)
        } else if dt == data_type::uint64() {
            FillValue::from(0u64)
        } else if dt == data_type::string() {
            FillValue::from("")
        } else {
            FillValue::from(0)
        };

        let array = ArrayBuilder::new(
            vec![vec.len() as u64],  // array shape
            vec![chunk_size as u64], // chunk shape
            dt,                      // data type
            fill,                    //
        )
        .bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(COMPRESSION_LEVEL, false))])
        .build(ws.clone(), key)?;

        array.store_metadata()?;

        let subset = Self::create_subset(0..vec.len() as u64);
        array.store_array_subset(&subset, vec)?;

        Ok(())
    }

    fn _open_vector(
        &self,
        key: &str,
    ) -> anyhow::Result<zarrs::array::Array<dyn ZReadStorageTraits>> {
        use zarrs::array::Array as ZArray;
        let ret = ZArray::open(self.read_store.clone(), key)?;
        Ok(ret)
    }

    /// Create an empty fixed-shape 1-D array with the given data type,
    /// chunk layout, and fill value. No data is written.
    ///
    /// Used by the streaming write path to pre-create `/by_column/*` and
    /// `/by_row/*` arrays at their final size before any triplets land.
    fn create_shaped_vector(
        &mut self,
        key: &str,
        dt: DataType,
        elem_bytes: usize,
        nelem: usize,
    ) -> anyhow::Result<()> {
        use zarrs::array::codec::ZstdCodec;
        use zarrs::array::ArrayBuilder;
        use zarrs::array::FillValue;

        let ws = self.write_store()?;

        let chunk_size = chunk_elems(nelem, elem_bytes);

        let fill = if dt == data_type::float32() {
            FillValue::from(zarrs::array::ZARR_NAN_F32)
        } else if dt == data_type::uint64() {
            FillValue::from(0u64)
        } else {
            FillValue::from(0)
        };

        let array = ArrayBuilder::new(
            vec![nelem.max(1) as u64],
            vec![chunk_size.max(1) as u64],
            dt,
            fill,
        )
        .bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(COMPRESSION_LEVEL, false))])
        .build(ws.clone(), key)?;

        array.store_metadata()?;
        Ok(())
    }

    /// Open an existing array through the writable filesystem store.
    /// `_open_vector` uses the read-only handle, so it can't be used
    /// to stage streaming writes.
    fn _open_writable_vector(
        &self,
        key: &str,
    ) -> anyhow::Result<zarrs::array::Array<FilesystemStore>> {
        use zarrs::array::Array as ZArray;
        let ws = self.write_store()?.clone();
        let ret = ZArray::open(ws, key)?;
        Ok(ret)
    }

    /// Write a `u64` slab at the given offset. The target array must
    /// already exist (see [`create_shaped_vector`]).
    fn write_slab_u64(&mut self, key: &str, offset: u64, data: &[u64]) -> anyhow::Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        let array = self._open_writable_vector(key)?;
        let subset = Self::create_subset(offset..offset + data.len() as u64);
        array.store_array_subset(&subset, data)?;
        Ok(())
    }

    /// Write an `f32` slab at the given offset.
    fn write_slab_f32(&mut self, key: &str, offset: u64, data: &[f32]) -> anyhow::Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        let array = self._open_writable_vector(key)?;
        let subset = Self::create_subset(offset..offset + data.len() as u64);
        array.store_array_subset(&subset, data)?;
        Ok(())
    }

    #[allow(clippy::type_complexity)]
    fn open_csc_triplets(
        &self,
    ) -> anyhow::Result<(
        zarrs::array::Array<dyn ZReadStorageTraits>,
        zarrs::array::Array<dyn ZReadStorageTraits>,
        zarrs::array::Array<dyn ZReadStorageTraits>,
    )> {
        Ok((
            self._open_vector("/by_column/indptr")?,
            self._open_vector("/by_column/data")?,
            self._open_vector("/by_column/indices")?,
        ))
    }

    /// Helper to create an ArraySubset from a range
    #[inline]
    fn create_subset(range: Range<u64>) -> ArraySubset {
        ArraySubset::new_with_ranges(&[range])
    }

    fn _retrieve_vector<V>(&self, key: &str) -> anyhow::Result<Vec<V>>
    where
        V: zarrs::array::ElementOwned,
    {
        let data = self._open_vector(key)?;
        let ntot = data.shape()[0];
        let subset = Self::create_subset(0..ntot);
        Ok(data.retrieve_array_subset::<Vec<V>>(&subset)?)
    }

    /////////////////////////////
    // purely helper functions //
    /////////////////////////////

    /// Helper function to set an attribute from a group named `group_name`
    fn _set_group_attr<V>(
        store: Arc<FilesystemStore>,
        group_name: &str,
        attr_name: &str,
        value: &V,
    ) -> anyhow::Result<()>
    where
        V: serde::Serialize,
    {
        use zarrs::group::Group;
        let mut group = Group::open(store, group_name)?;

        let new_value = serde_json::to_value(value)?;
        group
            .attributes_mut()
            .insert((*attr_name).to_string(), new_value);
        group.store_metadata()?;
        Ok(())
    }

    /// Helper function to get an attribute from a group named `group_name`
    fn _get_group_attr<V>(
        store: Arc<dyn ZReadStorageTraits>,
        group_name: &str,
        attr_name: &str,
    ) -> Option<V>
    where
        V: serde::de::DeserializeOwned,
    {
        zarrs::group::Group::open(store, group_name)
            .ok()
            .and_then(|grp| grp.attributes().get(attr_name).cloned())
            .and_then(|attr| serde_json::from_value(attr).ok())
    }

    fn _num_nnz(store: Arc<dyn ZReadStorageTraits>) -> Option<usize> {
        Self::_get_group_attr::<usize>(store, "/", "nnz")
    }

    fn _num_rows(store: Arc<dyn ZReadStorageTraits>) -> Option<usize> {
        Self::_get_group_attr::<usize>(store, "/", "nrow")
    }

    fn _num_columns(store: Arc<dyn ZReadStorageTraits>) -> Option<usize> {
        Self::_get_group_attr::<usize>(store, "/", "ncol")
    }

    /// Helper function to add a group in the writable store
    fn _add_group(&mut self, group_name: &str) -> anyhow::Result<()> {
        use zarrs::group::Group;
        let ws = self.write_store()?;

        if Group::open(ws.clone(), group_name).is_err() {
            let new_group = zarrs::group::GroupBuilder::new().build(ws.clone(), group_name)?;
            new_group.store_metadata()?;
        }

        Ok(())
    }
}

impl SparseIo for SparseMtxData {
    type IndexIter = Vec<usize>;

    /// Read row index pointers
    fn read_row_indptr(&mut self) -> anyhow::Result<()> {
        use zarrs::array::Array as Zarray;
        let key = "/by_row/indptr";
        if let Ok(indptr) = Zarray::open(self.read_store.clone(), key) {
            let indptr_vec = indptr.retrieve_array_subset::<Vec<u64>>(&indptr.subset_all())?;
            self.by_row_indptr.clear();
            self.by_row_indptr.extend(indptr_vec);
        }
        Ok(())
    }

    /// Read column index pointers
    fn read_column_indptr(&mut self) -> anyhow::Result<()> {
        use zarrs::array::Array as ZArray;
        let key = "/by_column/indptr";
        if let Ok(indptr) = ZArray::open(self.read_store.clone(), key) {
            let indptr_vec = indptr.retrieve_array_subset::<Vec<u64>>(&indptr.subset_all())?;
            self.by_column_indptr.clear();
            self.by_column_indptr.extend(indptr_vec);
        }
        Ok(())
    }

    fn clean_preloaded_columns(&mut self) {
        self.by_column_data = None;
        self.by_column_indices = None;
    }

    /// preload columns' values and indices
    fn preload_columns(&mut self) -> anyhow::Result<()> {
        use zarrs::array::Array as ZArray;

        let key = "/by_column/data";
        let data = ZArray::open(self.read_store.clone(), key)?;
        let key = "/by_column/indices";
        let indices = ZArray::open(self.read_store.clone(), key)?;

        let data = data.retrieve_array_subset::<Vec<f32>>(&data.subset_all())?;
        let indices = indices.retrieve_array_subset::<Vec<u64>>(&indices.subset_all())?;

        self.by_column_indices = Some(indices);
        self.by_column_data = Some(data);
        Ok(())
    }

    /// Helper function to keep the matrix shape
    fn record_mtx_shape(&mut self, mtx_shape: Option<(usize, usize, usize)>) -> anyhow::Result<()> {
        if let Some((nrow, ncol, nnz)) = mtx_shape {
            let ws = self.write_store()?;
            let read_store = self.read_store.clone();

            let check_set_attr = |attr_name: &str, value: usize| -> anyhow::Result<()> {
                let old_value = Self::_get_group_attr::<usize>(read_store.clone(), "/", attr_name);
                let new_value = serde_json::to_value(value)?;

                match old_value {
                    Some(old_value) => {
                        if old_value != new_value {
                            return Err(anyhow!("{} mismatch", attr_name));
                        }
                    }
                    _ => {
                        Self::_set_group_attr(ws.clone(), "/", attr_name, &new_value)?;
                    }
                }
                Ok(())
            };

            check_set_attr("nrow", nrow)?;
            check_set_attr("ncol", ncol)?;
            check_set_attr("nnz", nnz)?;
        }
        Ok(())
    }

    /// Helper function to create a new zarr backend file
    fn initialize_backend(&mut self) -> anyhow::Result<()> {
        use zarrs::group::GroupBuilder;

        self.remove_backend_file()?;
        let zarr_file = &self.file_name;
        let store = Arc::new(FilesystemStore::new(zarr_file)?);
        let root = GroupBuilder::new().build(store.clone(), "/")?;
        root.store_metadata()?;

        self.read_store = store.clone();
        self.write_store = Some(store);
        self.file_name = zarr_file.to_string();
        self.max_column_name_idx = MAX_COLUMN_NAME_IDX;
        self.max_row_name_idx = MAX_ROW_NAME_IDX;
        self.by_column_indptr = vec![];
        self.by_row_indptr = vec![];

        Ok(())
    }

    /// Clean up the backend file
    fn remove_backend_file(&self) -> anyhow::Result<()> {
        let backend = std::path::Path::new(&self.file_name);
        if backend.exists() {
            if backend.is_file() {
                std::fs::remove_file(backend)?;
            } else {
                std::fs::remove_dir_all(backend)?;
            }
        }
        Ok(())
    }

    /// Access file name of the zarr backend
    fn get_backend_file_name(&self) -> &str {
        &self.file_name
    }

    fn backend_type(&self) -> SparseIoBackend {
        SparseIoBackend::Zarr
    }

    /// Export the data to a mtx file. This will take time.
    /// * `mtx_file`: mtx file to be written
    fn to_mtx_file(&self, mtx_file: &str) -> anyhow::Result<()> {
        if let (Some(ncol), Some(nrow), Some(nnz)) =
            (self.num_columns(), self.num_rows(), self.num_non_zeros())
        {
            let (nrow, ncol, nnz) = (nrow, ncol, nnz);

            let mut buf = open_buf_writer(mtx_file)?;
            writeln!(buf, "%%MatrixMarket matrix coordinate real general")?;
            writeln!(buf, "{}\t{}\t{}", nrow, ncol, nnz)?;

            let (indptr, data, indices) = self.open_csc_triplets()?;
            let indptr = indptr.retrieve_array_subset::<Vec<u64>>(&indptr.subset_all())?;
            debug_assert!(indptr.len() == ncol + 1);

            for jj in 0..ncol {
                let (start, end) = (indptr[jj], indptr[jj + 1]);
                let subset = Self::create_subset(start..end);
                let data_slice = data.retrieve_array_subset::<Vec<f32>>(&subset)?;
                let indices_slice = indices.retrieve_array_subset::<Vec<u64>>(&subset)?;

                // write them with 1-based indices
                for k in 0..(end - start) as usize {
                    let val = data_slice[k];
                    let ii = indices_slice[k] as usize;
                    writeln!(buf, "{}\t{}\t{}", ii + 1, jj + 1, val)?;
                }
            }
            buf.flush()?;
            Ok(())
        } else {
            Err(anyhow!("Unable to figure out the size of the backend data"))
        }
    }

    /// Set row names for the matrix
    /// * `row_name_file`: a file each line contains row name words
    fn register_row_names_file(&mut self, row_name_file: &str) {
        let _ = self.register_names_file(
            "/row_names",
            row_name_file,
            0..self.max_row_name_idx,
            ROW_SEP,
        );
    }

    /// Set row names for the matrix
    /// * `rows`: a vector of row names
    fn register_row_names_vec(&mut self, rows: &[Box<str>]) {
        let _ = self.register_names_vec("/row_names", rows);
    }

    /// Set column names for the matrix
    /// * `column_name_file`: a file each line contains column name words
    fn register_column_names_file(&mut self, column_name_file: &str) {
        let _ = self.register_names_file(
            "/column_names",
            column_name_file,
            0..self.max_column_name_idx,
            COLUMN_SEP,
        );
    }

    /// Set column names for the matrix
    /// * `columns`: a vector of column names
    fn register_column_names_vec(&mut self, columns: &[Box<str>]) {
        let _ = self.register_names_vec("/column_names", columns);
    }

    /// Number of rows in the matrix
    fn num_rows(&self) -> Option<usize> {
        Self::_num_rows(self.read_store.clone())
    }

    /// Number of columns in the matrix
    fn num_columns(&self) -> Option<usize> {
        Self::_num_columns(self.read_store.clone())
    }

    /// Number of non-zero elements in the matrix
    fn num_non_zeros(&self) -> Option<usize> {
        Self::_num_nnz(self.read_store.clone())
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
        let name_data = read_lines_of_words(name_file, -1)?;
        let name_columns_vec: Vec<usize> = name_columns.collect();

        let names: Vec<String> = name_data
            .lines
            .iter()
            .map(|line| {
                name_columns_vec
                    .iter()
                    .filter_map(|&i| line.get(i))
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
                    .join(name_sep)
            })
            .collect();

        self.new_filled_vector(key, data_type::string(), &names)?;
        Ok(())
    }

    /// Add arbitrary names (a vector of strings)
    /// * `group_name`: group name
    /// * `names`: a file each line contains name words
    fn register_names_vec(&mut self, key: &str, names: &[Box<str>]) -> anyhow::Result<()> {
        let names_vec: Vec<String> = names.iter().map(|x| x.to_string()).collect();
        self.new_filled_vector(key, data_type::string(), &names_vec)?;
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
        Ok(self
            ._retrieve_vector::<String>(key)?
            .into_iter()
            .map(|s| s.into_boxed_str())
            .collect())
    }

    /// Read columns within the range and return a vector of triplets (row, col, value)
    /// * `col` : usize
    ///
    fn read_triplets_by_single_column(
        &self,
        j_data: usize,
    ) -> anyhow::Result<(usize, usize, Vec<(u64, u64, f32)>)> {
        use zarrs::array::Array as ZArray;

        debug_assert!(!self.by_column_indptr.is_empty()); // pre-loaded
        debug_assert!(j_data < self.num_columns().unwrap_or(0)); //

        let indptr = &self.by_column_indptr;

        debug_assert!((j_data + 1) < indptr.len());
        debug_assert!(indptr.len() > self.num_columns().unwrap_or(0));

        let nrow = self
            .num_rows()
            .ok_or(anyhow!("can't figure out the number of rows"))?;

        if let (Some(data), Some(indices)) = (&self.by_column_data, &self.by_column_indices) {
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
            let key = "/by_column/data";
            let data = ZArray::open(self.read_store.clone(), key)?;
            let key = "/by_column/indices";
            let indices = ZArray::open(self.read_store.clone(), key)?;

            let ncol_out = 1;
            let jj = 0;

            // [start, end)
            let start = indptr[j_data];
            let end = indptr[j_data + 1];

            let mut ret: Vec<(u64, u64, f32)> = Vec::with_capacity((end - start) as usize);

            if start < end {
                let subset = Self::create_subset(start..end);
                let data_slice = data.retrieve_array_subset::<Vec<f32>>(&subset)?;
                let indices_slice = indices.retrieve_array_subset::<Vec<u64>>(&subset)?;

                for k in 0..(end - start) {
                    let x_ij = data_slice[k as usize];
                    let ii = indices_slice[k as usize];
                    debug_assert!((ii as usize) < nrow);
                    ret.push((ii, jj, x_ij));
                }
            }

            Ok((nrow, ncol_out, ret))
        }
    }

    /// Read columns within the range and return dense `ndarray::Array2`
    /// * `columns` : range e.g., 0..3 -> [0, 1, 2] or vec![0, 1, 2]
    ///
    fn read_triplets_by_columns(
        &self,
        columns: Self::IndexIter,
    ) -> anyhow::Result<(usize, usize, Vec<(u64, u64, f32)>)> {
        use zarrs::array::Array as ZArray;

        debug_assert!(!self.by_column_indptr.is_empty());
        let indptr = &self.by_column_indptr;
        let columns_vec = columns.into_iter().collect::<Vec<usize>>();

        debug_assert!(indptr.len() > self.num_columns().unwrap_or(0));

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

        if let (Some(data), Some(indices)) = (&self.by_column_data, &self.by_column_indices) {
            let ncol_out = columns_vec.len();

            let mut ret: Vec<(u64, u64, f32)> = Vec::with_capacity((max_end - min_start) as usize);

            for (jj, &j_data) in columns_vec.iter().enumerate() {
                let jj = jj as u64;
                let start = indptr[j_data] as usize;
                let end = indptr[j_data + 1] as usize;
                for (&ii, &x_ij) in indices[start..end].iter().zip(data[start..end].iter()) {
                    ret.push((ii, jj, x_ij));
                }
            }

            Ok((nrow, ncol_out, ret))
        } else {
            let key = "/by_column/data";
            let data = ZArray::open(self.read_store.clone(), key)?;
            let key = "/by_column/indices";
            let indices = ZArray::open(self.read_store.clone(), key)?;

            let ncol_out = columns_vec.len();

            let mut ret: Vec<(u64, u64, f32)> = Vec::with_capacity((max_end - min_start) as usize);

            for (jj, &j_data) in columns_vec.iter().enumerate() {
                let jj = jj as u64;
                if j_data < ncol {
                    // [start, end)
                    let start = indptr[j_data];
                    let end = indptr[j_data + 1];

                    if start < end {
                        let subset = Self::create_subset(start..end);
                        let data_slice = data.retrieve_array_subset::<Vec<f32>>(&subset)?;
                        let indices_slice = indices.retrieve_array_subset::<Vec<u64>>(&subset)?;

                        for k in 0..(end - start) {
                            let x_ij = data_slice[k as usize];
                            let ii = indices_slice[k as usize];
                            debug_assert!((ii as usize) < nrow);
                            ret.push((ii, jj, x_ij));
                        }
                    }
                }
            }
            Ok((nrow, ncol_out, ret))
        }
    }

    /// Read rows within the range and return a vector of triplets (row, col, value)
    /// * `rows` : range e.g., 0..3 -> [0, 1, 2] or vec![0, 1, 2]
    ///
    fn read_triplets_by_rows(
        &self,
        rows: Self::IndexIter,
    ) -> anyhow::Result<(usize, usize, Vec<(u64, u64, f32)>)> {
        use zarrs::array::Array as ZArray;

        debug_assert!(!self.by_row_indptr.is_empty());
        let indptr = &self.by_row_indptr;
        debug_assert!(indptr.len() > self.num_rows().unwrap_or(0));

        let rows_vec = rows.into_iter().collect::<Vec<_>>();

        let key = "/by_row/data";
        let data = ZArray::open(self.read_store.clone(), key)?;
        let key = "/by_row/indices";
        let indices = ZArray::open(self.read_store.clone(), key)?;

        if let (Some(nrow), Some(ncol)) = (self.num_rows(), self.num_columns()) {
            let nrow_out = rows_vec.len();

            let min_start = rows_vec
                .iter()
                .map(|&j_data| indptr[j_data])
                .min()
                .unwrap_or(0);

            let max_end = rows_vec
                .iter()
                .map(|&j_data| indptr[j_data + 1])
                .max()
                .unwrap_or(0);

            let mut ret: Vec<(u64, u64, f32)> = Vec::with_capacity((max_end - min_start) as usize);

            for (ii, &i_data) in rows_vec.iter().enumerate() {
                let ii = ii as u64;
                if i_data < nrow {
                    debug_assert!((i_data + 1) < indptr.len());

                    // [start, end)
                    let start = indptr[i_data];
                    let end = indptr[i_data + 1];

                    if start < end {
                        let subset = Self::create_subset(start..end);
                        let data_slice = data.retrieve_array_subset::<Vec<f32>>(&subset)?;
                        let indices_slice = indices.retrieve_array_subset::<Vec<u64>>(&subset)?;

                        for k in 0..(end - start) {
                            let x_ij = data_slice[k as usize];
                            let jj = indices_slice[k as usize];
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
    /// CSR data structure in Zarr backend
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
        // open or create the group "/by_row"
        let key = "/by_row";
        self._add_group(key)?;

        let key = "/by_row/data";
        self.new_filled_vector(key, data_type::float32(), csr_vals)?;
        let key = "/by_row/indices";
        self.new_filled_vector(key, data_type::uint64(), csr_cols)?;
        let key = "/by_row/indptr";
        self.new_filled_vector(key, data_type::uint64(), csr_rowptr)?;

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
        &mut self,
        csc_rows: &[u64],
        csc_vals: &[f32],
        csc_colptr: &[u64],
    ) -> anyhow::Result<()> {
        // open or create the group "/by_column"
        let key = "/by_column";
        self._add_group(key)?;

        let key = "/by_column/data";
        self.new_filled_vector(key, data_type::float32(), csc_vals)?;
        let key = "/by_column/indices";
        self.new_filled_vector(key, data_type::uint64(), csc_rows)?;
        let key = "/by_column/indptr";
        self.new_filled_vector(key, data_type::uint64(), csc_colptr)?;

        Ok(())
    }

    fn cs_create(&mut self, key: CsKey, len: usize) -> anyhow::Result<()> {
        let (group, path, dt, elem_bytes) = match key {
            CsKey::CscData => (
                "/by_column",
                "/by_column/data",
                data_type::float32(),
                std::mem::size_of::<f32>(),
            ),
            CsKey::CscIndices => (
                "/by_column",
                "/by_column/indices",
                data_type::uint64(),
                std::mem::size_of::<u64>(),
            ),
            CsKey::CscIndptr => (
                "/by_column",
                "/by_column/indptr",
                data_type::uint64(),
                std::mem::size_of::<u64>(),
            ),
            CsKey::CsrData => (
                "/by_row",
                "/by_row/data",
                data_type::float32(),
                std::mem::size_of::<f32>(),
            ),
            CsKey::CsrIndices => (
                "/by_row",
                "/by_row/indices",
                data_type::uint64(),
                std::mem::size_of::<u64>(),
            ),
            CsKey::CsrIndptr => (
                "/by_row",
                "/by_row/indptr",
                data_type::uint64(),
                std::mem::size_of::<u64>(),
            ),
        };
        self._add_group(group)?;
        self.create_shaped_vector(path, dt, elem_bytes, len)
    }

    fn cs_write_u64(&mut self, key: CsKey, offset: u64, data: &[u64]) -> anyhow::Result<()> {
        let path = match key {
            CsKey::CscIndices => "/by_column/indices",
            CsKey::CscIndptr => "/by_column/indptr",
            CsKey::CsrIndices => "/by_row/indices",
            CsKey::CsrIndptr => "/by_row/indptr",
            CsKey::CscData | CsKey::CsrData => {
                return Err(anyhow!("cs_write_u64 called on f32 slot {:?}", key));
            }
        };
        self.write_slab_u64(path, offset, data)
    }

    fn cs_write_f32(&mut self, key: CsKey, offset: u64, data: &[f32]) -> anyhow::Result<()> {
        let path = match key {
            CsKey::CscData => "/by_column/data",
            CsKey::CsrData => "/by_row/data",
            _ => {
                return Err(anyhow!("cs_write_f32 called on u64 slot {:?}", key));
            }
        };
        self.write_slab_f32(path, offset, data)
    }
}
