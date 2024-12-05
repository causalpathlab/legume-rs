pub use ndarray::prelude::*;
pub use std::collections::HashMap;
pub use std::ops::Range;

#[allow(dead_code)]
pub trait SparseIo {
    /// Read columns within the range and return dense `ndarray::Array2`
    /// * `columns` : range e.g., 0..3 -> [0, 1, 2] or vec![0, 1, 2]
    ///
    fn read_columns<I>(&self, columns: I) -> anyhow::Result<Array2<f32>>
    where
        I: IntoIterator<Item = usize>;

    /// Read rows within the range and return dense `ndarray::Array2`
    /// * `rows` : range e.g., 0..3 -> [0, 1, 2] or vec![0, 1, 2]
    ///
    fn read_rows<I>(&self, rows: I) -> anyhow::Result<Array2<f32>>
    where
        I: IntoIterator<Item = usize>;

    /// Number of rows in the underlying data matrix
    fn num_rows(&self) -> Option<usize>;

    /// Number of columns in the underlying data matrix
    fn num_columns(&self) -> Option<usize>;

    /// Number of non-zero elements
    fn num_non_zeros(&self) -> Option<usize>;

    /// Set row names for the matrix
    /// * `row_name_file`: a file each line contains row name words
    fn register_row_names(&mut self, row_name_file: &str);

    /// Set column names for the matrix
    /// * `column_name_file`: a file each line contains column name words
    fn register_column_names(&mut self, column_name_file: &str);

    /// Add arbitrary names (a vector of strings)
    /// * `group_name`: group name
    /// * `name_file`: a file each line contains name words
    /// * `name_columns`: range of columns to be used for name
    /// * `name_sep`: separator for name columns
    fn register_names(
        &mut self,
        key: &str,
        name_file: &str,
        name_columns: Range<usize>,
        name_sep: &str,
    ) -> anyhow::Result<()>;

    fn row_names(&self) -> anyhow::Result<Vec<Box<str>>>;

    fn column_names(&self) -> anyhow::Result<Vec<Box<str>>>;

    /// Get back the registered names
    /// * `key`: key for the registered names
    fn retrieve_registered_names(&self, key: &str) -> anyhow::Result<Vec<Box<str>>>;

    /// Reposition rows in a new order specified by `remap`
    /// * `remap` - a hashmap of old row index to new row index
    fn remap_rows(&mut self, remap: HashMap<usize, usize>) -> anyhow::Result<()>;
}
