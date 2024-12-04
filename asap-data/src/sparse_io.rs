use ndarray::prelude::*;

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
}
