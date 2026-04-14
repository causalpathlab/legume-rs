#![allow(dead_code)]

use crate::common_io::{file_ext, write_lines};
use crate::parquet::{parquet_add_numeric_column, parquet_add_string_column, ParquetWriter};
use crate::traits::RunningStatOps;
use nalgebra_sparse::CscMatrix;
use num_traits::{Float, ToPrimitive, Zero};
use parquet::basic::Type as ParquetType;
use std::fmt::Display;
use std::iter::Sum;
use std::ops::AddAssign;

const STAT_COLUMN_NAMES: [&str; 4] = ["nnz", "tot", "mu", "sig"];

/// Denominator for mean/variance: clamp to a tiny positive to avoid
/// divide-by-zero when no samples have been accumulated yet.
fn safe_denom<T: Float>(n: usize) -> T {
    let n = T::from(n).unwrap_or(T::one());
    if n > T::zero() {
        n
    } else {
        T::from(1e-8).unwrap_or(T::one())
    }
}

/// Running statistics that accepts sparse column input but stores
/// sufficient statistics in dense vectors.
///
/// This is more efficient than `RunningStatistics<Ix1>` when the input
/// data is sparse and has many rows, as it avoids materializing dense
/// matrices during reads.
///
#[derive(Clone)]
pub struct SparseRunningStatistics<T>
where
    T: Float,
{
    nrows: usize,
    ncols_processed: usize,
    npos: Vec<T>,
    s1: Vec<T>,
    s2: Vec<T>,
}

impl<T> SparseRunningStatistics<T>
where
    T: Float + AddAssign + Sum + Zero,
{
    /// Create a new SparseRunningStatistics object
    ///
    /// # Arguments
    /// * `nrows` - Number of rows (features)
    ///
    pub fn new(nrows: usize) -> Self {
        SparseRunningStatistics {
            nrows,
            ncols_processed: 0,
            npos: vec![T::zero(); nrows],
            s1: vec![T::zero(); nrows],
            s2: vec![T::zero(); nrows],
        }
    }

    /// Add a sparse column (row indices + values) to the running
    /// statistics. The column advances `ncols_processed` by one.
    pub fn add_sparse_column(&mut self, row_indices: &[usize], values: &[T]) {
        debug_assert_eq!(row_indices.len(), values.len());

        for (&row, &val) in row_indices.iter().zip(values.iter()) {
            if val.is_finite() {
                if val > T::zero() {
                    self.npos[row] += T::one();
                }
                self.s1[row] += val;
                self.s2[row] += val * val;
            }
        }
        self.ncols_processed += 1;
    }

    pub fn nrows(&self) -> usize {
        self.nrows
    }

    pub fn ncols_processed(&self) -> usize {
        self.ncols_processed
    }

    fn denom(&self) -> T {
        safe_denom::<T>(self.ncols_processed)
    }

    /// Convert to owned vectors (npos, sum, mean, std)
    pub fn to_vecs(&self) -> (Vec<T>, Vec<T>, Vec<T>, Vec<T>) {
        (self.npos.clone(), self.s1.clone(), self.mean(), self.std())
    }

    /// Add columns from a CscMatrix
    ///
    /// # Arguments
    /// * `csc` - Sparse matrix in CSC format
    ///
    pub fn add_csc(&mut self, csc: &CscMatrix<T>) {
        for col in csc.col_iter() {
            let rows = col.row_indices();
            let vals = col.values();
            self.add_sparse_column(rows, vals);
        }
    }
}

impl<T> SparseRunningStatistics<T>
where
    T: Float + AddAssign + Sum + Zero + Display + ToPrimitive,
{
    /// Save the statistics to a file (parquet if `filename` ends with
    /// `.parquet`, otherwise a separator-delimited text file).
    pub fn save(&self, filename: &str, names: &[Box<str>], sep: &str) -> anyhow::Result<()> {
        let (nnz, tot, mu, sig) = self.to_f32_vecs();
        write_stat_file(
            filename,
            names,
            sep,
            StatColumns {
                nnz: &nnz,
                tot: &tot,
                mu: &mu,
                sig: &sig,
            },
        )
    }

    /// Get statistics as vectors (nnz, tot, mu, sig) converted to f32
    pub fn to_f32_vecs(&self) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let to_f32_slice =
            |v: &[T]| -> Vec<f32> { v.iter().map(|x| x.to_f32().unwrap_or(0.0)).collect() };
        let nnz = to_f32_slice(&self.npos);
        let tot = to_f32_slice(&self.s1);
        let mu = to_f32_slice(&self.mean());
        let sig = to_f32_slice(&self.std());
        (nnz, tot, mu, sig)
    }

    /// Convert statistics to string vectors for output
    pub fn to_string_vec(&self, names: &[Box<str>], sep: &str) -> anyhow::Result<Vec<Box<str>>> {
        if names.len() != self.nrows {
            anyhow::bail!(
                "The number of names ({}) does not match nrows ({})",
                names.len(),
                self.nrows
            );
        }

        let nnz = self.count_positives();
        let tot = self.sum();
        let mu = self.mean();
        let sig = self.std();

        let out: Vec<Box<str>> = (0..self.nrows)
            .map(|i| {
                format!(
                    "{}{}{}{}{}{}{}{}{}",
                    names[i],
                    sep,
                    format_value(nnz[i]),
                    sep,
                    format_value(tot[i]),
                    sep,
                    format_value(mu[i]),
                    sep,
                    format_value(sig[i])
                )
                .into_boxed_str()
            })
            .collect();
        Ok(out)
    }
}

/// Running statistics computed per-column directly from sparse (CSC) input.
///
/// Unlike [`SparseRunningStatistics`] which tracks per-row statistics across
/// columns, this tracks per-column statistics: nnz, sum, and sum-of-squares
/// for each column. Mean/variance use `nrows` as the denominator (implicit
/// zeros contribute zero values).
///
/// CSC slabs are consumed directly — no triplet materialization — and each
/// slab only needs to know its starting column offset in the global index
/// space.
#[derive(Clone)]
pub struct SparseColumnRunningStatistics<T>
where
    T: Float,
{
    nrows: usize,
    npos: Vec<T>,
    s1: Vec<T>,
    s2: Vec<T>,
}

impl<T> SparseColumnRunningStatistics<T>
where
    T: Float + AddAssign + Sum + Zero,
{
    /// Create a new per-column running statistics accumulator.
    ///
    /// # Arguments
    /// * `ncols` — total number of columns whose statistics will be tracked
    /// * `nrows` — row denominator used for mean/variance (number of rows
    ///   considered; for a row-filtered scan this should be the number of
    ///   rows kept)
    pub fn new(ncols: usize, nrows: usize) -> Self {
        Self {
            nrows,
            npos: vec![T::zero(); ncols],
            s1: vec![T::zero(); ncols],
            s2: vec![T::zero(); ncols],
        }
    }

    /// Accumulate statistics from a CSC slab. `col_offset` is the global
    /// column index of the slab's local column 0.
    pub fn add_csc(&mut self, csc: &CscMatrix<T>, col_offset: usize) {
        self.add_csc_inner(csc, col_offset, None);
    }

    /// Accumulate statistics from a CSC slab, keeping only rows where
    /// `row_mask[row]` is `true`. `row_mask` must cover every row index
    /// appearing in the CSC (length ≥ csc.nrows()).
    pub fn add_csc_masked(&mut self, csc: &CscMatrix<T>, col_offset: usize, row_mask: &[bool]) {
        debug_assert!(row_mask.len() >= csc.nrows());
        self.add_csc_inner(csc, col_offset, Some(row_mask));
    }

    fn add_csc_inner(&mut self, csc: &CscMatrix<T>, col_offset: usize, row_mask: Option<&[bool]>) {
        for (local_col, col) in csc.col_iter().enumerate() {
            let c = col_offset + local_col;
            let rows = col.row_indices();
            let vals = col.values();
            for (&row, &val) in rows.iter().zip(vals.iter()) {
                if let Some(mask) = row_mask {
                    if !mask[row] {
                        continue;
                    }
                }
                if !val.is_finite() {
                    continue;
                }
                if val > T::zero() {
                    self.npos[c] += T::one();
                }
                self.s1[c] += val;
                self.s2[c] += val * val;
            }
        }
    }

    pub fn ncols(&self) -> usize {
        self.npos.len()
    }

    pub fn nrows(&self) -> usize {
        self.nrows
    }

    fn denom(&self) -> T {
        safe_denom::<T>(self.nrows)
    }
}

impl<T> SparseColumnRunningStatistics<T>
where
    T: Float + AddAssign + Sum + Zero + Display + ToPrimitive,
{
    /// Save the per-column statistics to a file (parquet if `filename`
    /// ends with `.parquet`, otherwise a separator-delimited text file).
    pub fn save(&self, filename: &str, names: &[Box<str>], sep: &str) -> anyhow::Result<()> {
        let (nnz, tot, mu, sig) = self.to_f32_vecs();
        write_stat_file(
            filename,
            names,
            sep,
            StatColumns {
                nnz: &nnz,
                tot: &tot,
                mu: &mu,
                sig: &sig,
            },
        )
    }

    /// Get statistics as vectors (nnz, tot, mu, sig) converted to f32
    pub fn to_f32_vecs(&self) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let to_f32_slice =
            |v: &[T]| -> Vec<f32> { v.iter().map(|x| x.to_f32().unwrap_or(0.0)).collect() };
        let nnz = to_f32_slice(&self.npos);
        let tot = to_f32_slice(&self.s1);
        let mu = to_f32_slice(&self.mean());
        let sig = to_f32_slice(&self.std());
        (nnz, tot, mu, sig)
    }
}

impl<T> RunningStatOps<T> for SparseColumnRunningStatistics<T>
where
    T: Float + AddAssign + Sum + Zero,
{
    type Output = Vec<T>;

    fn clear(&mut self) {
        self.npos.fill(T::zero());
        self.s1.fill(T::zero());
        self.s2.fill(T::zero());
    }

    fn count_positives(&self) -> Vec<T> {
        self.npos.clone()
    }

    fn sum(&self) -> Vec<T> {
        self.s1.clone()
    }

    fn mean(&self) -> Vec<T> {
        let n = self.denom();
        self.s1.iter().map(|&s| s / n).collect()
    }

    fn variance(&self) -> Vec<T> {
        let n = self.denom();
        self.s1
            .iter()
            .zip(self.s2.iter())
            .map(|(&s1, &s2)| {
                let mu = s1 / n;
                s2 / n - mu * mu
            })
            .collect()
    }

    fn std(&self) -> Vec<T> {
        self.variance().into_iter().map(|v| v.sqrt()).collect()
    }
}

/// Four-column stat table (nnz, tot, mu, sig) to be written by
/// [`write_stat_file`]. All slices must have the same length.
struct StatColumns<'a> {
    nnz: &'a [f32],
    tot: &'a [f32],
    mu: &'a [f32],
    sig: &'a [f32],
}

impl StatColumns<'_> {
    fn len(&self) -> usize {
        self.nnz.len()
    }
}

/// Shared parquet/TSV writer for per-entity stat tables with the
/// standard (name, nnz, tot, mu, sig) schema.
fn write_stat_file(
    filename: &str,
    names: &[Box<str>],
    sep: &str,
    stats: StatColumns<'_>,
) -> anyhow::Result<()> {
    let n = stats.len();
    if names.len() != n {
        anyhow::bail!(
            "The number of names ({}) does not match stat length ({})",
            names.len(),
            n
        );
    }

    match file_ext(filename).unwrap_or(Box::from("")).as_ref() {
        "parquet" => {
            let column_names: Vec<Box<str>> =
                STAT_COLUMN_NAMES.iter().map(|s| (*s).into()).collect();
            let column_types = vec![ParquetType::FLOAT; STAT_COLUMN_NAMES.len()];

            let parquet_writer = ParquetWriter::new(
                filename,
                (n, 4),
                (Some(names), Some(&column_names)),
                Some(&column_types),
                None,
            )?;

            let mut writer = parquet_writer.get_writer()?;
            let mut row_group_writer = writer.next_row_group()?;

            parquet_add_string_column(&mut row_group_writer, names)?;
            parquet_add_numeric_column(&mut row_group_writer, stats.nnz)?;
            parquet_add_numeric_column(&mut row_group_writer, stats.tot)?;
            parquet_add_numeric_column(&mut row_group_writer, stats.mu)?;
            parquet_add_numeric_column(&mut row_group_writer, stats.sig)?;

            row_group_writer.close()?;
            writer.close()?;
        }
        _ => {
            let mut out: Vec<Box<str>> = (0..n)
                .map(|i| {
                    format!(
                        "{}{}{}{}{}{}{}{}{}",
                        names[i],
                        sep,
                        format_value(stats.nnz[i]),
                        sep,
                        format_value(stats.tot[i]),
                        sep,
                        format_value(stats.mu[i]),
                        sep,
                        format_value(stats.sig[i])
                    )
                    .into_boxed_str()
                })
                .collect();
            let header = format!("#name{}nnz{}tot{}mu{}sig", sep, sep, sep, sep);
            out.insert(0, header.into_boxed_str());
            write_lines(&out, filename)?;
        }
    }
    Ok(())
}

/// Save multiple group statistics to a single parquet file with a group column
pub fn save_grouped_stats_parquet(
    filename: &str,
    names: &[Box<str>],
    group_names: &[Box<str>],
    group_stats: &[SparseRunningStatistics<f32>],
) -> anyhow::Result<()> {
    use parquet::basic::{Compression, ZstdLevel};
    use parquet::file::properties::WriterProperties;
    use parquet::file::writer::SerializedFileWriter;
    use parquet::schema::types::Type as SchemaType;
    use std::sync::Arc;

    if group_names.len() != group_stats.len() {
        anyhow::bail!(
            "Number of group names ({}) does not match number of group stats ({})",
            group_names.len(),
            group_stats.len()
        );
    }

    // Calculate total rows
    let total_rows: usize = group_stats.iter().map(|s| s.nrows()).sum();

    // Build merged vectors
    let mut all_names: Vec<Box<str>> = Vec::with_capacity(total_rows);
    let mut all_groups: Vec<Box<str>> = Vec::with_capacity(total_rows);
    let mut all_nnz: Vec<f32> = Vec::with_capacity(total_rows);
    let mut all_tot: Vec<f32> = Vec::with_capacity(total_rows);
    let mut all_mu: Vec<f32> = Vec::with_capacity(total_rows);
    let mut all_sig: Vec<f32> = Vec::with_capacity(total_rows);

    for (group_name, stat) in group_names.iter().zip(group_stats.iter()) {
        let (nnz, tot, mu, sig) = stat.to_f32_vecs();
        for (i, name) in names.iter().enumerate() {
            all_names.push(name.clone());
            all_groups.push(group_name.clone());
            all_nnz.push(nnz[i]);
            all_tot.push(tot[i]);
            all_mu.push(mu[i]);
            all_sig.push(sig[i]);
        }
    }

    // Build schema: name, group, nnz, tot, mu, sig
    let fields = vec![
        Arc::new(
            SchemaType::primitive_type_builder("name", ParquetType::BYTE_ARRAY)
                .with_repetition(parquet::basic::Repetition::REQUIRED)
                .with_converted_type(parquet::basic::ConvertedType::UTF8)
                .build()?,
        ),
        Arc::new(
            SchemaType::primitive_type_builder("group", ParquetType::BYTE_ARRAY)
                .with_repetition(parquet::basic::Repetition::REQUIRED)
                .with_converted_type(parquet::basic::ConvertedType::UTF8)
                .build()?,
        ),
        Arc::new(
            SchemaType::primitive_type_builder("nnz", ParquetType::FLOAT)
                .with_repetition(parquet::basic::Repetition::REQUIRED)
                .build()?,
        ),
        Arc::new(
            SchemaType::primitive_type_builder("tot", ParquetType::FLOAT)
                .with_repetition(parquet::basic::Repetition::REQUIRED)
                .build()?,
        ),
        Arc::new(
            SchemaType::primitive_type_builder("mu", ParquetType::FLOAT)
                .with_repetition(parquet::basic::Repetition::REQUIRED)
                .build()?,
        ),
        Arc::new(
            SchemaType::primitive_type_builder("sig", ParquetType::FLOAT)
                .with_repetition(parquet::basic::Repetition::REQUIRED)
                .build()?,
        ),
    ];

    let schema = Arc::new(
        SchemaType::group_type_builder("grouped_stats")
            .with_fields(fields)
            .build()?,
    );

    let zstd_level = ZstdLevel::try_new(5)?;
    let props = Arc::new(
        WriterProperties::builder()
            .set_compression(Compression::ZSTD(zstd_level))
            .build(),
    );

    let file = std::fs::File::create(filename)?;
    let mut writer = SerializedFileWriter::new(file, schema, props)?;
    let mut row_group_writer = writer.next_row_group()?;

    parquet_add_string_column(&mut row_group_writer, &all_names)?;
    parquet_add_string_column(&mut row_group_writer, &all_groups)?;
    parquet_add_numeric_column(&mut row_group_writer, &all_nnz)?;
    parquet_add_numeric_column(&mut row_group_writer, &all_tot)?;
    parquet_add_numeric_column(&mut row_group_writer, &all_mu)?;
    parquet_add_numeric_column(&mut row_group_writer, &all_sig)?;

    row_group_writer.close()?;
    writer.close()?;

    Ok(())
}

fn format_value<T: Float + Display>(v: T) -> String {
    let v_f64 = v.to_f64().unwrap_or(0.0);
    if v_f64.abs() > 1e-4 {
        format!("{:.4}", v_f64)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    } else if v_f64.abs() > 1e-20 {
        format!("{:.4e}", v_f64)
    } else {
        "0".to_string()
    }
}

impl<T> RunningStatOps<T> for SparseRunningStatistics<T>
where
    T: Float + AddAssign + Sum + Zero,
{
    type Output = Vec<T>;

    fn clear(&mut self) {
        self.ncols_processed = 0;
        self.npos.fill(T::zero());
        self.s1.fill(T::zero());
        self.s2.fill(T::zero());
    }

    /// Count of positive (non-zero) values per row
    fn count_positives(&self) -> Vec<T> {
        self.npos.clone()
    }

    /// Sum per row
    fn sum(&self) -> Vec<T> {
        self.s1.clone()
    }

    /// Mean per row
    /// Uses ncols_processed as the denominator (implicit zeros count)
    fn mean(&self) -> Vec<T> {
        let n = self.denom();
        self.s1.iter().map(|&s| s / n).collect()
    }

    /// Variance per row
    fn variance(&self) -> Vec<T> {
        let n = self.denom();
        self.s1
            .iter()
            .zip(self.s2.iter())
            .map(|(&s1, &s2)| {
                let mu = s1 / n;
                s2 / n - mu * mu
            })
            .collect()
    }

    /// Standard deviation per row
    fn std(&self) -> Vec<T> {
        self.variance().into_iter().map(|v| v.sqrt()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_running_stat_basic() {
        let mut stat = SparseRunningStatistics::<f32>::new(4);

        // Column 0: [1, 0, 2, 0]
        stat.add_sparse_column(&[0, 2], &[1.0, 2.0]);

        // Column 1: [0, 3, 0, 4]
        stat.add_sparse_column(&[1, 3], &[3.0, 4.0]);

        assert_eq!(stat.ncols_processed(), 2);

        // npos: [1, 1, 1, 1]
        assert_eq!(stat.count_positives(), vec![1.0, 1.0, 1.0, 1.0]);

        // sum: [1, 3, 2, 4]
        assert_eq!(stat.sum(), vec![1.0, 3.0, 2.0, 4.0]);

        // mean: [0.5, 1.5, 1.0, 2.0]
        let mean = stat.mean();
        assert!((mean[0] - 0.5).abs() < 1e-6);
        assert!((mean[1] - 1.5).abs() < 1e-6);
        assert!((mean[2] - 1.0).abs() < 1e-6);
        assert!((mean[3] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_running_stat_csc() {
        use nalgebra_sparse::CooMatrix;

        let mut stat = SparseRunningStatistics::<f32>::new(3);

        // Create a 3x2 sparse matrix:
        // [1, 0]
        // [0, 2]
        // [3, 0]
        let mut coo: CooMatrix<f32> = CooMatrix::new(3, 2);
        coo.push(0, 0, 1.0);
        coo.push(1, 1, 2.0);
        coo.push(2, 0, 3.0);
        let csc = CscMatrix::from(&coo);

        stat.add_csc(&csc);

        assert_eq!(stat.ncols_processed(), 2);
        assert_eq!(stat.count_positives(), vec![1.0, 1.0, 1.0]);
        assert_eq!(stat.sum(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_sparse_running_stat_f64() {
        let mut stat = SparseRunningStatistics::<f64>::new(2);

        stat.add_sparse_column(&[0, 1], &[1.0, 2.0]);
        stat.add_sparse_column(&[0], &[3.0]);

        assert_eq!(stat.ncols_processed(), 2);
        assert_eq!(stat.sum(), vec![4.0, 2.0]);

        let mean = stat.mean();
        assert!((mean[0] - 2.0).abs() < 1e-10);
        assert!((mean[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_column_running_stat_csc() {
        use nalgebra_sparse::CooMatrix;

        // 3 rows x 4 columns:
        // col0 = [1, 0, 3]
        // col1 = [0, 2, 0]
        // col2 = [0, 0, 0]
        // col3 = [4, 5, 0]
        let mut coo: CooMatrix<f32> = CooMatrix::new(3, 4);
        coo.push(0, 0, 1.0);
        coo.push(2, 0, 3.0);
        coo.push(1, 1, 2.0);
        coo.push(0, 3, 4.0);
        coo.push(1, 3, 5.0);
        let csc = CscMatrix::from(&coo);

        let mut stat = SparseColumnRunningStatistics::<f32>::new(4, 3);
        stat.add_csc(&csc, 0);

        assert_eq!(stat.count_positives(), vec![2.0, 1.0, 0.0, 2.0]);
        assert_eq!(stat.sum(), vec![4.0, 2.0, 0.0, 9.0]);

        // mean = sum / nrows (denominator = 3)
        let mean = stat.mean();
        assert!((mean[0] - 4.0 / 3.0).abs() < 1e-6);
        assert!((mean[1] - 2.0 / 3.0).abs() < 1e-6);
        assert!((mean[2] - 0.0).abs() < 1e-6);
        assert!((mean[3] - 9.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_column_running_stat_block_offset() {
        use nalgebra_sparse::CooMatrix;

        // Simulate processing columns in two blocks of a 3x4 matrix:
        // Block A (cols 0..2): col0=[1,0,3], col1=[0,2,0]
        // Block B (cols 2..4): col2=[0,0,0], col3=[4,5,0]
        let mut coo_a: CooMatrix<f32> = CooMatrix::new(3, 2);
        coo_a.push(0, 0, 1.0);
        coo_a.push(2, 0, 3.0);
        coo_a.push(1, 1, 2.0);
        let csc_a = CscMatrix::from(&coo_a);

        let mut coo_b: CooMatrix<f32> = CooMatrix::new(3, 2);
        coo_b.push(0, 1, 4.0);
        coo_b.push(1, 1, 5.0);
        let csc_b = CscMatrix::from(&coo_b);

        let mut stat = SparseColumnRunningStatistics::<f32>::new(4, 3);
        stat.add_csc(&csc_a, 0);
        stat.add_csc(&csc_b, 2);

        assert_eq!(stat.count_positives(), vec![2.0, 1.0, 0.0, 2.0]);
        assert_eq!(stat.sum(), vec![4.0, 2.0, 0.0, 9.0]);
    }

    #[test]
    fn test_sparse_column_running_stat_masked() {
        use nalgebra_sparse::CooMatrix;

        // 4 rows x 2 columns, keep only rows [0, 2]
        // col0 = [1, 2, 3, 4] → kept: 1+3 = 4, nnz=2
        // col1 = [0, 5, 0, 6] → kept: 0, nnz=0
        let mut coo: CooMatrix<f32> = CooMatrix::new(4, 2);
        coo.push(0, 0, 1.0);
        coo.push(1, 0, 2.0);
        coo.push(2, 0, 3.0);
        coo.push(3, 0, 4.0);
        coo.push(1, 1, 5.0);
        coo.push(3, 1, 6.0);
        let csc = CscMatrix::from(&coo);

        let row_mask = vec![true, false, true, false];

        let mut stat = SparseColumnRunningStatistics::<f32>::new(2, 2);
        stat.add_csc_masked(&csc, 0, &row_mask);

        assert_eq!(stat.count_positives(), vec![2.0, 0.0]);
        assert_eq!(stat.sum(), vec![4.0, 0.0]);
    }
}
