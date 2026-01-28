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

    /// Add a sparse column to the running statistics
    ///
    /// # Arguments
    /// * `row_indices` - Row indices of non-zero values
    /// * `values` - Non-zero values
    ///
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

    /// Add from sparse triplets (row, col, value)
    ///
    /// # Arguments
    /// * `ncols` - Number of columns in this batch
    /// * `triplets` - Vector of (row, col, value) triplets
    ///
    pub fn add_triplets(&mut self, ncols: usize, triplets: &[(usize, usize, T)]) {
        for &(row, _, val) in triplets {
            if val.is_finite() {
                if val > T::zero() {
                    self.npos[row] += T::one();
                }
                self.s1[row] += val;
                self.s2[row] += val * val;
            }
        }
        self.ncols_processed += ncols;
    }

    /// Number of rows
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of columns processed so far
    pub fn ncols_processed(&self) -> usize {
        self.ncols_processed
    }

    /// Get the denominator for mean/variance calculations
    fn denom(&self) -> T {
        let n = T::from(self.ncols_processed).unwrap_or(T::one());
        if n > T::zero() {
            n
        } else {
            T::from(1e-8).unwrap_or(T::one())
        }
    }

    /// Convert to owned vectors (npos, sum, mean, std)
    pub fn to_vecs(&self) -> (Vec<T>, Vec<T>, Vec<T>, Vec<T>) {
        (
            self.npos.clone(),
            self.s1.clone(),
            self.mean(),
            self.std(),
        )
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
    /// Save the statistics to a file
    /// # Arguments
    /// * `filename` - The name of the file to save the statistics to
    /// * `names` - The names of the rows
    /// * `sep` - Separator string
    pub fn save(&self, filename: &str, names: &[Box<str>], sep: &str) -> anyhow::Result<()> {
        match file_ext(filename).unwrap_or(Box::from("")).as_ref() {
            "parquet" => {
                self.save_parquet(filename, names)?;
            }
            _ => {
                let mut out = self.to_string_vec(names, sep)?;
                let header = format!("#name{}nnz{}tot{}mu{}sig", sep, sep, sep, sep);
                out.insert(0, header.into_boxed_str());
                write_lines(&out, filename)?;
            }
        };
        Ok(())
    }

    /// Save statistics to a parquet file
    fn save_parquet(&self, filename: &str, names: &[Box<str>]) -> anyhow::Result<()> {
        if names.len() != self.nrows {
            anyhow::bail!(
                "The number of names ({}) does not match nrows ({})",
                names.len(),
                self.nrows
            );
        }

        let nnz: Vec<f32> = self
            .count_positives()
            .iter()
            .map(|v| v.to_f32().unwrap_or(0.0))
            .collect();
        let tot: Vec<f32> = self.sum().iter().map(|v| v.to_f32().unwrap_or(0.0)).collect();
        let mu: Vec<f32> = self.mean().iter().map(|v| v.to_f32().unwrap_or(0.0)).collect();
        let sig: Vec<f32> = self.std().iter().map(|v| v.to_f32().unwrap_or(0.0)).collect();

        let column_names: Vec<Box<str>> = vec!["nnz", "tot", "mu", "sig"]
            .into_iter()
            .map(|s| s.into())
            .collect();

        let column_types = vec![
            ParquetType::FLOAT,
            ParquetType::FLOAT,
            ParquetType::FLOAT,
            ParquetType::FLOAT,
        ];

        let parquet_writer = ParquetWriter::new(
            filename,
            (self.nrows, 4),
            (Some(names), Some(&column_names)),
            Some(&column_types),
        )?;

        let mut writer = parquet_writer.get_writer()?;
        let mut row_group_writer = writer.next_row_group()?;

        parquet_add_string_column(&mut row_group_writer, names)?;
        parquet_add_numeric_column(&mut row_group_writer, &nnz)?;
        parquet_add_numeric_column(&mut row_group_writer, &tot)?;
        parquet_add_numeric_column(&mut row_group_writer, &mu)?;
        parquet_add_numeric_column(&mut row_group_writer, &sig)?;

        row_group_writer.close()?;
        writer.close()?;

        Ok(())
    }

    /// Get statistics as vectors (nnz, tot, mu, sig) converted to f32
    pub fn to_f32_vecs(&self) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let nnz: Vec<f32> = self
            .count_positives()
            .iter()
            .map(|v| v.to_f32().unwrap_or(0.0))
            .collect();
        let tot: Vec<f32> = self.sum().iter().map(|v| v.to_f32().unwrap_or(0.0)).collect();
        let mu: Vec<f32> = self.mean().iter().map(|v| v.to_f32().unwrap_or(0.0)).collect();
        let sig: Vec<f32> = self.std().iter().map(|v| v.to_f32().unwrap_or(0.0)).collect();
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
}
