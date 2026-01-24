use anyhow::Result;
use candle_core::Tensor;
use matrix_util::traits::IoOps;
use parquet::basic::{Compression, ConvertedType, Type as ParquetType, ZstdLevel};
use parquet::data_type::{ByteArray, ByteArrayType, FloatType};
use parquet::file::properties::WriterProperties;
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::types::Type;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use super::traits::VariationalDistribution;

/// Trait for outputting variational distribution parameters.
///
/// Provides methods to save variational parameters to files in various formats.
pub trait VariationalOutput {
    /// Write mean parameters to file (format detected from extension).
    fn write_mean(&self, path: &str) -> Result<()>;

    /// Write variance parameters to file.
    fn write_var(&self, path: &str) -> Result<()>;

    /// Write standard deviation parameters to file.
    fn write_std(&self, path: &str) -> Result<()>;

    /// Write all standard outputs with a header prefix.
    /// Creates files like: {header}.mean.gz, {header}.std.gz
    fn write_all(&self, header: &str) -> Result<()>;

    /// Write to parquet in melted (long) format with row/column names.
    ///
    /// Output columns: row, column, mean, std
    fn to_parquet(
        &self,
        row_names: Option<&[Box<str>]>,
        column_names: Option<&[Box<str>]>,
        file_path: &str,
    ) -> Result<()>;
}

/// Extended output trait for sparse variational distributions (e.g., Susie).
pub trait SparseVariationalOutput: VariationalOutput {
    /// Write posterior inclusion probabilities to file.
    fn write_pip(&self, path: &str) -> Result<()>;

    /// Write component selection probabilities (alpha) to file.
    fn write_alpha(&self, path: &str) -> Result<()>;

    /// Write all outputs including sparse-specific ones.
    /// Creates files like: {header}.mean.gz, {header}.pip.gz, etc.
    fn write_all_sparse(&self, header: &str) -> Result<()>;

    /// Write to parquet in melted format including PIP.
    ///
    /// Output columns: row, column, mean, std, pip
    fn to_parquet_sparse(
        &self,
        row_names: Option<&[Box<str>]>,
        column_names: Option<&[Box<str>]>,
        file_path: &str,
    ) -> Result<()>;
}

/// Extract the format extension from a path, handling .gz compression.
fn get_format_ext(path: &Path) -> String {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let name_lower = name.to_lowercase();
    if name_lower.ends_with(".gz") {
        let stem = &name[..name.len() - 3];
        Path::new(stem)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase()
    } else {
        path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase()
    }
}

/// Helper to detect format and write tensor to file.
fn write_tensor(tensor: &Tensor, path: &str) -> Result<()> {
    let path = Path::new(path);
    let ext = get_format_ext(path);
    let tensor = tensor.to_device(&candle_core::Device::Cpu)?;

    match ext.as_str() {
        "csv" => tensor.to_csv(path.to_str().unwrap())?,
        "parquet" | "pq" => tensor.to_parquet(None, None, path.to_str().unwrap())?,
        _ => tensor.to_tsv(path.to_str().unwrap())?,
    }
    Ok(())
}

/// Helper to melt a 2D tensor to (values, row_indices, col_indices).
fn melt_tensor(tensor: &Tensor) -> Result<(Vec<f32>, Vec<usize>, Vec<usize>)> {
    let tensor = tensor.to_device(&candle_core::Device::Cpu)?;
    let dims = tensor.dims();
    let (nrows, ncols) = (dims[0], dims[1]);

    let data: Vec<f32> = tensor.flatten_all()?.to_vec1()?;

    let mut row_idx = Vec::with_capacity(data.len());
    let mut col_idx = Vec::with_capacity(data.len());

    for i in 0..nrows {
        for j in 0..ncols {
            row_idx.push(i);
            col_idx.push(j);
        }
    }

    Ok((data, row_idx, col_idx))
}

/// Create a parquet schema with given fields.
fn create_schema(
    name: &str,
    fields: &[(&str, ParquetType, ConvertedType)],
) -> Result<Arc<Type>> {
    Ok(Arc::new(
        Type::group_type_builder(name)
            .with_fields(
                fields
                    .iter()
                    .map(|(field_name, parquet_type, converted_type)| {
                        Arc::new(
                            Type::primitive_type_builder(field_name, *parquet_type)
                                .with_repetition(parquet::basic::Repetition::REQUIRED)
                                .with_converted_type(*converted_type)
                                .build()
                                .unwrap(),
                        )
                    })
                    .collect(),
            )
            .build()?,
    ))
}

/// Convert indices to ByteArray names.
fn indices_to_names(indices: &[usize], names: Option<&[Box<str>]>) -> Vec<ByteArray> {
    indices
        .iter()
        .map(|&i| {
            if let Some(names) = names {
                ByteArray::from(names[i].as_ref())
            } else {
                ByteArray::from(i.to_string().as_bytes())
            }
        })
        .collect()
}

// Implementation for GaussianVar
impl VariationalOutput for super::GaussianVar {
    fn write_mean(&self, path: &str) -> Result<()> {
        let mean = VariationalDistribution::mean(self)?;
        write_tensor(&mean, path)
    }

    fn write_var(&self, path: &str) -> Result<()> {
        let var = VariationalDistribution::var(self)?;
        write_tensor(&var, path)
    }

    fn write_std(&self, path: &str) -> Result<()> {
        let std = self.std()?;
        write_tensor(&std, path)
    }

    fn write_all(&self, header: &str) -> Result<()> {
        self.write_mean(&format!("{}.mean.gz", header))?;
        self.write_std(&format!("{}.std.gz", header))?;
        Ok(())
    }

    fn to_parquet(
        &self,
        row_names: Option<&[Box<str>]>,
        column_names: Option<&[Box<str>]>,
        file_path: &str,
    ) -> Result<()> {
        let schema = create_schema(
            "GaussianVar",
            &[
                ("row", ParquetType::BYTE_ARRAY, ConvertedType::UTF8),
                ("column", ParquetType::BYTE_ARRAY, ConvertedType::UTF8),
                ("mean", ParquetType::FLOAT, ConvertedType::NONE),
                ("std", ParquetType::FLOAT, ConvertedType::NONE),
            ],
        )?;

        let mean = VariationalDistribution::mean(self)?;
        let std = self.std()?;

        let (mean_vals, row_idx, col_idx) = melt_tensor(&mean)?;
        let (std_vals, _, _) = melt_tensor(&std)?;

        let rows = indices_to_names(&row_idx, row_names);
        let cols = indices_to_names(&col_idx, column_names);

        // Write parquet
        let file = File::create(file_path)?;
        let zstd_level = ZstdLevel::try_new(5)?;
        let props = Arc::new(
            WriterProperties::builder()
                .set_compression(Compression::ZSTD(zstd_level))
                .build(),
        );
        let mut writer = SerializedFileWriter::new(file, schema, props)?;
        let mut row_group = writer.next_row_group()?;

        // Write name columns
        for data in [&rows, &cols] {
            if let Some(mut col_writer) = row_group.next_column()? {
                col_writer.typed::<ByteArrayType>().write_batch(data, None, None)?;
                col_writer.close()?;
            }
        }

        // Write value columns
        for data in [&mean_vals, &std_vals] {
            if let Some(mut col_writer) = row_group.next_column()? {
                col_writer.typed::<FloatType>().write_batch(data, None, None)?;
                col_writer.close()?;
            }
        }

        row_group.close()?;
        writer.close()?;
        Ok(())
    }
}

// Implementation for SusieVar
impl VariationalOutput for super::SusieVar {
    fn write_mean(&self, path: &str) -> Result<()> {
        let mean = self.theta_mean()?;
        write_tensor(&mean, path)
    }

    fn write_var(&self, path: &str) -> Result<()> {
        let var = VariationalDistribution::var(self)?;
        write_tensor(&var, path)
    }

    fn write_std(&self, path: &str) -> Result<()> {
        let var = VariationalDistribution::var(self)?;
        let std = var.sqrt()?;
        write_tensor(&std, path)
    }

    fn write_all(&self, header: &str) -> Result<()> {
        self.write_mean(&format!("{}.mean.gz", header))?;
        self.write_std(&format!("{}.std.gz", header))?;
        Ok(())
    }

    fn to_parquet(
        &self,
        row_names: Option<&[Box<str>]>,
        column_names: Option<&[Box<str>]>,
        file_path: &str,
    ) -> Result<()> {
        let schema = create_schema(
            "SusieVar",
            &[
                ("row", ParquetType::BYTE_ARRAY, ConvertedType::UTF8),
                ("column", ParquetType::BYTE_ARRAY, ConvertedType::UTF8),
                ("mean", ParquetType::FLOAT, ConvertedType::NONE),
                ("std", ParquetType::FLOAT, ConvertedType::NONE),
            ],
        )?;

        let mean = self.theta_mean()?;
        let var = VariationalDistribution::var(self)?;
        let std = var.sqrt()?;

        let (mean_vals, row_idx, col_idx) = melt_tensor(&mean)?;
        let (std_vals, _, _) = melt_tensor(&std)?;

        let rows = indices_to_names(&row_idx, row_names);
        let cols = indices_to_names(&col_idx, column_names);

        let file = File::create(file_path)?;
        let zstd_level = ZstdLevel::try_new(5)?;
        let props = Arc::new(
            WriterProperties::builder()
                .set_compression(Compression::ZSTD(zstd_level))
                .build(),
        );
        let mut writer = SerializedFileWriter::new(file, schema, props)?;
        let mut row_group = writer.next_row_group()?;

        for data in [&rows, &cols] {
            if let Some(mut col_writer) = row_group.next_column()? {
                col_writer.typed::<ByteArrayType>().write_batch(data, None, None)?;
                col_writer.close()?;
            }
        }

        for data in [&mean_vals, &std_vals] {
            if let Some(mut col_writer) = row_group.next_column()? {
                col_writer.typed::<FloatType>().write_batch(data, None, None)?;
                col_writer.close()?;
            }
        }

        row_group.close()?;
        writer.close()?;
        Ok(())
    }
}

impl SparseVariationalOutput for super::SusieVar {
    fn write_pip(&self, path: &str) -> Result<()> {
        let pip = self.pip()?;
        write_tensor(&pip, path)
    }

    fn write_alpha(&self, path: &str) -> Result<()> {
        let alpha = self.alpha()?;
        let dims = alpha.dims();
        if dims.len() == 3 {
            let (l, p, k) = (dims[0], dims[1], dims[2]);
            let alpha_2d = alpha.reshape((l * p, k))?;
            write_tensor(&alpha_2d, path)
        } else {
            write_tensor(&alpha, path)
        }
    }

    fn write_all_sparse(&self, header: &str) -> Result<()> {
        self.write_all(header)?;
        self.write_pip(&format!("{}.pip.gz", header))?;
        self.write_alpha(&format!("{}.alpha.gz", header))?;
        Ok(())
    }

    fn to_parquet_sparse(
        &self,
        row_names: Option<&[Box<str>]>,
        column_names: Option<&[Box<str>]>,
        file_path: &str,
    ) -> Result<()> {
        let schema = create_schema(
            "SusieVar",
            &[
                ("row", ParquetType::BYTE_ARRAY, ConvertedType::UTF8),
                ("column", ParquetType::BYTE_ARRAY, ConvertedType::UTF8),
                ("mean", ParquetType::FLOAT, ConvertedType::NONE),
                ("std", ParquetType::FLOAT, ConvertedType::NONE),
                ("pip", ParquetType::FLOAT, ConvertedType::NONE),
            ],
        )?;

        let mean = self.theta_mean()?;
        let var = VariationalDistribution::var(self)?;
        let std = var.sqrt()?;
        let pip = self.pip()?;

        let (mean_vals, row_idx, col_idx) = melt_tensor(&mean)?;
        let (std_vals, _, _) = melt_tensor(&std)?;
        let (pip_vals, _, _) = melt_tensor(&pip)?;

        let rows = indices_to_names(&row_idx, row_names);
        let cols = indices_to_names(&col_idx, column_names);

        let file = File::create(file_path)?;
        let zstd_level = ZstdLevel::try_new(5)?;
        let props = Arc::new(
            WriterProperties::builder()
                .set_compression(Compression::ZSTD(zstd_level))
                .build(),
        );
        let mut writer = SerializedFileWriter::new(file, schema, props)?;
        let mut row_group = writer.next_row_group()?;

        for data in [&rows, &cols] {
            if let Some(mut col_writer) = row_group.next_column()? {
                col_writer.typed::<ByteArrayType>().write_batch(data, None, None)?;
                col_writer.close()?;
            }
        }

        for data in [&mean_vals, &std_vals, &pip_vals] {
            if let Some(mut col_writer) = row_group.next_column()? {
                col_writer.typed::<FloatType>().write_batch(data, None, None)?;
                col_writer.close()?;
            }
        }

        row_group.close()?;
        writer.close()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};
    use tempfile::tempdir;

    #[test]
    fn test_gaussian_var_output() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        let gaussian = super::super::GaussianVar::new(vb, 10, 3)?;

        let dir = tempdir()?;
        let header = dir.path().join("test_gaussian");
        gaussian.write_all(header.to_str().unwrap())?;

        assert!(dir.path().join("test_gaussian.mean.gz").exists());
        assert!(dir.path().join("test_gaussian.std.gz").exists());

        // Test parquet output
        let pq_path = dir.path().join("test_gaussian.parquet");
        gaussian.to_parquet(None, None, pq_path.to_str().unwrap())?;
        assert!(pq_path.exists());

        Ok(())
    }

    #[test]
    fn test_susie_var_output() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        let susie = super::super::SusieVar::new(vb, 3, 10, 2)?;

        let dir = tempdir()?;
        let header = dir.path().join("test_susie");
        susie.write_all_sparse(header.to_str().unwrap())?;

        assert!(dir.path().join("test_susie.mean.gz").exists());
        assert!(dir.path().join("test_susie.std.gz").exists());
        assert!(dir.path().join("test_susie.pip.gz").exists());
        assert!(dir.path().join("test_susie.alpha.gz").exists());

        // Test parquet output with PIP
        let pq_path = dir.path().join("test_susie.parquet");
        susie.to_parquet_sparse(None, None, pq_path.to_str().unwrap())?;
        assert!(pq_path.exists());

        Ok(())
    }

    #[test]
    fn test_parquet_with_names() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        let susie = super::super::SusieVar::new(vb, 2, 5, 2)?;

        let row_names: Vec<Box<str>> = (0..5).map(|i| format!("gene_{}", i).into()).collect();
        let col_names: Vec<Box<str>> = vec!["output_0".into(), "output_1".into()];

        let dir = tempdir()?;
        let pq_path = dir.path().join("named.parquet");
        susie.to_parquet_sparse(Some(&row_names), Some(&col_names), pq_path.to_str().unwrap())?;
        assert!(pq_path.exists());

        Ok(())
    }
}
