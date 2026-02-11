use crate::traits::*;
use matrix_util::traits::{IoOps, MeltOps};

use parquet::basic::Type as ParquetType;
use parquet::basic::{Compression, ConvertedType, ZstdLevel};
use parquet::data_type::{ByteArray, ByteArrayType, FloatType};
use parquet::file::properties::WriterProperties;
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::types::Type;
use std::fs::File;
use std::sync::Arc;

/// Pre-compute ByteArray lookup table for names.
/// If names are provided, converts them to ByteArray.
/// Otherwise, generates numeric strings "0", "1", "2", ... for the given count.
fn precompute_name_bytes(names: Option<&[Box<str>]>, count: usize) -> Vec<ByteArray> {
    match names {
        Some(n) => n.iter().map(|s| ByteArray::from(s.as_ref())).collect(),
        None => (0..count)
            .map(|i| ByteArray::from(i.to_string().as_str()))
            .collect(),
    }
}

/// Build parquet schema for parameter matrices.
/// If `include_factor` is true, includes a "factor" column between "column" and "mean".
fn build_parquet_schema(include_factor: bool) -> anyhow::Result<Arc<Type>> {
    let mut fields: Vec<(&str, ParquetType, ConvertedType)> = vec![
        ("row", ParquetType::BYTE_ARRAY, ConvertedType::UTF8),
        ("column", ParquetType::BYTE_ARRAY, ConvertedType::UTF8),
    ];

    if include_factor {
        fields.push(("factor", ParquetType::BYTE_ARRAY, ConvertedType::UTF8));
    }

    fields.extend([
        ("mean", ParquetType::FLOAT, ConvertedType::NONE),
        ("sd", ParquetType::FLOAT, ConvertedType::NONE),
        ("log_mean", ParquetType::FLOAT, ConvertedType::NONE),
        ("log_sd", ParquetType::FLOAT, ConvertedType::NONE),
    ]);

    Ok(Arc::new(
        Type::group_type_builder("GammaMatrix")
            .with_fields(
                fields
                    .into_iter()
                    .map(|(name, parquet_type, converted_type)| {
                        Arc::new(
                            Type::primitive_type_builder(name, parquet_type)
                                .with_repetition(parquet::basic::Repetition::REQUIRED)
                                .with_converted_type(converted_type)
                                .build()
                                .unwrap(),
                        )
                    })
                    .collect(),
            )
            .build()?,
    ))
}

/// consolidated input and output
pub trait ParamIo: Inference
where
    f32: From<<<Self as Inference>::Mat as MeltOps>::Scalar>,
{
    type Mat: IoOps + MeltOps;

    fn to_tsv(&self, header: &str) -> anyhow::Result<()> {
        self.posterior_log_mean()
            .to_tsv(&(header.to_string() + ".log_mean.gz"))?;

        self.posterior_log_sd()
            .to_tsv(&(header.to_string() + ".log_sd.gz"))?;

        self.posterior_mean()
            .to_tsv(&(header.to_string() + ".mean.gz"))?;

        self.posterior_sd()
            .to_tsv(&(header.to_string() + ".sd.gz"))?;

        Ok(())
    }

    fn to_parquet_with_names(
        &self,
        file_path: &str,
        row_names: (Option<&[Box<str>]>, Option<&str>),
        column_names: Option<&[Box<str>]>,
    ) -> anyhow::Result<()> {
        let row_names_slice = row_names.0;
        let schema = build_parquet_schema(false)?;

        // Pre-compute name ByteArrays once for efficient lookup
        let row_bytes = precompute_name_bytes(row_names_slice, self.nrows());
        let col_bytes = precompute_name_bytes(column_names, self.ncols());

        // prepare data - single traversal for all matrices
        let mat_mean = self.posterior_mean();
        let mat_sd = self.posterior_sd();
        let mat_log_mean = self.posterior_log_mean();
        let mat_log_sd = self.posterior_log_sd();

        let (mut values, idx) =
            mat_mean.melt_many_with_indexes(&[mat_sd, mat_log_mean, mat_log_sd]);
        debug_assert_eq!(values.len(), 4);

        // Pop in reverse order: log_sd, log_mean, sd, mean
        let log_sd: Vec<f32> = values
            .pop()
            .unwrap()
            .into_iter()
            .map(|x| x.into())
            .collect();
        let log_mean: Vec<f32> = values
            .pop()
            .unwrap()
            .into_iter()
            .map(|x| x.into())
            .collect();
        let sd: Vec<f32> = values
            .pop()
            .unwrap()
            .into_iter()
            .map(|x| x.into())
            .collect();
        let mean: Vec<f32> = values
            .pop()
            .unwrap()
            .into_iter()
            .map(|x| x.into())
            .collect();

        // Map indices to pre-computed ByteArrays
        let rows: Vec<_> = idx[0].iter().map(|&i| row_bytes[i].clone()).collect();
        let cols: Vec<_> = idx[1].iter().map(|&i| col_bytes[i].clone()).collect();

        let nelem = mean.len();
        assert_eq!(nelem, sd.len());
        assert_eq!(nelem, log_sd.len());
        assert_eq!(nelem, log_mean.len());

        // write data to parquet
        let file = File::create(file_path)?;
        let zstd_level = ZstdLevel::try_new(5)?; // Specify ZSTD compression level (e.g., 5)
        let writer_properties = Arc::new(
            WriterProperties::builder()
                .set_compression(Compression::ZSTD(zstd_level))
                .build(),
        );
        let mut writer = SerializedFileWriter::new(file, schema, writer_properties)?;

        let mut row_group_writer = writer.next_row_group()?;

        let name_columns = vec![&rows, &cols];

        for data in name_columns {
            if let Some(mut column_writer) = row_group_writer.next_column()? {
                let typed_writer = column_writer.typed::<ByteArrayType>();
                typed_writer.write_batch(data, None, None)?;
                column_writer.close()?;
            }
        }

        let val_columns: Vec<&[f32]> = vec![
            mean.as_slice(),
            sd.as_slice(),
            log_mean.as_slice(),
            log_sd.as_slice(),
        ];

        for data in val_columns {
            if let Some(mut column_writer) = row_group_writer.next_column()? {
                let typed_writer = column_writer.typed::<FloatType>();
                typed_writer.write_batch(data, None, None)?;
                column_writer.close()?;
            }
        }

        row_group_writer.close()?;
        writer.close()?;

        Ok(())
    }

    /// Write to parquet with default names
    fn to_parquet(&self, file_path: &str) -> anyhow::Result<()> {
        self.to_parquet_with_names(file_path, (None, None), None)
    }
}

/// Write down a vector of matrix parameters into one parquet file.
///
/// * `parameters`: a vector of row x column parameters (factors)
/// * `row_names`: a vector of row names
/// * `column_names`: a vector of column names
/// * `factor_names`: a vector of factor names
/// * `file_path`
pub fn to_parquet<Param: Inference>(
    parameters: &[Param],
    row_names: Option<&[Box<str>]>,
    column_names: Option<&[Box<str>]>,
    factor_names: Option<&[Box<str>]>,
    file_path: &str,
) -> anyhow::Result<()>
where
    f32: From<<<Param as Inference>::Mat as MeltOps>::Scalar>,
{
    let factor_names: Vec<Box<str>> = match factor_names {
        Some(x) => x.iter().cloned().collect(),
        _ => (0..parameters.len())
            .map(|x| x.to_string().into_boxed_str())
            .collect(),
    };

    if parameters.is_empty() {
        return Err(anyhow::anyhow!("parameters cannot be empty"));
    }

    if factor_names.len() != parameters.len() {
        return Err(anyhow::anyhow!(
            "number of the parameters and factor names should match"
        ));
    }

    let schema = build_parquet_schema(true)?;

    // Write data to parquet
    let file = File::create(file_path)?;
    let zstd_level = ZstdLevel::try_new(5)?;
    let writer_properties = Arc::new(
        WriterProperties::builder()
            .set_compression(Compression::ZSTD(zstd_level))
            .build(),
    );
    let mut writer = SerializedFileWriter::new(file, schema, writer_properties)?;

    // Pre-compute name ByteArrays once (reused across all factors)
    let first_param = &parameters[0];
    let row_bytes = precompute_name_bytes(row_names, first_param.nrows());
    let col_bytes = precompute_name_bytes(column_names, first_param.ncols());

    for (factor_idx, param) in parameters.iter().enumerate() {
        // Single traversal for all matrices
        let mat_mean = param.posterior_mean();
        let mat_sd = param.posterior_sd();
        let mat_log_mean = param.posterior_log_mean();
        let mat_log_sd = param.posterior_log_sd();

        let (mut values, idx) =
            mat_mean.melt_many_with_indexes(&[mat_sd, mat_log_mean, mat_log_sd]);
        debug_assert_eq!(values.len(), 4);

        // Pop in reverse order: log_sd, log_mean, sd, mean
        let log_sd: Vec<f32> = values
            .pop()
            .unwrap()
            .into_iter()
            .map(|x| x.into())
            .collect();
        let log_mean: Vec<f32> = values
            .pop()
            .unwrap()
            .into_iter()
            .map(|x| x.into())
            .collect();
        let sd: Vec<f32> = values
            .pop()
            .unwrap()
            .into_iter()
            .map(|x| x.into())
            .collect();
        let mean: Vec<f32> = values
            .pop()
            .unwrap()
            .into_iter()
            .map(|x| x.into())
            .collect();

        let factor_name = factor_names[factor_idx].clone();
        let factor_label = ByteArray::from(factor_name.as_bytes());

        // Map indices to pre-computed ByteArrays
        let rows: Vec<_> = idx[0].iter().map(|&i| row_bytes[i].clone()).collect();
        let cols: Vec<_> = idx[1].iter().map(|&i| col_bytes[i].clone()).collect();

        let nelem = mean.len();
        assert_eq!(nelem, sd.len());
        assert_eq!(nelem, log_sd.len());
        assert_eq!(nelem, log_mean.len());

        // Start a new row group for this inference
        let mut row_group_writer = writer.next_row_group()?;

        // Write the "inference", "row", and "column" columns
        let name_columns = vec![rows, cols, vec![factor_label; nelem]];

        for data in name_columns {
            if let Some(mut column_writer) = row_group_writer.next_column()? {
                let typed_writer = column_writer.typed::<ByteArrayType>();
                typed_writer.write_batch(&data, None, None)?;
                column_writer.close()?;
            }
        }

        // Write the "mean", "sd", "log_mean", and "log_sd" columns
        let val_columns: Vec<&[f32]> = vec![
            mean.as_slice(),
            sd.as_slice(),
            log_mean.as_slice(),
            log_sd.as_slice(),
        ];

        for data in val_columns {
            if let Some(mut column_writer) = row_group_writer.next_column()? {
                let typed_writer = column_writer.typed::<FloatType>();
                typed_writer.write_batch(data, None, None)?;
                column_writer.close()?;
            }
        }

        row_group_writer.close()?;
    }

    // Close the writer
    writer.close()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dmatrix_gamma::GammaMatrix;
    use parquet::file::reader::{FileReader, SerializedFileReader};
    use parquet::record::RowAccessor;
    use std::collections::HashMap;

    #[test]
    fn test_param_io_to_parquet() -> anyhow::Result<()> {
        // Create a small GammaMatrix
        let nrows = 3;
        let ncols = 2;
        let mut gamma = GammaMatrix::new((nrows, ncols), 2.0, 1.0);
        gamma.calibrate();

        // Write to a temp file
        let temp_dir = tempfile::tempdir()?;
        let file_path = temp_dir.path().join("test_output.parquet");
        let file_path_str = file_path.to_str().unwrap();

        let row_names: Vec<Box<str>> = vec!["r0".into(), "r1".into(), "r2".into()];
        let col_names: Vec<Box<str>> = vec!["c0".into(), "c1".into()];

        gamma.to_parquet(file_path_str, Some(&row_names), Some(&col_names), None)?;

        // Read back and verify
        let file = File::open(&file_path)?;
        let reader = SerializedFileReader::new(file)?;
        let mut iter = reader.get_row_iter(None)?;

        // Collect all rows into a map keyed by (row, col)
        let mut results: HashMap<(String, String), (f32, f32, f32, f32)> = HashMap::new();
        while let Some(row) = iter.next() {
            let row = row?;
            let row_name = row.get_string(0)?.to_string();
            let col_name = row.get_string(1)?.to_string();
            let mean = row.get_float(2)?;
            let sd = row.get_float(3)?;
            let log_mean = row.get_float(4)?;
            let log_sd = row.get_float(5)?;
            results.insert((row_name, col_name), (mean, sd, log_mean, log_sd));
        }

        // Should have nrows * ncols entries
        assert_eq!(results.len(), nrows * ncols);

        // Verify all row/col combinations exist
        for r in &row_names {
            for c in &col_names {
                assert!(
                    results.contains_key(&(r.to_string(), c.to_string())),
                    "Missing entry for ({}, {})",
                    r,
                    c
                );
            }
        }

        // Verify values match the posterior estimates
        let mean_mat = gamma.posterior_mean();
        let sd_mat = gamma.posterior_sd();
        let log_mean_mat = gamma.posterior_log_mean();
        let log_sd_mat = gamma.posterior_log_sd();

        for (ri, r) in row_names.iter().enumerate() {
            for (ci, c) in col_names.iter().enumerate() {
                let (mean, sd, log_mean, log_sd) =
                    results.get(&(r.to_string(), c.to_string())).unwrap();

                let expected_mean = mean_mat[(ri, ci)];
                let expected_sd = sd_mat[(ri, ci)];
                let expected_log_mean = log_mean_mat[(ri, ci)];
                let expected_log_sd = log_sd_mat[(ri, ci)];

                assert!(
                    (mean - expected_mean).abs() < 1e-6,
                    "mean mismatch at ({}, {}): {} vs {}",
                    r,
                    c,
                    mean,
                    expected_mean
                );
                assert!(
                    (sd - expected_sd).abs() < 1e-6,
                    "sd mismatch at ({}, {}): {} vs {}",
                    r,
                    c,
                    sd,
                    expected_sd
                );
                assert!(
                    (log_mean - expected_log_mean).abs() < 1e-6,
                    "log_mean mismatch at ({}, {}): {} vs {}",
                    r,
                    c,
                    log_mean,
                    expected_log_mean
                );
                assert!(
                    (log_sd - expected_log_sd).abs() < 1e-6,
                    "log_sd mismatch at ({}, {}): {} vs {}",
                    r,
                    c,
                    log_sd,
                    expected_log_sd
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_param_io_to_parquet_without_names() -> anyhow::Result<()> {
        // Test with numeric indices instead of names
        let nrows = 2;
        let ncols = 3;
        let mut gamma = GammaMatrix::new((nrows, ncols), 1.5, 0.5);
        gamma.calibrate();

        let temp_dir = tempfile::tempdir()?;
        let file_path = temp_dir.path().join("test_no_names.parquet");
        let file_path_str = file_path.to_str().unwrap();

        gamma.to_parquet_simple(file_path_str)?;

        // Read back and verify
        let file = File::open(&file_path)?;
        let reader = SerializedFileReader::new(file)?;
        let mut iter = reader.get_row_iter(None)?;

        let mut count = 0;
        while let Some(row) = iter.next() {
            let row = row?;
            let row_idx: usize = row.get_string(0)?.parse()?;
            let col_idx: usize = row.get_string(1)?.parse()?;

            assert!(row_idx < nrows, "row index out of bounds: {}", row_idx);
            assert!(col_idx < ncols, "col index out of bounds: {}", col_idx);

            count += 1;
        }

        assert_eq!(count, nrows * ncols);

        Ok(())
    }

    #[test]
    fn test_to_parquet_multiple_factors() -> anyhow::Result<()> {
        let nrows = 2;
        let ncols = 2;
        let n_factors = 3;

        // Create multiple GammaMatrix parameters with different hyperparameters
        let mut params: Vec<GammaMatrix> = Vec::new();
        for i in 0..n_factors {
            let mut gamma = GammaMatrix::new((nrows, ncols), 1.0 + i as f32, 0.5 + i as f32 * 0.1);
            gamma.calibrate();
            params.push(gamma);
        }

        let temp_dir = tempfile::tempdir()?;
        let file_path = temp_dir.path().join("test_multi_factor.parquet");
        let file_path_str = file_path.to_str().unwrap();

        let row_names: Vec<Box<str>> = vec!["gene1".into(), "gene2".into()];
        let col_names: Vec<Box<str>> = vec!["cell1".into(), "cell2".into()];
        let factor_names: Vec<Box<str>> =
            vec!["factor0".into(), "factor1".into(), "factor2".into()];

        to_parquet(
            &params,
            Some(&row_names),
            Some(&col_names),
            Some(&factor_names),
            file_path_str,
        )?;

        // Read back and verify
        let file = File::open(&file_path)?;
        let reader = SerializedFileReader::new(file)?;
        let mut iter = reader.get_row_iter(None)?;

        // Collect results keyed by (row, col, factor)
        let mut results: HashMap<(String, String, String), (f32, f32, f32, f32)> = HashMap::new();
        while let Some(row) = iter.next() {
            let row = row?;
            let row_name = row.get_string(0)?.to_string();
            let col_name = row.get_string(1)?.to_string();
            let factor_name = row.get_string(2)?.to_string();
            let mean = row.get_float(3)?;
            let sd = row.get_float(4)?;
            let log_mean = row.get_float(5)?;
            let log_sd = row.get_float(6)?;
            results.insert(
                (row_name, col_name, factor_name),
                (mean, sd, log_mean, log_sd),
            );
        }

        // Should have nrows * ncols * n_factors entries
        assert_eq!(results.len(), nrows * ncols * n_factors);

        // Verify values for each factor
        for (fi, param) in params.iter().enumerate() {
            let factor = &factor_names[fi];
            let mean_mat = param.posterior_mean();
            let sd_mat = param.posterior_sd();

            for (ri, r) in row_names.iter().enumerate() {
                for (ci, c) in col_names.iter().enumerate() {
                    let key = (r.to_string(), c.to_string(), factor.to_string());
                    let (mean, sd, _, _) = results.get(&key).expect("Missing entry");

                    let expected_mean = mean_mat[(ri, ci)];
                    let expected_sd = sd_mat[(ri, ci)];

                    assert!(
                        (mean - expected_mean).abs() < 1e-6,
                        "mean mismatch for factor {} at ({}, {})",
                        factor,
                        r,
                        c
                    );
                    assert!(
                        (sd - expected_sd).abs() < 1e-6,
                        "sd mismatch for factor {} at ({}, {})",
                        factor,
                        r,
                        c
                    );
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_to_parquet_empty_parameters() {
        let params: Vec<GammaMatrix> = vec![];
        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join("test_empty.parquet");
        let file_path_str = file_path.to_str().unwrap();

        let result = to_parquet::<GammaMatrix>(&params, None, None, None, file_path_str);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }
}
