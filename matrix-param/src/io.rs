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

    fn to_parquet(
        &self,
        row_names: Option<&[Box<str>]>,
        column_names: Option<&[Box<str>]>,
        file_path: &str,
    ) -> anyhow::Result<()> {
        // define schema
        let fields = vec![
            ("row", ParquetType::BYTE_ARRAY, ConvertedType::UTF8),
            ("column", ParquetType::BYTE_ARRAY, ConvertedType::UTF8),
            ("mean", ParquetType::FLOAT, ConvertedType::NONE),
            ("sd", ParquetType::FLOAT, ConvertedType::NONE),
            ("log_mean", ParquetType::FLOAT, ConvertedType::NONE),
            ("log_sd", ParquetType::FLOAT, ConvertedType::NONE),
        ];

        let schema = Arc::new(
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
        );

        // prepare data
        let (mean, idx) = self.posterior_mean().melt_with_indexes();

        let sd = self
            .posterior_sd()
            .melt()
            .into_iter()
            .map(|x| x.into())
            .collect::<Vec<f32>>();

        let log_mean = self
            .posterior_log_mean()
            .melt()
            .into_iter()
            .map(|x| x.into())
            .collect::<Vec<f32>>();

        let log_sd = self
            .posterior_log_sd()
            .melt()
            .into_iter()
            .map(|x| x.into())
            .collect::<Vec<f32>>();

        let rows = idx[0]
            .iter()
            .map(|&i| {
                if let Some(row_names) = row_names {
                    ByteArray::from(row_names[i].as_ref())
                } else {
                    ByteArray::from(i.to_string().as_bytes())
                }
            })
            .collect::<Vec<_>>();

        let cols = idx[1]
            .iter()
            .map(|&i| {
                if let Some(column_names) = column_names {
                    ByteArray::from(column_names[i].as_ref())
                } else {
                    ByteArray::from(i.to_string().as_bytes())
                }
            })
            .collect::<Vec<_>>();

        let mean = mean.into_iter().map(|x| x.into()).collect::<Vec<f32>>();

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

    if factor_names.len() != parameters.len() {
        return Err(anyhow::anyhow!(
            "number of the parameters and factor names should match"
        ));
    }

    // define schema
    let fields = vec![
        ("row", ParquetType::BYTE_ARRAY, ConvertedType::UTF8),
        ("column", ParquetType::BYTE_ARRAY, ConvertedType::UTF8),
        ("factor", ParquetType::BYTE_ARRAY, ConvertedType::UTF8),
        ("mean", ParquetType::FLOAT, ConvertedType::NONE),
        ("sd", ParquetType::FLOAT, ConvertedType::NONE),
        ("log_mean", ParquetType::FLOAT, ConvertedType::NONE),
        ("log_sd", ParquetType::FLOAT, ConvertedType::NONE),
    ];

    let schema = Arc::new(
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
    );

    // Write data to parquet
    let file = File::create(file_path)?;
    let zstd_level = ZstdLevel::try_new(5)?;
    let writer_properties = Arc::new(
        WriterProperties::builder()
            .set_compression(Compression::ZSTD(zstd_level))
            .build(),
    );
    let mut writer = SerializedFileWriter::new(file, schema, writer_properties)?;

    for (factor_idx, param) in parameters.iter().enumerate() {
        let (mean, idx) = param.posterior_mean().melt_with_indexes();

        let factor_name = factor_names[factor_idx].clone();

        let sd = param
            .posterior_sd()
            .melt()
            .into_iter()
            .map(|x| x.into())
            .collect::<Vec<f32>>();

        let log_mean = param
            .posterior_log_mean()
            .melt()
            .into_iter()
            .map(|x| x.into())
            .collect::<Vec<f32>>();

        let log_sd = param
            .posterior_log_sd()
            .melt()
            .into_iter()
            .map(|x| x.into())
            .collect::<Vec<f32>>();

        let factor_label = ByteArray::from(factor_name.as_bytes());

        let rows = idx[0]
            .iter()
            .map(|&i| {
                if let Some(row_names) = row_names {
                    ByteArray::from(row_names[i].as_ref())
                } else {
                    ByteArray::from(i.to_string().as_bytes())
                }
            })
            .collect::<Vec<_>>();

        let cols = idx[1]
            .iter()
            .map(|&i| {
                if let Some(column_names) = column_names {
                    ByteArray::from(column_names[i].as_ref())
                } else {
                    ByteArray::from(i.to_string().as_bytes())
                }
            })
            .collect::<Vec<_>>();

        let mean: Vec<f32> = mean.into_iter().map(|x| x.into()).collect();

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
