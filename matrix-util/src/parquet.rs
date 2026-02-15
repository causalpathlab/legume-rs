use parquet::basic::Type as ParquetType;
use parquet::basic::{Compression, ConvertedType, ZstdLevel};
use parquet::data_type::ByteArray;
use parquet::data_type::{ByteArrayType, DoubleType, FloatType, Int32Type, Int64Type};
use parquet::file::properties::WriterProperties;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::file::writer::{SerializedFileWriter, SerializedRowGroupWriter};
use parquet::record::RowAccessor;
use parquet::schema::types::Type as SchemaType;
use std::any::TypeId;
use std::collections::HashSet;
use std::fs::File;
use std::sync::Arc;

/// get field names by peeking into `file_path`
pub fn peek_parquet_field_names(file_path: &str) -> anyhow::Result<Vec<Box<str>>> {
    let file = File::open(file_path)?;
    let reader = SerializedFileReader::new(file)?;
    let metadata = reader.metadata();
    let fields = metadata.file_metadata().schema().get_fields();

    Ok(fields
        .iter()
        .map(|f| f.name().to_string().into_boxed_str())
        .collect())
}

pub struct ParquetReader {
    pub row_major_data: Vec<f64>,
    pub row_names: Vec<Box<str>>,
    pub column_names: Vec<Box<str>>,
}

impl ParquetReader {
    /// Create a new parquet reader for a matrix with row and column
    /// names.
    ///
    /// * `row_name_index`: if `None`, no column is treated as row names
    ///   (row names will be generated as "0", "1", "2", ...)
    ///
    /// * `select_column_index`: if `None`, use all the other columns
    ///
    /// * `select_column_names`: if `None`, use all the other columns
    pub fn new(
        file_path: &str,
        row_name_index: Option<usize>,
        select_columns_index: Option<&[usize]>,
        select_columns_names: Option<&[Box<str>]>,
    ) -> anyhow::Result<Self> {
        let file = File::open(file_path)?;
        let reader = SerializedFileReader::new(file)?;
        let metadata = reader.metadata();
        let nrows = metadata.file_metadata().num_rows() as usize;
        let fields = metadata.file_metadata().schema().get_fields();

        let select_columns: HashSet<usize> = {
            let mut indices = HashSet::new();

            // Add indices from `select_columns_index` if provided
            if let Some(select) = select_columns_index {
                indices.extend(select.iter().copied());
            }

            // Add indices from `select_columns_names` if provided
            if let Some(names) = select_columns_names {
                indices.extend(fields.iter().enumerate().filter_map(|(j, f)| {
                    if names.iter().any(|name| name.as_ref() == f.name()) {
                        Some(j)
                    } else {
                        None
                    }
                }));
            }

            // Default to all columns if neither is provided
            if indices.is_empty() {
                (0..fields.len()).collect()
            } else {
                indices
            }
        };

        // Get the type of the row name column for later use (if specified)
        let row_name_type = row_name_index.map(|idx| fields[idx].get_physical_type());

        let select_indices = fields
            .iter()
            .enumerate()
            .filter_map(|(j, f)| {
                // Exclude row_name_index column if specified
                let is_row_name_col = row_name_index == Some(j);
                if select_columns.contains(&j) && !is_row_name_col {
                    let tt = f.get_physical_type();
                    match tt {
                        parquet::basic::Type::FLOAT
                        | parquet::basic::Type::DOUBLE
                        | parquet::basic::Type::INT32
                        | parquet::basic::Type::INT64 => Some((tt, j)),
                        _ => None,
                    }
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        if select_indices.is_empty() {
            return Err(anyhow::anyhow!("no available columns"));
        }

        let ncols = select_indices.len();

        let column_names: Vec<Box<str>> = select_indices
            .iter()
            .map(|&(_, j)| fields[j].name().to_string().into_boxed_str())
            .collect();

        let row_iter = reader.get_row_iter(None)?;
        let mut row_names: Vec<Box<str>> = Vec::with_capacity(nrows);
        let mut row_major_data: Vec<f64> = Vec::with_capacity(nrows * ncols);

        for (row_counter, record) in row_iter.enumerate() {
            let row = record?;
            // Handle different column types for row names
            let row_name: Box<str> = match (row_name_index, row_name_type) {
                (Some(idx), Some(parquet::basic::Type::BYTE_ARRAY)) => {
                    row.get_string(idx)?.clone().into_boxed_str()
                }
                (Some(idx), Some(parquet::basic::Type::DOUBLE)) => {
                    row.get_double(idx)?.to_string().into_boxed_str()
                }
                (Some(idx), Some(parquet::basic::Type::FLOAT)) => {
                    row.get_float(idx)?.to_string().into_boxed_str()
                }
                (Some(idx), Some(parquet::basic::Type::INT32)) => {
                    row.get_int(idx)?.to_string().into_boxed_str()
                }
                (Some(idx), Some(parquet::basic::Type::INT64)) => {
                    row.get_long(idx)?.to_string().into_boxed_str()
                }
                (Some(idx), Some(_)) => {
                    // Fallback: try string, or use row index
                    row.get_string(idx)
                        .map(|s| s.clone().into_boxed_str())
                        .unwrap_or_else(|_| row_counter.to_string().into_boxed_str())
                }
                // No row name column specified, generate numeric names
                (None, _) | (_, None) => row_counter.to_string().into_boxed_str(),
            };
            row_names.push(row_name);

            let numbers: anyhow::Result<Vec<f64>> =
                select_indices
                    .iter()
                    .try_fold(Vec::with_capacity(ncols), |mut acc, &(tt, j)| {
                        let x = match tt {
                            parquet::basic::Type::DOUBLE => row.get_double(j)?,
                            parquet::basic::Type::FLOAT => row.get_float(j)? as f64,
                            parquet::basic::Type::INT32 | parquet::basic::Type::INT64 => {
                                row.get_int(j)? as f64
                            }
                            _ => {
                                unimplemented!("we just support integer and float/double for now")
                            }
                        };
                        acc.push(x);
                        Ok(acc)
                    });

            row_major_data.extend(numbers?);
        }

        Ok(Self {
            row_major_data,
            row_names,
            column_names,
        })
    }
}

pub struct ParquetWriter {
    file: std::fs::File,
    schema: Arc<SchemaType>,
    writer_properties: Arc<WriterProperties>,
    row_names: Vec<ByteArray>,
}

impl ParquetWriter {
    /// Create a new parquet writer for a matrix with row and column
    /// names.
    ///
    /// * `file_path`: output file path
    ///
    /// * `shape`: number of rows and columns
    ///
    /// * `names`: for row and column names, respectively; if `None`, just add `[0, n)` numbers.
    ///
    #[allow(clippy::type_complexity)]
    pub fn new(
        file_path: &str,
        shape: (usize, usize),
        names: (Option<&[Box<str>]>, Option<&[Box<str>]>),
        column_types: Option<&[ParquetType]>,
        row_column_name: Option<&str>,
    ) -> anyhow::Result<Self> {
        let (nrows, ncols) = shape;
        let (row_names, column_names) = names;

        let schema = build_columns_schema(ncols, column_names, column_types, row_column_name)?;

        let file = std::fs::File::create(file_path)?;

        let zstd_level = ZstdLevel::try_new(5)?;
        let writer_properties = std::sync::Arc::new(
            WriterProperties::builder()
                .set_compression(Compression::ZSTD(zstd_level))
                .build(),
        );

        let row_names: Vec<ByteArray> = match row_names {
            Some(row_names) => row_names
                .iter()
                .map(|r| ByteArray::from(r.as_ref()))
                .collect(),
            None => (0..nrows)
                .map(|i| ByteArray::from(i.to_string().as_bytes()))
                .collect(),
        };

        Ok(Self {
            file,
            schema,
            writer_properties,
            row_names,
        })
    }

    pub fn row_names_vec(&self) -> &Vec<ByteArray> {
        &self.row_names
    }

    pub fn get_writer(&self) -> anyhow::Result<SerializedFileWriter<File>> {
        Ok(SerializedFileWriter::new(
            self.file.try_clone()?,
            self.schema.clone(),
            self.writer_properties.clone(),
        )?)
    }
}

/// write down a string vector of `Box<str>` to `row_group_writer` by creating a
/// new `column_writer`
pub fn parquet_add_bytearray<'a>(
    row_group_writer: &mut SerializedRowGroupWriter<'a, File>,
    data: &[ByteArray],
) -> anyhow::Result<()> {
    if let Some(mut column_writer) = row_group_writer.next_column()? {
        let typed_writer = column_writer.typed::<ByteArrayType>();
        typed_writer.write_batch(data, None, None)?;
        column_writer.close()?;
    }

    Ok(())
}

/// write down a string vector of `Box<str>` to `row_group_writer` by creating a
/// new `column_writer`
pub fn parquet_add_string_column<'a>(
    row_group_writer: &mut SerializedRowGroupWriter<'a, File>,
    data: &[Box<str>],
) -> anyhow::Result<()> {
    let data_bytearray = data
        .iter()
        .map(|x| ByteArray::from(x.as_ref()))
        .collect::<Vec<_>>();

    parquet_add_bytearray(row_group_writer, &data_bytearray)?;
    Ok(())
}

/// write down a numeric vector to `row_group_writer` by creating a
/// new `column_writer`
pub fn parquet_add_numeric_column<'a, T: 'static + num_traits::ToPrimitive>(
    row_group_writer: &mut SerializedRowGroupWriter<'a, File>,
    data: &[T],
) -> anyhow::Result<()> {
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        if let Some(mut column_writer) = row_group_writer.next_column()? {
            let typed_writer = column_writer.typed::<DoubleType>();
            let data: Vec<f64> = data
                .iter()
                .map(|x| x.to_f64().expect("Failed to convert to f64"))
                .collect();
            typed_writer.write_batch(&data, None, None)?;
            column_writer.close()?;
        }
    } else if TypeId::of::<T>() == TypeId::of::<f32>() {
        if let Some(mut column_writer) = row_group_writer.next_column()? {
            let typed_writer = column_writer.typed::<FloatType>();
            let data: Vec<f32> = data
                .iter()
                .map(|x| x.to_f32().expect("Failed to convert to f32"))
                .collect();
            typed_writer.write_batch(&data, None, None)?;
            column_writer.close()?;
        }
    } else if TypeId::of::<T>() == TypeId::of::<i32>()
        || TypeId::of::<T>() == TypeId::of::<u32>()
        || TypeId::of::<T>() == TypeId::of::<usize>()
    {
        if let Some(mut column_writer) = row_group_writer.next_column()? {
            let typed_writer = column_writer.typed::<Int32Type>();
            let data: Vec<i32> = data
                .iter()
                .map(|x| x.to_i32().expect("Failed to convert to i32"))
                .collect();
            typed_writer.write_batch(&data, None, None)?;
            column_writer.close()?;
        }
    } else if TypeId::of::<T>() == TypeId::of::<i64>() || TypeId::of::<T>() == TypeId::of::<u64>() {
        if let Some(mut column_writer) = row_group_writer.next_column()? {
            let typed_writer = column_writer.typed::<Int64Type>();
            let data: Vec<i64> = data
                .iter()
                .map(|x| x.to_i64().expect("Failed to convert to i64"))
                .collect();
            typed_writer.write_batch(&data, None, None)?;
            column_writer.close()?;
        }
    } else {
        return Err(anyhow::anyhow!("Unsupported data type"));
    }

    Ok(())
}

fn build_columns_schema(
    ncols: usize,
    column_names: Option<&[Box<str>]>,
    column_types: Option<&[ParquetType]>,
    row_column_name: Option<&str>,
) -> anyhow::Result<Arc<SchemaType>> {
    if let Some(column_names) = column_names {
        if column_names.len() != ncols {
            return Err(anyhow::anyhow!(
                "Column names length ({}) does not match number of columns ({})",
                column_names.len(),
                ncols
            ));
        }
    }

    let row_col_name = row_column_name.unwrap_or("rowname");
    let mut fields = vec![Arc::new(
        SchemaType::primitive_type_builder(row_col_name, ParquetType::BYTE_ARRAY)
            .with_repetition(parquet::basic::Repetition::REQUIRED)
            .with_converted_type(ConvertedType::UTF8)
            .build()
            .unwrap(),
    )];

    let _column_names: Vec<Box<str>> = (0..ncols).map(|x| x.to_string().into_boxed_str()).collect();
    let _column_types = (0..ncols).map(|_x| ParquetType::FLOAT).collect::<Vec<_>>();

    let column_names: &[Box<str>] = column_names.unwrap_or(&_column_names);
    let column_types: &[ParquetType] = column_types.unwrap_or(&_column_types);

    for (column_name, &column_type) in column_names.iter().zip(column_types) {
        if column_type == ParquetType::BYTE_ARRAY {
            fields.push(Arc::new(
                SchemaType::primitive_type_builder(column_name, column_type)
                    .with_repetition(parquet::basic::Repetition::REQUIRED)
                    .with_converted_type(ConvertedType::UTF8)
                    .build()
                    .unwrap(),
            ));
        } else {
            fields.push(Arc::new(
                SchemaType::primitive_type_builder(column_name, column_type)
                    .with_repetition(parquet::basic::Repetition::REQUIRED)
                    .build()
                    .unwrap(),
            ));
        }
    }

    let schema = Arc::new(
        SchemaType::group_type_builder("2dMatrix")
            .with_fields(fields)
            .build()?,
    );

    Ok(schema)
}
