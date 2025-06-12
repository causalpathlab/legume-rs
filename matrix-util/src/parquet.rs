use parquet::basic::Type as ParquetType;
use parquet::basic::{Compression, ConvertedType, ZstdLevel};
use parquet::data_type::ByteArray;
use parquet::file::properties::WriterProperties;
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::types::Type;
use std::fs::File;
use std::sync::Arc;

pub struct ParquetWriter {
    file: std::fs::File,
    schema: Arc<Type>,
    writer_properties: Arc<WriterProperties>,
    pub row_names: Vec<ByteArray>,
}

impl ParquetWriter {
    pub fn new(
        file_path: &str,
        shape: (usize, usize),
        names: (Option<&[Box<str>]>, Option<&[Box<str>]>),
    ) -> anyhow::Result<Self> {
        let (nrows, ncols) = shape;
        let (row_names, column_names) = names;

        let schema = build_columns_schema(ncols, column_names)?;

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

    pub fn open(&self) -> anyhow::Result<SerializedFileWriter<File>> {
        Ok(SerializedFileWriter::new(
            self.file.try_clone()?,
            self.schema.clone(),
            self.writer_properties.clone(),
        )?)
    }
}

fn build_columns_schema(
    ncols: usize,
    column_names: Option<&[Box<str>]>,
) -> anyhow::Result<Arc<Type>> {
    if let Some(column_names) = column_names {
        if column_names.len() != ncols {
            return Err(anyhow::anyhow!(
                "Column names length ({}) does not match number of columns ({})",
                column_names.len(),
                ncols
            ));
        }
    }

    let mut fields = vec![Arc::new(
        Type::primitive_type_builder("row", ParquetType::BYTE_ARRAY)
            .with_repetition(parquet::basic::Repetition::REQUIRED)
            .with_converted_type(ConvertedType::UTF8)
            .build()
            .unwrap(),
    )];

    let _column_names: Vec<Box<str>> = (0..ncols).map(|x| x.to_string().into_boxed_str()).collect();

    let column_names: &[Box<str>] = column_names.unwrap_or(&_column_names);

    for column_name in column_names {
        fields.push(Arc::new(
            Type::primitive_type_builder(column_name, ParquetType::FLOAT)
                .with_repetition(parquet::basic::Repetition::REQUIRED)
                .build()
                .unwrap(),
        ));
    }

    let schema = Arc::new(
        Type::group_type_builder("2dMatrix")
            .with_fields(fields)
            .build()?,
    );

    Ok(schema)
}
