use crate::common_io::{read_lines_of_types, write_lines, Delimiter};
use crate::parquet::*;
use crate::traits::*;

use candle_core::{Device, Tensor};

impl IoOps for Tensor {
    type Scalar = f32;
    type Mat = Self;

    fn read_data(
        file_path: &str,
        delim: impl Into<Delimiter>,
        skip: Option<usize>,
        row_name_index: Option<usize>,
        column_indices: Option<&[usize]>,
        column_names: Option<&[Box<str>]>,
    ) -> anyhow::Result<MatWithNames<Self::Mat>> {
        let (rows, cols, data) = Self::read_data_vec_with_indices_names(
            file_path,
            delim,
            skip,
            row_name_index,
            column_indices,
            column_names,
        )?;

        let nrows = rows.len();
        let ncols = cols.len();
        Ok(MatWithNames {
            rows,
            cols,
            mat: Tensor::from_vec(data, (nrows, ncols), &Device::Cpu)?,
        })
    }

    fn read_file_delim(
        tsv_file: &str,
        delim: impl Into<Delimiter>,
        skip: Option<usize>,
    ) -> anyhow::Result<Self::Mat> {
        let hdr_line = match skip {
            Some(skip) => skip as i64,
            None => -1, // no skipping
        };

        let data = read_lines_of_types::<f32>(tsv_file, delim, hdr_line)?.lines;

        if data.is_empty() {
            return Err(anyhow::anyhow!("No data in file"));
        }

        let ncols = data[0].len();
        let nrows = data.len();
        let data = data.into_iter().flatten().collect::<Vec<_>>();

        Ok(Tensor::from_vec(data, (nrows, ncols), &Device::Cpu)?)
    }

    fn write_file_delim(&self, file: &str, delim: &str) -> anyhow::Result<()> {
        let dims = self.dims();

        if dims.len() != 2 {
            return Err(anyhow::anyhow!("Expected 2 dimensions, got {}", dims.len()));
        }

        let lines: Vec<Box<str>> = (0..dims[0])
            .map(|i| {
                let row = self.narrow(0, i, 1).expect("failed to narrow in");
                let flatten_row = row.flatten_to(1).expect("flatten");
                let row_vec = flatten_row.to_vec1::<f32>().expect("to_vec1");
                row_vec
                    .iter()
                    .map(|&x| format!("{}", x))
                    .collect::<Vec<_>>()
                    .join(delim)
                    .into_boxed_str()
            })
            .collect();

        write_lines(&lines, file)?;

        Ok(())
    }

    fn to_parquet(
        &self,
        row_names: Option<&[Box<str>]>,
        column_names: Option<&[Box<str>]>,
        file_path: &str,
    ) -> anyhow::Result<()> {
        let dims = self.dims();

        if dims.len() != 2 {
            return Err(anyhow::anyhow!("expected 2 dimensions, got {}", dims.len()));
        }

        let (nrows, ncols) = (dims[0], dims[1]);

        let writer =
            ParquetWriter::new(file_path, (nrows, ncols), (row_names, column_names), None)?;
        let row_names = writer.row_names_vec();

        if row_names.len() != nrows {
            return Err(anyhow::anyhow!("row names don't match"));
        }

        let mut writer = writer.get_writer()?;
        let mut row_group_writer = writer.next_row_group()?;
        parquet_add_bytearray(&mut row_group_writer, &row_names)?;

        let tensor = self.to_dtype(candle_core::DType::F32)?;

        for j in 0..ncols {
            let data_j = tensor.narrow(1, j, 1)?.flatten_all()?.to_vec1::<f32>()?;
            parquet_add_numeric_column(&mut row_group_writer, &data_j)?;
        }
        row_group_writer.close()?;
        writer.close()?;
        Ok(())
    }

    fn from_parquet_with_indices_names(
        file_path: &str,
        row_name_index: Option<usize>,
        column_indices: Option<&[usize]>,
        column_names: Option<&[Box<str>]>,
    ) -> anyhow::Result<MatWithNames<Self>> {
        let parquet = ParquetReader::new(file_path, row_name_index, column_indices, column_names)?;

        let nrows = parquet.row_names.len();
        let ncols = parquet.column_names.len();

        let data: Vec<f32> = parquet
            .row_major_data
            .into_iter()
            .map(|x| x as f32)
            .collect();

        Ok(MatWithNames {
            rows: parquet.row_names,
            cols: parquet.column_names,
            mat: Tensor::from_vec(data, (nrows, ncols), &Device::Cpu)?,
        })
    }
}
