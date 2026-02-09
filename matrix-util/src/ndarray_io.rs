use crate::common_io::{read_lines_of_types, write_lines, Delimiter};
use crate::parquet::*;
use crate::traits::*;
use ndarray::prelude::*;
use std::fmt::{Debug, Display};
use std::str::FromStr;

impl<T> IoOps for Array2<T>
where
    T: FromStr
        + Send
        + Display
        + Clone
        + Into<f64>
        + num_traits::FromPrimitive
        + num_traits::ToPrimitive
        + 'static,
    <T as FromStr>::Err: Debug,
{
    type Scalar = T;
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
            mat: Array2::from_shape_vec((nrows, ncols), data)?,
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

        let data = read_lines_of_types::<T>(tsv_file, delim, hdr_line)?.lines;

        if data.is_empty() {
            return Err(anyhow::anyhow!("No data in file"));
        }

        let ncols = data[0].len();
        let nrows = data.len();
        let data = data.into_iter().flatten().collect::<Vec<_>>();

        Ok(Array2::from_shape_vec((nrows, ncols), data)?)
    }

    fn write_file_delim(&self, out_file: &str, delim: &str) -> anyhow::Result<()> {
        let lines: Vec<Box<str>> = self
            .rows()
            .into_iter()
            .map(|row| {
                row.iter()
                    .map(|x| format!("{}", *x))
                    .collect::<Vec<String>>()
                    .join(delim)
                    .into_boxed_str()
            })
            .collect();
        write_lines(&lines, out_file)?;
        Ok(())
    }
    fn to_parquet_with_names(
        &self,
        file_path: &str,
        row_names: (Option<&[Box<str>]>, Option<&str>),
        column_names: Option<&[Box<str>]>,
    ) -> anyhow::Result<()> {
        let (nrows, ncols) = (self.nrows(), self.ncols());

        let (row_names_slice, row_column_name) = row_names;

        let writer =
            ParquetWriter::new(file_path, (nrows, ncols), (row_names_slice, column_names), None, row_column_name)?;
        let row_names = writer.row_names_vec();

        if row_names.len() != nrows {
            return Err(anyhow::anyhow!("row names don't match"));
        }

        let mut writer = writer.get_writer()?;
        let mut row_group_writer = writer.next_row_group()?;
        parquet_add_bytearray(&mut row_group_writer, &row_names)?;
        for j in 0..ncols {
            parquet_add_numeric_column(&mut row_group_writer, &self.column(j).to_vec())?;
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

        let data: Vec<T> = parquet
            .row_major_data
            .into_iter()
            .map(|x| T::from_f64(x).unwrap())
            .collect();

        Ok(MatWithNames {
            rows: parquet.row_names,
            cols: parquet.column_names,
            mat: Array2::from_shape_vec((nrows, ncols), data)?,
        })
    }
}
