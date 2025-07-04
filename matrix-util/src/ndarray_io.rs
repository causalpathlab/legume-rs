use crate::common_io::{read_lines_of_types, write_lines, Delimiter};
use crate::parquet::*;
use crate::traits::IoOps;
use ndarray::prelude::*;
use parquet::data_type::{ByteArrayType, DoubleType};
use std::fmt::{Debug, Display};
use std::str::FromStr;

impl<T> IoOps for Array2<T>
where
    T: FromStr + Send + Display + Clone + Into<f64> + num_traits::FromPrimitive,
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
    ) -> anyhow::Result<(Vec<Box<str>>, Vec<Box<str>>, Self::Mat)> {
        let (rows, cols, data) = Self::read_names_and_data_with_indices_names(
            file_path,
            delim,
            skip,
            row_name_index,
            column_indices,
            column_names,
        )?;

        let nrows = rows.len();
        let ncols = cols.len();
        Ok((rows, cols, Array2::from_shape_vec((nrows, ncols), data)?))
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
    fn to_parquet(
        &self,
        row_names: Option<&[Box<str>]>,
        column_names: Option<&[Box<str>]>,
        file_path: &str,
    ) -> anyhow::Result<()> {
        let (nrows, ncols) = (self.nrows(), self.ncols());

        let writer = ParquetWriter::new(file_path, (nrows, ncols), (row_names, column_names))?;
        let row_names = writer.row_names_vec();

        if row_names.len() != nrows {
            return Err(anyhow::anyhow!("row names don't match"));
        }

        let mut writer = writer.open()?;
        let mut row_group_writer = writer.next_row_group()?;

        if let Some(mut column_writer) = row_group_writer.next_column()? {
            let typed_writer = column_writer.typed::<ByteArrayType>();
            typed_writer.write_batch(&row_names, None, None)?;
            column_writer.close()?;
        }

        for j in 0..ncols {
            let data_j = self
                .column(j)
                .to_vec()
                .into_iter()
                .map(|x| x.into())
                .collect::<Vec<f64>>();
            if let Some(mut column_writer) = row_group_writer.next_column()? {
                let typed_writer = column_writer.typed::<DoubleType>();
                typed_writer.write_batch(data_j.as_slice(), None, None)?;
                column_writer.close()?;
            }
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
    ) -> anyhow::Result<(Vec<Box<str>>, Vec<Box<str>>, Self::Mat)> {
        let parquet = ParquetReader::new(file_path, row_name_index, column_indices, column_names)?;

        let nrows = parquet.row_names.len();
        let ncols = parquet.column_names.len();

        let data: Vec<T> = parquet
            .row_major_data
            .into_iter()
            .map(|x| T::from_f64(x).unwrap())
            .collect();

        Ok((
            parquet.row_names,
            parquet.column_names,
            Array2::from_shape_vec((nrows, ncols), data)?,
        ))
    }
}
