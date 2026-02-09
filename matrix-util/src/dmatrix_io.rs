use crate::common_io::{read_lines_of_types, write_lines, Delimiter};
use crate::parquet::*;
use crate::traits::*;
pub use nalgebra::{DMatrix, DVector};
pub use nalgebra_sparse::{coo::CooMatrix, csc::CscMatrix, csr::CsrMatrix};
use num_traits::*;
use parquet::basic::Type as ParquetType;
use std::any::TypeId;

use std::fmt::{Debug, Display};
use std::str::FromStr;

use num_traits::ToPrimitive;

impl<T> IoOps for DMatrix<T>
where
    T: PartialOrd
        + FromPrimitive
        + ToPrimitive
        + nalgebra::Scalar
        + Send
        + FromStr
        + Display
        + Copy,
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
            mat: DMatrix::<T>::from_row_iterator(nrows, ncols, data),
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

        Ok(DMatrix::<T>::from_row_iterator(nrows, ncols, data))
    }

    fn write_file_delim(&self, tsv_file: &str, delim: &str) -> anyhow::Result<()> {
        // par_iter() or par_bridge() will
        // mess up the order of the rows
        let lines = self
            .row_iter()
            .enumerate()
            .map(|(i, row)| {
                let line = row
                    .iter()
                    .map(|x| format!("{}", *x))
                    .collect::<Vec<String>>()
                    .join(delim)
                    .into_boxed_str();
                (i, line)
            })
            .collect::<Vec<_>>();

        let lines = lines.into_iter().map(|(_, line)| line).collect::<Vec<_>>();
        write_lines(&lines, tsv_file)?;
        Ok(())
    }

    fn to_parquet_with_names(
        &self,
        file_path: &str,
        row_names: (Option<&[Box<str>]>, Option<&str>),
        column_names: Option<&[Box<str>]>,
    ) -> anyhow::Result<()> {
        let (nrows, ncols) = (self.nrows(), self.ncols());

        let parquet_type = if TypeId::of::<T>() == TypeId::of::<f64>() {
            ParquetType::DOUBLE
        } else if TypeId::of::<T>() == TypeId::of::<f32>() {
            ParquetType::FLOAT
        } else if TypeId::of::<T>() == TypeId::of::<i32>()
            || TypeId::of::<T>() == TypeId::of::<u32>()
            || TypeId::of::<T>() == TypeId::of::<usize>()
        {
            ParquetType::INT32
        } else if TypeId::of::<T>() == TypeId::of::<i64>()
            || TypeId::of::<T>() == TypeId::of::<u64>()
        {
            ParquetType::INT64
        } else {
            return Err(anyhow::anyhow!("Unsupported data type"));
        };

        let column_types = vec![parquet_type; ncols];

        let (row_names_slice, row_column_name) = row_names;

        let writer = ParquetWriter::new(
            file_path,
            (nrows, ncols),
            (row_names_slice, column_names),
            Some(&column_types),
            row_column_name,
        )?;
        let row_names = writer.row_names_vec();

        if row_names.len() != nrows {
            return Err(anyhow::anyhow!("row names don't match"));
        }

        let mut writer = writer.get_writer()?;
        let mut row_group_writer = writer.next_row_group()?;
        parquet_add_bytearray(&mut row_group_writer, &row_names)?;

        for j in 0..ncols {
            parquet_add_numeric_column(
                &mut row_group_writer,
                &self.column(j).iter().copied().collect::<Vec<_>>(),
            )?;
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

        let data: Vec<T> = parquet
            .row_major_data
            .into_iter()
            .map(|x| T::from_f64(x).unwrap())
            .collect();

        let nrows = parquet.row_names.len();
        let ncols = parquet.column_names.len();

        Ok(MatWithNames {
            rows: parquet.row_names,
            cols: parquet.column_names,
            mat: DMatrix::<T>::from_row_iterator(nrows, ncols, data),
        })
    }
}
