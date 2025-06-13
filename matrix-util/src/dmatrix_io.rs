use crate::common_io::{read_lines_of_types, write_lines, Delimiter};
use crate::parquet::*;
use crate::traits::*;
pub use nalgebra::{DMatrix, DVector};
pub use nalgebra_sparse::{coo::CooMatrix, csc::CscMatrix, csr::CsrMatrix};
use parquet::data_type::{ByteArrayType, DoubleType};
use rayon::prelude::*;
use std::fmt::{Debug, Display};
use std::str::FromStr;

impl<T> IoOps for DMatrix<T>
where
    T: nalgebra::RealField + FromStr + Display + Copy,
    <T as FromStr>::Err: Debug,
    f32: From<T>,
    f64: From<T>,
{
    type Scalar = T;
    type Mat = Self;

    fn read_file_delim(
        tsv_file: &str,
        delim: impl Into<Delimiter>,
        skip: Option<usize>,
    ) -> anyhow::Result<Self::Mat> {
        let hdr_line = match skip {
            Some(skip) => skip as i64,
            None => -1, // no skipping
        };

        let (data, _) = read_lines_of_types::<T>(tsv_file, delim, hdr_line)?;

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
        let mut lines = self
            .row_iter()
            .enumerate()
            .par_bridge()
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

        lines.sort_by_key(|&(i, _)| i);
        let lines = lines.into_iter().map(|(_, line)| line).collect::<Vec<_>>();
        write_lines(&lines, tsv_file)?;
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
            let data_j: Vec<f64> = self.column(j).iter().map(|&x| x.into()).collect();

            if let Some(mut column_writer) = row_group_writer.next_column()? {
                let typed_writer = column_writer.typed::<DoubleType>();
                typed_writer.write_batch(&data_j, None, None)?;
                column_writer.close()?;
            }
        }

        row_group_writer.close()?;
        writer.close()?;
        Ok(())
    }

    fn from_parquet(file_path: &str) -> anyhow::Result<(Vec<Box<str>>, Vec<Box<str>>, Self::Mat)> {
        let parquet = ParquetReader::new(file_path, None)?;

        let data: Vec<T> = parquet
            .row_major_data
            .into_iter()
            .map(|x| T::from_f64(x).unwrap())
            .collect();

        let nrows = parquet.row_names.len();
        let ncols = parquet.column_names.len();

        Ok((
            parquet.row_names,
            parquet.column_names,
            DMatrix::<T>::from_row_iterator(nrows, ncols, data.into_iter()),
        ))
    }
}
