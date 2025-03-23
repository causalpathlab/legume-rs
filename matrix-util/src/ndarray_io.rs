use crate::common_io::{read_lines_of_types, write_lines};
use crate::traits::IoOps;
use ndarray::prelude::*;
use std::fmt::{Debug, Display};
use std::str::FromStr;

impl<T> IoOps for Array2<T>
where
    T: FromStr + Send + Display,
    <T as FromStr>::Err: Debug,
{
    type Scalar = T;
    type Mat = Self;

    fn read_file_delim(
        tsv_file: &str,
        delim: &str,
        skip: Option<usize>,
    ) -> anyhow::Result<Self::Mat> {
        let hdr_line = match skip {
            Some(skip) => skip as i64,
            None => -1, // no skipping
        };

        let (data, _) = read_lines_of_types::<T>(tsv_file, delim, hdr_line)?;

        if data.len() == 0 {
            return Err(anyhow::anyhow!("No data in file"));
        }

        let nrows = data[0].len();
        let ncols = data.len();
        let data = data.into_iter().flatten().collect::<Vec<_>>();

        Ok(Array2::from_shape_vec((nrows, ncols), data)?)
    }

    fn write_file_delim(&self, tsv_file: &str, delim: &str) -> anyhow::Result<()> {
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
        write_lines(&lines, &tsv_file)?;
        Ok(())
    }
}
