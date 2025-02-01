use crate::common_io::{read_lines_of_types, write_lines};
use crate::traits::*;
pub use nalgebra::{DMatrix, DVector};
pub use nalgebra_sparse::{coo::CooMatrix, csc::CscMatrix, csr::CsrMatrix};
use rayon::prelude::*;
use std::fmt::{Debug, Display};
use std::str::FromStr;

impl<T> IoOps for DMatrix<T>
where
    T: nalgebra::RealField + FromStr + Display + Copy,
    <T as FromStr>::Err: Debug,
{
    type Scalar = T;
    type Mat = Self;

    fn from_tsv(tsv_file: &str, skip: Option<usize>) -> anyhow::Result<Self::Mat> {
        let hdr_line = match skip {
            Some(skip) => skip as i64,
            None => -1, // no skipping
        };

        let (data, _) = read_lines_of_types::<T>(tsv_file, hdr_line)?;

        if data.len() == 0 {
            return Err(anyhow::anyhow!("No data in file"));
        }

        let nrows = data[0].len();
        let ncols = data.len();
        let data = data.into_iter().flatten().collect::<Vec<_>>();

        Ok(DMatrix::<T>::from_row_iterator(
            nrows,
            ncols,
            data.into_iter(),
        ))
    }
    fn to_tsv(&self, tsv_file: &str) -> anyhow::Result<()> {
        // par_iter() or par_bridge() will
        // mess up the order of the rows
        let mut lines = self
            .row_iter()
            .enumerate()
            .par_bridge()
            .map(|(i, row)| {
                let line = row
                    .iter()
                    .enumerate()
                    .map(|(_, x)| format!("{}", *x))
                    .collect::<Vec<String>>()
                    .join("\t")
                    .into_boxed_str();
                (i, line)
            })
            .collect::<Vec<_>>();

        lines.sort_by_key(|&(i, _)| i);
        let lines = lines.into_iter().map(|(_, line)| line).collect::<Vec<_>>();
        write_lines(&lines, &tsv_file)?;
        Ok(())
    }
}
