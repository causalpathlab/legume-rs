use crate::common_io::read_lines_of_words;
use crate::traits::*;
pub use nalgebra::{DMatrix, DVector};
pub use nalgebra_sparse::{coo::CooMatrix, csc::CscMatrix, csr::CsrMatrix};
use rayon::prelude::*;
use std::fmt::Debug;
use std::str::FromStr;

impl<T> IoOps for DMatrix<T>
where
    T: nalgebra::RealField + FromStr,
    <T as FromStr>::Err: Debug,
{
    type Scalar = T;
    type Mat = Self;

    fn from_tsv(tsv_file: &str, skip: Option<usize>) -> anyhow::Result<Self::Mat> {
        let hdr_line = match skip {
            Some(skip) => skip as i64,
            None => -1, // no skipping
        };

        let (lines_of_words, _) = read_lines_of_words(tsv_file, hdr_line)?;

        if lines_of_words.len() == 0 {
            return Err(anyhow::anyhow!("No data in file"));
        }

        let nrows = lines_of_words[0].len();
        let ncols = lines_of_words.len();

        let data: Vec<T> = lines_of_words
            .par_iter()
            .map(|words| {
                words
                    .iter()
                    .map(|v| v.parse::<T>().expect(""))
                    .collect::<Vec<T>>()
            })
            .flatten()
            .collect();

        Ok(DMatrix::<T>::from_row_iterator(
            nrows,
            ncols,
            data.into_iter(),
        ))
    }
}
