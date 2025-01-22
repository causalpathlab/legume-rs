use crate::common_io::{read_lines_of_words, write_lines};
use crate::traits::IoOps;
use ndarray::prelude::*;
use rayon::prelude::*;
use std::fmt::{Debug, Display};
use std::str::FromStr;

impl<T> IoOps for Array2<T>
where
    T: FromStr + Send + Display,
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

        let mut data = lines_of_words
            .iter()
            .enumerate()
            .par_bridge()
            .map(|(i, words)| {
                (
                    i,
                    words
                        .iter()
                        .map(|v| v.parse::<T>().expect(""))
                        .collect::<Vec<T>>(),
                )
            })
            .collect::<Vec<_>>();

        data.sort_by_key(|&(i, _)| i);

        let data = data
            .into_iter()
            .map(|(_, x)| x)
            .flatten()
            .collect::<Vec<_>>();

        Ok(Array2::from_shape_vec((nrows, ncols), data)?)
    }

    fn to_tsv(&self, tsv_file: &str) -> anyhow::Result<()> {
        let lines: Vec<Box<str>> = self
            .rows()
            .into_iter()
            .map(|row| {
                row.iter()
                    .map(|x| format!("{}", *x))
                    .collect::<Vec<String>>()
                    .join("\t")
                    .into_boxed_str()
            })
            .collect();
        write_lines(&lines, &tsv_file)?;
        Ok(())
    }
}
