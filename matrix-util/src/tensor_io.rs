use crate::common_io::{Delimiter, read_lines_of_types, write_lines};
use crate::traits::IoOps;
use candle_core::{Device, Tensor};

impl IoOps for Tensor {
    type Scalar = f32;
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

        let (data, _) = read_lines_of_types::<f32>(tsv_file, delim, hdr_line)?;

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
}
