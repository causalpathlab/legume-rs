use crate::common_io::{read_lines_of_types, write_lines, Delimiter};
use crate::parquet::*;
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

    fn to_parquet(
        &self,
        row_names: Option<&[Box<str>]>,
        column_names: Option<&[Box<str>]>,
        file_path: &str,
    ) -> anyhow::Result<()> {
        use parquet::data_type::{ByteArrayType, FloatType};

        let dims = self.dims();

        if dims.len() != 2 {
            return Err(anyhow::anyhow!("expected 2 dimensions, got {}", dims.len()));
        }

        let (nrows, ncols) = (dims[0], dims[1]);

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
            let data_j = self.narrow(1, j, 1)?.flatten_all()?.to_vec1::<f32>()?;
            if let Some(mut column_writer) = row_group_writer.next_column()? {
                let typed_writer = column_writer.typed::<FloatType>();
                typed_writer.write_batch(&data_j, None, None)?;
                column_writer.close()?;
            }
        }
        row_group_writer.close()?;
        writer.close()?;
        Ok(())
    }
}
