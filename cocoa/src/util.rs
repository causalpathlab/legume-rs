use matrix_param::{
    dmatrix_gamma::GammaMatrix,
    traits::{Inference, TwoStatParam},
};
use matrix_util::common_io::write_lines;

pub trait CocoaOut {
    fn to_summary_stat_tsv(
        &self,
        row_names: Vec<Box<str>>,
        column_names: Vec<Box<str>>,
        outfile: &str,
    ) -> anyhow::Result<()>;
}

impl CocoaOut for GammaMatrix {
    fn to_summary_stat_tsv(
        &self,
        row_names: Vec<Box<str>>,
        column_names: Vec<Box<str>>,
        outfile: &str,
    ) -> anyhow::Result<()> {
        let mean = self.posterior_mean();
        let sd = self.posterior_sd();
        let log_mean = self.posterior_log_mean();
        let log_sd = self.posterior_log_sd();

        assert_eq!(self.ncols(), column_names.len());
        assert_eq!(self.nrows(), row_names.len());

        let hdr = "row\tcol\tmean\tsd\tlog.mean\tlog.sd"
            .to_string()
            .into_boxed_str();

        let mut ret = Vec::with_capacity(column_names.len() * row_names.len() + 1);
        ret.push(hdr);
        for (c, col) in column_names.iter().enumerate() {
            for (r, row) in row_names.iter().enumerate() {
                let line = format!(
                    "{}\t{}\t{}\t{}\t{}\t{}",
                    row,
                    col,
                    mean[(r, c)],
                    sd[(r, c)],
                    log_mean[(r, c)],
                    log_sd[(r, c)]
                )
                .into_boxed_str();
                ret.push(line);
            }
        }
        write_lines(&ret, outfile)
    }
}
