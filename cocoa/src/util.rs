use crate::common::*;
use matrix_param::{dmatrix_gamma::GammaMatrix, traits::Inference};
use matrix_util::common_io::write_types;

pub trait CocoaOut {
    fn to_summary_stat_tsv(
        &self,
        row_names: Vec<Box<str>>,
        column_names: Vec<Box<str>>,
        outfile: &str,
    );
}

impl CocoaOut for GammaMatrix {
    fn to_summary_stat_tsv(
        &self,
        row_names: Vec<Box<str>>,
        column_names: Vec<Box<str>>,
        outfile: &str,
    ) {
        self.posterior_mean();
        self.posterior_sd();
        self.posterior_log_mean();
        self.posterior_log_sd();

        // type, gene, col, mu,
    }
}
