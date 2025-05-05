use asap_embed::asap_collapse_data::CollapsingStat;

use crate::cocoa_common::*;
use matrix_param::dmatrix_gamma::*;
use matrix_param::traits::Inference;
use matrix_param::traits::*;

use std::sync::{Arc, Mutex};

struct CocoaStat {
    mu: Vec<GammaMatrix>,
    delta: Vec<GammaMatrix>,
}

fn count_stat_each_sample(
    sample: usize,
    cells: &Vec<usize>,
    data_vec: &SparseIoVec,
    z_nk: &Mat,
    arc_stat: Arc<Mutex<&mut CocoaStat>>,
) {
    // y0[g,j] * z[j,k] * ind[j,s]
    // y1[g,j] * z[j,k] * ind[j,s]

    let n_cells = cells.len();
    let n_topics = z_nk.ncols();

    let k = 0;

    let z_n = Mat::from_vec(
        n_cells,
        1,
        cells.iter().cloned().map(|j| z_nk[(j, k)]).collect(),
    );

    let y1_dn = data_vec
        .read_columns_csc(cells.iter().cloned())
        .expect("read y1 cells");

    
}

pub trait CocoaCollapseOps {
    fn collect_stat(&self, latent: &Mat) -> anyhow::Result<()>;
}

impl CocoaCollapseOps for SparseIoVec {
    fn collect_stat(&self, logits_nk: &Mat) -> anyhow::Result<()> {
        let n_genes = self.num_rows()?;
        let n_cells = logits_nk.nrows();
        let n_topics = logits_nk.ncols();
        let n_samples = self.num_batches(); // batch == sample

        if n_cells != self.num_columns()? {
            return Err(anyhow::anyhow!(
                "{} samples in data vs. {} in latent membership",
                self.num_columns()?,
                n_cells
            ));
        }

        let mut stat = CocoaStat {
            mu: vec![],
            delta: vec![],
        };

        let (a0, b0) = (1., 1.);

        for _k in 0..n_topics {
            stat.mu.push(GammaMatrix::new((n_genes, n_samples), a0, b0));
            stat.delta
                .push(GammaMatrix::new((n_genes, n_samples), a0, b0));
        }

        let mut z_nk = logits_nk.map(|x| x.exp());
        let denom_n = z_nk.column_sum().map(|x| x.max(0.1));
        z_nk.column_iter_mut().for_each(|mut z| {
            z.component_div_assign(&denom_n);
        });

        let count = |sample: usize,
                     cells: &Vec<usize>,
                     z_nk: &Mat,
                     arc_stat: Arc<Mutex<&mut CocoaStat>>| {
            //
        };

        Ok(())
    }
}
