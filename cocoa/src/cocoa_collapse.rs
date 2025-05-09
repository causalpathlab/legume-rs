use crate::cocoa_common::*;
use crate::cocoa_stat::*;

use asap_embed::asap_normalization::NormalizeDistance;
use matrix_util::utils::partition_by_membership;

use std::sync::{Arc, Mutex};

pub struct CocoaCollapseIn<'a> {
    pub n_genes: usize,
    pub n_samples: usize,
    pub n_topics: usize,
    pub knn: usize,
    pub n_opt_iter: Option<usize>,
    pub hyper_param: Option<(f32, f32)>,
    pub cell_topic_nk: Mat,                   // cell x cell type topic
    pub sample_to_cells: &'a Vec<Vec<usize>>, // cells within samples
    pub sample_to_exposure: &'a Vec<usize>,   // exposure assignment
}

pub trait CocoaCollapseOps {
    fn collect_stat<'a>(&self, cocoa_input: &CocoaCollapseIn<'a>) -> anyhow::Result<CocoaStat<'a>>;
}

impl CocoaCollapseOps for SparseIoVec {
    fn collect_stat<'a>(&self, cocoa_input: &CocoaCollapseIn<'a>) -> anyhow::Result<CocoaStat<'a>> {
        let n_genes = cocoa_input.n_genes;
        let n_cells = self.num_columns()?;
        let n_topics = cocoa_input.n_topics;
        let n_samples = cocoa_input.n_samples;

        assert_eq!(n_samples, self.num_batches());
        assert_eq!(n_cells, self.num_columns()?);
        assert_eq!(n_genes, self.num_rows()?);
        assert_eq!(n_cells, cocoa_input.cell_topic_nk.nrows());
        assert_eq!(n_topics, cocoa_input.cell_topic_nk.ncols());

        let exposures = cocoa_input.sample_to_exposure;

        let mut cocoa_stat = CocoaStat::new(
            n_genes,
            n_topics,
            exposures,
            cocoa_input.n_opt_iter,
            cocoa_input.hyper_param,
        );

        self.visit_column_by_samples(
            cocoa_input.sample_to_cells,
            &collect_stat_each_sample,
            cocoa_input,
            &mut cocoa_stat,
        )?;

        Ok(cocoa_stat)
    }
}

fn collect_stat_each_sample(
    this_sample: usize,
    cells: &Vec<usize>,
    data: &SparseIoVec,
    input: &CocoaCollapseIn,
    arc_stat: Arc<Mutex<&mut CocoaStat>>,
) {
    debug_assert_eq!(data.num_rows().expect("data # features"), input.n_genes);
    debug_assert_eq!(input.n_genes, data.num_batches());

    let kk = input.n_topics;
    let mut z_nk = Mat::zeros(cells.len(), kk);

    for &j in cells.iter() {
        let mut z_n = z_nk.row_mut(j);
        z_n.copy_from(&input.cell_topic_nk.row(j));
    }

    let y1_dn = data
        .read_columns_csc(cells.iter().cloned())
        .expect("read y1 cells");

    // sum_j y1[g,j] * z[j,k] * ind[j,s]
    let y1_dk = y1_dn * &z_nk;

    let mut y0_dk = Mat::zeros(y1_dk.nrows(), y1_dk.ncols());

    let my_exposure = input.sample_to_exposure[this_sample];

    let target_samples: Vec<usize> = (0..input.n_samples)
        .filter(|&s| input.sample_to_exposure[s] != my_exposure)
        .collect();

    let (y0_matched_dm, matched, distances) = data
        .read_matched_columns_csc(cells.iter().cloned(), &target_samples, input.knn, true)
        .expect("read y0 cells");

    // sum_j (sum_a y0[g,a] * w[j,a]) * z[j,k] * ind[j,s]
    let y1_to_y0 = partition_by_membership(&matched, None);
    let target_exp_sum = 2_f32.ln();
    for (y1_col, y0_cols) in y1_to_y0 {
        let weights = y0_cols
            .iter()
            .map(|&j| distances[j])
            .normalized_exp(target_exp_sum);
        let denom = weights.iter().sum::<f32>();

        let z_k = &z_nk.row(y1_col);

        y0_cols.iter().zip(weights.iter()).for_each(|(&a, &w_j)| {
            let y0_a = y0_matched_dm.get_col(a).unwrap();
            let y0_rows = y0_a.row_indices();
            let y0_vals = y0_a.values();
            y0_rows.iter().zip(y0_vals.iter()).for_each(|(&g, &y0_gj)| {
                for (k, &z_jk) in z_k.iter().enumerate() {
                    y0_dk[(g, k)] += z_jk * y0_gj * w_j / denom;
                }
            });
        });
    }

    // update global statistics
    let mut stat = arc_stat.lock().expect("lock stat");
    for k in 0..input.n_topics {
        let mut y1_k_s = stat.y1_stat(k).column_mut(this_sample);
        y1_k_s += &y1_dk.column(k);
        let mut y0_k_s = stat.y0_stat(k).column_mut(this_sample);
        y0_k_s += &y0_dk.column(k);
        stat.size_stat(k)[this_sample] += &z_nk.column(k).sum();
    }
}
