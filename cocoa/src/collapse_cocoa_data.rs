use crate::common::*;
use crate::stat::*;
use matrix_util::utils::partition_by_membership;
use std::sync::{Arc, Mutex};

pub struct CocoaCollapseIn<'a> {
    pub n_genes: usize,
    pub n_topics: usize,
    pub knn: usize,
    pub n_opt_iter: Option<usize>,
    pub hyper_param: Option<(f32, f32)>,
    pub cell_topic_nk: Mat,                  // cell x cell type topic
    pub exposure_assignment: &'a Vec<usize>, // exposure assignment
}

pub trait CocoaCollapseOps {
    fn collect_cocoa_stat<'a>(
        &self,
        cocoa_input: &CocoaCollapseIn<'a>,
    ) -> anyhow::Result<CocoaStat>;
}

impl CocoaCollapseOps for SparseIoVec {
    fn collect_cocoa_stat<'a>(
        &self,
        cocoa_input: &CocoaCollapseIn<'a>,
    ) -> anyhow::Result<CocoaStat> {
        let n_genes = cocoa_input.n_genes;
        let n_topics = cocoa_input.n_topics;
        let n_cells = self.num_columns()?;
        let n_indv = self.num_batches();

        let pb_samples = self
            .take_grouped_columns()
            .ok_or(anyhow::anyhow!("should have pseudobulk samples assigned"))?;

        assert_eq!(n_cells, self.num_columns()?);
        assert_eq!(n_genes, self.num_rows()?);
        assert_eq!(n_cells, cocoa_input.cell_topic_nk.nrows());
        assert_eq!(n_topics, cocoa_input.cell_topic_nk.ncols());

        let n_samples = pb_samples.len();

        let mut cocoa_stat = CocoaStat::new(
            CocoaStatArgs {
                n_genes,
                n_topics,
                n_indv,
                n_samples,
            },
            cocoa_input.n_opt_iter,
            cocoa_input.hyper_param,
        );

        info!("pseudobulk data per individual and per topic (cell type)");
        self.visit_columns_by_group(&collect_basic_stat_visitor, cocoa_input, &mut cocoa_stat)?;
        info!("sum per individual");
        self.visit_columns_by_group(&collect_indv_stat_visitor, cocoa_input, &mut cocoa_stat)?;
        self.visit_columns_by_group(&collect_matched_stat_visitor, cocoa_input, &mut cocoa_stat)?;

        Ok(cocoa_stat)
    }
}

fn collect_indv_stat_visitor(
    this_sample: usize, // pseudo bulk sample
    cells: &[usize],    // cells within this pseudo bulk sample
    data: &SparseIoVec,
    input: &CocoaCollapseIn,
    arc_stat: Arc<Mutex<&mut CocoaStat>>,
) -> anyhow::Result<()> {
    let yy = data.read_columns_csc(cells.iter().cloned())?;
    let indv = data.get_batch_membership(cells.iter().cloned());

    let mut stat = arc_stat.lock().expect("lock");
    yy.col_iter()
        .zip(cells.iter().zip(indv.iter()))
        .for_each(|(y_j, (&j, &i))| {
            let rows = y_j.row_indices();
            let vals = y_j.values();

            let z_j = &input.cell_topic_nk.row(j);

            for (k, &z_jk) in z_j.iter().enumerate() {
                let y1 = stat.indv_y1_stat_mut(k);
                for (&g, &y_gj) in rows.iter().zip(vals.iter()) {
                    y1[(g, i)] += y_gj * z_jk;
                }
                let n_is = stat.indv_size_stat_mut(k);
                n_is[(i, this_sample)] += z_jk;
            }
        });

    Ok(())
}

fn collect_matched_stat_visitor(
    this_sample: usize,                   // pseudo bulk sample
    cells: &[usize],                      // cells within this pseudo bulk sample
    data: &SparseIoVec,                   // full data
    input: &CocoaCollapseIn,              //
    arc_stat: Arc<Mutex<&mut CocoaStat>>, // fill in y0 for this sample `s`
) -> anyhow::Result<()> {
    assert_eq!(data.num_rows().expect("data # features"), input.n_genes);

    let mut y0_dk = Mat::zeros(input.n_genes, input.n_topics);

    // break down cells into batch assignment groups
    let indv_vec = data.get_batch_membership(cells.iter().cloned());
    let indv_to_cells = partition_by_membership(&indv_vec, None);
    let n_indv = data.num_batches();

    let knn_batches = n_indv;
    let knn_cells = input.knn;

    for (indv_index, indv_cells) in indv_to_cells {
        ///////////////////////////////////////////////
        // let's avoid the same exposure individuals //
        ///////////////////////////////////////////////

        let this_exposure = input.exposure_assignment[indv_index];
        let same_exposure: Vec<usize> = (0..n_indv)
            .filter(|&i| input.exposure_assignment[i] == this_exposure)
            .collect();

        let i_cells = indv_cells.into_iter().map(|j| cells[j]);
        let (y0_mat, matched, distances) = data.read_neighbouring_columns_csc(
            i_cells,
            knn_batches,
            knn_cells,
            true,
            Some(&same_exposure),
        )?;

        // sum_j (sum_a y0[g,a] * w[j,a]) * z[j,k] * ind[j,s]
        let y1_to_y0 = partition_by_membership(&matched, None);

        for (y1_col, y0_cols) in y1_to_y0 {
            let z_j = &input.cell_topic_nk.row(y1_col);
            let weights: Vec<f32> = y0_cols.iter().map(|&j| (-distances[j]).exp()).collect();
            let denom = weights.iter().sum::<f32>();

            for (k, &z_jk) in z_j.iter().enumerate() {
                y0_cols.iter().zip(weights.iter()).for_each(|(&a, &w_j)| {
                    let y0_a = y0_mat.get_col(a).expect("cell a");
                    let y0_rows = y0_a.row_indices();
                    let y0_vals = y0_a.values();
                    y0_rows.iter().zip(y0_vals.iter()).for_each(|(&g, &y0_gj)| {
                        y0_dk[(g, k)] += z_jk * y0_gj * w_j / denom;
                    });
                });
            }
        }
    }

    // update statistics
    let mut stat = arc_stat.lock().expect("lock stat");
    for k in 0..input.n_topics {
        let mut y0_k_s = stat.y0_stat_mut(k).column_mut(this_sample);
        y0_k_s += &y0_dk.column(k);
    }

    Ok(())
}

fn collect_basic_stat_visitor(
    this_sample: usize,                   // sample id
    cells: &[usize],                      // cells within this sample
    data: &SparseIoVec,                   // full data
    input: &CocoaCollapseIn,              //
    arc_stat: Arc<Mutex<&mut CocoaStat>>, // fill in y1
) -> anyhow::Result<()> {
    assert_eq!(data.num_rows().expect("data # features"), input.n_genes);

    let kk = input.n_topics;
    let y1_dn = data.read_columns_csc(cells.iter().cloned())?;

    // sum_j y1[g,j] * z[j,k] * ind[j,s]
    let mut y1_dk = Mat::zeros(y1_dn.nrows(), kk);
    let mut size_k = DVec::zeros(kk);
    y1_dn
        .col_iter()
        .zip(cells.iter().cloned())
        .for_each(|(y, y1_col)| {
            let z_j = &input.cell_topic_nk.row(y1_col);
            for (k, &z_jk) in z_j.iter().enumerate() {
                let y_rows = y.row_indices();
                let y_vals = y.values();
                y_rows.iter().zip(y_vals.iter()).for_each(|(&g, &y_gj)| {
                    y1_dk[(g, k)] += z_jk * y_gj;
                });
                size_k[k] += z_jk;
            }
        });

    // update global statistics
    let mut stat = arc_stat.lock().expect("lock stat");
    for k in 0..input.n_topics {
        let mut y1_k_s = stat.y1_stat_mut(k).column_mut(this_sample);
        y1_k_s += &y1_dk.column(k);
        stat.size_stat_mut(k)[this_sample] += size_k[k];
    }

    Ok(())
}
