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
        let n_cells = self.num_columns();
        let n_indv = self.num_batches();

        let pb_samples = self
            .take_grouped_columns()
            .ok_or(anyhow::anyhow!("should have pseudobulk samples assigned"))?;

        assert_eq!(n_cells, self.num_columns());
        assert_eq!(n_genes, self.num_rows());
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

        info!("matching and collecting statistics per topic (cell type)");
        self.visit_columns_by_group(&collect_matched_stat_visitor, cocoa_input, &mut cocoa_stat)?;

        Ok(cocoa_stat)
    }
}

fn collect_matched_stat_visitor(
    this_sample: usize,                   // pseudo bulk sample
    cells: &[usize],                      // cells within this pseudo bulk sample
    data: &SparseIoVec,                   // full data
    input: &CocoaCollapseIn,              //
    arc_stat: Arc<Mutex<&mut CocoaStat>>, // fill in y1, y0, size, y1_di, size_ip
) -> anyhow::Result<()> {
    assert_eq!(data.num_rows(), input.n_genes);

    let n_genes = input.n_genes;
    let n_topics = input.n_topics;

    let mut y0_dk = Mat::zeros(n_genes, n_topics);
    let mut y1_dk = Mat::zeros(n_genes, n_topics);
    let mut size_k = DVec::zeros(n_topics);

    // Read ALL source cells in this sample once
    let y1_all = data.read_columns_csc(cells.iter().cloned())?;

    // Break down cells into batch (individual) assignment groups
    let indv_vec = data.get_batch_membership(cells.iter().cloned());
    let indv_to_cells = partition_by_membership(&indv_vec, None);
    let n_indv = data.num_batches();

    let knn_batches = n_indv;
    let knn_cells = input.knn;

    // Per-individual y1 and size (for indv-level stats)
    let mut indv_stats: Vec<(usize, Mat, DVec)> = Vec::new();

    for (indv_index, indv_cells) in indv_to_cells {
        ///////////////////////////////////////////////
        // let's avoid the same exposure individuals //
        ///////////////////////////////////////////////

        let this_exposure = input.exposure_assignment[indv_index];
        let same_exposure: Vec<usize> = (0..n_indv)
            .filter(|&i| input.exposure_assignment[i] == this_exposure)
            .collect();

        // indv_cells[p] is a position in `cells`; cells[indv_cells[p]] is global index
        let (y0_mat, matched, matched_glob, distances) = data.read_neighbouring_columns_csc(
            indv_cells.iter().map(|&j| cells[j]),
            knn_batches,
            knn_cells,
            true,
            Some(&same_exposure),
        )?;

        let y1_to_y0 = partition_by_membership(&matched, None);

        let mut indv_y1_dk = Mat::zeros(n_genes, n_topics);
        let mut indv_size_k = DVec::zeros(n_topics);

        // Process ALL source cells — only accumulate y1 alongside valid y0
        for &cell_pos in &indv_cells {
            let cell_glob = cells[cell_pos];
            let z_j = &input.cell_topic_nk.row(cell_glob);

            let y0_cols = match y1_to_y0.get(&cell_glob) {
                Some(cols) => cols.as_slice(),
                None => continue, // no matches at all — skip y1 and y0
            };

            for (k, &z_jk) in z_j.iter().enumerate() {
                if z_jk < 1e-8 {
                    continue;
                }

                // Weight by both distance and matched cell's topic membership
                let weights: Vec<f32> = y0_cols
                    .iter()
                    .map(|&a| {
                        let z_matched_k = input.cell_topic_nk[(matched_glob[a], k)];
                        (-distances[a]).exp() * z_matched_k
                    })
                    .collect();
                let denom = weights.iter().sum::<f32>();
                if denom < 1e-8 {
                    continue; // no same-type matches — skip both y1 and y0
                }

                // Accumulate y0
                y0_cols.iter().zip(weights.iter()).for_each(|(&a, &w_j)| {
                    let y0_a = y0_mat.get_col(a).expect("cell a");
                    y0_a.row_indices()
                        .iter()
                        .zip(y0_a.values().iter())
                        .for_each(|(&g, &y0_gj)| {
                            y0_dk[(g, k)] += z_jk * y0_gj * w_j / denom;
                        });
                });

                // Accumulate y1 symmetrically
                if let Some(y1_j) = y1_all.get_col(cell_pos) {
                    for (&g, &y_gj) in y1_j.row_indices().iter().zip(y1_j.values().iter()) {
                        y1_dk[(g, k)] += z_jk * y_gj;
                        indv_y1_dk[(g, k)] += z_jk * y_gj;
                    }
                }
                size_k[k] += z_jk;
                indv_size_k[k] += z_jk;
            }
        }

        if indv_size_k.iter().any(|&v| v > 0.) {
            indv_stats.push((indv_index, indv_y1_dk, indv_size_k));
        }
    }

    // Update statistics
    let mut stat = arc_stat.lock().expect("lock stat");
    for k in 0..n_topics {
        let mut y0_k_s = stat.y0_stat_mut(k).column_mut(this_sample);
        y0_k_s += &y0_dk.column(k);

        let mut y1_k_s = stat.y1_stat_mut(k).column_mut(this_sample);
        y1_k_s += &y1_dk.column(k);

        stat.size_stat_mut(k)[this_sample] += size_k[k];
    }

    for (indv_index, indv_y1, indv_size) in &indv_stats {
        for k in 0..n_topics {
            let y1_di = stat.indv_y1_stat_mut(k);
            for g in 0..n_genes {
                y1_di[(g, *indv_index)] += indv_y1[(g, k)];
            }
            stat.indv_size_stat_mut(k)[(*indv_index, this_sample)] += indv_size[k];
        }
    }

    Ok(())
}
