use crate::common::*;
use crate::stat::*;
use legume_numeric::matrix::utils::partition_by_membership;
use std::sync::{Arc, Mutex};

#[cfg(test)]
mod tests;

/// Inputs for the stage-1 accumulate pass. Every cell counts: stage 1 uses
/// cell state only, never the exposure labels, so it is computed once and
/// shared by every permutation draw. No batch δ enters here: scaling counts
/// by 1/δ would cancel the between-individual spread the permutation null is
/// built from (see README, "Why δ is a random effect").
pub struct CocoaCollapseIn {
    /// a pseudobulk contributes to a topic only if at least this many
    /// individuals have cells of that topic in it
    pub min_individuals_per_pb: usize,
    pub n_genes: usize,
    pub n_topics: usize,
    pub n_opt_iter: Option<usize>,
    pub hyper_param: Option<(f32, f32)>,
    pub cell_topic_nk: Mat, // cell x cell type topic
}

pub trait CocoaCollapseOps {
    fn collect_cocoa_stat(&self, cocoa_input: &CocoaCollapseIn) -> anyhow::Result<CocoaStat>;
}

impl CocoaCollapseOps for SparseIoVec {
    fn collect_cocoa_stat(&self, cocoa_input: &CocoaCollapseIn) -> anyhow::Result<CocoaStat> {
        let n_genes = cocoa_input.n_genes;
        let n_topics = cocoa_input.n_topics;
        let n_cells = self.num_columns();
        let n_indv = self.num_batches();

        let pb_samples = self
            .take_grouped_columns()
            .ok_or(anyhow::anyhow!("should have pseudobulk samples assigned"))?;

        assert_eq!(n_genes, self.num_rows());
        assert_eq!(n_cells, cocoa_input.cell_topic_nk.nrows());
        assert_eq!(n_topics, cocoa_input.cell_topic_nk.ncols());

        let n_samples = pb_samples.len();
        info!(
            "{} pseudobulk groups over {} cells ({:.1} cells per group, {} individuals)",
            n_samples,
            n_cells,
            n_cells as f32 / n_samples.max(1) as f32,
            n_indv
        );

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

        info!("collecting statistics per topic (cell type)");
        self.visit_columns_by_group(&collect_stat_visitor, cocoa_input, &mut cocoa_stat)?;

        Ok(cocoa_stat)
    }
}

/// Sum one pseudobulk's cells, weighted by topic, into y1(gene, pseudobulk),
/// y1(gene, individual), and n(individual, pseudobulk). A topic keeps this
/// pseudobulk only if enough individuals have cells of it here; otherwise
/// the pseudobulk is dropped from that topic and counted as dropped.
fn collect_stat_visitor(
    this_sample: usize, // pseudobulk sample
    cells: &[usize],    // cells within this pseudobulk sample
    data: &SparseIoVec, // full data
    input: &CocoaCollapseIn,
    arc_stat: Arc<Mutex<&mut CocoaStat>>, // fill in y1, y1_di, size_ip
) -> anyhow::Result<()> {
    assert_eq!(data.num_rows(), input.n_genes);
    let n_genes = input.n_genes;
    let n_topics = input.n_topics;

    let y1_all = data.read_columns_csc(cells.iter().cloned())?;
    let indv_vec = data.get_batch_membership(cells.iter().cloned());
    let indv_to_cells = partition_by_membership(&indv_vec, None);

    let mut y1_dk = Mat::zeros(n_genes, n_topics);
    let mut indv_stats: Vec<(usize, Mat, DVec)> = Vec::with_capacity(indv_to_cells.len());

    for (indv_index, indv_cells) in indv_to_cells {
        let mut indv_y1_dk = Mat::zeros(n_genes, n_topics);
        let mut indv_size_k = DVec::zeros(n_topics);
        for &cell_pos in &indv_cells {
            let z_j = &input.cell_topic_nk.row(cells[cell_pos]);
            let y1_j = y1_all.get_col(cell_pos);
            for (k, &z_jk) in z_j.iter().enumerate() {
                if z_jk < 1e-8 {
                    continue;
                }
                if let Some(y1_j) = &y1_j {
                    for (&g, &y_gj) in y1_j.row_indices().iter().zip(y1_j.values().iter()) {
                        y1_dk[(g, k)] += z_jk * y_gj;
                        indv_y1_dk[(g, k)] += z_jk * y_gj;
                    }
                }
                indv_size_k[k] += z_jk;
            }
        }
        if indv_size_k.iter().any(|&v| v > 0.) {
            indv_stats.push((indv_index, indv_y1_dk, indv_size_k));
        }
    }

    // individuals with cells of each topic in this pseudobulk
    let individuals: Vec<usize> = (0..n_topics)
        .map(|k| indv_stats.iter().filter(|(_, _, sz)| sz[k] > 0.0).count())
        .collect();
    let keep: Vec<bool> = individuals
        .iter()
        .map(|&n| n >= input.min_individuals_per_pb)
        .collect();

    let mut stat = arc_stat.lock().expect("lock stat");
    for k in 0..n_topics {
        let cells: f32 = indv_stats.iter().map(|(_, _, sz)| sz[k]).sum();
        if cells <= 0.0 {
            continue;
        }
        let mix = stat.mixing_mut(k);
        if keep[k] {
            mix.pseudobulks_kept += 1;
            mix.cells_kept += cells;
            mix.individuals_per_pseudobulk.push(individuals[k]);
        } else {
            mix.pseudobulks_dropped += 1;
            mix.cells_dropped += cells;
            continue;
        }
        let mut y1_k_s = stat.y1_stat_mut(k).column_mut(this_sample);
        y1_k_s += &y1_dk.column(k);
    }
    for (indv_index, indv_y1, indv_size) in &indv_stats {
        for k in (0..n_topics).filter(|&k| keep[k]) {
            let mut col = stat.indv_y1_stat_mut(k).column_mut(*indv_index);
            col += &indv_y1.column(k);
            stat.indv_size_stat_mut(k)[(*indv_index, this_sample)] += indv_size[k];
        }
    }
    Ok(())
}
