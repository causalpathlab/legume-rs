use crate::srt_common::*;
use crate::srt_gene_graph::GenePairGraph;

use matrix_param::dmatrix_gamma::*;
use matrix_param::traits::*;

/// Accumulated δ⁺ statistics per gene-pair per sample
#[allow(dead_code)]
pub struct GenePairCollapsedStat {
    pub delta_pos_ds: Mat,
    pub size_s: DVec,
    pub gene_pairs: Vec<(usize, usize)>,
    pub gene_names: Vec<Box<str>>,
    n_gene_pairs: usize,
    n_samples: usize,
}

/// Poisson-Gamma parameters for δ⁺ channel
pub struct GenePairParameters {
    pub delta_pos: GammaMatrix,
}

/// Shared input for delta computation visitor
struct DeltaSharedInput {
    pub gene_means: DVec,
    pub gene_adj: Vec<Vec<(usize, usize)>>,
    pub n_edges: usize,
    pub use_log1p: bool,
}

#[allow(dead_code)]
impl GenePairCollapsedStat {
    pub fn new(gene_graph: &GenePairGraph, n_samples: usize) -> Self {
        let n_gene_pairs = gene_graph.num_edges();
        Self {
            delta_pos_ds: Mat::zeros(n_gene_pairs, n_samples),
            size_s: DVec::zeros(n_samples),
            gene_pairs: gene_graph.gene_edges.clone(),
            gene_names: gene_graph.gene_names.clone(),
            n_gene_pairs,
            n_samples,
        }
    }

    pub fn num_gene_pairs(&self) -> usize {
        self.n_gene_pairs
    }

    pub fn num_samples(&self) -> usize {
        self.n_samples
    }

    /// Remove low-signal gene-pair edges.
    ///
    /// When `use_elbow` is true, applies a kneedle elbow on sorted row sums
    /// to prune noise-level edges (for union KNN or external networks).
    /// When false, removes only exact zeros (for reciprocal KNN).
    ///
    /// Mutates self and gene_graph in place. Returns number of removed edges.
    pub fn filter_empty_edges(&mut self, gene_graph: &mut GenePairGraph, use_elbow: bool) -> usize {
        let row_sums: Vec<f32> = (0..self.n_gene_pairs)
            .map(|i| self.delta_pos_ds.row(i).sum())
            .collect();

        let (threshold, elbow_rank) = if use_elbow {
            elbow_threshold(&row_sums)
        } else {
            (0.0, 0)
        };

        let keep_indices: Vec<usize> = (0..self.n_gene_pairs)
            .filter(|&i| row_sums[i] > threshold)
            .collect();

        let n_removed = self.n_gene_pairs - keep_indices.len();
        if n_removed == 0 {
            return 0;
        }

        let n_kept = keep_indices.len();
        let mut new_delta = Mat::zeros(n_kept, self.n_samples);
        for (new_idx, &old_idx) in keep_indices.iter().enumerate() {
            new_delta.row_mut(new_idx).copy_from(&self.delta_pos_ds.row(old_idx));
        }

        self.delta_pos_ds = new_delta;
        self.gene_pairs = keep_indices.iter().map(|&i| self.gene_pairs[i]).collect();
        self.n_gene_pairs = n_kept;

        gene_graph.filter_edges(&keep_indices);

        info!(
            "Elbow threshold: {:.4} (rank {}), kept {}/{} edges",
            threshold, elbow_rank, n_kept, n_kept + n_removed
        );

        n_removed
    }

    /// Fit Poisson-Gamma on δ⁺
    pub fn optimize(&self, hyper_param: Option<(f32, f32)>) -> anyhow::Result<GenePairParameters> {
        let (a0, b0) = hyper_param.unwrap_or((1_f32, 1_f32));
        let shape = (self.n_gene_pairs, self.n_samples);

        let mut delta_pos = GammaMatrix::new(shape, a0, b0);

        let size_s = &self.size_s.transpose();
        let sample_size_ds = Mat::from_rows(&vec![size_s.clone(); shape.0]);

        info!("Calibrating gene-pair statistics");

        delta_pos.update_stat(&self.delta_pos_ds, &sample_size_ds);
        delta_pos.calibrate();

        info!("Resolved gene-pair collapsed statistics");

        Ok(GenePairParameters { delta_pos })
    }

}

/// Visit all gene-pair interaction deltas for a single cell's sparse
/// expression vector. Calls `on_delta(edge_idx, delta)` for each
/// observed gene pair (g1, g2) where both genes are present.
///
/// When `use_log1p` is true, computes `log1p(g1)*log1p(g2) - μ̃_g1*μ̃_g2`.
/// When false, computes `g1*g2 - μ_g1*μ_g2` on raw counts.
#[inline]
pub fn visit_gene_pair_deltas(
    rows: &[usize],
    vals: &[f32],
    gene_adj: &[Vec<(usize, usize)>],
    gene_means: &DVec,
    use_log1p: bool,
    mut on_delta: impl FnMut(usize, f32),
) {
    let gene_vals: HashMap<usize, f32> = rows
        .iter()
        .zip(vals.iter())
        .map(|(&g, &v)| (g, if use_log1p { v.ln_1p() } else { v }))
        .collect();

    for (&g1, &val_g1) in rows.iter().zip(vals.iter()) {
        let t_g1 = if use_log1p { val_g1.ln_1p() } else { val_g1 };
        for &(g2, edge_idx) in gene_adj[g1].iter() {
            if let Some(&t_g2) = gene_vals.get(&g2) {
                let delta = t_g1 * t_g2 - gene_means[g1] * gene_means[g2];
                on_delta(edge_idx, delta);
            }
        }
    }
}

/// Compute gene-level raw means: μ_g = E[x_g] across all cells
pub fn compute_gene_raw_means(data_vec: &SparseIoVec, block_size: usize) -> anyhow::Result<DVec> {
    let n_genes = data_vec.num_rows();
    let n_cells = data_vec.num_columns();

    info!(
        "Computing gene raw means across {} cells, {} genes",
        n_cells, n_genes
    );

    let mut sums = DVec::zeros(n_genes);

    data_vec.visit_columns_by_block(&gene_raw_mean_visitor, &(), &mut sums, Some(block_size))?;

    sums /= n_cells as f32;
    Ok(sums)
}

fn gene_raw_mean_visitor(
    bound: (usize, usize),
    data_vec: &SparseIoVec,
    _: &(),
    arc_sums: Arc<Mutex<&mut DVec>>,
) -> anyhow::Result<()> {
    let (lb, ub) = bound;
    let yy = data_vec.read_columns_csc(lb..ub)?;

    let mut local_sums = DVec::zeros(yy.nrows());

    for y_j in yy.col_iter() {
        for (&gene, &val) in y_j.row_indices().iter().zip(y_j.values().iter()) {
            local_sums[gene] += val;
        }
    }

    let mut sums = arc_sums.lock().expect("lock gene_raw_mean sums");
    **sums += &local_sums;

    Ok(())
}

/// Compute gene-pair deltas by visiting cells grouped by sample.
///
/// For each cell c in sample s, for each gene pair (g1, g2):
///   When use_log1p=true:  δ = log1p(x_g1) × log1p(x_g2) - μ̃_g1 × μ̃_g2
///   When use_log1p=false: δ = x_g1 × x_g2 - μ_g1 × μ_g2
///   if δ > 0: δ⁺(edge, s) += δ
///   if δ < 0: δ⁻(edge, s) += |δ|
pub fn compute_gene_interaction_deltas(
    data_vec: &SparseIoVec,
    gene_graph: &GenePairGraph,
    gene_means: &DVec,
    n_samples: usize,
    use_log1p: bool,
) -> anyhow::Result<GenePairCollapsedStat> {
    let mut stat = GenePairCollapsedStat::new(gene_graph, n_samples);

    let gene_adj = gene_graph.build_directed_adjacency();
    let n_edges = gene_graph.num_edges();

    info!(
        "Computing gene-pair deltas: {} edges, {} samples, log1p={}",
        n_edges, n_samples, use_log1p,
    );

    let shared_in = DeltaSharedInput {
        gene_means: gene_means.clone(),
        gene_adj,
        n_edges,
        use_log1p,
    };

    data_vec.visit_columns_by_group(&gene_interaction_delta_visitor, &shared_in, &mut stat)?;

    Ok(stat)
}

fn gene_interaction_delta_visitor(
    sample: usize,
    cells: &[usize],
    data_vec: &SparseIoVec,
    shared_in: &DeltaSharedInput,
    arc_stat: Arc<Mutex<&mut GenePairCollapsedStat>>,
) -> anyhow::Result<()> {
    let gene_means = &shared_in.gene_means;
    let gene_adj = &shared_in.gene_adj;
    let n_edges = shared_in.n_edges;
    let use_log1p = shared_in.use_log1p;

    let yy = data_vec.read_columns_csc(cells.iter().cloned())?;

    let mut local_delta_pos = vec![0_f32; n_edges];
    let mut local_size = 0_f32;

    for y_j in yy.col_iter() {
        let rows = y_j.row_indices();
        let vals = y_j.values();

        visit_gene_pair_deltas(rows, vals, gene_adj, gene_means, use_log1p, |edge_idx, delta| {
            if delta > 0.0 {
                local_delta_pos[edge_idx] += delta;
            }
        });

        local_size += 1.0;
    }

    // Accumulate to shared stat
    let mut stat = arc_stat.lock().expect("lock delta stat");
    for (edge_idx, &dp) in local_delta_pos.iter().enumerate() {
        if dp > 0.0 {
            stat.delta_pos_ds[(edge_idx, sample)] += dp;
        }
    }
    stat.size_s[sample] += local_size;

    Ok(())
}

/// Simple preliminary collapse: sum observed expression per gene per sample.
/// Returns (gene_sum_ds, size_s) for building the gene graph.
pub fn preliminary_collapse(
    data_vec: &SparseIoVec,
    n_genes: usize,
    n_samples: usize,
) -> anyhow::Result<(Mat, DVec)> {
    info!(
        "Preliminary collapse: {} genes, {} samples",
        n_genes, n_samples,
    );

    let mut stat = PrelimStat {
        gene_sum_ds: Mat::zeros(n_genes, n_samples),
        size_s: DVec::zeros(n_samples),
    };

    data_vec.visit_columns_by_group(&prelim_collapse_visitor, &(), &mut stat)?;

    Ok((stat.gene_sum_ds, stat.size_s))
}

struct PrelimStat {
    gene_sum_ds: Mat,
    size_s: DVec,
}

fn prelim_collapse_visitor(
    sample: usize,
    cells: &[usize],
    data_vec: &SparseIoVec,
    _: &(),
    arc_stat: Arc<Mutex<&mut PrelimStat>>,
) -> anyhow::Result<()> {
    let yy = data_vec.read_columns_csc(cells.iter().cloned())?;

    let n_genes = yy.nrows();
    let mut local_sum = DVec::zeros(n_genes);
    let mut local_size = 0_f32;

    for y_j in yy.col_iter() {
        for (&gene, &y) in y_j.row_indices().iter().zip(y_j.values().iter()) {
            local_sum[gene] += y;
        }
        local_size += 1_f32;
    }

    let mut stat = arc_stat.lock().expect("lock prelim stat");
    let mut col = stat.gene_sum_ds.column_mut(sample);
    col += &local_sum;
    stat.size_s[sample] += local_size;

    Ok(())
}

/// Find elbow threshold from a set of values using the kneedle method.
///
/// Sorts positive values descending, normalizes rank and value to [0, 1],
/// then finds the point with maximum perpendicular distance from the line
/// connecting the first and last points.
///
/// Returns (threshold, elbow_rank). Threshold is 0.0 if no clear elbow
/// exists (all values similar or too few points).
fn elbow_threshold(values: &[f32]) -> (f32, usize) {
    // Only consider positive values for elbow detection
    let mut positive: Vec<f32> = values.iter().copied().filter(|&v| v > 0.0).collect();
    if positive.len() < 3 {
        return (0.0, 0);
    }

    positive.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let n = positive.len();

    // Normalize: x = rank/(n-1) in [0,1], y = value/max in [0,1]
    let y_max = positive[0] as f64;
    if y_max < 1e-12 {
        return (0.0, 0);
    }

    let x1 = 1.0_f64;
    let y1 = positive[n - 1] as f64 / y_max;

    // Line from (0, 1) to (1, y1)
    let dx = x1;
    let dy = y1 - 1.0;
    let line_len = (dx * dx + dy * dy).sqrt();

    let mut max_dist = 0.0_f64;
    let mut elbow_idx = 0;

    for i in 1..(n - 1) {
        let px = i as f64 / (n - 1) as f64;
        let py = positive[i] as f64 / y_max - 1.0;
        let dist = (px * dy - py * dx).abs() / line_len;
        if dist > max_dist {
            max_dist = dist;
            elbow_idx = i;
        }
    }

    (positive[elbow_idx], elbow_idx)
}
