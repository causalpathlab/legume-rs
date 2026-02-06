use crate::srt_common::*;
use crate::srt_gene_graph::GenePairGraph;

use matrix_param::dmatrix_gamma::*;
use matrix_param::traits::*;

/// Accumulated δ⁺/δ⁻ statistics per gene-pair per sample
pub struct GenePairCollapsedStat {
    pub delta_pos_ds: Mat,
    pub delta_neg_ds: Mat,
    pub size_s: DVec,
    pub gene_pairs: Vec<(usize, usize)>,
    pub gene_names: Vec<Box<str>>,
    n_gene_pairs: usize,
    n_samples: usize,
}

/// Poisson-Gamma parameters for δ⁺ and δ⁻ channels
pub struct GenePairParameters {
    pub delta_pos: GammaMatrix,
    pub delta_neg: GammaMatrix,
}

/// Shared input for delta computation visitor
pub struct DeltaSharedInput {
    pub gene_log_means: DVec,
    pub gene_adj: Vec<Vec<(usize, usize)>>,
    pub n_edges: usize,
}

#[allow(dead_code)]
impl GenePairCollapsedStat {
    pub fn new(
        gene_graph: &GenePairGraph,
        n_samples: usize,
    ) -> Self {
        let n_gene_pairs = gene_graph.num_edges();
        Self {
            delta_pos_ds: Mat::zeros(n_gene_pairs, n_samples),
            delta_neg_ds: Mat::zeros(n_gene_pairs, n_samples),
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

    /// Fit Poisson-Gamma on δ⁺ and δ⁻
    pub fn optimize(
        &self,
        hyper_param: Option<(f32, f32)>,
    ) -> anyhow::Result<GenePairParameters> {
        let (a0, b0) = hyper_param.unwrap_or((1_f32, 1_f32));
        let shape = (self.n_gene_pairs, self.n_samples);

        let mut delta_pos = GammaMatrix::new(shape, a0, b0);
        let mut delta_neg = GammaMatrix::new(shape, a0, b0);

        let size_s = &self.size_s.transpose();
        let sample_size_ds = Mat::from_rows(&vec![size_s.clone(); shape.0]);

        info!("Calibrating gene-pair statistics");

        delta_pos.update_stat(&self.delta_pos_ds, &sample_size_ds);
        delta_pos.calibrate();
        delta_neg.update_stat(&self.delta_neg_ds, &sample_size_ds);
        delta_neg.calibrate();

        info!("Resolved gene-pair collapsed statistics");

        Ok(GenePairParameters {
            delta_pos,
            delta_neg,
        })
    }

    /// Write gene-pair collapsed data to parquet
    pub fn to_parquet(&self, file_path: &str) -> anyhow::Result<()> {
        let edge_names: Vec<Box<str>> = self
            .gene_pairs
            .iter()
            .map(|&(g1, g2)| {
                format!("{}:{}", self.gene_names[g1], self.gene_names[g2]).into_boxed_str()
            })
            .collect();

        // Write δ⁺ and δ⁻ as separate files
        let pos_file = file_path.replace(".parquet", ".pos.parquet");
        self.delta_pos_ds
            .to_parquet(Some(&edge_names), None, &pos_file)?;

        let neg_file = file_path.replace(".parquet", ".neg.parquet");
        self.delta_neg_ds
            .to_parquet(Some(&edge_names), None, &neg_file)?;

        Ok(())
    }
}

/// Compute gene-level log1p means: μ̃_g = E[log1p(x_g)] across all cells
pub fn compute_gene_log_means(
    data_vec: &SparseIoVec,
    block_size: usize,
) -> anyhow::Result<DVec> {
    let n_genes = data_vec.num_rows();
    let n_cells = data_vec.num_columns();

    info!(
        "Computing gene log means across {} cells, {} genes",
        n_cells, n_genes
    );

    let mut sums = DVec::zeros(n_genes);

    data_vec.visit_columns_by_block(
        &gene_log_mean_visitor,
        &(),
        &mut sums,
        Some(block_size),
    )?;

    sums /= n_cells as f32;
    Ok(sums)
}

fn gene_log_mean_visitor(
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
            local_sums[gene] += val.ln_1p();
        }
    }

    let mut sums = arc_sums.lock().expect("lock gene_log_mean sums");
    **sums += &local_sums;

    Ok(())
}

/// Compute gene-pair deltas by visiting cells grouped by sample.
///
/// For each cell c in sample s, for each gene pair (g1, g2):
///   δ = log1p(x_g1(c)) × log1p(x_g2(c)) - μ̃_g1 × μ̃_g2
///   if δ > 0: δ⁺(edge, s) += δ
///   if δ < 0: δ⁻(edge, s) += |δ|
pub fn compute_gene_pair_deltas(
    data_vec: &SparseIoVec,
    gene_graph: &GenePairGraph,
    gene_log_means: &DVec,
    n_samples: usize,
) -> anyhow::Result<GenePairCollapsedStat> {
    let mut stat = GenePairCollapsedStat::new(gene_graph, n_samples);

    let gene_adj = gene_graph.build_directed_adjacency();
    let n_edges = gene_graph.num_edges();

    info!(
        "Computing gene-pair deltas: {} edges, {} samples",
        n_edges, n_samples,
    );

    let shared_in = DeltaSharedInput {
        gene_log_means: gene_log_means.clone(),
        gene_adj,
        n_edges,
    };

    data_vec.visit_columns_by_group(
        &delta_visitor,
        &shared_in,
        &mut stat,
    )?;

    Ok(stat)
}

fn delta_visitor(
    sample: usize,
    cells: &[usize],
    data_vec: &SparseIoVec,
    shared_in: &DeltaSharedInput,
    arc_stat: Arc<Mutex<&mut GenePairCollapsedStat>>,
) -> anyhow::Result<()> {
    let gene_log_means = &shared_in.gene_log_means;
    let gene_adj = &shared_in.gene_adj;
    let n_edges = shared_in.n_edges;

    let yy = data_vec.read_columns_csc(cells.iter().cloned())?;

    let mut local_delta_pos = vec![0_f32; n_edges];
    let mut local_delta_neg = vec![0_f32; n_edges];
    let mut local_size = 0_f32;

    for y_j in yy.col_iter() {
        let rows = y_j.row_indices();
        let vals = y_j.values();

        // Build sparse lookup: gene → log1p(x)
        let gene_log1p: HashMap<usize, f32> = rows
            .iter()
            .zip(vals.iter())
            .map(|(&g, &v)| (g, v.ln_1p()))
            .collect();

        // For each present gene g1, check directed neighbors g2 > g1
        for (&g1, &val_g1) in rows.iter().zip(vals.iter()) {
            let log1p_g1 = val_g1.ln_1p();

            for &(g2, edge_idx) in gene_adj[g1].iter() {
                if let Some(&log1p_g2) = gene_log1p.get(&g2) {
                    let delta =
                        log1p_g1 * log1p_g2 - gene_log_means[g1] * gene_log_means[g2];

                    if delta > 0.0 {
                        local_delta_pos[edge_idx] += delta;
                    } else {
                        local_delta_neg[edge_idx] += -delta;
                    }
                }
            }
        }

        local_size += 1.0;
    }

    // Accumulate to shared stat
    let mut stat = arc_stat.lock().expect("lock delta stat");
    for (edge_idx, &dp) in local_delta_pos.iter().enumerate() {
        if dp > 0.0 {
            stat.delta_pos_ds[(edge_idx, sample)] += dp;
        }
    }
    for (edge_idx, &dn) in local_delta_neg.iter().enumerate() {
        if dn > 0.0 {
            stat.delta_neg_ds[(edge_idx, sample)] += dn;
        }
    }
    stat.size_s[sample] += local_size;

    Ok(())
}

/// Nystrom projection: project individual cells onto the gene-pair
/// dictionary to obtain per-cell latent codes.
///
/// For each cell, computes delta values for present gene pairs and
/// projects onto the SVD basis (split into pos/neg halves).
pub fn nystrom_gene_pair_projection(
    data_vec: &SparseIoVec,
    gene_graph: &GenePairGraph,
    gene_log_means: &DVec,
    basis_dk: &Mat,
    block_size: usize,
) -> anyhow::Result<Mat> {
    let n_cells = data_vec.num_columns();
    let n_topics = basis_dk.ncols();
    let n_edges = gene_graph.num_edges();

    info!(
        "Nystrom gene-pair projection: {} cells, {} edges, {} topics",
        n_cells, n_edges, n_topics,
    );

    // Split basis into pos (top half) and neg (bottom half)
    let basis_pos = basis_dk.rows(0, n_edges).clone_owned();
    let basis_neg = basis_dk.rows(n_edges, n_edges).clone_owned();

    let gene_adj = gene_graph.build_directed_adjacency();

    let shared_in = NystromSharedInput {
        gene_log_means: gene_log_means.clone(),
        gene_adj,
        n_edges,
        basis_pos,
        basis_neg,
    };

    let mut proj_kn = Mat::zeros(n_topics, n_cells);

    data_vec.visit_columns_by_block(
        &nystrom_gene_pair_visitor,
        &shared_in,
        &mut proj_kn,
        Some(block_size),
    )?;

    Ok(proj_kn)
}

#[allow(dead_code)]
struct NystromSharedInput {
    gene_log_means: DVec,
    gene_adj: Vec<Vec<(usize, usize)>>,
    n_edges: usize,
    basis_pos: Mat,
    basis_neg: Mat,
}

fn nystrom_gene_pair_visitor(
    bound: (usize, usize),
    data_vec: &SparseIoVec,
    shared_in: &NystromSharedInput,
    arc_proj: Arc<Mutex<&mut Mat>>,
) -> anyhow::Result<()> {
    let (lb, ub) = bound;
    let gene_log_means = &shared_in.gene_log_means;
    let gene_adj = &shared_in.gene_adj;
    let basis_pos = &shared_in.basis_pos;
    let basis_neg = &shared_in.basis_neg;
    let n_topics = basis_pos.ncols();

    let yy = data_vec.read_columns_csc(lb..ub)?;

    let n_cells_block = ub - lb;
    let mut local_proj = Mat::zeros(n_topics, n_cells_block);

    for (cell_idx, y_j) in yy.col_iter().enumerate() {
        let rows = y_j.row_indices();
        let vals = y_j.values();

        // Build sparse lookup
        let gene_log1p: HashMap<usize, f32> = rows
            .iter()
            .zip(vals.iter())
            .map(|(&g, &v)| (g, v.ln_1p()))
            .collect();

        let mut proj_k = DVec::zeros(n_topics);

        for (&g1, &val_g1) in rows.iter().zip(vals.iter()) {
            let log1p_g1 = val_g1.ln_1p();

            for &(g2, edge_idx) in gene_adj[g1].iter() {
                if let Some(&log1p_g2) = gene_log1p.get(&g2) {
                    let delta =
                        log1p_g1 * log1p_g2 - gene_log_means[g1] * gene_log_means[g2];

                    if delta > 0.0 {
                        proj_k += delta * &basis_pos.row(edge_idx).transpose();
                    } else if delta < 0.0 {
                        proj_k += (-delta) * &basis_neg.row(edge_idx).transpose();
                    }
                }
            }
        }

        local_proj.column_mut(cell_idx).copy_from(&proj_k);
    }

    let mut proj_kn = arc_proj.lock().expect("lock nystrom proj");
    proj_kn.columns_range_mut(lb..ub).copy_from(&local_proj);

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

    data_vec.visit_columns_by_group(
        &prelim_collapse_visitor,
        &(),
        &mut stat,
    )?;

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

    let mut stat = arc_stat.lock().expect("lock prelim stat");

    for y_j in yy.col_iter() {
        let rows = y_j.row_indices();
        let vals = y_j.values();
        for (&gene, &y) in rows.iter().zip(vals.iter()) {
            stat.gene_sum_ds[(gene, sample)] += y;
        }
        stat.size_s[sample] += 1_f32;
    }

    Ok(())
}
