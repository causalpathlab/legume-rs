//! Within-cluster peak→gene refinement (pb-per-cluster).
//!
//! Re-scores the links on each cluster's pb columns.

use crate::common::*;
use crate::p2g::link_map::{link_peaks_to_genes, LinkParams, PeakGeneEdge};
use genomic_data::coordinates::{GeneTss, PeakCoord};

/// A peak–gene edge scored inside one cell/pb cluster.
#[derive(Clone, Debug, PartialEq)]
pub struct ClusterLink {
    pub edge: PeakGeneEdge,
    pub cluster: usize,
}

/// For each cluster with ≥ `min_samples` columns, run [`link_peaks_to_genes`] on the
/// column subset and tag edges with that cluster id.
pub fn refine_within_clusters(
    rna_pb: &Mat,
    atac_pb: &Mat,
    gene_tss: &[Option<GeneTss>],
    peak_coords: &[Option<PeakCoord>],
    sample_cluster: &[Option<usize>],
    min_samples: usize,
    params: &LinkParams,
) -> anyhow::Result<Vec<ClusterLink>> {
    let n_samples = rna_pb.ncols();
    anyhow::ensure!(
        atac_pb.ncols() == n_samples,
        "RNA/ATAC sample count mismatch"
    );
    anyhow::ensure!(
        sample_cluster.len() == n_samples,
        "sample_cluster length {} != samples {n_samples}",
        sample_cluster.len()
    );

    let max_c = sample_cluster.iter().flatten().copied().max();
    let Some(max_c) = max_c else {
        return Ok(Vec::new());
    };
    let n_clusters = max_c + 1;

    let mut out = Vec::new();
    for c in 0..n_clusters {
        let cols: Vec<usize> = sample_cluster
            .iter()
            .enumerate()
            .filter_map(|(j, lab)| (lab == &Some(c)).then_some(j))
            .collect();
        if cols.len() < min_samples {
            continue;
        }
        let rna_sub = subset_columns(rna_pb, &cols);
        let atac_sub = subset_columns(atac_pb, &cols);
        let edges = link_peaks_to_genes(&rna_sub, &atac_sub, gene_tss, peak_coords, params)?;
        out.extend(
            edges
                .into_iter()
                .map(|edge| ClusterLink { edge, cluster: c }),
        );
    }
    Ok(out)
}

fn subset_columns(m: &Mat, cols: &[usize]) -> Mat {
    let r = m.nrows();
    let mut out = Mat::zeros(r, cols.len());
    for (j, &c) in cols.iter().enumerate() {
        for i in 0..r {
            out[(i, j)] = m[(i, c)];
        }
    }
    out
}
