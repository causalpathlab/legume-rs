//! Within-cluster peak→gene refinement (pb-per-cluster).
//!
//! Re-runs the rough ABC co-occurrence map on each cluster's pb columns.

use crate::common::*;
use crate::p2g::abc_map::{rough_abc_map, AbcMapParams, PeakGeneEdge};
use genomic_data::coordinates::{GeneTss, PeakCoord};

/// A peak–gene edge scored inside one cell/pb cluster.
#[derive(Clone, Debug, PartialEq)]
pub struct ClusterLink {
    pub edge: PeakGeneEdge,
    pub cluster: usize,
}

/// For each cluster with ≥ `min_samples` columns, run [`rough_abc_map`] on the
/// column subset and tag edges with that cluster id.
pub fn refine_within_clusters(
    rna_pb: &Mat,
    atac_pb: &Mat,
    gene_tss: &[Option<GeneTss>],
    peak_coords: &[Option<PeakCoord>],
    sample_cluster: &[Option<usize>],
    min_samples: usize,
    params: &AbcMapParams,
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
        let edges = rough_abc_map(&rna_sub, &atac_sub, gene_tss, peak_coords, params)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use genomic_data::coordinates::{GeneTss, PeakCoord};

    #[test]
    fn link_appears_only_in_active_cluster() {
        let s = 40usize;
        let mut rna = Mat::zeros(1, s);
        let mut atac = Mat::zeros(1, s);
        let mut sample_cluster = vec![Some(0usize); s];
        for j in 0..s {
            if j < 20 {
                let z = (j as f32) * 0.2;
                let signal = z.sin().abs() + 0.1;
                rna[(0, j)] = signal * 10.0;
                atac[(0, j)] = signal * 8.0;
                sample_cluster[j] = Some(0);
            } else {
                // Cluster 1: gene and peak fluctuate independently.
                rna[(0, j)] = ((j % 5) as f32) * 0.5 + 0.1;
                atac[(0, j)] = ((j % 7) as f32) * 0.4 + 0.1;
                sample_cluster[j] = Some(1);
            }
        }

        let gene_tss = vec![Some(GeneTss {
            chr: "1".into(),
            tss: 100_000,
        })];
        let peak_coords = vec![Some(PeakCoord {
            chr: "1".into(),
            start: 100_000,
            end: 100_500,
        })];
        let params = AbcMapParams {
            cis_window: 500_000,
            max_cis: 10,
            min_weight: 0.3,
        };

        let links = refine_within_clusters(
            &rna,
            &atac,
            &gene_tss,
            &peak_coords,
            &sample_cluster,
            5,
            &params,
        )
        .unwrap();

        let in0 = links.iter().filter(|l| l.cluster == 0).count();
        let in1 = links.iter().filter(|l| l.cluster == 1).count();
        assert!(
            in0 >= 1,
            "active cluster should keep the cis link; got {links:?}"
        );
        assert_eq!(
            in1, 0,
            "inactive cluster should not pass min_weight; got {links:?}"
        );
    }
}
