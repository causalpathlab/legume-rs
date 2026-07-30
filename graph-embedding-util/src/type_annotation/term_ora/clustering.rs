//! Step 4 of the firm pipeline: the grouping the over-representation test is run over.
//!
//! Leiden over the cells' own cosine kNN graph, split in two — the graph is built once and
//! reused, while the partition is redrawn per bootstrap replicate — plus the per-cluster
//! cell counts every output reports the call's power by.

use super::TermOraConfig;
use anyhow::Result;
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::knn_graph::{KnnGraph, KnnGraphArgs};

/// Cells per cluster.
pub(super) fn cluster_sizes(community: &[usize], n_comm: usize) -> Vec<usize> {
    let mut sizes = vec![0usize; n_comm];
    for &k in community {
        if k < n_comm {
            sizes[k] += 1;
        }
    }
    sizes
}

/// Leiden communities over a cosine cell kNN graph (cells L2-normalized for the
/// graph; gem `e_cell` is already unit, so this matches the assignment geometry).
///
/// The kNN graph is now **deterministic** (matrix-util's seeded instant-distance backend), so this
/// step reproduces run-to-run and `seed` pins Leiden on top of a fixed graph. Historically, under
/// the old un-seedable `hnsw_rs` backend it was *not* reproducible: four identical invocations on
/// 15,315 cord-blood cells at `--resolution 8` gave 990 / 132 / 137 / 138 communities, agreeing on
/// only 83–94% of the labels. That cross-run instability is resolved.
///
/// A single partition still should not be over-trusted: within a run, Leiden picks among near-equal
/// modularity optima, so [`super::MarkerBootstrapConfig::recluster`] reseeds it once per bootstrap
/// replicate and lets that partition-choice uncertainty land in the per-cell support.
pub(super) fn cell_knn_graph(
    cell_flat: &[f32],
    n: usize,
    h: usize,
    cfg: &TermOraConfig,
) -> Result<KnnGraph> {
    let mut cell_u = cell_flat.to_vec();
    crate::type_annotation::score::l2_normalize_rows(&mut cell_u, n, h);
    let cell_mat = DMatrix::<f32>::from_row_iterator(n, h, cell_u.iter().copied());
    KnnGraph::from_rows(
        &cell_mat,
        KnnGraphArgs {
            knn: cfg.knn.clamp(1, n - 1),
            block_size: 1000,
            reciprocal: false,
        },
    )
}

/// Leiden over a **prebuilt** cell kNN graph.
///
/// The graph is built once and reused by every bootstrap replicate, and that is deliberate. The
/// bootstrap resamples the *marker panel*; the cell embedding it clusters is identical on every
/// draw, so the graph's input never changes. The graph is also deterministic now (seeded
/// instant-distance), so rebuilding it would reproduce the identical graph — nothing to gain.
///
/// Leiden's `seed` is a different animal and *is* redrawn per replicate: modularity has many
/// near-equal optima and which one the optimiser lands in is a real, load-bearing arbitrary choice.
/// Holding the partition fixed across replicates makes the bootstrap abstain on nothing (measured
/// on the fixed graph: 0% unassigned, and its support falls from AUC 0.93 to 0.69 at separating
/// spurious calls), because a cluster's argmax will not flip when only the panel jiggles. The
/// partition is where the within-run instability lives, so the partition is what gets resampled.
pub(super) fn cluster_cells(
    graph: &KnnGraph,
    n: usize,
    cfg: &TermOraConfig,
    seed: u64,
) -> Vec<usize> {
    crate::type_annotation::layout::leiden_from_graph(graph, n, cfg.resolution, seed)
}
