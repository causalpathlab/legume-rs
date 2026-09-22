//! Init gene and peak feature partitions for hierarchical embed.
//!
//! Each axis is warm-started from **frozen finest-level pb profiles** (same
//! signal as ge-util module warm). Cis links couple the two partitions: a
//! linked peak starts in the module of its strongest linked gene, so peaks that
//! target one gene program share a module; unlinked peaks are clustered on
//! their own ATAC profile into modules of their own.

use crate::common::Mat;
use crate::p2g::link_map::PeakGeneEdge;
use crate::p2g::pb_levels::PbLevels;
use graph_embedding_util::fit::hier::partition::Partition;
use graph_embedding_util::fit::module_warm::warm_start_module_labels;
use legume_numeric::matrix::rand_util::name_seed;

/// Build coarse gene levels by summing child pb columns (same tree as the
/// atac/rna collapse). `parent[l][i]` is the level-`l + 1` column that holds
/// level-`l` column `i`.
pub fn coarsen_profile_levels(finest: &Mat, parent: &[Vec<usize>]) -> Vec<Mat> {
    let mut levels = vec![finest.clone()];
    for map in parent {
        let prev = levels.last().expect("finest level");
        let n_parents = map.iter().max().map_or(0, |&p| p + 1);
        let mut out = Mat::zeros(prev.nrows(), n_parents);
        for (child, &par) in map.iter().enumerate() {
            for r in 0..prev.nrows() {
                out[(r, par)] += prev[(r, child)];
            }
        }
        levels.push(out);
    }
    levels
}

/// Warm-start the gene partition on the finest gene profiles and the peak
/// partition from the links plus the finest ATAC profiles.
///
/// Peak module ids `0..n_gene_modules` are the gene modules: a linked peak
/// takes the module of its strongest linked gene (ties: the first such edge).
/// Peaks with no link are k-means clustered on their ATAC profile into
/// `n_peak_modules` further modules, ids `n_gene_modules..`, so the two kinds
/// never share an id. The peak partition therefore has
/// `n_gene_modules + n_peak_modules` modules; those with no member are empty.
pub fn init_gene_peak_partitions(
    levels: &PbLevels,
    gene_finest: &Mat,
    edges: &[PeakGeneEdge],
    n_gene_modules: usize,
    n_peak_modules: usize,
    seed: u64,
) -> anyhow::Result<(Partition, Partition)> {
    let atac_finest = levels.atac.first();
    anyhow::ensure!(
        gene_finest.nrows() > 0 && atac_finest.is_some_and(|a| a.nrows() > 0),
        "need gene and peak profiles"
    );
    let atac_finest = atac_finest.expect("checked above");
    let n_genes = gene_finest.nrows();
    let n_peaks = atac_finest.nrows();
    let gene_labels =
        warm_start_module_labels(gene_finest, n_gene_modules, name_seed(seed, "gene modules"));

    // Strongest link per peak.
    let mut best: Vec<Option<(usize, f32)>> = vec![None; n_peaks];
    for e in edges {
        if e.peak >= n_peaks || e.gene >= n_genes {
            continue;
        }
        let slot = &mut best[e.peak];
        if slot.is_none_or(|(_, w)| e.weight > w) {
            *slot = Some((e.gene, e.weight));
        }
    }
    let unlinked: Vec<usize> = (0..n_peaks).filter(|&p| best[p].is_none()).collect();
    let mut peak_labels = vec![0u32; n_peaks];
    for (p, b) in best.iter().enumerate() {
        if let Some((g, _)) = b {
            peak_labels[p] = gene_labels[*g];
        }
    }
    if !unlinked.is_empty() {
        let sub = Mat::from_fn(unlinked.len(), atac_finest.ncols(), |r, j| {
            atac_finest[(unlinked[r], j)]
        });
        let sub_labels =
            warm_start_module_labels(&sub, n_peak_modules, name_seed(seed, "peak modules"));
        for (r, &p) in unlinked.iter().enumerate() {
            peak_labels[p] = n_gene_modules as u32 + sub_labels[r];
        }
    }
    Ok((
        Partition::from_labels(&gene_labels, n_gene_modules),
        Partition::from_labels(&peak_labels, n_gene_modules + n_peak_modules),
    ))
}
