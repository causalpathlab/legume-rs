//! Multi-seed sketch coarsening for cells and unified features.
//!
//! Cell coarsening: K `FeatureCoarsening`s over the unified cell axis
//! built from sparse-triplet random-projection sketches.
//!
//! Feature coarsening: K unified `FeatureCoarsening`s over the
//! `[n_genes + n_peaks]` feature axis, modality-preserving — gene
//! fine indices map to the first `G_blocks_k` coarse blocks, peak
//! fine indices map to the next `P_blocks_k` coarse blocks. No
//! coarse block ever mixes genes and peaks. The unified per-seed
//! coarsening drives a single shared `E_feat` table at training time.
//!
//! Sparsity-aware: sketches built in O(nnz). Output is per-seed
//! `FeatureCoarsening` (fine→coarse map + block lists), used at
//! training time to mean-pool only touched coarse blocks.

use crate::common::*;
use crate::embed::data::Triplet;
use data_beans_alg::feature_coarsening::compute_feature_coarsening;
use nalgebra::DMatrix;
use rand::{rngs::StdRng, RngExt, SeedableRng};
use rand_distr::StandardNormal;

/// One axis's K coarsenings.
pub struct AxisCoarsenings {
    pub coarsenings: Vec<FeatureCoarsening>,
    /// Fine cardinality (G or P or N).
    #[allow(dead_code)]
    pub n_fine: usize,
}

impl AxisCoarsenings {
    /// Average coarse-block cardinality across the K coarsenings.
    pub fn avg_n_coarse(&self) -> f32 {
        if self.coarsenings.is_empty() {
            return 0.0;
        }
        let sum: usize = self.coarsenings.iter().map(|c| c.num_coarse).sum();
        sum as f32 / self.coarsenings.len() as f32
    }
}

/// Inputs for `build_unified_feature_coarsenings`.
pub struct UnifiedFeatureCoarseningArgs<'a> {
    /// Unified triplets (RNA + ATAC concatenated, with feature indices
    /// already offset so peaks are at `[n_genes, n_genes + n_peaks)`).
    pub triplets: &'a [Triplet],
    pub n_genes: usize,
    pub n_peaks: usize,
    pub n_cells: usize,
    pub target_gene_blocks: usize,
    pub target_peak_blocks: usize,
    pub sketch_dim: usize,
    pub n_seeds: usize,
    pub base_seed: u64,
}

/// Build `K` unified gene+peak coarsenings, modality-preserving.
///
/// Each per-seed coarsening is a `FeatureCoarsening` of length
/// `n_genes + n_peaks` where:
/// - gene fine indices `[0, n_genes)` map to coarse blocks `[0, G_k)`
/// - peak fine indices `[n_genes, n_genes + n_peaks)` map to coarse
///   blocks `[G_k, G_k + P_k)`
///
/// No coarse block mixes the two modalities.
pub fn build_unified_feature_coarsenings(
    args: UnifiedFeatureCoarseningArgs,
) -> anyhow::Result<AxisCoarsenings> {
    let mut coarsenings = Vec::with_capacity(args.n_seeds);
    let n_features = args.n_genes + args.n_peaks;

    for k in 0..args.n_seeds {
        let seed = args.base_seed.wrapping_add(k as u64);

        // Build separate sketches per modality, restricted to that
        // modality's triplets. The gene sketch uses RNA edges only;
        // the peak sketch uses ATAC edges only.
        let gene_sketch = sketch_features_via_random_projection(
            args.triplets,
            args.n_genes,
            args.n_cells,
            args.sketch_dim,
            seed,
            0,
            args.n_genes as u32,
        );
        let peak_sketch = sketch_features_via_random_projection(
            args.triplets,
            args.n_peaks,
            args.n_cells,
            args.sketch_dim,
            seed.wrapping_add(0xA7AC),
            args.n_genes as u32,
            (args.n_genes + args.n_peaks) as u32,
        );

        let gene_fc = compute_feature_coarsening(&gene_sketch, args.target_gene_blocks)?;
        let peak_fc = compute_feature_coarsening(&peak_sketch, args.target_peak_blocks)?;

        let unified = concat_modality_coarsenings(&gene_fc, &peak_fc, args.n_genes);
        info!(
            "Feature coarsening seed={}: {} genes → {} blocks, {} peaks → {} blocks (unified {} blocks)",
            k,
            args.n_genes,
            gene_fc.num_coarse,
            args.n_peaks,
            peak_fc.num_coarse,
            unified.num_coarse,
        );
        coarsenings.push(unified);
    }
    Ok(AxisCoarsenings {
        coarsenings,
        n_fine: n_features,
    })
}

/// Concatenate a gene-side and peak-side `FeatureCoarsening` into a
/// unified one over `[0, n_genes + n_peaks)` with peak coarse blocks
/// offset by `gene_fc.num_coarse` and peak fine indices offset by
/// `n_genes`. Block boundaries between the two modalities are preserved.
fn concat_modality_coarsenings(
    gene_fc: &FeatureCoarsening,
    peak_fc: &FeatureCoarsening,
    n_genes: usize,
) -> FeatureCoarsening {
    let g_coarse = gene_fc.num_coarse;
    let n_features = n_genes + peak_fc.fine_to_coarse.len();

    let mut fine_to_coarse = Vec::with_capacity(n_features);
    for &c in &gene_fc.fine_to_coarse {
        fine_to_coarse.push(c);
    }
    for &c in &peak_fc.fine_to_coarse {
        fine_to_coarse.push(c + g_coarse);
    }

    let mut coarse_to_fine = Vec::with_capacity(g_coarse + peak_fc.num_coarse);
    for fines in &gene_fc.coarse_to_fine {
        coarse_to_fine.push(fines.clone());
    }
    for fines in &peak_fc.coarse_to_fine {
        let shifted: Vec<usize> = fines.iter().map(|&f| f + n_genes).collect();
        coarse_to_fine.push(shifted);
    }

    FeatureCoarsening {
        fine_to_coarse,
        coarse_to_fine,
        num_coarse: g_coarse + peak_fc.num_coarse,
    }
}

/// Inputs for `build_cell_coarsenings`.
pub struct CellCoarseningArgs<'a> {
    /// Unified triplets — both modalities feed the cell sketch.
    pub triplets: &'a [Triplet],
    pub n_cells: usize,
    pub n_features: usize,
    pub target_blocks: usize,
    pub sketch_dim: usize,
    pub n_seeds: usize,
    pub base_seed: u64,
}

/// Build `K` cell coarsenings from the unified triplet stream.
pub fn build_cell_coarsenings(args: CellCoarseningArgs) -> anyhow::Result<AxisCoarsenings> {
    let mut coarsenings = Vec::with_capacity(args.n_seeds);
    for k in 0..args.n_seeds {
        let seed = args.base_seed.wrapping_add(k as u64);
        let sketch = sketch_cells_via_random_projection(
            args.triplets,
            args.n_cells,
            args.n_features,
            args.sketch_dim,
            seed,
        );
        let fc = compute_feature_coarsening(&sketch, args.target_blocks)?;
        info!(
            "Cell coarsening seed={}: {} → {} blocks",
            k, args.n_cells, fc.num_coarse
        );
        coarsenings.push(fc);
    }
    Ok(AxisCoarsenings {
        coarsenings,
        n_fine: args.n_cells,
    })
}

/// Produce a `[n_features_in_range, sketch_dim]` sketch from the
/// triplets whose `feature` falls in `[feat_lo, feat_hi)`.
///
/// `n_features_in_range` should equal `feat_hi - feat_lo` (the
/// modality's fine cardinality). Feature indices in the output are
/// local to the modality (subtract `feat_lo`).
fn sketch_features_via_random_projection(
    triplets: &[Triplet],
    n_features_in_range: usize,
    n_cells: usize,
    sketch_dim: usize,
    seed: u64,
    feat_lo: u32,
    feat_hi: u32,
) -> DMatrix<f32> {
    let r = gaussian_matrix(n_cells, sketch_dim, seed);
    let mut sketch = DMatrix::<f32>::zeros(n_features_in_range, sketch_dim);
    for t in triplets {
        if t.feature < feat_lo || t.feature >= feat_hi {
            continue;
        }
        let f_local = (t.feature - feat_lo) as usize;
        let c = t.cell as usize;
        let row_r = r.row(c);
        let mut row = sketch.row_mut(f_local);
        for s in 0..sketch_dim {
            row[s] += t.count * row_r[s];
        }
    }
    sketch
}

/// Produce `[N, S]` sketch where row `c` is the sum, over non-zero
/// `(f, c, x)` triplets, of `x * R[f, :]`.
fn sketch_cells_via_random_projection(
    triplets: &[Triplet],
    n_cells: usize,
    n_features: usize,
    sketch_dim: usize,
    seed: u64,
) -> DMatrix<f32> {
    let r = gaussian_matrix(n_features, sketch_dim, seed);
    let mut sketch = DMatrix::<f32>::zeros(n_cells, sketch_dim);
    for t in triplets {
        let c = t.cell as usize;
        let f = t.feature as usize;
        let row_r = r.row(f);
        let mut row = sketch.row_mut(c);
        for s in 0..sketch_dim {
            row[s] += t.count * row_r[s];
        }
    }
    sketch
}

fn gaussian_matrix(rows: usize, cols: usize, seed: u64) -> DMatrix<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    DMatrix::from_fn(rows, cols, |_, _| {
        let v: f32 = rng.sample(StandardNormal);
        v
    })
}
