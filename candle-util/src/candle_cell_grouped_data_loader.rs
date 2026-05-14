//! Cell-grouped data loader for the hierarchical cell→PB pooling topic
//! model (`senna cell-embedded-topic`).
//!
//! Where [`crate::candle_indexed_data_loader::IndexedInMemoryData`] treats
//! each pseudobulk (PB) sample as a single dense atom, this loader keeps
//! the **cell → PB membership** and feeds the encoder the genuinely sparse
//! single-cell atoms:
//!
//!   - **Foreground (FG)** — each PB's member cells, packed `[M, K_fg]`
//!     (M = total member cells in the minibatch) plus a `cell_to_pb [M]`
//!     segment map into `[0, N)`. The encoder pools cell embeddings within
//!     each PB (two-level gene→cell→PB pool).
//!   - **Background (BG)** — the PB-level `μ_residual` profile (or the
//!     observed PB profile when no batch residual was fit), single-level
//!     top-K, packed `[N, K_bg]`.
//!   - **Decoder** — top-K of the PB-level `target` profile, identical
//!     union + scatter-pos packing as the indexed loader.
//!
//! Cell extraction is level-independent: the caller extracts the per-cell
//! top-K samples + library-size factors once and shares them across every
//! PB level via `Arc`.

use crate::candle_data_loader_util::Minibatches;
use crate::candle_indexed_data_loader::{
    build_indexed_samples, build_union_and_scatter_pos, compute_log_selection_freq,
    gather_per_feature_at_indices, labeled_bar, pack_indices_values, slice_log_q_at_union,
    sum_sample_values, touched_rho_tensor, IndexedSample,
};

use candle_core::{Device, Tensor};
use indicatif::ParallelProgressIterator;
use matrix_util::traits::CandleDataLoaderOps;
use rayon::prelude::*;
use std::sync::Arc;

////////////////////////////////////////////////////////////////////////
// Packed minibatch
////////////////////////////////////////////////////////////////////////

/// Packed cell-grouped minibatch.
///
/// FG side is flattened across the `N` PBs of the minibatch into `M`
/// member cells; `cell_to_pb` is the segment map back into `[0, N)`.
/// BG and decoder sides are `[N, K]`-packed exactly like the indexed
/// loader. Padding (fewer than `K` features) is filled with `(idx=0,
/// val=0)`; the encoder masks padded FG/BG slots by `value > 0`.
pub struct CellGroupedMinibatchData {
    /// [M, K_fg] u32 in [0, D) — per-cell encoder feature ids
    pub fg_indices: Tensor,
    /// [M, K_fg] f32 — per-cell raw values at those ids
    pub fg_values: Tensor,
    /// [M, 1] f32 — per-cell library-size factor `s_c = Σ_g y_cg`
    pub fg_size_factor: Tensor,
    /// [M, K_fg] f32 — per-gene NB-Fisher weight gathered at `fg_indices`
    pub fg_fisher: Tensor,
    /// [M] u32 in [0, N) — segment map: member cell → its PB row
    pub cell_to_pb: Tensor,
    /// [N, K_bg] u32 in [0, D) — PB-level background feature ids
    pub bg_indices: Tensor,
    /// [N, K_bg] f32 — background values (μ_residual / observed PB profile)
    pub bg_values: Tensor,
    /// [N, K_bg] f32 — per-gene NB-Fisher weight gathered at `bg_indices`
    pub bg_fisher: Tensor,
    /// [S] u32 in [0, D) — sorted union of decoder per-PB top-K ids
    pub output_union_indices: Tensor,
    /// [N, K_out] u32 in [0, S) — per-PB positions of values in the union
    pub output_scatter_pos: Tensor,
    /// [N, K_out] f32 — decoder feature values (per-PB index order)
    pub output_values: Tensor,
    /// [N, K_out] f32 — per-gene NB-Fisher weight at the decoder ids
    pub output_values_weight: Tensor,
    /// [1, S] f32 — log selection frequency at union positions
    pub output_log_q_s: Tensor,
    /// [T] u32 — sorted-unique union of every feature id this minibatch
    /// touches (FG cell ids ∪ BG ids ∪ decoder union) — the rows of the
    /// shared ρ `[D, H]` table that receive a nonzero gradient.
    pub touched_rho_indices: Tensor,
}

impl CellGroupedMinibatchData {
    /// Upload every tensor field to `dev`. Cached minibatches are built
    /// host-side; the training loop calls this once per minibatch.
    pub fn to_device(&self, dev: &Device) -> anyhow::Result<CellGroupedMinibatchData> {
        Ok(CellGroupedMinibatchData {
            fg_indices: self.fg_indices.to_device(dev)?,
            fg_values: self.fg_values.to_device(dev)?,
            fg_size_factor: self.fg_size_factor.to_device(dev)?,
            fg_fisher: self.fg_fisher.to_device(dev)?,
            cell_to_pb: self.cell_to_pb.to_device(dev)?,
            bg_indices: self.bg_indices.to_device(dev)?,
            bg_values: self.bg_values.to_device(dev)?,
            bg_fisher: self.bg_fisher.to_device(dev)?,
            output_union_indices: self.output_union_indices.to_device(dev)?,
            output_scatter_pos: self.output_scatter_pos.to_device(dev)?,
            output_values: self.output_values.to_device(dev)?,
            output_values_weight: self.output_values_weight.to_device(dev)?,
            output_log_q_s: self.output_log_q_s.to_device(dev)?,
            touched_rho_indices: self.touched_rho_indices.to_device(dev)?,
        })
    }
}

////////////////////////////////////////////////////////////////////////
// In-memory loader (one per PB level)
////////////////////////////////////////////////////////////////////////

/// Per-level cell-grouped in-memory loader. Minibatches iterate over PB
/// ids; each draws its member cells (FG), the PB background (BG), and the
/// PB decoder target.
pub struct CellGroupedInMemoryData {
    /// Per-cell top-K samples — shared across all levels (`Arc`).
    cell_samples: Arc<Vec<IndexedSample>>,
    /// Per-cell library-size factor `s_c` (floored ≥ 1) — shared (`Arc`).
    cell_size_factor: Arc<Vec<f32>>,
    /// `pb_to_cells[pb]` = member cell ids of PB `pb` at this level.
    pb_to_cells: Vec<Vec<usize>>,
    /// Per-PB background top-K (μ_residual / observed PB profile).
    bg_samples: Vec<IndexedSample>,
    /// Per-PB decoder-target top-K.
    output_samples: Vec<IndexedSample>,
    /// Per-feature log selection frequency from `output_samples`.
    output_log_q: Vec<f32>,
    /// Per-feature NB-Fisher weight (length = D).
    feature_fisher_weights: Vec<f32>,
    n_features: usize,
    fg_context_size: usize,
    bg_context_size: usize,
    dec_context_size: usize,
    total_output_count: f32,
    minibatches: Minibatches,
    cached_batches: Vec<CellGroupedMinibatchData>,
}

/// Constructor args for one PB level of [`CellGroupedInMemoryData`].
pub struct CellGroupedArgs<'a, D>
where
    D: CandleDataLoaderOps + Sync,
{
    /// Per-cell top-K samples, shared across levels.
    pub cell_samples: Arc<Vec<IndexedSample>>,
    /// Per-cell library-size factor `s_c` (floored ≥ 1), shared.
    pub cell_size_factor: Arc<Vec<f32>>,
    /// `cell_to_pb[c]` = PB id of cell `c` at this level.
    pub cell_to_pb: &'a [usize],
    /// PB-level background source `[N_pb, D]` (μ_residual or observed).
    pub bg_source: &'a D,
    /// PB-level decoder-target source `[N_pb, D]`.
    pub target_source: &'a D,
    pub n_features: usize,
    pub fg_context_size: usize,
    pub bg_context_size: usize,
    pub dec_context_size: usize,
    /// Per-feature weights used to score top-K candidates (BG + decoder).
    pub shortlist_weights: &'a [f32],
    /// Per-feature NB-Fisher weight gathered into the FG/BG/decoder packs.
    pub feature_fisher_weights: &'a [f32],
}

impl CellGroupedInMemoryData {
    /// Build one PB level. Inverts `cell_to_pb` into `pb_to_cells`, then
    /// top-K-selects the BG and decoder-target profiles.
    pub fn new<D>(args: CellGroupedArgs<D>) -> anyhow::Result<Self>
    where
        D: CandleDataLoaderOps + Sync,
    {
        let (n_pb, _) = args.bg_source.data_shape();
        let (n_pb_t, _) = args.target_source.data_shape();
        anyhow::ensure!(
            n_pb == n_pb_t,
            "bg/target PB count mismatch: {n_pb} vs {n_pb_t}"
        );
        anyhow::ensure!(
            args.cell_to_pb.len() == args.cell_samples.len(),
            "cell_to_pb length {} != n_cells {}",
            args.cell_to_pb.len(),
            args.cell_samples.len(),
        );
        anyhow::ensure!(
            args.feature_fisher_weights.len() == args.n_features,
            "feature_fisher_weights length {} != n_features {}",
            args.feature_fisher_weights.len(),
            args.n_features,
        );

        // Invert cell→pb into pb→cells.
        let mut pb_to_cells: Vec<Vec<usize>> = vec![Vec::new(); n_pb];
        for (cell, &pb) in args.cell_to_pb.iter().enumerate() {
            anyhow::ensure!(pb < n_pb, "cell {cell} maps to pb {pb} >= {n_pb}");
            pb_to_cells[pb].push(cell);
        }

        let bg_context_size = args.bg_context_size.min(args.n_features);
        let dec_context_size = args.dec_context_size.min(args.n_features);

        let bg_samples = build_indexed_samples(
            args.bg_source,
            n_pb,
            bg_context_size,
            args.shortlist_weights,
            "Top-K (bg)",
        );
        let output_samples = build_indexed_samples(
            args.target_source,
            n_pb,
            dec_context_size,
            args.shortlist_weights,
            "Top-K (decoder)",
        );

        let output_log_q = compute_log_selection_freq(&output_samples, args.n_features);
        let total_output_count = sum_sample_values(&output_samples);

        Ok(CellGroupedInMemoryData {
            cell_samples: args.cell_samples,
            cell_size_factor: args.cell_size_factor,
            pb_to_cells,
            bg_samples,
            output_samples,
            output_log_q,
            feature_fisher_weights: args.feature_fisher_weights.to_vec(),
            n_features: args.n_features,
            fg_context_size: args.fg_context_size.min(args.n_features),
            bg_context_size,
            dec_context_size,
            total_output_count,
            minibatches: Minibatches {
                samples: (0..n_pb).collect(),
                chunks: vec![],
            },
            cached_batches: vec![],
        })
    }

    pub fn shuffle_minibatch(&mut self, batch_size: usize) {
        self.minibatches.shuffle_minibatch(batch_size);
        self.cached_batches.clear();
    }

    /// Pre-build all minibatch tensors for the current shuffle order
    /// (host-side; the training loop uploads each on demand).
    pub fn precompute_all_minibatches(&mut self) -> anyhow::Result<()> {
        let n_chunks = self.minibatches.chunks.len() as u64;
        let prog_bar = labeled_bar("Minibatch precompute", n_chunks);
        self.cached_batches = self
            .minibatches
            .chunks
            .par_iter()
            .progress_with(prog_bar.clone())
            .map(|pb_indices| self.build_minibatch(pb_indices, &Device::Cpu))
            .collect::<anyhow::Result<Vec<_>>>()?;
        prog_bar.finish_and_clear();
        Ok(())
    }

    /// Retrieve a pre-computed minibatch. Panics if
    /// `precompute_all_minibatches` was not called after the last shuffle.
    pub fn minibatch_cached(&self, batch_idx: usize) -> &CellGroupedMinibatchData {
        &self.cached_batches[batch_idx]
    }

    pub fn num_data(&self) -> usize {
        self.minibatches.samples.len()
    }

    pub fn num_minibatch(&self) -> usize {
        self.minibatches.chunks.len()
    }

    pub fn fg_context_size(&self) -> usize {
        self.fg_context_size
    }

    pub fn bg_context_size(&self) -> usize {
        self.bg_context_size
    }

    pub fn dec_context_size(&self) -> usize {
        self.dec_context_size
    }

    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Sum of all decoder-side values across every PB — the per-epoch
    /// count total, invariant to minibatch shuffling.
    pub fn total_output_count(&self) -> f32 {
        self.total_output_count
    }

    /// Build a packed minibatch from a list of PB ids.
    fn build_minibatch(
        &self,
        pb_indices: &[usize],
        dev: &Device,
    ) -> anyhow::Result<CellGroupedMinibatchData> {
        let k_fg = self.fg_context_size;
        let k_bg = self.bg_context_size;
        let k_out = self.dec_context_size;

        // FG: flatten every batch PB's member cells into [M] and record
        // the cell → PB-row segment map.
        let mut fg_cell_ids: Vec<usize> = Vec::new();
        let mut cell_to_pb: Vec<u32> = Vec::new();
        for (row, &pb) in pb_indices.iter().enumerate() {
            for &c in &self.pb_to_cells[pb] {
                fg_cell_ids.push(c);
                cell_to_pb.push(row as u32);
            }
        }
        let m = fg_cell_ids.len();

        let (fg_indices, fg_values) =
            pack_indices_values(&self.cell_samples, &fg_cell_ids, k_fg, dev)?;
        let fg_fisher = gather_per_feature_at_indices(
            &self.cell_samples,
            &fg_cell_ids,
            &self.feature_fisher_weights,
            k_fg,
            dev,
        )?;
        let sf_buf: Vec<f32> = fg_cell_ids
            .iter()
            .map(|&c| self.cell_size_factor[c])
            .collect();
        let fg_size_factor = Tensor::from_vec(sf_buf, (m, 1), dev)?;
        let cell_to_pb_t = Tensor::from_vec(cell_to_pb, (m,), dev)?;

        // BG: PB-level single-level top-K pack.
        let (bg_indices, bg_values) =
            pack_indices_values(&self.bg_samples, pb_indices, k_bg, dev)?;
        let bg_fisher = gather_per_feature_at_indices(
            &self.bg_samples,
            pb_indices,
            &self.feature_fisher_weights,
            k_bg,
            dev,
        )?;

        // Decoder: union + per-PB scatter positions + values.
        let (output_union_indices, output_scatter_pos, output_union_vec) =
            build_union_and_scatter_pos(&self.output_samples, pb_indices, self.n_features, k_out, dev)?;
        let (_, output_values) =
            pack_indices_values(&self.output_samples, pb_indices, k_out, dev)?;
        let output_values_weight = gather_per_feature_at_indices(
            &self.output_samples,
            pb_indices,
            &self.feature_fisher_weights,
            k_out,
            dev,
        )?;
        let output_log_q_s = slice_log_q_at_union(&self.output_log_q, &output_union_vec, dev)?;

        // Touched ρ rows: FG cell ids ∪ BG ids ∪ decoder union.
        let touched_rho_indices = touched_rho_tensor(
            &output_union_vec,
            fg_cell_ids
                .iter()
                .flat_map(|&c| self.cell_samples[c].indices.iter().copied())
                .chain(
                    pb_indices
                        .iter()
                        .flat_map(|&pb| self.bg_samples[pb].indices.iter().copied()),
                ),
            dev,
        )?;

        Ok(CellGroupedMinibatchData {
            fg_indices,
            fg_values,
            fg_size_factor,
            fg_fisher,
            cell_to_pb: cell_to_pb_t,
            bg_indices,
            bg_values,
            bg_fisher,
            output_union_indices,
            output_scatter_pos,
            output_values,
            output_values_weight,
            output_log_q_s,
            touched_rho_indices,
        })
    }

    /// Build a minibatch from an ordered (non-shuffled) PB-id range.
    /// Used by tests.
    #[cfg(test)]
    fn minibatch_ordered(
        &self,
        lb: usize,
        ub: usize,
        dev: &Device,
    ) -> anyhow::Result<CellGroupedMinibatchData> {
        let pb_indices: Vec<usize> = (lb..ub).collect();
        self.build_minibatch(&pb_indices, dev)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    /// 2-PB / 4-cell toy: verify the `cell_to_pb` segment map and that the
    /// FG `index_add` segment pool reproduces a hand-summed `[N, H]`.
    #[test]
    fn test_cell_grouped_segment_pool() {
        let dev = Device::Cpu;
        let n_features = 5;
        let h = 3;

        // 4 cells, each a sparse top-K sample over 5 features.
        let cell_samples = Arc::new(vec![
            IndexedSample {
                indices: vec![0, 2],
                values: vec![4.0, 9.0],
            },
            IndexedSample {
                indices: vec![1, 3],
                values: vec![1.0, 1.0],
            },
            IndexedSample {
                indices: vec![0, 4],
                values: vec![16.0, 25.0],
            },
            IndexedSample {
                indices: vec![2],
                values: vec![1.0],
            },
        ]);
        // Library-size factors (floored ≥ 1).
        let cell_size_factor = Arc::new(vec![13.0f32, 2.0, 41.0, 1.0]);
        // PB 0 = cells {0,1}; PB 1 = cells {2,3}.
        let cell_to_pb = vec![0usize, 0, 1, 1];

        // PB-level BG / target sources [N_pb=2, D=5].
        let bg = DMatrix::<f32>::from_row_slice(
            2,
            5,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        );
        let target = bg.clone();
        let fisher = vec![1.0f32; n_features];

        let loader = CellGroupedInMemoryData::new(CellGroupedArgs {
            cell_samples: cell_samples.clone(),
            cell_size_factor: cell_size_factor.clone(),
            cell_to_pb: &cell_to_pb,
            bg_source: &bg,
            target_source: &target,
            n_features,
            fg_context_size: 2,
            bg_context_size: 3,
            dec_context_size: 3,
            shortlist_weights: &fisher,
            feature_fisher_weights: &fisher,
        })
        .unwrap();

        let mb = loader.minibatch_ordered(0, 2, &dev).unwrap();

        // M = 2 + 2 = 4 member cells; segment map [0,0,1,1].
        let c2p: Vec<u32> = mb.cell_to_pb.to_vec1().unwrap();
        assert_eq!(c2p, vec![0, 0, 1, 1]);
        assert_eq!(mb.fg_indices.dims(), &[4, 2]);
        assert_eq!(mb.fg_size_factor.dims(), &[4, 1]);
        assert_eq!(mb.bg_indices.dims(), &[2, 3]);

        // Hand pool: per-cell value-weighted embedding (raw values, no
        // Anscombe / size-factor — just the index_add segment sum), then
        // segment-summed by PB. Mirrors the encoder's index_add step.
        let rho: Vec<Vec<f32>> = (0..n_features)
            .map(|d| (0..h).map(|j| (d * h + j) as f32).collect())
            .collect();
        let fg_idx: Vec<Vec<u32>> = mb.fg_indices.to_vec2().unwrap();
        let fg_val: Vec<Vec<f32>> = mb.fg_values.to_vec2().unwrap();
        let mut want = vec![vec![0.0f32; h]; 2];
        for cell in 0..4 {
            let pb = c2p[cell] as usize;
            for k in 0..2 {
                let feat = fg_idx[cell][k] as usize;
                let v = fg_val[cell][k];
                for j in 0..h {
                    want[pb][j] += v * rho[feat][j];
                }
            }
        }

        // index_add reproduction on the host tensors.
        let rho_flat: Vec<f32> = rho.iter().flatten().copied().collect();
        let rho_t = Tensor::from_vec(rho_flat, (n_features, h), &dev).unwrap();
        let e = rho_t
            .index_select(&mb.fg_indices.flatten_all().unwrap(), 0)
            .unwrap()
            .reshape((4, 2, h))
            .unwrap();
        let h_m = e
            .broadcast_mul(&mb.fg_values.unsqueeze(2).unwrap())
            .unwrap()
            .sum(1)
            .unwrap();
        let pooled = Tensor::zeros((2, h), candle_core::DType::F32, &dev)
            .unwrap()
            .index_add(&mb.cell_to_pb, &h_m, 0)
            .unwrap();
        let got: Vec<Vec<f32>> = pooled.to_vec2().unwrap();
        for pb in 0..2 {
            for j in 0..h {
                assert!(
                    (got[pb][j] - want[pb][j]).abs() < 1e-4,
                    "pb {pb} dim {j}: got {} want {}",
                    got[pb][j],
                    want[pb][j]
                );
            }
        }
    }
}
