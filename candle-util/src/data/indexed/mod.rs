//! Indexed (top-K) data loader for the senna topic models.
//!
//! Packs each row's top-K features into `[N, K]` host buffers, never an
//! `[N, D]` dense matrix. This is the **encoder's** bounded view of a row;
//! what a decoder scores is the consumer's choice over the whole feature
//! axis, so the loader carries `row_ids` and stops there.
//!
//! Module layout (split 2026-05-15 from a single 1570-line file):
//! - [`types`] — public data types (`IndexedSample`, `IndexedMinibatchData`,
//!   `IndexedInMemoryArgs`).
//! - [`top_k`] — weighted top-K selection (`top_k_indices_weighted`,
//!   `csc_columns_to_indexed_samples`, `build_indexed_samples`).
//! - [`pack`] — parallel `[N, K]` pack/gather helpers.
//! - [`graph_adj`] — feature-feature `GraphCsr`, per-cell sub-adjacency
//!   cache, and `SparseEdgeBatch` builders for the γ-gated GCN block.

use crate::data::loader_util::Minibatches;

use candle_core::{Device, Tensor};
use indicatif::{ParallelProgressIterator, ProgressBar};
use matrix_util::traits::CandleDataLoaderOps;
use rayon::prelude::*;
use std::sync::Arc;

pub mod graph_adj;
pub mod pack;
pub mod splice_tracks;
pub mod top_k;
pub mod types;

pub use graph_adj::{build_sparse_edges_from_tensor, GraphCsr, SparseEdgeBatch};
pub use pack::gather_per_feature_at_indices;
pub use splice_tracks::{
    gem_samples_from_csc, top_k_genes_from_row, GemIndexedArgs, GemIndexedData, GemMinibatchData,
    GemSample, GeneTrackMap,
};
pub use top_k::{csc_columns_to_indexed_samples, top_k_indices_weighted};
pub use types::{IndexedInMemoryArgs, IndexedMinibatchData, IndexedSample};

// Crate-public re-exports.
pub use pack::pack_indices_values;
pub use top_k::build_indexed_samples;

use graph_adj::{scatter_sparse_edges, SubAdjCache};
use pack::pack_null_at_indices;

///////////////////////////////////////////////////////////////////////////
// Progress bar helper (used here and externally by cell-grouped loader) //
///////////////////////////////////////////////////////////////////////////

/// A bounded progress bar in the **canonical workspace style** (see
/// [`matrix_util::progress::new_progress_bar`]): `[elapsed] bar pos/len (eta) {msg}`,
/// cyan/blue, and — crucially — registered with the shared `MULTI_PROGRESS` so `-v`
/// log output interleaves cleanly above it. `label` is the initial `{msg}` (e.g.
/// "Epochs", "Null rows"); the epoch trainers overwrite it each step with a live metric
/// (`prog_bar.set_message`), matching `senna bge`. Delegating here keeps every
/// candle-util bar on ONE style and ONE bridged `MultiProgress` — a locally-styled
/// `ProgressBar::new` spawns a SECOND, unbridged bar that corrupts log output under
/// `-v` (see the `matrix-util::progress` module doc).
#[must_use]
pub fn labeled_bar(label: &str, len: u64) -> ProgressBar {
    matrix_util::progress::new_progress_bar(len).with_message(label.to_string())
}

/////////////////////////////////////////////////////////////////////
// IndexedInMemoryData — minibatch source for indexed-topic models //
/////////////////////////////////////////////////////////////////////

/// Indexed minibatch source.
///
/// Built via [`IndexedInMemoryData::from_dense`] from a `CandleDataLoaderOps`
/// source. Call [`shuffle_minibatch`] then [`precompute_all_minibatches`]
/// once per epoch; the training loop retrieves prebuilt batches via
/// [`minibatch_cached`].
///
/// [`shuffle_minibatch`]: IndexedInMemoryData::shuffle_minibatch
/// [`precompute_all_minibatches`]: IndexedInMemoryData::precompute_all_minibatches
/// [`minibatch_cached`]: IndexedInMemoryData::minibatch_cached
pub struct IndexedInMemoryData {
    input_samples: Vec<IndexedSample>,
    input_null_rows: Option<Vec<Vec<f32>>>,
    n_input_features: usize,
    input_context_size: usize,
    /// Per-feature mean expression rate `μ_d` (encoder side) gathered
    /// into `input_values_mean [N, K]` at minibatch build time.
    input_mean: Option<Vec<f32>>,
    /// Optional feature-feature graph. When set, the indexed encoder's
    /// GCN diffusion block consumes a per-minibatch [`SparseEdgeBatch`]
    /// scattered straight from `sub_adj_cache`.
    graph_csr: Option<Arc<GraphCsr>>,
    /// Pre-computed and pre-normalised per-cell sub-adjacency triples
    /// in slot-space (see [`graph_adj`] module docs).
    sub_adj_cache: Option<SubAdjCache>,
    minibatches: Minibatches,
    cached_batches: Vec<IndexedMinibatchData>,
}

impl IndexedInMemoryData {
    /// Build indexed data from a dense matrix.
    pub fn from_dense<D>(args: IndexedInMemoryArgs<D>) -> anyhow::Result<Self>
    where
        D: CandleDataLoaderOps + Sync,
    {
        let (n_samples, n_input_features) = args.input.data_shape();
        let input_context_size = args.input_context_size.min(n_input_features);

        let input_samples = build_indexed_samples(
            args.input,
            n_samples,
            input_context_size,
            args.input_shortlist_weights,
            "Top-K (encoder)",
        );

        let null_rows: Option<Vec<Vec<f32>>> = args.input_null.map(|d| {
            let (n, _) = d.data_shape();
            let prog_bar = labeled_bar("Null rows", n as u64);
            let rows: Vec<Vec<f32>> = (0..n)
                .into_par_iter()
                .progress_with(prog_bar.clone())
                .map(|i| d.row_to_f32_vec(i))
                .collect();
            prog_bar.finish_and_clear();
            rows
        });

        let rows: Vec<usize> = (0..n_samples).collect();

        let input_mean = args.input_mean.map(|s| s.to_vec());
        if let Some(ref b) = input_mean {
            anyhow::ensure!(
                b.len() == n_input_features,
                "input_mean length {} != n_input_features {}",
                b.len(),
                n_input_features
            );
        }

        Ok(IndexedInMemoryData {
            input_samples,
            input_null_rows: null_rows,
            n_input_features,
            input_context_size,
            input_mean,
            graph_csr: None,
            sub_adj_cache: None,
            minibatches: Minibatches {
                samples: rows,
                chunks: vec![],
            },
            cached_batches: vec![],
        })
    }

    /// Attach a feature-feature graph to this loader. Builds a one-shot
    /// per-cell sub-adjacency cache so subsequent epoch shuffles only
    /// scatter cached triples instead of re-walking the CSR per cell
    /// per minibatch. Pass `None` to detach and drop the cache.
    pub fn set_graph_csr(&mut self, graph_csr: Option<Arc<GraphCsr>>) {
        if let Some(ref g) = graph_csr {
            assert_eq!(
                g.n_features, self.n_input_features,
                "feature graph has {} features but encoder input has {}",
                g.n_features, self.n_input_features
            );
            let cache =
                SubAdjCache::build(&self.input_samples, g.as_ref(), self.input_context_size);
            let mb_triples = cache.triples.len();
            let mb_bytes = mb_triples * std::mem::size_of::<(u16, u16, f32)>();
            log::info!(
                "built per-cell adjacency cache: {} cells, {} triples (~{} MB)",
                self.input_samples.len(),
                mb_triples,
                mb_bytes >> 20,
            );
            self.sub_adj_cache = Some(cache);
        } else {
            self.sub_adj_cache = None;
        }
        self.graph_csr = graph_csr;
        self.cached_batches.clear();
    }

    /// Whether a feature graph is attached.
    pub fn has_graph(&self) -> bool {
        self.graph_csr.is_some()
    }

    pub fn shuffle_minibatch(&mut self, batch_size: usize) {
        self.minibatches.shuffle_minibatch(batch_size);
        self.cached_batches.clear();
    }

    /// Pre-build all minibatch tensors for the current shuffle order.
    ///
    /// Cached batches are built host-side (`Device::Cpu`); the consumer
    /// uploads each minibatch on demand via
    /// [`IndexedMinibatchData::to_device`].
    pub fn precompute_all_minibatches(&mut self) -> anyhow::Result<()> {
        let n_chunks = self.minibatches.chunks.len() as u64;
        let prog_bar = labeled_bar("Minibatch precompute", n_chunks);
        self.cached_batches = self
            .minibatches
            .chunks
            .par_iter()
            .progress_with(prog_bar.clone())
            .map(|sample_indices| self.build_minibatch(sample_indices, &Device::Cpu))
            .collect::<anyhow::Result<Vec<_>>>()?;
        prog_bar.finish_and_clear();
        Ok(())
    }

    /// Retrieve a pre-computed minibatch. Panics if
    /// [`precompute_all_minibatches`] was not called after the last
    /// [`shuffle_minibatch`].
    pub fn minibatch_cached(&self, batch_idx: usize) -> &IndexedMinibatchData {
        &self.cached_batches[batch_idx]
    }

    pub fn num_data(&self) -> usize {
        self.minibatches.samples.len()
    }

    pub fn num_minibatch(&self) -> usize {
        self.minibatches.chunks.len()
    }

    pub fn input_context_size(&self) -> usize {
        self.input_context_size
    }

    pub fn n_input_features(&self) -> usize {
        self.n_input_features
    }

    /// Build a packed minibatch: `(indices, values)` straight into
    /// `[N, K_in]`, plus the source `row_ids` so a consumer can index
    /// per-row state it holds itself. Nothing is materialised at `[N, D]`
    /// shape on host or device.
    fn build_minibatch(
        &self,
        sample_indices: &[usize],
        target_device: &Device,
    ) -> anyhow::Result<IndexedMinibatchData> {
        let k_in = self.input_context_size;

        let row_ids = Tensor::from_vec(
            sample_indices.iter().map(|&i| i as u32).collect::<Vec<_>>(),
            (sample_indices.len(),),
            target_device,
        )?;
        let (input_indices, input_values) =
            pack_indices_values(&self.input_samples, sample_indices, k_in, target_device)?;
        let input_values_null = match self.input_null_rows.as_ref() {
            Some(rows) => Some(pack_null_at_indices(
                &self.input_samples,
                rows,
                sample_indices,
                k_in,
                target_device,
            )?),
            None => None,
        };
        let input_values_mean = match self.input_mean.as_ref() {
            Some(b) => Some(gather_per_feature_at_indices(
                &self.input_samples,
                sample_indices,
                b,
                k_in,
                target_device,
            )?),
            None => None,
        };

        Ok(IndexedMinibatchData {
            row_ids,
            input_indices,
            input_values,
            input_values_null,
            input_values_mean,
        })
    }

    /// Lazy on-demand build of the sparse per-minibatch edge batch
    /// consumed by the γ-gated GCN block. Returns `Ok(None)` when no
    /// feature graph is attached. Scatters pre-normalised per-cell
    /// triples directly onto `target_device`.
    pub fn minibatch_sparse_edges(
        &self,
        batch_idx: usize,
        target_device: &Device,
    ) -> anyhow::Result<Option<SparseEdgeBatch>> {
        let Some(cache) = self.sub_adj_cache.as_ref() else {
            return Ok(None);
        };
        let sample_indices = self.minibatches.chunks.get(batch_idx).ok_or_else(|| {
            anyhow::anyhow!(
                "invalid minibatch index {batch_idx} vs total {}",
                self.minibatches.chunks.len()
            )
        })?;
        let edges = scatter_sparse_edges(
            cache,
            sample_indices,
            self.input_context_size,
            target_device,
        )?;
        Ok(Some(edges))
    }

    /// Build an indexed minibatch from the shuffled indices.
    pub fn minibatch_shuffled(
        &self,
        batch_idx: usize,
        target_device: &Device,
    ) -> anyhow::Result<IndexedMinibatchData> {
        let sample_indices = self.minibatches.chunks.get(batch_idx).ok_or_else(|| {
            anyhow::anyhow!(
                "invalid batch index {} vs total {}",
                batch_idx,
                self.minibatches.chunks.len()
            )
        })?;

        self.build_minibatch(sample_indices, target_device)
    }

    /// Build an indexed minibatch of exactly `n` rows, cycling over the
    /// data when `n` exceeds the row count. Training pads every batch to
    /// the full minibatch size by bootstrap (`bootstrap_indices`), so a
    /// memory probe must measure full-size batches too; a truncated
    /// probe batch would under-measure the per-row cost.
    pub fn minibatch_cycled(
        &self,
        n: usize,
        target_device: &Device,
    ) -> anyhow::Result<IndexedMinibatchData> {
        let n_data = self.num_data();
        anyhow::ensure!(n_data > 0, "minibatch_cycled on an empty loader");
        let sample_indices: Vec<usize> = (0..n).map(|i| i % n_data).collect();
        self.build_minibatch(&sample_indices, target_device)
    }

    /// Build an indexed minibatch from an ordered (non-shuffled) range.
    pub fn minibatch_ordered(
        &self,
        lb: usize,
        ub: usize,
        target_device: &Device,
    ) -> anyhow::Result<IndexedMinibatchData> {
        let sample_indices: Vec<usize> = (lb..ub).collect();
        self.build_minibatch(&sample_indices, target_device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn test_indexed_from_dense() {
        let data = DMatrix::<f32>::from_row_slice(
            4,
            6,
            &[
                0.1, 0.5, 0.3, 0.9, 0.2, 0.7, // sample 0: top-3 = {1,3,5}
                0.8, 0.1, 0.6, 0.2, 0.9, 0.3, // sample 1: top-3 = {0,2,4}
                0.3, 0.7, 0.1, 0.4, 0.6, 0.5, // sample 2: top-3 = {1,4,5}
                0.2, 0.3, 0.8, 0.1, 0.5, 0.9, // sample 3: top-3 = {2,4,5}
            ],
        );

        let w = vec![1.0f32; 6];
        let args = IndexedInMemoryArgs {
            input: &data,
            input_null: None,
            input_context_size: 3,
            input_shortlist_weights: &w,
            input_mean: None,
        };

        let indexed = IndexedInMemoryData::from_dense(args).unwrap();
        assert_eq!(indexed.num_data(), 4);
        assert_eq!(indexed.n_input_features(), 6);
        assert_eq!(indexed.input_context_size(), 3);

        assert_eq!(indexed.input_samples[0].indices, vec![1, 3, 5]);
        assert_eq!(indexed.input_samples[0].values, vec![0.5, 0.9, 0.7]);

        assert_eq!(indexed.input_samples[1].indices, vec![0, 2, 4]);
        assert_eq!(indexed.input_samples[1].values, vec![0.8, 0.6, 0.9]);
    }

    fn three_row_loader(context: usize) -> IndexedInMemoryData {
        let data = DMatrix::<f32>::from_row_slice(
            3,
            6,
            &[
                0.1, 0.5, 0.3, 0.9, 0.2, 0.7, // top-2 = {3,5}
                0.8, 0.1, 0.6, 0.2, 0.9, 0.3, // top-2 = {0,4}
                0.3, 0.7, 0.1, 0.4, 0.6, 0.5, // top-2 = {1,4}
            ],
        );
        let w = vec![1.0f32; 6];
        IndexedInMemoryData::from_dense(IndexedInMemoryArgs {
            input: &data,
            input_null: None,
            input_context_size: context,
            input_shortlist_weights: &w,
            input_mean: None,
        })
        .unwrap()
    }

    #[test]
    fn test_packed_minibatch_shapes() {
        let indexed = three_row_loader(2);
        let mb = indexed.minibatch_ordered(0, 3, &Device::Cpu).unwrap();

        assert_eq!(mb.input_indices.dims(), &[3, 2]);
        assert_eq!(mb.input_values.dims(), &[3, 2]);
        assert_eq!(mb.row_ids.dims(), &[3]);

        let in_idx: Vec<Vec<u32>> = mb.input_indices.to_vec2().unwrap();
        let in_val: Vec<Vec<f32>> = mb.input_values.to_vec2().unwrap();
        assert_eq!(in_idx[0], vec![3, 5]);
        assert!((in_val[0][0] - 0.9).abs() < 1e-6);
        assert!((in_val[0][1] - 0.7).abs() < 1e-6);
    }

    /// The minibatch must say which source row each of its rows came from:
    /// that is how a consumer indexes per-row state it owns (a target table,
    /// a free latent) without re-deriving the shuffle.
    #[test]
    fn minibatch_carries_its_row_ids() {
        let indexed = three_row_loader(2);

        let ordered = indexed.minibatch_ordered(1, 3, &Device::Cpu).unwrap();
        let ids: Vec<u32> = ordered.row_ids.to_vec1().unwrap();
        assert_eq!(ids, vec![1, 2]);

        // Cycling repeats the data in order, and `row_ids` says so.
        let cycled = indexed.minibatch_cycled(7, &Device::Cpu).unwrap();
        let ids: Vec<u32> = cycled.row_ids.to_vec1().unwrap();
        assert_eq!(ids, vec![0, 1, 2, 0, 1, 2, 0]);

        // Duplicated rows carry the same id and the same content.
        let vals: Vec<Vec<f32>> = cycled.input_values.to_vec2().unwrap();
        assert_eq!(vals[0], vals[3]);
    }
}
