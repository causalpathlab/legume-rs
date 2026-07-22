//! Indexed (top-K) data loader for the senna topic models.
//!
//! Encoder side is fully packed `[N, K_in]` (never `[N, S_in]`); decoder
//! side builds a per-minibatch union of feature ids and per-cell scatter
//! positions so the importance-weighted conditional softmax never has
//! to materialise an `[N, S]` dense matrix.
//!
//! Module layout (split 2026-05-15 from a single 1570-line file):
//! - [`types`] — public data types (`IndexedSample`, `IndexedMinibatchData`,
//!   `IndexedInMemoryArgs`).
//! - [`top_k`] — weighted top-K selection (`top_k_indices_weighted`,
//!   `csc_columns_to_indexed_samples`, `build_indexed_samples`).
//! - [`pack`] — parallel `[N, K]` pack/gather helpers.
//! - [`union_scatter`] — decoder-side union construction with a
//!   thread-local generation-tagged lookup.
//! - [`graph_adj`] — feature-feature `GraphCsr`, per-cell sub-adjacency
//!   cache, and `SparseEdgeBatch` builders for the γ-gated GCN block.
//! - [`stats`] — per-feature / per-sample aggregates.

use crate::data::loader_util::Minibatches;

use candle_core::{Device, Tensor};
use indicatif::{ParallelProgressIterator, ProgressBar};
use matrix_util::traits::CandleDataLoaderOps;
use rayon::prelude::*;
use std::sync::Arc;

pub mod splice_tracks;
pub mod graph_adj;
pub mod pack;
pub mod stats;
pub mod top_k;
pub mod types;
pub mod union_scatter;

pub use splice_tracks::{
    gem_samples_from_csc, top_k_genes_from_row, GemIndexedArgs, GemIndexedData,
    GemMinibatchData, GemSample, GeneTrackMap,
};
pub use graph_adj::{build_sparse_edges_from_tensor, GraphCsr, SparseEdgeBatch};
pub use pack::gather_per_feature_at_indices;
pub use stats::{compute_log_selection_freq, sum_sample_values};
pub use top_k::{csc_columns_to_indexed_samples, top_k_indices_weighted};
pub use types::{IndexedInMemoryArgs, IndexedMinibatchData, IndexedSample};
pub use union_scatter::{build_union_and_scatter_pos, slice_log_q_at_union};

// Crate-public re-exports.
pub use pack::pack_indices_values;
pub use top_k::build_indexed_samples;

use graph_adj::{scatter_sparse_edges, SubAdjCache};
use pack::{pack_null_at_indices, pack_values_only};

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
    output_samples: Vec<IndexedSample>,
    n_input_features: usize,
    n_output_features: usize,
    input_context_size: usize,
    output_context_size: usize,
    /// Per-feature `log(q_d)` where `q_d = P(feature d in top-K)`. Used
    /// for importance-weighted conditional softmax (Jean et al., 2015).
    output_log_q: Vec<f32>,
    /// Per-feature mean expression rate `μ_d` (encoder side) gathered
    /// into `input_values_mean [N, K]` at minibatch build time.
    input_mean: Option<Vec<f32>>,
    /// Per-feature NB-Fisher weight (decoder side) gathered into
    /// `output_values_weight [N, K]` at minibatch build time.
    output_fisher_weights: Option<Vec<f32>>,
    /// Optional feature-feature graph. When set, the indexed encoder's
    /// GCN diffusion block consumes a per-minibatch [`SparseEdgeBatch`]
    /// scattered straight from `sub_adj_cache`.
    graph_csr: Option<Arc<GraphCsr>>,
    /// Pre-computed and pre-normalised per-cell sub-adjacency triples
    /// in slot-space (see [`graph_adj`] module docs).
    sub_adj_cache: Option<SubAdjCache>,
    /// Sum of all `output_samples` values — the per-epoch decoder count
    /// total, invariant to shuffling.
    total_output_count: f32,
    minibatches: Minibatches,
    cached_batches: Vec<IndexedMinibatchData>,
}

impl IndexedInMemoryData {
    /// Build indexed data from dense matrices.
    ///
    /// Input and output get independent top-K selections from their
    /// respective sources.
    pub fn from_dense<D>(args: IndexedInMemoryArgs<D>) -> anyhow::Result<Self>
    where
        D: CandleDataLoaderOps + Sync,
    {
        let (n_samples, n_input_features) = args.input.data_shape();
        let (_, n_output_features) = args.output.data_shape();
        let input_context_size = args.input_context_size.min(n_input_features);
        let output_context_size = args.output_context_size.min(n_output_features);

        let input_samples = build_indexed_samples(
            args.input,
            n_samples,
            input_context_size,
            args.input_shortlist_weights,
            "Top-K (encoder)",
        );
        let output_samples = build_indexed_samples(
            args.output,
            n_samples,
            output_context_size,
            args.output_shortlist_weights,
            "Top-K (decoder)",
        );

        let output_log_q = compute_log_selection_freq(&output_samples, n_output_features);
        let total_output_count = sum_sample_values(&output_samples);

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
        let output_fisher_weights = args.output_fisher_weights.map(|s| s.to_vec());
        if let Some(ref b) = input_mean {
            anyhow::ensure!(
                b.len() == n_input_features,
                "input_mean length {} != n_input_features {}",
                b.len(),
                n_input_features
            );
        }
        if let Some(ref w) = output_fisher_weights {
            anyhow::ensure!(
                w.len() == n_output_features,
                "output_fisher_weights length {} != n_output_features {}",
                w.len(),
                n_output_features
            );
        }

        Ok(IndexedInMemoryData {
            input_samples,
            input_null_rows: null_rows,
            output_samples,
            n_input_features,
            n_output_features,
            input_context_size,
            output_context_size,
            output_log_q,
            input_mean,
            output_fisher_weights,
            graph_csr: None,
            sub_adj_cache: None,
            total_output_count,
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

    pub fn output_context_size(&self) -> usize {
        self.output_context_size
    }

    pub fn n_input_features(&self) -> usize {
        self.n_input_features
    }

    pub fn n_output_features(&self) -> usize {
        self.n_output_features
    }

    pub fn total_output_count(&self) -> f32 {
        self.total_output_count
    }

    /// Build a packed minibatch.
    ///
    /// Encoder side packs `(indices, values)` directly into `[N, K_in]`
    /// — no union and no `[N, S_in]`. Decoder side builds the union and
    /// per-cell scatter positions; values are packed in per-cell index
    /// order into `[N, K_out]`. Nothing is materialised at `[N, S]`
    /// shape on host or device.
    #[allow(clippy::type_complexity)]
    fn build_minibatch(
        &self,
        sample_indices: &[usize],
        target_device: &Device,
    ) -> anyhow::Result<IndexedMinibatchData> {
        let k_in = self.input_context_size;
        let k_out = self.output_context_size;

        let (input_result, output_result) = rayon::join(
            || -> anyhow::Result<(Tensor, Tensor, Option<Tensor>, Option<Tensor>)> {
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
                Ok((
                    input_indices,
                    input_values,
                    input_values_null,
                    input_values_mean,
                ))
            },
            || -> anyhow::Result<(Tensor, Tensor, Tensor, Vec<u32>, Option<Tensor>)> {
                let (union_indices, scatter_pos, union_vec) = build_union_and_scatter_pos(
                    &self.output_samples,
                    sample_indices,
                    self.n_output_features,
                    k_out,
                    target_device,
                )?;
                let values =
                    pack_values_only(&self.output_samples, sample_indices, k_out, target_device)?;
                let output_values_weight = match self.output_fisher_weights.as_ref() {
                    Some(w) => Some(gather_per_feature_at_indices(
                        &self.output_samples,
                        sample_indices,
                        w,
                        k_out,
                        target_device,
                    )?),
                    None => None,
                };
                Ok((
                    union_indices,
                    scatter_pos,
                    values,
                    union_vec,
                    output_values_weight,
                ))
            },
        );
        let (input_indices, input_values, input_values_null, input_values_mean) = input_result?;
        let (
            output_union_indices,
            output_scatter_pos,
            output_values,
            output_union_vec,
            output_values_weight,
        ) = output_result?;

        let output_log_q_s =
            slice_log_q_at_union(&self.output_log_q, &output_union_vec, target_device)?;

        Ok(IndexedMinibatchData {
            input_indices,
            input_values,
            input_values_null,
            input_values_mean,
            output_union_indices,
            output_scatter_pos,
            output_values,
            output_values_weight,
            output_log_q_s,
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

    fn assert_union_eq_set(union: &[u32], expected: &[u32]) {
        let mut sorted = union.to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, expected);
    }

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
            output: &data,
            input_context_size: 3,
            output_context_size: 3,
            input_shortlist_weights: &w,
            output_shortlist_weights: &w,
            input_mean: None,
            output_fisher_weights: None,
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

    #[test]
    fn test_packed_minibatch_shapes() {
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
        let args = IndexedInMemoryArgs {
            input: &data,
            input_null: None,
            output: &data,
            input_context_size: 2,
            output_context_size: 2,
            input_shortlist_weights: &w,
            output_shortlist_weights: &w,
            input_mean: None,
            output_fisher_weights: None,
        };

        let indexed = IndexedInMemoryData::from_dense(args).unwrap();
        let mb = indexed.minibatch_ordered(0, 3, &Device::Cpu).unwrap();

        assert_eq!(mb.input_indices.dims(), &[3, 2]);
        assert_eq!(mb.input_values.dims(), &[3, 2]);

        let in_idx: Vec<Vec<u32>> = mb.input_indices.to_vec2().unwrap();
        let in_val: Vec<Vec<f32>> = mb.input_values.to_vec2().unwrap();
        assert_eq!(in_idx[0], vec![3, 5]);
        assert!((in_val[0][0] - 0.9).abs() < 1e-6);
        assert!((in_val[0][1] - 0.7).abs() < 1e-6);

        let union: Vec<u32> = mb.output_union_indices.to_vec1().unwrap();
        assert_union_eq_set(&union, &[0, 1, 3, 4, 5]);
        assert_eq!(mb.output_scatter_pos.dims(), &[3, 2]);
        assert_eq!(mb.output_values.dims(), &[3, 2]);

        let scat: Vec<Vec<u32>> = mb.output_scatter_pos.to_vec2().unwrap();
        let vals: Vec<Vec<f32>> = mb.output_values.to_vec2().unwrap();

        assert_eq!(union[scat[0][0] as usize], 3);
        assert_eq!(union[scat[0][1] as usize], 5);
        assert!((vals[0][0] - 0.9).abs() < 1e-6);
        assert!((vals[0][1] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_packed_minibatch_separate_output() {
        let input = DMatrix::<f32>::from_row_slice(
            2,
            4,
            &[
                0.9, 0.1, 0.5, 0.3, // top-2 = {0,2}
                0.2, 0.8, 0.4, 0.7, // top-2 = {1,3}
            ],
        );
        let output = DMatrix::<f32>::from_row_slice(
            2,
            4,
            &[
                10.0, 20.0, 30.0, 40.0, // top-2 = {2,3}
                50.0, 60.0, 70.0, 80.0, // top-2 = {2,3}
            ],
        );

        let w = vec![1.0f32; 4];
        let args = IndexedInMemoryArgs {
            input: &input,
            input_null: None,
            output: &output,
            input_context_size: 2,
            output_context_size: 2,
            input_shortlist_weights: &w,
            output_shortlist_weights: &w,
            input_mean: None,
            output_fisher_weights: None,
        };

        let indexed = IndexedInMemoryData::from_dense(args).unwrap();
        let mb = indexed.minibatch_ordered(0, 2, &Device::Cpu).unwrap();

        let in_idx: Vec<Vec<u32>> = mb.input_indices.to_vec2().unwrap();
        assert_eq!(in_idx[0], vec![0, 2]);
        assert_eq!(in_idx[1], vec![1, 3]);

        let output_union: Vec<u32> = mb.output_union_indices.to_vec1().unwrap();
        assert_union_eq_set(&output_union, &[2, 3]);

        let vals: Vec<Vec<f32>> = mb.output_values.to_vec2().unwrap();
        assert!((vals[0][0] - 30.0).abs() < 1e-6);
        assert!((vals[0][1] - 40.0).abs() < 1e-6);
        assert!((vals[1][0] - 70.0).abs() < 1e-6);
        assert!((vals[1][1] - 80.0).abs() < 1e-6);

        let scat: Vec<Vec<u32>> = mb.output_scatter_pos.to_vec2().unwrap();
        for row in &scat {
            for (kk, &pos) in row.iter().enumerate() {
                let expected_feat = if kk == 0 { 2u32 } else { 3u32 };
                assert_eq!(output_union[pos as usize], expected_feat);
            }
        }
    }

    #[test]
    fn test_packed_minibatch_different_context_sizes() {
        let data = DMatrix::<f32>::from_row_slice(
            2,
            6,
            &[
                0.1, 0.5, 0.3, 0.9, 0.2, 0.7, // input top-3={1,3,5}, output top-1={3}
                0.8, 0.1, 0.6, 0.2, 0.9, 0.3, // input top-3={0,2,4}, output top-1={4}
            ],
        );

        let w = vec![1.0f32; 6];
        let args = IndexedInMemoryArgs {
            input: &data,
            input_null: None,
            output: &data,
            input_context_size: 3,
            output_context_size: 1,
            input_shortlist_weights: &w,
            output_shortlist_weights: &w,
            input_mean: None,
            output_fisher_weights: None,
        };

        let indexed = IndexedInMemoryData::from_dense(args).unwrap();
        assert_eq!(indexed.input_context_size(), 3);
        assert_eq!(indexed.output_context_size(), 1);

        let mb = indexed.minibatch_ordered(0, 2, &Device::Cpu).unwrap();

        assert_eq!(mb.input_indices.dims(), &[2, 3]);
        assert_eq!(mb.input_values.dims(), &[2, 3]);
        assert_eq!(mb.output_scatter_pos.dims(), &[2, 1]);
        assert_eq!(mb.output_values.dims(), &[2, 1]);
        let union: Vec<u32> = mb.output_union_indices.to_vec1().unwrap();
        assert_union_eq_set(&union, &[3, 4]);
    }
}
