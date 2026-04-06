use crate::candle_data_loader_util::Minibatches;

use candle_core::{Device, Tensor};
use matrix_util::traits::CandleDataLoaderOps;
use rayon::prelude::*;

/// Per-sample: top-K features selected from dense data
#[derive(Clone)]
pub struct IndexedSample {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

/// Batched minibatch with separate input (encoder) and output (decoder) windows.
pub struct IndexedMinibatchData {
    /// [S_in] u32 — sorted union of per-sample top-K indices for encoder
    pub input_union_indices: Tensor,
    /// [N, S_in] f32 — encoder input values at union positions
    pub input_indexed_x: Tensor,
    /// [N, S_in] f32 — encoder null/batch correction at union positions
    pub input_indexed_x_null: Option<Tensor>,
    /// [S_out] u32 — sorted union of per-sample top-K indices for decoder
    pub output_union_indices: Tensor,
    /// [N, S_out] f32 — decoder target values at union positions
    pub output_indexed_x: Tensor,
    /// [S_out] f32 — log selection frequency at union positions for importance weighting
    pub output_log_q_s: Tensor,
}

/// Adaptive feature window data loader with decoupled encoder/decoder windows.
///
/// Each sample keeps its top-K features by value independently for input (encoder)
/// and output (decoder) sides. Batches use the union of selected indices within
/// each side, producing separate [N, S_in] and [N, S_out] tensors.
///
/// Call `precompute_all_minibatches` after `shuffle_minibatch` to cache all
/// minibatch tensors for the jitter interval. Use `minibatch_cached` to retrieve.
pub struct IndexedInMemoryData {
    input_samples: Vec<IndexedSample>,
    input_null_rows: Option<Vec<Vec<f32>>>,
    output_samples: Vec<IndexedSample>,
    n_input_features: usize,
    n_output_features: usize,
    input_context_size: usize,
    output_context_size: usize,
    /// Per-feature log selection frequency: log(q_d) where q_d = P(feature d in top-K).
    /// Used for importance-weighted conditional softmax (Jean et al., 2015).
    output_log_q: Vec<f32>,
    minibatches: Minibatches,
    cached_batches: Vec<IndexedMinibatchData>,
}

pub struct IndexedInMemoryArgs<'a, D>
where
    D: CandleDataLoaderOps,
{
    pub input: &'a D,
    pub input_null: Option<&'a D>,
    pub output: &'a D,
    pub input_context_size: usize,
    pub output_context_size: usize,
}

/// Select top-K indices by value from a dense row.
/// Returns (sorted_indices, values_at_those_indices).
pub fn top_k_indices(row: &[f32], k: usize) -> (Vec<u32>, Vec<f32>) {
    let k = k.min(row.len());

    // Collect (value, index) pairs and partial sort to find top-K
    let mut indexed: Vec<(f32, u32)> = row
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i as u32))
        .collect();

    // Partial sort: move top-K elements to the front
    indexed.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
        b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Take top-K, then sort by index for deterministic order
    let mut top_k: Vec<(u32, f32)> = indexed[..k].iter().map(|&(v, i)| (i, v)).collect();
    top_k.sort_unstable_by_key(|&(idx, _)| idx);

    let indices: Vec<u32> = top_k.iter().map(|&(i, _)| i).collect();
    let values: Vec<f32> = top_k.iter().map(|&(_, v)| v).collect();
    (indices, values)
}

struct UnionScatterOut {
    union_indices: Tensor,
    indexed_x: Tensor,
    union_vec: Vec<u32>,
    /// Lookup table: feature_id -> position in union_vec. Indexed by feature_id,
    /// entries are usize::MAX for features not in the union.
    pos_lookup: Vec<usize>,
}

/// Build union indices and scattered [N, S] matrix from IndexedSamples.
fn build_union_scatter(
    samples: &[IndexedSample],
    sample_indices: &[usize],
    n_features: usize,
    target_device: &Device,
) -> anyhow::Result<UnionScatterOut> {
    let n_batch = sample_indices.len();

    // Single-pass: build position lookup and sorted union vec together
    let mut pos_lookup = vec![usize::MAX; n_features];
    let mut union_vec = Vec::new();
    for &si in sample_indices {
        for &idx in &samples[si].indices {
            let fi = idx as usize;
            if pos_lookup[fi] == usize::MAX {
                pos_lookup[fi] = union_vec.len();
                union_vec.push(idx);
            }
        }
    }
    // Sort union by feature index for deterministic output
    union_vec.sort_unstable();
    for (pos, &idx) in union_vec.iter().enumerate() {
        pos_lookup[idx as usize] = pos;
    }
    let s = union_vec.len();

    // Scatter values into [N, S] buffer
    let mut x_data = vec![0.0f32; n_batch * s];
    for (row, &si) in sample_indices.iter().enumerate() {
        let sample = &samples[si];
        let row_offset = row * s;
        for (k, &feat_idx) in sample.indices.iter().enumerate() {
            let col = pos_lookup[feat_idx as usize];
            x_data[row_offset + col] = sample.values[k];
        }
    }

    let union_indices = Tensor::from_vec(union_vec.clone(), (s,), target_device)?
        .to_dtype(candle_core::DType::U32)?;
    let indexed_x = Tensor::from_vec(x_data, (n_batch, s), target_device)?;

    Ok(UnionScatterOut {
        union_indices,
        indexed_x,
        union_vec,
        pos_lookup,
    })
}

/// Build IndexedSamples from a data source in parallel.
fn build_indexed_samples<D: CandleDataLoaderOps + Sync>(
    data: &D,
    n_samples: usize,
    context_size: usize,
) -> Vec<IndexedSample> {
    (0..n_samples)
        .into_par_iter()
        .map(|i| {
            let row = data.row_to_f32_vec(i);
            let (indices, values) = top_k_indices(&row, context_size);
            IndexedSample { indices, values }
        })
        .collect()
}

/// Compute log selection frequency for each feature from indexed samples.
///
/// q_d = (# samples containing feature d) / (total samples), clamped to [1/n, 1].
/// Returns log(q_d) for each of n_features features.
///
/// Used for importance-weighted conditional softmax (Jean et al., 2015,
/// "On Using Very Large Target Vocabulary for Neural Machine Translation").
fn compute_log_selection_freq(samples: &[IndexedSample], n_features: usize) -> Vec<f32> {
    let n = samples.len().max(1) as f32;
    let mut counts = vec![0u32; n_features];
    for sample in samples {
        for &idx in &sample.indices {
            counts[idx as usize] += 1;
        }
    }
    counts
        .iter()
        .map(|&c| ((c.max(1) as f32) / n).ln())
        .collect()
}

impl IndexedInMemoryData {
    /// Build indexed data from dense matrices.
    ///
    /// Input and output get independent top-K selections from their respective sources.
    pub fn from_dense<D>(args: IndexedInMemoryArgs<D>) -> anyhow::Result<Self>
    where
        D: CandleDataLoaderOps + Sync,
    {
        let (n_samples, n_input_features) = args.input.data_shape();
        let (_, n_output_features) = args.output.data_shape();
        let input_context_size = args.input_context_size.min(n_input_features);
        let output_context_size = args.output_context_size.min(n_output_features);

        let input_samples = build_indexed_samples(args.input, n_samples, input_context_size);
        let output_samples = build_indexed_samples(args.output, n_samples, output_context_size);

        let output_log_q = compute_log_selection_freq(&output_samples, n_output_features);

        // Pre-extract null rows in parallel
        let null_rows: Option<Vec<Vec<f32>>> = args.input_null.map(|d| {
            let (n, _) = d.data_shape();
            (0..n)
                .into_par_iter()
                .map(|i| d.row_to_f32_vec(i))
                .collect()
        });

        let rows: Vec<usize> = (0..n_samples).collect();

        Ok(IndexedInMemoryData {
            input_samples,
            input_null_rows: null_rows,
            output_samples,
            n_input_features,
            n_output_features,
            input_context_size,
            output_context_size,
            output_log_q,
            minibatches: Minibatches {
                samples: rows,
                chunks: vec![],
            },
            cached_batches: vec![],
        })
    }

    /// Build indexed data from pre-computed samples (e.g., from sparse I/O).
    ///
    /// Both input and output share the same samples (encoder = decoder target).
    pub fn from_samples(
        samples: Vec<IndexedSample>,
        n_features: usize,
        context_size: usize,
    ) -> Self {
        let n = samples.len();
        let rows: Vec<usize> = (0..n).collect();
        // Split: output gets the original, input gets a clone.
        // Both are needed because the struct stores two separate Vecs.
        let output_samples = samples;
        let input_samples = output_samples.clone();
        let output_log_q = compute_log_selection_freq(&output_samples, n_features);
        IndexedInMemoryData {
            input_samples,
            input_null_rows: None,
            output_samples,
            n_input_features: n_features,
            n_output_features: n_features,
            input_context_size: context_size,
            output_context_size: context_size,
            output_log_q,
            minibatches: Minibatches {
                samples: rows,
                chunks: vec![],
            },
            cached_batches: vec![],
        }
    }

    pub fn shuffle_minibatch(&mut self, batch_size: usize) {
        self.minibatches.shuffle_minibatch(batch_size);
        self.cached_batches.clear();
    }

    /// Pre-build all minibatch tensors for the current shuffle order.
    ///
    /// Call this once after `shuffle_minibatch` to avoid rebuilding
    /// union+scatter on every `minibatch_cached` call within a jitter interval.
    pub fn precompute_all_minibatches(&mut self, target_device: &Device) -> anyhow::Result<()> {
        self.cached_batches = (0..self.minibatches.chunks.len())
            .map(|b| {
                let sample_indices = &self.minibatches.chunks[b];
                self.build_minibatch(sample_indices, target_device)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(())
    }

    /// Retrieve a pre-computed minibatch. Panics if `precompute_all_minibatches`
    /// was not called after the last `shuffle_minibatch`.
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

    /// Build an indexed minibatch with separate input/output unions.
    fn build_minibatch(
        &self,
        sample_indices: &[usize],
        target_device: &Device,
    ) -> anyhow::Result<IndexedMinibatchData> {
        let n_batch = sample_indices.len();

        // Build input side (encoder)
        let input_out = build_union_scatter(
            &self.input_samples,
            sample_indices,
            self.n_input_features,
            target_device,
        )?;

        // Scatter null data at input union positions
        let input_indexed_x_null = if let Some(ref null_data) = self.input_null_rows {
            let s = input_out.union_vec.len();
            let mut buf = vec![0.0f32; n_batch * s];
            for (row, &si) in sample_indices.iter().enumerate() {
                let null_row = &null_data[si];
                let row_offset = row * s;
                for &feat_idx in &input_out.union_vec {
                    let col = input_out.pos_lookup[feat_idx as usize];
                    buf[row_offset + col] = null_row[feat_idx as usize];
                }
            }
            Some(Tensor::from_vec(buf, (n_batch, s), target_device)?)
        } else {
            None
        };

        // Build output side (decoder)
        let output_out = build_union_scatter(
            &self.output_samples,
            sample_indices,
            self.n_output_features,
            target_device,
        )?;

        // Slice log selection frequency at output union positions
        let log_q_s: Vec<f32> = output_out
            .union_vec
            .iter()
            .map(|&idx| self.output_log_q[idx as usize])
            .collect();
        let s_out = output_out.union_vec.len();
        let output_log_q_s = Tensor::from_vec(log_q_s, (1, s_out), target_device)?;

        Ok(IndexedMinibatchData {
            input_union_indices: input_out.union_indices,
            input_indexed_x: input_out.indexed_x,
            input_indexed_x_null,
            output_union_indices: output_out.union_indices,
            output_indexed_x: output_out.indexed_x,
            output_log_q_s,
        })
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

    /// Build an indexed minibatch from ordered (non-shuffled) sample range.
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
    fn test_top_k_indices() {
        let row = vec![0.1, 0.5, 0.3, 0.9, 0.2, 0.7];
        let (indices, values) = top_k_indices(&row, 3);
        // Top-3 by value: 0.9 (idx=3), 0.7 (idx=5), 0.5 (idx=1)
        // Sorted by index: 1, 3, 5
        assert_eq!(indices, vec![1, 3, 5]);
        assert_eq!(values, vec![0.5, 0.9, 0.7]);
    }

    #[test]
    fn test_top_k_all_features() {
        let row = vec![0.1, 0.5, 0.3];
        let (indices, values) = top_k_indices(&row, 10); // K > D
        assert_eq!(indices, vec![0, 1, 2]);
        assert_eq!(values, vec![0.1, 0.5, 0.3]);
    }

    #[test]
    fn test_indexed_from_dense() {
        // 4 samples, 6 features, context_size=3
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

        let args = IndexedInMemoryArgs {
            input: &data,
            input_null: None,
            output: &data,
            input_context_size: 3,
            output_context_size: 3,
        };

        let indexed = IndexedInMemoryData::from_dense(args).unwrap();
        assert_eq!(indexed.num_data(), 4);
        assert_eq!(indexed.n_input_features(), 6);
        assert_eq!(indexed.input_context_size(), 3);

        // Verify input sample 0: top-3 should be indices 1, 3, 5
        assert_eq!(indexed.input_samples[0].indices, vec![1, 3, 5]);
        assert_eq!(indexed.input_samples[0].values, vec![0.5, 0.9, 0.7]);

        // Verify input sample 1: top-3 should be indices 0, 2, 4
        assert_eq!(indexed.input_samples[1].indices, vec![0, 2, 4]);
        assert_eq!(indexed.input_samples[1].values, vec![0.8, 0.6, 0.9]);
    }

    #[test]
    fn test_indexed_minibatch_union() {
        let data = DMatrix::<f32>::from_row_slice(
            3,
            6,
            &[
                0.1, 0.5, 0.3, 0.9, 0.2, 0.7, // top-2 = {3,5}
                0.8, 0.1, 0.6, 0.2, 0.9, 0.3, // top-2 = {0,4}
                0.3, 0.7, 0.1, 0.4, 0.6, 0.5, // top-2 = {1,4}
            ],
        );

        let args = IndexedInMemoryArgs {
            input: &data,
            input_null: None,
            output: &data,
            input_context_size: 2,
            output_context_size: 2,
        };

        let indexed = IndexedInMemoryData::from_dense(args).unwrap();

        // Use ordered minibatch with all 3 samples
        let mb = indexed.minibatch_ordered(0, 3, &Device::Cpu).unwrap();

        // Union of {3,5}, {0,4}, {1,4} = {0,1,3,4,5} => S=5
        let union: Vec<u32> = mb.input_union_indices.to_vec1().unwrap();
        assert_eq!(union, vec![0, 1, 3, 4, 5]);

        // input_indexed_x shape should be [3, 5]
        assert_eq!(mb.input_indexed_x.dims(), &[3, 5]);

        // Verify values: sample 0 has features 3,5 active with values 0.9, 0.7
        let x_vals: Vec<Vec<f32>> = mb.input_indexed_x.to_vec2().unwrap();
        // union positions: 0->0, 1->1, 3->2, 4->3, 5->4
        // sample 0: feat 3 at pos 2 = 0.9, feat 5 at pos 4 = 0.7, rest = 0
        assert!((x_vals[0][2] - 0.9).abs() < 1e-6); // feat 3
        assert!((x_vals[0][4] - 0.7).abs() < 1e-6); // feat 5
        assert!((x_vals[0][0]).abs() < 1e-6); // feat 0 not selected
    }

    #[test]
    fn test_indexed_with_separate_output() {
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
                10.0, 20.0, 30.0, 40.0, // top-2 = {30,40} -> {2,3}
                50.0, 60.0, 70.0, 80.0, // top-2 = {70,80} -> {2,3}
            ],
        );

        let args = IndexedInMemoryArgs {
            input: &input,
            input_null: None,
            output: &output,
            input_context_size: 2,
            output_context_size: 2,
        };

        let indexed = IndexedInMemoryData::from_dense(args).unwrap();
        let mb = indexed.minibatch_ordered(0, 2, &Device::Cpu).unwrap();

        // Input union: {0,2} ∪ {1,3} = {0,1,2,3} => S_in=4
        let input_union: Vec<u32> = mb.input_union_indices.to_vec1().unwrap();
        assert_eq!(input_union, vec![0, 1, 2, 3]);

        // Output union: {2,3} ∪ {2,3} = {2,3} => S_out=2
        let output_union: Vec<u32> = mb.output_union_indices.to_vec1().unwrap();
        assert_eq!(output_union, vec![2, 3]);

        // Output values: sample 0 at positions {2,3} = [30.0, 40.0]
        let y_vals: Vec<Vec<f32>> = mb.output_indexed_x.to_vec2().unwrap();
        assert!((y_vals[0][0] - 30.0).abs() < 1e-6);
        assert!((y_vals[0][1] - 40.0).abs() < 1e-6);
        // sample 1 at positions {2,3} = [70.0, 80.0]
        assert!((y_vals[1][0] - 70.0).abs() < 1e-6);
        assert!((y_vals[1][1] - 80.0).abs() < 1e-6);
    }

    #[test]
    fn test_indexed_different_context_sizes() {
        let data = DMatrix::<f32>::from_row_slice(
            2,
            6,
            &[
                0.1, 0.5, 0.3, 0.9, 0.2, 0.7, // input top-3={1,3,5}, output top-1={3}
                0.8, 0.1, 0.6, 0.2, 0.9, 0.3, // input top-3={0,2,4}, output top-1={4}
            ],
        );

        let args = IndexedInMemoryArgs {
            input: &data,
            input_null: None,
            output: &data,
            input_context_size: 3,
            output_context_size: 1,
        };

        let indexed = IndexedInMemoryData::from_dense(args).unwrap();
        assert_eq!(indexed.input_context_size(), 3);
        assert_eq!(indexed.output_context_size(), 1);

        let mb = indexed.minibatch_ordered(0, 2, &Device::Cpu).unwrap();

        // Input union: {1,3,5} ∪ {0,2,4} = {0,1,2,3,4,5} => S_in=6
        assert_eq!(mb.input_indexed_x.dims()[1], 6);

        // Output union: {3} ∪ {4} = {3,4} => S_out=2
        assert_eq!(mb.output_indexed_x.dims()[1], 2);
    }
}
