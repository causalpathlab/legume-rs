use crate::candle_data_loader_util::Minibatches;

use candle_core::{Device, Tensor};
use matrix_util::traits::CandleDataLoaderOps;
use std::collections::{BTreeSet, HashMap};

/// Per-sample: top-K features selected from dense data
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
}

/// Adaptive feature window data loader with decoupled encoder/decoder windows.
///
/// Each sample keeps its top-K features by value independently for input (encoder)
/// and output (decoder) sides. Batches use the union of selected indices within
/// each side, producing separate [N, S_in] and [N, S_out] tensors.
pub struct IndexedInMemoryData {
    input_samples: Vec<IndexedSample>,
    input_null_rows: Option<Vec<Vec<f32>>>,
    output_samples: Vec<IndexedSample>,
    n_input_features: usize,
    n_output_features: usize,
    input_context_size: usize,
    output_context_size: usize,
    minibatches: Minibatches,
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

/// Extract dense row as Vec<f32> from a tensor
fn tensor_to_row(t: &Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

struct UnionScatterOut {
    union_indices: Tensor,
    indexed_x: Tensor,
    union_vec: Vec<u32>,
    pos_map: HashMap<u32, usize>,
}

/// Build union indices and scattered [N, S] matrix from IndexedSamples.
fn build_union_scatter(
    samples: &[IndexedSample],
    sample_indices: &[usize],
    target_device: &Device,
) -> anyhow::Result<UnionScatterOut> {
    let n_batch = sample_indices.len();

    let mut union_set = BTreeSet::new();
    for &si in sample_indices {
        for &idx in &samples[si].indices {
            union_set.insert(idx);
        }
    }
    let union_vec: Vec<u32> = union_set.into_iter().collect();
    let s = union_vec.len();

    let pos_map: HashMap<u32, usize> = union_vec
        .iter()
        .enumerate()
        .map(|(pos, &idx)| (idx, pos))
        .collect();

    let mut x_data = vec![0.0f32; n_batch * s];
    for (row, &si) in sample_indices.iter().enumerate() {
        let sample = &samples[si];
        for (k, &feat_idx) in sample.indices.iter().enumerate() {
            let col = pos_map[&feat_idx];
            x_data[row * s + col] = sample.values[k];
        }
    }

    let union_indices = Tensor::from_vec(union_vec.clone(), (s,), target_device)?
        .to_dtype(candle_core::DType::U32)?;
    let indexed_x = Tensor::from_vec(x_data, (n_batch, s), target_device)?;

    Ok(UnionScatterOut {
        union_indices,
        indexed_x,
        union_vec,
        pos_map,
    })
}

impl IndexedInMemoryData {
    /// Build indexed data from dense matrices.
    ///
    /// Input and output get independent top-K selections from their respective sources.
    pub fn from_dense<D>(args: IndexedInMemoryArgs<D>) -> anyhow::Result<Self>
    where
        D: CandleDataLoaderOps,
    {
        let input_rows = args.input.rows_to_tensor_vec();
        let output_rows = args.output.rows_to_tensor_vec();
        let n_samples = input_rows.len();
        let n_input_features = input_rows[0].flatten_all()?.to_vec1::<f32>()?.len();
        let n_output_features = output_rows[0].flatten_all()?.to_vec1::<f32>()?.len();
        let input_context_size = args.input_context_size.min(n_input_features);
        let output_context_size = args.output_context_size.min(n_output_features);

        // Pre-extract null rows
        let null_rows: Option<Vec<Vec<f32>>> = args
            .input_null
            .map(|d| d.rows_to_tensor_vec().iter().map(tensor_to_row).collect());

        // Build input indexed samples
        let mut input_samples = Vec::with_capacity(n_samples);
        for row_tensor in &input_rows {
            let row = tensor_to_row(row_tensor);
            let (indices, values) = top_k_indices(&row, input_context_size);
            input_samples.push(IndexedSample { indices, values });
        }

        // Build output indexed samples
        let mut output_samples = Vec::with_capacity(n_samples);
        for row_tensor in &output_rows {
            let row = tensor_to_row(row_tensor);
            let (indices, values) = top_k_indices(&row, output_context_size);
            output_samples.push(IndexedSample { indices, values });
        }

        let rows: Vec<usize> = (0..n_samples).collect();

        Ok(IndexedInMemoryData {
            input_samples,
            input_null_rows: null_rows,
            output_samples,
            n_input_features,
            n_output_features,
            input_context_size,
            output_context_size,
            minibatches: Minibatches {
                samples: rows,
                chunks: vec![],
            },
        })
    }

    pub fn shuffle_minibatch(&mut self, batch_size: usize) {
        self.minibatches.shuffle_minibatch(batch_size);
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
        let input_out = build_union_scatter(&self.input_samples, sample_indices, target_device)?;

        // Scatter null data at input union positions
        let input_indexed_x_null = if let Some(ref null_data) = self.input_null_rows {
            let s = input_out.union_vec.len();
            let mut buf = vec![0.0f32; n_batch * s];
            for (row, &si) in sample_indices.iter().enumerate() {
                let null_row = &null_data[si];
                for &feat_idx in &input_out.union_vec {
                    let col = input_out.pos_map[&feat_idx];
                    buf[row * s + col] = null_row[feat_idx as usize];
                }
            }
            Some(Tensor::from_vec(buf, (n_batch, s), target_device)?)
        } else {
            None
        };

        // Build output side (decoder)
        let output_out = build_union_scatter(&self.output_samples, sample_indices, target_device)?;

        Ok(IndexedMinibatchData {
            input_union_indices: input_out.union_indices,
            input_indexed_x: input_out.indexed_x,
            input_indexed_x_null,
            output_union_indices: output_out.union_indices,
            output_indexed_x: output_out.indexed_x,
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
