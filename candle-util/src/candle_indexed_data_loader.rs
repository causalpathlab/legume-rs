use crate::candle_data_loader_util::Minibatches;

use candle_core::{Device, Tensor};
use matrix_util::traits::CandleDataLoaderOps;
use std::collections::{BTreeSet, HashMap};

/// Per-sample: top-K features selected from dense data
pub struct IndexedSample {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

/// Batched minibatch with union-of-indexed-windows
pub struct IndexedMinibatchData {
    /// [S] u32 — sorted union of per-sample top-K indices
    pub union_indices: Tensor,
    /// [N, S] f32 — input values at union positions
    pub indexed_x: Tensor,
    /// [N, S] f32 — null/batch correction at union positions
    pub indexed_x_null: Option<Tensor>,
    /// [N, S] f32 — output/target at union positions
    pub indexed_y: Option<Tensor>,
    /// [N, S] f32 — output null at union positions
    pub indexed_y_null: Option<Tensor>,
}

/// Adaptive feature window data loader.
///
/// Each sample keeps only its top-K features by value (K = context_size).
/// Batches use the union of selected indices across the minibatch,
/// producing [N, S] tensors where S is the union size.
pub struct IndexedInMemoryData {
    samples: Vec<IndexedSample>,
    null_samples: Option<Vec<Vec<f32>>>,
    output_samples: Option<Vec<Vec<f32>>>,
    output_null_samples: Option<Vec<Vec<f32>>>,
    n_features: usize,
    context_size: usize,
    minibatches: Minibatches,
}

pub struct IndexedInMemoryArgs<'a, D>
where
    D: CandleDataLoaderOps,
{
    pub input: &'a D,
    pub input_null: Option<&'a D>,
    pub output: Option<&'a D>,
    pub output_null: Option<&'a D>,
    pub context_size: usize,
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

impl IndexedInMemoryData {
    /// Build indexed data from dense matrices.
    ///
    /// For each row of `input`, finds top-K indices by value.
    /// The same indices are used to index all four data sources.
    pub fn from_dense<D>(args: IndexedInMemoryArgs<D>) -> anyhow::Result<Self>
    where
        D: CandleDataLoaderOps,
    {
        let input_rows = args.input.rows_to_tensor_vec();
        let n_samples = input_rows.len();
        let n_features = input_rows[0].flatten_all()?.to_vec1::<f32>()?.len();
        let context_size = args.context_size.min(n_features);

        // Pre-extract all rows for auxiliary data
        let null_rows: Option<Vec<Vec<f32>>> = args
            .input_null
            .map(|d| d.rows_to_tensor_vec().iter().map(tensor_to_row).collect());
        let output_rows: Option<Vec<Vec<f32>>> = args
            .output
            .map(|d| d.rows_to_tensor_vec().iter().map(tensor_to_row).collect());
        let output_null_rows: Option<Vec<Vec<f32>>> = args
            .output_null
            .map(|d| d.rows_to_tensor_vec().iter().map(tensor_to_row).collect());

        // Build indexed samples from input
        let mut samples = Vec::with_capacity(n_samples);
        for row_tensor in &input_rows {
            let row = tensor_to_row(row_tensor);
            let (indices, values) = top_k_indices(&row, context_size);
            samples.push(IndexedSample { indices, values });
        }

        let rows: Vec<usize> = (0..n_samples).collect();

        Ok(IndexedInMemoryData {
            samples,
            null_samples: null_rows,
            output_samples: output_rows,
            output_null_samples: output_null_rows,
            n_features,
            context_size,
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

    pub fn context_size(&self) -> usize {
        self.context_size
    }

    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Build an indexed minibatch from the given sample indices.
    ///
    /// 1. Collect top-K indices from each sample into BTreeSet (sorted union)
    /// 2. Build position map: feature index -> position in union
    /// 3. Scatter values into [N, S] matrices
    fn build_minibatch(
        &self,
        sample_indices: &[usize],
        target_device: &Device,
    ) -> anyhow::Result<IndexedMinibatchData> {
        let n_batch = sample_indices.len();

        // 1. Build sorted union of all indices in this batch
        let mut union_set = BTreeSet::new();
        for &si in sample_indices {
            for &idx in &self.samples[si].indices {
                union_set.insert(idx);
            }
        }
        let union_vec: Vec<u32> = union_set.into_iter().collect();
        let s = union_vec.len();

        // 2. Position map: feature index -> position in union
        let pos_map: HashMap<u32, usize> = union_vec
            .iter()
            .enumerate()
            .map(|(pos, &idx)| (idx, pos))
            .collect();

        // 3. Build [N, S] matrices by scattering values
        let mut x_data = vec![0.0f32; n_batch * s];
        let mut x_null_data = self
            .null_samples
            .as_ref()
            .map(|_| vec![0.0f32; n_batch * s]);
        let mut y_data = self
            .output_samples
            .as_ref()
            .map(|_| vec![0.0f32; n_batch * s]);
        let mut y_null_data = self
            .output_null_samples
            .as_ref()
            .map(|_| vec![0.0f32; n_batch * s]);

        for (row, &si) in sample_indices.iter().enumerate() {
            let sample = &self.samples[si];

            // Scatter input values at selected positions
            for (k, &feat_idx) in sample.indices.iter().enumerate() {
                let col = pos_map[&feat_idx];
                x_data[row * s + col] = sample.values[k];
            }

            // Scatter auxiliary data at all union positions
            if let Some(ref null_data) = self.null_samples {
                let buf = x_null_data.as_mut().unwrap();
                let null_row = &null_data[si];
                for &feat_idx in &union_vec {
                    let col = pos_map[&feat_idx];
                    buf[row * s + col] = null_row[feat_idx as usize];
                }
            }

            if let Some(ref out_data) = self.output_samples {
                let buf = y_data.as_mut().unwrap();
                let out_row = &out_data[si];
                for &feat_idx in &union_vec {
                    let col = pos_map[&feat_idx];
                    buf[row * s + col] = out_row[feat_idx as usize];
                }
            }

            if let Some(ref out_null) = self.output_null_samples {
                let buf = y_null_data.as_mut().unwrap();
                let out_null_row = &out_null[si];
                for &feat_idx in &union_vec {
                    let col = pos_map[&feat_idx];
                    buf[row * s + col] = out_null_row[feat_idx as usize];
                }
            }
        }

        // Convert to tensors
        let union_indices =
            Tensor::from_vec(union_vec, (s,), target_device)?.to_dtype(candle_core::DType::U32)?;
        let indexed_x = Tensor::from_vec(x_data, (n_batch, s), target_device)?;
        let indexed_x_null = x_null_data
            .map(|buf| Tensor::from_vec(buf, (n_batch, s), target_device))
            .transpose()?;
        let indexed_y = y_data
            .map(|buf| Tensor::from_vec(buf, (n_batch, s), target_device))
            .transpose()?;
        let indexed_y_null = y_null_data
            .map(|buf| Tensor::from_vec(buf, (n_batch, s), target_device))
            .transpose()?;

        Ok(IndexedMinibatchData {
            union_indices,
            indexed_x,
            indexed_x_null,
            indexed_y,
            indexed_y_null,
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
            output: None,
            output_null: None,
            context_size: 3,
        };

        let indexed = IndexedInMemoryData::from_dense(args).unwrap();
        assert_eq!(indexed.num_data(), 4);
        assert_eq!(indexed.n_features(), 6);
        assert_eq!(indexed.context_size(), 3);

        // Verify sample 0: top-3 should be indices 1, 3, 5
        assert_eq!(indexed.samples[0].indices, vec![1, 3, 5]);
        assert_eq!(indexed.samples[0].values, vec![0.5, 0.9, 0.7]);

        // Verify sample 1: top-3 should be indices 0, 2, 4
        assert_eq!(indexed.samples[1].indices, vec![0, 2, 4]);
        assert_eq!(indexed.samples[1].values, vec![0.8, 0.6, 0.9]);
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
            output: None,
            output_null: None,
            context_size: 2,
        };

        let indexed = IndexedInMemoryData::from_dense(args).unwrap();

        // Use ordered minibatch with all 3 samples
        let mb = indexed.minibatch_ordered(0, 3, &Device::Cpu).unwrap();

        // Union of {3,5}, {0,4}, {1,4} = {0,1,3,4,5} => S=5
        let union: Vec<u32> = mb.union_indices.to_vec1().unwrap();
        assert_eq!(union, vec![0, 1, 3, 4, 5]);

        // indexed_x shape should be [3, 5]
        assert_eq!(mb.indexed_x.dims(), &[3, 5]);

        // Verify values: sample 0 has features 3,5 active with values 0.9, 0.7
        let x_vals: Vec<Vec<f32>> = mb.indexed_x.to_vec2().unwrap();
        // union positions: 0->0, 1->1, 3->2, 4->3, 5->4
        // sample 0: feat 3 at pos 2 = 0.9, feat 5 at pos 4 = 0.7, rest = 0
        assert!((x_vals[0][2] - 0.9).abs() < 1e-6); // feat 3
        assert!((x_vals[0][4] - 0.7).abs() < 1e-6); // feat 5
        assert!((x_vals[0][0]).abs() < 1e-6); // feat 0 not selected
    }

    #[test]
    fn test_indexed_with_output() {
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
                10.0, 20.0, 30.0, 40.0, //
                50.0, 60.0, 70.0, 80.0, //
            ],
        );

        let args = IndexedInMemoryArgs {
            input: &input,
            input_null: None,
            output: Some(&output),
            output_null: None,
            context_size: 2,
        };

        let indexed = IndexedInMemoryData::from_dense(args).unwrap();
        let mb = indexed.minibatch_ordered(0, 2, &Device::Cpu).unwrap();

        // Union of {0,2} and {1,3} = {0,1,2,3} => S=4
        let union: Vec<u32> = mb.union_indices.to_vec1().unwrap();
        assert_eq!(union, vec![0, 1, 2, 3]);

        // Output should be present and indexed at same positions
        assert!(mb.indexed_y.is_some());
        let y_vals: Vec<Vec<f32>> = mb.indexed_y.unwrap().to_vec2().unwrap();
        // sample 0: all 4 features in union, output = [10, 20, 30, 40]
        assert!((y_vals[0][0] - 10.0).abs() < 1e-6);
        assert!((y_vals[0][2] - 30.0).abs() < 1e-6);
        // sample 1: output = [50, 60, 70, 80]
        assert!((y_vals[1][1] - 60.0).abs() < 1e-6);
        assert!((y_vals[1][3] - 80.0).abs() < 1e-6);
    }
}
