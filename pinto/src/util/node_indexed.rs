//! Per-node indexed data construction for the indexed link community encoder.
//!
//! Builds `IndexedSample`s for nodes (cells or super-nodes) at any coarsening
//! level. At the finest level each node = one cell; at coarsened levels each
//! node = pseudobulk sum of member cells' sparse expression vectors.

use crate::util::common::*;
use candle_util::candle_core::{Device, Tensor};
use candle_util::candle_indexed_data_loader::{top_k_indices, IndexedSample};
use matrix_util::utils::generate_minibatch_intervals;

/// Build `IndexedSample`s for nodes at a given coarsening level.
///
/// * `data`         – sparse expression [n_genes × n_cells]
/// * `cell_labels`  – cell → super-node index (identity for finest level)
/// * `n_nodes`      – number of super-nodes at this level
/// * `context_size` – top-K genes per node
/// * `block_size`   – cells per I/O block
pub fn build_node_indexed_samples(
    data: &SparseIoVec,
    cell_labels: &[usize],
    n_nodes: usize,
    context_size: usize,
    block_size: usize,
) -> anyhow::Result<Vec<IndexedSample>> {
    let n_genes = data.num_rows();
    let n_cells = data.num_columns();
    let context_size = context_size.min(n_genes);

    let mut node_cells: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
    for (cell, &label) in cell_labels.iter().enumerate() {
        if cell < n_cells {
            node_cells[label].push(cell);
        }
    }

    let intervals = generate_minibatch_intervals(n_nodes, block_size.max(1));

    let pb = new_progress_bar(
        intervals.len() as u64,
        "Node samples {bar:40} {pos}/{len} blocks ({eta})",
    );

    let partial: Vec<Vec<(usize, IndexedSample)>> = intervals
        .par_iter()
        .progress_with(pb.clone())
        .map(|&(lb, ub)| {
            let mut unique_cells: Vec<usize> = Vec::new();
            for cells in &node_cells[lb..ub] {
                unique_cells.extend_from_slice(cells);
            }
            unique_cells.sort_unstable();
            unique_cells.dedup();

            if unique_cells.is_empty() {
                return (lb..ub)
                    .map(|node| {
                        (
                            node,
                            IndexedSample {
                                indices: vec![0; context_size.min(1)],
                                values: vec![0.0; context_size.min(1)],
                            },
                        )
                    })
                    .collect();
            }

            let x_dn = data
                .read_columns_csc(unique_cells.iter().copied())
                .expect("failed to read columns");

            let cell_to_col: HashMap<usize, usize> = unique_cells
                .iter()
                .enumerate()
                .map(|(col, &cell)| (cell, col))
                .collect();

            let mut results = Vec::with_capacity(ub - lb);
            let mut dense = vec![0.0f32; n_genes];

            for (node, node_cell_list) in node_cells[lb..ub].iter().enumerate() {
                dense.iter_mut().for_each(|x| *x = 0.0);
                for &cell in node_cell_list {
                    if let Some(&col) = cell_to_col.get(&cell) {
                        let col_slice = x_dn.col(col);
                        for (&row, &val) in col_slice
                            .row_indices()
                            .iter()
                            .zip(col_slice.values().iter())
                        {
                            dense[row] += val;
                        }
                    }
                }

                let (indices, values) = top_k_indices(&dense, context_size);
                results.push((lb + node, IndexedSample { indices, values }));
            }

            results
        })
        .collect();

    pb.finish_and_clear();

    let samples: Vec<IndexedSample> = partial
        .into_iter()
        .flat_map(|block| block.into_iter().map(|(_, sample)| sample))
        .collect();

    Ok(samples)
}

/// Build union indices and scatter matrix for a batch of node samples.
///
/// Returns `(union_indices [S], indexed_x [N, S])` as Tensors.
pub fn build_node_minibatch(
    samples: &[IndexedSample],
    node_indices: &[usize],
    n_features: usize,
    device: &Device,
) -> anyhow::Result<(Tensor, Tensor)> {
    let n_batch = node_indices.len();

    let mut pos_lookup = vec![usize::MAX; n_features];
    let mut union_vec: Vec<u32> = Vec::new();
    for &ni in node_indices {
        for &idx in &samples[ni].indices {
            let fi = idx as usize;
            if pos_lookup[fi] == usize::MAX {
                pos_lookup[fi] = 0; // placeholder
                union_vec.push(idx);
            }
        }
    }
    union_vec.sort_unstable();
    for (pos, &idx) in union_vec.iter().enumerate() {
        pos_lookup[idx as usize] = pos;
    }
    let s = union_vec.len();

    // Scatter values into [N, S]
    let mut x_data = vec![0.0f32; n_batch * s];
    for (row, &ni) in node_indices.iter().enumerate() {
        let sample = &samples[ni];
        let row_offset = row * s;
        for (k, &feat_idx) in sample.indices.iter().enumerate() {
            let col = pos_lookup[feat_idx as usize];
            x_data[row_offset + col] = sample.values[k];
        }
    }

    let union_indices = Tensor::from_vec(union_vec, (s,), device)?
        .to_dtype(candle_util::candle_core::DType::U32)?;
    let indexed_x = Tensor::from_vec(x_data, (n_batch, s), device)?;

    Ok((union_indices, indexed_x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_node_minibatch_basic() {
        let samples = vec![
            IndexedSample {
                indices: vec![0, 3, 7],
                values: vec![1.0, 2.0, 3.0],
            },
            IndexedSample {
                indices: vec![1, 3, 5],
                values: vec![4.0, 5.0, 6.0],
            },
        ];
        let node_indices = vec![0, 1];
        let device = Device::Cpu;
        let (union_idx, indexed_x) =
            build_node_minibatch(&samples, &node_indices, 10, &device).unwrap();

        let union: Vec<u32> = union_idx.to_vec1().unwrap();
        assert_eq!(union, vec![0, 1, 3, 5, 7]); // sorted union

        let x: Vec<Vec<f32>> = indexed_x.to_vec2().unwrap();
        // Row 0: features at [0,3,7] → positions [0,2,4]
        assert_eq!(x[0], vec![1.0, 0.0, 2.0, 0.0, 3.0]);
        // Row 1: features at [1,3,5] → positions [1,2,3]
        assert_eq!(x[1], vec![0.0, 4.0, 5.0, 6.0, 0.0]);
    }
}
