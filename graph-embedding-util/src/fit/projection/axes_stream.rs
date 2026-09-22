//! Cell groups for the per-axis projection, streamed from one sparse backend
//! per axis: the columns are walked in global order in groups, each group's
//! nonzeros are read per backend, and only one group is resident at a time.

use super::block_sgd::CellGroup;
use data_beans::sparse_io_vector::SparseIoVec;

/// Groups of `group_size` consecutive cells over `backends` (one per axis,
/// columns aligned). Refuses backends whose column counts differ.
pub fn stream_cell_groups<'a>(
    backends: &'a [&'a SparseIoVec],
    group_size: usize,
) -> anyhow::Result<impl Iterator<Item = anyhow::Result<CellGroup>> + 'a> {
    anyhow::ensure!(!backends.is_empty(), "at least one backend");
    let n_cells = backends[0].num_columns();
    for (a, b) in backends.iter().enumerate().skip(1) {
        anyhow::ensure!(
            b.num_columns() == n_cells,
            "backend {a} has {} columns, backend 0 has {n_cells}",
            b.num_columns()
        );
    }
    let size = group_size.max(1);
    Ok((0..n_cells).step_by(size).map(move |start| {
        let end = (start + size).min(n_cells);
        read_group(backends, start, end)
    }))
}

/// One group's sparse rows on every axis, each cell's features ascending.
fn read_group(backends: &[&SparseIoVec], start: usize, end: usize) -> anyhow::Result<CellGroup> {
    let n = end - start;
    let mut axes: Vec<Vec<(Vec<u32>, Vec<f32>)>> = Vec::with_capacity(backends.len());
    for b in backends {
        let mut rows: Vec<(Vec<u32>, Vec<f32>)> = vec![(Vec::new(), Vec::new()); n];
        b.for_each_triplet(start..end, n, |row, local_col, v| {
            if v > 0.0 {
                let r = &mut rows[local_col as usize];
                r.0.push(row as u32);
                r.1.push(v);
            }
        })?;
        for (f, c) in &mut rows {
            let mut idx: Vec<usize> = (0..f.len()).collect();
            idx.sort_unstable_by_key(|&i| f[i]);
            *f = idx.iter().map(|&i| f[i]).collect();
            *c = idx.iter().map(|&i| c[i]).collect();
        }
        axes.push(rows);
    }
    Ok(CellGroup {
        cells: (start as u32..end as u32).collect(),
        axes,
    })
}
