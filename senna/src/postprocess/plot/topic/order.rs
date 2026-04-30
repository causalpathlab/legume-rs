//! Cell and topic ordering helpers used by the structure-bar layout:
//! a global topic-prevalence ranking and the per-batch cell sort
//! (argmax-based or coordinate-based).

use crate::embed_common::*;

/// Global display order for the K topic columns: descending total
/// prevalence (sum of probabilities across all cells). The most
/// prevalent topic comes first, so the same color block lands at the
/// same horizontal position in every batch panel — gives the structure
/// plot visual continuity across batches without changing the topic ↔
/// color identity (which is still keyed by topic-id, see `topic_colors`
/// in the entry-point module).
pub(super) fn global_topic_order(probs: &[f32], n_topics: usize) -> Vec<usize> {
    if n_topics == 0 || probs.is_empty() {
        return (0..n_topics).collect();
    }
    let n_cells = probs.len() / n_topics;
    let mut totals = vec![0.0f32; n_topics];
    for i in 0..n_cells {
        let row = &probs[i * n_topics..(i + 1) * n_topics];
        for (j, &v) in row.iter().enumerate() {
            if v.is_finite() && v > 0.0 {
                totals[j] += v;
            }
        }
    }
    let mut order: Vec<usize> = (0..n_topics).collect();
    order.sort_by(|&a, &b| {
        totals[b]
            .partial_cmp(&totals[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    order
}

/// Argmax-then-dominant-prob ordering inside one batch's cell list. The
/// primary sort key is the *display rank* of each cell's argmax topic
/// (so cells dominated by the same topic land at the same horizontal
/// position across batches); secondary key is descending dominant prob.
pub(super) fn order_by_argmax(
    cells: &[usize],
    probs: &[f32],
    n_topics: usize,
    topic_rank: &[usize],
) -> Vec<usize> {
    let mut keyed: Vec<(usize, usize, f32)> = cells
        .iter()
        .map(|&i| {
            let row = &probs[i * n_topics..(i + 1) * n_topics];
            let mut best_j = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for (j, &v) in row.iter().enumerate() {
                if v > best_v {
                    best_v = v;
                    best_j = j;
                }
            }
            (i, topic_rank[best_j], best_v)
        })
        .collect();
    keyed.sort_by(|a, b| {
        a.1.cmp(&b.1)
            .then(b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal))
    });
    keyed.into_iter().map(|(i, _, _)| i).collect()
}

/// Sort cells by ascending x from `cell_coords.parquet`. Falls back to
/// argmax order if the file is missing or has no `x` column.
pub(super) fn order_by_coord(
    cells: &[usize],
    cell_coords_path: Option<&str>,
    probs: &[f32],
    n_topics: usize,
    topic_rank: &[usize],
) -> anyhow::Result<Vec<usize>> {
    let Some(path) = cell_coords_path else {
        info!("--order coord: no layout.cell_coords in manifest, falling back to argmax");
        return Ok(order_by_argmax(cells, probs, n_topics, topic_rank));
    };
    let MatWithNames { cols, mat, .. } = Mat::from_parquet(path)?;
    let Some(xj) = cols.iter().position(|c| &**c == "x") else {
        info!("--order coord: no 'x' column in {path}, falling back to argmax");
        return Ok(order_by_argmax(cells, probs, n_topics, topic_rank));
    };
    if mat.nrows() < cells.iter().copied().max().unwrap_or(0) + 1 {
        info!("--order coord: cell_coords too short, falling back to argmax");
        return Ok(order_by_argmax(cells, probs, n_topics, topic_rank));
    }
    let mut keyed: Vec<(usize, f32)> = cells.iter().map(|&i| (i, mat[(i, xj)])).collect();
    keyed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    Ok(keyed.into_iter().map(|(i, _)| i).collect())
}
