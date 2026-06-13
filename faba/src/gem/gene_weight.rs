//! Per-gene **ubiquity** diagnostic.
//!
//! Abundance/housekeeping balance is handled entirely by the sampler's
//! `count^τ` tempering (`--tau`): with τ=1 positives are drawn strictly
//! count-proportionally; lowering τ flattens the draw toward uniform over
//! expressed `(gene, cell)` entries, so the highest-count housekeeping /
//! ribosomal genes stop monopolising the gradient. There is no separate
//! down-weighting term — a second, opposing knob on the same axis.
//!
//! This module computes per-gene **ubiquity** (`ubiquity_from_count_pool`):
//! the fraction of cells expressing a gene — a *breadth* diagnostic written
//! to `{out}.ubiquity.parquet`; not consumed by the model.

use super::pseudobulk::StratumPool;

/// Per-gene **ubiquity** `u_g ∈ (0, 1]` = fraction of cells expressing
/// gene `g`, derived from the **cell-axis** count-comp pool. Each pool
/// entry is one nonzero `(gene, cell)` total (`aggregate_pools` keys on
/// `(gene_id, axis_id)`), so the number of entries per gene is exactly the
/// count of cells with nonzero expression. Divided by `n_cells`.
///
/// Ubiquity separates genes that are high *everywhere* (true housekeeping:
/// MT-/RP-, u ≈ 1) from genes that are high in *one* lineage (Hb: u ≈ 0.1).
/// Written to `{out}.ubiquity.parquet` as a *breadth* diagnostic.
pub fn ubiquity_from_count_pool(
    count_comp: &StratumPool,
    n_genes: usize,
    n_cells: usize,
) -> Vec<f32> {
    let mut cells_with = vec![0u32; n_genes];
    for i in 0..count_comp.len() {
        let g = count_comp.gene_ids[i] as usize;
        if g < n_genes {
            cells_with[g] += 1;
        }
    }
    let inv_n = 1.0 / n_cells.max(1) as f32;
    cells_with
        .iter()
        .map(|&c| (c as f32 * inv_n).clamp(0.0, 1.0))
        .collect()
}
