use rustc_hash::FxHashMap as HashMap;
use std::ops::AddAssign;
use std::sync::{Arc, Mutex};

use anyhow::{bail, Result};
use log::info;
use nalgebra::{DMatrix, DVector};

use auxiliary_data::cell_annotations::{CellAnnotations, CellTypeMembership};
use data_beans::sparse_data_visitors::VisitColumnsOps;
use data_beans::sparse_io_vector::SparseIoVec;
use matrix_param::dmatrix_gamma::GammaMatrix;
use matrix_param::traits::TwoStatParam;

/// Accumulates gene-level count sums per (individual × cell_type).
///
/// Layout: genes × (n_individuals * n_cell_types)
/// Column index for (individual i, cell_type k) = k * n_individuals + i
struct CollapseStat {
    /// Weighted sum of counts: genes × (n_indv * n_ct)
    count_sum: DMatrix<f32>,
    /// Effective cell count per (individual, cell_type): n_indv * n_ct
    cell_weight: DVector<f32>,
    /// Number of cell types
    n_cell_types: usize,
}

/// Input to the visitor: cell-level membership matrix.
struct VisitorInput {
    /// Cell membership: cells × cell_types.
    membership: DMatrix<f32>,
}

/// Poisson-Gamma pseudobulk output.
#[derive(Debug)]
pub struct CollapsedPseudobulk {
    /// Per cell type: GammaMatrix (genes × individuals)
    pub gamma_params: Vec<GammaMatrix>,
    /// Ordered cell type names
    pub cell_type_names: Vec<Box<str>>,
    /// Ordered individual IDs
    pub individual_ids: Vec<Box<str>>,
    /// Gene names
    pub gene_names: Vec<Box<str>>,
    /// Effective cell count per (cell_type_idx, individual_idx) — fractional for soft membership
    pub cell_weights: Vec<Vec<f32>>,
}

/// Collapse single-cell counts into Poisson-Gamma pseudobulk parameters.
///
/// Groups cells by individual, then weights each cell's expression by its
/// cell type membership (hard one-hot or soft probabilities).
///
/// # Arguments
/// * `data_vec` - Sparse single-cell count matrix (SparseIoVec over one or more backends)
/// * `annotations` - Cell-to-individual mapping
/// * `membership` - Membership matrix (cells × cell_types) and cell type names
/// * `a0` - Gamma prior shape parameter (default: 1.0)
/// * `b0` - Gamma prior rate parameter (default: 1.0)
pub fn collapse_pseudobulk(
    data_vec: SparseIoVec,
    annotations: &CellAnnotations,
    membership: &CellTypeMembership,
    a0: f32,
    b0: f32,
) -> Result<CollapsedPseudobulk> {
    collapse_pseudobulk_weighted(data_vec, annotations, membership, a0, b0, None)
}

/// Same as [`collapse_pseudobulk`], but row-scales the per-(individual,
/// cell_type) count sums by `gene_weights[g]` before the Gamma update. This
/// is the NB-Fisher housekeeping adjustment used by pinto (see
/// `gene_weighting::compute_nb_fisher_weights`); housekeeping genes get
/// attenuated in the posterior mean while informative genes stay at w≈1.
pub fn collapse_pseudobulk_weighted(
    mut data_vec: SparseIoVec,
    annotations: &CellAnnotations,
    membership: &CellTypeMembership,
    a0: f32,
    b0: f32,
    gene_weights: Option<&[f32]>,
) -> Result<CollapsedPseudobulk> {
    let num_genes = data_vec.num_rows();
    let num_cells = data_vec.num_columns();
    let n_cell_types = membership.cell_type_names.len();

    if num_genes == 0 || num_cells == 0 {
        bail!(
            "SC backend has no data (rows={}, cols={})",
            num_genes,
            num_cells
        );
    }

    if membership.matrix.nrows() != num_cells || membership.matrix.ncols() != n_cell_types {
        bail!(
            "Membership matrix shape ({}, {}) doesn't match (cells={}, cell_types={})",
            membership.matrix.nrows(),
            membership.matrix.ncols(),
            num_cells,
            n_cell_types
        );
    }

    let column_names = data_vec.column_names()?;
    let gene_names = data_vec.row_names()?;

    info!("SC backend: {} genes × {} cells", num_genes, num_cells);

    let individual_labels = annotations.to_column_aligned_vec(&column_names, "");
    let matched = individual_labels.iter().filter(|s| !s.is_empty()).count();

    info!("Matched {}/{} cells to annotations", matched, num_cells);

    if matched == 0 {
        bail!("No cells matched between SC backend and annotations");
    }

    // Assign groups by individual
    data_vec.assign_groups(&individual_labels, None);

    let group_keys = data_vec
        .group_keys()
        .ok_or_else(|| anyhow::anyhow!("groups not assigned"))?
        .clone();

    // Map group_key -> individual_idx, skipping unmatched cells (empty key)
    let ind_lookup: HashMap<&str, usize> = annotations
        .individual_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (id.as_ref(), i))
        .collect();

    let n_groups = group_keys.len();
    let n_individuals = annotations.individual_ids.len();

    info!(
        "Created {} individual groups, {} cell types",
        n_groups, n_cell_types
    );

    // Initialize accumulator: genes × (n_groups * n_cell_types)
    let stat_cols = n_groups * n_cell_types;
    let mut stat = CollapseStat {
        count_sum: DMatrix::zeros(num_genes, stat_cols),
        cell_weight: DVector::zeros(stat_cols),
        n_cell_types,
    };

    let visitor_input = VisitorInput {
        membership: membership.matrix.clone(),
    };

    // Visit groups in parallel
    data_vec.visit_columns_by_group(&collect_stat_visitor, &visitor_input, &mut stat)?;

    // Build GammaMatrix per cell type
    let mut gamma_params: Vec<GammaMatrix> = Vec::with_capacity(n_cell_types);
    let mut cell_weights: Vec<Vec<f32>> = Vec::with_capacity(n_cell_types);

    // Map group_keys back to individual indices, skipping unmatched group
    let group_to_ind: Vec<Option<usize>> = group_keys
        .iter()
        .map(|key| ind_lookup.get(key.as_ref()).copied())
        .collect();

    if let Some(w) = gene_weights {
        if w.len() != num_genes {
            bail!("gene_weights length {} != num_genes {}", w.len(), num_genes);
        }
    }

    for ct_idx in 0..n_cell_types {
        let mut count_sum_ct = DMatrix::<f32>::zeros(num_genes, n_individuals);
        let mut weight_ct = DVector::<f32>::zeros(n_individuals);

        for (group_idx, ind_opt) in group_to_ind.iter().enumerate() {
            let Some(&ind_idx) = ind_opt.as_ref() else {
                continue;
            };
            let src_col = ct_idx * n_groups + group_idx;
            count_sum_ct
                .column_mut(ind_idx)
                .add_assign(&stat.count_sum.column(src_col));
            weight_ct[ind_idx] += stat.cell_weight[src_col];
        }

        if let Some(w) = gene_weights {
            for (g, &wg) in w.iter().enumerate() {
                count_sum_ct.row_mut(g).scale_mut(wg);
            }
        }

        let denom = DMatrix::from_fn(num_genes, n_individuals, |_g, i| weight_ct[i]);

        let mut gamma = GammaMatrix::new((num_genes, n_individuals), a0, b0);
        gamma.update_stat(&count_sum_ct, &denom);
        gamma.calibrate();

        let cw: Vec<f32> = weight_ct.iter().copied().collect();
        let n_with_cells = cw.iter().filter(|&&c| c > 0.0).count();
        let total_weight: f32 = cw.iter().sum();

        info!(
            "Cell type {}: {} individuals with cells, total weight {:.1}",
            membership.cell_type_names[ct_idx], n_with_cells, total_weight
        );

        gamma_params.push(gamma);
        cell_weights.push(cw);
    }

    Ok(CollapsedPseudobulk {
        gamma_params,
        cell_type_names: membership.cell_type_names.clone(),
        individual_ids: annotations.individual_ids.clone(),
        gene_names,
        cell_weights,
    })
}

/// Visitor: for each individual group, multiply sparse counts by membership
/// weights and accumulate.
///
/// Computes: `y_gk = Y_gn * Z_nk` (genes × cell_types) per individual
fn collect_stat_visitor(
    group_id: usize,
    cells: &[usize],
    data: &SparseIoVec,
    input: &VisitorInput,
    arc_stat: Arc<Mutex<&mut CollapseStat>>,
) -> Result<()> {
    if cells.is_empty() {
        return Ok(());
    }

    let y_gn = data.read_columns_csc(cells.iter().cloned())?;
    let n_cells = cells.len();
    let n_ct = input.membership.ncols();

    let mut z_nk = DMatrix::<f32>::zeros(n_cells, n_ct);
    for (local_idx, &global_idx) in cells.iter().enumerate() {
        z_nk.row_mut(local_idx)
            .copy_from(&input.membership.row(global_idx));
    }

    let y_gk: DMatrix<f32> = &y_gn * &z_nk;
    let z_k = DVector::from_fn(n_ct, |k, _| z_nk.column(k).sum());

    let mut stat = arc_stat.lock().expect("lock");
    let n_groups = stat.cell_weight.len() / stat.n_cell_types;

    for k in 0..n_ct {
        let col = k * n_groups + group_id;
        stat.count_sum.column_mut(col).add_assign(&y_gk.column(k));
        stat.cell_weight[col] += z_k[k];
    }

    Ok(())
}

#[cfg(test)]
#[path = "pseudobulk_tests.rs"]
mod tests;
