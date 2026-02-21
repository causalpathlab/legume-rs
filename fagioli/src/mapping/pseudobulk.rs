use std::collections::HashMap;
use std::ops::AddAssign;
use std::sync::{Arc, Mutex};

use anyhow::{bail, Result};
use log::info;
use nalgebra::{DMatrix, DVector};

use data_beans::sparse_data_visitors::VisitColumnsOps;
use data_beans::sparse_io_vector::SparseIoVec;
use matrix_param::dmatrix_gamma::GammaMatrix;
use matrix_param::traits::TwoStatParam;
use matrix_util::common_io::read_lines_of_words_delim;
use matrix_util::traits::IoOps;

//////////////////////////////////////////
// Cell annotations and membership I/O  //
//////////////////////////////////////////

/// Parsed cell annotations: maps each cell to an individual (and optionally a cell type).
#[derive(Debug, Clone)]
pub struct CellAnnotations {
    /// cell_id -> individual_idx
    pub cell_to_individual: HashMap<Box<str>, usize>,
    /// Ordered individual IDs (index corresponds to individual_idx)
    pub individual_ids: Vec<Box<str>>,
}

/// Read cell annotations from a delimited file (TSV, CSV, or space-separated).
///
/// Supports gzip-compressed files (.gz) and multiple delimiters (tab, comma, space).
/// First line is treated as a header and skipped.
///
/// Requires at least 2 columns: `cell_id individual_id`.
/// A 3rd column (`cell_type`) is accepted but ignored here;
/// use [`build_onehot_membership`] to convert it into a membership matrix.
pub fn read_cell_annotations(path: &str) -> Result<CellAnnotations> {
    info!("Reading cell annotations from {}", path);

    let parsed = read_lines_of_words_delim(path, &['\t', ',', ' '], 0)?;

    let mut individual_to_idx: HashMap<Box<str>, usize> = HashMap::new();
    let mut individual_ids: Vec<Box<str>> = Vec::new();
    let mut cell_to_individual: HashMap<Box<str>, usize> = HashMap::new();

    for words in &parsed.lines {
        if words.len() < 2 {
            continue;
        }

        let cell_id = words[0].clone();
        let ind_id = words[1].clone();

        let ind_idx = *individual_to_idx.entry(ind_id.clone()).or_insert_with(|| {
            let idx = individual_ids.len();
            individual_ids.push(ind_id);
            idx
        });

        cell_to_individual.insert(cell_id, ind_idx);
    }

    info!(
        "Loaded {} cell annotations: {} individuals",
        cell_to_individual.len(),
        individual_ids.len(),
    );

    Ok(CellAnnotations {
        cell_to_individual,
        individual_ids,
    })
}

/// Infer cell annotations from cell names by splitting on `@`.
///
/// Cell names like `ACGT@IND_A` map to individual `IND_A`.
/// Names without `@` are assigned to a single individual `"all"`.
pub fn infer_cell_annotations(column_names: &[Box<str>]) -> CellAnnotations {
    info!("Inferring individuals from cell names (barcode@indiv)");

    let mut individual_to_idx: HashMap<Box<str>, usize> = HashMap::new();
    let mut individual_ids: Vec<Box<str>> = Vec::new();
    let mut cell_to_individual: HashMap<Box<str>, usize> = HashMap::new();

    for cell_name in column_names {
        let indiv: Box<str> = if let Some(pos) = cell_name.rfind('@') {
            Box::from(&cell_name[pos + 1..])
        } else {
            Box::from("all")
        };
        let idx = *individual_to_idx.entry(indiv.clone()).or_insert_with(|| {
            let i = individual_ids.len();
            individual_ids.push(indiv);
            i
        });
        cell_to_individual.insert(cell_name.clone(), idx);
    }

    info!(
        "Inferred {} individuals from cell names",
        individual_ids.len()
    );

    CellAnnotations {
        cell_to_individual,
        individual_ids,
    }
}

/// Membership matrix with cell type names.
pub struct Membership {
    /// Membership matrix: cells × cell_types (aligned to SC backend column order)
    pub matrix: DMatrix<f32>,
    /// Ordered cell type names (columns of the membership matrix)
    pub cell_type_names: Vec<Box<str>>,
}

/// Build a one-hot membership matrix from hard cell-type annotations.
///
/// Reads a delimited file with 3+ columns: `cell_id individual_id cell_type`.
/// The resulting matrix has one-hot rows aligned to the SC backend `column_names`.
pub fn build_onehot_membership(path: &str, column_names: &[Box<str>]) -> Result<Membership> {
    info!("Building one-hot membership from {}", path);

    let parsed = read_lines_of_words_delim(path, &['\t', ',', ' '], 0)?;

    let mut celltype_to_idx: HashMap<Box<str>, usize> = HashMap::new();
    let mut cell_type_names: Vec<Box<str>> = Vec::new();
    let mut cell_to_ct: HashMap<Box<str>, usize> = HashMap::new();

    for words in &parsed.lines {
        if words.len() < 3 {
            continue;
        }

        let cell_id = words[0].clone();
        let ct_name = words[2].clone();

        let ct_idx = *celltype_to_idx.entry(ct_name.clone()).or_insert_with(|| {
            let idx = cell_type_names.len();
            cell_type_names.push(ct_name);
            idx
        });

        cell_to_ct.insert(cell_id, ct_idx);
    }

    let n_cells = column_names.len();
    let n_ct = cell_type_names.len();
    let mut matrix = DMatrix::<f32>::zeros(n_cells, n_ct);
    let mut matched = 0usize;

    for (cell_idx, cell_name) in column_names.iter().enumerate() {
        if let Some(&ct_idx) = cell_to_ct.get(cell_name) {
            matrix[(cell_idx, ct_idx)] = 1.0;
            matched += 1;
        }
    }

    info!(
        "One-hot membership: {}/{} cells matched, {} cell types",
        matched, n_cells, n_ct
    );

    if matched == 0 {
        bail!("No cells matched between SC backend and annotation file");
    }

    Ok(Membership {
        matrix,
        cell_type_names,
    })
}

/// Read soft cell-type membership proportions from a parquet file.
///
/// The parquet should have cell IDs as row names and cell type names as
/// column headers.  Values are probabilities (rows sum to ~1).
/// Cells are matched to the SC backend by `column_names`.
pub fn read_membership_proportions(
    file_path: &str,
    column_names: &[Box<str>],
) -> Result<Membership> {
    info!("Reading membership proportions from {}", file_path);

    let mat_with_names = DMatrix::<f32>::from_parquet(file_path)?;
    let cell_type_names = mat_with_names.cols;
    let n_cell_types = cell_type_names.len();
    let n_cells = column_names.len();

    // Build lookup: parquet row name -> parquet row index
    let parquet_cell_lookup: HashMap<&str, usize> = mat_with_names
        .rows
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_ref(), i))
        .collect();

    // Build membership matrix aligned to column_names order
    let mut matrix = DMatrix::<f32>::zeros(n_cells, n_cell_types);
    let mut matched = 0usize;

    for (cell_idx, cell_name) in column_names.iter().enumerate() {
        if let Some(&pq_row) = parquet_cell_lookup.get(cell_name.as_ref()) {
            for k in 0..n_cell_types {
                matrix[(cell_idx, k)] = mat_with_names.mat[(pq_row, k)];
            }
            matched += 1;
        }
    }

    info!(
        "Matched {}/{} cells to membership parquet ({} cell types)",
        matched, n_cells, n_cell_types
    );

    if matched == 0 {
        bail!("No cells matched between SC backend and membership parquet");
    }

    Ok(Membership {
        matrix,
        cell_type_names,
    })
}

/////////////////////////////////////////
// Poisson-Gamma pseudobulk collapse   //
/////////////////////////////////////////

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
    mut data_vec: SparseIoVec,
    annotations: &CellAnnotations,
    membership: &Membership,
    a0: f32,
    b0: f32,
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

    // Build individual labels for grouping, skipping unmatched cells
    let mut individual_labels: Vec<Box<str>> = Vec::with_capacity(num_cells);
    let mut matched = 0usize;

    for col_name in &column_names {
        if let Some(&ind_idx) = annotations.cell_to_individual.get(col_name) {
            individual_labels.push(annotations.individual_ids[ind_idx].clone());
            matched += 1;
        } else {
            // Unmatched cells get empty label — will be excluded via filter
            individual_labels.push(Box::from(""));
        }
    }

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
    let n_genes = y_gn.nrows();
    let n_cells = cells.len();
    let n_ct = input.membership.ncols();

    let mut z_nk = DMatrix::<f32>::zeros(n_cells, n_ct);
    for (local_idx, &global_idx) in cells.iter().enumerate() {
        for k in 0..n_ct {
            z_nk[(local_idx, k)] = input.membership[(global_idx, k)];
        }
    }

    let y_gk: DMatrix<f32> = &y_gn * &z_nk;
    let z_k = DVector::from_fn(n_ct, |k, _| z_nk.column(k).sum());

    let mut stat = arc_stat.lock().expect("lock");
    let n_groups = stat.cell_weight.len() / stat.n_cell_types;

    for k in 0..n_ct {
        let col = k * n_groups + group_id;
        for g in 0..n_genes {
            stat.count_sum[(g, col)] += y_gk[(g, k)];
        }
        stat.cell_weight[col] += z_k[k];
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use data_beans::sparse_io::{create_sparse_from_triplets, SparseIoBackend};
    use matrix_param::traits::Inference;

    #[test]
    fn test_read_cell_annotations_roundtrip() -> Result<()> {
        use std::io::Write;

        let dir = tempfile::tempdir()?;
        let path = dir.path().join("cells.tsv");
        let path_str = path.to_str().unwrap();

        let mut f = std::fs::File::create(&path)?;
        writeln!(f, "cell_id\tindividual_id\tcell_type")?;
        writeln!(f, "cell_0\tIND_A\tT_cell")?;
        writeln!(f, "cell_1\tIND_A\tB_cell")?;
        writeln!(f, "cell_2\tIND_B\tT_cell")?;
        writeln!(f, "cell_3\tIND_B\tT_cell")?;
        f.flush()?;

        let anno = read_cell_annotations(path_str)?;
        assert_eq!(anno.individual_ids.len(), 2);
        assert_eq!(anno.cell_to_individual.len(), 4);

        let &ind_idx = anno.cell_to_individual.get(&Box::from("cell_0")).unwrap();
        assert_eq!(anno.individual_ids[ind_idx].as_ref(), "IND_A");

        Ok(())
    }

    fn make_test_data(
        dir: &tempfile::TempDir,
    ) -> Result<(SparseIoVec, CellAnnotations, Membership)> {
        let path = dir.path().join("test.zarr");
        let path_str = path.to_str().unwrap();

        // 3 genes × 6 cells
        // IND_A: cell_0 (T_cell), cell_1 (T_cell), cell_2 (B_cell)
        // IND_B: cell_3 (T_cell), cell_4 (B_cell), cell_5 (B_cell)
        let triplets: Vec<(u64, u64, f32)> = vec![
            (0, 0, 5.0),
            (0, 1, 3.0),
            (0, 2, 2.0),
            (0, 3, 4.0),
            (0, 4, 1.0),
            (0, 5, 6.0),
            (1, 0, 10.0),
            (1, 1, 20.0),
            (1, 3, 15.0),
            (1, 5, 25.0),
            (2, 0, 1.0),
            (2, 2, 7.0),
            (2, 4, 3.0),
        ];

        let gene_names: Vec<Box<str>> = vec!["gene_0".into(), "gene_1".into(), "gene_2".into()];
        let cell_names: Vec<Box<str>> = (0..6).map(|i| format!("cell_{}", i).into()).collect();

        let nnz = triplets.len();
        let mut backend = create_sparse_from_triplets(
            &triplets,
            (3, 6, nnz),
            Some(path_str),
            Some(&SparseIoBackend::Zarr),
        )?;
        backend.register_row_names_vec(&gene_names);
        backend.register_column_names_vec(&cell_names);

        let mut cell_to_individual = HashMap::new();
        cell_to_individual.insert(Box::from("cell_0"), 0); // IND_A
        cell_to_individual.insert(Box::from("cell_1"), 0); // IND_A
        cell_to_individual.insert(Box::from("cell_2"), 0); // IND_A
        cell_to_individual.insert(Box::from("cell_3"), 1); // IND_B
        cell_to_individual.insert(Box::from("cell_4"), 1); // IND_B
        cell_to_individual.insert(Box::from("cell_5"), 1); // IND_B

        let annotations = CellAnnotations {
            cell_to_individual,
            individual_ids: vec![Box::from("IND_A"), Box::from("IND_B")],
        };

        // One-hot: T_cell=col0, B_cell=col1
        // cell_0 T, cell_1 T, cell_2 B, cell_3 T, cell_4 B, cell_5 B
        let membership = Membership {
            matrix: DMatrix::from_row_slice(
                6,
                2,
                &[
                    1.0, 0.0, // cell_0: T
                    1.0, 0.0, // cell_1: T
                    0.0, 1.0, // cell_2: B
                    1.0, 0.0, // cell_3: T
                    0.0, 1.0, // cell_4: B
                    0.0, 1.0, // cell_5: B
                ],
            ),
            cell_type_names: vec![Box::from("T_cell"), Box::from("B_cell")],
        };

        let mut data_vec = SparseIoVec::new();
        data_vec.push(Arc::from(backend), None)?;

        Ok((data_vec, annotations, membership))
    }

    #[test]
    fn test_collapse_hard_membership() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let (data_vec, annotations, membership) = make_test_data(&dir)?;

        let result = collapse_pseudobulk(data_vec, &annotations, &membership, 1.0, 1.0)?;

        assert_eq!(result.individual_ids.len(), 2);
        assert_eq!(result.cell_type_names.len(), 2);
        assert_eq!(result.gene_names.len(), 3);
        assert_eq!(result.gamma_params.len(), 2);

        let t_idx = result
            .cell_type_names
            .iter()
            .position(|x| x.as_ref() == "T_cell")
            .unwrap();
        let b_idx = result
            .cell_type_names
            .iter()
            .position(|x| x.as_ref() == "B_cell")
            .unwrap();
        let a_idx = result
            .individual_ids
            .iter()
            .position(|x| x.as_ref() == "IND_A")
            .unwrap();
        let b_ind_idx = result
            .individual_ids
            .iter()
            .position(|x| x.as_ref() == "IND_B")
            .unwrap();

        // T_cell weights: IND_A=2.0, IND_B=1.0
        assert!((result.cell_weights[t_idx][a_idx] - 2.0).abs() < 1e-6);
        assert!((result.cell_weights[t_idx][b_ind_idx] - 1.0).abs() < 1e-6);

        // B_cell weights: IND_A=1.0, IND_B=2.0
        assert!((result.cell_weights[b_idx][a_idx] - 1.0).abs() < 1e-6);
        assert!((result.cell_weights[b_idx][b_ind_idx] - 2.0).abs() < 1e-6);

        // Posterior mean = (a0 + sum) / (b0 + weight)
        // T_cell, IND_A, gene 0: sum=5+3=8, weight=2 → (1+8)/(1+2) = 3.0
        let t_mean = result.gamma_params[t_idx].posterior_mean();
        let expected = (1.0 + 8.0) / (1.0 + 2.0);
        assert!(
            (t_mean[(0, a_idx)] - expected).abs() < 1e-4,
            "T_cell IND_A gene0: got {}, expected {}",
            t_mean[(0, a_idx)],
            expected
        );

        // T_cell, IND_A, gene 1: sum=10+20=30, weight=2 → (1+30)/(1+2) ≈ 10.33
        let expected = (1.0 + 30.0) / (1.0 + 2.0);
        assert!(
            (t_mean[(1, a_idx)] - expected).abs() < 1e-4,
            "T_cell IND_A gene1: got {}, expected {}",
            t_mean[(1, a_idx)],
            expected
        );

        // B_cell, IND_B, gene 0: sum=1+6=7, weight=2 → (1+7)/(1+2) ≈ 2.67
        let b_mean = result.gamma_params[b_idx].posterior_mean();
        let expected = (1.0 + 7.0) / (1.0 + 2.0);
        assert!(
            (b_mean[(0, b_ind_idx)] - expected).abs() < 1e-4,
            "B_cell IND_B gene0: got {}, expected {}",
            b_mean[(0, b_ind_idx)],
            expected
        );

        Ok(())
    }

    #[test]
    fn test_collapse_soft_membership() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let (data_vec, annotations, _) = make_test_data(&dir)?;

        let soft = Membership {
            matrix: DMatrix::from_row_slice(
                6,
                2,
                &[
                    0.7, 0.3, // cell_0: mostly T
                    0.8, 0.2, // cell_1: mostly T
                    0.1, 0.9, // cell_2: mostly B
                    0.9, 0.1, // cell_3: mostly T
                    0.2, 0.8, // cell_4: mostly B
                    0.3, 0.7, // cell_5: mostly B
                ],
            ),
            cell_type_names: vec![Box::from("T_cell"), Box::from("B_cell")],
        };

        let result = collapse_pseudobulk(data_vec, &annotations, &soft, 1.0, 1.0)?;

        let t_idx = result
            .cell_type_names
            .iter()
            .position(|x| x.as_ref() == "T_cell")
            .unwrap();
        let a_idx = result
            .individual_ids
            .iter()
            .position(|x| x.as_ref() == "IND_A")
            .unwrap();

        // T_cell weight for IND_A = 0.7 + 0.8 + 0.1 = 1.6
        let expected_weight = 0.7 + 0.8 + 0.1;
        assert!(
            (result.cell_weights[t_idx][a_idx] - expected_weight).abs() < 1e-4,
            "T_cell IND_A weight: got {}, expected {}",
            result.cell_weights[t_idx][a_idx],
            expected_weight
        );

        let expected_sum = 5.0 * 0.7 + 3.0 * 0.8 + 2.0 * 0.1;
        let expected_mean = (1.0 + expected_sum) / (1.0 + expected_weight);
        let t_mean = result.gamma_params[t_idx].posterior_mean();
        assert!(
            (t_mean[(0, a_idx)] - expected_mean).abs() < 1e-3,
            "Soft T_cell IND_A gene0: got {}, expected {}",
            t_mean[(0, a_idx)],
            expected_mean
        );

        Ok(())
    }

    #[test]
    fn test_collapse_posterior_sd_decreases_with_more_cells() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let (data_vec, annotations, membership) = make_test_data(&dir)?;

        let result = collapse_pseudobulk(data_vec, &annotations, &membership, 1.0, 1.0)?;

        let t_idx = result
            .cell_type_names
            .iter()
            .position(|x| x.as_ref() == "T_cell")
            .unwrap();
        let a_idx = result
            .individual_ids
            .iter()
            .position(|x| x.as_ref() == "IND_A")
            .unwrap();
        let b_idx = result
            .individual_ids
            .iter()
            .position(|x| x.as_ref() == "IND_B")
            .unwrap();

        let t_sd = result.gamma_params[t_idx].posterior_sd();

        // IND_A has 2 T_cells, IND_B has 1 T_cell → IND_A should have lower SD
        assert!(
            t_sd[(0, a_idx)] < t_sd[(0, b_idx)],
            "More cells should give lower SD: IND_A={} vs IND_B={}",
            t_sd[(0, a_idx)],
            t_sd[(0, b_idx)]
        );

        Ok(())
    }
}
