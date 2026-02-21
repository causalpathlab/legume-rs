use std::collections::HashMap;

use anyhow::{bail, Result};
use log::info;
use nalgebra::DMatrix;

use matrix_util::common_io::read_lines_of_words_delim;
use matrix_util::traits::IoOps;

use crate::mapping::pseudobulk::{CellAnnotations, Membership};

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
