use std::collections::HashMap;
use std::io::{BufRead, BufReader};

use anyhow::{bail, Result};
use flate2::read::GzDecoder;
use log::info;
use nalgebra::DMatrix;

use data_beans::sparse_io::SparseIo;

/// Parsed cell annotations mapping each cell to an individual and cell type.
#[derive(Debug, Clone)]
pub struct CellAnnotations {
    /// cell_id -> (individual_idx, cell_type_idx)
    pub cell_map: HashMap<Box<str>, (usize, usize)>,
    /// Ordered individual IDs (index corresponds to individual_idx)
    pub individual_ids: Vec<Box<str>>,
    /// Ordered cell type names (index corresponds to cell_type_idx)
    pub cell_type_names: Vec<Box<str>>,
}

/// Pseudobulk expression data aggregated from single-cell counts.
#[derive(Debug)]
pub struct PseudobulkData {
    /// One matrix per cell type: N_individuals × N_genes
    pub expression: Vec<DMatrix<f32>>,
    /// Cell counts per (cell_type, individual), same shape as expression matrices
    pub cell_counts: Vec<DMatrix<f32>>,
    /// Ordered cell type names
    pub cell_type_names: Vec<Box<str>>,
    /// Ordered individual IDs
    pub individual_ids: Vec<Box<str>>,
    /// Gene names (row names from the SC backend)
    pub gene_names: Vec<Box<str>>,
}

/// Read cell annotations from a TSV.GZ file.
///
/// Expected columns: cell_id, individual_id, cell_type
/// (tab-separated, first line is header)
pub fn read_cell_annotations(path: &str) -> Result<CellAnnotations> {
    let file = std::fs::File::open(path)?;

    let reader: Box<dyn BufRead> = if path.ends_with(".gz") {
        Box::new(BufReader::new(GzDecoder::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };

    let mut individual_to_idx: HashMap<Box<str>, usize> = HashMap::new();
    let mut celltype_to_idx: HashMap<Box<str>, usize> = HashMap::new();
    let mut individual_ids: Vec<Box<str>> = Vec::new();
    let mut cell_type_names: Vec<Box<str>> = Vec::new();
    let mut cell_map: HashMap<Box<str>, (usize, usize)> = HashMap::new();

    let mut lines = reader.lines();

    // Skip header
    let header = lines.next().ok_or_else(|| anyhow::anyhow!("Empty annotation file"))??;
    let hdr_fields: Vec<&str> = header.split('\t').collect();
    if hdr_fields.len() < 3 {
        bail!(
            "Cell annotation file must have at least 3 tab-separated columns (cell_id, individual_id, cell_type), got {}",
            hdr_fields.len()
        );
    }

    for line in lines {
        let line = line?;
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 3 {
            continue;
        }

        let cell_id: Box<str> = Box::from(fields[0]);
        let ind_id: Box<str> = Box::from(fields[1]);
        let ct_name: Box<str> = Box::from(fields[2]);

        let ind_idx = *individual_to_idx.entry(ind_id.clone()).or_insert_with(|| {
            let idx = individual_ids.len();
            individual_ids.push(ind_id);
            idx
        });

        let ct_idx = *celltype_to_idx.entry(ct_name.clone()).or_insert_with(|| {
            let idx = cell_type_names.len();
            cell_type_names.push(ct_name);
            idx
        });

        cell_map.insert(cell_id, (ind_idx, ct_idx));
    }

    info!(
        "Loaded {} cell annotations: {} individuals, {} cell types",
        cell_map.len(),
        individual_ids.len(),
        cell_type_names.len()
    );

    Ok(CellAnnotations {
        cell_map,
        individual_ids,
        cell_type_names,
    })
}

/// Aggregate single-cell counts into pseudobulk expression per cell type.
///
/// For each cell type, produces an N_individuals × N_genes matrix of summed counts.
/// Individuals with fewer than `min_cells` in a given cell type get zeros
/// (use the `cell_counts` matrix to filter downstream).
pub fn aggregate_pseudobulk(
    sc_backend: &dyn SparseIo<IndexIter = Vec<usize>>,
    annotations: &CellAnnotations,
    min_cells: usize,
) -> Result<PseudobulkData> {
    let num_genes = sc_backend.num_rows().unwrap_or(0);
    let num_cells = sc_backend.num_columns().unwrap_or(0);
    let num_individuals = annotations.individual_ids.len();
    let num_cell_types = annotations.cell_type_names.len();

    if num_genes == 0 || num_cells == 0 {
        bail!("SC backend has no data (rows={}, cols={})", num_genes, num_cells);
    }

    // Get column names (cell IDs) from backend
    let column_names = sc_backend.column_names()?;
    let gene_names = sc_backend.row_names()?;

    info!(
        "SC backend: {} genes × {} cells",
        num_genes, num_cells
    );

    // Map backend column index -> (individual_idx, cell_type_idx)
    let mut col_annotations: Vec<Option<(usize, usize)>> = Vec::with_capacity(num_cells);
    let mut matched = 0usize;
    for col_name in &column_names {
        if let Some(&(ind_idx, ct_idx)) = annotations.cell_map.get(col_name) {
            col_annotations.push(Some((ind_idx, ct_idx)));
            matched += 1;
        } else {
            col_annotations.push(None);
        }
    }

    info!(
        "Matched {}/{} cells to annotations",
        matched, num_cells
    );

    if matched == 0 {
        bail!("No cells matched between SC backend and annotations");
    }

    // Initialize pseudobulk matrices (N_ind × N_genes) per cell type
    let mut expression: Vec<DMatrix<f32>> = (0..num_cell_types)
        .map(|_| DMatrix::zeros(num_individuals, num_genes))
        .collect();

    // Count cells per (cell_type, individual) for min_cells filtering
    let mut cells_per_ct_ind: Vec<Vec<usize>> = vec![vec![0; num_individuals]; num_cell_types];

    // Read data by column (cell) using triplets
    // Process in batches to avoid memory issues
    let batch_size = 1000;
    for batch_start in (0..num_cells).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(num_cells);
        let batch_cols: Vec<usize> = (batch_start..batch_end).collect();

        let (_nrow, _ncol, triplets) = sc_backend.read_triplets_by_columns(batch_cols)?;

        for (row_idx, col_idx, value) in triplets {
            let global_col = batch_start + col_idx as usize;
            if let Some((ind_idx, ct_idx)) = col_annotations[global_col] {
                expression[ct_idx][(ind_idx, row_idx as usize)] += value;
            }
        }

        // Track cell counts for this batch
        for local_col in 0..(batch_end - batch_start) {
            let global_col = batch_start + local_col;
            if let Some((ind_idx, ct_idx)) = col_annotations[global_col] {
                cells_per_ct_ind[ct_idx][ind_idx] += 1;
            }
        }
    }

    // Build cell counts matrix (same shape as expression, but stores per-individual cell count
    // replicated across genes for convenience)
    let cell_counts_mat: Vec<DMatrix<f32>> = (0..num_cell_types)
        .map(|ct| {
            let mut mat = DMatrix::zeros(num_individuals, num_genes);
            for ind in 0..num_individuals {
                let count = cells_per_ct_ind[ct][ind] as f32;
                for gene in 0..num_genes {
                    mat[(ind, gene)] = count;
                }
            }
            mat
        })
        .collect();

    // Zero out individuals with < min_cells
    for ct in 0..num_cell_types {
        for ind in 0..num_individuals {
            if cells_per_ct_ind[ct][ind] < min_cells {
                for gene in 0..num_genes {
                    expression[ct][(ind, gene)] = 0.0;
                }
            }
        }
    }

    let total_cells: usize = cells_per_ct_ind.iter().flat_map(|v| v.iter()).sum();
    info!(
        "Pseudobulk aggregation complete: {} cell types, {} individuals, {} genes, {} total cell assignments",
        num_cell_types, num_individuals, num_genes, total_cells
    );

    Ok(PseudobulkData {
        expression,
        cell_counts: cell_counts_mat,
        cell_type_names: annotations.cell_type_names.clone(),
        individual_ids: annotations.individual_ids.clone(),
        gene_names,
    })
}

/// Get a boolean mask of individuals with >= min_cells for a given cell type.
pub fn individual_mask(
    cells_per_ind: &DMatrix<f32>,
    min_cells: f32,
) -> Vec<bool> {
    // cells_per_ind is N_ind × N_genes, but cell count is same across genes;
    // just check column 0
    (0..cells_per_ind.nrows())
        .map(|i| cells_per_ind[(i, 0)] >= min_cells)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(anno.cell_type_names.len(), 2);
        assert_eq!(anno.cell_map.len(), 4);

        let (ind_idx, ct_idx) = anno.cell_map[&Box::from("cell_0")];
        assert_eq!(anno.individual_ids[ind_idx].as_ref(), "IND_A");
        assert_eq!(anno.cell_type_names[ct_idx].as_ref(), "T_cell");

        Ok(())
    }

    #[test]
    fn test_individual_mask() {
        let counts = DMatrix::from_row_slice(3, 2, &[10.0, 10.0, 3.0, 3.0, 5.0, 5.0]);
        let mask = individual_mask(&counts, 5.0);
        assert_eq!(mask, vec![true, false, true]);
    }
}
