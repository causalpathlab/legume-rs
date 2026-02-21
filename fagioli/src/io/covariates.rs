use anyhow::Result;
use log::info;
use nalgebra::DMatrix;

/// Load additional covariate files and return their matrices, matched and centered.
pub fn load_covariate_files(
    covariate_files: &[String],
    matched_individual_ids: &[&str],
    n_matched: usize,
) -> Result<Vec<DMatrix<f32>>> {
    let ind_lookup: std::collections::HashMap<&str, usize> = matched_individual_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    let mut extra = Vec::new();

    for cov_path in covariate_files {
        info!("Loading covariates from {}", cov_path);
        let parsed = matrix_util::common_io::read_lines_of_words_delim(cov_path, &['\t', ','], 0)?;

        let n_cov_cols = if let Some(first_line) = parsed.lines.first() {
            first_line.len().saturating_sub(1)
        } else {
            anyhow::bail!("Empty covariate file: {}", cov_path);
        };

        if n_cov_cols == 0 {
            anyhow::bail!(
                "Covariate file {} has no covariate columns (need ID + values)",
                cov_path
            );
        }

        let mut cov_mat = DMatrix::<f32>::zeros(n_matched, n_cov_cols);
        let mut n_found = 0usize;

        for row in &parsed.lines {
            let id = row[0].as_ref();
            if let Some(&mat_row) = ind_lookup.get(id) {
                for (j, val_str) in row[1..].iter().enumerate() {
                    if j < n_cov_cols {
                        cov_mat[(mat_row, j)] = val_str.parse::<f32>().unwrap_or(0.0);
                    }
                }
                n_found += 1;
            }
        }

        info!(
            "  {} covariates, {}/{} individuals matched",
            n_cov_cols, n_found, n_matched
        );

        center_columns(&mut cov_mat);
        extra.push(cov_mat);
    }

    Ok(extra)
}

/// Center each column of a matrix to zero mean.
fn center_columns(mat: &mut DMatrix<f32>) {
    let n = mat.nrows() as f32;
    for j in 0..mat.ncols() {
        let mean = mat.column(j).sum() / n;
        for i in 0..mat.nrows() {
            mat[(i, j)] -= mean;
        }
    }
}
