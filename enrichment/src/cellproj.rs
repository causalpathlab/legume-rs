//! Cell-level projection: P = cell_membership · Q, row-normalize, argmax.

use crate::Mat;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelWithConfidence {
    pub cell_name: Box<str>,
    pub label: Box<str>,
    pub confidence: f32,
}

/// Compute cell × celltype posterior matrix from (cell × K) membership and
/// (K × C) Q. Rows are re-normalized; zero-mass rows stay zero.
pub fn label_cells(
    cell_membership_nk: &Mat,
    q_kc: &Mat,
    cell_names: &[Box<str>],
    celltype_names: &[Box<str>],
    min_confidence: f32,
) -> (Mat, Vec<LabelWithConfidence>) {
    let mut posterior = cell_membership_nk * q_kc;
    let n = posterior.nrows();
    let c = posterior.ncols();
    for i in 0..n {
        let s: f32 = (0..c).map(|j| posterior[(i, j)]).sum();
        if s > 1e-12 {
            for j in 0..c {
                posterior[(i, j)] /= s;
            }
        }
    }

    let mut labels: Vec<LabelWithConfidence> = Vec::with_capacity(n);
    for (i, cell_name) in cell_names.iter().enumerate().take(n) {
        let row = posterior.row(i);
        let s: f32 = row.iter().sum();
        if s < 1e-12 {
            labels.push(LabelWithConfidence {
                cell_name: cell_name.clone(),
                label: "unassigned".into(),
                confidence: 0.0,
            });
            continue;
        }
        let (argmax, max_val) = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        let label = if *max_val >= min_confidence {
            celltype_names[argmax].clone()
        } else {
            "unassigned".into()
        };
        labels.push(LabelWithConfidence {
            cell_name: cell_name.clone(),
            label,
            confidence: *max_val,
        });
    }
    (posterior, labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hard_membership_picks_right_celltype() {
        // Two cells, K=3, C=2. Cell 0 is all in topic 0 (→ celltype 0).
        // Cell 1 is all in topic 2 (→ celltype 1).
        let membership = Mat::from_row_slice(2, 3, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        let q = Mat::from_row_slice(3, 2, &[1.0, 0.0, 0.5, 0.5, 0.0, 1.0]);
        let names = vec!["cell0".into(), "cell1".into()];
        let cts: Vec<Box<str>> = vec!["A".into(), "B".into()];
        let (_p, labels) = label_cells(&membership, &q, &names, &cts, 0.5);
        assert_eq!(labels[0].label.as_ref(), "A");
        assert_eq!(labels[1].label.as_ref(), "B");
    }
}
