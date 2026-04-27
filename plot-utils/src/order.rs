//! Row/column orderings for matrix-style summary plots.
//!
//! Used by Hinton diagrams, GSEA tables, and any other (rows × cols)
//! layout where readers expect mass to cluster along the diagonal.

/// Diagonalizing permutation: places high-magnitude entries on the main
/// diagonal so blocks read as bands.
///
/// Two-pass — same recipe as `Util.R::order.pair`:
/// 1. Sort rows by argmax column (rows whose peak is in early columns
///    come first). Ties broken by row index for determinism.
/// 2. Sort columns by the *position* of their argmax row in the
///    row-sorted order — column whose peak appears at row position 0
///    moves to col position 0, etc.
///
/// Non-finite entries are treated as `-∞` (never the argmax). Matrices
/// with all-NaN rows/cols still get a stable order via the tie-breaker.
///
/// `mat` is row-major (`mat[r * ncols + c]`).
#[must_use]
pub fn diagonalize_order(mat: &[f32], nrows: usize, ncols: usize) -> (Vec<usize>, Vec<usize>) {
    debug_assert_eq!(mat.len(), nrows * ncols);
    if nrows == 0 || ncols == 0 {
        return ((0..nrows).collect(), (0..ncols).collect());
    }

    let argmax_col = |r: usize| -> usize {
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for c in 0..ncols {
            let v = mat[r * ncols + c];
            if v.is_finite() && v > bv {
                bv = v;
                best = c;
            }
        }
        best
    };
    let mut row_order: Vec<usize> = (0..nrows).collect();
    row_order.sort_by_key(|&r| (argmax_col(r), r));

    let argmax_row_pos = |c: usize, ro: &[usize]| -> usize {
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (pos, &r) in ro.iter().enumerate() {
            let v = mat[r * ncols + c];
            if v.is_finite() && v > bv {
                bv = v;
                best = pos;
            }
        }
        best
    };
    let mut col_order: Vec<usize> = (0..ncols).collect();
    col_order.sort_by_key(|&c| (argmax_row_pos(c, &row_order), c));

    (row_order, col_order)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagonalizes_block_matrix() {
        // Three "blocks": rows {0,3} peak at col 0; {1,4} at col 1; {2} at col 2.
        // Original column order: 2, 0, 1 → expect col_order to swap them
        // so peaks line up with the row order.
        #[rustfmt::skip]
        let mat = [
            0.1, 0.9, 0.0,  // row 0 peaks col 1 (originally)
            0.9, 0.1, 0.0,  // row 1 peaks col 0
            0.0, 0.0, 1.0,  // row 2 peaks col 2
            0.2, 0.8, 0.0,  // row 3 peaks col 1
            0.8, 0.2, 0.0,  // row 4 peaks col 0
        ];
        let (ro, co) = diagonalize_order(&mat, 5, 3);
        assert_eq!(ro.len(), 5);
        assert_eq!(co.len(), 3);
        // First two rows should peak at col_order[0], next two at col_order[1], last at col_order[2].
        let peak_in = |r: usize, target_col: usize| -> bool {
            let row = r * 3;
            (0..3).all(|c| {
                let v = mat[row + co[c]];
                if c == target_col {
                    v >= mat[row + co[(target_col + 1) % 3]]
                } else {
                    true
                }
            })
        };
        assert!(peak_in(ro[0], 0));
        assert!(peak_in(ro[1], 0));
        assert!(peak_in(ro[2], 1));
        assert!(peak_in(ro[3], 1));
        assert!(peak_in(ro[4], 2));
    }

    #[test]
    fn handles_empty() {
        let (r, c) = diagonalize_order(&[], 0, 0);
        assert!(r.is_empty() && c.is_empty());
    }

    #[test]
    fn handles_nan() {
        let mat = [f32::NAN, 1.0, 0.0, f32::NAN];
        let (r, c) = diagonalize_order(&mat, 2, 2);
        assert_eq!(r.len(), 2);
        assert_eq!(c.len(), 2);
    }
}
