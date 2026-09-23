//! The fold-in accumulated cell by cell equals the fold-in over pseudobulk
//! counts: a pseudobulk's counts are the sums of its cells', and the fold-in
//! needs only `Σ_u n_pu ẽ_u` and `Σ_u n_pu` per peak.

use chickpea::p2g::peak_foldin::{fold_in_peaks, FoldInDesign, PeakMoments};
use nalgebra::DMatrix;
use nalgebra_sparse::{CooMatrix, CsrMatrix};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

#[test]
fn accumulating_cells_matches_the_pseudobulk_fold_in() {
    let mut rng = StdRng::seed_from_u64(11);
    let (n_peaks, n_units, n_cells, h) = (15, 6, 40, 3);
    let e = DMatrix::from_fn(n_units, h, |_, _| rng.random::<f32>() - 0.5);
    let pb_of_cell: Vec<usize> = (0..n_cells).map(|c| c % n_units).collect();

    // Sparse cells; pseudobulk counts are their sums.
    let mut cells: Vec<Vec<(u32, f32)>> = Vec::new();
    let mut pb = DMatrix::<f32>::zeros(n_peaks, n_units);
    for &u in &pb_of_cell {
        let mut cell: Vec<(u32, f32)> = Vec::new();
        for p in 0..n_peaks as u32 {
            if rng.random::<f32>() < 0.4 {
                cell.push((p, (1.0 + 4.0 * rng.random::<f32>()).floor()));
            }
        }
        for &(p, x) in &cell {
            pb[(p as usize, u)] += x;
        }
        cells.push(cell);
    }
    // Unit sizes: the pseudobulks' total counts.
    let size: Vec<f32> = (0..n_units).map(|u| pb.column(u).sum()).collect();

    let mut coo = CooMatrix::new(n_peaks, n_units);
    for p in 0..n_peaks {
        for u in 0..n_units {
            if pb[(p, u)] > 0.0 {
                coo.push(p, u, pb[(p, u)]);
            }
        }
    }
    let want = fold_in_peaks(&CsrMatrix::from(&coo), &size, &e, 1e-3).unwrap();

    let design = FoldInDesign::new(&e, &size, 1e-3).unwrap();
    let mut moments = PeakMoments::new(n_peaks, h);
    for (cell, &u) in cells.iter().zip(&pb_of_cell) {
        moments.add_cell(&design, u, cell);
    }
    let got = design.finish(&moments);

    for p in 0..n_peaks {
        for k in 0..h {
            assert!(
                (got.phi[(p, k)] - want.phi[(p, k)]).abs() < 1e-4,
                "peak {p} coord {k}"
            );
        }
        assert!((got.bias[p] - want.bias[p]).abs() < 1e-4, "peak {p} bias");
    }
}
