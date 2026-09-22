//! Per-cell embedding onto the frozen gene / peak dictionaries.

mod common;

use chickpea::common::Mat;
use chickpea::p2g::cells::{
    cluster_labels_to_pb, embed_cells, write_cell_parquet, FrozenAxis, PbWarmStart,
};
use common::{backend, barcodes, mat};
use legume_numeric::candle::candle_core::Device;
use std::path::Path;

/// A dictionary that puts program-A features along +e0 and program-B along +e1.
fn rows(n: usize, split: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|f| {
            let mut r = vec![0f32; dim];
            r[usize::from(f >= split)] = 1.0;
            r[2] = 0.01 * f as f32;
            r
        })
        .collect()
}

/// 12 cells: 0..6 express the first `split` features, 6..12 the rest.
fn counts(n_feat: usize, split: usize) -> Mat {
    mat(n_feat, 12, |f, c| {
        let a = c < 6;
        if (f < split) == a {
            8.0 + (c % 3) as f32
        } else {
            1.0
        }
    })
}

/// Two pseudobulks, one per program, with rows along +e0 and +e1, and cells
/// 0..6 in the first, 6..12 in the second.
fn pb_table(dim: usize) -> (Vec<Vec<f32>>, Vec<usize>) {
    let mut rows = vec![vec![0f32; dim]; 2];
    rows[0][0] = 1.0;
    rows[1][1] = 1.0;
    (rows, (0..12).map(|c| usize::from(c >= 6)).collect())
}

#[test]
fn cells_land_on_their_programs_side_and_the_parquet_has_one_row_per_barcode() {
    let dim = 4;
    let rna = counts(4, 2);
    let atac = counts(6, 3);
    let gene_rows = rows(4, 2, dim);
    let peak_rows = rows(6, 3, dim);
    let (gb, pb) = (vec![0f32; 4], vec![0f32; 6]);
    let axes = [
        FrozenAxis {
            label: "gene",
            rows: &gene_rows,
            bias: &gb,
        },
        FrozenAxis {
            label: "peak",
            rows: &peak_rows,
            bias: &pb,
        },
    ];
    let (rb, ab) = (backend(&rna), backend(&atac));
    let (rows, cell_to_pb) = pb_table(dim);
    let warm = PbWarmStart {
        rows: &rows,
        cell_to_pb: &cell_to_pb,
    };
    let theta = embed_cells(&axes, &[&rb, &ab], &warm, dim, &Device::Cpu).unwrap();
    assert_eq!((theta.nrows(), theta.ncols()), (12, dim));
    for c in 0..12 {
        let (own, other) = if c < 6 { (0, 1) } else { (1, 0) };
        assert!(theta[(c, own)] > theta[(c, other)], "cell {c}");
    }
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("run").to_string_lossy().into_owned();
    write_cell_parquet(&prefix, &theta, &barcodes(12)).unwrap();
    assert!(Path::new(&format!("{prefix}.cell_embedding.parquet")).is_file());
}

#[test]
fn a_single_axis_run_embeds_on_that_axis_alone() {
    let dim = 4;
    let atac = counts(6, 3);
    let peak_rows = rows(6, 3, dim);
    let pb = vec![0f32; 6];
    let axes = [FrozenAxis {
        label: "peak",
        rows: &peak_rows,
        bias: &pb,
    }];
    let ab = backend(&atac);
    let (rows, cell_to_pb) = pb_table(dim);
    let warm = PbWarmStart {
        rows: &rows,
        cell_to_pb: &cell_to_pb,
    };
    let theta = embed_cells(&axes, &[&ab], &warm, dim, &Device::Cpu).unwrap();
    for c in 0..12 {
        let (own, other) = if c < 6 { (0, 1) } else { (1, 0) };
        assert!(theta[(c, own)] > theta[(c, other)], "cell {c}");
    }
}

#[test]
fn a_dictionary_that_does_not_match_the_backend_is_refused() {
    let dim = 4;
    let atac = counts(6, 3);
    let peak_rows = rows(5, 3, dim);
    let pb = vec![0f32; 5];
    let axes = [FrozenAxis {
        label: "peak",
        rows: &peak_rows,
        bias: &pb,
    }];
    let ab = backend(&atac);
    let (rows, cell_to_pb) = pb_table(dim);
    let warm = PbWarmStart {
        rows: &rows,
        cell_to_pb: &cell_to_pb,
    };
    let err = embed_cells(&axes, &[&ab], &warm, dim, &Device::Cpu).expect_err("refused");
    assert!(err.to_string().contains("peak"), "{err}");
}

#[test]
fn a_warm_start_that_does_not_cover_every_cell_is_refused() {
    let dim = 4;
    let atac = counts(6, 3);
    let peak_rows = rows(6, 3, dim);
    let pb = vec![0f32; 6];
    let axes = [FrozenAxis {
        label: "peak",
        rows: &peak_rows,
        bias: &pb,
    }];
    let ab = backend(&atac);
    let (rows, _) = pb_table(dim);
    let cell_to_pb = vec![0usize; 5];
    let warm = PbWarmStart {
        rows: &rows,
        cell_to_pb: &cell_to_pb,
    };
    let err = embed_cells(&axes, &[&ab], &warm, dim, &Device::Cpu).expect_err("refused");
    assert!(err.to_string().contains("warm start"), "{err}");
}

#[test]
fn pb_labels_are_the_majority_of_their_cells() {
    // pb 0 has cells {0,1,2} labeled 1,1,0; pb 1 has {3,4} labeled None,2; pb 2 has no labeled cell.
    let cell_label = vec![Some(1), Some(1), Some(0), None, Some(2), None];
    let cell_to_pb = vec![0, 0, 0, 1, 1, 2];
    assert_eq!(
        cluster_labels_to_pb(&cell_label, &cell_to_pb, 3),
        vec![Some(1), Some(2), None]
    );
}

#[test]
fn a_tie_goes_to_the_lowest_cluster_id() {
    let cell_label = vec![Some(3), Some(1)];
    let cell_to_pb = vec![0, 0];
    assert_eq!(
        cluster_labels_to_pb(&cell_label, &cell_to_pb, 1),
        vec![Some(1)]
    );
}

/// Phase-1 cell units: at most `k` cells per finest pb and per coarser pb,
/// unioned, read from the backends in axis order; an ATAC-only run has an
/// empty gene axis for every cell.
#[test]
fn phase1_cell_units_keep_k_per_pseudobulk_at_every_level() {
    use chickpea::p2g::cells::phase1_cell_units;
    let rna = counts(4, 2);
    let atac = counts(6, 3);
    let (rb, ab) = (backend(&rna), backend(&atac));
    // 12 cells: finest pbs of 3 (4 pbs), one coarse level of 2 pbs.
    let cell_to_pb: Vec<usize> = (0..12).map(|c| c / 3).collect();
    let parent: Vec<Vec<usize>> = vec![vec![0, 0, 1, 1]];
    let g = phase1_cell_units(&[&rb, &ab], &cell_to_pb, &parent, 1, 5).unwrap();
    assert_eq!(g.axes.len(), 2);
    assert!(
        g.cells.len() >= 4 && g.cells.len() <= 6,
        "{} cells",
        g.cells.len()
    );
    for pb in 0..4 {
        let n = g
            .cells
            .iter()
            .filter(|&&c| cell_to_pb[c as usize] == pb)
            .count();
        assert!((1..=2).contains(&n), "pb {pb} keeps {n}");
    }
    for (i, &c) in g.cells.iter().enumerate() {
        assert_eq!(g.axes[0][i].0.len(), 4, "cell {c} gene row");
        assert_eq!(g.axes[1][i].0.len(), 6, "cell {c} peak row");
    }
    // ATAC-only: one backend, the gene axis is empty rows.
    let g = phase1_cell_units(&[&ab], &cell_to_pb, &parent, 1, 5).unwrap();
    assert_eq!(g.axes.len(), 2);
    assert!(g.axes[0].iter().all(|(f, _)| f.is_empty()));
    assert!(g.axes[1].iter().all(|(f, _)| f.len() == 6));
    // k = 0: no cell units.
    let g = phase1_cell_units(&[&rb, &ab], &cell_to_pb, &parent, 0, 5).unwrap();
    assert!(g.cells.is_empty());
}
