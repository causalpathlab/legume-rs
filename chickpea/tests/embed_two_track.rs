//! End to end on a planted fixture: RNA genes + ATAC peaks on one multiome
//! axis (module-only ATAC). Cell programs separate.

mod common;

use chickpea::p2g::two_track::{embed_two_track, TwoTrackConfig, TwoTrackInput};
use common::{
    gene_positions, index, kernel, program_of, write_fixture, N_CELLS, N_GENES, PEAKS_PER_GENE,
};
use nalgebra::DMatrix;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let d: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let n = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    d / (n(a) * n(b)).max(1e-12)
}

fn row(m: &DMatrix<f32>, r: usize) -> Vec<f32> {
    m.row(r).iter().copied().collect()
}

#[test]
fn programs_separate_on_the_multiome_axis() {
    let dir = tempfile::tempdir().unwrap();
    let (rna, atac) = write_fixture(dir.path());
    let genes = gene_positions();
    let work = dir.path().join("run").to_string_lossy().into_owned();
    let cfg = TwoTrackConfig {
        embedding_dim: 8,
        epochs: 60,
        num_levels: 2,
        sort_dim: 3,
        proj_dim: 8,
        feature_modules: 4,
        phase1_cells_per_pb: 4,
        // 24 genes < 30 ≤ 48 peaks → ATAC module-only, RNA keeps residuals.
        module_only_min_rows: 30,
        ..TwoTrackConfig::default()
    };
    let input = TwoTrackInput {
        rna_file: &rna,
        atac_file: &atac,
        batch_file: None,
        gene_positions: &genes,
        kernel: &kernel(),
        work_prefix: &work,
    };
    let out = embed_two_track(&input, &cfg).unwrap();

    assert_eq!(out.genes.len(), N_GENES);
    assert_eq!(out.rna_rows.nrows(), N_GENES);
    assert_eq!(out.peak_rows.nrows(), N_GENES * PEAKS_PER_GENE);
    assert_eq!(out.module_of_peak.len(), out.peak_rows.nrows());
    assert_eq!(out.cell_rows.nrows(), N_CELLS);

    let prog: Vec<usize> = out.barcodes.iter().map(|b| program_of(index(b))).collect();
    let (mut same, mut cross, mut ns, mut nc) = (0f32, 0f32, 0, 0);
    for a in 0..N_CELLS {
        for b in (a + 1)..N_CELLS {
            let c = cosine(&row(&out.cell_rows, a), &row(&out.cell_rows, b));
            if prog[a] == prog[b] {
                same += c;
                ns += 1;
            } else {
                cross += c;
                nc += 1;
            }
        }
    }
    let (same, cross) = (same / ns as f32, cross / nc as f32);
    assert!(
        same > cross + 0.3,
        "cells: same {same:.3} vs cross {cross:.3}"
    );

    // Peaks in the same ATAC module share a row (module_only).
    let mut by_mod: std::collections::HashMap<u32, Vec<usize>> = std::collections::HashMap::new();
    for (p, &m) in out.module_of_peak.iter().enumerate() {
        by_mod.entry(m).or_default().push(p);
    }
    let shared = by_mod.values().any(|v| v.len() > 1);
    assert!(
        shared,
        "expected at least one ATAC module with multiple peaks"
    );
    for ps in by_mod.values().filter(|v| v.len() > 1) {
        let r0 = row(&out.peak_rows, ps[0]);
        for &p in &ps[1..] {
            let rp = row(&out.peak_rows, p);
            assert!(
                (cosine(&r0, &rp) - 1.0).abs() < 1e-4,
                "peaks in one module must share μ"
            );
        }
    }
}
