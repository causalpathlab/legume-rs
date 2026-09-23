//! End to end on a planted fixture: RNA plus peak-aggregated gene rows on two
//! tracks, one short fit. The two cell programs separate, and each gene's two
//! rows agree more than rows of unrelated genes.

mod common;

use chickpea::p2g::two_track::{embed_two_track, TwoTrackConfig, TwoTrackInput};
use common::{gene_positions, index, kernel, program_of, write_fixture, N_CELLS, N_GENES};
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
fn programs_separate_and_a_genes_two_rows_agree() {
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

    // Shapes: every gene has an RNA row; every gene here also has cis peaks.
    assert_eq!(out.genes.len(), N_GENES);
    assert_eq!(out.rna_rows.nrows(), N_GENES);
    assert_eq!(out.atac_rows.nrows(), N_GENES);
    assert_eq!(out.cell_rows.nrows(), N_CELLS);
    assert!(out.atac_gene.iter().all(Option::is_some));

    // Pseudobulks: the finest level's rows, and every cell's pseudobulk.
    assert_eq!(out.cell_to_pb.len(), N_CELLS);
    assert!(out.cell_to_pb.iter().all(|&u| u < out.pb_rows.nrows()));
    assert_eq!(out.pb_rows.ncols(), out.rna_rows.ncols());

    // Cells: same-program pairs are closer than cross-program pairs.
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

    // Genes: a gene's RNA and ATAC rows agree more than an opposite-program pair.
    let row_of_gene: Vec<usize> = {
        let mut v = vec![0; N_GENES];
        for (r, name) in out.genes.iter().enumerate() {
            v[index(name)] = r;
        }
        v
    };
    let atac_row = |g: usize| out.atac_gene[row_of_gene[g]].unwrap();
    let (mut own, mut other) = (0f32, 0f32);
    for (g, &r) in row_of_gene.iter().enumerate() {
        let rna = row(&out.rna_rows, r);
        let opp = (g + 1) % N_GENES; // the next gene is in the other program
        own += cosine(&rna, &row(&out.atac_rows, atac_row(g)));
        other += cosine(&rna, &row(&out.atac_rows, atac_row(opp)));
    }
    let (own, other) = (own / N_GENES as f32, other / N_GENES as f32);
    assert!(own > other + 0.3, "genes: own {own:.3} vs other {other:.3}");
}
