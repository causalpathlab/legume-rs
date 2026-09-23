//! End to end on a planted fixture: RNA plus peak-aggregated gene rows on two
//! tracks, one short fit. The two cell programs separate, and each gene's two
//! rows agree more than rows of unrelated genes.

use chickpea::p2g::cis::AbcKernel;
use chickpea::p2g::two_track::{embed_two_track, TwoTrackConfig, TwoTrackInput};
use data_beans::sparse_io::{create_sparse_from_triplets, SparseIoBackend};
use genomic_data::coordinates::GeneTss;
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const N_GENES: usize = 24;
const N_CELLS: usize = 120;
/// Two peaks per gene: one at the TSS, one 3 kb upstream.
const PEAKS_PER_GENE: usize = 2;

/// Gene `g` belongs to program `g % 2`; cell `c` to program `c % 2`.
fn program_of(i: usize) -> usize {
    i % 2
}

fn tss(g: usize) -> i64 {
    100_000 * (g as i64 + 1)
}

fn names(prefix: &str, n: usize) -> Vec<Box<str>> {
    (1..=n).map(|i| format!("{prefix}{i}").into()).collect()
}

/// Writes `{dir}/rna.zarr` and `{dir}/atac.zarr` for the same barcodes.
fn write_fixture(dir: &std::path::Path) -> (String, String) {
    let mut rng = StdRng::seed_from_u64(7);
    let mut count = |on: bool| -> f32 {
        let mean = if on { 8.0 } else { 0.3 };
        // Small deterministic Poisson-like draw.
        let u: f32 = rng.random();
        (mean * (0.5 + u)).floor()
    };
    let barcodes = names("CELL", N_CELLS);

    let mut rna = Vec::new();
    for c in 0..N_CELLS {
        for g in 0..N_GENES {
            let v = count(program_of(g) == program_of(c));
            if v > 0.0 {
                rna.push((g as u64, c as u64, v));
            }
        }
    }
    let n_peaks = N_GENES * PEAKS_PER_GENE;
    let mut atac = Vec::new();
    for c in 0..N_CELLS {
        for p in 0..n_peaks {
            let v = count(program_of(p / PEAKS_PER_GENE) == program_of(c));
            if v > 0.0 {
                atac.push((p as u64, c as u64, v));
            }
        }
    }
    let peak_names: Vec<Box<str>> = (0..n_peaks)
        .map(|p| {
            let mid = tss(p / PEAKS_PER_GENE) - 3_000 * (p % PEAKS_PER_GENE) as i64;
            format!("chr1:{}-{}", mid - 250, mid + 250).into()
        })
        .collect();

    let write = |name: &str, trip: &[(u64, u64, f32)], rows: &[Box<str>]| -> String {
        let path = dir.join(name).to_string_lossy().into_owned();
        let mut m = create_sparse_from_triplets(
            trip,
            (rows.len(), N_CELLS, trip.len()),
            Some(&path),
            Some(&SparseIoBackend::Zarr),
        )
        .unwrap();
        m.register_row_names_vec(rows);
        m.register_column_names_vec(&barcodes);
        path
    };
    (
        write("rna.zarr", &rna, &names("GENE", N_GENES)),
        write("atac.zarr", &atac, &peak_names),
    )
}

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
    let genes: Vec<Option<GeneTss>> = (0..N_GENES)
        .map(|g| {
            Some(GeneTss {
                chr: "chr1".into(),
                tss: tss(g),
            })
        })
        .collect();
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
        kernel: &AbcKernel {
            window: 10_000,
            ..AbcKernel::default()
        },
        work_prefix: &work,
    };
    let out = embed_two_track(&input, &cfg).unwrap();

    // Shapes: every gene has an RNA row; every gene here also has cis peaks.
    assert_eq!(out.genes.len(), N_GENES);
    assert_eq!(out.rna_rows.nrows(), N_GENES);
    assert_eq!(out.atac_rows.nrows(), N_GENES);
    assert_eq!(out.cell_rows.nrows(), N_CELLS);
    assert!(out.atac_gene.iter().all(Option::is_some));

    // The index after the name prefix, 0-based: GENE3 -> 2.
    let index = |name: &str| -> usize {
        name.trim_start_matches(|ch: char| !ch.is_ascii_digit())
            .parse::<usize>()
            .unwrap()
            - 1
    };

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
