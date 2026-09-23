//! Shared planted multiome fixture: two cell programs over genes with two
//! peaks each, written as tiny RNA and ATAC zarrs.
#![allow(dead_code)]

use chickpea::p2g::cis::AbcKernel;
use data_beans::sparse_io::{create_sparse_from_triplets, SparseIoBackend};
use genomic_data::coordinates::GeneTss;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

pub const N_GENES: usize = 24;
pub const N_CELLS: usize = 120;
/// Two peaks per gene: one at the TSS, one 3 kb upstream.
pub const PEAKS_PER_GENE: usize = 2;

/// Gene `g` belongs to program `g % 2`; cell `c` to program `c % 2`.
pub fn program_of(i: usize) -> usize {
    i % 2
}

pub fn tss(g: usize) -> i64 {
    100_000 * (g as i64 + 1)
}

pub fn names(prefix: &str, n: usize) -> Vec<Box<str>> {
    (1..=n).map(|i| format!("{prefix}{i}").into()).collect()
}

/// Writes `{dir}/rna.zarr` and `{dir}/atac.zarr` for the same barcodes.
pub fn write_fixture(dir: &std::path::Path) -> (String, String) {
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

/// Every gene at its TSS on chr1.
pub fn gene_positions() -> Vec<Option<GeneTss>> {
    (0..N_GENES)
        .map(|g| {
            Some(GeneTss {
                chr: "chr1".into(),
                tss: tss(g),
            })
        })
        .collect()
}

/// A window that reaches each gene's own two peaks only.
pub fn kernel() -> AbcKernel {
    AbcKernel {
        window: 10_000,
        ..AbcKernel::default()
    }
}

/// The index after the name prefix, 0-based: GENE3 -> 2.
pub fn index(name: &str) -> usize {
    name.trim_start_matches(|ch: char| !ch.is_ascii_digit())
        .parse::<usize>()
        .unwrap()
        - 1
}
