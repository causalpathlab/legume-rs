//! Peak counts aggregated onto genes through the fixed weights `W`.

use chickpea::p2g::cis::{build_cis_pairs, AbcKernel, CisPairs};
use chickpea::p2g::gene_track::PeakToGenes;
use genomic_data::coordinates::{GeneTss, PeakCoord};

/// GENE1 (peaks 0, 1), GENE2 (peak 1), GENE3 (no position), GENE4 (peak 3);
/// peak 2 reaches no gene.
fn pairs() -> CisPairs {
    let g = |chr: &str, tss| {
        Some(GeneTss {
            chr: chr.into(),
            tss,
        })
    };
    let p = |chr: &str, start, end| {
        Some(PeakCoord {
            chr: chr.into(),
            start,
            end,
        })
    };
    let genes = vec![g("chr1", 10_000), g("chr1", 30_000), None, g("chr2", 5_000)];
    let peaks = vec![
        p("chr1", 9_900, 10_100),
        p("chr1", 19_900, 20_100),
        p("chr1", 200_000, 200_200),
        p("chr2", 5_900, 6_100),
    ];
    let k = AbcKernel {
        window: 15_000,
        ..AbcKernel::default()
    };
    build_cis_pairs(&genes, &peaks, &k)
}

#[test]
fn the_inverse_lists_every_gene_a_peak_feeds() {
    let pairs = pairs();
    let map = PeakToGenes::new(&pairs, 4);
    assert_eq!(map.genes_of(0), [0]);
    assert_eq!(map.genes_of(1), [0, 1], "the shared peak feeds both genes");
    assert!(map.genes_of(2).is_empty());
    assert_eq!(map.genes_of(3), [3]);
}

#[test]
fn a_cells_gene_counts_are_the_weighted_sums_of_its_peak_counts() {
    let pairs = pairs();
    let map = PeakToGenes::new(&pairs, 4);
    let w0 = &pairs.weight[pairs.gene(0)];
    let cell = [(0u32, 2.0f32), (1, 4.0), (3, 1.0)];
    let got = map.aggregate(&cell);
    let want = [(0u32, 2.0 * w0[0] + 4.0 * w0[1]), (1, 4.0), (3, 1.0)];
    assert_eq!(got.len(), want.len(), "{got:?}");
    for ((g, v), (wg, wv)) in got.iter().zip(want) {
        assert_eq!(*g, wg);
        assert!((v - wv).abs() < 1e-6, "gene {g}: {v} vs {wv}");
    }
}

#[test]
fn a_cell_with_only_unreached_peaks_has_no_gene_counts() {
    let map = PeakToGenes::new(&pairs(), 4);
    assert!(map.aggregate(&[(2u32, 5.0f32)]).is_empty());
}

#[test]
fn genes_come_out_ascending_and_without_zeros() {
    let map = PeakToGenes::new(&pairs(), 4);
    let got = map.aggregate(&[(3u32, 1.0f32), (1, 0.0), (0, 1.0)]);
    let genes: Vec<u32> = got.iter().map(|&(g, _)| g).collect();
    assert_eq!(genes, [0, 3]);
}

mod written {
    use super::pairs;
    use chickpea::p2g::gene_track::{write_gene_track, PeakToGenes};
    use data_beans::sparse_io::{
        create_sparse_from_triplets, open_sparse_matrix_by_path, SparseIoBackend,
    };

    /// Four peaks × three cells (CELL1..CELL3) on disk, the pairs' layout.
    fn atac_file(dir: &std::path::Path) -> String {
        let path = dir.join("atac.zarr").to_string_lossy().into_owned();
        let trip: Vec<(u64, u64, f32)> = vec![
            (0, 0, 2.0),
            (1, 0, 4.0),
            (3, 0, 1.0),
            (2, 1, 5.0), // CELL2: only the unreached peak
            (1, 2, 3.0),
        ];
        let mut m = create_sparse_from_triplets(
            &trip,
            (4, 3, trip.len()),
            Some(&path),
            Some(&SparseIoBackend::Zarr),
        )
        .unwrap();
        let names = |v: &[&str]| v.iter().map(|&s| Box::from(s)).collect::<Vec<Box<str>>>();
        m.register_row_names_vec(&names(&["P1", "P2", "P3", "P4"]));
        m.register_column_names_vec(&names(&["CELL1", "CELL2", "CELL3"]));
        path
    }

    fn gene_names() -> Vec<Box<str>> {
        ["GENE1", "GENE2", "GENE3", "GENE4"]
            .into_iter()
            .map(Box::from)
            .collect()
    }

    #[test]
    fn the_track_has_a_row_per_gene_with_cis_peaks_and_every_cell() {
        let dir = tempfile::tempdir().unwrap();
        let atac = open_sparse_matrix_by_path(&atac_file(dir.path())).unwrap();
        let pairs = pairs();
        let map = PeakToGenes::new(&pairs, 4);
        let out = dir.path().join("track.zarr").to_string_lossy().into_owned();
        let summary =
            write_gene_track(atac.as_ref(), &map, &pairs, &gene_names(), &out, 2).unwrap();
        assert_eq!(summary.n_genes, 3);
        assert_eq!(summary.n_cells, 3);

        let t = open_sparse_matrix_by_path(&out).unwrap();
        let rows: Vec<String> = t
            .row_names()
            .unwrap()
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            rows,
            ["GENE1", "GENE2", "GENE4"],
            "GENE3 has no position, so no row"
        );
        let cols: Vec<String> = t
            .column_names()
            .unwrap()
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(cols, ["CELL1", "CELL2", "CELL3"], "cells unchanged");
    }

    #[test]
    fn the_written_counts_are_the_weighted_sums() {
        let dir = tempfile::tempdir().unwrap();
        let atac = open_sparse_matrix_by_path(&atac_file(dir.path())).unwrap();
        let pairs = pairs();
        let map = PeakToGenes::new(&pairs, 4);
        let out = dir.path().join("track.zarr").to_string_lossy().into_owned();
        // A block of 2 cells splits the three cells across two slabs.
        write_gene_track(atac.as_ref(), &map, &pairs, &gene_names(), &out, 2).unwrap();

        let t = open_sparse_matrix_by_path(&out).unwrap();
        let dense = t.read_columns_dmatrix((0..3).collect()).unwrap();
        let w0 = &pairs.weight[pairs.gene(0)];
        let want = [
            [2.0 * w0[0] + 4.0 * w0[1], 0.0, 3.0 * w0[1]], // GENE1
            [4.0, 0.0, 3.0],                               // GENE2
            [1.0, 0.0, 0.0],                               // GENE4
        ];
        for (r, row) in want.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                assert!(
                    (dense[(r, c)] - v).abs() < 1e-5,
                    "({r}, {c}): {} vs {v}",
                    dense[(r, c)]
                );
            }
        }
    }
}
