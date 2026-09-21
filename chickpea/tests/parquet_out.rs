//! E2G-like parquet writers.

use chickpea::p2g::link_map::PeakGeneEdge;
mod common;

use chickpea::p2g::parquet_out::*;
use chickpea::p2g::refine::ClusterLink;
use common::{peak, tss};
use genomic_data::coordinates::{GeneTss, PeakCoord};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;

#[test]
fn writes_peaks_clusters_and_chr_partitioned_links() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().to_string_lossy().into_owned();

    let peak_coords = vec![
        peak(100),
        Some(PeakCoord {
            chr: "2".into(),
            start: 300,
            end: 400,
        }),
    ];
    let gene_tss = vec![
        tss(150),
        Some(GeneTss {
            chr: "2".into(),
            tss: 350,
        }),
    ];
    let gene_names: Vec<Box<str>> = vec!["G1".into(), "G2".into()];
    let peaks = peaks_from_coords(&peak_coords);
    let clusters = vec![
        ClusterRow {
            id: "0".into(),
            name: "cluster_0".into(),
        },
        ClusterRow {
            id: "1".into(),
            name: "cluster_1".into(),
        },
    ];
    let links = vec![
        ClusterLink {
            edge: PeakGeneEdge {
                peak: 0,
                gene: 0,
                weight: 0.9,
            },
            cluster: 0,
        },
        ClusterLink {
            edge: PeakGeneEdge {
                peak: 1,
                gene: 1,
                weight: 0.8,
            },
            cluster: 1,
        },
    ];

    write_e2g_tables(
        &out,
        &peaks,
        &clusters,
        &links,
        &peak_coords,
        &gene_tss,
        &gene_names,
    )
    .unwrap();

    assert!(dir.path().join("peaks.parquet").is_file());
    assert!(dir.path().join("clusters.parquet").is_file());
    assert!(dir.path().join("peak_gene").join("chr1.parquet").is_file());
    assert!(dir.path().join("peak_gene").join("chr2.parquet").is_file());

    let file = File::open(dir.path().join("peaks.parquet")).unwrap();
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap()
        .build()
        .unwrap();
    let mut n = 0usize;
    for batch in reader {
        n += batch.unwrap().num_rows();
    }
    assert_eq!(n, 2);
}
