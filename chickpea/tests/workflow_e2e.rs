//! End-to-end workflow smoke test on synthetic pb matrices.

mod common;

use chickpea::common::Mat;
use chickpea::p2g::link_map::LinkParams;
use chickpea::p2g::workflow::*;
use common::{peak, tss};
use graph_embedding_util::fne::FneConfig;
use legume_numeric::candle::candle_core::Device;
use std::path::Path;

#[test]
fn e2e_writes_e2g_parquet_from_synthetic_pb() {
    let s = 48usize;
    let mut rna = Mat::zeros(2, s);
    let mut atac = Mat::zeros(3, s);
    for j in 0..s {
        let z = (j as f32) * 0.15;
        let sig0 = z.sin().abs() + 0.1;
        let sig1 = z.cos().abs() + 0.1;
        rna[(0, j)] = sig0 * 10.0;
        rna[(1, j)] = sig1 * 10.0;
        atac[(0, j)] = sig0 * 8.0;
        atac[(1, j)] = sig1 * 8.0;
        atac[(2, j)] = ((j % 5) as f32) * 0.2;
    }

    let gene_names: Vec<Box<str>> = vec!["G0".into(), "G1".into()];
    let peak_names: Vec<Box<str>> = vec![
        "chr1:100000-100500".into(),
        "chr1:200000-200500".into(),
        "chr1:150000-150500".into(),
    ];
    let gene_tss = vec![tss(100_000), tss(200_000)];
    let peak_coords = vec![peak(100_000), peak(200_000), peak(150_000)];

    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().to_string_lossy().into_owned();
    let params = WorkflowParams {
        abc: LinkParams {
            cis_window: 500_000,
            max_cis: 50,
            min_weight: 0.1,
            ..LinkParams::default()
        },
        fne: FneConfig {
            dim: 8,
            epochs: 2,
            batch_size: 8,
            num_batch_negs: 2,
            num_uniform_negs: 2,
            wd: Some(0.0),
            eval_fraction: 0.0,
            seed: 3,
            device: Device::Cpu,
            ..FneConfig::default()
        },
        min_cluster_samples: 5,
        target_clusters: Some(2),
    };

    run_from_pseudobulk(
        &PbMultiome {
            rna_pb: &rna,
            atac_pb: &atac,
            gene_tss: &gene_tss,
            peak_coords: &peak_coords,
            gene_names: &gene_names,
            peak_names: &peak_names,
        },
        &out,
        &params,
    )
    .unwrap();

    assert!(Path::new(&out).join("peaks.parquet").is_file());
    assert!(Path::new(&out).join("clusters.parquet").is_file());
    assert!(
        Path::new(&out)
            .join("peak_gene")
            .join("chr1.parquet")
            .is_file(),
        "expected peak_gene/chr1.parquet under {out}"
    );
    // Senna-style peak / gene / cell embedding parquets (prefix = out dir path).
    assert!(
        Path::new(&format!("{out}.peak_embedding.parquet")).is_file(),
        "expected {out}.peak_embedding.parquet"
    );
    assert!(
        Path::new(&format!("{out}.gene_embedding.parquet")).is_file(),
        "expected {out}.gene_embedding.parquet"
    );
    assert!(
        Path::new(&format!("{out}.cell_embedding.parquet")).is_file(),
        "expected {out}.cell_embedding.parquet"
    );
}
