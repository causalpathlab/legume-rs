//! Dump pb-sample embeddings for UMAP plotting (see `tests/umap/`).

use super::*;
use crate::p2g::abc_map::rough_abc_map;
use crate::p2g::cluster::cluster_cells;
use crate::p2g::embed_ge::train_peak_gene_embeds;
use graph_embedding_util::fne::FneConfig;
use legume_numeric::candle::candle_core::Device;
use std::io::Write;
use std::path::Path;

/// Output directory for TSVs consumed by `tests/umap/plot_pb_sample_umap.R`.
fn umap_out_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/umap/out")
}

#[test]
fn dump_pb_embeds_for_umap() {
    let s = 120usize;
    let mut rna = Mat::zeros(4, s);
    let mut atac = Mat::zeros(4, s);
    // Two latent programs across samples → separable clusters in embed space.
    for j in 0..s {
        let t = j as f32 / s as f32;
        let a = if j < s / 2 {
            (t * 6.0).sin().abs() + 0.2
        } else {
            0.05
        };
        let b = if j >= s / 2 {
            (t * 6.0).cos().abs() + 0.2
        } else {
            0.05
        };
        rna[(0, j)] = a * 12.0;
        rna[(1, j)] = a * 10.0;
        rna[(2, j)] = b * 12.0;
        rna[(3, j)] = b * 10.0;
        atac[(0, j)] = a * 9.0;
        atac[(1, j)] = a * 8.0;
        atac[(2, j)] = b * 9.0;
        atac[(3, j)] = b * 8.0;
    }

    let gene_names: Vec<Box<str>> = (0..4).map(|i| format!("G{i}").into()).collect();
    let peak_names: Vec<Box<str>> = (0..4)
        .map(|i| format!("chr1:{}-{}", 100_000 + i * 50_000, 100_500 + i * 50_000).into())
        .collect();
    let gene_tss: Vec<_> = (0..4)
        .map(|i| {
            Some(GeneTss {
                chr: "1".into(),
                tss: 100_000 + i * 50_000,
            })
        })
        .collect();
    let peak_coords: Vec<_> = (0..4)
        .map(|i| {
            let start = 100_000 + i * 50_000;
            Some(PeakCoord {
                chr: "1".into(),
                start,
                end: start + 500,
            })
        })
        .collect();

    let abc = AbcMapParams {
        cis_window: 500_000,
        max_cis: 50,
        min_weight: 0.05,
    };
    let edges = rough_abc_map(&rna, &atac, &gene_tss, &peak_coords, &abc).unwrap();
    let fne = FneConfig {
        dim: 16,
        epochs: 8,
        lr: 0.1,
        batch_size: 16,
        num_batch_negs: 4,
        num_uniform_negs: 4,
        wd: Some(0.0),
        wd_interval: 50,
        eval_fraction: 0.0,
        eval_min_per_relation: 1,
        relation_repeats: Vec::new(),
        preset: None,
        seed: 11,
        device: Device::Cpu,
    };
    let embeds = train_peak_gene_embeds(&edges, &peak_names, &gene_names, &fne).unwrap();
    let sample_mat = embed_pb_samples(&rna, &embeds.gene).unwrap();
    let clusters = cluster_cells(&sample_mat, Some(2), 5).unwrap();

    let dir = umap_out_dir();
    std::fs::create_dir_all(&dir).unwrap();
    let emb_path = dir.join("pb_sample_embedding.tsv");
    let lab_path = dir.join("pb_sample_cluster.tsv");
    {
        let mut f = std::fs::File::create(&emb_path).unwrap();
        let header = (0..sample_mat.ncols())
            .map(|d| format!("e{d}"))
            .collect::<Vec<_>>()
            .join("\t");
        writeln!(f, "sample\t{header}").unwrap();
        for i in 0..sample_mat.nrows() {
            write!(f, "{i}").unwrap();
            for d in 0..sample_mat.ncols() {
                write!(f, "\t{:.8}", sample_mat[(i, d)]).unwrap();
            }
            writeln!(f).unwrap();
        }
    }
    {
        let mut f = std::fs::File::create(&lab_path).unwrap();
        writeln!(f, "sample\tcluster").unwrap();
        for (i, lab) in clusters.label.iter().enumerate() {
            let c = lab.map(|x| x as i64).unwrap_or(-1);
            writeln!(f, "{i}\t{c}").unwrap();
        }
    }
    assert!(emb_path.is_file());
    assert!(lab_path.is_file());
}
