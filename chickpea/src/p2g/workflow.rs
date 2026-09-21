//! End-to-end peak-to-gene workflow on pb matrices.
//!
//! abc_map → ge-util FNE → pb-sample embeds → cluster → within-cluster refine → E2G parquet.

use crate::common::*;
use crate::p2g::abc_map::{rough_abc_map, AbcMapParams};
use crate::p2g::cluster::cluster_cells;
use crate::p2g::embed_ge::train_peak_gene_embeds;
use crate::p2g::parquet_out::{peaks_from_coords, write_e2g_tables, ClusterRow};
use crate::p2g::refine::refine_within_clusters;
use genomic_data::coordinates::{GeneTss, PeakCoord};
use graph_embedding_util::fne::FneConfig;
use log::info;
use nalgebra::DMatrix;

/// Knobs for [`run_from_pseudobulk`].
#[derive(Clone, Debug)]
pub struct WorkflowParams {
    pub abc: AbcMapParams,
    pub fne: FneConfig,
    /// Min pb samples (or cells) to keep a cluster.
    pub min_cluster_samples: usize,
    pub target_clusters: Option<usize>,
}

/// Paired RNA/ATAC pb matrices plus feature metadata for the workflow.
pub struct PbMultiome<'a> {
    pub rna_pb: &'a Mat,
    pub atac_pb: &'a Mat,
    pub gene_tss: &'a [Option<GeneTss>],
    pub peak_coords: &'a [Option<PeakCoord>],
    pub gene_names: &'a [Box<str>],
    pub peak_names: &'a [Box<str>],
}

/// Run the ge-util peak-to-gene workflow from paired RNA/ATAC pseudobulk matrices.
pub fn run_from_pseudobulk(
    data: &PbMultiome<'_>,
    out_dir: &str,
    params: &WorkflowParams,
) -> anyhow::Result<()> {
    let PbMultiome {
        rna_pb,
        atac_pb,
        gene_tss,
        peak_coords,
        gene_names,
        peak_names,
    } = data;

    anyhow::ensure!(
        rna_pb.nrows() == gene_names.len() && rna_pb.nrows() == gene_tss.len(),
        "RNA rows vs gene names/TSS mismatch"
    );
    anyhow::ensure!(
        atac_pb.nrows() == peak_names.len() && atac_pb.nrows() == peak_coords.len(),
        "ATAC rows vs peak names/coords mismatch"
    );
    anyhow::ensure!(
        rna_pb.ncols() == atac_pb.ncols(),
        "RNA/ATAC sample count mismatch"
    );

    info!("Building rough ABC / co-occurrence map...");
    let edges = rough_abc_map(rna_pb, atac_pb, gene_tss, peak_coords, &params.abc)?;
    anyhow::ensure!(!edges.is_empty(), "no cis peak–gene edges above min_weight");
    info!("ABC map: {} edges", edges.len());

    info!(
        "Training peak/gene embeddings via graph-embedding-util FNE (dim={}, epochs={})...",
        params.fne.dim, params.fne.epochs
    );
    let embeds = train_peak_gene_embeds(&edges, peak_names, gene_names, &params.fne)?;

    info!("Embedding pb samples from gene embeddings...");
    let sample_mat = embed_pb_samples(rna_pb, &embeds.gene)?;
    info!(
        "Clustering {} pb samples (min_cluster_samples={})...",
        sample_mat.nrows(),
        params.min_cluster_samples
    );
    let clusters = cluster_cells(
        &sample_mat,
        params.target_clusters,
        params.min_cluster_samples,
    )?;
    info!(
        "Kept {} clusters (sizes={:?})",
        clusters.n_clusters, clusters.sizes
    );

    info!("Refining peak→gene within clusters...");
    let links = refine_within_clusters(
        rna_pb,
        atac_pb,
        gene_tss,
        peak_coords,
        &clusters.label,
        params.min_cluster_samples,
        &params.abc,
    )?;
    anyhow::ensure!(!links.is_empty(), "no within-cluster links after refine");
    info!("Refined links: {}", links.len());

    let peaks = peaks_from_coords(peak_coords);
    let cluster_rows: Vec<ClusterRow> = (0..clusters.n_clusters)
        .map(|c| ClusterRow {
            id: c.to_string().into_boxed_str(),
            name: format!("cluster_{c}").into_boxed_str(),
        })
        .collect();

    mkdir_parent(&format!("{out_dir}/peaks.parquet"))?;
    write_e2g_tables(
        out_dir,
        &peaks,
        &cluster_rows,
        &links,
        peak_coords,
        gene_tss,
        gene_names,
    )?;
    Ok(())
}

/// Pb-sample embedding: `Σ_g log1p(RNA_{g,s}) · e_gene[g]`, L2-normalized.
fn embed_pb_samples(rna_pb: &Mat, gene_emb: &[Vec<f32>]) -> anyhow::Result<DMatrix<f32>> {
    let n_genes = rna_pb.nrows();
    let n_samples = rna_pb.ncols();
    anyhow::ensure!(
        gene_emb.len() == n_genes,
        "gene embedding rows != RNA genes"
    );
    let dim = gene_emb.first().map_or(0, |r| r.len());
    anyhow::ensure!(dim > 0, "empty gene embedding");
    anyhow::ensure!(
        gene_emb.iter().all(|r| r.len() == dim),
        "ragged gene embedding"
    );

    let mut out = DMatrix::<f32>::zeros(n_samples, dim);
    for s in 0..n_samples {
        for g in 0..n_genes {
            let w = rna_pb[(g, s)].ln_1p();
            if w == 0.0 {
                continue;
            }
            for d in 0..dim {
                out[(s, d)] += w * gene_emb[g][d];
            }
        }
        let mut norm = 0.0f32;
        for d in 0..dim {
            norm += out[(s, d)] * out[(s, d)];
        }
        let norm = norm.sqrt();
        if norm > 1e-8 {
            for d in 0..dim {
                out[(s, d)] /= norm;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
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
            atac[(0, j)] = sig0 * 8.0; // cis to gene 0
            atac[(1, j)] = sig1 * 8.0; // cis to gene 1
            atac[(2, j)] = ((j % 5) as f32) * 0.2; // cis bystander
        }

        let gene_names: Vec<Box<str>> = vec!["G0".into(), "G1".into()];
        let peak_names: Vec<Box<str>> = vec![
            "chr1:100000-100500".into(),
            "chr1:200000-200500".into(),
            "chr1:150000-150500".into(),
        ];
        let gene_tss = vec![
            Some(GeneTss {
                chr: "1".into(),
                tss: 100_000,
            }),
            Some(GeneTss {
                chr: "1".into(),
                tss: 200_000,
            }),
        ];
        let peak_coords = vec![
            Some(PeakCoord {
                chr: "1".into(),
                start: 100_000,
                end: 100_500,
            }),
            Some(PeakCoord {
                chr: "1".into(),
                start: 200_000,
                end: 200_500,
            }),
            Some(PeakCoord {
                chr: "1".into(),
                start: 150_000,
                end: 150_500,
            }),
        ];

        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().to_string_lossy().into_owned();
        let params = WorkflowParams {
            abc: AbcMapParams {
                cis_window: 500_000,
                max_cis: 50,
                min_weight: 0.1,
            },
            fne: FneConfig {
                dim: 8,
                epochs: 2,
                lr: 0.1,
                batch_size: 8,
                num_batch_negs: 2,
                num_uniform_negs: 2,
                wd: Some(0.0),
                wd_interval: 50,
                eval_fraction: 0.0,
                eval_min_per_relation: 1,
                relation_repeats: Vec::new(),
                preset: None,
                seed: 3,
                device: Device::Cpu,
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
    }
}
