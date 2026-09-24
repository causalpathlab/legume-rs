//! The link workflow: multiome embed, peak module readout, gates, per-cluster tables.
//!
//! Outputs, all `{out}.*.parquet`:
//!
//! - `gene_embedding`: RNA gene rows;
//! - `cell_embedding`, `cell_clusters`;
//! - `peak_embedding`: each peak's ATAC-module row `μ_{m(p)}`;
//! - `peaks`: `peak`, `chromosome`, `start`, `end`;
//! - `links`: `gene`, `peak`, `distance`, `abc`, `gate` (the trained gate
//!   weight `w`; `0` for a closed pair);
//! - `links_by_cluster`: `gene_idx`, `peak_idx`, `cluster`, `gate`, `abc`
//!   (indices into the RNA / peak name axes; shares re-weighted by cluster
//!   accessibility). Dense string copies of every pair×cluster row OOMs at
//!   genome scale.

use super::context::for_each_context_share;
use super::two_track::{embed_two_track, TwoTrackConfig, TwoTrackEmbedding, TwoTrackInput};
use crate::common::*;
use data_beans::sparse_io::open_sparse_matrix_by_path;
use genomic_data::coordinates::parse_peak_coordinates;
use graph_embedding_util as ge;
use legume_numeric::candle::candle_core::Device;
use legume_numeric::matrix::parquet::{write_table, Column};
use legume_numeric::matrix::traits::ConvertMatOps;
use nalgebra::DMatrix;
use rustc_hash::{FxHashMap, FxHashSet};

#[derive(Debug, Clone)]
pub struct LinkConfig {
    pub embed: TwoTrackConfig,
    /// Target Leiden cluster count; `None` lets the resolution decide.
    pub n_clusters: Option<usize>,
    /// ATAC cells per block when streaming the cluster accessibility rates.
    pub block_size: usize,
}

impl Default for LinkConfig {
    fn default() -> Self {
        Self {
            embed: TwoTrackConfig::default(),
            n_clusters: None,
            block_size: 1000,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LinkSummary {
    pub n_genes: usize,
    pub n_peaks: usize,
    pub n_pairs: usize,
    pub n_clusters: usize,
    pub theta0: f64,
    pub theta1: f64,
    pub theta3: f64,
    pub gamma1: f64,
    pub gamma2: f64,
}

pub fn run_links(
    input: &TwoTrackInput,
    cfg: &LinkConfig,
    keep: Option<&FxHashSet<Box<str>>>,
) -> anyhow::Result<LinkSummary> {
    let out = input.work_prefix;
    let emb = embed_two_track(input, &cfg.embed)?;

    let kept: Vec<usize> = match keep {
        Some(set) => (0..emb.barcodes.len())
            .filter(|&c| {
                let b = emb.barcodes[c].as_ref();
                set.contains(b) || b.split_once('@').is_some_and(|(raw, _)| set.contains(raw))
            })
            .collect(),
        None => (0..emb.barcodes.len()).collect(),
    };
    anyhow::ensure!(!kept.is_empty(), "no embedded cell passed QC");
    info!("Cells: {} of {} pass QC", kept.len(), emb.barcodes.len());
    let cell_rows = emb.cell_rows.select_rows(kept.iter());
    let barcodes: Vec<Box<str>> = kept.iter().map(|&c| emb.barcodes[c].clone()).collect();

    let dev = Device::Cpu;
    let (kept_labels, _) = ge::cell_clusters(&cell_rows.to_tensor(&dev)?, cfg.n_clusters)?;
    let n_clusters = kept_labels.iter().max().map_or(0, |m| m + 1);
    let mut labels: Vec<Option<usize>> = vec![None; emb.barcodes.len()];
    for (&c, &l) in kept.iter().zip(&kept_labels) {
        labels[c] = Some(l);
    }

    let lambda = stream_cluster_rates(input.atac_file, &emb, &labels, n_clusters, cfg)?;
    info!(
        "Gates (phase-1 mix): θ0={:.3} θ1={:.3} θ3={:.3}, γ1={:.3} γ2={:.3}",
        emb.gate_theta0, emb.gate_theta1, emb.gate_theta3, emb.gate_gamma1, emb.gate_gamma2
    );

    let save = |stem: &str, m: &DMatrix<f32>, names: &[Box<str>], axis: &str| {
        ge::save_embedding(
            &format!("{out}.{stem}.parquet"),
            &m.to_tensor(&dev)?,
            names,
            axis,
        )
    };
    save("gene_embedding", &emb.rna_rows, &emb.genes, "gene")?;
    save("cell_embedding", &cell_rows, &barcodes, "cell")?;
    save("peak_embedding", &emb.peak_rows, &emb.peak_names, "peak")?;

    let label_i32: Vec<i32> = kept_labels.iter().map(|&l| l as i32).collect();
    write_table(
        &format!("{out}.cell_clusters.parquet"),
        &[
            ("cell".into(), Column::Str(&barcodes)),
            ("cluster".into(), Column::I32(&label_i32)),
        ],
    )?;
    write_peaks(&format!("{out}.peaks.parquet"), &emb.peak_names)?;
    write_links(out, &emb, &emb.gate_w, &lambda)?;

    Ok(LinkSummary {
        n_genes: emb.rna_genes.len(),
        n_peaks: emb.peak_names.len(),
        n_pairs: emb.pairs.n_pairs(),
        n_clusters,
        theta0: f64::from(emb.gate_theta0),
        theta1: f64::from(emb.gate_theta1),
        theta3: f64::from(emb.gate_theta3),
        gamma1: f64::from(emb.gate_gamma1),
        gamma2: f64::from(emb.gate_gamma2),
    })
}

fn stream_cluster_rates(
    atac_file: &str,
    emb: &TwoTrackEmbedding,
    labels: &[Option<usize>],
    n_clusters: usize,
    cfg: &LinkConfig,
) -> anyhow::Result<DMatrix<f32>> {
    let atac = open_sparse_matrix_by_path(atac_file)?;
    let n_peaks = emb.peak_names.len();
    let cell_of: FxHashMap<&str, usize> = emb
        .barcodes
        .iter()
        .enumerate()
        .map(|(i, b)| (b.as_ref(), i))
        .collect();
    let columns = atac.column_names()?;
    let cell_idx: Vec<Option<(usize, usize)>> = columns
        .iter()
        .map(|b| {
            let c = *cell_of.get(b.as_ref())?;
            Some((c, labels[c]?))
        })
        .collect();
    let n_matched = cell_idx.iter().filter(|c| c.is_some()).count();
    anyhow::ensure!(
        n_matched > 0,
        "no ATAC barcode matches an embedded cell that passed QC"
    );
    let block = cfg.block_size.max(1);
    let blocks: Vec<std::ops::Range<usize>> = (0..columns.len())
        .step_by(block)
        .map(|lb| lb..(lb + block).min(columns.len()))
        .collect();

    let mut cl_size = vec![0f32; n_clusters];
    let mut counts = DMatrix::<f32>::zeros(n_peaks, n_clusters);
    for cols in &blocks {
        let csc = atac.read_columns_csc(cols.clone().collect())?;
        for (j, col) in csc.col_iter().enumerate() {
            let Some((_, k)) = cell_idx[cols.start + j] else {
                continue;
            };
            let depth: f32 = col.values().iter().sum();
            cl_size[k] += depth;
            for (&p, &x) in col.row_indices().iter().zip(col.values()) {
                counts[(p, k)] += x;
            }
        }
    }
    Ok(DMatrix::from_fn(n_peaks, n_clusters, |p, k| {
        if cl_size[k] > 0.0 {
            counts[(p, k)] / cl_size[k]
        } else {
            0.0
        }
    }))
}

fn write_peaks(path: &str, names: &[Box<str>]) -> anyhow::Result<()> {
    let coords = parse_peak_coordinates(names);
    let chr: Vec<Box<str>> = coords
        .iter()
        .map(|c| c.as_ref().map_or_else(|| Box::from(""), |c| c.chr.clone()))
        .collect();
    let start: Vec<i64> = coords
        .iter()
        .map(|c| c.as_ref().map_or(-1, |c| c.start))
        .collect();
    let end: Vec<i64> = coords
        .iter()
        .map(|c| c.as_ref().map_or(-1, |c| c.end))
        .collect();
    write_table(
        path,
        &[
            ("peak".into(), Column::Str(names)),
            ("chromosome".into(), Column::Str(&chr)),
            ("start".into(), Column::I64(&start)),
            ("end".into(), Column::I64(&end)),
        ],
    )
}

fn write_links(
    out: &str,
    emb: &TwoTrackEmbedding,
    w: &[f32],
    lambda: &DMatrix<f32>,
) -> anyhow::Result<()> {
    let pairs = &emb.pairs;
    let mut gene: Vec<Box<str>> = Vec::with_capacity(pairs.n_pairs());
    for g in 0..pairs.n_genes() {
        gene.extend(pairs.gene(g).map(|_| emb.rna_genes[g].clone()));
    }
    let peak: Vec<Box<str>> = pairs
        .peak
        .iter()
        .map(|&p| emb.peak_names[p as usize].clone())
        .collect();
    write_table(
        &format!("{out}.links.parquet"),
        &[
            ("gene".into(), Column::Str(&gene)),
            ("peak".into(), Column::Str(&peak)),
            ("distance".into(), Column::I64(&pairs.dist)),
            ("abc".into(), Column::F32(&pairs.weight)),
            ("gate".into(), Column::F32(w)),
        ],
    )?;

    // Integer indices — not gene/peak name strings — so a genome-scale
    // pair×cluster table stays in the hundreds of MB, not tens of GB.
    let (mut g_i, mut p_i, mut k_i) = (Vec::<i32>::new(), Vec::<i32>::new(), Vec::<i32>::new());
    let (mut a_c, mut b_c) = (Vec::<f32>::new(), Vec::<f32>::new());
    let mut gene_of_pair = vec![0i32; pairs.n_pairs()];
    for g in 0..pairs.n_genes() {
        for k in pairs.gene(g) {
            gene_of_pair[k] = g as i32;
        }
    }
    for_each_context_share(pairs, w, lambda, |k, j, a, b| {
        g_i.push(gene_of_pair[k]);
        p_i.push(pairs.peak[k] as i32);
        k_i.push(j as i32);
        a_c.push(a);
        b_c.push(b);
    })?;
    write_table(
        &format!("{out}.links_by_cluster.parquet"),
        &[
            ("gene_idx".into(), Column::I32(&g_i)),
            ("peak_idx".into(), Column::I32(&p_i)),
            ("cluster".into(), Column::I32(&k_i)),
            ("gate".into(), Column::F32(&a_c)),
            ("abc".into(), Column::F32(&b_c)),
        ],
    )
}
