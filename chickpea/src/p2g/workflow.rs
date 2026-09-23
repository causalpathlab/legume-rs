//! The link workflow: gene-centric embedding, peak fold-in, localized
//! attention, per-cluster tables.
//!
//! Outputs, all `{out}.*.parquet`:
//!
//! - `gene_embedding` / `gene_atac_embedding`: the two tracks' gene rows;
//! - `cell_embedding`, `cell_clusters` (`cell`, `cluster`);
//! - `peak_embedding`, `peaks` (`peak`, `chromosome`, `start`, `end`, `bias`);
//! - `links`: one row per cis pair (`gene`, `peak`, `distance`, `abc`,
//!   `attention`), shares summing to 1 per gene;
//! - `links_by_cluster`: (`gene`, `peak`, `cluster`, `attention`, `abc`), the
//!   shares re-weighted by each cluster's accessibility; pairs whose peak is
//!   closed in a cluster are left out.

use super::attention::{fit_attention, AttentionConfig};
use super::context::context_shares;
use super::peak_foldin::{FoldInDesign, PeakMoments};
use super::two_track::{embed_two_track, TwoTrackConfig, TwoTrackEmbedding, TwoTrackInput};
use crate::common::*;
use data_beans::sparse_io::open_sparse_matrix_by_path;
use genomic_data::coordinates::parse_peak_coordinates;
use graph_embedding_util as ge;
use legume_numeric::candle::candle_core::Device;
use legume_numeric::matrix::parquet::{write_table, Column};
use legume_numeric::matrix::traits::ConvertMatOps;
use nalgebra::DMatrix;
use rustc_hash::FxHashMap;

#[derive(Debug, Clone)]
pub struct LinkConfig {
    pub embed: TwoTrackConfig,
    pub attention: AttentionConfig,
    /// Ridge of the peak fold-in, relative to the unit design's mean diagonal.
    pub foldin_ridge: f32,
    /// Target Leiden cluster count; `None` lets the resolution decide.
    pub n_clusters: Option<usize>,
}

impl Default for LinkConfig {
    fn default() -> Self {
        Self {
            embed: TwoTrackConfig::default(),
            attention: AttentionConfig::default(),
            foldin_ridge: 1e-3,
            n_clusters: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LinkSummary {
    pub n_genes: usize,
    pub n_peaks: usize,
    pub n_pairs: usize,
    pub n_clusters: usize,
    pub gamma: f64,
    pub pseudocount: f64,
    pub attention_loss: Vec<f64>,
}

/// Run everything and write the tables under `inp.work_prefix`.
pub fn run_links(inp: &TwoTrackInput, cfg: &LinkConfig) -> anyhow::Result<LinkSummary> {
    let out = inp.work_prefix;
    let emb = embed_two_track(inp, &cfg.embed)?;

    let dev = Device::Cpu;
    let (labels, _) = ge::cell_clusters(&emb.cell_rows.to_tensor(&dev)?, cfg.n_clusters)?;
    let n_clusters = labels.iter().max().map_or(0, |m| m + 1);

    ///////////////////////////////////////////
    // ATAC, streamed: fold-in and λ per cluster //
    ///////////////////////////////////////////
    let (fold, lambda) = stream_atac(inp.atac_file, &emb, &labels, n_clusters, cfg)?;
    info!(
        "Peak fold-in: {} peaks × {} dims; {n_clusters} clusters",
        fold.phi.nrows(),
        fold.phi.ncols()
    );

    ////////////////////////////
    // Attention and contexts //
    ////////////////////////////
    let rho = rna_rows_in_pair_order(&emb)?;
    let fit = fit_attention(&rho, &fold.phi, &emb.pairs, &cfg.attention)?;
    info!(
        "Attention: γ = {:.3}, pseudocount = {:.0} bp, loss {:.4} → {:.4}",
        fit.gamma,
        fit.pseudocount,
        fit.loss.first().copied().unwrap_or(f64::NAN),
        fit.loss.last().copied().unwrap_or(f64::NAN)
    );
    let ctx_att = context_shares(&emb.pairs, &fit.pi, &lambda)?;
    let ctx_abc = context_shares(&emb.pairs, &emb.pairs.weight, &lambda)?;

    ////////////
    // Tables //
    ////////////
    let save = |stem: &str, m: &DMatrix<f32>, names: &[Box<str>], axis: &str| {
        ge::save_embedding(
            &format!("{out}.{stem}.parquet"),
            &m.to_tensor(&dev)?,
            names,
            axis,
        )
    };
    save("gene_embedding", &emb.rna_rows, &emb.genes, "gene")?;
    let atac_names: Vec<Box<str>> = emb
        .atac_gene
        .iter()
        .zip(&emb.genes)
        .filter(|(a, _)| a.is_some())
        .map(|(_, g)| g.clone())
        .collect();
    save("gene_atac_embedding", &emb.atac_rows, &atac_names, "gene")?;
    save("cell_embedding", &emb.cell_rows, &emb.barcodes, "cell")?;
    save("peak_embedding", &fold.phi, &emb.peak_names, "peak")?;

    let label_i32: Vec<i32> = labels.iter().map(|&l| l as i32).collect();
    write_table(
        &format!("{out}.cell_clusters.parquet"),
        &[
            ("cell".into(), Column::Str(&emb.barcodes)),
            ("cluster".into(), Column::I32(&label_i32)),
        ],
    )?;
    write_peaks(&format!("{out}.peaks.parquet"), &emb.peak_names, &fold.bias)?;
    write_links(out, &emb, &fit.pi, &ctx_att, &ctx_abc)?;

    Ok(LinkSummary {
        n_genes: emb.rna_genes.len(),
        n_peaks: emb.peak_names.len(),
        n_pairs: emb.pairs.n_pairs(),
        n_clusters,
        gamma: fit.gamma,
        pseudocount: fit.pseudocount,
        attention_loss: fit.loss,
    })
}

/// Two passes over the ATAC cells: unit and cluster depths, then the fold-in
/// moments and each cluster's counts. Returns the fold-in and `λ` (per-cluster
/// accessibility rates, `[peaks × clusters]`).
fn stream_atac(
    atac_file: &str,
    emb: &TwoTrackEmbedding,
    labels: &[usize],
    n_clusters: usize,
    cfg: &LinkConfig,
) -> anyhow::Result<(super::peak_foldin::PeakFoldIn, DMatrix<f32>)> {
    let atac = open_sparse_matrix_by_path(atac_file)?;
    let n_peaks = emb.peak_names.len();
    let cell_of: FxHashMap<&str, usize> = emb
        .barcodes
        .iter()
        .enumerate()
        .map(|(i, b)| (b.as_ref(), i))
        .collect();
    let columns = atac.column_names()?;
    let cell_idx: Vec<Option<usize>> = columns
        .iter()
        .map(|b| cell_of.get(b.as_ref()).copied())
        .collect();
    let n_matched = cell_idx.iter().filter(|c| c.is_some()).count();
    anyhow::ensure!(n_matched > 0, "no ATAC barcode matches an embedded cell");
    if n_matched < columns.len() {
        info!(
            "ATAC: {} of {} cells were not embedded and are skipped",
            columns.len() - n_matched,
            columns.len()
        );
    }
    let block = cfg.embed.block_size.max(1);
    let blocks: Vec<std::ops::Range<usize>> = (0..columns.len())
        .step_by(block)
        .map(|lb| lb..(lb + block).min(columns.len()))
        .collect();

    let n_pb = emb.pb_rows.nrows();
    let mut pb_size = vec![0f32; n_pb];
    let mut cl_size = vec![0f32; n_clusters];
    for cols in &blocks {
        let csc = atac.read_columns_csc(cols.clone().collect())?;
        for (j, col) in csc.col_iter().enumerate() {
            if let Some(c) = cell_idx[cols.start + j] {
                let depth: f32 = col.values().iter().sum();
                pb_size[emb.cell_to_pb[c]] += depth;
                cl_size[labels[c]] += depth;
            }
        }
    }

    let design = FoldInDesign::new(&emb.pb_rows, &pb_size, cfg.foldin_ridge)?;
    let mut moments = PeakMoments::new(n_peaks, design.dim());
    let mut counts = DMatrix::<f32>::zeros(n_peaks, n_clusters);
    let mut cell: Vec<(u32, f32)> = Vec::new();
    for cols in &blocks {
        let csc = atac.read_columns_csc(cols.clone().collect())?;
        for (j, col) in csc.col_iter().enumerate() {
            let Some(c) = cell_idx[cols.start + j] else {
                continue;
            };
            cell.clear();
            cell.extend(
                col.row_indices()
                    .iter()
                    .zip(col.values())
                    .map(|(&p, &x)| (p as u32, x)),
            );
            moments.add_cell(&design, emb.cell_to_pb[c], &cell);
            for &(p, x) in &cell {
                counts[(p as usize, labels[c])] += x;
            }
        }
    }
    let lambda = DMatrix::from_fn(n_peaks, n_clusters, |p, k| {
        if cl_size[k] > 0.0 {
            counts[(p, k)] / cl_size[k]
        } else {
            0.0
        }
    });
    Ok((design.finish(&moments), lambda))
}

/// The RNA rows reordered to the pairs' gene order (the RNA file's).
fn rna_rows_in_pair_order(emb: &TwoTrackEmbedding) -> anyhow::Result<DMatrix<f32>> {
    let row_of: FxHashMap<&str, usize> = emb
        .genes
        .iter()
        .enumerate()
        .map(|(r, g)| (g.as_ref(), r))
        .collect();
    let missing = emb
        .rna_genes
        .iter()
        .filter(|g| !row_of.contains_key(g.as_ref()))
        .count();
    anyhow::ensure!(
        missing == 0,
        "{missing} RNA genes have no embedded row (names changed on load?)"
    );
    Ok(emb
        .rna_rows
        .select_rows(emb.rna_genes.iter().map(|g| &row_of[g.as_ref()])))
}

fn write_peaks(path: &str, names: &[Box<str>], bias: &[f32]) -> anyhow::Result<()> {
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
            ("bias".into(), Column::F32(bias)),
        ],
    )
}

fn write_links(
    out: &str,
    emb: &TwoTrackEmbedding,
    pi: &[f32],
    ctx_att: &DMatrix<f32>,
    ctx_abc: &DMatrix<f32>,
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
            ("attention".into(), Column::F32(pi)),
        ],
    )?;

    let (mut g_c, mut p_c, mut k_c, mut a_c, mut b_c) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for k in 0..pairs.n_pairs() {
        for j in 0..ctx_att.ncols() {
            let (a, b) = (ctx_att[(k, j)], ctx_abc[(k, j)]);
            if a > 0.0 || b > 0.0 {
                g_c.push(gene[k].clone());
                p_c.push(peak[k].clone());
                k_c.push(Box::<str>::from(j.to_string()));
                a_c.push(a);
                b_c.push(b);
            }
        }
    }
    write_table(
        &format!("{out}.links_by_cluster.parquet"),
        &[
            ("gene".into(), Column::Str(&g_c)),
            ("peak".into(), Column::Str(&p_c)),
            ("cluster".into(), Column::Str(&k_c)),
            ("attention".into(), Column::F32(&a_c)),
            ("abc".into(), Column::F32(&b_c)),
        ],
    )
}
