//! Peak/gene/pb embeddings via `graph-embedding-util` hierarchical trainer.
//!
//! Frozen pb units and sparse count axes (genes, peaks), not FNE count edges.

use crate::common::Mat;
use crate::p2g::link_map::PeakGeneEdge;
use crate::p2g::module_init::{coarsen_profile_levels, init_gene_peak_partitions};
use crate::p2g::pb_levels::PbLevels;
use graph_embedding_util::data::Triplet;
use graph_embedding_util::fit::hier::{
    train_partitions, HierConfig, HierOutput, Partition, UnitTable,
};
use graph_embedding_util::fit::projection::CellGroup;
use graph_embedding_util::save_embedding;
use legume_numeric::candle::candle_core::{Device, Tensor};
use legume_numeric::matrix::dense_mat_io::{axis_id_names, l2_normalize_rows_inplace};
use legume_numeric::matrix::traits::ConvertMatOps;
use log::info;
use nalgebra::DMatrix;
use std::sync::atomic::AtomicBool;

/// Row embeddings after hierarchical training (CPU `[n, dim]` f32).
#[derive(Clone, Debug)]
pub struct PeakGeneEmbeds {
    pub peak: Vec<Vec<f32>>,
    pub gene: Vec<Vec<f32>>,
    /// Pseudobulk rows per level, finest first.
    pub pb: Vec<Vec<Vec<f32>>>,
    pub dim: usize,
    pub peak_names: Vec<Box<str>>,
    pub gene_names: Vec<Box<str>>,
    /// `⌈n_units / units_per_step⌉` from the hier trainer (for smoke checks).
    pub steps_per_epoch: usize,
    /// Per-feature bias of the gene axis, for the per-cell projection.
    pub gene_bias: Vec<f32>,
    /// Per-feature bias of the peak axis.
    pub peak_bias: Vec<f32>,
}

impl PeakGeneEmbeds {
    /// The finest pb rows, L2-normalised, as written to `pb_embedding.parquet`.
    fn finest_unit_rows(&self) -> Mat {
        let rows = &self.pb[0];
        let mut m = Mat::from_fn(rows.len(), self.dim, |i, d| rows[i][d]);
        l2_normalize_rows_inplace(&mut m);
        m
    }
}

/// Knobs for [`train_peak_gene_embeds`].
#[derive(Clone, Debug)]
pub struct HierEmbedConfig {
    pub dim: usize,
    pub epochs: usize,
    pub seed: u64,
    pub device: Device,
    pub n_gene_modules: usize,
    pub n_peak_modules: usize,
    pub units_per_step: usize,
    pub modules_per_unit: usize,
    pub lr: f32,
    pub merge_every: usize,
    pub merge_cosine: f32,
    /// Cells trained as units next to the pseudobulks: at most this many per
    /// pseudobulk at every level, unioned. `0` trains on pseudobulks alone.
    pub cells_per_pb: usize,
}

impl Default for HierEmbedConfig {
    fn default() -> Self {
        Self {
            dim: 32,
            epochs: 10,
            seed: 42,
            device: Device::Cpu,
            n_gene_modules: 128,
            n_peak_modules: 128,
            units_per_step: 256,
            modules_per_unit: 8,
            lr: 0.05,
            merge_every: 0,
            merge_cosine: 0.95,
            cells_per_pb: 16,
        }
    }
}

fn profile_to_triplets(m: &Mat) -> Vec<Triplet> {
    let mut out = Vec::new();
    for s in 0..m.ncols() {
        for f in 0..m.nrows() {
            let c = m[(f, s)];
            if c > 0.0 {
                out.push(Triplet {
                    cell: s as u32,
                    feature: f as u32,
                    count: c,
                });
            }
        }
    }
    out
}

fn build_units_and_partitions(
    levels: &PbLevels,
    gene_finest: &Mat,
    edges: &[PeakGeneEdge],
    cells: &CellGroup,
    cfg: &HierEmbedConfig,
) -> anyhow::Result<(UnitTable, Vec<Partition>)> {
    let n_pb = levels.n_pb_per_level();
    // Borrow the RNA levels when there are any; only the ATAC-only surrogate
    // needs its coarse levels built here.
    let coarsened;
    let gene_levels: Vec<&Mat> = match &levels.rna {
        Some(rna) => rna.iter().collect(),
        None => {
            coarsened = coarsen_profile_levels(gene_finest, &levels.parent);
            coarsened.iter().collect()
        }
    };
    anyhow::ensure!(
        gene_levels.len() == levels.atac.len(),
        "gene and ATAC level count mismatch"
    );
    for (l, (g, a)) in gene_levels.iter().zip(&levels.atac).enumerate() {
        anyhow::ensure!(
            g.ncols() == a.ncols() && g.ncols() == n_pb[l],
            "level {l} sample count mismatch"
        );
    }

    let gene_blobs: Vec<Vec<Triplet>> =
        gene_levels.iter().map(|m| profile_to_triplets(m)).collect();
    let peak_blobs: Vec<Vec<Triplet>> = levels.atac.iter().map(profile_to_triplets).collect();
    let gene_blob_refs: Vec<&[Triplet]> = gene_blobs.iter().map(Vec::as_slice).collect();
    let peak_blob_refs: Vec<&[Triplet]> = peak_blobs.iter().map(Vec::as_slice).collect();
    let axis_blobs = [&gene_blob_refs[..], &peak_blob_refs[..]];
    let n_features = [gene_finest.nrows(), levels.atac[0].nrows()];
    let units = UnitTable::from_pseudobulk_axes_and_cells(&axis_blobs, &n_pb, &n_features, cells);
    let (gene_part, peak_part) = init_gene_peak_partitions(
        levels,
        gene_finest,
        edges,
        cfg.n_gene_modules,
        cfg.n_peak_modules,
        cfg.seed,
    )?;
    Ok((units, vec![gene_part, peak_part]))
}

fn dmatrix_rows(m: &DMatrix<f32>) -> Vec<Vec<f32>> {
    (0..m.nrows())
        .map(|i| m.row(i).iter().copied().collect())
        .collect()
}

fn hier_output_to_embeds(
    out: &HierOutput,
    levels: &PbLevels,
    peak_names: &[Box<str>],
    gene_names: &[Box<str>],
    dim: usize,
) -> anyhow::Result<PeakGeneEmbeds> {
    anyhow::ensure!(out.axes.len() >= 2, "expected gene and peak axes");
    let gene = dmatrix_rows(&out.axes[0].rho);
    let peak = dmatrix_rows(&out.axes[1].rho);
    anyhow::ensure!(gene.len() == gene_names.len() && peak.len() == peak_names.len());

    let n_pb = levels.n_pb_per_level();
    let mut pb = Vec::with_capacity(n_pb.len());
    let mut off = 0usize;
    for &n in &n_pb {
        let mut level = Vec::with_capacity(n);
        for i in 0..n {
            let row: Vec<f32> = (0..dim).map(|d| out.e_u[(off + i, d)]).collect();
            level.push(row);
        }
        pb.push(level);
        off += n;
    }
    // Cell units, when phase 1 had them, follow the pseudobulk levels.
    anyhow::ensure!(off <= out.e_u.nrows(), "unit row count vs pb levels");

    Ok(PeakGeneEmbeds {
        peak,
        gene,
        pb,
        dim,
        peak_names: peak_names.to_vec(),
        gene_names: gene_names.to_vec(),
        steps_per_epoch: out.steps_per_epoch,
        gene_bias: out.axes[0].b_feat.clone(),
        peak_bias: out.axes[1].b_feat.clone(),
    })
}

/// Train peak/gene/pb embeddings via hierarchical softmax over frozen pb units.
pub fn train_peak_gene_embeds(
    edges: &[PeakGeneEdge],
    levels: &PbLevels,
    peak_names: &[Box<str>],
    gene_names: &[Box<str>],
    gene_finest: &Mat,
    cells: &CellGroup,
    cfg: &HierEmbedConfig,
) -> anyhow::Result<PeakGeneEmbeds> {
    anyhow::ensure!(!edges.is_empty(), "no peak–gene edges to embed");
    let (units, partitions) = build_units_and_partitions(levels, gene_finest, edges, cells, cfg)?;
    let n_u = units.n_units();
    info!(
        "Hier embed: {n_u} units ({} frozen pb + {} cells), {} genes, {} peaks, M=[{}, {}], H={}",
        n_u - cells.cells.len(),
        cells.cells.len(),
        gene_names.len(),
        peak_names.len(),
        partitions[0].n_modules(),
        partitions[1].n_modules(),
        cfg.dim
    );
    let stop = AtomicBool::new(false);
    let hcfg = HierConfig {
        n_modules: cfg.n_gene_modules,
        epochs: cfg.epochs,
        units_per_step: cfg.units_per_step,
        modules_per_unit: cfg.modules_per_unit,
        lr: cfg.lr,
        weight_decay: 0.0,
        seed: cfg.seed,
        offset_l2: 0.0,
        offset_rank: 1,
        device: cfg.device.clone(),
        merge_every: cfg.merge_every,
        merge_cosine: cfg.merge_cosine,
        // Peaks are module-level: the row is the module's, the bias the peak's
        // share of it. Peak-level rows come from a later refinement stage.
        module_only: vec![1],
    };
    let out = train_partitions(&units, &partitions, cfg.dim, &hcfg, None, &[], &stop)?;
    info!(
        "Hier embed finished: steps_per_epoch={} (⌈{n_u}/{}⌉)",
        out.steps_per_epoch, cfg.units_per_step
    );
    hier_output_to_embeds(&out, levels, peak_names, gene_names, cfg.dim)
}

/// Write peak / gene / finest-pb embedding parquets.
///
/// Finest units are `{prefix}.pb_embedding.parquet`, not `cell_embedding`,
/// because this path has no per-barcode projection onto the trained tables.
pub fn write_embedding_parquets(prefix: &str, embeds: &PeakGeneEmbeds) -> anyhow::Result<()> {
    anyhow::ensure!(embeds.dim > 0, "empty embedding dim");
    anyhow::ensure!(
        embeds.peak.len() == embeds.peak_names.len(),
        "peak rows vs names mismatch"
    );
    anyhow::ensure!(
        embeds.gene.len() == embeds.gene_names.len(),
        "gene rows vs names mismatch"
    );
    anyhow::ensure!(!embeds.pb.is_empty(), "no pb level to write");

    let peak_path = format!("{prefix}.peak_embedding.parquet");
    save_embedding(
        &peak_path,
        &rows_to_tensor(&embeds.peak, embeds.dim)?,
        &embeds.peak_names,
        "peak",
    )?;
    let gene_path = format!("{prefix}.gene_embedding.parquet");
    save_embedding(
        &gene_path,
        &rows_to_tensor(&embeds.gene, embeds.dim)?,
        &embeds.gene_names,
        "gene",
    )?;
    let finest = embeds.finest_unit_rows();
    let pb_path = format!("{prefix}.pb_embedding.parquet");
    save_embedding(
        &pb_path,
        &finest.to_tensor(&Device::Cpu)?,
        &axis_id_names("pb_", finest.nrows()),
        "pb",
    )?;
    info!(
        "Wrote embeddings: {peak_path} ({} × {}), {gene_path} ({} × {}), {pb_path} ({} × {})",
        embeds.peak.len(),
        embeds.dim,
        embeds.gene.len(),
        embeds.dim,
        finest.nrows(),
        embeds.dim
    );

    if embeds.pb.len() > 1 {
        let rows: Vec<Vec<f32>> = embeds.pb.iter().flatten().cloned().collect();
        let names: Vec<Box<str>> = embeds
            .pb
            .iter()
            .enumerate()
            .flat_map(|(l, level)| axis_id_names(&format!("L{l}:"), level.len()))
            .collect();
        let tree_path = format!("{prefix}.pb_tree_embedding.parquet");
        save_embedding(
            &tree_path,
            &rows_to_tensor(&rows, embeds.dim)?,
            &names,
            "pb",
        )?;
        info!(
            "Wrote pb tree embedding: {tree_path} ({} × {}, {} levels)",
            rows.len(),
            embeds.dim,
            embeds.pb.len()
        );
    }
    Ok(())
}

fn rows_to_tensor(rows: &[Vec<f32>], dim: usize) -> anyhow::Result<Tensor> {
    let n = rows.len();
    let mut flat = Vec::with_capacity(n * dim);
    for (i, row) in rows.iter().enumerate() {
        anyhow::ensure!(
            row.len() == dim,
            "ragged embedding at row {i}: len {} != {dim}",
            row.len()
        );
        flat.extend_from_slice(row);
    }
    Ok(Tensor::from_vec(flat, (n, dim), &Device::Cpu)?)
}
