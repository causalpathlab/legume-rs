//! End-to-end peak-to-gene workflow on pb matrices.
//!
//! link_map → hierarchical embed over frozen pb units → cluster finest pb
//! rows → within-cluster refine → E2G parquet.

use crate::common::*;
use crate::p2g::cells::{cluster_labels_to_pb, embed_cells, write_cell_parquet, FrozenAxis};
use crate::p2g::cluster::cluster_cells;
use crate::p2g::embed_ge::{
    train_peak_gene_embeds, write_embedding_parquets, HierEmbedConfig, PeakGeneEmbeds,
};
use crate::p2g::link_map::{link_peaks_to_genes, LinkParams};
use crate::p2g::parquet_out::{peaks_from_coords, write_e2g_tables, ClusterRow};
use crate::p2g::pb_levels::PbLevels;
use crate::p2g::refine::refine_within_clusters;
use data_beans::sparse_io_vector::SparseIoVec;
use genomic_data::coordinates::{GeneTss, PeakCoord};
use legume_numeric::matrix::dense_mat_io::l2_normalize_rows_inplace;
use log::info;

/// The per-cell inputs of phase 2: one backend per frozen axis, in the order
/// the axes are handed to the embed (genes then peaks; ATAC-only: peaks only).
pub struct CellInputs<'a> {
    pub backends: Vec<&'a SparseIoVec>,
    pub barcodes: Vec<Box<str>>,
    /// Finest-level pb of every cell (column order of the backends).
    pub cell_to_pb: &'a [usize],
}

/// Knobs for [`run_from_pseudobulk`].
#[derive(Clone, Debug)]
pub struct WorkflowParams {
    pub abc: LinkParams,
    pub embed: HierEmbedConfig,
    pub min_cluster_samples: usize,
    pub target_clusters: Option<usize>,
}

/// The pb levels (links are scored on level 0) and feature metadata. In an
/// ATAC-only run `levels.rna` is `None` and `gene_activity` is the RNA
/// surrogate the links are scored against.
pub struct PbMultiome<'a> {
    pub levels: &'a PbLevels,
    pub gene_activity: Option<&'a Mat>,
    pub gene_tss: &'a [Option<GeneTss>],
    pub peak_coords: &'a [Option<PeakCoord>],
    pub gene_names: &'a [Box<str>],
    pub peak_names: &'a [Box<str>],
}

/// Run the ge-util peak-to-gene workflow from paired RNA/ATAC pseudobulk
/// matrices and the cells behind them: every cell is projected onto the frozen
/// dictionaries, the clusters are cell clusters, and each finest pb takes the
/// majority label of its cells for the refine.
pub fn run_from_pseudobulk(
    data: &PbMultiome<'_>,
    cells: &CellInputs<'_>,
    out_dir: &str,
    params: &WorkflowParams,
) -> anyhow::Result<()> {
    let PbMultiome {
        levels,
        gene_activity,
        gene_tss,
        peak_coords,
        gene_names,
        peak_names,
    } = data;
    anyhow::ensure!(levels.n_levels() > 0, "no pb level");
    let atac_pb: &Mat = &levels.atac[0];
    let rna_pb: &Mat = match (gene_activity, &levels.rna) {
        (Some(activity), _) => activity,
        (None, Some(rna)) => &rna[0],
        (None, None) => anyhow::bail!("neither RNA nor a gene activity surrogate"),
    };

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

    info!("Scoring peak–gene links ({:?})...", params.abc.score);
    let edges = link_peaks_to_genes(rna_pb, atac_pb, gene_tss, peak_coords, &params.abc)?;
    anyhow::ensure!(!edges.is_empty(), "no cis peak–gene edges above min_weight");
    info!("Link map: {} edges", edges.len());

    info!(
        "Training peak/gene/pb embeddings via hierarchical trainer \
         (dim={}, epochs={}, {} pb levels, units/step={})...",
        params.embed.dim,
        params.embed.epochs,
        levels.n_levels(),
        params.embed.units_per_step
    );
    let embeds = train_peak_gene_embeds(
        &edges,
        levels,
        peak_names,
        gene_names,
        rna_pb,
        &params.embed,
    )?;

    write_embedding_parquets(out_dir, &embeds)?;

    // Sample labels for the refine: cell clusters mapped onto the finest pb columns.
    let (sample_label, n_clusters) =
        cluster_cells_onto_pbs(cells, &embeds, levels, out_dir, params)?;

    info!("Refining peak→gene within clusters...");
    let links = refine_within_clusters(
        rna_pb,
        atac_pb,
        gene_tss,
        peak_coords,
        &sample_label,
        params.min_cluster_samples,
        &params.abc,
    )?;
    anyhow::ensure!(!links.is_empty(), "no within-cluster links after refine");
    info!("Refined links: {}", links.len());

    let peaks = peaks_from_coords(peak_coords);
    let cluster_rows: Vec<ClusterRow> = (0..n_clusters)
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

/// Phase 2 and the cell clustering: project every cell onto the frozen axes,
/// write the cell parquet, Leiden-cluster the L2-normalised rows, and label
/// each finest pb by the majority of its cells.
fn cluster_cells_onto_pbs(
    c: &CellInputs<'_>,
    embeds: &PeakGeneEmbeds,
    levels: &PbLevels,
    out_dir: &str,
    params: &WorkflowParams,
) -> anyhow::Result<(Vec<Option<usize>>, usize)> {
    let gene_axis = FrozenAxis {
        label: "gene",
        rows: &embeds.gene,
        bias: &embeds.gene_bias,
    };
    let peak_axis = FrozenAxis {
        label: "peak",
        rows: &embeds.peak,
        bias: &embeds.peak_bias,
    };
    // ATAC-only: the gene axis was trained on a pb-level surrogate with no
    // per-cell counts, so the cells are projected on the peaks alone.
    let axes: Vec<FrozenAxis> = if levels.rna.is_some() {
        vec![gene_axis, peak_axis]
    } else {
        info!("Cell embed: ATAC-only run, projecting on the peak axis alone");
        vec![peak_axis]
    };
    let mut rows = embed_cells(&axes, &c.backends, params.embed.dim, &params.embed.device)?;
    write_cell_parquet(out_dir, &rows, &c.barcodes)?;
    l2_normalize_rows_inplace(&mut rows);
    info!(
        "Clustering {} cells (min_cluster_samples={})...",
        rows.nrows(),
        params.min_cluster_samples
    );
    let clusters = cluster_cells(&rows, params.target_clusters, params.min_cluster_samples)?;
    info!(
        "Kept {} cell clusters (sizes={:?})",
        clusters.n_clusters, clusters.sizes
    );
    anyhow::ensure!(
        c.cell_to_pb.len() == clusters.label.len(),
        "cell_to_pb covers {} cells, embedding has {}",
        c.cell_to_pb.len(),
        clusters.label.len()
    );
    Ok((
        cluster_labels_to_pb(&clusters.label, c.cell_to_pb, levels.n_pb(0)),
        clusters.n_clusters,
    ))
}
