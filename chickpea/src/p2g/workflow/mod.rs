//! End-to-end peak-to-gene workflow on pb matrices.
//!
//! link_map → joint FNE over links + the pb levels → cluster the finest pb
//! rows → within-cluster refine → E2G parquet.

use crate::common::*;
use crate::p2g::cluster::cluster_cells;
use crate::p2g::embed_ge::{train_peak_gene_embeds, write_embedding_parquets};
use crate::p2g::link_map::{link_peaks_to_genes, LinkParams};
use crate::p2g::parquet_out::{peaks_from_coords, write_e2g_tables, ClusterRow};
use crate::p2g::pb_levels::PbLevels;
use crate::p2g::refine::refine_within_clusters;
use genomic_data::coordinates::{GeneTss, PeakCoord};
use graph_embedding_util::fne::FneConfig;
use log::info;

/// Knobs for [`run_from_pseudobulk`].
#[derive(Clone, Debug)]
pub struct WorkflowParams {
    pub abc: LinkParams,
    pub fne: FneConfig,
    /// SIMBA expression levels per modality in the pb × feature relations.
    pub context_bins: usize,
    /// Min pb samples (or cells) to keep a cluster.
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

/// Run the ge-util peak-to-gene workflow from paired RNA/ATAC pseudobulk matrices.
pub fn run_from_pseudobulk(
    data: &PbMultiome<'_>,
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
        "Training peak/gene/pb embeddings jointly via graph-embedding-util FNE \
         (dim={}, epochs={}, {} pb levels, {} bins)...",
        params.fne.dim,
        params.fne.epochs,
        levels.n_levels(),
        params.context_bins
    );
    let embeds = train_peak_gene_embeds(
        &edges,
        levels,
        peak_names,
        gene_names,
        params.context_bins,
        &params.fne,
    )?;

    write_embedding_parquets(out_dir, &embeds)?;
    let sample_mat = embeds.finest_unit_rows();
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
