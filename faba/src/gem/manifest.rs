//! On-disk outputs: parquet writers for the trained params + a JSON
//! manifest that senna (Mode A or Mode B) consumes via `--from`.

use super::common::candle_core;
use anyhow::{Context, Result};
use candle_core::Tensor;
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::traits::IoOps;
use serde::{Deserialize, Serialize};

use graph_embedding_util::data::UnifiedData;

use super::feature_table::FeatureTable;
use super::model::GemModel;
use super::pseudobulk::PseudobulkData;

pub const MANIFEST_VERSION: u32 = 1;
pub const GEM_KIND: &str = "gem";
pub const FEATURE_CONVENTION: &str = "gene/modality/detail";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct GemManifest {
    pub kind: String,
    pub version: u32,
    pub prefix: String,

    pub embedding_dim: usize,
    pub n_programs: usize,
    pub n_modalities: usize,
    pub n_regions: usize,
    pub n_cells: usize,
    pub feature_convention: String,

    /// β_g — `[G × H]` base gene embedding; first column = gene name.
    pub gene_embedding: String,
    /// β_g again under senna's frozen-feature name (`[G × H]`). Lets
    /// `senna itopic --freeze-feature-embedding {prefix}` reuse the faba
    /// gene embedding as a fixed ρ.
    pub feature_embedding: String,
    /// z_g — `[G × K]` matrix; first column = gene name. (Mode B only.)
    pub gene_program_loadings: String,
    /// δ — long format `[(K·M) × H]`; row name `program_{k}/modality_{m}`.
    /// `δ_{k,m,:}` is the program×modality deviation **direction**
    /// (replaces the old scalar Q). (Mode B only.)
    pub program_modality_deviation: String,
    /// γ — long format `[(M·R) × H]`; row name `modality_{m}/region_{r}`.
    /// Additive per-(modality, region) log-space offset. (Mode B only.)
    pub modality_region_offset: String,
    /// `[M × 2]` (modality_id, modality_name) lookup.
    pub modality_axis: String,
    /// `[G × M]` binary measured mask.
    pub measured_mask: String,

    /// `e_cell` — `[N_cells × H]`; row name = cell barcode.
    pub cell_embedding: String,
    /// `b_cell` — `[N_cells × 1]`.
    pub cell_bias: String,
    /// `cell_to_pb` — `[N_cells × num_levels]` partition (coarsest-first
    /// columns) for downstream `pool(e_cell, cell_to_pb)` views.
    pub cell_to_pb: String,

    // Optional / diagnostic.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agg_bias: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comp_bias: Option<String>,

    // Archetype-based topics (`--resolve-topics`). Senna topic-model
    // layout: `latent` = log θ [N×K], `dictionary` = β [G×K],
    // `topic_embedding` = α [K×H]. Absent when topics weren't resolved.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_topics: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latent: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dictionary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub topic_embedding: Option<String>,
}

/// Write every model artifact (parquets) under the configured prefix.
/// The manifest is emitted separately by [`write_manifest`] **after**
/// optional topic resolution, so the durable embeddings land first and
/// the manifest can reference the resolved-topic files.
pub fn write_outputs(
    prefix: &str,
    table: &FeatureTable,
    pb: &PseudobulkData,
    model: &GemModel,
    unified: &UnifiedData,
    // QC-passed cell indices (see `--min-cell-nnz`). The per-cell outputs
    // (`cell_embedding`, `cell_bias`, `cell_to_pb`) are restricted to these
    // rows — a write-time selection, no backend rewrite. Feature-side
    // outputs are unaffected.
    keep_idx: &[usize],
) -> Result<()> {
    let path_gene_emb = format!("{prefix}.gene_embedding.parquet");
    let path_z = format!("{prefix}.gene_program_loadings.parquet");
    let path_delta = format!("{prefix}.program_modality_deviation.parquet");
    let path_gamma = format!("{prefix}.modality_region_offset.parquet");
    let path_mod_axis = format!("{prefix}.modality_axis.parquet");
    let path_measured = format!("{prefix}.measured_mask.parquet");
    let path_agg_bias = format!("{prefix}.agg_bias.parquet");
    let path_comp_bias = format!("{prefix}.comp_bias.parquet");
    let path_cell_emb = format!("{prefix}.cell_embedding.parquet");
    let path_cell_bias = format!("{prefix}.cell_bias.parquet");
    let path_cell_to_pb = format!("{prefix}.cell_to_pb.parquet");

    let dim_names: Vec<Box<str>> = (0..model.embedding_dim)
        .map(|h| format!("dim_{h}").into_boxed_str())
        .collect();
    let prog_names: Vec<Box<str>> = (0..model.n_programs)
        .map(|k| format!("program_{k}").into_boxed_str())
        .collect();

    // β_g
    let rho = tensor_to_dmatrix2(&model.beta)?;
    rho.to_parquet_with_names(
        &path_gene_emb,
        (Some(&table.gene_names), Some("gene")),
        Some(&dim_names),
    )
    .with_context(|| format!("writing {path_gene_emb}"))?;

    // Same matrix under senna's frozen-feature-side name so `senna itopic
    // --freeze-feature-embedding {prefix}` (topic layout, bias = 0) picks
    // up β_g as a fixed ρ [G, H] with no renaming. faba's own tooling
    // keeps using `gene_embedding.parquet`.
    let path_feat_emb = format!("{prefix}.feature_embedding.parquet");
    rho.to_parquet_with_names(
        &path_feat_emb,
        (Some(&table.gene_names), Some("gene")),
        Some(&dim_names),
    )
    .with_context(|| format!("writing {path_feat_emb}"))?;

    // z_g
    let z = tensor_to_dmatrix2(&model.z)?;
    z.to_parquet_with_names(
        &path_z,
        (Some(&table.gene_names), Some("gene")),
        Some(&prog_names),
    )
    .with_context(|| format!("writing {path_z}"))?;

    // δ in long format: rows = (K programs × M modalities), columns = H
    // dims. Row name `program_{k}/modality_{m}`; entry is the deviation
    // direction δ_{k,m,:}.
    let k = model.n_programs;
    let m = model.n_modalities;
    let r = model.n_regions;
    let h = model.embedding_dim;
    let delta_mat = tensor_to_dmatrix2(&model.delta.reshape((k * m, h))?)?;
    let delta_row_names: Vec<Box<str>> = (0..k)
        .flat_map(|kk| {
            table
                .modality_names
                .iter()
                .map(move |mn| format!("program_{kk}/{mn}").into_boxed_str())
        })
        .collect();
    delta_mat
        .to_parquet_with_names(
            &path_delta,
            (Some(&delta_row_names), Some("program_modality")),
            Some(&dim_names),
        )
        .with_context(|| format!("writing {path_delta}"))?;

    // γ in long format: rows = (M modalities × R regions), columns = H
    // dims. Row name `{modality}/region_{r}`.
    let gamma_mat = tensor_to_dmatrix2(&model.gamma.reshape((m * r, h))?)?;
    let gamma_row_names: Vec<Box<str>> = table
        .modality_names
        .iter()
        .flat_map(|mn| (0..r).map(move |rr| format!("{mn}/region_{rr}").into_boxed_str()))
        .collect();
    gamma_mat
        .to_parquet_with_names(
            &path_gamma,
            (Some(&gamma_row_names), Some("modality_region")),
            Some(&dim_names),
        )
        .with_context(|| format!("writing {path_gamma}"))?;

    // modality_axis: row = modality_id (0..M-1), single column = modality_name.
    write_modality_axis(&table.modality_names, &path_mod_axis)?;

    // measured_mask [G × M]
    write_measured_mask(table, &path_measured)?;

    // Biases
    let b_agg = tensor_to_dmatrix1(&model.b_agg)?;
    b_agg
        .to_parquet_with_names(
            &path_agg_bias,
            (Some(&table.gene_names), Some("gene")),
            Some(&[Box::from("b_agg")]),
        )
        .with_context(|| format!("writing {path_agg_bias}"))?;

    let b_comp = tensor_to_dmatrix2(&model.b_comp)?;
    b_comp
        .to_parquet_with_names(
            &path_comp_bias,
            (Some(&table.gene_names), Some("gene")),
            Some(&table.modality_names),
        )
        .with_context(|| format!("writing {path_comp_bias}"))?;

    ////////////////////////////////////////
    // Cell-axis head — restricted to QC-passed cells (`keep_idx`). Rows are
    // selected at write time; no backend is rewritten.
    ////////////////////////////////////////
    let barcodes_kept: Vec<Box<str>> = keep_idx
        .iter()
        .map(|&i| unified.barcodes[i].clone())
        .collect();

    let e_cell = tensor_to_dmatrix2(&model.e_cell)?.select_rows(keep_idx.iter());
    e_cell
        .to_parquet_with_names(
            &path_cell_emb,
            (Some(&barcodes_kept), Some("cell")),
            Some(&dim_names),
        )
        .with_context(|| format!("writing {path_cell_emb}"))?;
    let b_cell = tensor_to_dmatrix1(&model.b_cell)?.select_rows(keep_idx.iter());
    b_cell
        .to_parquet_with_names(
            &path_cell_bias,
            (Some(&barcodes_kept), Some("cell")),
            Some(&[Box::from("b_cell")]),
        )
        .with_context(|| format!("writing {path_cell_bias}"))?;

    ////////////////////////////////////////
    // cell_to_pb partition (kept cells only)
    ////////////////////////////////////////
    let kept_cell_to_pb: Vec<Vec<usize>> = pb
        .cell_to_pb_per_level
        .iter()
        .map(|c2p| keep_idx.iter().map(|&i| c2p[i]).collect())
        .collect();
    write_cell_to_pb(&kept_cell_to_pb, &barcodes_kept, &path_cell_to_pb)?;

    Ok(())
}

/// Emit `{prefix}.faba.json`. Called **after** [`write_outputs`] and any
/// `--resolve-topics` step so the manifest references the resolved-topic
/// artifacts when present. Output parquet paths are reconstructed from
/// `prefix` (deterministic, matching the names [`write_outputs`] uses).
pub fn write_manifest(
    prefix: &str,
    model: &GemModel,
    // Number of QC-passed cells actually written to the per-cell outputs.
    n_cells_written: usize,
    topics: Option<&super::topics::ResolvedTopics>,
) -> Result<()> {
    let path_manifest = format!("{prefix}.faba.json");
    let manifest = GemManifest {
        kind: GEM_KIND.into(),
        version: MANIFEST_VERSION,
        prefix: prefix.into(),
        embedding_dim: model.embedding_dim,
        n_programs: model.n_programs,
        n_modalities: model.n_modalities,
        n_regions: model.n_regions,
        n_cells: n_cells_written,
        feature_convention: FEATURE_CONVENTION.into(),
        gene_embedding: format!("{prefix}.gene_embedding.parquet"),
        feature_embedding: format!("{prefix}.feature_embedding.parquet"),
        gene_program_loadings: format!("{prefix}.gene_program_loadings.parquet"),
        program_modality_deviation: format!("{prefix}.program_modality_deviation.parquet"),
        modality_region_offset: format!("{prefix}.modality_region_offset.parquet"),
        modality_axis: format!("{prefix}.modality_axis.parquet"),
        measured_mask: format!("{prefix}.measured_mask.parquet"),
        cell_embedding: format!("{prefix}.cell_embedding.parquet"),
        cell_bias: format!("{prefix}.cell_bias.parquet"),
        cell_to_pb: format!("{prefix}.cell_to_pb.parquet"),
        agg_bias: Some(format!("{prefix}.agg_bias.parquet")),
        comp_bias: Some(format!("{prefix}.comp_bias.parquet")),
        num_topics: topics.map(|t| t.k),
        latent: topics.map(|t| t.latent.clone()),
        dictionary: topics.map(|t| t.dictionary.clone()),
        topic_embedding: topics.map(|t| t.topic_embedding.clone()),
    };

    let json = serde_json::to_string_pretty(&manifest)?;
    std::fs::write(&path_manifest, json).with_context(|| format!("writing {path_manifest}"))?;
    log::info!("manifest → {path_manifest}");
    Ok(())
}

////////////////////////////////////////
// Helpers
////////////////////////////////////////

fn tensor_to_dmatrix2(t: &Tensor) -> Result<DMatrix<f32>> {
    let shape = t.shape().dims();
    anyhow::ensure!(shape.len() == 2, "expected 2D tensor, got {:?}", shape);
    let rows = shape[0];
    let cols = shape[1];
    let v: Vec<Vec<f32>> = t.to_vec2::<f32>()?;
    Ok(DMatrix::from_row_iterator(
        rows,
        cols,
        v.into_iter().flatten(),
    ))
}

fn tensor_to_dmatrix1(t: &Tensor) -> Result<DMatrix<f32>> {
    let shape = t.shape().dims();
    anyhow::ensure!(shape.len() == 1, "expected 1D tensor, got {:?}", shape);
    let n = shape[0];
    let v: Vec<f32> = t.to_vec1::<f32>()?;
    Ok(DMatrix::from_column_slice(n, 1, &v))
}

fn write_modality_axis(modality_names: &[Box<str>], path: &str) -> Result<()> {
    // A single-column "name" DMatrix isn't expressible directly (matrix
    // values are numeric). Encode as an indexed 1-column f32 matrix
    // where the row name carries the modality string and the value is
    // the modality id — symmetric with how the measured_mask is keyed.
    let n = modality_names.len();
    let ids: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let mat = DMatrix::from_column_slice(n, 1, &ids);
    mat.to_parquet_with_names(
        path,
        (Some(modality_names), Some("modality")),
        Some(&[Box::from("modality_id")]),
    )
    .with_context(|| format!("writing {path}"))
}

fn write_cell_to_pb(
    cell_to_pb_per_level: &[Vec<usize>],
    barcodes: &[Box<str>],
    path: &str,
) -> Result<()> {
    let n_cells = barcodes.len();
    let n_levels = cell_to_pb_per_level.len();
    if n_levels == 0 {
        // Nothing to write — emit an empty single-column matrix to keep
        // the manifest contract uniform.
        let mat = DMatrix::<f32>::zeros(n_cells, 1);
        let dummy: Vec<Box<str>> = vec![Box::from("level_0")];
        return mat
            .to_parquet_with_names(path, (Some(barcodes), Some("cell")), Some(&dummy))
            .with_context(|| format!("writing {path}"));
    }
    let level_names: Vec<Box<str>> = (0..n_levels)
        .map(|l| format!("level_{l}").into_boxed_str())
        .collect();
    let mut mat = DMatrix::<f32>::zeros(n_cells, n_levels);
    for (l, c2p) in cell_to_pb_per_level.iter().enumerate() {
        for (c, &p) in c2p.iter().enumerate() {
            mat[(c, l)] = p as f32;
        }
    }
    mat.to_parquet_with_names(path, (Some(barcodes), Some("cell")), Some(&level_names))
        .with_context(|| format!("writing {path}"))
}

fn write_measured_mask(table: &FeatureTable, path: &str) -> Result<()> {
    let g = table.n_genes();
    let m = table.n_modalities();
    let mut mat = DMatrix::<f32>::zeros(g, m);
    for gi in 0..g {
        for mi in 0..m {
            mat[(gi, mi)] = if table.measured[gi][mi] { 1.0 } else { 0.0 };
        }
    }
    mat.to_parquet_with_names(
        path,
        (Some(&table.gene_names), Some("gene")),
        Some(&table.modality_names),
    )
    .with_context(|| format!("writing {path}"))
}
