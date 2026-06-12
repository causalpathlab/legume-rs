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
use super::model::{GemModel, PARAM_INIT_STD};
use super::pseudobulk::PseudobulkData;

use statrs::distribution::{ChiSquared, ContinuousCDF};

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
    /// `[G × 3]` per-feature prior-score QC (`emb_sq_norm`, `chisq_stat`,
    /// `prior_pval`); flags rows training never moved off the init prior.
    /// `#[serde(default)]` so manifests written before this field parse.
    #[serde(default)]
    pub feature_prior_score: String,
    /// `[N_cells × 3]` per-cell prior-score QC (`emb_sq_norm`, `chisq_stat`,
    /// `prior_pval`); flags cells whose phase-2 projection solved near-zero
    /// (expressed only dead genes).  Empty string when phase-2 was skipped.
    /// `#[serde(default)]` so manifests written before this field parse.
    #[serde(default)]
    pub cell_prior_score: String,

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

/// Cell-axis QC outputs passed to [`write_outputs`].
///
/// `keep_idx` — QC-passed cell indices (spliced auto cell-calling); the per-cell
/// outputs (`cell_embedding`, `cell_bias`, `cell_to_pb`) are restricted to
/// these rows at write time, no backend rewrite.
///
/// `cell_nrms` — pre-L2-normalisation norms from phase-2 cell projection,
/// one per model cell (0..n_cells).  Empty slice when phase-2 was skipped.
pub struct CellQcOutputs<'a> {
    pub keep_idx: &'a [usize],
    pub cell_nrms: &'a [f32],
}

/// Write every model artifact (parquets) under the configured prefix.
/// The manifest is emitted separately by [`write_manifest`] **after**
/// optional topic resolution, so the durable embeddings land first and
/// the manifest can reference the resolved-topic files.
///
/// `score_prefix` controls the output path for the prior-score parquets
/// (`feature_prior_score`, `cell_prior_score`).  Pass the same value as
/// `prefix` for single-pass runs; pass `"{prefix}.pass1"` for the first
/// pass of a `--refine` run so pass-1 scores are preserved alongside the
/// final pass-2 artifacts.
pub fn write_outputs(
    prefix: &str,
    score_prefix: &str,
    table: &FeatureTable,
    pb: &PseudobulkData,
    model: &GemModel,
    unified: &UnifiedData,
    cell: CellQcOutputs<'_>,
) -> Result<()> {
    let CellQcOutputs {
        keep_idx,
        cell_nrms,
    } = cell;
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

    // Per-feature prior-score QC. A row that training never moved off its
    // `N(0, σ²I)` init (σ = PARAM_INIT_STD) is uninformed noise — the dense
    // small-norm blob that contaminates downstream UMAP / clustering /
    // archetype topics. Score each β_g row against that prior null.
    let path_feat_prior = format!("{score_prefix}.feature_prior_score.parquet");
    write_feature_prior_score(
        &rho,
        &table.gene_names,
        model.embedding_dim,
        &path_feat_prior,
    )?;

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

    ////////////////////////////////////////
    // Per-cell prior-score QC
    ////////////////////////////////////////
    if !cell_nrms.is_empty() {
        let path_cell_prior = format!("{score_prefix}.cell_prior_score.parquet");
        write_cell_prior_score(
            cell_nrms,
            keep_idx,
            &barcodes_kept,
            model.embedding_dim,
            &path_cell_prior,
        )?;
    }

    Ok(())
}

/// Emit `{prefix}.faba.json`. Called **after** [`write_outputs`] and any
/// `--resolve-topics` step so the manifest references the resolved-topic
/// artifacts when present. Output parquet paths are reconstructed from
/// `prefix` (deterministic, matching the names [`write_outputs`] uses).
///
/// `has_cell_prior`: set `true` when `write_outputs` produced a
/// `{prefix}.cell_prior_score.parquet` (i.e. phase-2 ran and returned
/// non-empty norms).
pub fn write_manifest(
    prefix: &str,
    model: &GemModel,
    // Number of QC-passed cells actually written to the per-cell outputs.
    n_cells_written: usize,
    topics: Option<&super::topics::ResolvedTopics>,
    has_cell_prior: bool,
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
        feature_prior_score: format!("{prefix}.feature_prior_score.parquet"),
        cell_prior_score: if has_cell_prior {
            format!("{prefix}.cell_prior_score.parquet")
        } else {
            String::new()
        },
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
// Manifest reading (shared by gem-plot / gem-annotate)
////////////////////////////////////////

/// Parse a `{prefix}.faba.json` manifest, returning it plus the directory
/// it lives in (for resolving its sibling parquet paths).
pub(crate) fn load_manifest(from: &str) -> Result<(GemManifest, std::path::PathBuf)> {
    let txt = std::fs::read_to_string(from).with_context(|| format!("reading manifest {from}"))?;
    let manifest: GemManifest =
        serde_json::from_str(&txt).with_context(|| format!("parsing manifest {from}"))?;
    let dir = std::path::Path::new(from)
        .parent()
        .map(std::path::Path::to_path_buf)
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    Ok((manifest, dir))
}

/// Resolve a manifest-stored parquet path against the manifest's own
/// directory (by basename), so a relocated run directory still works.
pub(crate) fn resolve(dir: &std::path::Path, stored: &str) -> String {
    let p = std::path::Path::new(stored);
    let name = p.file_name().map(std::path::Path::new).unwrap_or(p);
    dir.join(name).to_string_lossy().into_owned()
}

/// Default output prefix: the manifest directory + the prefix basename.
pub(crate) fn default_out(dir: &std::path::Path, prefix: &str) -> String {
    let base = std::path::Path::new(prefix)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| prefix.to_owned());
    dir.join(base).to_string_lossy().into_owned()
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

const PRIOR_SCORE_COL_NAMES: [&str; 3] = ["emb_sq_norm", "chisq_stat", "prior_pval"];

/// Compute the three prior-score columns for one embedding row given its
/// pre-squared norm.  Returns `[sq_norm_f32, chisq_f32, pval_f32]`.
fn prior_score_cols(sq_norm: f64, var: f64, dist: &ChiSquared) -> [f32; 3] {
    let chisq = sq_norm / var;
    [sq_norm as f32, chisq as f32, dist.sf(chisq) as f32]
}

/// Per-feature prior-score QC, written to `{prefix}.feature_prior_score.parquet`.
///
/// Under the null "training never informed this row", the embedding is a draw
/// from the initialiser prior `e_f ~ N(0, σ²I_H)` (σ = [`PARAM_INIT_STD`]), so
///
/// ```text
///     chisq = ‖e_f‖² / σ²  ~  χ²_H .
/// ```
///
/// `prior_pval = P(χ²_H ≥ chisq)` is then a standard upper-tail p-value for
/// "distinguishable from the prior": a trained row grows its norm off the tiny
/// init shell → large `chisq` → `prior_pval → 0` (live feature); an uninformed
/// row stays on the shell → `chisq ≈ H` → `prior_pval ≈ U(0,1)` (mean 0.5).
/// Flag dead features with e.g. `prior_pval > 0.05`. `chisq` is the unbounded,
/// precision-stable monotone score (large = live); `prior_pval` is the
/// calibrated reading.
fn write_feature_prior_score(
    rho: &DMatrix<f32>,
    gene_names: &[Box<str>],
    h: usize,
    path: &str,
) -> Result<()> {
    let g = rho.nrows();
    let var = PARAM_INIT_STD * PARAM_INIT_STD;
    let dist = ChiSquared::new(h as f64).context("chi-squared dof (embedding_dim)")?;

    let mut out = DMatrix::<f32>::zeros(g, 3);
    let mut n_prior_like = 0usize;
    for r in 0..g {
        let sq_norm = rho
            .row(r)
            .iter()
            .map(|&x| (x as f64) * (x as f64))
            .sum::<f64>();
        let cols = prior_score_cols(sq_norm, var, &dist);
        if cols[2] > 0.05 {
            n_prior_like += 1;
        }
        out[(r, 0)] = cols[0];
        out[(r, 1)] = cols[1];
        out[(r, 2)] = cols[2];
    }

    let col_names: Vec<Box<str>> = PRIOR_SCORE_COL_NAMES
        .iter()
        .map(|s| Box::from(*s))
        .collect();
    out.to_parquet_with_names(path, (Some(gene_names), Some("gene")), Some(&col_names))
        .with_context(|| format!("writing {path}"))?;

    log::info!(
        "feature prior-score → {} ({} / {} rows ≈ prior at p>0.05 — candidate uninformed/dead features)",
        path,
        n_prior_like,
        g
    );
    Ok(())
}

/// Per-cell prior-score QC, written to `{score_prefix}.cell_prior_score.parquet`.
///
/// Symmetric with [`write_feature_prior_score`] but applied to the
/// **pre-L2-normalisation** norms from the phase-2 Poisson-MAP projection.
///
/// Under the null "the cell's expressed genes are all near-zero (dead)", the
/// IRLS ridge prior dominates and pulls `e_cell_raw → 0`, so `nrm ≈ 0`.
/// The same statistic `chisq = nrm² / σ²  ~  χ²_H` is used, but its
/// direction is inverted relative to features:
///
/// ```text
/// dead cell:  nrm ≈ 0  →  chisq ≈ 0  →  prior_pval ≈ 1  (flag: pval > threshold)
/// live cell:  nrm > 0  →  chisq > 0  →  prior_pval < 1
/// ```
///
/// `prior_pval > threshold` (same threshold direction as features) marks dead cells.
/// Only kept cells (`keep_idx`) are written.
fn write_cell_prior_score(
    cell_nrms: &[f32],
    keep_idx: &[usize],
    barcodes: &[Box<str>],
    h: usize,
    path: &str,
) -> Result<()> {
    let n = keep_idx.len();
    let var = PARAM_INIT_STD * PARAM_INIT_STD;
    let dist = ChiSquared::new(h as f64).context("chi-squared dof (embedding_dim)")?;

    let mut out = DMatrix::<f32>::zeros(n, 3);
    let mut n_dead = 0usize;
    for (row, &ci) in keep_idx.iter().enumerate() {
        let nrm = cell_nrms[ci] as f64;
        let cols = prior_score_cols(nrm * nrm, var, &dist);
        if cols[2] > 0.05 {
            n_dead += 1;
        }
        out[(row, 0)] = cols[0];
        out[(row, 1)] = cols[1];
        out[(row, 2)] = cols[2];
    }

    let col_names: Vec<Box<str>> = PRIOR_SCORE_COL_NAMES
        .iter()
        .map(|s| Box::from(*s))
        .collect();
    out.to_parquet_with_names(path, (Some(barcodes), Some("cell")), Some(&col_names))
        .with_context(|| format!("writing {path}"))?;

    log::info!(
        "cell prior-score → {} ({} / {} cells dead at p>0.05 — candidate dead-gene-region cells)",
        path,
        n_dead,
        n
    );
    Ok(())
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
