//! On-disk outputs: parquet writers for the trained params + a JSON
//! manifest that senna (Mode A or Mode B) consumes via `--from`.

use anyhow::{Context, Result};
use candle_core::Tensor;
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::traits::IoOps;
use serde::{Deserialize, Serialize};

use graph_embedding_util::data::UnifiedData;

use super::feature_table::FeatureTable;
use super::model::RnaModEmbedModel;
use super::pseudobulk::PseudobulkData;

pub const MANIFEST_VERSION: u32 = 1;
pub const RNA_MOD_EMBED_KIND: &str = "rna-mod-embed";
pub const FEATURE_CONVENTION: &str = "gene/modality/detail";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct RnaModEmbedManifest {
    pub kind: String,
    pub version: u32,
    pub prefix: String,

    pub embedding_dim: usize,
    pub n_programs: usize,
    pub n_modalities: usize,
    pub n_cells: usize,
    pub feature_convention: String,

    /// ρ_g — `[G × H]` matrix; first column = gene name.
    pub gene_embedding: String,
    /// z_g — `[G × K]` matrix; first column = gene name. (Mode B only.)
    pub gene_program_loadings: String,
    /// Q — long format with columns
    /// `(program, modality, dim_0..dim_{H-1})`. (Mode B only.)
    pub program_signatures: String,
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
}

/// Write every model artifact under the configured prefix, then emit
/// `{prefix}.faba.json`.
pub fn write_outputs(
    prefix: &str,
    table: &FeatureTable,
    pb: &PseudobulkData,
    model: &RnaModEmbedModel,
    unified: &UnifiedData,
) -> Result<()> {
    let path_gene_emb = format!("{prefix}.gene_embedding.parquet");
    let path_z = format!("{prefix}.gene_program_loadings.parquet");
    let path_q = format!("{prefix}.program_signatures.parquet");
    let path_mod_axis = format!("{prefix}.modality_axis.parquet");
    let path_measured = format!("{prefix}.measured_mask.parquet");
    let path_agg_bias = format!("{prefix}.agg_bias.parquet");
    let path_comp_bias = format!("{prefix}.comp_bias.parquet");
    let path_cell_emb = format!("{prefix}.cell_embedding.parquet");
    let path_cell_bias = format!("{prefix}.cell_bias.parquet");
    let path_cell_to_pb = format!("{prefix}.cell_to_pb.parquet");
    let path_manifest = format!("{prefix}.faba.json");

    let dim_names: Vec<Box<str>> = (0..model.embedding_dim)
        .map(|h| format!("dim_{h}").into_boxed_str())
        .collect();
    let prog_names: Vec<Box<str>> = (0..model.n_programs)
        .map(|k| format!("program_{k}").into_boxed_str())
        .collect();

    // ρ_g
    let rho = tensor_to_dmatrix2(&model.rho)?;
    rho.to_parquet_with_names(
        &path_gene_emb,
        (Some(&table.gene_names), Some("gene")),
        Some(&dim_names),
    )
    .with_context(|| format!("writing {path_gene_emb}"))?;

    // z_g
    let z = tensor_to_dmatrix2(&model.z)?;
    z.to_parquet_with_names(
        &path_z,
        (Some(&table.gene_names), Some("gene")),
        Some(&prog_names),
    )
    .with_context(|| format!("writing {path_z}"))?;

    // Q in long format: row = (program, modality), columns = dim_0..dim_{H-1}.
    write_q_long(
        &model.q,
        &prog_names,
        &table.modality_names,
        &dim_names,
        &path_q,
    )?;

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
    // Cell-axis head
    ////////////////////////////////////////
    let e_cell = tensor_to_dmatrix2(&model.e_cell)?;
    e_cell
        .to_parquet_with_names(
            &path_cell_emb,
            (Some(&unified.barcodes), Some("cell")),
            Some(&dim_names),
        )
        .with_context(|| format!("writing {path_cell_emb}"))?;
    let b_cell = tensor_to_dmatrix1(&model.b_cell)?;
    b_cell
        .to_parquet_with_names(
            &path_cell_bias,
            (Some(&unified.barcodes), Some("cell")),
            Some(&[Box::from("b_cell")]),
        )
        .with_context(|| format!("writing {path_cell_bias}"))?;

    ////////////////////////////////////////
    // cell_to_pb partition
    ////////////////////////////////////////
    write_cell_to_pb(
        &pb.cell_to_pb_per_level,
        &unified.barcodes,
        &path_cell_to_pb,
    )?;

    let manifest = RnaModEmbedManifest {
        kind: RNA_MOD_EMBED_KIND.into(),
        version: MANIFEST_VERSION,
        prefix: prefix.into(),
        embedding_dim: model.embedding_dim,
        n_programs: model.n_programs,
        n_modalities: model.n_modalities,
        n_cells: model.n_cells,
        feature_convention: FEATURE_CONVENTION.into(),
        gene_embedding: path_gene_emb,
        gene_program_loadings: path_z,
        program_signatures: path_q,
        modality_axis: path_mod_axis,
        measured_mask: path_measured,
        cell_embedding: path_cell_emb,
        cell_bias: path_cell_bias,
        cell_to_pb: path_cell_to_pb,
        agg_bias: Some(path_agg_bias),
        comp_bias: Some(path_comp_bias),
    };

    let json = serde_json::to_string_pretty(&manifest)?;
    std::fs::write(&path_manifest, json).with_context(|| format!("writing {path_manifest}"))?;
    log::info!("rmodem: manifest → {path_manifest}");
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

fn write_q_long(
    q: &Tensor,
    program_names: &[Box<str>],
    modality_names: &[Box<str>],
    dim_names: &[Box<str>],
    path: &str,
) -> Result<()> {
    // Q shape: [K, M, H]
    let dims = q.shape().dims();
    anyhow::ensure!(dims.len() == 3, "Q must be 3D, got {:?}", dims);
    let (k, m, h) = (dims[0], dims[1], dims[2]);
    // Pull as Vec<Vec<Vec<f32>>>
    let data: Vec<Vec<Vec<f32>>> = q.to_vec3::<f32>()?;
    let n_rows = k * m;
    let row_names: Vec<Box<str>> = (0..k)
        .flat_map(|ki| {
            (0..m).map(move |mi| {
                format!("{}/{}", program_names[ki], modality_names[mi]).into_boxed_str()
            })
        })
        .collect();
    let mut mat = DMatrix::<f32>::zeros(n_rows, h);
    for (ki, prog_mat) in data.iter().enumerate() {
        for (mi, dim_vec) in prog_mat.iter().enumerate() {
            let row = ki * m + mi;
            for (hi, &v) in dim_vec.iter().enumerate() {
                mat[(row, hi)] = v;
            }
        }
    }
    mat.to_parquet_with_names(
        path,
        (Some(&row_names), Some("program_modality")),
        Some(dim_names),
    )
    .with_context(|| format!("writing {path}"))
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
