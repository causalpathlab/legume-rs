//! `{out}.feature_contrast.parquet` / `{out}.feature_contrast_bias.parquet`:
//! one row per gene-and-modality, contrasting that modality's two channel
//! tracks on the RAW embedding (`model.e_feat`, the same object
//! `{out}.feature_embedding.parquet` holds), not the SIMBA co-embed
//! (`{out}.feature_coembedding.parquet`), which is a lossy convex-combination
//! view of the cell manifold that a subtraction is not well posed on.
//!
//! Every modality on the axis contrasts a fixed `(numerator, denominator)`
//! channel pair ([`super::tracks::contrast_channels`]): `count` is
//! `unspliced − spliced`, `m6a` is `methylated − unmethylated`, `atoi` is
//! `edited − unedited`, `apa` is `proximal − distal`. A gene carrying rows on
//! BOTH channels of a modality gets one row, `{gene}/{modality}`; a gene
//! missing either channel (no `--modality` file for that modality, or an
//! axis with no `unspliced` track at all) is skipped, and skipped genes are
//! counted once per modality rather than reported gene by gene. A modality
//! with no track for one of its two channels anywhere on the axis (e.g. a
//! `count`-only axis with no `unspliced` row at all) contributes no rows.
//!
//! This table is per-FEATURE. A cell whose counts sit only on a non-base
//! track carries a floor `b_cell` from the per-track polish (an absent base
//! track reads at the score-clamp floor); that is a per-CELL fact already
//! recorded in `{out}.cell_bias.parquet`, written the same way `senna bge`
//! writes it; nothing here special-cases it.

use crate::bge::driver::FitArtifacts;
use crate::gem::tracks::{contrast_channels, TrackPlan};
use graph_embedding_util as ge;
use matrix_util::parquet::{write_named_table, Column};
use rustc_hash::FxHashMap;
use senna::embed_common::*;

/// The contrast rows for one axis: `rows[i] = "{gene[i]}/{modality[i]}"`,
/// `delta[i, ..]` the `[H]` loading difference, `bias[i]` the scalar feature
/// bias difference.
pub(crate) struct ContrastTable {
    pub rows: Vec<Box<str>>,
    pub modality: Vec<Box<str>>,
    pub gene: Vec<Box<str>>,
    /// `[n × H]`, `H = rho.ncols()`.
    pub delta: Mat,
    pub bias: Vec<f32>,
}

/// Build the contrast table from the plan's row grammar and the RAW loading
/// (`rho` = `model.e_feat` as `[D × H]`, `b_feat` = `model.b_feat`). Pure
/// function of its inputs; see the module docs for the skip rule.
pub(crate) fn contrast_rows(
    plan: &TrackPlan,
    feature_names: &[Box<str>],
    rho: &Mat,
    b_feat: &[f32],
) -> ContrastTable {
    debug_assert_eq!(rho.nrows(), feature_names.len());
    debug_assert_eq!(b_feat.len(), feature_names.len());
    let h = rho.ncols();

    // (gene id, track id) -> row, one pass over the axis.
    let mut row_of: FxHashMap<(u32, u32), usize> = FxHashMap::default();
    for (r, (&t, &g)) in plan.row_track.iter().zip(&plan.row_gene).enumerate() {
        row_of.insert((g, t), r);
    }

    // Every modality this axis carries, alphabetical (deterministic output
    // order) and de-duplicated (a modality names two tracks, one per
    // channel).
    let mut modalities: Vec<&str> = plan.tracks.iter().map(|t| t.modality.as_ref()).collect();
    modalities.sort_unstable();
    modalities.dedup();

    let mut rows: Vec<Box<str>> = Vec::new();
    let mut modality_col: Vec<Box<str>> = Vec::new();
    let mut gene_col: Vec<Box<str>> = Vec::new();
    let mut delta_flat: Vec<f32> = Vec::new();
    let mut bias: Vec<f32> = Vec::new();

    for &modality in &modalities {
        // Fixed per the row grammar's own vocabulary; `None` only for a
        // modality outside {count, m6a, atoi, apa}, which `assign_tracks`
        // already rejects, so every track's modality resolves here.
        let Some((pos, neg)) = contrast_channels(modality) else {
            continue;
        };
        let (Some(pos_track), Some(neg_track)) =
            (plan.track_of(modality, pos), plan.track_of(modality, neg))
        else {
            // This axis never got both channels of this modality at all
            // (e.g. no `unspliced` row anywhere): nothing to contrast.
            continue;
        };

        let mut n_written = 0usize;
        let mut n_skipped = 0usize;
        for (gid, gene) in plan.gene_names.iter().enumerate() {
            let g = gid as u32;
            let (Some(&pr), Some(&nr)) = (row_of.get(&(g, pos_track)), row_of.get(&(g, neg_track)))
            else {
                n_skipped += 1;
                continue;
            };
            rows.push(format!("{gene}/{modality}").into_boxed_str());
            modality_col.push(Box::from(modality));
            gene_col.push(gene.clone());
            delta_flat.extend((0..h).map(|j| rho[(pr, j)] - rho[(nr, j)]));
            bias.push(b_feat[pr] - b_feat[nr]);
            n_written += 1;
        }
        log::info!(
            "gem contrast {modality}: {n_written} gene(s) written ({pos} − {neg}), {n_skipped} \
             skipped (missing one channel)"
        );
    }

    let delta = Mat::from_row_slice(rows.len(), h, &delta_flat);
    ContrastTable {
        rows,
        modality: modality_col,
        gene: gene_col,
        delta,
        bias,
    }
}

/// Write an already-built [`ContrastTable`] to `{prefix}.feature_contrast.parquet`
/// and `{prefix}.feature_contrast_bias.parquet`. Split from [`write_contrast`]
/// so tests can drive it directly, off a hand-built table, without a real fit.
fn write_contrast_table(prefix: &str, table: &ContrastTable) -> anyhow::Result<()> {
    let h_names = ge::embedding_col_names(table.delta.ncols());
    let h_cols: Vec<Vec<f32>> = (0..table.delta.ncols())
        .map(|j| table.delta.column(j).iter().copied().collect())
        .collect();
    let mut columns: Vec<(Box<str>, Column)> = vec![
        (Box::from("modality"), Column::Str(&table.modality)),
        (Box::from("gene"), Column::Str(&table.gene)),
    ];
    columns.extend(
        h_names
            .iter()
            .zip(&h_cols)
            .map(|(name, col)| (name.clone(), Column::F32(col))),
    );
    write_named_table(
        &format!("{prefix}.feature_contrast.parquet"),
        "feature",
        &table.rows,
        &columns,
    )?;
    write_named_table(
        &format!("{prefix}.feature_contrast_bias.parquet"),
        "feature",
        &table.rows,
        &[
            (Box::from("modality"), Column::Str(&table.modality)),
            (Box::from("gene"), Column::Str(&table.gene)),
            (Box::from("bias"), Column::F32(&table.bias)),
        ],
    )?;
    log::info!(
        "Wrote {prefix}.feature_contrast.parquet / feature_contrast_bias.parquet ({} row(s))",
        table.rows.len()
    );
    Ok(())
}

/// `senna gem`'s `after_fit` hook: build the contrast table from the
/// finished fit's RAW loading and write it. `plan` is the same
/// [`TrackPlan`] the driver trained with (`run_gem_embedding` clones it onto
/// `FitConfig.tracks` and keeps this borrow for the closure).
pub(crate) fn write_contrast(a: &FitArtifacts<'_>, plan: &TrackPlan) -> anyhow::Result<()> {
    let cpu = candle_core::Device::Cpu;
    let rho = Mat::from_tensor(&a.out.model.e_feat.to_device(&cpu)?)?;
    let b_feat = a.out.model.b_feat.to_device(&cpu)?.to_vec1::<f32>()?;
    let table = contrast_rows(plan, &a.unified.feature_names, &rho, &b_feat);
    write_contrast_table(a.prefix, &table)
}

#[cfg(test)]
#[path = "contrast/tests.rs"]
mod tests;
