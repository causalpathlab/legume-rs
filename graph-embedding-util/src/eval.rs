//! Output writers shared by all callers.
//!
//! The two embedding tables are written under caller-chosen stems (see
//! [`EmbeddingFileNames`]); the bias tables are fixed:
//! - cell × H, col prefix `h` — `latent` by default, `cell_embedding` for gem
//! - feature × H, col prefix `h` — `dictionary` by default, `feature_embedding` for gem
//! - `{out}.feature_bias.parquet` (per-feature bias).
//! - `{out}.cell_bias.parquet` (per-cell bias `b_cell`, the depth sink).

use crate::model::JointEmbedModel;
use candle_util::candle_core::Tensor;
use log::info;
use matrix_util::traits::IoOps;

/// File stems for the two embedding tables.
///
/// [`Default`] is the legacy senna convention (`latent` / `dictionary`), kept
/// so pre-existing manifests still describe their files correctly. senna's
/// embedding commands now use [`EmbeddingFileNames::SENNA_EMBEDDING`], which
/// moves the cell table to `cell_embedding` and leaves `latent` to mean log θ.
/// `senna gem` uses [`EmbeddingFileNames::EXPLICIT`], and its own downstream
/// (`senna lineage` / `senna annotate-gem`) reads those names. The conventions do
/// not mix: a reader expects one or the other.
#[derive(Clone, Copy, Debug)]
pub struct EmbeddingFileNames {
    /// Stem for the cell × H table.
    pub cell: &'static str,
    /// Stem for the feature × H table.
    pub feature: &'static str,
}

impl Default for EmbeddingFileNames {
    fn default() -> Self {
        Self {
            cell: "latent",
            feature: "dictionary",
        }
    }
}

impl EmbeddingFileNames {
    /// The explicit spelling used by `senna gem`: say what the table *is* rather
    /// than what role it plays in a topic model.
    pub const EXPLICIT: Self = Self {
        cell: "cell_embedding",
        feature: "feature_embedding",
    };

    /// The convention used by senna's embedding commands (`bge`, `fne`).
    ///
    /// Cell side is `cell_embedding`, never `latent`: `latent` is reserved for
    /// log θ, so a run that also resolves topics can emit both without either
    /// table's meaning depending on which flags were passed.
    ///
    /// The feature side deliberately stays `dictionary` rather than following
    /// [`Self::EXPLICIT`] — in senna, `{out}.feature_embedding.parquet` is
    /// owned by the SIMBA co-embed (features re-placed onto the cell manifold),
    /// and writing the raw off-manifold ρ there would hand
    /// `annotate-by-projection` an ill-posed nearest-centroid problem.
    pub const SENNA_EMBEDDING: Self = Self {
        cell: "cell_embedding",
        feature: "dictionary",
    };
}

pub struct OutputContext<'a> {
    /// Names for the rows of `model.e_feat` / `model.b_feat`. The
    /// feature-embedding parquet is keyed directly on these — gbe trains
    /// `E_feat` at fine gene resolution, so no replication is needed.
    pub feature_names: &'a [Box<str>],
    pub barcodes: &'a [Box<str>],
    /// Optional cell keep-mask (QC): emit only these cell rows of the
    /// per-cell embedding, with matching barcodes. `None` = emit every cell.
    pub cell_keep_idx: Option<&'a [usize]>,
}

/// Write the outputs under senna's `latent` / `dictionary` convention.
pub fn save_outputs(
    model: &JointEmbedModel,
    ctx: &OutputContext,
    out_prefix: &str,
) -> anyhow::Result<()> {
    save_outputs_named(model, ctx, out_prefix, EmbeddingFileNames::default())
}

/// Write the outputs, choosing the stems for the two embedding tables.
pub fn save_outputs_named(
    model: &JointEmbedModel,
    ctx: &OutputContext,
    out_prefix: &str,
    names: EmbeddingFileNames,
) -> anyhow::Result<()> {
    let latent_path = format!("{out_prefix}.{}.parquet", names.cell);
    let cell_bias_path = format!("{out_prefix}.cell_bias.parquet");
    match ctx.cell_keep_idx {
        // Drop QC-failed cells from the per-cell outputs (the row indices and
        // barcodes are subset by the same mask, so they cannot desync).
        Some(keep) => {
            let idx: Vec<u32> = keep.iter().map(|&i| i as u32).collect();
            let idx_t = Tensor::from_vec(idx, keep.len(), model.e_cell.device())?;
            let kept = model.e_cell.index_select(&idx_t, 0)?;
            let kept_b = model.b_cell.index_select(&idx_t, 0)?;
            let names: Vec<Box<str>> = keep.iter().map(|&i| ctx.barcodes[i].clone()).collect();
            save_embedding(&latent_path, &kept, &names, "cell")?;
            save_bias(&cell_bias_path, &kept_b, &names, "cell")?;
        }
        None => {
            save_embedding(&latent_path, &model.e_cell, ctx.barcodes, "cell")?;
            save_bias(&cell_bias_path, &model.b_cell, ctx.barcodes, "cell")?;
        }
    }
    save_embedding(
        &format!("{out_prefix}.{}.parquet", names.feature),
        &model.e_feat,
        ctx.feature_names,
        "feature",
    )?;
    save_bias(
        &format!("{out_prefix}.feature_bias.parquet"),
        &model.b_feat,
        ctx.feature_names,
        "feature",
    )?;
    // The LEARNED gate's inclusion probabilities `σ(S)`, when that gate is what
    // selected. Rows align with the feature dictionary. Every entry is an INDEPENDENT
    // probability in (0,1) for that (gene, dim): neither rows nor columns sum to
    // anything in particular, and a gene may load several dims. This is the variational
    // estimate of the same estimand `--posterior` samples into `feature_pip` /
    // `beta_pip`.
    //
    // Skipped for an ungated model, AND under `--posterior`: there the selection pass
    // installed a `pip`, training stopped reading these logits, and `feature_pip` is
    // already the table to read. Emitting both would ship two files with identical
    // bytes and a comment inviting the reader to compare them.
    //
    // The predecessor was a per-dim softmax over genes, where a COLUMN carried one
    // unit of mass and 1.0 meant "neutral". Anything reading these tables against that
    // convention will misread them — the neutral point is gone, and the range is now
    // (0,1) rather than (0, D).
    if let Some(selection) = model.feature_selection()? {
        let sel_path = format!("{out_prefix}.feature_selection.parquet");
        save_embedding(&sel_path, &selection, ctx.feature_names, "feature")?;
        info!("Per-gene feature inclusion probability σ(S) → {sel_path}");
    }
    // Per-gene VELOCITY inclusion `σ(s_delta)` — the independent δ-gate readout
    // (motion driver genes); `Some` only for a factored model with velocity.
    if let Some(velocity_sel) = model.velocity_selection()? {
        let vsel_path = format!("{out_prefix}.velocity_selection.parquet");
        save_embedding(&vsel_path, &velocity_sel, ctx.feature_names, "feature")?;
        info!("Per-gene velocity inclusion probability σ(s_δ) → {vsel_path}");
    }
    Ok(())
}

/// Canonical embedding-coordinate column names `h0..h{H-1}` — the single
/// convention shared by every embedding writer (senna `save_embedding`,
/// `senna gem`, `pinto cage`, `annotate-by-projection`). One source of truth so
/// the `{out}.*_embedding.parquet` column schema never drifts (`h` vs `dim_` vs
/// `e`). Embedding columns are always read positionally, so the name is purely
/// for human/schema legibility — but it must be consistent across tools.
pub fn embedding_col_names(h: usize) -> Vec<Box<str>> {
    (0..h).map(|i| format!("h{i}").into_boxed_str()).collect()
}

/// Write an embedding table `[N, H]` to parquet with `h0..h{H-1}` columns and
/// `row_names` on the `row_axis`. Public so callers can emit auxiliary
/// embeddings (e.g. bge's pre-cell-QC "before" cell embedding) in the same
/// layout as the standard latent output.
pub fn save_embedding(
    path: &str,
    table: &Tensor,
    row_names: &[Box<str>],
    row_axis: &str,
) -> anyhow::Result<()> {
    let h = table.dim(1)?;
    let cols = embedding_col_names(h);
    table.to_parquet_with_names(path, (Some(row_names), Some(row_axis)), Some(&cols))?;
    Ok(())
}

/// Write a 1-D bias tensor `[N]` as an `[N, 1]` `bias` column parquet.
/// Public so callers that emit a custom output layout (e.g. `senna bge
/// --resolve-etm`) reuse the same bias format instead of re-implementing it.
pub fn save_bias(
    path: &str,
    bias: &Tensor,
    row_names: &[Box<str>],
    row_axis: &str,
) -> anyhow::Result<()> {
    let bias_2d = bias.unsqueeze(1)?;
    let col = vec![Box::<str>::from("bias")];
    bias_2d.to_parquet_with_names(path, (Some(row_names), Some(row_axis)), Some(&col))?;
    Ok(())
}

/// SIMBA-style feature co-embedding, shared by `senna bge` and `senna rest`:
/// re-embed every feature onto the cell manifold (feature = softmax-over-cells
/// weighted average of the cell embeddings) via [`crate::feature_coembedding`]
/// and write it as `{out}.feature_embedding.parquet`, *overriding* the raw
/// learned feature embedding (the raw embedding is the disjoint off-manifold
/// cloud — nothing downstream consumes it, so it is not written). `e_cell` is
/// the reference cell embedding (left unchanged, SIMBA's anchor) and `e_feat`
/// the raw feature embedding (both `[*, H]` on the same device); `target_eff`
/// is the eff-cells temperature target from [`crate::cell_clusters`].
///
/// `confidence`, when given, is a per-feature weight in `[0, 1]` — the phase-1
/// selection posterior's `max_h PIP[g, h]` — applied AFTER the softmax so the
/// attention and its calibrated temperature are untouched. See
/// [`crate::postprocess::shrink_by_confidence`]. Its spread is logged either way:
/// a saturated posterior makes the shrinkage a constant rescale, and that has to
/// be visible rather than silently doing nothing.
pub fn write_feature_coembedding(
    out_prefix: &str,
    e_cell: &Tensor,
    e_feat: &Tensor,
    feature_names: &[Box<str>],
    target_eff: f64,
    confidence: Option<&[f32]>,
) -> anyhow::Result<()> {
    let (coembed, t) = crate::feature_coembedding(e_cell, e_feat, target_eff)?;
    let coembed = match confidence {
        None => coembed,
        Some(w) => {
            let s = crate::postprocess::confidence_spread(w);
            if s.is_degenerate() {
                log::warn!(
                    "co-embed confidence shrinkage is inert: per-feature weight spans \
                     [{:.3}, {:.3}] over {} features, so it rescales every feature by \
                     ~the same constant and moves nothing relative to anything else. \
                     A selection posterior estimated on many more dims than the embedding \
                     actually uses saturates this way — check the effective rank.",
                    s.min,
                    s.max,
                    s.n
                );
            } else {
                info!(
                    "co-embed confidence shrinkage: weight min {:.3}, p05 {:.3}, median {:.3}, \
                     max {:.3}; {} of {} features pulled at least halfway to the origin",
                    s.min, s.p05, s.median, s.max, s.n_below_half, s.n
                );
            }
            crate::postprocess::shrink_by_confidence(&coembed, w)?
        }
    };
    save_embedding(
        &format!("{out_prefix}.feature_embedding.parquet"),
        &coembed,
        feature_names,
        "feature",
    )?;
    info!("Feature co-embedding (SIMBA-style, T={t:.4}) → {out_prefix}.feature_embedding.parquet");
    Ok(())
}

/// Write both gates' tables for the β-sharing (splice) model, keyed by GENE so
/// they join directly against `{out}.beta_feature_embedding.parquet`.
///
/// Emits `{out}.{beta,delta}_{pip,posterior_mean}.parquet`. Genes whose `δ` is
/// not identified — no spliced counts, so only `β + δ` is pinned — are written as
/// `NaN` in the δ tables rather than as the prior draw the sampler happened to
/// take, and their count is logged. A number there would be indistinguishable
/// from a measurement.
pub fn write_splice_posterior_tables(
    out_prefix: &str,
    res: &crate::posterior::pb_gibbs::SpliceGibbsResult,
    gene_names: &[Box<str>],
) -> anyhow::Result<()> {
    let g = gene_names.len();
    anyhow::ensure!(
        res.beta_pip.len() == g * res.h && res.delta_pip.len() == g * res.h,
        "splice posterior tables: {} β / {} δ values for {g} genes × {} dims",
        res.beta_pip.len(),
        res.delta_pip.len(),
        res.h
    );
    let n_unid = res.delta_identified.iter().filter(|&&x| !x).count();
    if n_unid > 0 {
        info!(
            "δ posterior: {n_unid} of {g} gene(s) have no spliced counts, so only β+δ is \
             identified for them — their δ rows are written as NaN, not as a prior draw"
        );
    }
    let mask_unidentified = |v: &[f32]| -> Vec<f32> {
        v.chunks_exact(res.h)
            .zip(&res.delta_identified)
            .flat_map(|(row, &ok)| {
                row.iter()
                    .map(move |&x| if ok { x } else { f32::NAN })
                    .collect::<Vec<f32>>()
            })
            .collect()
    };
    let dev = candle_util::candle_core::Device::Cpu;
    let delta_pip = mask_unidentified(&res.delta_pip);
    let delta_mean = mask_unidentified(&res.delta_mean);
    for (values, suffix) in [
        (&res.beta_pip, "beta_pip"),
        (&res.beta_mean, "beta_posterior_mean"),
        (&delta_pip, "delta_pip"),
        (&delta_mean, "delta_posterior_mean"),
    ] {
        let t = Tensor::from_slice(values.as_slice(), (g, res.h), &dev)?;
        let path = format!("{out_prefix}.{suffix}.parquet");
        save_embedding(&path, &t, gene_names, "feature")?;
        info!("wrote {path}");
    }
    Ok(())
}

/// Write the phase-1 posterior's two feature-side tables.
///
/// * `{out}.feature_pip.parquet` — `[D × H]` per-`(feature, dim)` inclusion
///   probability. Read it per ENTRY: inclusion is independent per dim, so a row
///   does not sum to 1 and may sum well past it.
/// * `{out}.feature_posterior_mean.parquet` — `[D × H]` `E[z · β]`, the
///   **effective** loading, with excluded draws contributing zero. This is what
///   the model uses, and it is also what was written back into the fit; it is
///   emitted so a reader can join loadings against their own uncertainty.
///
/// Both are on the full feature axis in model order, so they join positionally
/// and by name against `{out}.feature_embedding.parquet`.
pub fn write_pb_posterior_tables(
    out_prefix: &str,
    res: &crate::posterior::pb_gibbs::PbGibbsResult,
    feature_names: &[Box<str>],
) -> anyhow::Result<()> {
    let d = feature_names.len();
    anyhow::ensure!(
        res.pip.len() == d * res.h,
        "posterior tables: {} PIP values for {d} features × {} dims",
        res.pip.len(),
        res.h
    );
    let dev = candle_util::candle_core::Device::Cpu;
    for (values, suffix, what) in [
        (&res.pip, "feature_pip", "per-dim inclusion probability"),
        (
            &res.mean_beta,
            "feature_posterior_mean",
            "posterior-mean effective loading E[z·β]",
        ),
    ] {
        let t = Tensor::from_slice(values.as_slice(), (d, res.h), &dev)?;
        let path = format!("{out_prefix}.{suffix}.parquet");
        save_embedding(&path, &t, feature_names, "feature")?;
        info!("wrote {path} ({what})");
    }
    Ok(())
}
