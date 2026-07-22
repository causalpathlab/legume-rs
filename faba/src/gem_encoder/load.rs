//! Data loading for `faba gem-encoder`.
//!
//! Reads gem-format gene matrices (`{gene}/count/{spliced|unspliced}` rows),
//! resolves per-cell batch identity the same way `faba gem` does, optionally
//! cuts to the top-N variable genes, and collapses cells into a multilevel
//! pseudobulk hierarchy that the masked model trains on.
//!
//! This mirrors senna's `load_and_collapse` (`senna/src/topic/common.rs`), which
//! faba cannot call — faba does not depend on senna. Everything it needs is
//! already a faba dependency: `auxiliary_data` for the read, `data_beans_alg`
//! for the projection and collapse.

use anyhow::Context;
use auxiliary_data::data_loading::{read_data_on_shared_rows, ReadSharedRowsArgs, SparseDataWithBatch};
use auxiliary_data::feature_names::FeatureNameKind;
use candle_util::data::indexed::GeneTrackMap;
use data_beans::sparse_io_vector::{ColumnAlignment, RowAlignment, SparseIoVec};
use data_beans_alg::random_projection::RandProjOps;
use data_beans_alg::collapse_data::{
    collapse_columns_multilevel_with_hierarchy, CollapsedOut, MultilevelParams,
};
use log::info;
use matrix_util::common_io::basename;
use nalgebra::DMatrix;
use rustc_hash::FxHashMap;

use crate::gem_encoder::args::GemEncoderArgs;

type Mat = DMatrix<f32>;

/// Split a gem feature row `{gene}/count/{spliced|unspliced}` into its gene key
/// and whether it is the **nascent** (unspliced) track. `None` when the row is
/// not a gene-level count row at all.
///
/// Goes through [`crate::feature_name::parse_feature_row`] rather than matching
/// on `/count/` directly, because a bare `rsplit_once` **cannot tell "spliced"
/// apart from "not a count row"** — both fall to the same branch. It used to,
/// and the consequence was silent: `BRCA2/m6a/methylated` became a mature gene
/// literally named `BRCA2/m6a/methylated`, and the sub-gene form
/// `{gene}/count/{site}/{channel}` became a mature row of the right gene. The
/// `n_nascent > 0` guard below catches a wholly spliced input, not a
/// contaminated one.
///
/// A `subunit` is rejected on purpose: this model is gene-resolution, so a
/// per-site or per-component row is not a thing it can pair across tracks.
#[must_use]
pub fn split_count_row(name: &str) -> Option<(&str, bool)> {
    use crate::feature_name::{parse_feature_row, COUNT, SPLICED, UNSPLICED};
    let row = parse_feature_row(name)?;
    if row.modality != COUNT || row.subunit.is_some() {
        return None;
    }
    match row.channel {
        SPLICED => Some((row.gene, false)),
        UNSPLICED => Some((row.gene, true)),
        _ => None,
    }
}

/// Intern each row's gene key to a dense gene id, returning the row→gene map,
/// the nascent flags, and the id-ordered gene names.
#[must_use]
pub fn build_gene_track_map(feature_names: &[Box<str>]) -> (GeneTrackMap, Vec<Box<str>>) {
    let mut ids: FxHashMap<Box<str>, u32> = FxHashMap::default();
    let mut row_to_gene = Vec::with_capacity(feature_names.len());
    let mut row_is_nascent = Vec::with_capacity(feature_names.len());
    let mut gene_names: Vec<Box<str>> = Vec::new();
    let mut skipped: Vec<&str> = Vec::new();
    for name in feature_names {
        let Some((gene, is_nascent)) = split_count_row(name) else {
            // Keep the row on the axis so every index still lines up with the
            // matrix, but give it its own gene so it can never be paired with a
            // real one. The warning below is the only place this surfaces.
            if skipped.len() < 3 {
                skipped.push(name);
            } else {
                skipped.push("");
            }
            let g = ids.len() as u32;
            ids.insert(name.clone(), g);
            gene_names.push(name.clone());
            row_to_gene.push(g);
            row_is_nascent.push(false);
            continue;
        };
        let gid = match ids.get(gene) {
            Some(&g) => g,
            None => {
                let g = ids.len() as u32;
                ids.insert(gene.into(), g);
                gene_names.push(gene.into());
                g
            }
        };
        row_to_gene.push(gid);
        row_is_nascent.push(is_nascent);
    }
    if !skipped.is_empty() {
        let shown: Vec<&str> = skipped.iter().copied().filter(|s| !s.is_empty()).collect();
        log::warn!(
            "{} of {} feature rows are not `{{gene}}/count/{{spliced|unspliced}}` and were \
             given their own single-track gene id — they cannot pair across tracks and will \
             contribute nothing but noise. Examples: {}. `faba gem-encoder` expects a \
             gene-level count matrix (`*_genes.zarr.zip`); a mixed-modality or per-site \
             matrix is not a supported input.",
            skipped.len(),
            feature_names.len(),
            shown.join(", ")
        );
    }
    let n_genes = gene_names.len();
    (
        GeneTrackMap {
            row_to_gene,
            row_is_nascent,
            n_genes,
        },
        gene_names,
    )
}

/// Everything the fit needs from the input files.
pub struct PreparedData {
    pub data_vec: SparseIoVec,
    /// Pseudobulk levels, coarse-first … finest-last.
    pub collapsed_levels: Vec<CollapsedOut>,
    /// Row → (gene, track) map for the (possibly HVG-subset) feature axis.
    pub map: GeneTrackMap,
    /// Gene names in gene-id order (length `map.n_genes`).
    pub gene_names: Vec<Box<str>>,
    pub cell_names: Vec<Box<str>>,
    /// Raw per-ROW mean over every cell — the splice-ratio QC compares the model
    /// against this rather than against another smoothed summary.
    pub row_mean: Vec<f32>,
    /// Per-GENE shortlist weight (NB-Fisher, pooled across a gene's two tracks),
    /// used to rank the encoder's top-K.
    pub gene_weights: Vec<f32>,
}


/// Read the gene matrices, pick HVGs, project, and collapse into pseudobulks.
pub fn load_and_collapse(args: &GemEncoderArgs) -> anyhow::Result<PreparedData> {
    let files = args.genes()?;

    // Under Union alignment cells merge by raw barcode, so tag each file's
    // barcodes with its sample id to keep distinct samples apart.
    //
    // Uses `faba gem`'s own helper AND its auto-strip rule, so a gem run and a
    // gem-encoder run over the same files agree on cell identity. They did not
    // before: gem falls back to the longest common `_`-suffix when
    // `--genes-sample-strip` is empty, so it tagged `@rep1_wt` where this tagged
    // `@rep1_wt_genes`, and barcodes from the two tools would not join.
    let do_tag = args.batch_files.is_none() && files.len() > 1;
    let per_file_barcode_suffix = if do_tag {
        let strip: Box<str> = if args.genes_sample_strip.is_empty() {
            let bn: Vec<Box<str>> = files
                .iter()
                .map(|f| basename(f))
                .collect::<anyhow::Result<_>>()?;
            let s = crate::gem::sample_id::longest_common_underscore_suffix(&bn);
            if !s.is_empty() {
                info!("auto-strip: --genes-sample-strip = {:?}", s.as_ref());
            }
            s
        } else {
            args.genes_sample_strip.clone()
        };
        let ids = files
            .iter()
            .map(|f| crate::gem::sample_id::file_sample_id(f, &strip).map(Some))
            .collect::<anyhow::Result<Vec<_>>>()?;
        info!("tagging barcodes with per-file @sample id for batch identity");
        Some(ids)
    } else {
        None
    };

    let SparseDataWithBatch {
        data: mut data_vec,
        batch: mut batch_membership,
        ..
    } = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: files.to_vec(),
        batch_files: args.batch_files.clone(),
        preload: args.runtime.preload_data,
        // gem rows are `{gene}/count/{track}`; the per-gene pairing depends on
        // that exact path, so canonicalization must not split on `_`.
        feature_kind: Some(FeatureNameKind::Exact),
        row_alignment: RowAlignment::Union,
        column_alignment: ColumnAlignment::Union,
        qc: None,
        qc_block_size: None,
        qc_report_out: None,
        per_file_feature_suffix: None,
        per_file_barcode_suffix,
    })
    .context("reading gene matrices")?;

    if args.ignore_batch {
        info!("--ignore-batch: collapsing all cells to a single batch");
        let all: Box<str> = "all".into();
        for b in batch_membership.iter_mut() {
            *b = all.clone();
        }
    }

    info!(
        "loaded {} feature rows × {} cells",
        data_vec.num_rows(),
        data_vec.num_columns()
    );

    let feature_names = data_vec.row_names().context("row names")?;
    let (map, gene_names) = build_gene_track_map(&feature_names);
    let n_nascent = map.row_is_nascent.iter().filter(|&&b| b).count();
    anyhow::ensure!(
        n_nascent > 0,
        "no `{{gene}}/count/unspliced` rows found among {} feature rows (e.g. `{}`). \
         `faba gem-encoder` models the nascent→mature transition, so a spliced-only \
         input has nothing for it to learn — use `faba gem` instead.",
        feature_names.len(),
        feature_names.first().map_or("", |s| s.as_ref()),
    );
    info!(
        "gene axis: {} genes from {} rows ({} nascent, {} mature)",
        map.n_genes,
        feature_names.len(),
        n_nascent,
        feature_names.len() - n_nascent,
    );

    info!("computing per-row statistics (HVG ranking + shortlist weights)");
    let row_stats =
        data_beans_alg::sparse_streaming::streaming_sparse_running_stats(&data_vec, None, "row stats")
            .context("row statistics")?;

    // Random projection sketch drives the pseudobulk partition. HVG enters ONLY
    // here, as a 0/1 row weight — features with weight 0 are excluded from the
    // projection geometry and from nothing else.
    let hvg_w = {
        use matrix_util::traits::RunningStatOps;
        hvg_row_weights(&row_stats.mean(), &row_stats.variance(), &map, args.n_hvg)
    };
    // Derived here so the private stats type stays inside this module, and so
    // the single streaming pass serves both consumers.
    let row_weights = data_beans_alg::gene_weighting::fisher_weights_from_stats(
        &row_stats,
        data_vec.num_columns(),
    );
    let gene_weights = per_gene_weights(&row_weights, &map);
    let row_mean: Vec<f32> = {
        use matrix_util::traits::RunningStatOps;
        row_stats.mean().to_vec()
    };
    let proj_kn = data_vec
        .project_columns_weighted(
            args.proj_dim.max(args.n_latent),
            None,
            Some(&batch_membership),
            &hvg_w,
        )
        .context("random projection")?
        .proj;
    info!("projection sketch: {} × {}", proj_kn.nrows(), proj_kn.ncols());

    info!("multi-level collapsing into pseudobulk samples ...");
    let ml = MultilevelParams {
        knn_pb_samples: args.knn_pb,
        num_levels: args.num_levels,
        sort_dim: args.sort_dim,
        num_opt_iter: args.num_opt_iter,
        refine: Some(data_beans_alg::refine_multilevel::RefineParams::default()),
        // Only posterior means are read downstream (training matrices, per-gene
        // track means, the batch residual), so skip the sd / log-mean / log-sd
        // planes — that is the bulk of collapse-stage memory at high pb counts.
        // Same choice `faba gem` makes.
        output_calibration: matrix_param::traits::CalibrateTarget::MeanOnly,
    };
    let out = collapse_columns_multilevel_with_hierarchy(
        &mut data_vec,
        &proj_kn,
        &batch_membership,
        &ml,
    )
    .context("multilevel collapse")?;

    // data-beans-alg returns levels finest-first; the trainer wants
    // coarse-first so the shared encoder sees broad structure before detail.
    let mut collapsed_levels = out.levels;
    collapsed_levels.reverse();
    for (i, lvl) in collapsed_levels.iter().enumerate() {
        info!(
            "  level {i}: {} pseudobulk samples",
            matrix_param::traits::Inference::posterior_mean(&lvl.mu_observed).ncols()
        );
    }

    let cell_names = data_vec.column_names().context("column names")?;

    Ok(PreparedData {
        data_vec,
        collapsed_levels,
        map,
        gene_names,
        cell_names,
        row_mean,
        gene_weights,
    })
}

/// Per-ROW projection weight: `1.0` on the rows of the top-N most variable
/// genes, `0.0` elsewhere. All ones when `n_hvg` is 0 or covers every gene.
///
/// # HVG restricts the SKETCH, not the model
///
/// The weight is handed to `project_columns_weighted`, whose zero-weight rows
/// are "excluded from the projection geometry" — and from nothing else. The
/// pseudobulk partition is then built on the variable genes, where the
/// structure is, while the model still trains on, and still emits, **every**
/// gene.
///
/// This used to mask the rows off `data_vec` outright, which restricted the
/// whole run: the collapse, the encoder's vocabulary, the decoder's simplex,
/// and every output. That is the wrong scope for a partitioning heuristic. Its
/// worst consequence was silent and downstream — a marker gene that missed the
/// cut was not down-weighted in `dictionary.parquet`, it was **absent**, so
/// `faba annotate` scored that cell type on whatever fraction of its panel
/// happened to survive and still returned a confident-looking call.
///
/// Ranking pools a gene's two tracks onto one entry before scoring: ranking
/// rows instead returns well under N genes, because a gene's two correlated
/// tracks both rank high and then collapse to one gene on dedup. The pooled
/// mean is exact (`E[s+u] = E[s] + E[u]`); the pooled variance sums the tracks,
/// a lower bound that ignores cross-track covariance — fine for ranking.
#[must_use]
fn hvg_row_weights(means: &[f32], vars: &[f32], map: &GeneTrackMap, n_hvg: usize) -> Vec<f32> {
    use data_beans_alg::hvg::select_hvg_by_stats;

    let n_rows = map.row_to_gene.len();
    let n_genes = map.n_genes;
    if n_hvg == 0 || n_hvg >= n_genes {
        info!("HVG: --n-hvg {n_hvg} covers all {n_genes} genes; projection uses every row");
        return vec![1.0; n_rows];
    }

    let mut gmean = vec![0f32; n_genes];
    let mut gvar = vec![0f32; n_genes];
    for (r, (&m, &v)) in means.iter().zip(vars.iter()).enumerate() {
        gmean[map.row_to_gene[r] as usize] += m;
        gvar[map.row_to_gene[r] as usize] += v;
    }

    let keep: rustc_hash::FxHashSet<usize> = select_hvg_by_stats(&gmean, &gvar, n_hvg)
        .into_iter()
        .collect();
    let w: Vec<f32> = map
        .row_to_gene
        .iter()
        .map(|&g| f32::from(u8::from(keep.contains(&(g as usize)))))
        .collect();
    info!(
        "HVG (--n-hvg {n_hvg}): {} of {n_genes} genes weight the projection \
         ({} of {n_rows} rows); the model still sees all {n_genes}",
        keep.len(),
        w.iter().filter(|&&x| x > 0.0).count(),
    );
    w
}

/// Per-gene mean rate for one track, from a `[D_rows, n_pb]` pseudobulk
/// posterior mean. Genes with no row on that track get `0`, which the encoder's
/// divisive Anscombe floors — it never divides by zero.
///
/// NOTE the orientation: `mu_dp` is `[D_rows, N_pb]` — **un-transposed** — so
/// `.row(r)` is feature row `r` across pseudobulks and the `/n_pb` averages over
/// them. The caller feeds the same underlying matrix to `GemIndexedArgs::input`
/// in its TRANSPOSED `[N_pb, D_rows]` form. Both are correct as written, but two
/// orientations of one matrix a few lines apart is easy to break; check which
/// one you have before touching either.
#[must_use]
pub fn per_gene_track_mean(mu_dp: &Mat, map: &GeneTrackMap, nascent: bool) -> Vec<f32> {
    let n_pb = mu_dp.ncols().max(1) as f32;
    let mut out = vec![0f32; map.n_genes];
    for (r, (&g, &is_n)) in map
        .row_to_gene
        .iter()
        .zip(map.row_is_nascent.iter())
        .enumerate()
    {
        if is_n == nascent {
            out[g as usize] += mu_dp.row(r).iter().sum::<f32>() / n_pb;
        }
    }
    out
}

/// Pool per-**row** shortlist weights onto genes by taking the max across a
/// gene's two tracks.
///
/// Max rather than mean: the weight scores how informative a gene is, and a
/// gene whose mature track is highly informative should be selected even when
/// its nascent track is shallow and uninformative — which is the common case,
/// since nascent counts are the sparse side.
#[must_use]
pub fn per_gene_weights(row_weights: &[f32], map: &GeneTrackMap) -> Vec<f32> {
    let mut out = vec![0f32; map.n_genes];
    for (r, &g) in map.row_to_gene.iter().enumerate() {
        let w = &mut out[g as usize];
        *w = w.max(row_weights[r]);
    }
    out
}

#[cfg(test)]
#[path = "load_tests.rs"]
mod tests;
