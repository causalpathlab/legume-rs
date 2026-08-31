//! Query-side projection onto a frozen `senna svd` dictionary.
//!
//! An svd run writes no encoder checkpoint, so `predict` and `impute` cannot
//! reach it the way they reach the topic families. What it does write is the
//! dictionary `u` — orthonormal columns from the Nyström rsvd — and that is
//! enough: a new cell is put through the same per-column transform training
//! used ([`super::nystrom_preprocess_columns`]) and multiplied through `u`.
//!
//! **This is `u`, not the whitened basis `u·diag(1/s)` training projected
//! with.** The singular values are not persisted, so the scores here differ
//! from `{model}.latent.parquet` by a per-component scale. That is harmless
//! as long as everything being compared goes through THIS function — which is
//! why both consumers do, and why `impute` re-projects its reference rather
//! than reading the stored latent.

use crate::embed_common::*;
use crate::topic::eval::{
    build_gene_remap_with, ensure_gene_coverage, hide_features, QueryNameOpts,
};

/// The training-time normalization scale, replayed from the manifest's
/// recorded fit arguments through the typed [`crate::run_manifest::RunManifest::train_args_as`]
/// reader. Runs predating the manifest (or the `train_args` record) take the
/// fit's long-standing default.
///
/// A blob this senna cannot deserialize warns and falls back rather than
/// failing the run: `run_manifest`'s header reserves blob PARSING for
/// `update` precisely so that a senna which gained a flag can still OPEN a
/// run, and turning that into a hard error here would make this the second
/// command a renamed `SvdArgs` field can break — for one `f32` it does not
/// otherwise need.
pub(crate) fn column_sum_norm(model: &str) -> f32 {
    const DEFAULT: f32 = 1e4;
    let Ok((manifest, _)) = crate::run_manifest::load_for(model) else {
        info!("no manifest for {model}; using the default normalization scale");
        return DEFAULT;
    };
    if manifest.train_args.is_none() {
        info!("manifest for {model} predates train_args; using the default normalization scale");
        return DEFAULT;
    }
    match manifest.train_args_as::<crate::svd::SvdArgs>(model) {
        Ok(recorded) => recorded.column_sum_norm,
        Err(e) => {
            log::warn!(
                "{model}: cannot replay the recorded fit configuration ({e}); falling back to \
                 the default normalization scale. If the run was trained with a non-default \
                 --column-sum-norm, the projection here differs from its training space."
            );
            DEFAULT
        }
    }
}

/// Say so when an svd model carries a batch correction this path cannot replay.
///
/// Training divides each column by `mu_residual`, a per-PSEUDOBULK-group
/// residual, before fitting the dictionary. That matrix is never written to
/// disk — `{model}.delta.parquet` holds the per-BATCH `delta`, a different
/// object — and a new cell has no training pseudobulk membership to index it
/// by even if it were. So the correction cannot be reproduced from the run's
/// outputs, and cells are projected without it: consistent with each other,
/// but not with the space the dictionary was fitted in.
pub(crate) fn warn_if_batch_corrected(model: &str, batch_flag_given: bool) {
    if std::path::Path::new(&format!("{model}.delta.parquet")).is_file() {
        log::warn!(
            "{model}: this svd run was fitted with a batch correction, which the query-side \
             projection cannot replay (the per-pseudobulk residual it used is not persisted). \
             Cells are projected without it, so residual batch structure survives into the \
             scores."
        );
    }
    if batch_flag_given {
        log::warn!(
            "--batch-files has no effect on an svd model: the projection is per cell against \
             a frozen dictionary, with no per-batch term to estimate"
        );
    }
}

struct SvdProjParam<'a> {
    u_dk: &'a Mat,
    /// data row → dictionary row, by name.
    remap: &'a [Option<usize>],
    column_sum_norm: f32,
}

/// `[N, K]` projection of every column of `data` onto the dictionary:
/// remap rows by name onto the training axis, then the training run's own
/// per-chunk transform ([`super::nystrom_preprocess_columns`]), and
/// multiply through `u`.
///
/// `opts` carries the caller's query-axis rules and is NOT optional in
/// spirit: this used to build its own `QueryNameOpts::default()`, which made
/// the projection deaf to `--feature-name-kind` and — worse — to
/// `--ablate-features`, so a latent was fitted on the very genes it was then
/// scored against. Hiding is applied AFTER the coverage gate, because the
/// hidden features are withheld on purpose and counting them as missing would
/// refuse every ablated run.
pub(crate) fn projection_remap(
    data: &SparseIoVec,
    train_genes: &[Box<str>],
    opts: &QueryNameOpts,
    what: &str,
) -> anyhow::Result<crate::topic::eval::GeneRemap> {
    let data_genes = data.row_names()?;
    let mut remap = build_gene_remap_with(train_genes, &data_genes, opts);
    ensure_gene_coverage(&remap, opts.min_overlap, "--feature-name-kind")?;
    if let Some(hide) = opts.hide.as_deref() {
        hide_features(&mut remap, &data_genes, hide)?;
    }
    info!(
        "{what}: {} of the model's {} genes present across {} cells",
        remap.n_mapped,
        train_genes.len(),
        data.num_columns()
    );
    Ok(remap)
}

/// Project one already-read block onto the dictionary, `[K, n_block]`.
///
/// The transform lives HERE and only here, so a caller that has the block's
/// CSC in hand (and wants to do something else with it) reuses the same
/// arithmetic instead of streaming the data a second time.
pub(crate) fn project_block(
    csc_in: &nalgebra_sparse::CscMatrix<f32>,
    remap: &[Option<usize>],
    u_dk: &Mat,
    column_sum_norm: f32,
) -> Mat {
    let d_train = u_dk.nrows();
    // Remap onto the training gene axis, staying sparse for the reasons on
    // `nystrom_preprocess_columns`.
    let mut coo = nalgebra_sparse::CooMatrix::new(d_train, csc_in.ncols());
    for c_local in 0..csc_in.ncols() {
        let col = csc_in.col(c_local);
        for (&row_id, &v) in col.row_indices().iter().zip(col.values().iter()) {
            if let Some(t) = remap[row_id] {
                coo.push(t, c_local, v);
            }
        }
    }
    let mut x_dn = nalgebra_sparse::CscMatrix::from(&coo);
    super::nystrom_preprocess_columns(&mut x_dn, column_sum_norm, None);
    (x_dn.transpose() * u_dk).transpose()
}

pub(crate) fn project_onto_dictionary(
    data: &SparseIoVec,
    train_genes: &[Box<str>],
    u_dk: &Mat,
    column_sum_norm: f32,
    opts: &QueryNameOpts,
    block_size: Option<usize>,
    what: &str,
) -> anyhow::Result<Mat> {
    let remap = projection_remap(data, train_genes, opts, what)?;

    let n = data.num_columns();
    let k = u_dk.ncols();
    let param = SvdProjParam {
        u_dk,
        remap: &remap.new_to_train,
        column_sum_norm,
    };
    let mut proj_kn = Mat::zeros(k, n);
    data.visit_columns_by_block(&proj_visitor, &param, &mut proj_kn, block_size)?;
    Ok(proj_kn.transpose())
}

fn proj_visitor(
    job: (usize, usize),
    data: &SparseIoVec,
    param: &SvdProjParam,
    arc_proj_kn: Arc<Mutex<&mut Mat>>,
) -> anyhow::Result<()> {
    let (lb, ub) = job;
    let csc_in = data.read_columns_csc(lb..ub)?;
    let chunk = project_block(&csc_in, param.remap, param.u_dk, param.column_sum_norm);
    let mut proj_kn = arc_proj_kn.lock().expect("lock proj in svd projection");
    proj_kn.columns_range_mut(lb..ub).copy_from(&chunk);
    Ok(())
}

#[cfg(test)]
#[path = "project_tests.rs"]
mod tests;
