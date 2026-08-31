//! `senna impute` — post-hoc kNN imputation against a reference dataset.
//!
//! Pipeline:
//! 1. Place the new sparse data in the model's latent space — through
//!    `senna predict` for the encoder/projection families, or a direct
//!    dictionary projection for `svd` runs.
//! 2. Place the reference cells in the SAME space: the training run's
//!    latent (topic families), its cell embedding (`bge`), or the same
//!    dictionary projection applied to the reference data (`svd`).
//! 3. Build an HNSW kNN index over the reference and, for each new cell,
//!    find the K nearest reference cells.
//! 4. Convert kNN L2 distances to softmax weights with temperature τ
//!    (smaller τ ⇒ sharper, fewer-neighbor-effective).
//! 5. For each new cell, weighted-average the reference cells'
//!    full-feature counts to produce the imputed full-feature row.
//! 6. Write `{out}.imputed.parquet` (`N_new` × `n_ref_features`).
//!
//! Steps 3–5 are model-agnostic and live in
//! [`data_beans_alg::retrieval_impute`]; this module owns step 1–2's
//! per-family latent semantics:
//!
//! - topic family (`topic`, `masked-topic`, `masked-sbp`, `joint-topic`):
//!   `log θ` rows, mapped to the simplex by softmax.
//! - `vae` / `masked-vae`: a Gaussian `z`; the same softmax mapping is a
//!   deliberate choice — it is exactly `softmax(z)`, the proportions the
//!   head would read out, and for `log θ` rows it is a no-op.
//! - `bge`: an H-dimensional embedding where magnitude carries depth, not
//!   identity — rows are L2-normalized so L2 ≈ cosine.
//! - `svd`: the saved latent's whitening scale is not persisted, so BOTH
//!   sides are re-projected through the frozen `dictionary.parquet`
//!   with the training-time transform, then row-normalized.
//!
//! Information-theoretic note (per the rag-augmentation memory): the
//! imputed expression is a function of the query latent through the
//! reference's distribution-preserving retrieval. It is the genuine RAG
//! payoff (residual full-rank covariance β can't carry), not a
//! deterministic β·θ readout.

use crate::embed_common::*;
use crate::predict::{predict_model, PredictArgs};
use crate::run_manifest::{self, RunKind};
use crate::topic::eval::{build_gene_remap_with, ensure_gene_coverage, QueryNameOpts};
use auxiliary_data::data_loading::{read_data_on_shared_rows, ReadSharedRowsArgs};
use clap::Args;
use data_beans::sparse_data_visitors::VisitColumnsOps;
use data_beans::sparse_io_vector::SparseIoVec;
use data_beans_alg::retrieval_impute::{retrieval_impute, RetrievalImputeConfig};
use log::info;
use matrix_util::traits::IoOps;

#[derive(Args, Debug)]
pub struct ImputeArgs {
    #[arg(
        required = true,
        value_delimiter = ',',
        help = "New (typically sparse-panel) data files (.zarr or .h5) to impute"
    )]
    pub data_files: Vec<Box<str>>,

    #[arg(
        long,
        required = true,
        help = "Trained model prefix (topic family, masked family, vae, bge, or svd run)"
    )]
    pub model: Box<str>,

    #[arg(short, long, required = true, help = "Output file prefix")]
    pub out: Box<str>,

    #[arg(
        long,
        help = "Reference run prefix or manifest (defaults to --model's own run)",
        long_help = "Run prefix (or `.senna.json` path) whose training cells are the reference.\n\
                     Its manifest supplies the reference latent and data files,\n\
                     so neither --reference-latent nor --reference-data is needed.\n\
                     Defaults to the --model run itself: the model's training cells\n\
                     are the reference. Explicit --reference-* flags override\n\
                     the manifest's values piecewise."
    )]
    pub reference: Option<Box<str>>,

    #[arg(
        long,
        help = "Reference latent parquet (overrides the manifest's latent)",
        long_help = "Reference latent parquet, e.g. `{train_out}.latent.parquet`.\n\
                     Must live in the model's matching space: log θ for the topic\n\
                     families, the H-space cell embedding for a bge model.\n\
                     Ignored for svd models — their matching latent is re-projected\n\
                     from the reference data because the training whitening scale\n\
                     is not persisted."
    )]
    pub reference_latent: Option<Box<str>>,

    #[arg(
        long,
        value_delimiter = ',',
        help = "Reference data files (overrides the manifest's data files)"
    )]
    pub reference_data: Option<Vec<Box<str>>>,

    #[arg(
        long,
        value_delimiter = ',',
        help = "Per-file batch labels for the reference data"
    )]
    pub reference_batch_files: Option<Vec<Box<str>>>,

    #[arg(
        short,
        long,
        value_delimiter = ',',
        help = "Per-file batch labels for --data-files"
    )]
    pub batch_files: Option<Vec<Box<str>>>,

    #[arg(
        long,
        default_value_t = 25,
        help = "Number of reference nearest neighbours to pool per new cell"
    )]
    pub knn: usize,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Softmax temperature on kNN distances (lower = sharper neighbour weights)"
    )]
    pub knn_temperature: f32,

    #[arg(long, default_value_t = 500, help = "Predict / read minibatch size")]
    pub minibatch_size: usize,

    #[arg(
        long,
        help = "Cells per delta-estimation block (auto by default)",
        hide = true
    )]
    pub block_size: Option<usize>,

    #[arg(
        long,
        help = "Load all columns into memory before evaluation",
        hide = true
    )]
    pub preload_data: bool,

    #[arg(short, long, help = "Verbose logging")]
    pub verbose: bool,
}

/// The reference side, after the manifest/flag negotiation: where the
/// latent lives (`None` for svd, which re-projects), and which backends
/// hold the reference counts.
#[derive(Debug)]
struct ReferenceSpec {
    latent: Option<Box<str>>,
    data_files: Vec<Box<str>>,
    batch_files: Option<Vec<Box<str>>>,
}

/// Resolve the reference from `--reference` / the model's own manifest,
/// with the explicit `--reference-*` flags winning piecewise.
///
/// The manifest is consulted unless every needed piece was given
/// explicitly — so the legacy invocation (`--reference-latent` +
/// `--reference-data`, no manifest anywhere) keeps working unchanged.
fn resolve_reference(args: &ImputeArgs, model_kind: RunKind) -> anyhow::Result<ReferenceSpec> {
    let needs_latent = !matches!(model_kind, RunKind::Svd | RunKind::JointSvd);
    if args.reference_latent.is_some() && !needs_latent {
        log::warn!(
            "--reference-latent is ignored for a {model_kind} model: its matching latent \
             is re-projected from the reference data (the training whitening scale is \
             not persisted)"
        );
    }

    let latent_settled = !needs_latent || args.reference_latent.is_some();
    let data_settled = args.reference_data.is_some();
    if args.reference.is_none() && latent_settled && data_settled {
        return Ok(ReferenceSpec {
            latent: args.reference_latent.clone(),
            data_files: args.reference_data.clone().expect("data_settled"),
            batch_files: args.reference_batch_files.clone(),
        });
    }

    let prefix = args.reference.as_deref().unwrap_or(args.model.as_ref());
    let (manifest, dir) = run_manifest::load_for(prefix).map_err(|e| {
        anyhow::anyhow!(
            "resolving the reference from `{prefix}`: {e}\n\
             Pass --reference-latent and --reference-data explicitly if the \
             reference run has no manifest."
        )
    })?;

    // The matching space comes from the MODEL; a reference whose cells live
    // in a different space cannot be matched against, however plausible the
    // dimensions look.
    anyhow::ensure!(
        manifest.kind.cell_space() == model_kind.cell_space(),
        "reference run `{prefix}` is a `{}` run whose cells live in a different \
         space than the `{model_kind}` model's; match against a reference of the \
         same family",
        manifest.kind,
    );
    if manifest.kind != model_kind {
        log::warn!(
            "reference run `{prefix}` is `{}` while the model is `{model_kind}`; \
             proceeding because both latents share a cell space, but the K axes \
             must genuinely correspond",
            manifest.kind,
        );
    }

    let to_box = |s: &str| -> Box<str> {
        run_manifest::resolve(&dir, s)
            .to_string_lossy()
            .into_owned()
            .into()
    };

    let latent = if !needs_latent {
        None
    } else if let Some(explicit) = args.reference_latent.clone() {
        Some(explicit)
    } else {
        // bge and its embedding kin keep log θ in `latent` (when resolved) and
        // their H-space Z in `cell_embedding`; `geometry_latent` prefers the
        // latter, which is the space `predict` projects a query into.
        let rel = match model_kind.cell_space() {
            run_manifest::CellSpace::Embedding => manifest.outputs.geometry_latent(),
            _ => manifest.outputs.latent.as_deref(),
        };
        let rel = rel.ok_or_else(|| {
            anyhow::anyhow!(
                "reference run `{prefix}` records no latent in its manifest; \
                 pass --reference-latent explicitly"
            )
        })?;
        Some(to_box(rel))
    };

    let data_files = match args.reference_data.clone() {
        Some(files) => files,
        None => {
            let files: Vec<Box<str>> = manifest.data.input.iter().map(|s| to_box(s)).collect();
            anyhow::ensure!(
                !files.is_empty(),
                "reference run `{prefix}` records no data files; pass --reference-data"
            );
            files
        }
    };
    let batch_files = match args.reference_batch_files.clone() {
        Some(files) => Some(files),
        None => {
            let files: Vec<Box<str>> = manifest.data.batch.iter().map(|s| to_box(s)).collect();
            (!files.is_empty()).then_some(files)
        }
    };

    Ok(ReferenceSpec {
        latent,
        data_files,
        batch_files,
    })
}

pub fn impute_model(args: &ImputeArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;

    let kind = crate::topic::model_metadata::resolve_run_kind(&args.model)?;
    match kind {
        RunKind::Topic
        | RunKind::Itopic
        | RunKind::JointTopic
        | RunKind::MaskedVae
        | RunKind::Vae
        | RunKind::Bge
        | RunKind::Svd => {}
        RunKind::JointSvd => anyhow::bail!(
            "impute does not support joint-svd runs yet: the joint dictionary spans \
             stacked modalities and the query-side projection is not defined for a \
             single-modality panel. Impute against the per-modality `senna svd` run."
        ),
        other => anyhow::bail!(
            "impute needs a run with a transferable per-cell latent; `{other}` runs \
             have no query-side projection. Supported: topic, masked-topic, \
             masked-sbp, masked-vae, joint-topic, vae, bge, svd."
        ),
    }

    let reference = resolve_reference(args, kind)?;

    // Reference counts — needed by every arm (they are the imputation
    // payload), and by the svd arm for its reference-side projection too.
    info!(
        "Opening reference data ({} file(s))",
        reference.data_files.len()
    );
    let ref_loaded = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: reference.data_files.clone(),
        batch_files: reference.batch_files.clone(),
        preload: args.preload_data,
        ..Default::default()
    })?;
    let ref_data: SparseIoVec = ref_loaded.data;

    let (new_cell_names, theta_new, theta_ref) = match kind {
        RunKind::Svd => svd_matching_latents(args, &ref_data)?,
        _ => predict_matching_latents(args, kind, &reference)?,
    };
    let n_new = theta_new.nrows();

    let imputed = retrieval_impute(
        &theta_new,
        &theta_ref,
        &ref_data,
        &RetrievalImputeConfig {
            knn: args.knn,
            temperature: args.knn_temperature,
            chunk: args.minibatch_size,
        },
    )?;

    let g_ref = ref_data.num_rows();
    let ref_gene_names = ref_data.row_names()?;
    let imputed_path = format!("{}.imputed.parquet", args.out);
    imputed.to_parquet_with_names(
        &imputed_path,
        (Some(&new_cell_names), Some("cell")),
        Some(&ref_gene_names),
    )?;
    info!("Wrote imputed {n_new}× {g_ref} matrix to {imputed_path}");

    Ok(())
}

//////////////////////////////////////////
// Predict-backed families (non-svd)    //
//////////////////////////////////////////

/// Query latent via `senna predict`, reference latent from the training
/// run's parquet, both mapped onto the model's matching space.
fn predict_matching_latents(
    args: &ImputeArgs,
    kind: RunKind,
    reference: &ReferenceSpec,
) -> anyhow::Result<(Vec<Box<str>>, Mat, Mat)> {
    // 1. Run senna-predict on the new data → writes {out}.predict_tmp.latent.parquet
    let predict_prefix: Box<str> = format!("{}.predict_tmp", args.out).into();
    info!("Projecting new data through the model (predict → {predict_prefix})");
    let predict_args = PredictArgs {
        ablate_features: None,
        null_from: None,
        eval_features: None,
        data_files: args.data_files.clone(),
        model: args.model.clone(),
        out: predict_prefix.clone(),
        batch_files: args.batch_files.clone(),
        minibatch_size: args.minibatch_size,
        block_size: args.block_size,
        preload_data: args.preload_data,
        refine_steps: 0,
        refine_lr: 0.01,
        refine_reg: 1.0,
        decoder_only: false,
        delta_iters: 3,
        // Coverage is reported; imputation adds no gate of its own.
        min_gene_overlap: 0.0,
        verbose: args.verbose,
        residual_out: None,
        residual_include_delta: false,
        residual_threshold: 0.0,
        feature_name_kind: crate::masked_topic::FeatureNameKindArg::Exact,
        feature_name_suffix_delim: None,
        keep_feature_suffix: None,
    };
    predict_model(&predict_args)?;

    // 2. Load the two latents and map both onto the matching space.
    let theta_new_path = format!("{predict_prefix}.latent.parquet");
    let ref_latent_path = reference
        .latent
        .as_deref()
        .expect("resolve_reference guarantees a latent for the predict families");
    info!("Loading projected latent and reference latent ({ref_latent_path})");
    let theta_new_parq = Mat::from_parquet_with_row_names(&theta_new_path, Some(0))?;
    let theta_ref_parq = Mat::from_parquet_with_row_names(ref_latent_path, Some(0))?;
    let new_cell_names: Vec<Box<str>> = theta_new_parq.rows;
    let mut theta_new = theta_new_parq.mat;
    let mut theta_ref = theta_ref_parq.mat;
    let (n_new, k_new) = (theta_new.nrows(), theta_new.ncols());
    let (n_ref, k_ref) = (theta_ref.nrows(), theta_ref.ncols());
    anyhow::ensure!(
        k_new == k_ref,
        "latent dimension mismatch: theta_new K={k_new} vs reference K={k_ref}"
    );
    info!("  θ_new: {n_new} cells × {k_new}; θ_ref: {n_ref} cells × {k_ref}");

    match kind {
        // Map both latents onto the simplex. L2 there correlates with cosine
        // for the cell-cell matching the L2-backed kNN index supports.
        //
        // exp + per-row renorm rather than bare exp: for a log-θ latent the
        // renorm is a no-op (rows already sum to 1), and for a Gaussian `z`
        // (vae / masked-vae) it is exactly the `softmax(z)` that gives its
        // proportions. Bare `exp` left those rows unnormalized, so cells were
        // matched on library-scale-like magnitude.
        RunKind::Topic
        | RunKind::Itopic
        | RunKind::JointTopic
        | RunKind::MaskedVae
        | RunKind::Vae => {
            crate::embed_common::softmax_rows_inplace(&mut theta_new);
            crate::embed_common::softmax_rows_inplace(&mut theta_ref);
        }
        // An embedding's magnitude carries depth, not identity: L2-normalize
        // so the index's L2 becomes cosine on the unit sphere.
        RunKind::Bge => {
            l2_normalize_rows_inplace(&mut theta_new);
            l2_normalize_rows_inplace(&mut theta_ref);
        }
        other => unreachable!("impute_model admits no other predict-backed kind: {other}"),
    }

    Ok((new_cell_names, theta_new, theta_ref))
}

////////////////////////////
// SVD dictionary path    //
////////////////////////////

/// Both sides projected through the frozen `dictionary.parquet` with the
/// training-time transform.
///
/// The saved training latent is NOT reused: it was projected through the
/// Nyström basis `u·diag(1/s)`, and `s` is not persisted — so a query
/// projected through `u` alone would sit in a per-component rescaling of
/// that space. Re-projecting the reference through the same map as the
/// query keeps the two sides exactly comparable, at the cost of one
/// extra streaming pass over the reference.
fn svd_matching_latents(
    args: &ImputeArgs,
    ref_data: &SparseIoVec,
) -> anyhow::Result<(Vec<Box<str>>, Mat, Mat)> {
    let (train_genes, u_dk) = crate::topic::model_metadata::load_dictionary(&args.model)?;
    let column_sum_norm = svd_column_sum_norm(&args.model);

    info!("Loading new data for the dictionary projection");
    let new_loaded = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: args.data_files.clone(),
        batch_files: args.batch_files.clone(),
        preload: args.preload_data,
        ..Default::default()
    })?;
    let new_data: SparseIoVec = new_loaded.data;
    let new_cell_names = new_data.column_names()?;

    let mut theta_new = project_onto_svd_dictionary(
        &new_data,
        &train_genes,
        &u_dk,
        column_sum_norm,
        args.block_size,
        "query",
    )?;
    let mut theta_ref = project_onto_svd_dictionary(
        ref_data,
        &train_genes,
        &u_dk,
        column_sum_norm,
        args.block_size,
        "reference",
    )?;

    // Direction, not magnitude: after the per-cell standardization the row
    // norms mostly reflect how much of the dictionary a panel covers.
    l2_normalize_rows_inplace(&mut theta_new);
    l2_normalize_rows_inplace(&mut theta_ref);

    Ok((new_cell_names, theta_new, theta_ref))
}

/// The training-time normalization scale, from the run manifest's recorded
/// fit arguments when available.
fn svd_column_sum_norm(model: &str) -> f32 {
    const DEFAULT: f32 = 1e4;
    let Ok((manifest, _)) = run_manifest::load_for(model) else {
        info!("no manifest for {model}; using the default normalization scale");
        return DEFAULT;
    };
    let recorded = manifest
        .train_args
        .as_ref()
        .and_then(|t| t.args.get("column_sum_norm"))
        .and_then(serde_json::Value::as_f64);
    match recorded {
        Some(c) if c > 0.0 => c as f32,
        _ => {
            info!("manifest for {model} records no column_sum_norm; using the default");
            DEFAULT
        }
    }
}

struct SvdProjParam<'a> {
    u_dk: &'a Mat,
    /// data row → dictionary row, by name.
    remap: &'a [Option<usize>],
    column_sum_norm: f32,
}

/// `[N, K]` projection of every column of `data` onto the dictionary:
/// remap rows by name onto the training axis, then the same transform the
/// training run applied per cell — L2-normalize to the recorded scale,
/// `log1p`, standardize — and multiply through `u`.
fn project_onto_svd_dictionary(
    data: &SparseIoVec,
    train_genes: &[Box<str>],
    u_dk: &Mat,
    column_sum_norm: f32,
    block_size: Option<usize>,
    what: &str,
) -> anyhow::Result<Mat> {
    let data_genes = data.row_names()?;
    // Defaults reproduce the exact-then-flexible name matching `predict`
    // applies when `--feature-name-kind` is left at `exact`.
    let opts = QueryNameOpts::default();
    let remap = build_gene_remap_with(train_genes, &data_genes, &opts);
    ensure_gene_coverage(&remap, 0.0, "--feature-name-kind")?;
    info!(
        "{what}: {} of the model's {} genes present across {} cells",
        remap.n_mapped,
        train_genes.len(),
        data.num_columns()
    );

    let n = data.num_columns();
    let k = u_dk.ncols();
    let param = SvdProjParam {
        u_dk,
        remap: &remap.new_to_train,
        column_sum_norm,
    };
    let mut proj_kn = Mat::zeros(k, n);
    data.visit_columns_by_block(&svd_proj_visitor, &param, &mut proj_kn, block_size)?;
    Ok(proj_kn.transpose())
}

fn svd_proj_visitor(
    job: (usize, usize),
    data: &SparseIoVec,
    param: &SvdProjParam,
    arc_proj_kn: Arc<Mutex<&mut Mat>>,
) -> anyhow::Result<()> {
    let (lb, ub) = job;
    let csc_in = data.read_columns_csc(lb..ub)?;
    let d_train = param.u_dk.nrows();

    // Remap onto the training gene axis but STAY SPARSE: the training-time
    // `scale_columns_inplace` standardizes over a column's stored entries
    // only, so running the same chain on a densified copy would standardize
    // against the zeros too and land in a different space.
    let mut coo = nalgebra_sparse::CooMatrix::new(d_train, ub - lb);
    for c_local in 0..csc_in.ncols() {
        let col = csc_in.col(c_local);
        for (&row_id, &v) in col.row_indices().iter().zip(col.values().iter()) {
            if let Some(t) = param.remap[row_id] {
                coo.push(t, c_local, v);
            }
        }
    }
    let mut x_dn = nalgebra_sparse::CscMatrix::from(&coo);

    x_dn.normalize_columns_inplace();
    x_dn *= param.column_sum_norm;
    x_dn.log1p_inplace();
    x_dn.scale_columns_inplace();

    let chunk = (x_dn.transpose() * param.u_dk).transpose();
    let mut proj_kn = arc_proj_kn.lock().expect("lock proj in svd impute");
    proj_kn.columns_range_mut(lb..ub).copy_from(&chunk);
    Ok(())
}

/// Unit-norm rows in place; an all-zero row stays zero (and is later
/// skipped by the retrieval core rather than matched arbitrarily).
fn l2_normalize_rows_inplace(m: &mut Mat) {
    for mut row in m.row_iter_mut() {
        let norm = row.norm();
        if norm > 0.0 {
            row /= norm;
        }
    }
}

#[cfg(test)]
#[path = "impute/tests.rs"]
mod tests;
