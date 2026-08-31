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
//! per-family latent semantics, encoded once in [`matching_plan`].
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

/// How a run kind's cells reach the matching space.
///
/// This is the ONE authority every per-kind decision in this module
/// consults — the admission gate, the reference negotiation, and the
/// latent transform all derive from it, so a kind cannot be admitted in
/// one place and forgotten in another.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MatchingPlan {
    /// `predict` → softmax rows onto the simplex.
    ///
    /// For a `log θ` latent the renormalization is a no-op. For `masked-vae`
    /// it is what the head itself does — that decoder log-softmaxes `z` over
    /// K before the dictionary. For plain `vae` it is neither: the
    /// `GaussianNbDecoder` softmaxes `z·W` over the GENE axis and never over
    /// `z`'s own coordinates, so this is a projection chosen to make the
    /// unconstrained latent comparable under L2, not a readout of the head.
    /// It is still the right matching map — bare `exp` would leave rows
    /// matched on library-scale-like magnitude — but do not read a vae
    /// impute's neighbours as topic proportions.
    SoftmaxSimplex,
    /// `predict` → L2-normalized embedding rows: magnitude carries depth,
    /// not identity, so the index's L2 becomes cosine on the unit sphere.
    CosineEmbedding,
    /// No reusable stored latent — the training whitening scale is not
    /// persisted — so BOTH sides are re-projected through the frozen
    /// dictionary with the training-time transform. The batch correction
    /// that transform applied at training is not recoverable either; see
    /// [`warn_if_batch_corrected`].
    DictionaryProjection,
}

/// The impute policy per run kind. Deliberately a full `match` with no
/// `_` arm, for the same reason as [`RunKind::cell_space`]: a new kind
/// must say how it imputes (or that it cannot) before this compiles.
fn matching_plan(kind: RunKind) -> anyhow::Result<MatchingPlan> {
    match kind {
        RunKind::Topic
        | RunKind::Itopic
        | RunKind::JointTopic
        | RunKind::MaskedVae
        | RunKind::Vae => Ok(MatchingPlan::SoftmaxSimplex),
        RunKind::Bge => Ok(MatchingPlan::CosineEmbedding),
        RunKind::Svd => Ok(MatchingPlan::DictionaryProjection),
        // joint-svd stacks modalities, so a single-modality panel has no
        // defined projection; the graph/embedding kinds have no query-side
        // projection at all.
        RunKind::JointSvd
        | RunKind::Fne
        | RunKind::ResolveEmbeddingSpace
        | RunKind::Gem
        | RunKind::GemEncoder => anyhow::bail!(
            "impute needs a run with a transferable per-cell latent; `{kind}` runs \
             have no query-side projection here"
        ),
    }
}

/// The reference side, after the manifest/flag negotiation: where the
/// latent lives (`None` for svd, which re-projects), and which backends
/// hold the reference counts.
#[derive(Debug)]
struct ReferenceSpec {
    latent: Option<Box<str>>,
    data_files: Vec<Box<str>>,
}

/// Resolve the reference from `--reference` / the model's own manifest,
/// with the explicit `--reference-*` flags winning piecewise.
///
/// The manifest is consulted unless every needed piece was given
/// explicitly — so the legacy invocation (`--reference-latent` +
/// `--reference-data`, no manifest anywhere) keeps working unchanged.
fn resolve_reference(args: &ImputeArgs, model_kind: RunKind) -> anyhow::Result<ReferenceSpec> {
    // Re-derived rather than passed in: a (kind, plan) pair in the signature
    // could be constructed inconsistently, and the call is pure and cheap.
    let plan = matching_plan(model_kind)?;
    let needs_latent = plan != MatchingPlan::DictionaryProjection;
    if args.reference_latent.is_some() && !needs_latent {
        log::warn!(
            "--reference-latent is ignored for a {model_kind} model: its matching latent \
             is re-projected from the reference data (the training whitening scale is \
             not persisted)"
        );
    }

    let latent_settled = !needs_latent || args.reference_latent.is_some();
    if args.reference.is_none() && latent_settled && args.reference_data.is_some() {
        return Ok(ReferenceSpec {
            // `None` for the dictionary plan on every path, even when the
            // (just-warned-as-ignored) flag was passed.
            latent: needs_latent
                .then(|| args.reference_latent.clone())
                .flatten(),
            data_files: args.reference_data.clone().expect("checked above"),
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

    // The reference must share the MODEL's matching plan, not merely its
    // cell space: `Signed` covers both a vae (softmaxed z) and an svd run —
    // whose stored latent this module refuses to reuse even for svd itself,
    // because the whitening scale is not persisted. Plan equality is the
    // real invariant.
    let ref_plan = matching_plan(manifest.kind).map_err(|_| {
        anyhow::anyhow!(
            "reference run `{prefix}` is a `{}` run, which has no matching \
             latent to serve as an impute reference",
            manifest.kind
        )
    })?;
    anyhow::ensure!(
        ref_plan == plan,
        "reference run `{prefix}` is a `{}` run whose cells live in a different \
         matching space than the `{model_kind}` model's; match against a \
         reference of the same family",
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
        let rel = if plan == MatchingPlan::CosineEmbedding {
            manifest.outputs.geometry_latent()
        } else {
            manifest.outputs.latent.as_deref()
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

    Ok(ReferenceSpec { latent, data_files })
}

pub fn impute_model(args: &ImputeArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;

    let kind = crate::topic::model_metadata::resolve_run_kind(&args.model)?;
    let plan = matching_plan(kind)?;
    let reference = resolve_reference(args, kind)?;

    let open_reference = || -> anyhow::Result<SparseIoVec> {
        info!(
            "Opening reference data ({} file(s))",
            reference.data_files.len()
        );
        let loaded = read_data_on_shared_rows(ReadSharedRowsArgs {
            data_files: reference.data_files.clone(),
            batch_files: args.reference_batch_files.clone(),
            preload: args.preload_data,
            ..Default::default()
        })?;
        Ok(loaded.data)
    };

    let (new_cell_names, theta_new, theta_ref, ref_data) = match plan {
        // The dictionary projection consumes the reference counts itself.
        MatchingPlan::DictionaryProjection => {
            let ref_data = open_reference()?;
            let (names, theta_new, theta_ref) = svd_matching_latents(args, &ref_data)?;
            (names, theta_new, theta_ref, ref_data)
        }
        // The predict families never touch the reference counts before the
        // retrieval step, so the (potentially preloaded, multi-GB) reference
        // opens only after predict has succeeded.
        _ => {
            let (names, theta_new, theta_ref) = predict_matching_latents(args, plan, &reference)?;
            let ref_data = open_reference()?;
            (names, theta_new, theta_ref, ref_data)
        }
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
    plan: MatchingPlan,
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

    // The transform each plan calls for is documented on `MatchingPlan`.
    if plan == MatchingPlan::CosineEmbedding {
        l2_normalize_rows_inplace(&mut theta_new);
        l2_normalize_rows_inplace(&mut theta_ref);
    } else {
        softmax_rows_inplace(&mut theta_new);
        softmax_rows_inplace(&mut theta_ref);
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
    warn_if_batch_corrected(args);

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

/// The training-time normalization scale, replayed from the manifest's
/// recorded fit arguments through the typed [`RunManifest::train_args_as`]
/// reader. Runs predating the manifest (or the `train_args` record) take the
/// fit's long-standing default.
///
/// A blob this senna cannot deserialize warns and falls back rather than
/// failing the run: `run_manifest`'s header reserves blob PARSING for
/// `update` precisely so that a senna which gained a flag can still OPEN a
/// run, and turning that into a hard error here would make `impute` the
/// second command a renamed `SvdArgs` field can break — for one `f32` it
/// does not otherwise need.
fn svd_column_sum_norm(model: &str) -> f32 {
    const DEFAULT: f32 = 1e4;
    let Ok((manifest, _)) = run_manifest::load_for(model) else {
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
                 --column-sum-norm, the matching space here differs from its training space."
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
/// outputs, and both sides are projected without it: consistent with each
/// other, but not with the space the dictionary was fitted in. Residual
/// batch structure can therefore drive the retrieval.
///
/// Warned rather than refused, because the query may legitimately carry no
/// batch structure worth correcting, and the same projection is applied to
/// both sides. The fix is on the training side — persist the projection
/// state — and is tracked in `plans/impute-all-models.md`.
fn warn_if_batch_corrected(args: &ImputeArgs) {
    if std::path::Path::new(&format!("{}.delta.parquet", args.model)).is_file() {
        log::warn!(
            "{}: this svd run was fitted with a batch correction, which impute cannot \
             replay (the per-pseudobulk residual it used is not persisted). Query and \
             reference are projected without it, so retrieval may match on residual \
             batch structure rather than biology.",
            args.model
        );
    }
    if args.batch_files.is_some() || args.reference_batch_files.is_some() {
        log::warn!(
            "--batch-files / --reference-batch-files have no effect on an svd model: the \
             projection is per cell against a frozen dictionary, with no per-batch term \
             to estimate"
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
/// per-chunk transform ([`crate::svd::nystrom_preprocess_columns`]), and
/// multiply through `u`.
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

    // Remap onto the training gene axis, staying sparse for the reasons on
    // `nystrom_preprocess_columns`.
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
    crate::svd::nystrom_preprocess_columns(&mut x_dn, param.column_sum_norm, None);

    let chunk = (x_dn.transpose() * param.u_dk).transpose();
    let mut proj_kn = arc_proj_kn.lock().expect("lock proj in svd impute");
    proj_kn.columns_range_mut(lb..ub).copy_from(&chunk);
    Ok(())
}

#[cfg(test)]
#[path = "impute/tests.rs"]
mod tests;
