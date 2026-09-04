//! `senna impute` — post-hoc kNN imputation against a reference dataset.
//!
//! Pipeline:
//! 1. Place the new sparse data in the model's latent space — through
//!    `senna predict` for the encoder/projection families, or a direct
//!    dictionary projection for `svd` runs.
//! 2. Place the reference cells in the SAME space: the training run's
//!    latent (topic families), its cell embedding (`bge`, `simba`), or the same
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
use auxiliary_data::data_loading::{read_data_on_shared_rows, ReadSharedRowsArgs};
use clap::Args;
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
        help = "Trained model prefix (topic family, masked family, vae, bge, simba, or svd run)"
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
                     families, the H-space cell embedding for a bge or simba model.\n\
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

    #[arg(
        long,
        default_value_t = crate::embed_common::ComputeDevice::Cpu,
        value_enum,
        help = "Compute device for the inner predict",
        long_help = "Compute device for the inner predict.\n\
                     `cuda` / `metal` require the matching cargo feature.\n\
                     Matters for a bge or simba model, whose projection re-runs\n\
                     the per-cell SGD of training; the encoder families infer\n\
                     with one forward pass."
    )]
    pub device: crate::embed_common::ComputeDevice,

    #[arg(long, default_value_t = 0, help = "Device ordinal (for cuda/metal)")]
    pub device_no: usize,
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
        RunKind::Topic | RunKind::Itopic | RunKind::MaskedVae | RunKind::Vae => {
            Ok(MatchingPlan::SoftmaxSimplex)
        }
        // simba projects a query through bge's path (frozen gene table, zero
        // bias), so it matches in its cell embedding exactly as bge does.
        RunKind::Bge | RunKind::Simba => Ok(MatchingPlan::CosineEmbedding),
        RunKind::Svd => Ok(MatchingPlan::DictionaryProjection),
        // The joint families write no encoder checkpoint (`has_model: false`),
        // so `predict` cannot load one and the simplex arm would fail deep
        // inside it rather than here. Refuse up front and name the way out.
        RunKind::JointTopic | RunKind::JointSvd => anyhow::bail!(
            "impute does not support `{kind}` runs: they write no encoder checkpoint, \
             so there is no query-side projection. Impute against the per-modality \
             `senna {}` run instead.",
            if kind == RunKind::JointTopic {
                "topic"
            } else {
                "svd"
            }
        ),
        // The graph / co-embedding kinds have no query-side projection at all.
        RunKind::Fne | RunKind::ResolveEmbeddingSpace | RunKind::Gem | RunKind::GemEncoder => {
            anyhow::bail!(
                "impute needs a run with a transferable per-cell latent; `{kind}` runs \
                 have no query-side projection here"
            )
        }
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
    write_model_imputed_genes(args, &new_cell_names)?;

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
        // bge: the latent comes from pass 1 on the matched genes (unchanged), and
        // predict also initializes the genes the model never saw and writes their
        // per-cell rates, which become the model-imputed table below — retrieval
        // cannot reach a gene the reference has no counts for.
        no_init_genes: false,
        init_neighbours: graph_embedding_util::transfer::DEFAULT_INIT_NEIGHBOURS,
        init_similarity_floor: graph_embedding_util::transfer::DEFAULT_SIMILARITY_FLOOR,
        init_genes_in_fit: false,
        emit_gene_rates: true,
        null_from: None,
        eval_features: None,
        data_files: args.data_files.clone(),
        bulk: Vec::new(),
        bulk_table: crate::embed_common::BulkTableArgs::default(),
        model: args.model.clone(),
        out: predict_prefix.clone(),
        batch_files: args.batch_files.clone(),
        minibatch_size: args.minibatch_size,
        block_size: args.block_size,
        preload_data: args.preload_data,
        device: args.device.clone(),
        device_no: args.device_no,
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
    let column_sum_norm = crate::svd::project::column_sum_norm(&args.model);
    crate::svd::project::warn_if_batch_corrected(
        &args.model,
        args.batch_files.is_some() || args.reference_batch_files.is_some(),
    );

    info!("Loading new data for the dictionary projection");
    let new_loaded = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files: args.data_files.clone(),
        batch_files: args.batch_files.clone(),
        preload: args.preload_data,
        ..Default::default()
    })?;
    let new_data: SparseIoVec = new_loaded.data;
    let new_cell_names = new_data.column_names()?;

    let mut theta_new = crate::svd::project::project_onto_dictionary(
        &new_data,
        &train_genes,
        &u_dk,
        column_sum_norm,
        // impute exposes no query-axis flags; the defaults reproduce the
        // exact-then-flexible matching `predict` uses at `--feature-name-kind
        // exact`, and nothing is hidden.
        &crate::topic::eval::QueryNameOpts::default(),
        args.block_size,
        "query",
    )?;
    let mut theta_ref = crate::svd::project::project_onto_dictionary(
        ref_data,
        &train_genes,
        &u_dk,
        column_sum_norm,
        &crate::topic::eval::QueryNameOpts::default(),
        args.block_size,
        "reference",
    )?;

    // Direction, not magnitude: after the per-cell standardization the row
    // norms mostly reflect how much of the dictionary a panel covers.
    l2_normalize_rows_inplace(&mut theta_new);
    l2_normalize_rows_inplace(&mut theta_ref);

    Ok((new_cell_names, theta_new, theta_ref))
}

/// The genes the model never saw, imputed by the MODEL rather than retrieved:
/// read back predict's alignment and rate tables (written under the
/// `predict_tmp` prefix when the query carried unseen genes), keep the
/// initialized columns, and write `{out}.imputed_model_genes.parquet`. Absent
/// tables — a plain run, or no unseen gene — write nothing.
fn write_model_imputed_genes(args: &ImputeArgs, new_cell_names: &[Box<str>]) -> anyhow::Result<()> {
    let prefix = format!("{}.predict_tmp", args.out);
    let al_path = format!("{prefix}.gene_alignment.parquet");
    let rates_path = format!("{prefix}.gene_rates.parquet");
    if !(std::path::Path::new(&al_path).exists() && std::path::Path::new(&rates_path).exists()) {
        return Ok(());
    }
    let mut cols =
        matrix_util::parquet::read_parquet_string_columns_by_name(&al_path, &["gene", "status"])?;
    let al_status = cols.pop().expect("status column");
    let al_genes = cols.pop().expect("gene column");
    let rates = <Mat as IoOps>::from_parquet(&rates_path)?;
    anyhow::ensure!(
        rates.mat.nrows() == new_cell_names.len(),
        "{rates_path}: {} rows for {} query cells",
        rates.mat.nrows(),
        new_cell_names.len()
    );
    let (names, m) = model_imputed_columns(&al_genes, &al_status, &rates.cols, &rates.mat);
    if names.is_empty() {
        return Ok(());
    }
    let path = format!("{}.imputed_model_genes.parquet", args.out);
    m.to_parquet_with_names(&path, (Some(new_cell_names), Some("cell")), Some(&names))?;
    info!(
        "Wrote {path}: {} genes the model never saw, imputed from their initialized rows \
         (provenance in {al_path}); they are NOT in the retrieved matrix",
        names.len()
    );
    Ok(())
}

/// The model-imputed columns for the genes the model never saw: from predict's
/// `gene_alignment` (status per union gene) and `gene_rates` (per-cell rates of
/// the missing and initialized genes), keep exactly the `initialized` columns, in
/// the alignment's order. Retrieval cannot impute these genes — the reference has
/// no counts for them — so they are written as their own table with their own
/// provenance rather than mixed into the retrieved matrix.
pub(crate) fn model_imputed_columns(
    alignment_genes: &[Box<str>],
    alignment_status: &[Box<str>],
    rate_genes: &[Box<str>],
    rates: &Mat,
) -> (Vec<Box<str>>, Mat) {
    let col_of: std::collections::HashMap<&str, usize> = rate_genes
        .iter()
        .enumerate()
        .map(|(j, g)| (g.as_ref(), j))
        .collect();
    let keep: Vec<(Box<str>, usize)> = alignment_genes
        .iter()
        .zip(alignment_status)
        .filter(|(_, st)| {
            graph_embedding_util::transfer::GeneStatus::parse(st)
                == Some(graph_embedding_util::transfer::GeneStatus::Initialized)
        })
        .filter_map(|(g, _)| col_of.get(g.as_ref()).map(|&j| (g.clone(), j)))
        .collect();
    let mut out = Mat::zeros(rates.nrows(), keep.len());
    for (k, (_, j)) in keep.iter().enumerate() {
        out.set_column(k, &rates.column(*j));
    }
    (keep.into_iter().map(|(g, _)| g).collect(), out)
}

#[cfg(test)]
#[path = "impute/tests.rs"]
mod tests;
