//! `pinto impute` — impute full-feature expression for a new sample by
//! kNN over community propensities against a trained run's cells.
//!
//! The matching space is the per-cell community propensity — the one
//! partition-level cell representation every pinto model (`cage`, `lc`,
//! `dsvd`) publishes — so the retrieval step is uniform while the
//! query-side projection dispatches on the manifest's `command`:
//!
//! - **cage**: the full `pinto predict` pipeline runs first, writing its
//!   usual outputs under `{out}`; the query propensity is its result, and
//!   the reference propensity is the training run's own
//!   `{model}.propensity.parquet`. Both sides are incident-edge fractions
//!   over the same trained communities.
//! - **lc / dsvd** (no frozen gene dictionary): each cell's propensity is
//!   estimated from `{model}.gene_community.parquet` by a per-cell EM fit
//!   of a multinomial mixture over the community expression profiles —
//!   and it is estimated the SAME way for the reference cells, so the two
//!   sides come from one map rather than comparing an edge-based
//!   propensity against an expression-based one. The query's profile
//!   columns are renormalized over its matched genes (a panel observes a
//!   window of each profile; conditioning on that window is what makes
//!   the per-cell likelihood proper).
//!
//! The retrieval itself (kNN over propensity rows, softmax distance
//! weights, streamed weighted average of reference counts) lives in
//! [`data_beans_alg::retrieval_impute`]. A cell with no pairs (cage) or
//! no counts on any model gene (profiles) has a zero propensity row and
//! is skipped rather than matched arbitrarily.
//!
//! Writes `{out}.imputed.parquet` (`N_query` × `n_ref_features`), plus
//! `{out}.propensity.parquet` for the profile-projected query.

use crate::link_community::outputs::write_propensity_matrix;
use crate::predict::{predict_cage, PredictArgs};
use crate::util::common::*;
use crate::util::metadata::PintoMetadata;
use clap::Args;
use data_beans::sparse_data_visitors::VisitColumnsOps;
use data_beans::sparse_io_vector::SparseIoVec;
use data_beans_alg::retrieval_impute::{retrieval_impute, RetrievalImputeConfig};
use log::info;
use matrix_util::common_io::mkdir_parent;
use matrix_util::traits::IoOps;
use std::path::Path;

#[derive(Args, Debug)]
pub struct ImputeArgs {
    #[command(flatten)]
    pub predict: PredictArgs,

    #[arg(
        long,
        value_delimiter = ',',
        help = "Reference data files (default: the model manifest's data files)",
        long_help = "Sparse backends holding the reference cells' counts — the\n\
                     training data of the --model run, in the same order.\n\
                     Defaults to the data files recorded in {model}.pinto.json;\n\
                     pass this explicitly when the data has moved."
    )]
    pub reference_data: Option<Vec<Box<str>>>,

    #[arg(
        long,
        help = "Reference propensity parquet (default {model}.propensity.parquet)",
        long_help = "Cage models only. The training run's cell propensity;\n\
                     its rows must be the cells of --reference-data, in order.\n\
                     For lc / dsvd models the reference propensity is\n\
                     re-estimated from the reference data instead, so both\n\
                     sides come from the same profile projection."
    )]
    pub reference_propensity: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 25,
        help = "Reference nearest neighbours pooled per query cell",
        long_help = "Named --impute-knn (not -k): the SrtInput -k sizes the\n\
                     spatial pair graph, this sizes the retrieval pool."
    )]
    pub impute_knn: usize,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Softmax temperature on kNN distances (lower = sharper neighbour weights)"
    )]
    pub impute_temperature: f32,

    #[arg(
        long,
        default_value_t = 100,
        help = "EM iterations per cell for the lc / dsvd profile projection",
        hide = true
    )]
    pub profile_em_iters: usize,
}

pub fn run_impute(args: &ImputeArgs) -> anyhow::Result<()> {
    let c = &args.predict.common;
    let model = args.predict.model.as_ref();
    mkdir_parent(&c.out)?;

    // The manifest is the authority on what the model IS; artifact probing
    // (e.g. for feature_embedding.parquet) would silently reroute a cage
    // model with a misplaced file — or a `pinto predict` output prefix —
    // onto the profile path with different semantics. The probe survives
    // only as a fallback for pre-manifest runs.
    let meta = PintoMetadata::read(Path::new(&format!("{model}.pinto.json"))).ok();
    let is_cage = match meta.as_ref().map(|m| m.command.as_str()) {
        Some("cage") => true,
        Some("lc" | "dsvd") => false,
        Some(other) => anyhow::bail!(
            "{model} is a `{other}` run; impute needs a trained cage, lc, or dsvd model"
        ),
        None => {
            let probe = Path::new(&format!("{model}.feature_embedding.parquet")).is_file();
            log::warn!(
                "{model}.pinto.json not found; dispatching by artifact probe \
                 ({} path)",
                if probe { "cage" } else { "lc/dsvd profile" }
            );
            probe
        }
    };

    let reference_files = resolve_reference_data(args, meta.as_ref())?;
    let (query_prop, query_cells, ref_prop, ref_data) = if is_cage {
        cage_propensities(args, &reference_files)?
    } else {
        profile_propensities(args, &reference_files)?
    };
    anyhow::ensure!(
        query_prop.ncols() == ref_prop.ncols(),
        "query propensity has {} communities but the reference has {}; the two \
         files are from different runs",
        query_prop.ncols(),
        ref_prop.ncols()
    );

    let imputed = retrieval_impute(
        &query_prop,
        &ref_prop,
        &ref_data,
        &RetrievalImputeConfig {
            knn: args.impute_knn,
            temperature: args.impute_temperature,
            chunk: c.block_size.unwrap_or(512),
        },
    )?;

    let ref_gene_names = ref_data.row_names()?;
    let imputed_path = format!("{}.imputed.parquet", c.out);
    imputed.to_parquet_with_names(
        &imputed_path,
        (Some(&query_cells), Some("cell")),
        Some(&ref_gene_names),
    )?;
    info!(
        "Wrote imputed {} × {} matrix to {imputed_path}",
        imputed.nrows(),
        imputed.ncols()
    );
    Ok(())
}

/// Reference data files: the explicit flag, else the model's manifest.
fn resolve_reference_data(
    args: &ImputeArgs,
    meta: Option<&PintoMetadata>,
) -> anyhow::Result<Vec<Box<str>>> {
    if let Some(files) = args.reference_data.clone() {
        return Ok(files);
    }
    let model = args.predict.model.as_ref();
    let meta = meta.ok_or_else(|| {
        anyhow::anyhow!("cannot read {model}.pinto.json. Pass --reference-data explicitly.")
    })?;
    let files: Vec<Box<str>> = meta
        .data_files
        .clone()
        .unwrap_or_default()
        .into_iter()
        .map(Into::into)
        .collect();
    anyhow::ensure!(
        !files.is_empty(),
        "{model}.pinto.json records no data files; pass --reference-data explicitly"
    );
    for f in &files {
        anyhow::ensure!(
            Path::new(f.as_ref()).exists(),
            "reference data file {f} (recorded in {model}.pinto.json) does not exist; \
             pass --reference-data with the current location"
        );
    }
    Ok(files)
}

/// Open the reference backends. The multi-file `@basename` cell-name
/// suffixing matches what training wrote into its propensity table, so the
/// cage path's name check compares like with like.
fn open_reference(files: &[Box<str>], preload: bool) -> anyhow::Result<SparseIoVec> {
    info!("Opening reference data ({} file(s))", files.len());
    let loaded = auxiliary_data::data_loading::read_data_on_shared_rows(
        auxiliary_data::data_loading::ReadSharedRowsArgs {
            data_files: files.to_vec(),
            preload,
            ..Default::default()
        },
    )?;
    Ok(loaded.data)
}

//////////////////////////////////
// cage: predict-backed side    //
//////////////////////////////////

/// Run `pinto predict` for the query propensity; read the training run's
/// propensity for the reference side. The reference opens only after
/// predict has succeeded, so a preloaded atlas is not held in memory
/// through the whole predict pass.
fn cage_propensities(
    args: &ImputeArgs,
    reference_files: &[Box<str>],
) -> anyhow::Result<(Mat, Vec<Box<str>>, Mat, SparseIoVec)> {
    let model = args.predict.model.as_ref();

    info!("cage model: projecting the query sample through `pinto predict`");
    let (query_prop, query_cells) = predict_cage(&args.predict)?;

    let ref_data = open_reference(reference_files, args.predict.common.preload_data)?;
    let ref_prop_path = args
        .reference_propensity
        .clone()
        .unwrap_or_else(|| format!("{model}.propensity.parquet").into());
    let none = HashSet::default();
    let (ref_prop, _, _, ref_cells) =
        crate::plot::load::read_propensity(Path::new(ref_prop_path.as_ref()), &none)?;

    // Positional pairing of the reference propensity with the reference
    // backend is the load-bearing assumption; check it by name, not count.
    let data_cells = ref_data.column_names()?;
    anyhow::ensure!(
        ref_cells == data_cells,
        "{ref_prop_path}: its {} cells do not match the reference data's {} — \
         pass the training run's own data files (in training order) as \
         --reference-data",
        ref_cells.len(),
        data_cells.len()
    );
    Ok((query_prop, query_cells, ref_prop, ref_data))
}

///////////////////////////////////////
// lc / dsvd: profile projection    //
///////////////////////////////////////

/// Estimate BOTH sides' propensities from the model's gene-community
/// profiles, one EM per cell, then write the query's under `{out}`.
fn profile_propensities(
    args: &ImputeArgs,
    reference_files: &[Box<str>],
) -> anyhow::Result<(Mat, Vec<Box<str>>, Mat, SparseIoVec)> {
    let c = &args.predict.common;
    let model = args.predict.model.as_ref();
    if args.reference_propensity.is_some() {
        log::warn!(
            "--reference-propensity is ignored for an lc / dsvd model: the reference \
             propensity is re-estimated from the reference data so both sides come \
             from the same profile projection"
        );
    }

    let gc_path = format!("{model}.gene_community.parquet");
    anyhow::ensure!(
        Path::new(&gc_path).is_file(),
        "{model}: gene_community.parquet does not exist — not an lc / dsvd model prefix"
    );
    let (profiles, model_genes) = crate::plot::load::read_gene_community(Path::new(&gc_path))?;
    info!(
        "lc/dsvd model: projecting propensities through {} gene × {} community profiles",
        profiles.nrows(),
        profiles.ncols()
    );

    anyhow::ensure!(!c.data_files.is_empty(), "impute: no data files given");
    let query_data = open_reference(&c.data_files, c.preload_data)?;
    let feature_kind = args
        .predict
        .gene_name_mode
        .resolve_kind(&query_data.row_names()?);
    let query_cells = query_data.column_names()?;
    let query_prop = project_profile_propensity(
        &query_data,
        &profiles,
        &model_genes,
        &feature_kind,
        args.profile_em_iters,
        c.block_size,
        "query",
    )?;
    write_propensity_matrix(&c.out, &query_prop, &query_cells)?;
    info!("Wrote {}.propensity.parquet (profile-projected)", c.out);

    // The reference goes through the SAME map — not through the training
    // run's edge-based propensity — so the kNN compares like with like.
    let ref_data = open_reference(reference_files, c.preload_data)?;
    let ref_prop = project_profile_propensity(
        &ref_data,
        &profiles,
        &model_genes,
        &feature_kind,
        args.profile_em_iters,
        c.block_size,
        "reference",
    )?;
    Ok((query_prop, query_cells, ref_prop, ref_data))
}

struct ProfileEmParam {
    /// `[K, G_matched]` per-community gene probabilities, TRANSPOSED so a
    /// gene's K values are one contiguous column, renormalized over the
    /// matched genes (f64 for the per-cell EM accumulators).
    p_t: nalgebra::DMatrix<f64>,
    /// data row → matched-profile column.
    row_to_matched: Vec<Option<usize>>,
    /// Already clamped to ≥ 1.
    iters: usize,
}

/// `[N, K]` per-cell community proportions by EM over the profile mixture.
///
/// Per cell: `π ← Σ_g x_g · r_g` with responsibilities
/// `r_{g,k} ∝ π_k p̃_k(g)`, iterated to convergence. `p̃_k` is the profile
/// conditioned on this dataset's matched genes. A cell with no counts on
/// any matched gene keeps a zero row (skipped by the retrieval core).
fn project_profile_propensity(
    data: &SparseIoVec,
    profiles: &Mat,
    model_genes: &[Box<str>],
    feature_kind: &auxiliary_data::feature_names::FeatureNameKind,
    iters: usize,
    block_size: Option<usize>,
    what: &str,
) -> anyhow::Result<Mat> {
    let k = profiles.ncols();
    let data_genes = data.row_names()?;

    // Model gene → profile row, on canonicalized names.
    let model_idx: HashMap<Box<str>, usize> = model_genes
        .iter()
        .enumerate()
        .map(|(i, g)| (g.clone(), i))
        .collect();
    let mut matched_profile_rows: Vec<usize> = Vec::new();
    let mut row_to_matched: Vec<Option<usize>> = vec![None; data_genes.len()];
    let mut seen: HashMap<usize, usize> = HashMap::default();
    for (row, name) in data_genes.iter().enumerate() {
        let key = feature_kind.canonicalize(name);
        if let Some(&g) = model_idx.get(&key) {
            let slot = *seen.entry(g).or_insert_with(|| {
                matched_profile_rows.push(g);
                matched_profile_rows.len() - 1
            });
            row_to_matched[row] = Some(slot);
        }
    }
    anyhow::ensure!(
        !matched_profile_rows.is_empty(),
        "{what}: none of its {} genes match the model's {} — check --gene-name-mode",
        data_genes.len(),
        model_genes.len()
    );
    info!(
        "{what}: {} of the model's {} genes are present across {} cells",
        matched_profile_rows.len(),
        model_genes.len(),
        data.num_columns()
    );

    // Condition each community's profile on the matched window and drop the
    // rest: `p̃_k(g) = λ_gk / Σ_{g∈matched} λ_gk`. Stored [K, G_matched] so
    // the EM's per-gene access is one contiguous column.
    let g_m = matched_profile_rows.len();
    let mut p_t = nalgebra::DMatrix::<f64>::zeros(k, g_m);
    for (slot, &g) in matched_profile_rows.iter().enumerate() {
        for c in 0..k {
            p_t[(c, slot)] = f64::from(profiles[(g, c)]).max(0.0);
        }
    }
    for c in 0..k {
        let total: f64 = p_t.row(c).iter().sum();
        if total > 0.0 {
            let mut row = p_t.row_mut(c);
            row /= total;
        }
    }

    let param = ProfileEmParam {
        p_t,
        row_to_matched,
        iters: iters.max(1),
    };
    let n = data.num_columns();
    let mut prop_kn = Mat::zeros(k, n);
    data.visit_columns_by_block(&profile_em_visitor, &param, &mut prop_kn, block_size)?;
    Ok(prop_kn.transpose())
}

fn profile_em_visitor(
    job: (usize, usize),
    data: &SparseIoVec,
    param: &ProfileEmParam,
    arc_prop_kn: Arc<Mutex<&mut Mat>>,
) -> anyhow::Result<()> {
    let (lb, ub) = job;
    let csc = data.read_columns_csc(lb..ub)?;
    let k = param.p_t.nrows();
    let mut chunk = Mat::zeros(k, ub - lb);

    // One set of buffers for the whole block; the per-cell loop below only
    // clears and refills them.
    let mut obs: Vec<(usize, f64)> = Vec::new();
    let mut pi = vec![0.0f64; k];
    let mut num = vec![0.0f64; k];
    let mut resp = vec![0.0f64; k];

    for c_local in 0..csc.ncols() {
        let col = csc.col(c_local);
        // The cell's counts on the matched window.
        obs.clear();
        obs.extend(
            col.row_indices()
                .iter()
                .zip(col.values().iter())
                .filter_map(|(&row, &v)| {
                    param.row_to_matched[row].map(|slot| (slot, f64::from(v)))
                }),
        );
        if obs.iter().map(|&(_, x)| x).sum::<f64>() <= 0.0 {
            continue; // zero row → skipped downstream
        }

        pi.fill(1.0 / k as f64);
        for _ in 0..param.iters {
            num.fill(0.0);
            for &(slot, x) in &obs {
                // `resp_c = π_c · p̃_c(g)`, computed once for both the
                // denominator and the numerator; the profile column is
                // contiguous, so this is a straight cache-line scan.
                let p_col = param.p_t.column(slot);
                let mut denom = 0.0f64;
                for c in 0..k {
                    resp[c] = pi[c] * p_col[c];
                    denom += resp[c];
                }
                if denom <= 0.0 {
                    continue; // gene absent from every community
                }
                let f = x / denom;
                for c in 0..k {
                    num[c] += resp[c] * f;
                }
            }
            let assigned: f64 = num.iter().sum();
            if assigned <= 0.0 {
                pi.fill(0.0);
                break;
            }
            let mut delta = 0.0f64;
            for (c, pi_c) in pi.iter_mut().enumerate() {
                let next = num[c] / assigned;
                delta = delta.max((next - *pi_c).abs());
                *pi_c = next;
            }
            if delta < 1e-6 {
                break;
            }
        }
        for (c, &pi_c) in pi.iter().enumerate() {
            chunk[(c, c_local)] = pi_c as f32;
        }
    }

    let mut prop_kn = arc_prop_kn.lock().expect("lock propensity in impute");
    prop_kn.columns_range_mut(lb..ub).copy_from(&chunk);
    Ok(())
}

#[cfg(test)]
#[path = "impute/tests.rs"]
mod tests;
