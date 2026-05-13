//! `senna layout phate` — raw-gene-space PB landmarks laid out
//! with PHATE diffusion embedding over the log1p-CPM PB features.
//! Cells placed by cheap Nyström in proj space.

use super::fit_layout_common::{
    finalize_viz, preprocess_layout_data, resolve_inputs, LayoutCommonArgs, LayoutPrep,
    PbLayoutPrep, PhateCliArgs, ResolvedViz,
};
use super::viz_prep::apply_svd_preprocessing;
use crate::embed_common::*;
use crate::fit_pseudotime::{compute_pseudotime, PseudotimeArtifacts, RootSpec};
use crate::geometry::orient::rotate_root_to_bottom;
use crate::geometry::phate::phate_layout_2d;
use crate::principal_graph::PrincipalGraphArgs;
use crate::run_manifest::resolve;

#[derive(Args, Debug)]
pub struct LayoutPhateArgs {
    #[clap(flatten)]
    common: LayoutCommonArgs,

    #[clap(flatten)]
    phate: PhateCliArgs,

    #[arg(
        long,
        default_value_t = 0,
        help = "SVD preprocessing: keep top N components (0 = skip, use raw features)"
    )]
    svd_dims: usize,

    #[arg(
        long,
        default_value_t = 2000,
        help = "Number of highly variable genes to select (0 = use all genes)"
    )]
    n_hvg: usize,

    #[arg(
        long,
        default_value_t = false,
        help = "Rotate so the pseudotime root sits at the bottom of the layout",
        long_help = "Rigid 2D rotation of the PHATE layout so the pseudotime root \
                     anchor sits below the tip anchor along the y-axis. PHATE \
                     coords are only defined up to rotation/reflection, so this \
                     is principled — preserves all distances and cluster \
                     structure.\n\n\
                     Uses `manifest.pseudotime` if already populated; otherwise \
                     fits a fresh principal graph from `outputs.latent` (using \
                     `--root-cell` / `--root-node`) in-memory without writing \
                     any artifacts. Run `senna fit-pseudotime` first if you \
                     want non-default principal-graph hyperparameters."
    )]
    orient_by_root: bool,

    #[arg(
        long,
        help = "Root cell name for in-memory pseudotime when manifest.pseudotime is missing"
    )]
    root_cell: Option<Box<str>>,

    #[arg(
        long,
        help = "Root principal-graph node id (alternative to --root-cell)"
    )]
    root_node: Option<usize>,

    #[arg(
        long,
        default_value_t = 0.10,
        help = "Anchor-bin quantile width for orientation (bottom/top fraction)"
    )]
    orient_tip_quantile: f32,
}

// Defaults mirror clap `default_value_t` on each field above. Used by
// callers that build the args programmatically (e.g. `senna plot`
// auto-running layout when `manifest.layout.cell_coords` is missing).
impl Default for LayoutPhateArgs {
    fn default() -> Self {
        Self {
            common: LayoutCommonArgs::default(),
            phate: PhateCliArgs::default(),
            svd_dims: 0,
            n_hvg: 2000,
            orient_by_root: false,
            root_cell: None,
            root_node: None,
            orient_tip_quantile: 0.10,
        }
    }
}

pub fn fit_layout_phate(args: &LayoutPhateArgs) -> anyhow::Result<()> {
    let mut resolved = resolve_inputs(&args.common)?;
    // PHATE here is PB-level (diffusion-MDS over PB features → 2D);
    // DirectCells manifests (RunKind::Gbe) have no PB scaffolding so
    // we redirect.
    let LayoutPrep::PbThenNystrom(prep) = preprocess_layout_data(&args.common, &resolved)? else {
        anyhow::bail!(
            "`senna layout phate` does not support DirectCells manifests \
             (e.g. RunKind::Gbe). Use `senna layout umap --from <manifest>` instead — \
             UMAP runs cell-level directly on the GBE embedding."
        );
    };

    // HVG selection: keep top N genes by residual variance (mean-variance corrected)
    let features = if args.n_hvg > 0 && args.n_hvg < prep.pb_features.ncols() {
        info!(
            "Selecting top {} HVGs from {} genes (mean-variance corrected)",
            args.n_hvg,
            prep.pb_features.ncols()
        );
        data_beans_alg::hvg::select_hvg(&prep.pb_features, args.n_hvg)
    } else {
        prep.pb_features.clone()
    };

    let features = if args.svd_dims > 0 {
        apply_svd_preprocessing(&features, args.svd_dims)?
    } else {
        features
    };

    let pb_coords = phate_layout_2d(&features, &(&args.phate).into());

    let pb_coords = if args.orient_by_root {
        let pseudotime = obtain_pseudotime(args, &resolved, &prep)?;
        let pb_pt = aggregate_cell_pt_to_pb(&pseudotime, &prep);
        rotate_root_to_bottom(&pb_coords, &pb_pt, args.orient_tip_quantile)?
    } else {
        pb_coords
    };

    finalize_viz(&args.common, &mut resolved, &prep, &pb_coords)
}

/// Resolve per-cell pseudotime for orientation. Prefers a cached
/// `manifest.pseudotime.pseudotime` parquet; falls back to fitting a
/// principal graph in-memory from `outputs.latent` using the user-supplied
/// root, without writing any artifacts.
fn obtain_pseudotime(
    args: &LayoutPhateArgs,
    resolved: &ResolvedViz,
    prep: &PbLayoutPrep,
) -> anyhow::Result<Vec<f32>> {
    let cell_names = prep.data_vec.column_names()?;

    let manifest = resolved.manifest.as_ref().ok_or_else(|| {
        anyhow::anyhow!(
            "--orient-by-root requires --from <manifest>; \
             pseudotime needs a latent or a pre-computed pseudotime artifact to align by"
        )
    })?;
    let manifest_dir = resolved
        .manifest_path
        .as_ref()
        .and_then(|p| p.parent())
        .map(std::path::Path::to_path_buf)
        .unwrap_or_else(|| std::path::PathBuf::from("."));

    if let Some(pt_rel) = manifest.pseudotime.pseudotime.as_deref() {
        let pt_path = resolve(&manifest_dir, pt_rel).to_string_lossy().into_owned();
        info!("Reading cached pseudotime from {pt_path}");
        let MatWithNames {
            rows: pt_cells,
            mat: pt_mat,
            ..
        } = read_mat(&pt_path)?;
        anyhow::ensure!(
            pt_mat.nrows() == cell_names.len(),
            "pseudotime has {} rows but data has {} cells",
            pt_mat.nrows(),
            cell_names.len()
        );
        if pt_cells != cell_names {
            log::warn!(
                "pseudotime row names differ from data cell names — using positional alignment"
            );
        }
        let pt: Vec<f32> = (0..pt_mat.nrows()).map(|i| pt_mat[(i, 0)]).collect();
        return Ok(pt);
    }

    // No cached pseudotime → fit fresh from the latent.
    let latent_rel = manifest.outputs.latent.as_deref().ok_or_else(|| {
        anyhow::anyhow!(
            "--orient-by-root requested but manifest has neither pseudotime.pseudotime \
             nor outputs.latent; run `senna fit-pseudotime` first or provide a latent"
        )
    })?;
    let latent_path = resolve(&manifest_dir, latent_rel)
        .to_string_lossy()
        .into_owned();
    info!("Reading latent from {latent_path} for in-memory pseudotime");
    let MatWithNames {
        rows: latent_cells,
        mat: latent,
        ..
    } = read_mat(&latent_path)?;
    anyhow::ensure!(
        latent.nrows() == cell_names.len(),
        "latent has {} rows but data has {} cells",
        latent.nrows(),
        cell_names.len()
    );
    if latent_cells != cell_names {
        log::warn!("latent row names differ from data cell names — using positional alignment");
    }

    let root_spec = match (args.root_node, args.root_cell.as_deref()) {
        (Some(id), _) => RootSpec::Node(id),
        (None, Some(name)) => RootSpec::Cell(name),
        (None, None) => RootSpec::DefaultFirstCell,
    };

    let n_centroids = (latent.nrows() / 10).clamp(5, 200);
    let pg_args = PrincipalGraphArgs {
        n_centroids,
        ..PrincipalGraphArgs::default()
    };
    info!(
        "Fitting principal graph in-memory: K={n_centroids}, γ={:.2}",
        pg_args.gamma
    );

    let PseudotimeArtifacts { pseudotime, .. } =
        compute_pseudotime(&latent, &latent_cells, &pg_args, root_spec)?;
    Ok(pseudotime)
}

/// Mean per-PB pseudotime; NaN for PBs with no contributing cells (dropped
/// by coverage filter or all-NaN inputs). Aligns to `pb_coords` row order
/// via `prep.pb_membership_kept`.
fn aggregate_cell_pt_to_pb(pseudotime: &[f32], prep: &PbLayoutPrep) -> Vec<f32> {
    let n_pb = prep.pb_features.nrows();
    let mut sum = vec![0.0_f32; n_pb];
    let mut count = vec![0_usize; n_pb];
    for (cell_i, &pb_id) in prep.pb_membership_kept.iter().enumerate() {
        if pb_id == usize::MAX {
            continue;
        }
        let pt = pseudotime[cell_i];
        if !pt.is_finite() {
            continue;
        }
        sum[pb_id] += pt;
        count[pb_id] += 1;
    }
    (0..n_pb)
        .map(|i| {
            if count[i] > 0 {
                sum[i] / count[i] as f32
            } else {
                f32::NAN
            }
        })
        .collect()
}
