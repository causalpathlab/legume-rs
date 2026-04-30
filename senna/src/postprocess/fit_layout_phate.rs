//! `senna layout phate` — raw-gene-space PB landmarks laid out
//! with PHATE diffusion embedding over the log1p-CPM PB features.
//! Cells placed by cheap Nyström in proj space.

use super::fit_layout_common::{
    finalize_viz, preprocess_layout_data, resolve_inputs, LayoutCommonArgs, PhateCliArgs,
};
use super::viz_prep::apply_svd_preprocessing;
use crate::embed_common::*;
use crate::geometry::phate::phate_layout_2d;

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
        }
    }
}

/// Run `senna layout phate` against an existing manifest with all
/// defaults — exactly equivalent to `senna layout phate --from PATH`
/// from the CLI. The manifest is updated in place with the resulting
/// `layout.cell_coords` / `layout.pb_coords` / `layout.pb_gene_mean`
/// paths. `preload` mirrors `--preload-data` (slow disks).
///
/// Output files land in the manifest's own directory (rooted at the
/// manifest's `prefix` basename), not the caller's CWD — so callers
/// like `senna plot` invoking this from arbitrary working dirs
/// produce stable layout artifacts next to the manifest.
pub fn run_default_phate_layout(manifest_path: &str, preload: bool) -> anyhow::Result<()> {
    use crate::run_manifest::RunManifest;
    use std::path::Path;

    let mut args = LayoutPhateArgs::default();
    args.common.from = Some(Box::from(manifest_path));
    args.common.preload_data = preload;

    // Anchor `out` to the manifest's directory so writes don't leak
    // into the caller's CWD. Strip any directory portion from
    // `manifest.prefix` (it may already be a path like
    // "results/run1") and re-attach to the manifest dir.
    let (manifest, manifest_dir) = RunManifest::load(Path::new(manifest_path))?;
    let prefix_base = Path::new(&manifest.prefix)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| manifest.prefix.clone());
    let abs_out = manifest_dir.join(prefix_base);
    args.common.out = Some(abs_out.to_string_lossy().into_owned().into_boxed_str());

    fit_layout_phate(&args)
}

pub fn fit_layout_phate(args: &LayoutPhateArgs) -> anyhow::Result<()> {
    let mut resolved = resolve_inputs(&args.common)?;
    let prep = preprocess_layout_data(&args.common, &resolved)?;

    // HVG selection: keep top N genes by residual variance (mean-variance corrected)
    let features = if args.n_hvg > 0 && args.n_hvg < prep.pb_features.ncols() {
        info!(
            "Selecting top {} HVGs from {} genes (mean-variance corrected)",
            args.n_hvg,
            prep.pb_features.ncols()
        );
        data_beans_alg::hvg_selection::select_hvg(&prep.pb_features, args.n_hvg)
    } else {
        prep.pb_features.clone()
    };

    let features = if args.svd_dims > 0 {
        apply_svd_preprocessing(&features, args.svd_dims)?
    } else {
        features
    };

    let pb_coords = phate_layout_2d(&features, &(&args.phate).into());
    finalize_viz(&args.common, &mut resolved, &prep, &pb_coords)
}
