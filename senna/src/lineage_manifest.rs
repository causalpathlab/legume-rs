//! Run-manifest glue for the `lineage` crate.
//!
//! `lineage` owns the trajectory implementations but must not depend on
//! `senna` — that would cycle, since `senna` re-exports it. So the manifest
//! side of each lineage command lives here: resolve `-f run.senna.json` into
//! the plain paths and loaded matrices the library takes, then write the
//! artifact paths it returns back into the manifest. This is the only place
//! that touches [`RunManifest`] on behalf of a lineage command.

use anyhow::Result;
use log::info;
use std::path::PathBuf;

use lineage::lineage::args::LineageArgs;
use lineage::lineage::run::{run_lineage, LineageInputs};
use lineage::lineage::LatentContract;
use lineage::pseudotime::{run_pseudotime, PseudotimeArgs, PseudotimeInputs, PseudotimeOutputs};

use crate::marker_embedding::load_marker_feature_embedding_from;
use crate::run_manifest::{
    default_path, derive_out_prefix, load_for, rel_to_manifest, resolve, RunManifest,
};

/// What `{prefix}`'s manifest says about its per-cell tables, in the form the
/// θ resolver in `lineage` takes. A missing or unreadable manifest is not
/// fatal: `--theta-from auto` then assumes the geometry rather than reading
/// it, which is what [`LatentContract::unknown`] means.
pub fn latent_contract(prefix: &str) -> LatentContract {
    let source = default_path(&derive_out_prefix(prefix));
    let Ok((manifest, _)) = load_for(prefix) else {
        return LatentContract::unknown(source);
    };
    contract_from_manifest(&manifest, source)
}

fn contract_from_manifest(manifest: &RunManifest, source: impl Into<Box<str>>) -> LatentContract {
    LatentContract {
        latent_is_log_simplex: manifest.kind.latent_is_log_simplex(),
        kind: Some(manifest.kind.to_string().into_boxed_str()),
        source: source.into(),
    }
}

/// `senna lineage` / `lupin lineage`: resolve the manifest-derived inputs,
/// then run the fit.
pub fn run_lineage_from_manifest(args: &LineageArgs) -> Result<()> {
    let prefix = args.from.as_ref();
    let source = default_path(&derive_out_prefix(prefix));
    let (contract, feature_embedding) = match load_for(prefix) {
        Ok((manifest, dir)) => {
            let contract = contract_from_manifest(&manifest, source);
            let feature_embedding = args
                .markers
                .is_some()
                .then(|| load_marker_feature_embedding_from(&manifest, &dir, prefix))
                .transpose()?;
            (contract, feature_embedding)
        }
        Err(_) => {
            anyhow::ensure!(
                args.markers.is_none(),
                "--markers needs a readable run.senna.json under {prefix} to locate \
                 feature_coembedding"
            );
            (LatentContract::unknown(source), None)
        }
    };
    let inputs = LineageInputs {
        contract,
        feature_embedding,
    };
    run_lineage(args, &inputs)
}

/// `senna pseudotime` / `lupin pseudotime`: resolve `--latent` / `--from`,
/// run the fit, and record the artifacts it wrote in the manifest.
pub fn run_pseudotime_from_manifest(args: &PseudotimeArgs) -> Result<()> {
    let manifest_ctx = load_manifest_ctx(args)?;
    let inputs = PseudotimeInputs {
        latent: resolve_latent_path(args, manifest_ctx.as_ref())?,
        cell_coords: manifest_ctx.as_ref().and_then(resolve_cell_coords),
    };

    let outputs = run_pseudotime(args, &inputs)?;

    if let Some(ctx) = manifest_ctx {
        update_manifest(ctx, &outputs)?;
    }
    Ok(())
}

/// Loaded manifest plus its on-disk path and resolved directory. Held across
/// the whole pseudotime run so we can both read existing entries (e.g.
/// `outputs.latent`, `layout.cell_coords`) and write new ones back.
struct ManifestCtx {
    manifest: RunManifest,
    path: PathBuf,
    dir: PathBuf,
}

fn load_manifest_ctx(args: &PseudotimeArgs) -> Result<Option<ManifestCtx>> {
    let Some(from) = args.manifest() else {
        return Ok(None);
    };
    let path = PathBuf::from(from);
    let (manifest, dir) = RunManifest::load(&path)?;
    Ok(Some(ManifestCtx {
        manifest,
        path,
        dir,
    }))
}

fn resolve_latent_path(args: &PseudotimeArgs, ctx: Option<&ManifestCtx>) -> Result<String> {
    if let Some(p) = args.latent() {
        return Ok(p.to_string());
    }
    let ctx = ctx.ok_or_else(|| anyhow::anyhow!("either --latent or --from is required"))?;
    let rel = ctx.manifest.outputs.geometry_latent().ok_or_else(|| {
        anyhow::anyhow!(
            "manifest {} has no outputs.cell_embedding or outputs.latent",
            ctx.path.display()
        )
    })?;
    Ok(resolve(&ctx.dir, rel).to_string_lossy().into_owned())
}

/// The run's 2D cell layout, which is what lets the pseudotime fit also
/// project its centroids into layout space. Absent is ordinary — the layout is
/// a separate command — so this says how to get one rather than failing.
fn resolve_cell_coords(ctx: &ManifestCtx) -> Option<String> {
    let Some(rel) = ctx.manifest.layout.cell_coords.as_deref() else {
        info!(
            "manifest has no layout.cell_coords; skipping 2D centroid projection \
             (run `senna layout phate --from ...` first to enable plot overlay)"
        );
        return None;
    };
    Some(resolve(&ctx.dir, rel).to_string_lossy().into_owned())
}

fn update_manifest(mut ctx: ManifestCtx, outputs: &PseudotimeOutputs) -> Result<()> {
    let rel = |p: &str| rel_to_manifest(&ctx.dir, p);
    ctx.manifest.pseudotime.pseudotime = Some(rel(&outputs.pseudotime));
    ctx.manifest.pseudotime.nodes_latent = Some(rel(&outputs.nodes_latent));
    ctx.manifest.pseudotime.nodes_2d = outputs.nodes_2d.as_deref().map(rel);
    ctx.manifest.pseudotime.edges = Some(rel(&outputs.edges));
    ctx.manifest.pseudotime.root_node = Some(outputs.root);
    // Tree layout is now produced by `senna layout tree`; clear any stale
    // paths from a previous run so downstream readers don't pick up a tree
    // that no longer matches this pseudotime fit.
    ctx.manifest.pseudotime.tree_cell_coords = None;
    ctx.manifest.pseudotime.tree_nodes_2d = None;
    ctx.manifest.save(&ctx.path)?;
    info!("Updated manifest {}", ctx.path.display());
    Ok(())
}

#[cfg(test)]
#[path = "lineage_manifest_tests.rs"]
mod tests;
