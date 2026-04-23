//! Manifest + parquet → `enrichment::GroupInputs` + marker matrix.

use crate::embed_common::Mat;
use crate::marker_support::build_annotation_matrix;
use crate::run_manifest::{self, RunManifest};
use enrichment::{GroupInputs, SpecificityMode};
use matrix_util::traits::IoOps;
use std::path::Path;

/// Senna-side companion to manifest.kind. The enrichment crate doesn't need
/// to know — it derives behavior from `SpecificityMode`. We keep this here
/// for the load-time dispatch (log-space exp, derive_pb_latent).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Topic,
    Svd,
}

pub struct LoadedInputs {
    pub group: GroupInputs,
    pub markers_gc: Mat,
    pub celltype_names: Vec<Box<str>>,
    pub specificity: SpecificityMode,
}

pub fn load_from_manifest(
    manifest_path: &str,
    markers_path: &str,
) -> anyhow::Result<(LoadedInputs, RunManifest, std::path::PathBuf)> {
    let (manifest, manifest_dir) = RunManifest::load(Path::new(manifest_path))?;
    log::info!(
        "Loaded manifest ({}): kind={}",
        manifest_path,
        manifest.kind
    );

    let resolve = |rel: &str| -> String {
        run_manifest::resolve(&manifest_dir, rel)
            .to_string_lossy()
            .into_owned()
    };

    // β / profile (G × K). Prefer the empirical NB-Fisher-weighted
    // dictionary (full gene resolution, column-simplex) when present;
    // fall back to the trained `dictionary` (feature-coarsened then
    // expanded — lossy for rare informative genes).
    let (dict_rel, dict_source) = match manifest.outputs.dictionary_empirical.as_deref() {
        Some(rel) => (rel, "empirical (NB-Fisher weighted)"),
        None => match manifest.outputs.dictionary.as_deref() {
            Some(rel) => (rel, "trained (feature-coarsened, expanded)"),
            None => anyhow::bail!(
                "manifest missing both outputs.dictionary_empirical and outputs.dictionary"
            ),
        },
    };
    let dict_path = resolve(dict_rel);
    let mut dict = Mat::from_parquet_with_row_names(&dict_path, Some(0))?;
    log::info!(
        "Loaded dictionary {} ({}): {}×{}",
        dict_path,
        dict_source,
        dict.mat.nrows(),
        dict.mat.ncols()
    );
    if dict.mat.max() <= 0.0 {
        log::info!("Detected log-space dictionary (max ≤ 0); exponentiating to probabilities");
        dict.mat = dict.mat.map(f32::exp);
    }

    // θ cell (N × K).
    let latent_rel = manifest
        .outputs
        .latent
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("manifest missing outputs.latent"))?;
    let latent_path = resolve(latent_rel);
    let latent = Mat::from_parquet_with_row_names(&latent_path, Some(0))?;
    log::info!(
        "Loaded latent {}: {}×{}",
        latent_path,
        latent.mat.nrows(),
        latent.mat.ncols()
    );

    let (profile_kind, spec_mode) = match manifest.kind.as_str() {
        // Topic kinds: β is already housekeeping-adjusted during training
        // (NB Fisher weighting). Rank directly by raw β rather than double-
        // adjusting via specificity transform — avoids over-suppressing
        // informative high-mean genes.
        "topic" | "itopic" | "joint-topic" => (Kind::Topic, SpecificityMode::Raw),
        "svd" | "joint-svd" => (Kind::Svd, SpecificityMode::Abs),
        other => {
            anyhow::bail!(
                "unsupported manifest.kind {:?} — senna annotate supports topic / itopic / \
                 joint-topic / svd / joint-svd",
                other
            )
        }
    };

    let cell_membership_nk: Mat = if matches!(profile_kind, Kind::Topic) && latent.mat.max() <= 0.0
    {
        log::info!("Detected log-space latent (max ≤ 0); exponentiating to probabilities");
        latent.mat.map(f32::exp)
    } else {
        latent.mat.clone()
    };

    anyhow::ensure!(
        latent.mat.ncols() == dict.mat.ncols(),
        "latent K {} ≠ dictionary K {}",
        latent.mat.ncols(),
        dict.mat.ncols()
    );

    // pb_gene (G × P). Required for the sample-permutation null.
    let pb_gene_rel = manifest.outputs.pb_gene.as_deref().ok_or_else(|| {
        anyhow::anyhow!(
            "manifest missing outputs.pb_gene — senna annotate's sample-permutation null needs \
             PB aggregates. Retrain with an up-to-date senna to populate this field."
        )
    })?;
    let pb_gene_path = resolve(pb_gene_rel);
    let pb_gene = Mat::from_parquet_with_row_names(&pb_gene_path, Some(0))?;
    log::info!(
        "Loaded pb_gene {}: {}×{}",
        pb_gene_path,
        pb_gene.mat.nrows(),
        pb_gene.mat.ncols()
    );

    // pb_latent (P × K). Optional — derive from pb_gene · β when absent.
    let pb_membership_pk: Mat = match manifest.outputs.pb_latent.as_deref() {
        Some(rel) => {
            let path = resolve(rel);
            let loaded = Mat::from_parquet_with_row_names(&path, Some(0))?;
            log::info!(
                "Loaded pb_latent {}: {}×{}",
                path,
                loaded.mat.nrows(),
                loaded.mat.ncols()
            );
            loaded.mat
        }
        None => {
            log::info!("pb_latent not in manifest — deriving from pb_gene · β (E-step proxy)");
            derive_pb_latent(&pb_gene.mat, &dict.mat, profile_kind)
        }
    };

    anyhow::ensure!(
        dict.mat.nrows() == pb_gene.mat.nrows(),
        "dictionary G {} ≠ pb_gene G {}",
        dict.mat.nrows(),
        pb_gene.mat.nrows()
    );
    anyhow::ensure!(
        pb_gene.mat.ncols() == pb_membership_pk.nrows(),
        "pb_gene P {} ≠ pb_membership P {}",
        pb_gene.mat.ncols(),
        pb_membership_pk.nrows()
    );
    anyhow::ensure!(
        pb_membership_pk.ncols() == dict.mat.ncols(),
        "pb_membership K {} ≠ dictionary K {}",
        pb_membership_pk.ncols(),
        dict.mat.ncols()
    );

    let annot = build_annotation_matrix(markers_path, &dict.rows)?;
    log::info!(
        "Marker matrix: {} genes × {} celltypes",
        annot.membership_ga.nrows(),
        annot.membership_ga.ncols()
    );

    let group = GroupInputs {
        profile_gk: dict.mat,
        pb_gene_gp: pb_gene.mat,
        pb_membership_pk,
        cell_membership_nk,
        gene_names: dict.rows,
        cell_names: latent.rows,
    };

    Ok((
        LoadedInputs {
            group,
            markers_gc: annot.membership_ga,
            celltype_names: annot.annot_names,
            specificity: spec_mode,
        },
        manifest,
        manifest_dir,
    ))
}

/// Derive a PB × K membership matrix from `pb_gene` (G × P) and `β` (G × K)
/// when the training kind didn't persist pb_latent.
///
/// Topic kinds: θ_PB[p, :] ∝ pb_gene[:, p]^T · β, row-normalized.
/// SVD kinds: use |β| so the product is non-negative, then row-normalize.
fn derive_pb_latent(pb_gene_gp: &Mat, dict_gk: &Mat, kind: Kind) -> Mat {
    let p = pb_gene_gp.ncols();
    let k = dict_gk.ncols();
    let dict_signed = match kind {
        Kind::Topic => dict_gk.clone(),
        Kind::Svd => dict_gk.map(f32::abs),
    };
    let raw_pk = pb_gene_gp.transpose() * &dict_signed;
    let mut out = Mat::zeros(p, k);
    for pi in 0..p {
        let s: f32 = (0..k).map(|kk| raw_pk[(pi, kk)].max(0.0)).sum();
        if s <= 0.0 {
            let uniform = 1.0 / k as f32;
            for kk in 0..k {
                out[(pi, kk)] = uniform;
            }
            continue;
        }
        for kk in 0..k {
            out[(pi, kk)] = raw_pk[(pi, kk)].max(0.0) / s;
        }
    }
    out
}
