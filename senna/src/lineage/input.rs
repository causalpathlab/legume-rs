//! Which per-cell table supplies θ, and the metric it is fitted and laid out in.
//!
//! Split out of [`super::run`] because "read the right manifold in the right
//! geometry" is a decision with its own preconditions (the manifest kind, the
//! stamped latent contract, the producer's own scaling note), not a step in the
//! fit. The fit downstream is metric-agnostic: it takes whatever matrix this
//! module hands it.

use anyhow::{Context, Result};
use log::{info, warn};
use std::path::Path;

use matrix_util::dmatrix_io::DMatrix;
use matrix_util::traits::IoOps;

use super::args::*;
use super::layout::l2_normalize_rows;
use crate::run_manifest::RunKind;

/// θ and δ as they came off disk, plus what they are.
///
/// `theta` is UNTRANSFORMED here — [`apply_geometry`] is applied by the caller,
/// which keeps this pair available afterwards for the velocity field. The arrows
/// are projected from the native θ/δ space onto whatever 2D coordinates the
/// transformed θ produces, the same separation scVelo makes: a metric chosen to
/// lay cells out well is not necessarily one δ is expressed in.
pub(super) struct LoadedTheta {
    pub cell_names: Vec<Box<str>>,
    pub theta: DMatrix<f32>,
    pub velocity: Option<DMatrix<f32>>,
}

/// Resolve `--theta-from auto` against the run manifest.
///
/// `latent` requires a run whose `latent.parquet` is on the probability simplex
/// — `senna gem-encoder`, not `senna gem`. `gem` writes no latent at all: its
/// per-cell table is a Euclidean embedding, and `exp()`-ing it would produce a
/// plausible wrong θ rather than an error.
///
/// The stamped-contract check this used to carry is gone with the file it read.
/// It existed because gem-encoder runs before 2026-07-21 wrote raw logits into
/// `latent.parquet` under the same model type, so the kind alone could not tell
/// them apart. The shared manifest did not exist then, so any prefix that has
/// one is necessarily newer than that split, and
/// [`RunKind::latent_is_log_simplex`] now carries the whole answer.
pub(super) fn resolve_theta_from(requested: ThetaFrom, prefix: &str) -> Result<ThetaFrom> {
    // A missing manifest is not fatal for `auto` — it just means the geometry
    // has to be assumed rather than read — so this stays an Option.
    let kind = crate::run_manifest::load_for(prefix)
        .ok()
        .map(|(m, _)| m.kind);
    let is_log_theta = kind.is_some_and(RunKind::latent_is_log_simplex);
    let manifest =
        crate::run_manifest::default_path(&crate::run_manifest::derive_out_prefix(prefix));

    match requested {
        ThetaFrom::CellEmbedding => Ok(ThetaFrom::CellEmbedding),
        ThetaFrom::Latent => {
            anyhow::ensure!(
                is_log_theta,
                "--theta-from latent needs a run whose latent is on the simplex; {manifest} \
                 reports {}. Only `senna gem-encoder` writes latent.parquet / \
                 velocity_factor.parquet; `senna gem` writes a Euclidean cell_embedding.",
                kind.map_or_else(|| "no manifest".to_string(), |k| k.to_string())
            );
            Ok(ThetaFrom::Latent)
        }
        ThetaFrom::Auto => {
            if is_log_theta {
                info!(
                    "--theta-from auto → latent: {manifest} is a {} run, so the fit reads the \
                     SIMPLEX directly rather than the θ·α co-embedding",
                    kind.expect("is_log_theta implies a kind")
                );
                Ok(ThetaFrom::Latent)
            } else {
                if kind.is_none() {
                    warn!(
                        "no readable manifest at {manifest}, so --theta-from auto falls back to \
                         cell_embedding. Pass --theta-from explicitly if that is not what this \
                         prefix holds."
                    );
                }
                Ok(ThetaFrom::CellEmbedding)
            }
        }
    }
}

/// Resolve `--latent-geometry auto` from where θ came from: a simplex gets
/// Hellinger, a raw cell embedding gets cosine (what `senna gem` documents for
/// its own output).
pub(super) fn resolve_geometry(requested: LatentGeometry, from: ThetaFrom) -> LatentGeometry {
    match requested {
        LatentGeometry::Auto => match from {
            ThetaFrom::Latent => LatentGeometry::Hellinger,
            _ => LatentGeometry::Cosine,
        },
        explicit => explicit,
    }
}

/// Put θ into the requested metric.
///
/// Hellinger is `√θ` — Euclidean distance on the result is Hellinger distance
/// (up to a constant), the proper metric on a simplex. Because `Σθ = 1`, the
/// result already has unit L2 norm per row, so cosine and Euclidean coincide on
/// it and no further normalization is wanted. Negative entries cannot occur on a
/// simplex but are clamped rather than trusted, since a caller can force
/// `hellinger` on a table that is not one.
pub(super) fn apply_geometry(theta: &DMatrix<f32>, geometry: LatentGeometry) -> DMatrix<f32> {
    match geometry {
        LatentGeometry::Euclidean => theta.clone(),
        LatentGeometry::Cosine => l2_normalize_rows(theta),
        LatentGeometry::Hellinger => theta.map(|v| v.max(0.0).sqrt()),
        // Resolved by `resolve_geometry` before this point.
        LatentGeometry::Auto => theta.clone(),
    }
}

/// Read the θ/δ pair named by `from`.
///
/// On the `latent` path `latent.parquet` holds LOG θ, so it is exponentiated
/// here — that is the whole content of the `log-theta` contract the resolver
/// checked. Its δ partner is `velocity_factor.parquet` (K space), not
/// `velocity.parquet` (H space); pairing θ from one space with δ from the other
/// would silently produce nonsense, so the two travel together.
pub(super) fn load_theta(prefix: &str, from: ThetaFrom, no_velocity: bool) -> Result<LoadedTheta> {
    let (theta_file, velocity_file, space) = match from {
        ThetaFrom::Latent => ("latent", "velocity_factor", "K (topic simplex)"),
        _ => ("cell_embedding", "velocity", "H (gene-embedding)"),
    };

    let theta_path = format!("{prefix}.{theta_file}.parquet");
    let cell = DMatrix::<f32>::from_parquet(&theta_path)
        .with_context(|| format!("reading θ from {theta_path}"))?;
    let cell_names = cell.rows;
    let theta = if from == ThetaFrom::Latent {
        // log θ → θ. The contract was verified upstream; this is the map it names.
        cell.mat.map(f32::exp)
    } else {
        cell.mat
    };
    let n = theta.nrows();
    info!(
        "θ from {theta_path}: {n} cells × {} dims, {space} space",
        theta.ncols()
    );

    let velocity_path = format!("{prefix}.{velocity_file}.parquet");
    let velocity = if no_velocity {
        None
    } else if Path::new(&velocity_path).exists() {
        let vel = DMatrix::<f32>::from_parquet(&velocity_path)
            .with_context(|| format!("reading velocity {velocity_path}"))?;
        anyhow::ensure!(
            vel.mat.nrows() == n,
            "velocity {velocity_path} has {} rows but θ has {n}",
            vel.mat.nrows()
        );
        anyhow::ensure!(
            vel.mat.ncols() == theta.ncols(),
            "velocity {velocity_path} has {} columns but θ has {} — δ must live in θ's space",
            vel.mat.ncols(),
            theta.ncols()
        );
        Some(vel.mat)
    } else {
        warn!("velocity file {velocity_path} absent; forest falls back to the geometric MST");
        None
    };

    Ok(LoadedTheta {
        cell_names,
        theta,
        velocity,
    })
}

/// Read `cell_embedding.parquet` for the `--markers` node calls.
///
/// Marker scoring is a nearest-centroid statistic against the CO-EMBEDDED gene
/// vectors in `feature_embedding.parquet`, which live in H space. So it reads
/// this table even when the trajectory itself was fitted on the K-space simplex:
/// the two answer different questions and only one of them needs the gene
/// vectors to share a metric with the cells.
pub(super) fn load_marker_theta(prefix: &str, cell_names: &[Box<str>]) -> Result<DMatrix<f32>> {
    let path = format!("{prefix}.cell_embedding.parquet");
    let emb = DMatrix::<f32>::from_parquet(&path)
        .with_context(|| format!("reading {path} for --markers (marker scoring is H-space)"))?;
    anyhow::ensure!(
        emb.mat.nrows() == cell_names.len(),
        "{path} has {} rows but θ has {} — the two tables disagree on the cell set",
        emb.mat.nrows(),
        cell_names.len()
    );
    Ok(emb.mat)
}

#[cfg(test)]
#[path = "input_tests.rs"]
mod input_tests;
