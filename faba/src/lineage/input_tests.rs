//! Which table θ comes from, and the metric it lands in.

use super::*;
use faba::manifest::{self, RunKind};
use serde_json::{Map, Value};

/// A unique scratch prefix per test, so the manifests written here cannot collide.
fn scratch(tag: &str) -> String {
    std::env::temp_dir()
        .join(format!("faba_lin_input_{}_{tag}", std::process::id()))
        .to_string_lossy()
        .into_owned()
}

/// Stamp `{prefix}.gem.json` the way a producer would.
fn write_manifest(prefix: &str, kind: RunKind, latent: Option<&str>) {
    let mut extra = Map::new();
    if let Some(l) = latent {
        extra.insert("latent".into(), Value::String(l.into()));
    }
    manifest::write(prefix, kind, extra).unwrap();
}

#[test]
fn auto_reads_the_simplex_only_when_the_run_stamps_log_theta() {
    // A topic run that states the contract → the simplex.
    let p = scratch("topic_ok");
    write_manifest(&p, RunKind::Topic, Some("log-theta"));
    assert_eq!(
        resolve_theta_from(ThetaFrom::Auto, &p).unwrap(),
        ThetaFrom::Latent
    );

    // A topic run that does NOT: pre-2026-07-21 files put raw logits in latent.parquet
    // under this same model_type, so auto must decline rather than exp() them.
    let p = scratch("topic_unstamped");
    write_manifest(&p, RunKind::Topic, None);
    assert_eq!(
        resolve_theta_from(ThetaFrom::Auto, &p).unwrap(),
        ThetaFrom::CellEmbedding
    );

    // An embedding run writes no latent.parquet at all.
    let p = scratch("embedding");
    write_manifest(&p, RunKind::Embedding, Some("embedding"));
    assert_eq!(
        resolve_theta_from(ThetaFrom::Auto, &p).unwrap(),
        ThetaFrom::CellEmbedding
    );

    // No manifest → nothing says, so keep the historical behaviour.
    let p = scratch("no_manifest");
    assert_eq!(
        resolve_theta_from(ThetaFrom::Auto, &p).unwrap(),
        ThetaFrom::CellEmbedding
    );
}

#[test]
fn explicit_latent_refuses_a_run_that_cannot_supply_it() {
    // An embedding run has no latent.parquet: fail loudly rather than read a
    // file that does not exist, or one that means something else.
    let p = scratch("explicit_embedding");
    write_manifest(&p, RunKind::Embedding, Some("embedding"));
    assert!(resolve_theta_from(ThetaFrom::Latent, &p).is_err());

    // An unstamped topic run is the dangerous case: latent.parquet EXISTS and has the
    // right shape, but may hold raw logits. Refusing beats a plausible wrong θ.
    let p = scratch("explicit_unstamped");
    write_manifest(&p, RunKind::Topic, None);
    assert!(resolve_theta_from(ThetaFrom::Latent, &p).is_err());

    let p = scratch("explicit_ok");
    write_manifest(&p, RunKind::Topic, Some("log-theta"));
    assert_eq!(
        resolve_theta_from(ThetaFrom::Latent, &p).unwrap(),
        ThetaFrom::Latent
    );
}

#[test]
fn cell_embedding_is_honoured_without_consulting_the_manifest() {
    // Even on a topic run stamping log-theta, an explicit request stands: it is the
    // escape hatch for comparing the co-embedding against the simplex.
    let p = scratch("forced_embedding");
    write_manifest(&p, RunKind::Topic, Some("log-theta"));
    assert_eq!(
        resolve_theta_from(ThetaFrom::CellEmbedding, &p).unwrap(),
        ThetaFrom::CellEmbedding
    );
}

#[test]
fn auto_geometry_follows_the_manifold() {
    assert_eq!(
        resolve_geometry(LatentGeometry::Auto, ThetaFrom::Latent),
        LatentGeometry::Hellinger,
        "a simplex gets its own metric"
    );
    assert_eq!(
        resolve_geometry(LatentGeometry::Auto, ThetaFrom::CellEmbedding),
        LatentGeometry::Cosine,
        "gem writes cell_embedding raw with its norm carrying depth"
    );
    // An explicit choice is never overridden by the manifold.
    assert_eq!(
        resolve_geometry(LatentGeometry::Euclidean, ThetaFrom::Latent),
        LatentGeometry::Euclidean
    );
}

#[test]
fn hellinger_puts_a_simplex_on_the_unit_sphere() {
    // Two cells on a 4-topic simplex: one diffuse, one nearly pure.
    let theta = DMatrix::from_row_slice(2, 4, &[0.25, 0.25, 0.25, 0.25, 0.97, 0.01, 0.01, 0.01]);
    let h = apply_geometry(&theta, LatentGeometry::Hellinger);
    for i in 0..2 {
        let norm: f32 = (0..4).map(|j| h[(i, j)] * h[(i, j)]).sum();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "Σθ = 1 ⇒ ‖√θ‖ = 1, so cosine and Euclidean coincide; row {i} had {norm}"
        );
    }
    assert!((h[(0, 0)] - 0.5).abs() < 1e-6, "√0.25 = 0.5");

    // Hellinger SEPARATES what θ·α would have merged: the diffuse and pure cells are
    // far apart here, which is the whole reason for reading the simplex directly.
    let d: f32 = (0..4).map(|j| (h[(0, j)] - h[(1, j)]).powi(2)).sum::<f32>();
    assert!(d.sqrt() > 0.5, "distinct compositions stay distinct, got {d}");
}

#[test]
fn cosine_normalizes_rows_and_euclidean_leaves_them_alone() {
    // Same direction, 10x apart in magnitude — the depth axis gem warns about.
    let theta = DMatrix::from_row_slice(2, 2, &[3.0, 4.0, 30.0, 40.0]);

    let c = apply_geometry(&theta, LatentGeometry::Cosine);
    for i in 0..2 {
        let norm: f32 = (0..2).map(|j| c[(i, j)] * c[(i, j)]).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "row {i} normalized to {norm}");
    }
    let gap: f32 = (0..2).map(|j| (c[(0, j)] - c[(1, j)]).powi(2)).sum();
    assert!(
        gap.sqrt() < 1e-5,
        "cells differing only in library size collapse together under cosine"
    );

    let e = apply_geometry(&theta, LatentGeometry::Euclidean);
    assert_eq!(e, theta, "euclidean is the identity");
    let gap: f32 = (0..2).map(|j| (e[(0, j)] - e[(1, j)]).powi(2)).sum::<f32>();
    assert!(
        gap.sqrt() > 20.0,
        "under Euclidean the same two cells sit far apart — the depth axis dominating"
    );
}

#[test]
fn hellinger_clamps_rather_than_trusting_a_forced_non_simplex() {
    // `--latent-geometry hellinger` can be aimed at a table that is not a simplex.
    // A negative entry must not produce NaN and poison every downstream distance.
    let m = DMatrix::from_row_slice(1, 3, &[-1.0, 0.0, 4.0]);
    let h = apply_geometry(&m, LatentGeometry::Hellinger);
    assert!(h.iter().all(|v| v.is_finite()), "no NaN escapes: {h:?}");
    assert_eq!(h[(0, 0)], 0.0);
    assert_eq!(h[(0, 2)], 2.0);
}
