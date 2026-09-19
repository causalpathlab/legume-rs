//! What a run manifest tells the lineage θ resolver.
//!
//! The resolver itself is tested in `lineage`; what is senna's to get right is
//! the mapping from a stamped `kind` to the contract — which used to be inside
//! `lineage::lineage::input`.

use super::*;
use crate::run_manifest::write_kind_only;

/// A unique scratch prefix per test, so the manifests written here cannot collide.
fn scratch(tag: &str) -> String {
    std::env::temp_dir()
        .join(format!(
            "senna_lineage_manifest_{}_{tag}",
            std::process::id()
        ))
        .to_string_lossy()
        .into_owned()
}

#[test]
fn a_topic_run_promises_a_log_simplex_latent() {
    let p = scratch("topic");
    write_kind_only(&p, RunKind::Topic).unwrap();
    let c = latent_contract(&p);
    assert!(c.latent_is_log_simplex);
    assert!(!c.is_gem);
    assert_eq!(c.kind.as_deref(), Some("topic"));
}

#[test]
fn a_gem_run_promises_a_euclidean_co_embedding_and_no_velocity() {
    let p = scratch("gem");
    write_kind_only(&p, RunKind::Gem).unwrap();
    let c = latent_contract(&p);
    assert!(
        !c.latent_is_log_simplex,
        "gem writes no latent.parquet at all"
    );
    assert!(c.is_gem, "the caller uses this to explain the missing δ");
}

#[test]
fn an_unreadable_manifest_yields_an_unknown_contract() {
    // Nothing stamped at this prefix: the resolver must be told that, not
    // handed a default that reads as a real claim.
    let c = latent_contract(&scratch("nothing_here"));
    assert!(c.kind.is_none());
    assert!(!c.latent_is_log_simplex);
    assert!(
        c.source.contains("senna.json"),
        "the fallback warning names where a manifest was looked for, got {}",
        c.source
    );
}
