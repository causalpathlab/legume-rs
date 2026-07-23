use super::*;
use faba::manifest::{write as write_manifest, RunKind};

fn prefix_in(dir: &std::path::Path, name: &str) -> String {
    dir.join(name).to_string_lossy().to_string()
}

/// What a bare `faba annotate -f <prefix>` resolves to, across every kind of
/// prefix it can be handed.
///
/// - no manifest → projection: every `faba gem` run produced before the manifest
///   existed, where a moving default would silently change what already-published
///   annotations mean.
/// - topic → enrichment: the whole point. Without it a no-`--mode` annotate ran
///   nearest-centroid on a topic model, which the flag's own help forbids.
/// - embedding → projection, resolved POSITIVELY rather than by absence — the
///   difference between "this is a gem run" and "I found nothing and guessed".
#[test]
fn the_manifest_decides_the_mode_when_none_was_given() {
    let dir = tempfile::tempdir().unwrap();
    for (name, kind, expect) in [
        ("mystery_run", None, Mode::Projection),
        ("topic_run", Some(RunKind::Topic), Mode::Enrichment),
        ("embed_run", Some(RunKind::Embedding), Mode::Projection),
    ] {
        let prefix = prefix_in(dir.path(), name);
        if let Some(kind) = kind {
            write_manifest(&prefix, kind, serde_json::Map::new()).unwrap();
        }
        assert_eq!(resolve_mode(&prefix, None), expect, "{name}");
    }
}

/// An explicit choice always wins, in every combination — including the
/// discouraged one, which warns and proceeds rather than erroring so that
/// reproducing a previously published call stays possible.
#[test]
fn an_explicit_mode_is_never_overridden() {
    let dir = tempfile::tempdir().unwrap();
    let topic = prefix_in(dir.path(), "topic_run");
    let embed = prefix_in(dir.path(), "embed_run");
    write_manifest(&topic, RunKind::Topic, serde_json::Map::new()).unwrap();
    write_manifest(&embed, RunKind::Embedding, serde_json::Map::new()).unwrap();

    for (prefix, requested) in [
        (&topic, Mode::Projection),
        (&topic, Mode::Enrichment),
        (&embed, Mode::Projection),
        (&embed, Mode::Enrichment),
    ] {
        assert_eq!(resolve_mode(prefix, Some(requested)), requested);
    }
}

/// A prefix written before the rename still resolves to enrichment. Those runs
/// exist on disk, and the legacy name was only ever written by gem-encoder, so
/// dropping support would strand exactly the runs that most need enrichment.
#[test]
fn a_legacy_manifest_still_resolves_to_enrichment() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = prefix_in(dir.path(), "old_topic_run");
    std::fs::write(
        faba::manifest::legacy_path(&prefix),
        r#"{"model_type": "gem-encoder-softmax", "latent": "log-theta"}"#,
    )
    .unwrap();
    assert_eq!(resolve_mode(&prefix, None), Mode::Enrichment);
}
