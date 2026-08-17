use super::*;
use crate::run_manifest::{write_kind_only, RunKind};

fn prefix_in(dir: &std::path::Path, name: &str) -> String {
    dir.join(name).to_string_lossy().to_string()
}

/// What a bare `senna annotate-gem -f <prefix>` resolves to, across every kind of
/// prefix it can be handed.
///
/// - no manifest → projection: every `senna gem` run produced before the manifest
///   existed, where a moving default would silently change what already-published
///   annotations mean.
/// - gem-encoder → enrichment: the whole point. Without it a no-`--mode` annotate
///   ran nearest-centroid on a topic model, which the flag's own help forbids.
/// - gem → projection, resolved POSITIVELY rather than by absence — the
///   difference between "this is a gem run" and "I found nothing and guessed".
#[test]
fn the_manifest_decides_the_mode_when_none_was_given() {
    let dir = tempfile::tempdir().unwrap();
    for (name, kind, expect) in [
        ("mystery_run", None, Mode::Projection),
        (
            "gem_encoder_run",
            Some(RunKind::GemEncoder),
            Mode::Enrichment,
        ),
        ("gem_run", Some(RunKind::Gem), Mode::Projection),
    ] {
        let prefix = prefix_in(dir.path(), name);
        if let Some(kind) = kind {
            write_kind_only(&prefix, kind).unwrap();
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
    let topic = prefix_in(dir.path(), "gem_encoder_run");
    let embed = prefix_in(dir.path(), "gem_run");
    write_kind_only(&topic, RunKind::GemEncoder).unwrap();
    write_kind_only(&embed, RunKind::Gem).unwrap();

    for (prefix, requested) in [
        (&topic, Mode::Projection),
        (&topic, Mode::Enrichment),
        (&embed, Mode::Projection),
        (&embed, Mode::Enrichment),
    ] {
        assert_eq!(resolve_mode(prefix, Some(requested)), requested);
    }
}
