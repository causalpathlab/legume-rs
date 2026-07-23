use super::*;

/// Write `{prefix}.model.json` with the given body and hand back the prefix.
fn with_model_json(dir: &std::path::Path, name: &str, body: &str) -> String {
    let prefix = dir.join(name).to_string_lossy().to_string();
    std::fs::write(format!("{prefix}.model.json"), body).unwrap();
    prefix
}

/// A `faba gem` prefix has NO model.json, and must keep getting projection —
/// this is every already-published embedding run, and the default moving under
/// them would silently change what their annotations mean.
#[test]
fn absent_model_json_resolves_to_projection() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("gem_run").to_string_lossy().to_string();
    assert!(!is_topic_model(&prefix));
    assert_eq!(resolve_mode(&prefix, None), Mode::Projection);
}

/// The whole point: a gem-encoder prefix picks enrichment on its own. Without
/// this, a no-`--mode` annotate ran nearest-centroid on a topic model, which the
/// flag's own help forbids.
#[test]
fn gem_encoder_model_json_resolves_to_enrichment() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = with_model_json(
        dir.path(),
        "topic_run",
        r#"{"model_type": "gem-encoder-softmax", "latent": "log-theta", "n_genes": 12}"#,
    );
    assert!(is_topic_model(&prefix));
    assert_eq!(resolve_mode(&prefix, None), Mode::Enrichment);
}

/// Detection is on the `gem-encoder` PREFIX, not the exact string, so older
/// files written under a different simplex map still read as topic models. Those
/// exist on disk and need enrichment just as much.
#[test]
fn older_simplex_map_still_reads_as_a_topic_model() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = with_model_json(
        dir.path(),
        "sbp_run",
        r#"{"model_type": "gem-encoder-sbp", "n_genes": 12}"#,
    );
    assert!(is_topic_model(&prefix));
    assert_eq!(resolve_mode(&prefix, None), Mode::Enrichment);
}

/// An explicit choice always wins — including the discouraged one, which warns
/// and proceeds rather than erroring. Reproducing a previously published call
/// has to stay possible.
#[test]
fn an_explicit_mode_is_never_overridden() {
    let dir = tempfile::tempdir().unwrap();
    let topic = with_model_json(
        dir.path(),
        "topic_run",
        r#"{"model_type": "gem-encoder-softmax"}"#,
    );
    let embed = dir.path().join("gem_run").to_string_lossy().to_string();

    assert_eq!(
        resolve_mode(&topic, Some(Mode::Projection)),
        Mode::Projection
    );
    assert_eq!(
        resolve_mode(&topic, Some(Mode::Enrichment)),
        Mode::Enrichment
    );
    assert_eq!(
        resolve_mode(&embed, Some(Mode::Enrichment)),
        Mode::Enrichment
    );
    assert_eq!(
        resolve_mode(&embed, Some(Mode::Projection)),
        Mode::Projection
    );
}

/// Malformed or truncated JSON must not panic and must not guess "topic". An
/// interrupted write is the realistic case, and falling back to projection keeps
/// the pre-existing behaviour rather than inventing a new one.
#[test]
fn unparseable_model_json_falls_back_to_projection() {
    let dir = tempfile::tempdir().unwrap();
    for (name, body) in [
        ("truncated", r#"{"model_type": "gem-encoder-soft"#),
        ("empty", ""),
        ("no_model_type", r#"{"latent": "log-theta"}"#),
        ("wrong_type", r#"{"model_type": 42}"#),
        ("other_model", r#"{"model_type": "senna-topic"}"#),
    ] {
        let prefix = with_model_json(dir.path(), name, body);
        assert!(
            !is_topic_model(&prefix),
            "{name} must not read as a topic model"
        );
        assert_eq!(resolve_mode(&prefix, None), Mode::Projection, "{name}");
    }
}
