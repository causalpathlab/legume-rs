use super::*;

fn prefix_in(dir: &std::path::Path, name: &str) -> String {
    dir.join(name).to_string_lossy().to_string()
}

/// Round-trip: what a producer writes is what a consumer reads back. This is the
/// whole contract, and it is the thing that breaks silently if either side is
/// edited alone.
#[test]
fn write_then_detect_round_trips_both_kinds() {
    let dir = tempfile::tempdir().unwrap();
    for kind in [RunKind::Embedding, RunKind::Topic] {
        let prefix = prefix_in(dir.path(), &format!("{kind:?}_run"));
        write(&prefix, kind, Map::new()).unwrap();
        assert_eq!(
            detect(&prefix),
            Some(Detected {
                kind,
                legacy: false
            })
        );
    }
}

/// Extra fields must survive alongside the stamped `model_type` — producers put
/// the latent contract and `velocity_common_mode` here and readers depend on
/// them.
#[test]
fn extra_fields_are_preserved_and_model_type_is_stamped() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = prefix_in(dir.path(), "run");
    let mut extra = Map::new();
    extra.insert("latent".into(), Value::String("log-theta".into()));
    extra.insert("n_genes".into(), Value::from(12u32));
    write(&prefix, RunKind::Topic, extra).unwrap();

    let text = std::fs::read_to_string(path(&prefix)).unwrap();
    let v: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(
        v["model_type"],
        Value::String(RunKind::Topic.model_type().into())
    );
    assert_eq!(v["latent"], Value::String("log-theta".into()));
    assert_eq!(v["n_genes"], Value::from(12u32));
}

/// A prefix written before the rename still resolves, and says so. Those runs
/// exist on disk; refusing them would strand every published gem-encoder output.
#[test]
fn the_legacy_name_is_still_read_and_flagged() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = prefix_in(dir.path(), "old_run");
    std::fs::write(
        legacy_path(&prefix),
        r#"{"model_type": "gem-encoder-softmax", "latent": "log-theta"}"#,
    )
    .unwrap();
    assert_eq!(
        detect(&prefix),
        Some(Detected {
            kind: RunKind::Topic,
            legacy: true
        })
    );
}

/// The current name wins when both exist, so a re-run upgrades a prefix in place
/// rather than being shadowed by the stale file beside it.
#[test]
fn the_current_name_takes_precedence_over_the_legacy_one() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = prefix_in(dir.path(), "both");
    std::fs::write(
        legacy_path(&prefix),
        r#"{"model_type": "gem-encoder-softmax"}"#,
    )
    .unwrap();
    write(&prefix, RunKind::Embedding, Map::new()).unwrap();
    assert_eq!(
        detect(&prefix),
        Some(Detected {
            kind: RunKind::Embedding,
            legacy: false
        })
    );
}

/// Every way of saying "this prefix does not identify itself" must come back
/// `None`, so no consumer can accidentally treat a guess as an answer.
#[test]
fn unidentifiable_prefixes_detect_as_none() {
    let dir = tempfile::tempdir().unwrap();
    assert_eq!(detect(&prefix_in(dir.path(), "absent")), None);

    for (name, body) in [
        ("truncated", r#"{"model_type": "gem-enco"#),
        ("empty", ""),
        ("no_field", r#"{"latent": "log-theta"}"#),
        ("not_a_string", r#"{"model_type": 42}"#),
        ("foreign", r#"{"model_type": "senna-topic"}"#),
    ] {
        let prefix = prefix_in(dir.path(), name);
        std::fs::write(path(&prefix), body).unwrap();
        assert_eq!(detect(&prefix), None, "{name} must not resolve to a kind");
    }
}

/// Classification is on the prefix of `model_type`, so a producer can change its
/// simplex map without a consumer needing a new arm.
#[test]
fn model_type_is_classified_by_prefix() {
    assert_eq!(
        RunKind::from_model_type("gem-encoder-softmax"),
        Some(RunKind::Topic)
    );
    assert_eq!(
        RunKind::from_model_type("gem-encoder-sbp"),
        Some(RunKind::Topic)
    );
    assert_eq!(
        RunKind::from_model_type("gem-embedding"),
        Some(RunKind::Embedding)
    );
    assert_eq!(RunKind::from_model_type("senna-topic"), None);
    assert_eq!(RunKind::from_model_type(""), None);
}
