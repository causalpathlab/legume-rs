//! `n_encoder_groups` round-trips through save/load, and reads as `None` for
//! a model written before the field existed. Below that: rebuilding the
//! masked encoder's `coarse_groups` from a checkpoint's recorded group count
//! (`resolve_encoder_coarse_groups`), refused loudly rather than left to
//! `VarMap::load`; and the query-gene-axis guard on that same path
//! (`ensure_coarse_encoder_matches_query_axis`).

use super::{
    ensure_coarse_encoder_matches_query_axis, resolve_encoder_coarse_groups,
    save_coarsening_levels, save_feature_mean, TopicModelMetadata,
};
use candle_util::candle_core::Device;
use data_beans_alg::feature_coarsening::FeatureCoarsening;

fn minimal_metadata() -> TopicModelMetadata {
    TopicModelMetadata {
        model_type: "indexed_topic_masked".into(),
        decoder_types: vec!["nb".into()],
        decoder_weights: vec![1.0],
        n_features_encoder: 10,
        n_features_full: 10,
        n_topics: 3,
        encoder_hidden: vec![8],
        num_levels: 1,
        level_decoder_dims: vec![10],
        adj_method: "residual".into(),
        has_coarsening: false,
        embedding_dim: Some(4),
        enc_context_size: Some(10),
        theta_mean: None,
        n_train_cells: None,
        n_gene_modules: None,
        query_rank: None,
        n_encoder_groups: None,
    }
}

#[test]
fn n_encoder_groups_round_trips_through_save_and_load() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("run").to_string_lossy().into_owned();

    let mut m = minimal_metadata();
    m.n_encoder_groups = Some(1000);
    m.has_coarsening = true;
    m.save(&prefix).unwrap();

    let loaded = TopicModelMetadata::load(&prefix).unwrap();
    assert_eq!(loaded.n_encoder_groups, Some(1000));
    assert!(loaded.has_coarsening);
}

/// A model written before `n_encoder_groups` existed has no such key in its
/// `model.json` at all — exactly what `serde(default)` has to paper over.
#[test]
fn a_model_written_before_the_field_existed_reads_as_none() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("run").to_string_lossy().into_owned();

    let mut json = serde_json::to_value(minimal_metadata()).unwrap();
    json.as_object_mut().unwrap().remove("n_encoder_groups");
    std::fs::write(
        format!("{prefix}.model.json"),
        serde_json::to_string_pretty(&json).unwrap(),
    )
    .unwrap();

    let loaded = TopicModelMetadata::load(&prefix).unwrap();
    assert_eq!(loaded.n_encoder_groups, None);
}

/// 6 genes, 3 groups — small enough to hand-check, shared by every
/// `resolve_encoder_coarse_groups` test below.
fn coarsening_6_3() -> FeatureCoarsening {
    FeatureCoarsening {
        fine_to_coarse: vec![0, 0, 1, 1, 1, 2],
        coarse_to_fine: vec![vec![0, 1], vec![2, 3, 4], vec![5]],
        num_coarse: 3,
    }
}

/// Write a `{prefix}.coarsening.json` + `{prefix}.feature_mean.parquet` pair
/// for [`coarsening_6_3`] — the on-disk half of a coarse-trained checkpoint
/// `resolve_encoder_coarse_groups` reads back.
fn write_coarse_checkpoint(prefix: &str) {
    save_coarsening_levels(&[Some(coarsening_6_3())], prefix).unwrap();
    let genes: Vec<Box<str>> = (0..6)
        .map(|i| Box::from(format!("g{i}").as_str()))
        .collect();
    save_feature_mean(&[2.0, 1.0, 1.0, 1.0, 2.0, 4.0], &genes, prefix).unwrap();
}

/// A model whose encoder pools through a fixed coarse grouping rebuilds the
/// SAME grouping from nothing but its recorded group count, the saved
/// coarsening file, and `feature_mean` — the ingredients `predict`, `probe`
/// and `counterfactual` all have on hand.
#[test]
fn resolve_encoder_coarse_groups_rebuilds_the_saved_grouping() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("run").to_string_lossy().into_owned();
    write_coarse_checkpoint(&prefix);

    let mut m = minimal_metadata();
    m.n_features_full = 6;
    m.n_encoder_groups = Some(3);

    let map = resolve_encoder_coarse_groups(&prefix, &m, &Device::Cpu)
        .unwrap()
        .expect("Some map for a grouped encoder");
    assert_eq!(map.n_coarse(), 3);
    assert_eq!(map.n_fine(), 6);
}

/// `n_encoder_groups: None` (the shape every model was trained at before this
/// feature existed) rebuilds per gene — `None`, no coarsening file touched —
/// so an old checkpoint with no coarsening.json at all still loads.
#[test]
fn resolve_encoder_coarse_groups_is_none_for_a_per_gene_encoder() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("run").to_string_lossy().into_owned();
    let m = minimal_metadata();
    assert_eq!(m.n_encoder_groups, None);

    let out = resolve_encoder_coarse_groups(&prefix, &m, &Device::Cpu).unwrap();
    assert!(out.is_none());
}

/// `n_encoder_groups` says the encoder pools groups, but no coarsening file
/// is on disk: refuse loudly and name the prefix, rather than letting
/// `VarMap::load` fail on `enc.group.embeddings` with a raw missing-tensor
/// error.
#[test]
fn resolve_encoder_coarse_groups_refuses_a_missing_coarsening_file() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("run").to_string_lossy().into_owned();
    let mut m = minimal_metadata();
    m.n_features_full = 6;
    m.n_encoder_groups = Some(3);
    // Deliberately no `{prefix}.coarsening.json` written.

    let err = resolve_encoder_coarse_groups(&prefix, &m, &Device::Cpu)
        .map(|_| ())
        .unwrap_err()
        .to_string();
    assert!(err.contains(&prefix), "must name the model prefix: {err}");
    assert!(err.contains('3'), "must name the recorded count: {err}");
}

/// The recorded group count and the loaded grouping's actual count disagree:
/// refuse, naming both numbers, rather than trusting a stale or hand-edited
/// coarsening file.
#[test]
fn resolve_encoder_coarse_groups_refuses_a_disagreeing_group_count() {
    let dir = tempfile::tempdir().unwrap();
    let prefix = dir.path().join("run").to_string_lossy().into_owned();
    write_coarse_checkpoint(&prefix);

    let mut m = minimal_metadata();
    m.n_features_full = 6;
    m.n_encoder_groups = Some(5); // the file actually has 3

    let err = resolve_encoder_coarse_groups(&prefix, &m, &Device::Cpu)
        .map(|_| ())
        .unwrap_err()
        .to_string();
    assert!(
        err.contains('5') && err.contains('3'),
        "must name both the recorded (5) and found (3) counts: {err}"
    );
}

/// A per-gene encoder, or a query on the model's own axis, never trips the
/// gene-axis guard.
#[test]
fn query_axis_check_passes_off_the_coarse_path_or_on_a_matching_axis() {
    assert!(ensure_coarse_encoder_matches_query_axis("m", None, 6, 9).is_ok());
    assert!(ensure_coarse_encoder_matches_query_axis("m", Some(0), 6, 9).is_ok());
    assert!(ensure_coarse_encoder_matches_query_axis("m", Some(3), 6, 6).is_ok());
}

/// A coarse encoder's grouping is keyed to the model's own gene axis; a query
/// on a different-length axis is refused, naming both lengths, rather than
/// silently applying a grouping keyed to the wrong genes.
#[test]
fn query_axis_check_refuses_a_changed_axis_on_the_coarse_path() {
    let err = ensure_coarse_encoder_matches_query_axis("m", Some(3), 6, 9)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains('6') && err.contains('9'),
        "must name both axis lengths: {err}"
    );
}
