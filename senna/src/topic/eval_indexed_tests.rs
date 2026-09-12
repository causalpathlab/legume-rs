//! Which read a masked checkpoint takes, decided from its metadata alone.

use super::MaskedRead;

/// A model trained window-free records no context size, and reads dense.
#[test]
fn no_recorded_window_reads_dense() {
    let read = MaskedRead::resolve(None, None).expect("a window-free model needs nothing else");
    assert!(
        matches!(read, MaskedRead::Dense),
        "a model with no recorded window must read every gene"
    );
    // A stale shortlist alongside a window-free model does not resurrect the
    // window: the metadata decides, not what happens to be on disk.
    let w = vec![1.0f32; 4];
    let read = MaskedRead::resolve(None, Some(&w)).unwrap();
    assert!(matches!(read, MaskedRead::Dense));
}

/// An OLD model records its window and keeps scoring exactly as it did: top-K
/// per cell, ranked by the shortlist weights it was trained with.
#[test]
fn a_recorded_window_reads_the_indexed_path() {
    let w = vec![0.5f32, 1.0, 2.0, 0.25];
    match MaskedRead::resolve(Some(3), Some(&w)).expect("an old model loads") {
        MaskedRead::Windowed {
            context_size,
            shortlist_weights,
        } => {
            assert_eq!(context_size, 3);
            assert_eq!(shortlist_weights, w.as_slice());
        }
        MaskedRead::Dense => panic!("a recorded window must take the indexed path"),
    }
}

/// A recorded window with no weights on disk is a broken model, not a reason to
/// silently read it some other way: scoring it dense would be a different model.
#[test]
fn a_recorded_window_without_its_weights_is_refused_by_name() {
    let msg = MaskedRead::resolve(Some(3), None)
        .expect_err("a windowed model cannot be read without its shortlist")
        .to_string();
    assert!(
        msg.contains("shortlist_weights"),
        "the error must name the missing file; got: {msg}"
    );
}

/// `enc_context_size` is what decides the read, so it has to survive the round
/// trip through `model.json` — a `None` that came back as anything else would
/// score a window-free model as if it had a window, and the other way round.
#[test]
fn the_field_that_decides_the_read_round_trips() {
    use crate::topic::model_metadata::TopicModelMetadata;

    let base = TopicModelMetadata {
        model_type: crate::topic::model_metadata::MODEL_TYPE_MASKED_VAE.into(),
        decoder_types: vec!["nb".into()],
        decoder_weights: vec![1.0],
        n_features_encoder: 6,
        n_features_full: 6,
        n_topics: 2,
        encoder_hidden: vec![8],
        num_levels: 1,
        level_decoder_dims: vec![6],
        adj_method: "residual".into(),
        has_coarsening: false,
        embedding_dim: Some(4),
        enc_context_size: None,
        theta_mean: None,
        n_train_cells: None,
        n_gene_modules: None,
        query_rank: None,
    };
    let dir = tempfile::tempdir().unwrap();

    let dense = dir.path().join("dense").to_string_lossy().into_owned();
    base.save(&dense).unwrap();
    let back = TopicModelMetadata::load(&dense).unwrap();
    assert_eq!(
        back.enc_context_size, None,
        "a window-free model records none"
    );
    assert!(matches!(
        MaskedRead::resolve(back.enc_context_size, None).unwrap(),
        MaskedRead::Dense
    ));

    let windowed = dir.path().join("windowed").to_string_lossy().into_owned();
    let mut m = base;
    m.enc_context_size = Some(512);
    m.save(&windowed).unwrap();
    assert_eq!(
        TopicModelMetadata::load(&windowed)
            .unwrap()
            .enc_context_size,
        Some(512),
        "an old model's window must come back as itself"
    );
}
