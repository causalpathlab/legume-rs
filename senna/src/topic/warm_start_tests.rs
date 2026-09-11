//! What a continued fit may and may not change about the feature side.

use super::check_gene_modules;
use crate::topic::model_metadata::TopicModelMetadata;

/// `M` decides which weights the encoder registers at all: at `0` a free
/// `[D, H]` table, above it a `[D, M]` membership and an `[M, H]` dictionary.
/// Crossing that switch, in either direction, is a different model wearing the
/// same checkpoint's name.
#[test]
fn a_checkpoint_cannot_cross_the_free_composed_switch() {
    assert!(check_gene_modules(0, 0).is_ok());
    assert!(check_gene_modules(32, 32).is_ok());

    let err = check_gene_modules(0, 32).unwrap_err().to_string();
    assert!(err.contains("--gene-modules"), "{err}");
    assert!(check_gene_modules(32, 0).is_err());
}

/// A model written before the field existed carries no count, and reads as the
/// free table it was trained as — so it meets the check like any other.
#[test]
fn a_model_from_before_modules_reads_as_a_free_table() {
    let json = r#"{"model_type":"indexed_topic","decoder_types":["nb"],"decoder_weights":[1.0],
        "n_features_encoder":4,"n_features_full":4,"n_topics":2,"encoder_hidden":[8],
        "num_levels":1,"level_decoder_dims":[4],"adj_method":"batch","has_coarsening":false}"#;
    let metadata: TopicModelMetadata = serde_json::from_str(json).expect("older metadata");
    assert_eq!(metadata.gene_modules(), 0);
    assert!(check_gene_modules(metadata.gene_modules(), 0).is_ok());
}

/// A different module count is not added capacity either: sparsemax gives an
/// appended module no gradient, so `M` moves by splitting a loaded one, which
/// is not something `--init-from` does on its own.
#[test]
fn a_different_module_count_is_refused_by_the_number() {
    let err = check_gene_modules(32, 64).unwrap_err().to_string();
    assert!(err.contains("32") && err.contains("64"), "{err}");
    // Not the same refusal as crossing zero: this one is a missing capability,
    // and the message has to say so rather than imply M is frozen for good.
    assert!(err.contains("not implemented yet"), "{err}");
    let crossing = check_gene_modules(0, 32).unwrap_err().to_string();
    assert!(!crossing.contains("not implemented yet"), "{crossing}");
}

/// The encoder's input is the composed embedding followed by each module's
/// level and coverage, so widening `H` inserts columns in the MIDDLE of a
/// trained weight. The corner copy every other growth uses would slide the
/// module halves onto the wrong inputs, so the combination is refused by name.
#[test]
fn widening_the_embedding_does_not_compose_with_modules() {
    use super::check_embedding_growth;
    assert!(
        check_embedding_growth(0, 16).is_ok(),
        "a free table appends"
    );
    assert!(check_embedding_growth(32, 0).is_ok(), "not growing is fine");
    let err = check_embedding_growth(32, 16).unwrap_err().to_string();
    assert!(
        err.contains("--add-embedding-dim") && err.contains("--gene-modules"),
        "{err}"
    );
}
