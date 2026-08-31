//! Ablation is the flag that turns the score from a reconstruction into a
//! prediction, so its gate has to be exact: a feature named for hiding must
//! leave the encoder's view, and nothing else may move.
//!
//! Driven through `build_remap` rather than the hiding helper directly, because
//! the ordering is half the contract — hiding must happen AFTER the coverage
//! gate, or every ablated run is refused for "missing" the genes it withheld on
//! purpose.

use super::*;

fn names(v: &[&str]) -> Vec<Box<str>> {
    v.iter().map(|s| Box::from(*s)).collect()
}

fn opts_hiding(hidden: &[&str], min_overlap: f32) -> QueryNameOpts {
    QueryNameOpts {
        min_overlap,
        hide: Some(std::sync::Arc::new(
            hidden.iter().map(|s| Box::from(*s)).collect(),
        )),
        ..Default::default()
    }
}

#[test]
fn ablation_hides_exactly_the_named_features() {
    let genes = names(&["a", "b", "c", "d"]);
    // Axes match, so the remap would normally be `None`; hiding forces an
    // identity one, and only the named rows differ from it.
    let out = build_remap(&genes, &genes, &opts_hiding(&["b", "d"], 0.0))
        .expect("remap")
        .expect("hiding always yields a remap");
    assert_eq!(out.new_to_train, vec![Some(0), None, Some(2), None]);
    assert_eq!(out.n_mapped, 2);
}

#[test]
fn hiding_survives_a_real_axis_mismatch() {
    // Query carries a gene the model lacks; hiding must not resurrect it or
    // renumber the survivors.
    let training = names(&["a", "b", "c"]);
    let query = names(&["a", "zzz", "b", "c"]);
    let out = build_remap(&training, &query, &opts_hiding(&["c"], 0.0))
        .expect("remap")
        .expect("hiding always yields a remap");
    assert_eq!(out.new_to_train[1], None, "unmatched gene stays unmatched");
    assert_eq!(out.new_to_train[3], None, "named gene is hidden");
    assert_eq!(out.n_mapped, 2);
}

#[test]
fn a_name_that_matches_nothing_is_an_error_not_a_silent_reconstruction() {
    // The failure this guards: a typo'd file leaves every gene visible, the run
    // succeeds, and the reported number is a plain reconstruction wearing the
    // ablation's name.
    let genes = names(&["a", "b"]);
    assert!(build_remap(&genes, &genes, &opts_hiding(&["zzz"], 0.0)).is_err());
}

#[test]
fn hiding_every_feature_is_an_error() {
    let genes = names(&["a", "b"]);
    assert!(build_remap(&genes, &genes, &opts_hiding(&["a", "b"], 0.0)).is_err());
}

#[test]
fn coverage_is_gated_before_hiding_not_after() {
    // Hiding half the axis must not be read as half the axis going missing.
    // Ordering it the other way refuses every ablated run under any real
    // --min-gene-overlap.
    let genes = names(&["a", "b", "c", "d"]);
    let out = build_remap(&genes, &genes, &opts_hiding(&["a", "b"], 0.9))
        .expect("a 90% floor must still pass: nothing is missing, two are withheld")
        .expect("hiding always yields a remap");
    assert_eq!(out.n_mapped, 2);
}

/// The panel file and the data may disagree on case while naming the same
/// genes. The remap matches lowercased, so the model resolves fine — but the
/// hide set used to match exactly, so a lowercase panel against uppercase rows
/// hid nothing and errored with "matched no feature", pointing at the wrong
/// cause entirely.
#[test]
fn hiding_matches_case_insensitively_like_the_remap_does() {
    let genes = names(&["Cd8a", "GZMB", "ms4a1"]);
    let out = build_remap(&genes, &genes, &opts_hiding(&["CD8A", "Ms4a1"], 0.0))
        .expect("case must not defeat the hide")
        .expect("hiding always yields a remap");
    assert_eq!(out.new_to_train, vec![None, Some(1), None]);
}

/// The scoring cap exists for one reason: a dense block's working set must not
/// be multiplied by every thread. It has to bite at whole-transcriptome width
/// and stay out of the way at the coarsened widths the topic paths score on,
/// which have never had a memory problem.
mod block_concurrency {
    use super::super::dense_block_concurrency;

    #[test]
    fn a_whole_transcriptome_block_is_capped_well_below_the_thread_count() {
        // The reported OOM's shape: ~58k genes at the default minibatch.
        let conc = dense_block_concurrency(500, 57_843);
        assert!(
            conc <= 8,
            "58k-gene blocks must not run wide open; got {conc}"
        );
        assert!(conc >= 1, "the cap must always admit at least one block");
    }

    #[test]
    fn a_coarsened_block_is_not_throttled() {
        // What a dense topic model actually scores on after coarsening: the
        // cap must return the full thread count, i.e. change nothing.
        let conc = dense_block_concurrency(500, 2_000);
        assert_eq!(conc, rayon::current_num_threads());
    }

    #[test]
    fn an_absurd_width_still_admits_one_block() {
        assert_eq!(dense_block_concurrency(usize::MAX, usize::MAX), 1);
    }
}
