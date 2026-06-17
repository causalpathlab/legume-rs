//! Unit tests for the private helpers in [`super`] (the `type_annotation`
//! module). Kept in a sibling file; `use super::*` still reaches the private
//! items since this is a child module of `type_annotation`.

use super::*;

fn names(v: &[&str]) -> Vec<Box<str>> {
    v.iter().map(|s| Box::from(*s)).collect()
}

#[test]
fn lexical_label_shared_prefix_and_suffix() {
    let tn = names(&[
        "CD8 Naive",
        "CD8 Effector_1",
        "CD8 Memory", // 0,1,2
        "Naive B",
        "Memory B",
        "pre B", // 3,4,5
        "CD14 Mono",
        "CD16 Mono", // 6,7
        "Platelet",
        "NK", // 8,9 (no overlap pair)
    ]);
    // shared leading token
    assert_eq!(lexical_label(&[0, 1, 2], &tn).as_ref(), "CD8");
    // shared trailing token
    assert_eq!(lexical_label(&[3, 4, 5], &tn).as_ref(), "B");
    // shared mid token
    assert_eq!(lexical_label(&[6, 7], &tn).as_ref(), "Mono");
    // no shared token → representative (first/most-enriched) member
    assert_eq!(lexical_label(&[8, 9], &tn).as_ref(), "Platelet");
    // singleton keeps its own name
    assert_eq!(lexical_label(&[4], &tn).as_ref(), "Memory B");
}

#[test]
fn merge_map_groups_by_peak_community() {
    // 3 fine types, 2 communities. enrich[k*C + t]:
    // types 0,1 peak in community 0; type 2 peaks in community 1.
    let n_types = 3;
    let n_comm = 2;
    let enrich = vec![
        // comm 0:
        2.0, 1.5, -1.0, // comm 1:
        0.1, 0.2, 3.0,
    ];
    let sizes = vec![100, 100];
    let cof = build_merge_map(&enrich, &sizes, n_comm, n_types);
    assert_eq!(cof, vec![0, 0, 1]);
}

#[test]
fn merge_map_size_weight_resists_tiny_noisy_community() {
    // type 0: modest enrichment in big community 0, noisy-high in tiny
    // community 1. Size weighting should keep it in community 0.
    let n_types = 1;
    let n_comm = 2;
    let enrich = vec![1.0 /*comm0*/, 3.0 /*comm1*/];
    let sizes = vec![2000, 5];
    let cof = build_merge_map(&enrich, &sizes, n_comm, n_types);
    assert_eq!(cof, vec![0]); // 1.0*√2000 ≫ 3.0*√5
}

#[test]
fn top_enriched_members_picks_positive_top() {
    let n_types = 3;
    let n_comm = 2;
    let enrich = vec![
        // comm 0: types 0,1 positive (0 strongest), 2 negative
        2.0, 1.5, -1.0, // comm 1: only type 2 positive
        -0.5, -0.2, 3.0,
    ];
    let w = vec![1.0, 1.0, 1.0];
    let m = top_enriched_members(&enrich, &w, n_comm, n_types, 6);
    // community 0: positive top, strongest first
    assert_eq!(m[0], vec![0, 1]);
    // community 1: only the single positive type
    assert_eq!(m[1], vec![2]);
}

#[test]
fn top_enriched_weight_favors_larger_marker_set() {
    // one community, two positive types: type 0 slightly higher raw enrich,
    // but type 1 has a much bigger marker set → weight flips the top pick.
    let n_types = 2;
    let n_comm = 1;
    let enrich = vec![1.0, 0.8];
    let weight = vec![1.0, 3.0]; // √1 vs √9
    let m = top_enriched_members(&enrich, &weight, n_comm, n_types, 6);
    assert_eq!(m[0], vec![1, 0]); // 0.8*3=2.4 > 1.0*1
}

#[test]
fn pnorm_upper_matches_known_quantiles() {
    assert!((pnorm_upper(0.0) - 0.5).abs() < 1e-5);
    assert!((pnorm_upper(1.96) - 0.025).abs() < 1e-3);
    assert!((pnorm_upper(-1.96) - 0.975).abs() < 1e-3);
    // monotone decreasing
    assert!(pnorm_upper(3.0) < pnorm_upper(1.0));
}
