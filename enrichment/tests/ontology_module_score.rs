//! Integration test for the descriptive GO/GMT module-score scorer
//! (`enrichment::ontology_module_score`).

use enrichment::{ontology_module_score, Mat};

#[test]
fn module_score_signature_discriminates_and_cancels_common() {
    // G × K profile (K = 3). SYMMETRIC by construction: each cluster has
    // exactly one unique high block (A→k0, B→k1, D→k2) PLUS a common block C
    // high in all three. Equal high-mass per cluster ⇒ C's per-cluster score
    // is identical ⇒ the cross-cluster contrast cancels it exactly, while a
    // unique block stands out only in its own cluster.
    let g = 200;
    let k = 3;
    // term column → (gene block, hot cluster or None for "all").
    let blocks: [(std::ops::Range<usize>, Option<usize>); 4] = [
        (0..20, Some(0)),  // A → k0
        (20..40, Some(1)), // B → k1
        (40..60, None),    // C → all (common)
        (60..80, Some(2)), // D → k2
    ];
    let mut profile = Mat::from_element(g, k, 1.0f32);
    for (range, hot) in &blocks {
        for gi in range.clone() {
            match hot {
                Some(c) => profile[(gi, *c)] = 50.0,
                None => (0..k).for_each(|c| profile[(gi, c)] = 50.0),
            }
        }
    }
    let terms: Vec<(Box<str>, Vec<usize>)> = blocks
        .iter()
        .zip(["A", "B", "C", "D"])
        .map(|((r, _), id)| (id.into(), r.clone().collect()))
        .collect();
    let universe: Vec<usize> = (0..g).collect();
    let ms = ontology_module_score(&profile, &terms, &universe).unwrap();
    let d = &ms.effect_kt; // 3 × 4 cols A,B,C,D

    // A is k0's signature (clearly its strongest cluster), B is k1's.
    assert!(d[(0, 0)] > 0.5, "A in k0 d={}", d[(0, 0)]);
    assert!(d[(0, 0)] > d[(1, 0)] + 1.0, "A discriminates k0 vs k1");
    assert!(d[(1, 1)] > 0.5, "B in k1 d={}", d[(1, 1)]);
    assert!(d[(1, 1)] > d[(0, 1)] + 1.0, "B discriminates k1 vs k0");
    // C high in all three → identical per-cluster score → contrast ≈ 0.
    for c in 0..k {
        assert!(d[(c, 2)].abs() < 1e-3, "C cancels in k{c}: d={}", d[(c, 2)]);
    }
}
