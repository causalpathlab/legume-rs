use super::*;

#[test]
fn test_simulate_basic() {
    let params = ScapeSimParams {
        n_fragments: 500,
        ..Default::default()
    };
    let (frags, labels) = simulate_fragments(&params);
    assert_eq!(frags.len(), labels.len());
    assert!(frags.len() > 100, "should generate substantial fragments");

    // Check labels are within range
    for &lbl in &labels {
        assert!(lbl < params.weights.len());
    }

    // Check some junction reads exist
    let n_junction = frags.iter().filter(|f| f.is_junction).count();
    assert!(n_junction > 0, "should have junction reads");

    // Check noise reads exist
    let n_noise = labels.iter().filter(|&&l| l == 0).count();
    assert!(n_noise > 0, "should have noise reads");
}

#[test]
fn test_simulate_deterministic() {
    let params = ScapeSimParams {
        n_fragments: 200,
        seed: 123,
        ..Default::default()
    };
    let (frags1, labels1) = simulate_fragments(&params);
    let (frags2, labels2) = simulate_fragments(&params);
    assert_eq!(labels1, labels2);
    assert_eq!(frags1.len(), frags2.len());
    for (a, b) in frags1.iter().zip(frags2.iter()) {
        assert_eq!(a.x, b.x);
        assert_eq!(a.l, b.l);
    }
}
