use super::*;

fn part(labels: &[usize]) -> Partition {
    let m = labels.iter().max().map_or(0, |&x| x + 1);
    (labels.to_vec(), m)
}

/// ARI is 1 for identical partitions, and **invariant to how the clusters are named** — the two
/// Leiden runs we compare never agree on cluster *ids*, only on which cells go together.
#[test]
fn ari_ignores_the_cluster_labels() {
    let a: Vec<usize> = (0..100).map(|i| i % 4).collect();
    let relabelled: Vec<usize> = a.iter().map(|&x| (x + 2) % 4).collect();
    assert!((ari(&a, &a) - 1.0).abs() < 1e-9);
    assert!(
        (ari(&a, &relabelled) - 1.0).abs() < 1e-9,
        "a renamed partition is the same partition"
    );
}

/// Chance agreement scores ~0, not the ~0.25 raw Rand would give here. That correction is the
/// whole reason this is ARI and not accuracy: without it the medoid would drift toward whichever
/// partition had the most clusters.
#[test]
fn ari_of_independent_partitions_is_about_zero() {
    let a: Vec<usize> = (0..400).map(|i| i % 4).collect();
    let b: Vec<usize> = (0..400).map(|i| (i / 4) % 4).collect(); // orthogonal to `a` by construction
    let r = ari(&a, &b);
    assert!(
        r.abs() < 0.05,
        "independent partitions should score ~0, got {r}"
    );
}

/// **The point of the medoid.** Five partitions: four that agree closely, one wild outlier. The
/// reported partition must be one of the four, never the outlier — which is exactly the failure
/// the old code could hit, since it reported whichever draw `--seed` happened to produce.
#[test]
fn the_medoid_is_a_typical_partition_not_the_outlier() {
    let base: Vec<usize> = (0..200).map(|i| i / 50).collect(); // 4 clusters of 50
    let mut parts = Vec::new();
    for k in 0..4 {
        // Four near-copies: each moves a couple of cells across a boundary.
        let mut p = base.clone();
        p[k * 7] = (p[k * 7] + 1) % 4;
        p[k * 7 + 1] = (p[k * 7 + 1] + 1) % 4;
        parts.push(part(&p));
    }
    // The outlier: a completely different, orthogonal partition.
    let wild: Vec<usize> = (0..200).map(|i| i % 4).collect();
    parts.push(part(&wild));

    let m = medoid(&parts);
    let (best, mean_ari) = (m.best, m.agreement);
    assert!(
        best < 4,
        "the medoid must not be the outlier (got index {best})"
    );
    assert!(
        m.agreement >= m.ensemble_mean,
        "the medoid must agree with the ensemble at least as well as an average draw does"
    );
    assert!(
        mean_ari > 0.0 && mean_ari < 1.0,
        "mean ARI to the rest should be a real number in (0,1), got {mean_ari}"
    );

    // And the outlier really is the worst choice: its own mean agreement is far lower.
    let outlier_only = medoid(&[parts[4].clone(), parts[4].clone()]);
    assert!(
        (outlier_only.agreement - 1.0).abs() < 1e-9,
        "a partition agrees with itself"
    );
}

/// One partition is not a choice.
#[test]
fn a_single_partition_is_its_own_medoid() {
    let p = part(&(0..50).map(|i| i / 10).collect::<Vec<_>>());
    let m = medoid(&[p]);
    assert_eq!((m.best, m.agreement), (0, 1.0));
}

/// Every partition identical ⇒ any index is the medoid, and the agreement is perfect. (The `0/0`
/// guard in `ari` matters here: with one cluster holding everything, chance agreement *is* total
/// agreement.)
#[test]
fn identical_partitions_agree_perfectly() {
    let p = part(&(0..100).map(|i| i / 25).collect::<Vec<_>>());
    let mean = medoid(&[p.clone(), p.clone(), p.clone()]).agreement;
    assert!(
        (mean - 1.0).abs() < 1e-9,
        "identical partitions must score 1.0, got {mean}"
    );

    let one = part(&vec![0usize; 100]); // everything in a single cluster
    let mean_one = medoid(&[one.clone(), one.clone()]).agreement;
    assert!(
        (mean_one - 1.0).abs() < 1e-9,
        "the degenerate 0/0 case must be 1.0, got {mean_one}"
    );
}
