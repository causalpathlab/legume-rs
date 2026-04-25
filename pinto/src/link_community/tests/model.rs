use crate::link_community::model::*;

fn make_synthetic_profiles(n_edges: usize, m: usize, k: usize) -> (LinkProfileStore, Vec<usize>) {
    let mut profiles = vec![0.0f32; n_edges * m];
    let mut labels = vec![0usize; n_edges];

    // Create planted structure: edges in community c have higher values in gene c
    for e in 0..n_edges {
        let c = e % k;
        labels[e] = c;
        for g in 0..m {
            let base_val = 1.0;
            let signal = if g % k == c { 5.0 } else { 0.0 };
            profiles[e * m + g] = base_val + signal;
        }
    }

    (LinkProfileStore::new(profiles, n_edges, m), labels)
}

#[test]
fn test_from_profiles_basic() {
    let (store, labels) = make_synthetic_profiles(20, 6, 3);
    let stats = LinkCommunityStats::from_profiles(&store, 3, &labels);

    assert_eq!(stats.k, 3);
    assert_eq!(stats.m, 6);
    assert_eq!(stats.n_edges, 20);

    // Each community should have roughly n_edges/k edges
    for c in 0..3 {
        assert!(stats.edge_count[c] > 0);
    }
    let total_count: usize = stats.edge_count.iter().sum();
    assert_eq!(total_count, 20);
}

#[test]
fn test_delta_move_consistency() {
    let (store, labels) = make_synthetic_profiles(30, 4, 3);
    let mut stats = LinkCommunityStats::from_profiles(&store, 3, &labels);

    // Move edge 0 from community 0 to community 1
    let old_c = stats.membership[0];
    let new_c = (old_c + 1) % 3;
    stats.delta_move(0, old_c, new_c, &store);

    // Recompute from scratch and compare
    let stats_recomputed = LinkCommunityStats::from_profiles(&store, 3, &stats.membership);
    let score_delta = stats.total_score();
    let score_recomputed = stats_recomputed.total_score();

    assert!(
        (score_delta - score_recomputed).abs() < 1e-8,
        "Incremental vs recomputed: {} vs {}",
        score_delta,
        score_recomputed
    );
}

#[test]
fn test_recompute_matches() {
    let (store, labels) = make_synthetic_profiles(20, 4, 2);
    let mut stats = LinkCommunityStats::from_profiles(&store, 2, &labels);

    // Do a few moves
    stats.delta_move(0, 0, 1, &store);
    stats.delta_move(3, 1, 0, &store);
    stats.delta_move(5, 1, 0, &store);

    let score_before = stats.total_score();
    stats.recompute(&store);
    let score_after = stats.total_score();

    assert!(
        (score_before - score_after).abs() < 1e-8,
        "Recompute drift: {} vs {}",
        score_before,
        score_after
    );
}

#[test]
fn test_delta_score_matches_hand_computed_dot_product() {
    // Fast DC-SBM scoring: log_probs[t] must equal
    //   Σ_g y_eg · (log_rate[t,g] - log_rate[current_c,g])
    // where log_rate[k,g] = log_gene[k,g] + log_size_offset[k].
    let (store, labels) = make_synthetic_profiles(15, 4, 3);
    let stats = LinkCommunityStats::from_profiles(&store, 3, &labels);
    let mut log_probs = vec![0.0f64; 3];

    for e in 0..stats.n_edges {
        let current_c = stats.membership[e];
        let (cols, vals) = store.row(e);
        let sf = store.size_factors[e] as f64;
        compute_log_probs_for_edge(e, &stats, &store, None, None, &mut log_probs);
        assert!(
            log_probs[current_c].abs() < 1e-10,
            "current community must score 0"
        );
        for (t, &got) in log_probs.iter().enumerate() {
            if t == current_c {
                continue;
            }
            let mut expected = sf * (stats.log_size_offset[t] - stats.log_size_offset[current_c]);
            for (&col, &y_f32) in cols.iter().zip(vals.iter()) {
                let y = y_f32 as f64;
                let g = col as usize;
                expected +=
                    y * (stats.log_gene[t * stats.m + g] - stats.log_gene[current_c * stats.m + g]);
            }
            assert!(
                (got - expected).abs() < 1e-10,
                "e={e} t={t}: got={got:.10} expected={expected:.10}"
            );
        }
    }
}

#[test]
fn test_delta_move_keeps_caches_in_sync() {
    // After any sequence of delta_moves, log_gene[k,g] must equal
    // ln(gene_sum[k,g] + ε) and log_size_offset[k] must equal
    // -ln(size_sum[k] + M·ε).
    let (store, labels) = make_synthetic_profiles(30, 5, 3);
    let mut stats = LinkCommunityStats::from_profiles(&store, 3, &labels);
    let m_eps = (stats.m as f64) * LOG_EPS;

    // Apply a handful of moves.
    let moves: &[(usize, usize)] = &[(0, 1), (3, 2), (7, 0), (11, 2), (14, 1), (2, 0)];
    for &(e, new_c) in moves {
        let old_c = stats.membership[e];
        if old_c != new_c {
            stats.delta_move(e, old_c, new_c, &store);
        }
    }

    // Verify every cache entry matches what from-scratch would give.
    for k in 0..stats.k {
        let off_expected = -((stats.size_sum[k] + m_eps).ln());
        assert!(
            (stats.log_size_offset[k] - off_expected).abs() < 1e-12,
            "log_size_offset[{k}] drift"
        );
        for g in 0..stats.m {
            let idx = k * stats.m + g;
            let lg_expected = (stats.gene_sum[idx] + LOG_EPS).ln();
            assert!(
                (stats.log_gene[idx] - lg_expected).abs() < 1e-12,
                "log_gene[{k},{g}] drift"
            );
        }
    }

    // Incremental caches must also agree with a fresh rebuild.
    let fresh = LinkCommunityStats::from_profiles(&store, 3, &stats.membership);
    for i in 0..stats.log_gene.len() {
        assert!((stats.log_gene[i] - fresh.log_gene[i]).abs() < 1e-12);
    }
    for k in 0..stats.k {
        assert!((stats.log_size_offset[k] - fresh.log_size_offset[k]).abs() < 1e-12);
    }
}

#[test]
fn test_size_factors() {
    let profiles = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let store = LinkProfileStore::new(profiles, 2, 3);
    assert_eq!(store.size_factors[0], 6.0);
    assert_eq!(store.size_factors[1], 15.0);
}

#[test]
fn test_classifier_recovers_planted() {
    // Create planted profiles: K=3 communities with distinct gene signatures
    let k = 3;
    let m = 6;
    let n_edges = 60;

    let (store, labels) = make_synthetic_profiles(n_edges, m, k);
    let stats = LinkCommunityStats::from_profiles(&store, k, &labels);

    let classifier = LinkCommunityClassifier::from_stats(&stats);
    let predicted = classifier.predict_labels(&store);

    // Classifier should recover the planted labels exactly
    // (since it's trained on the same data)
    assert_eq!(
        predicted, labels,
        "Classifier should recover planted labels exactly"
    );
}

#[test]
fn test_classifier_generalizes() {
    // Train classifier on one set of edges, test on another
    let k = 2;
    let m = 8;
    let n_train = 100;
    let n_test = 50;

    // Training: planted profiles
    let mut train_profiles = vec![0.0f32; n_train * m];
    let mut train_labels = vec![0usize; n_train];
    for e in 0..n_train {
        let c = e % k;
        train_labels[e] = c;
        for g in 0..m {
            let signal = if (g < m / 2) == (c == 0) { 10.0 } else { 1.0 };
            train_profiles[e * m + g] = signal;
        }
    }
    let train_store = LinkProfileStore::new(train_profiles, n_train, m);
    let stats = LinkCommunityStats::from_profiles(&train_store, k, &train_labels);

    let classifier = LinkCommunityClassifier::from_stats(&stats);

    // Test: similar profiles with noise
    let mut test_profiles = vec![0.0f32; n_test * m];
    let mut test_labels = vec![0usize; n_test];
    for e in 0..n_test {
        let c = e % k;
        test_labels[e] = c;
        for g in 0..m {
            let signal = if (g < m / 2) == (c == 0) { 8.0 } else { 2.0 };
            test_profiles[e * m + g] = signal;
        }
    }
    let test_store = LinkProfileStore::new(test_profiles, n_test, m);

    let predicted = classifier.predict_labels(&test_store);
    let correct: usize = predicted
        .iter()
        .zip(test_labels.iter())
        .filter(|(&p, &t)| p == t)
        .count();

    assert!(
        correct == n_test,
        "Classifier should generalize to noisy test data: {}/{}",
        correct,
        n_test
    );
}

#[test]
fn test_classifier_parallel_matches_sequential() {
    let k = 3;
    let m = 6;
    let n_edges = 120;

    let (store, labels) = make_synthetic_profiles(n_edges, m, k);
    let stats = LinkCommunityStats::from_profiles(&store, k, &labels);
    let classifier = LinkCommunityClassifier::from_stats(&stats);

    let seq = classifier.predict_labels(&store);
    let par = classifier.predict_labels_parallel(&store);

    assert_eq!(seq, par, "Parallel predictions should match sequential");
}
