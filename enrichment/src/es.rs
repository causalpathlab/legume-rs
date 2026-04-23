//! Rank transformation and weighted KS enrichment score walker.

/// Returns an index order that sorts `scores` in descending order
/// (stable on ties). The returned `Vec<u32>` is a *permutation of indices* —
/// position `i` in the output holds the original gene index that ranks `i`-th.
pub fn rank_descending(scores: &[f32]) -> Vec<u32> {
    let n = scores.len();
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_by(|&a, &b| {
        scores[b as usize]
            .partial_cmp(&scores[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    order
}

/// Weighted KS enrichment score over a ranked gene list.
///
/// * `ranked_indices` — indices of genes in descending-specificity order.
///   `ranked_indices[0]` is the most-specific gene for this topic.
/// * `hit_weight` — per-gene marker weight (length = number of genes).
///   `hit_weight[g] == 0` ⇒ gene `g` is a non-marker (miss).
///   `hit_weight[g] > 0`  ⇒ gene `g` is a marker; the value is its
///   importance (e.g. IDF weight `ln(C / c_g)` from a marker DB so genes
///   listed by many celltypes contribute less than highly-specific markers).
///
/// Hit step at marker `g`: `+ hit_weight[g] / Σ_{g' marker} hit_weight[g']`.
/// Miss step at non-marker: `− 1 / (#non-markers)`.
/// Returns the signed maximum absolute deviation of the running sum.
pub fn weighted_ks_es(ranked_indices: &[u32], hit_weight: &[f32]) -> f32 {
    // Single pass over hit_weight: total mass + number of markers.
    let mut total_hit: f64 = 0.0;
    let mut n_hit: usize = 0;
    for &w in hit_weight {
        let w = w.max(0.0);
        if w > 0.0 {
            total_hit += w as f64;
            n_hit += 1;
        }
    }
    let n_miss = ranked_indices.len().saturating_sub(n_hit) as f64;

    if total_hit <= 0.0 || n_miss <= 0.0 {
        return 0.0;
    }

    let hit_norm = 1.0 / total_hit;
    let miss_norm = 1.0 / n_miss;

    let mut running: f64 = 0.0;
    let mut max_dev: f64 = 0.0;
    let mut max_abs: f64 = 0.0;

    for &g in ranked_indices {
        let w = hit_weight[g as usize].max(0.0) as f64;
        if w > 0.0 {
            running += w * hit_norm;
        } else {
            running -= miss_norm;
        }
        let abs = running.abs();
        if abs > max_abs {
            max_abs = abs;
            max_dev = running;
        }
    }

    max_dev as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_descending_orders_correctly() {
        let scores = vec![0.1, 0.9, 0.3, 0.7];
        let order = rank_descending(&scores);
        assert_eq!(order, vec![1, 3, 2, 0]);
    }

    #[test]
    fn rank_descending_handles_ties_stably() {
        let scores = vec![0.5, 0.5, 0.5];
        let order = rank_descending(&scores);
        assert_eq!(order.len(), 3);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2]);
    }

    fn binary_hit(g: usize, marker_idx: &[usize]) -> Vec<f32> {
        let mut hw = vec![0.0; g];
        for &i in marker_idx {
            hw[i] = 1.0;
        }
        hw
    }

    #[test]
    fn es_all_markers_at_top_gives_max_positive() {
        let ranked: Vec<u32> = (0..10).collect();
        let hit = binary_hit(10, &[0, 1, 2]);
        let es = weighted_ks_es(&ranked, &hit);
        assert!(es > 0.9, "expected near-1 ES, got {es}");
    }

    #[test]
    fn es_all_markers_at_bottom_gives_negative() {
        let ranked: Vec<u32> = (0..10).collect();
        let hit = binary_hit(10, &[7, 8, 9]);
        let es = weighted_ks_es(&ranked, &hit);
        assert!(es < -0.5, "expected strongly negative ES, got {es}");
    }

    #[test]
    fn es_uniformly_spread_markers_gives_small_magnitude() {
        let ranked: Vec<u32> = (0..10).collect();
        let hit = binary_hit(10, &[0, 3, 6, 9]);
        let es = weighted_ks_es(&ranked, &hit);
        assert!(
            es.abs() < 0.3,
            "expected small ES for uniform markers, got {es}"
        );
    }

    #[test]
    fn es_empty_marker_set_returns_zero() {
        let ranked: Vec<u32> = (0..10).collect();
        let hit = vec![0.0; 10];
        let es = weighted_ks_es(&ranked, &hit);
        assert_eq!(es, 0.0);
    }

    #[test]
    fn es_idf_weighted_concentrates_on_specific_markers() {
        // 10 genes: 3 are markers (positions 5, 6, 9). The first two are
        // "generic" markers (low IDF, e.g. listed by many celltypes); the
        // last is a "specific" marker (high IDF). The specific marker is
        // the most informative and is at the bottom of the ranking — under
        // unweighted KS, a "wide spread" → modest negative ES; under IDF
        // weighting that emphasizes the specific marker, ES is dominated
        // by its (later) hit and is even more negative.
        let ranked: Vec<u32> = (0..10).collect();
        let mut hit_unweighted = vec![0.0; 10];
        hit_unweighted[5] = 1.0;
        hit_unweighted[6] = 1.0;
        hit_unweighted[9] = 1.0;
        let mut hit_idf = vec![0.0; 10];
        hit_idf[5] = 0.1; // generic (e.g. ln(C/many) small)
        hit_idf[6] = 0.1;
        hit_idf[9] = 5.0; // specific (e.g. ln(C/1) large)
        let es_uw = weighted_ks_es(&ranked, &hit_unweighted);
        let es_w = weighted_ks_es(&ranked, &hit_idf);
        // IDF version is dominated by gene 9 → walk dips much more before
        // recovering at rank 9.
        assert!(
            es_w < es_uw,
            "IDF version should be more negative: uw={es_uw}, w={es_w}"
        );
    }
}
