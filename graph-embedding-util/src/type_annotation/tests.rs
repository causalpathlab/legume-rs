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
    let m = top_enriched_members(&enrich, n_comm, n_types, 6);
    // community 0: positive top, strongest first
    assert_eq!(m[0], vec![0, 1]);
    // community 1: only the single positive type
    assert_eq!(m[1], vec![2]);
}

#[test]
fn top_enriched_ranks_by_enrichment_not_marker_count() {
    // Two positive types; the higher-enrichment one leads regardless of how
    // many markers each has (marker-count weighting was removed).
    let n_types = 2;
    let n_comm = 1;
    let enrich = vec![1.0, 0.8];
    let m = top_enriched_members(&enrich, n_comm, n_types, 6);
    assert_eq!(m[0], vec![0, 1]); // 1.0 > 0.8, no √|markers| flip
}

#[test]
fn top2_margin_reports_top1_minus_top2() {
    // 2 cells × 3 types: cell 0 has a clear winner, cell 1 is a near-tie.
    let score = vec![
        5.0, 1.0, 0.0, /* cell 0 */ 2.0, 1.9, 0.0, /* cell 1 */
    ];
    let m = top2_margin(&score, 2, 3);
    assert!((m[0] - 4.0).abs() < 1e-6, "clear winner margin");
    assert!((m[1] - 0.1).abs() < 1e-5, "near-tie margin");
    // A single type is always definitive (+∞ margin).
    let m1 = top2_margin(&[1.0, 2.0], 2, 1);
    assert!(m1.iter().all(|&v| v.is_infinite()));
}

#[test]
fn pnorm_upper_matches_known_quantiles() {
    assert!((pnorm_upper(0.0) - 0.5).abs() < 1e-5);
    assert!((pnorm_upper(1.96) - 0.025).abs() < 1e-3);
    assert!((pnorm_upper(-1.96) - 0.975).abs() < 1e-3);
    // monotone decreasing
    assert!(pnorm_upper(3.0) < pnorm_upper(1.0));
}

/// With `cfg.layout`, `annotate_by_projection` returns finite, non-degenerate
/// `[N×2]` UMAP and PHATE cell coordinates. (Strict cluster separation on a
/// 50-point toy SGD is unreliable; separation quality is verified end-to-end
/// on real data.)
#[test]
fn layout_produces_finite_separated_coords() {
    let h = 4;
    let per = 25;
    let n = 2 * per; // 50 cells, ≤ phate_max_direct → direct PHATE
                     // Group A points ~ +e0, group B ~ +e1, with small deterministic jitter.
    let mut cell = vec![0f32; n * h];
    for c in 0..n {
        let g = c / per; // 0 or 1
        let jit = ((c * 7) % 5) as f32 * 0.01;
        cell[c * h + g] = 1.0 + jit;
        cell[c * h + (g + 2)] = 0.05 * jit;
    }
    // Two genes: gene 0 marks group A (e0), gene 1 marks group B (e1).
    let n_feat = 2;
    let mut feat = vec![0f32; n_feat * h];
    feat[0] = 1.0; // gene 0 ~ e0
    feat[h + 1] = 1.0; // gene 1 ~ e1
    let type_markers = vec![vec![(0u32, 1.0f32)], vec![(1u32, 1.0f32)]];
    let type_names = names(&["A", "B"]);

    let cfg = AnnotateProjConfig {
        n_perm: 0,
        layout: true,
        phate: true,
        phate_max_direct: 10_000,
        knn: 10,
        ..AnnotateProjConfig::default()
    };
    let res = annotate_by_projection(
        &ProjInputs {
            feature_emb: &feat,
            n_features: n_feat,
            cell_emb: &cell,
            n_cells: n,
            type_markers: &type_markers,
            type_names: &type_names,
            h,
        },
        &cfg,
    )
    .expect("annotate_by_projection");

    let umap = res.cell_umap.expect("umap present");
    let phate = res.cell_phate.expect("phate present");
    assert_eq!(umap.len(), n * 2);
    assert_eq!(phate.len(), n * 2);
    assert!(umap.iter().all(|v| v.is_finite()), "umap finite");
    assert!(phate.iter().all(|v| v.is_finite()), "phate finite");

    // Non-degenerate: both layouts have real spread along x (not collapsed).
    let var_x = |coords: &[f32]| {
        let mean = (0..n).map(|c| coords[c * 2]).sum::<f32>() / n as f32;
        (0..n).map(|c| (coords[c * 2] - mean).powi(2)).sum::<f32>() / n as f32
    };
    assert!(
        var_x(&umap) > 1e-6,
        "UMAP degenerate (var_x={:.3e})",
        var_x(&umap)
    );
    assert!(
        var_x(&phate) > 1e-9,
        "PHATE degenerate (var_x={:.3e})",
        var_x(&phate)
    );

    // The two groups' UMAP centroids are at least distinct.
    let centroid = |lo: usize, hi: usize| {
        let (mut x, mut y) = (0.0f32, 0.0f32);
        for c in lo..hi {
            x += umap[c * 2];
            y += umap[c * 2 + 1];
        }
        let m = (hi - lo) as f32;
        (x / m, y / m)
    };
    let (ax, ay) = centroid(0, per);
    let (bx, by) = centroid(per, n);
    let between = ((ax - bx).powi(2) + (ay - by).powi(2)).sqrt();
    assert!(
        between > 1e-2,
        "UMAP group centroids coincide: {between:.4}"
    );
}
