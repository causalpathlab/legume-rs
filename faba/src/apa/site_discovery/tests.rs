use super::*;
use genomic_data::sam::{CellBarcode, UmiBarcode};

fn make_junction_frag(pa_pos: f32) -> FragmentRecord {
    FragmentRecord {
        x: pa_pos - 200.0,
        l: 200.0,
        r: 50.0,
        is_junction: true,
        pa_site: Some(pa_pos),
        cell_barcode: CellBarcode::Missing,
        umi: UmiBarcode::Missing,
    }
}

#[test]
fn test_bisect_splits_two_clusters() {
    // Two tight masses at ~100 and ~600 (gap 500 ≫ min_gap=50).
    let mut xs: Vec<f32> = Vec::new();
    for i in 0..40 {
        xs.push(100.0 + (i % 5) as f32);
        xs.push(600.0 + (i % 5) as f32);
    }
    xs.sort_by(f32::total_cmp);
    let sites = discover_sites_bisect(&xs, 50.0, 10);
    assert_eq!(sites.len(), 2, "expected 2 clusters, got {sites:?}");
    assert_eq!(sites.iter().map(|s| s.1).sum::<usize>(), xs.len());
    let mut pos: Vec<f32> = sites.iter().map(|s| s.0).collect();
    pos.sort_by(f32::total_cmp);
    assert!((pos[0] - 102.0).abs() < 5.0 && (pos[1] - 602.0).abs() < 5.0);
}

#[test]
fn test_bisect_single_cluster_and_min_count() {
    // One tight mass → one site (no interior gap ≥ min_gap).
    let mut xs: Vec<f32> = (0..30).map(|i| 300.0 + (i % 7) as f32).collect();
    xs.sort_by(f32::total_cmp);
    assert_eq!(discover_sites_bisect(&xs, 50.0, 10).len(), 1);
    // A tiny 2nd mass (< min_count) must NOT spawn its own site.
    let mut xs2 = xs.clone();
    xs2.extend([900.0, 901.0, 902.0]); // only 3 reads, min_count=10
    xs2.sort_by(f32::total_cmp);
    assert_eq!(discover_sites_bisect(&xs2, 50.0, 10).len(), 1);
    // Below min_count total → no sites.
    assert!(discover_sites_bisect(&[100.0, 101.0], 50.0, 10).is_empty());
}

#[test]
fn test_bisect_min_count_zero_no_panic() {
    // min_count == 0 must not panic (the old bound indexed sorted[n]); it
    // resolves as if each cluster needs ≥ 1 read.
    assert!(discover_sites_bisect(&[], 50.0, 0).is_empty());
    assert_eq!(discover_sites_bisect(&[300.0], 50.0, 0).len(), 1);
    let mut xs: Vec<f32> = (0..20).map(|i| 300.0 + (i % 5) as f32).collect();
    xs.extend([900.0, 901.0]); // a well-separated 2nd mass
    xs.sort_by(f32::total_cmp);
    assert_eq!(discover_sites_bisect(&xs, 50.0, 0).len(), 2);
}

#[test]
fn test_bisect_deep_no_stack_overflow() {
    // Many well-separated clusters used to recurse one level per cluster; the
    // iterative version must handle it without overflowing the stack.
    let xs: Vec<f32> = (0..20_000).map(|i| i as f32 * 1000.0).collect();
    assert_eq!(discover_sites_bisect(&xs, 50.0, 1).len(), 20_000);
}

#[test]
fn test_junction_site_discovery() {
    let mut fragments = Vec::new();
    // 50 junction reads near alpha=300 (scattered ±10bp)
    for i in 0..50 {
        let offset = (i % 5) as f32 - 2.0; // -2..+2
        fragments.push(make_junction_frag(300.0 + offset));
    }
    // 50 junction reads near alpha=700
    for i in 0..50 {
        let offset = (i % 5) as f32 - 2.0;
        fragments.push(make_junction_frag(700.0 + offset));
    }

    let sites = discover_sites_from_junctions(&fragments, 5);
    assert!(
        sites.len() >= 2,
        "should discover at least 2 site clusters, got {}",
        sites.len()
    );

    // Check that sites are within 20bp of true positions
    let has_300 = sites.iter().any(|&s| (s - 300.0).abs() < 20.0);
    let has_700 = sites.iter().any(|&s| (s - 700.0).abs() < 20.0);
    assert!(has_300, "should find site near 300, got {:?}", sites);
    assert!(has_700, "should find site near 700, got {:?}", sites);
}

#[test]
fn test_merge_nearby_sites() {
    // Sites clustered at 300, 305, 310 + site at 700
    let sites = vec![300.0, 305.0, 310.0, 700.0];

    // Create fragments to give counts
    let mut fragments = Vec::new();
    for &pos in &[300.0, 305.0, 310.0] {
        for _ in 0..10 {
            fragments.push(make_junction_frag(pos));
        }
    }
    for _ in 0..20 {
        fragments.push(make_junction_frag(700.0));
    }

    let merged = merge_nearby_sites(&sites, &fragments, 20.0);
    assert_eq!(merged.len(), 2, "should merge to 2 sites, got {:?}", merged);

    let has_near_300 = merged.iter().any(|&s| (s - 305.0).abs() < 15.0);
    let has_near_700 = merged.iter().any(|&s| (s - 700.0).abs() < 5.0);
    assert!(has_near_300, "should have merged site near 300-310");
    assert!(has_near_700, "should have site at 700");
}
