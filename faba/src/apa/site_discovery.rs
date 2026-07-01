use crate::apa::fragment::FragmentRecord;
use crate::mixture::kernel_smooth::*;
use rustc_hash::FxHashMap as HashMap;

/// Discover candidate pA sites from junction reads in fragment records.
/// Returns sorted positions (UTR-relative) with counts above min_coverage.
pub fn discover_sites_from_junctions(
    fragments: &[FragmentRecord],
    min_coverage: usize,
) -> Vec<f32> {
    let mut site_counts: HashMap<i64, usize> = HashMap::default();

    for frag in fragments {
        if let Some(pa_pos) = frag.pa_site {
            let binned = pa_pos.round() as i64;
            *site_counts.entry(binned).or_default() += 1;
        }
    }

    let mut sites: Vec<f32> = site_counts
        .into_iter()
        .filter(|(_, count)| *count >= min_coverage)
        .map(|(pos, _)| pos as f32)
        .collect();

    sites.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sites
}

/// Discover candidate pA sites from coverage modes (kernel smoothing).
/// Fallback when junction reads are absent or insufficient.
/// Uses read 3'-end positions (which pile up near pA sites) to find peaks.
pub fn discover_sites_from_coverage(
    fragments: &[FragmentRecord],
    utr_length: f32,
    bandwidth: f32,
) -> Vec<f32> {
    if fragments.is_empty() || utr_length <= 0.0 {
        return Vec::new();
    }

    // Collect fragment end positions (x + l = approximate 3' end in UTR coords)
    let end_positions: Vec<f32> = fragments.iter().map(|f| f.x + f.l).collect();

    // Create coverage histogram at 1bp resolution (or 10bp for speed)
    let resolution = 10.0;
    let (hist_x, hist_y) = coverage_histogram(&end_positions, utr_length, resolution);

    // Smooth with Gaussian kernel
    let smoothed = gaussian_kernel_smooth(&hist_x, &hist_y, &hist_x, bandwidth);

    // Find modes (peaks)
    let mode_indices = find_modes(&smoothed);

    // Convert mode indices back to UTR positions
    let mut sites: Vec<f32> = mode_indices
        .into_iter()
        .filter(|&i| smoothed[i] > 0.0)
        .map(|i| hist_x[i])
        .collect();

    sites.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sites
}

/// Merge nearby candidate sites within a given distance.
/// Keeps the site with the highest count as the representative.
pub fn merge_nearby_sites(
    sites: &[f32],
    fragments: &[FragmentRecord],
    merge_dist: f32,
) -> Vec<f32> {
    if sites.is_empty() {
        return Vec::new();
    }

    // Count junction reads per site (or approximate from fragment ends)
    let mut site_counts: HashMap<i64, usize> = HashMap::default();
    for frag in fragments {
        if let Some(pa_pos) = frag.pa_site {
            let binned = pa_pos.round() as i64;
            *site_counts.entry(binned).or_default() += 1;
        } else {
            // Use fragment 3'-end position for non-junction reads
            let end_pos = (frag.x + frag.l).round() as i64;
            *site_counts.entry(end_pos).or_default() += 1;
        }
    }

    let mut merged = Vec::new();
    let mut i = 0;

    while i < sites.len() {
        let cluster_start = i;
        let mut best_pos = sites[i];
        let mut best_count = nearby_count(&site_counts, sites[i], merge_dist / 2.0);

        // Extend cluster while sites are within merge_dist
        while i + 1 < sites.len() && sites[i + 1] - sites[cluster_start] <= merge_dist {
            i += 1;
            let count = nearby_count(&site_counts, sites[i], merge_dist / 2.0);
            if count > best_count {
                best_count = count;
                best_pos = sites[i];
            }
        }

        merged.push(best_pos);
        i += 1;
    }

    merged
}

/// Discover pA-site clusters by **recursive mass bisection** — no histogram, no
/// kernel smoothing. Given the sorted read 3'-end / poly-A positions, split at
/// the largest interior gap that still leaves `min_count` reads on each side,
/// and recurse ("divide the mass until it can't be split"). A cluster is a leaf
/// when no such gap ≥ `min_gap` exists. Returns each leaf's `(median position,
/// read count)`. One sort up front (caller) + O(n) work per recursion level over
/// O(#sites) levels ⇒ effectively O(n log n), replacing the O(bins²)-ish KDE
/// fallback + the per-site `merge_nearby_sites` scan for the fast-PDUI path.
pub fn discover_sites_bisect(sorted: &[f32], min_gap: f32, min_count: usize) -> Vec<(f32, usize)> {
    let n = sorted.len();
    if n < min_count {
        return Vec::new();
    }
    // Interior split index i (between sorted[i] and sorted[i+1]) with the largest
    // gap, keeping ≥ min_count on each side. `lo..hi` enforces that balance.
    let lo = min_count.saturating_sub(1);
    let hi = n.saturating_sub(min_count);
    let mut best_gap = min_gap;
    let mut best_i: Option<usize> = None;
    for i in lo..hi {
        let gap = sorted[i + 1] - sorted[i];
        if gap > best_gap {
            best_gap = gap;
            best_i = Some(i);
        }
    }
    match best_i {
        None => vec![(sorted[n / 2], n)], // leaf: one site (median representative)
        Some(i) => {
            let mut left = discover_sites_bisect(&sorted[..=i], min_gap, min_count);
            left.extend(discover_sites_bisect(&sorted[i + 1..], min_gap, min_count));
            left
        }
    }
}

/// Count fragments near a position within a window.
fn nearby_count(counts: &HashMap<i64, usize>, pos: f32, window: f32) -> usize {
    let lo = (pos - window).round() as i64;
    let hi = (pos + window).round() as i64;
    (lo..=hi).filter_map(|p| counts.get(&p)).sum()
}

#[cfg(test)]
mod tests {
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
}
