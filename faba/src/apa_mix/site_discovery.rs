use crate::apa_mix::fragment::FragmentRecord;
use crate::apa_mix::kernel_smooth::*;
use fnv::FnvHashMap as HashMap;

/// Discover candidate pA sites from junction reads in fragment records.
/// Returns sorted positions (UTR-relative) with counts above min_coverage.
pub fn discover_sites_from_junctions(
    fragments: &[FragmentRecord],
    min_coverage: usize,
) -> Vec<f64> {
    let mut site_counts: HashMap<i64, usize> = HashMap::default();

    for frag in fragments {
        if let Some(pa_pos) = frag.pa_site {
            let binned = pa_pos.round() as i64;
            *site_counts.entry(binned).or_default() += 1;
        }
    }

    let mut sites: Vec<f64> = site_counts
        .into_iter()
        .filter(|(_, count)| *count >= min_coverage)
        .map(|(pos, _)| pos as f64)
        .collect();

    sites.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sites
}

/// Discover candidate pA sites from coverage modes (kernel smoothing).
/// Fallback when junction reads are absent or insufficient.
/// Uses read 3'-end positions (which pile up near pA sites) to find peaks.
pub fn discover_sites_from_coverage(
    fragments: &[FragmentRecord],
    utr_length: f64,
    bandwidth: f64,
) -> Vec<f64> {
    if fragments.is_empty() || utr_length <= 0.0 {
        return Vec::new();
    }

    // Collect fragment end positions (x + l = approximate 3' end in UTR coords)
    let end_positions: Vec<f64> = fragments.iter().map(|f| f.x + f.l).collect();

    // Create coverage histogram at 1bp resolution (or 10bp for speed)
    let resolution = 10.0;
    let (hist_x, hist_y) = coverage_histogram(&end_positions, utr_length, resolution);

    // Smooth with Gaussian kernel
    let smoothed = gaussian_kernel_smooth(&hist_x, &hist_y, &hist_x, bandwidth);

    // Find modes (peaks)
    let mode_indices = find_modes(&smoothed);

    // Convert mode indices back to UTR positions
    let mut sites: Vec<f64> = mode_indices
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
    sites: &[f64],
    fragments: &[FragmentRecord],
    merge_dist: f64,
) -> Vec<f64> {
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

/// Count fragments near a position within a window.
fn nearby_count(counts: &HashMap<i64, usize>, pos: f64, window: f64) -> usize {
    let lo = (pos - window).round() as i64;
    let hi = (pos + window).round() as i64;
    (lo..=hi).filter_map(|p| counts.get(&p)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use genomic_data::sam::{CellBarcode, UmiBarcode};

    fn make_junction_frag(pa_pos: f64) -> FragmentRecord {
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
    fn test_junction_site_discovery() {
        let mut fragments = Vec::new();
        // 50 junction reads near alpha=300 (scattered ±10bp)
        for i in 0..50 {
            let offset = (i % 5) as f64 - 2.0; // -2..+2
            fragments.push(make_junction_frag(300.0 + offset));
        }
        // 50 junction reads near alpha=700
        for i in 0..50 {
            let offset = (i % 5) as f64 - 2.0;
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
