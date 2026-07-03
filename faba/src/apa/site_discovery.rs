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
    // A cluster needs ≥1 read (and ≥ min_count overall); `.max(1)` also makes a
    // `min_count == 0` request well-defined instead of panicking.
    let floor = min_count.max(1);
    if sorted.len() < floor {
        return Vec::new();
    }
    // Iterative (explicit work stack) rather than recursive: a long UTR can peel
    // into very many clusters, and one-leaf-per-level recursion would overflow the
    // (rayon worker) stack. Pushing right-then-left keeps the leaves in ascending
    // position order (the recursive `left.extend(right)` order).
    let mut leaves = Vec::new();
    let mut stack: Vec<&[f32]> = vec![sorted];
    while let Some(seg) = stack.pop() {
        let n = seg.len();
        // Interior split index i (between seg[i] and seg[i+1]) with the largest gap,
        // keeping ≥ `floor` reads on each side. `hi = n - floor` bounds i ≤ n-2, so
        // `seg[i + 1]` is always in range.
        let lo = floor.saturating_sub(1);
        let hi = n.saturating_sub(floor);
        let mut best_gap = min_gap;
        let mut best_i: Option<usize> = None;
        for i in lo..hi {
            let gap = seg[i + 1] - seg[i];
            if gap > best_gap {
                best_gap = gap;
                best_i = Some(i);
            }
        }
        match best_i {
            None => leaves.push((seg[n / 2], n)), // leaf: one site (median representative)
            Some(i) => {
                stack.push(&seg[i + 1..]);
                stack.push(&seg[..=i]);
            }
        }
    }
    leaves
}

/// Count fragments near a position within a window.
fn nearby_count(counts: &HashMap<i64, usize>, pos: f32, window: f32) -> usize {
    let lo = (pos - window).round() as i64;
    let hi = (pos + window).round() as i64;
    (lo..=hi).filter_map(|p| counts.get(&p)).sum()
}

#[cfg(test)]
mod tests;
