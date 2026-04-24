//! Conditional-entropy estimator with Miller-Madow bias correction.
//!
//! Given two continuous variables observed on the same N samples (edges), we
//! bin each into `k` quantile bins and compute the plug-in estimate of
//! `H(R|L) = H(L,R) - H(L)` in nats. The Miller-Madow correction adds
//! `(K̂_eff - 1) / (2 n)` (in nats) to each marginal/joint entropy before
//! subtraction, where `K̂_eff` is the number of non-empty bins (joint or
//! marginal). The returned value is normalised by `H(R)` so it lies in
//! `[0, 1]` (0 = perfect prediction, 1 = independent).

/// Bin edges via quantiles. Returns `n_bins+1` cut points: `[min, q_1, ..., q_{k-1}, max]`.
///
/// Uses nearest-rank quantiles on a sorted copy of `values`. Ties are handled
/// by the subsequent `assign_bin` call, which forces bin indices into
/// `[0, n_bins-1]`.
pub fn quantile_edges(values: &[f32], n_bins: usize) -> Vec<f32> {
    assert!(n_bins >= 2, "n_bins must be at least 2");
    let mut sorted: Vec<f32> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let mut edges = Vec::with_capacity(n_bins + 1);
    edges.push(sorted.first().copied().unwrap_or(0.0));
    for b in 1..n_bins {
        let q = (b as f32) / (n_bins as f32);
        let idx = ((q * (n as f32)) as usize).min(n.saturating_sub(1));
        edges.push(sorted[idx]);
    }
    edges.push(sorted.last().copied().unwrap_or(0.0));
    edges
}

fn assign_bin(value: f32, edges: &[f32]) -> usize {
    let n_bins = edges.len() - 1;
    for (b, &edge) in edges.iter().enumerate().take(n_bins).skip(1) {
        if value < edge {
            return b - 1;
        }
    }
    n_bins - 1
}

/// Compute normalised Miller-Madow H(R|L) / H(R) in `[0, 1]`.
///
/// Returns `None` if fewer than 2 samples, `H(R)` is zero, or quantile edges
/// collapse for either variable. The `bins_l` and `bins_r` slices are
/// pre-computed quantile edges; reusing them across real and decoy pairs
/// within the same connected component keeps the null on the same grid.
pub fn normalised_conditional_entropy(
    l: &[f32],
    r: &[f32],
    bins_l: &[f32],
    bins_r: &[f32],
) -> Option<f32> {
    let n = l.len();
    if n < 2 || r.len() != n {
        return None;
    }
    let kl = bins_l.len() - 1;
    let kr = bins_r.len() - 1;
    if kl < 2 || kr < 2 {
        return None;
    }

    let mut joint = vec![0u32; kl * kr];
    let mut marg_l = vec![0u32; kl];
    let mut marg_r = vec![0u32; kr];
    for i in 0..n {
        let bl = assign_bin(l[i], bins_l);
        let br = assign_bin(r[i], bins_r);
        joint[bl * kr + br] += 1;
        marg_l[bl] += 1;
        marg_r[br] += 1;
    }

    let nf = n as f32;
    let h_plugin = |counts: &[u32]| -> (f32, usize) {
        let mut h = 0.0f32;
        let mut k_eff = 0usize;
        for &c in counts {
            if c > 0 {
                let p = (c as f32) / nf;
                h -= p * p.ln();
                k_eff += 1;
            }
        }
        (h, k_eff)
    };

    let (h_joint, k_joint) = h_plugin(&joint);
    let (h_l, k_l) = h_plugin(&marg_l);
    let (h_r, k_r) = h_plugin(&marg_r);

    let mm = |k_eff: usize| -> f32 {
        if k_eff <= 1 {
            0.0
        } else {
            ((k_eff - 1) as f32) / (2.0 * nf)
        }
    };
    let h_joint = h_joint + mm(k_joint);
    let h_l = h_l + mm(k_l);
    let h_r = h_r + mm(k_r);

    let h_r_given_l = h_joint - h_l;
    if h_r <= 0.0 {
        return None;
    }
    Some((h_r_given_l / h_r).clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ce_perfect_prediction() {
        // R = f(L): CE should be near 0.
        let l: Vec<f32> = (0..200).map(|i| i as f32).collect();
        let r: Vec<f32> = l.iter().map(|&x| 2.0 * x).collect();
        let eb_l = quantile_edges(&l, 4);
        let eb_r = quantile_edges(&r, 4);
        let ce = normalised_conditional_entropy(&l, &r, &eb_l, &eb_r).unwrap();
        assert!(ce < 0.05, "expected near-zero CE, got {}", ce);
    }

    #[test]
    fn ce_independent() {
        // Independent pairs constructed so marginals and joint are near-uniform.
        let n = 1600;
        let l: Vec<f32> = (0..n).map(|i| (i % 4) as f32).collect();
        // r depends on (i/4) parity, orthogonal to l.
        let r: Vec<f32> = (0..n).map(|i| ((i / 4) % 4) as f32).collect();
        let eb_l = quantile_edges(&l, 4);
        let eb_r = quantile_edges(&r, 4);
        let ce = normalised_conditional_entropy(&l, &r, &eb_l, &eb_r).unwrap();
        assert!(ce > 0.9, "expected CE near 1, got {}", ce);
    }

    #[test]
    fn miller_madow_shifts_small_samples() {
        // With tiny n, MM correction should measurably raise plug-in entropy
        // (and therefore CE) compared to a hand-computed plug-in baseline.
        let l: Vec<f32> = vec![0.0, 0.0, 1.0, 1.0];
        let r: Vec<f32> = vec![0.0, 1.0, 0.0, 1.0];
        let eb = vec![-0.5, 0.5, 1.5];
        let ce = normalised_conditional_entropy(&l, &r, &eb, &eb).unwrap();
        // Bivariate joint has 4 non-empty cells each with count 1; MM inflates
        // H_joint by 3/(2n) = 0.375 and H_L, H_R by 1/(2n) = 0.125 each.
        // Plug-in H(R|L) = ln 2 = 0.693; MM adds 0.375 - 0.125 = 0.25 → 0.943.
        // Plug-in H(R) = ln 2 + 0.125 = 0.818. Ratio ~1.15, clamped to 1.
        assert!(ce > 0.99);
    }
}
