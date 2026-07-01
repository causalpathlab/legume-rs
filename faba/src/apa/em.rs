use crate::apa::likelihood::{log_lik_fragment_given_site_robust, log_lik_noise};

/// EM algorithm parameters.
pub struct EmParams {
    /// Maximum number of EM iterations
    pub max_iter: usize,
    /// Convergence tolerance for log-likelihood change
    pub tol: f32,
    /// Minimum component weight before pruning
    pub min_weight: f32,
    /// Mixing weight of the per-site uniform "skirt" inside each site's
    /// emission (heavy-tail robustness). 0 disables.
    pub skirt_eta: f32,
    /// Skirt half-width in units of beta: W = skirt_mult * beta.
    pub skirt_mult: f32,
    /// Post-EM merge: collapse selected sites with |alpha_i - alpha_j| <
    /// `merge_beta_mult * max(beta_i, beta_j)`. 0 disables.
    pub merge_beta_mult: f32,
    /// Cap on the candidate pool considered by BIC site-selection (top-N by
    /// coverage). Bounds the K-loop / per-column θ-marginalisation on long
    /// UTRs. 0 = unlimited.
    pub max_sites: usize,
}

impl Default for EmParams {
    fn default() -> Self {
        Self {
            // The weighted EM converges in a handful of iters; 40 caps the wasted
            // tail on the (opt-in) --mixture path with negligible accuracy loss.
            max_iter: 40,
            tol: 1e-4,
            min_weight: 0.01,
            skirt_eta: 0.05,
            skirt_mult: 3.0,
            merge_beta_mult: 2.0,
            max_sites: 0,
        }
    }
}

/// Precomputed data for a single UTR needed by the EM routines.
///
/// `theta_lik_matrix` is flat row-major over **fragment clusters**, not
/// raw fragments: row `m` lives at `[m * theta_grid.len() .. (m+1) *
/// theta_grid.len()]` and holds `log p(x_m, l_m, r_m | θ_t)` for cluster
/// `m`. `cluster_counts[m] = c_m` is the multiplicity (= number of
/// original fragments that fell into cluster `m`); these weight the EM
/// log-likelihood and M-step γ accumulator. `n_for_bic` is the original
/// fragment count, used in the BIC penalty so K-selection isn't biased
/// upward by clustering.
pub struct SiteModelData<'a> {
    pub alpha_arr: &'a [f32],
    pub beta_arr: &'a [f32],
    pub theta_lik_matrix: &'a [f32],
    pub theta_grid: &'a [f32],
    pub cluster_counts: &'a [f32],
    pub n_for_bic: usize,
    pub utr_length: f32,
    pub max_polya: f32,
}

impl<'a> SiteModelData<'a> {
    /// Number of fragment **clusters** (= rows of `theta_lik_matrix`).
    #[inline]
    pub fn n_clusters(&self) -> usize {
        if self.theta_grid.is_empty() {
            0
        } else {
            self.theta_lik_matrix.len() / self.theta_grid.len()
        }
    }

    #[inline]
    pub fn theta_row(&self, m: usize) -> &[f32] {
        let stride = self.theta_grid.len();
        &self.theta_lik_matrix[m * stride..(m + 1) * stride]
    }
}

/// Result of EM inference for one UTR.
///
/// `gamma` is flat row-major (`n_obs × n_components`). Use `gamma_row(n)`
/// for slice access.
pub struct EmResult {
    /// Estimated mixing weights pi_0..pi_K (pi_0 = noise)
    pub weights: Vec<f32>,
    /// pA site positions alpha_k (for k=1..K)
    pub alphas: Vec<f32>,
    /// pA site dispersions beta_k (for k=1..K)
    pub betas: Vec<f32>,
    /// Flat posterior gamma; column k=0 is the noise component.
    pub gamma: Vec<f32>,
    /// Number of mixture components (= row stride of `gamma` = K+1 with noise).
    pub n_components: usize,
    /// Final log-likelihood
    pub log_lik: f32,
    /// BIC
    pub bic: f32,
    /// Number of iterations
    pub n_iter: usize,
}

impl EmResult {
    #[inline]
    pub fn gamma_row(&self, n: usize) -> &[f32] {
        let start = n * self.n_components;
        &self.gamma[start..start + self.n_components]
    }
}

/// Constrained EM: alpha and beta are fixed, only estimate weights pi_k.
/// This is SCAPE's `fixed_inference()`.
pub fn fixed_inference(data: &SiteModelData, params: &EmParams) -> EmResult {
    let n_frag = data.n_clusters();
    let n_components = data.alpha_arr.len(); // K APA components
    let n_total = n_components + 1; // +1 for noise component (k=0)

    // Flat (n_frag × n_total) component log-likelihood matrix. Column 0
    // is the noise term; columns 1..=K are the per-site likelihoods.
    let mut component_log_liks = vec![0.0_f32; n_frag * n_total];
    let noise_ll = log_lik_noise(data.utr_length, data.max_polya);

    for n in 0..n_frag {
        let base = n * n_total;
        component_log_liks[base] = noise_ll;
        let theta_row = data.theta_row(n);
        for k in 0..n_components {
            component_log_liks[base + k + 1] = log_lik_fragment_given_site_robust(
                theta_row,
                data.theta_grid,
                data.alpha_arr[k],
                data.beta_arr[k],
                params.skirt_eta,
                params.skirt_mult,
            );
        }
    }

    run_fixed_em(
        &component_log_liks,
        n_total,
        data.alpha_arr,
        data.beta_arr,
        Some(data.cluster_counts),
        data.n_for_bic,
        params,
    )
}

/// Run the generic fixed EM on precomputed component log-likelihoods.
///
/// `component_log_liks` is flat row-major with stride `n_total`.
/// `cluster_counts` (optional) supplies the per-row multiplicity `c_m`
/// for clustered fragments; `n_for_bic` is the original fragment count
/// (so the BIC penalty stays calibrated to the un-collapsed dataset).
fn run_fixed_em(
    component_log_liks: &[f32],
    n_total: usize,
    alpha_arr: &[f32],
    beta_arr: &[f32],
    cluster_counts: Option<&[f32]>,
    n_for_bic: usize,
    params: &EmParams,
) -> EmResult {
    let n_components = alpha_arr.len();
    debug_assert_eq!(n_total, n_components + 1, "expected one noise column");
    let generic_params = crate::mixture::em::EmParams {
        max_iter: params.max_iter,
        tol: params.tol,
        min_weight: params.min_weight,
    };

    // alpha and beta are fixed; only mixing weights are free.
    // K+1 weights with sum-to-1 ⇒ K free parameters.
    let n_free_params = n_components;
    let result = crate::mixture::em::fixed_em_weighted(
        component_log_liks,
        n_total,
        cluster_counts,
        Some(n_for_bic),
        n_free_params,
        &generic_params,
    );

    EmResult {
        weights: result.weights,
        alphas: alpha_arr.to_vec(),
        betas: beta_arr.to_vec(),
        gamma: result.gamma,
        n_components: result.n_components,
        log_lik: result.log_lik,
        bic: result.bic,
        n_iter: result.n_iter,
    }
}

/// Select the best number of pA sites by BIC model selection.
///
/// Candidate sites are ranked by `site_order` (indices into `alpha_arr`,
/// sorted by descending coverage). Tries K=1, 2, ..., N sites in that order,
/// fitting an EM for each K. Returns the model with lowest BIC, stopping
/// early after 2 consecutive BIC increases.
///
/// If there is only one candidate site, runs a single EM (no selection needed).
pub fn select_sites_by_bic(
    data: &SiteModelData,
    params: &EmParams,
    site_order: &[usize],
) -> EmResult {
    let n_frag = data.n_clusters();
    // Cap the candidate pool to the top-N by coverage (site_order is
    // coverage-ranked). Bounds the K-loop and per-column cost on long UTRs.
    let n_candidates = if params.max_sites > 0 {
        site_order.len().min(params.max_sites)
    } else {
        site_order.len()
    };

    if n_candidates <= 1 {
        return fixed_inference(data, params);
    }

    // Per-candidate fragment log-likelihoods in fragment-major layout
    // (`all_site_lls[n * n_candidates + j]` = log p(frag n | site_order[j])).
    // Filled LAZILY inside the K-loop: the greedy search early-stops, so on
    // long UTRs (many candidates, few real sites) only the first `K_final`
    // columns are ever computed — we never pay the dominant per-fragment
    // θ-marginalisation for candidates the search doesn't reach.
    let noise_ll = log_lik_noise(data.utr_length, data.max_polya);
    let mut all_site_lls = vec![0.0_f32; n_frag * n_candidates];
    let mut n_filled = 0usize; // columns 0..n_filled are materialised
    let mut col_scratch = vec![0.0_f32; n_frag];

    let skirt_eta = params.skirt_eta;
    let skirt_mult = params.skirt_mult;
    let theta_grid = data.theta_grid;
    const PARALLEL_THRESHOLD: usize = 4096;

    let mut best_result: Option<EmResult> = None;
    let mut n_worse = 0u32;

    let mut selected_alphas: Vec<f32> = Vec::with_capacity(n_candidates);
    let mut selected_betas: Vec<f32> = Vec::with_capacity(n_candidates);
    // Scratch buffer for `component_log_liks` at the current K; reallocated
    // (resize) each iteration with stride `n_total = k + 1`. The fragment-
    // major layout of `all_site_lls` keeps the inner copy contiguous.
    let mut component_log_liks: Vec<f32> = Vec::new();

    for k in 1..=n_candidates {
        selected_alphas.push(data.alpha_arr[site_order[k - 1]]);
        selected_betas.push(data.beta_arr[site_order[k - 1]]);

        // Materialise any not-yet-computed candidate columns up to k (usually
        // just column k-1). Compute a contiguous scratch column (parallel over
        // fragments on heavy UTRs), then scatter it into the fragment-major
        // buffer so the copy below stays a contiguous slice.
        while n_filled < k {
            let idx = site_order[n_filled];
            let (a_j, b_j) = (data.alpha_arr[idx], data.beta_arr[idx]);
            let fill = |n: usize| {
                log_lik_fragment_given_site_robust(
                    data.theta_row(n),
                    theta_grid,
                    a_j,
                    b_j,
                    skirt_eta,
                    skirt_mult,
                )
            };
            if n_frag >= PARALLEL_THRESHOLD {
                use rayon::prelude::*;
                col_scratch
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(n, s)| *s = fill(n));
            } else {
                for (n, s) in col_scratch.iter_mut().enumerate() {
                    *s = fill(n);
                }
            }
            let j = n_filled;
            for n in 0..n_frag {
                all_site_lls[n * n_candidates + j] = col_scratch[n];
            }
            n_filled += 1;
        }

        let n_total = k + 1;
        component_log_liks.clear();
        component_log_liks.resize(n_frag * n_total, 0.0);
        for n in 0..n_frag {
            let dst = n * n_total;
            component_log_liks[dst] = noise_ll;
            let src = n * n_candidates;
            component_log_liks[dst + 1..dst + n_total].copy_from_slice(&all_site_lls[src..src + k]);
        }

        let result = run_fixed_em(
            &component_log_liks,
            n_total,
            &selected_alphas,
            &selected_betas,
            Some(data.cluster_counts),
            data.n_for_bic,
            params,
        );

        let is_better = match &best_result {
            None => true,
            Some(prev) => result.bic < prev.bic,
        };

        if is_better {
            best_result = Some(result);
            n_worse = 0;
        } else {
            n_worse += 1;
            if n_worse >= 2 {
                break;
            }
        }
    }

    let best = best_result.unwrap();
    merge_close_sites(
        best,
        &all_site_lls,
        n_candidates,
        noise_ll,
        data.cluster_counts,
        data.n_for_bic,
        params,
    )
}

/// Collapse selected sites whose alphas are within `merge_beta_mult * max(beta_i, beta_j)`
/// of each other into a single site (keep the higher-pi one), then re-fit weights.
fn merge_close_sites(
    result: EmResult,
    all_site_lls: &[f32],
    n_candidates: usize,
    noise_ll: f32,
    cluster_counts: &[f32],
    n_for_bic: usize,
    params: &EmParams,
) -> EmResult {
    if params.merge_beta_mult <= 0.0 || result.alphas.len() < 2 {
        return result;
    }

    let k = result.alphas.len();
    // Only consider live (pi > 0) components as merge candidates; pruned slots
    // must not ride into the refit's candidate list.
    let live: Vec<usize> = (0..k).filter(|&i| result.weights[i + 1] > 0.0).collect();
    if live.len() < 2 {
        return result;
    }

    // NaN-safe ordering: treat NaN as the smallest value so the sort is total.
    let cmp_f32 = |a: f32, b: f32| {
        a.partial_cmp(&b).unwrap_or(match (a.is_nan(), b.is_nan()) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => std::cmp::Ordering::Equal,
        })
    };

    // Index by descending pi (skip noise weight at [0]); break ties by lower alpha.
    let mut order = live.clone();
    order.sort_by(|&a, &b| {
        cmp_f32(result.weights[b + 1], result.weights[a + 1])
            .then(cmp_f32(result.alphas[a], result.alphas[b]))
    });

    // Greedy keep: walk in descending pi, keep a site unless it is within
    // merge_beta_mult * max(beta) of an already-kept site.
    let mut keep: Vec<usize> = Vec::with_capacity(order.len());
    for &i in &order {
        let close = keep.iter().any(|&j| {
            let tol = params.merge_beta_mult * result.betas[i].max(result.betas[j]);
            (result.alphas[i] - result.alphas[j]).abs() < tol
        });
        if !close {
            keep.push(i);
        }
    }

    if keep.len() == live.len() {
        return result; // nothing to merge
    }

    // Refit weights on the surviving subset.
    keep.sort_by(|&a, &b| cmp_f32(result.alphas[a], result.alphas[b]));
    let n_frag = if n_candidates == 0 {
        0
    } else {
        all_site_lls.len() / n_candidates
    };
    let kept_alphas: Vec<f32> = keep.iter().map(|&i| result.alphas[i]).collect();
    let kept_betas: Vec<f32> = keep.iter().map(|&i| result.betas[i]).collect();
    let n_total = keep.len() + 1;
    let mut component_log_liks = vec![0.0_f32; n_frag * n_total];
    for n in 0..n_frag {
        let base = n * n_total;
        component_log_liks[base] = noise_ll;
        let src_base = n * n_candidates;
        for (slot, &i) in keep.iter().enumerate() {
            component_log_liks[base + 1 + slot] = all_site_lls[src_base + i];
        }
    }

    let merged = run_fixed_em(
        &component_log_liks,
        n_total,
        &kept_alphas,
        &kept_betas,
        Some(cluster_counts),
        n_for_bic,
        params,
    );

    // Only accept the merge if BIC strictly improves; otherwise keep the
    // BIC-best model that motivated the merge attempt.
    if merged.bic.is_finite() && merged.bic < result.bic {
        merged
    } else {
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::apa::fragment::{cluster_fragments, ClusterBins};
    use crate::apa::likelihood::{precompute_theta_lik_matrix, LikelihoodParams};
    use crate::apa::simulate::{simulate_fragments, ScapeSimParams};

    /// Cluster simulated fragments (with the default bin sizes used in
    /// production) and return the inputs needed to build a SiteModelData:
    /// flat θ-likelihood matrix, θ grid, per-cluster counts, and the
    /// original-fragment count for BIC.
    fn cluster_for_em(
        fragments: &[crate::apa::fragment::FragmentRecord],
        params: &ScapeSimParams,
        lik_params: &LikelihoodParams,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, usize) {
        let (clusters, _cluster_idx) = cluster_fragments(fragments, ClusterBins::default());
        let counts: Vec<f32> = clusters.iter().map(|c| c.count as f32).collect();
        let (mat, grid) = precompute_theta_lik_matrix(&clusters, params.utr_length, lik_params);
        (mat, grid, counts, fragments.len())
    }

    fn run_em_on_sim(params: &ScapeSimParams) -> EmResult {
        let (fragments, _labels) = simulate_fragments(params);
        let lik_params = LikelihoodParams {
            mu_f: params.mu_f,
            sigma_f: params.sigma_f,
            theta_step: 10,
            max_polya: params.max_polya,
            min_polya: params.min_polya,
        };
        let (theta_lik_matrix, theta_grid, cluster_counts, n_for_bic) =
            cluster_for_em(&fragments, params, &lik_params);

        // Baseline EM behavior: disable skirt and post-EM merge so these
        // tests cover the non-robust path. Robust path has its own test.
        let em_params = EmParams {
            max_iter: 200,
            tol: 1e-6,
            min_weight: 0.005,
            skirt_eta: 0.0,
            skirt_mult: 0.0,
            merge_beta_mult: 0.0,
            max_sites: 0,
        };

        let site_data = SiteModelData {
            alpha_arr: &params.alphas,
            beta_arr: &params.betas,
            theta_lik_matrix: &theta_lik_matrix,
            theta_grid: &theta_grid,
            cluster_counts: &cluster_counts,
            n_for_bic,
            utr_length: params.utr_length,
            max_polya: params.max_polya,
        };
        fixed_inference(&site_data, &em_params)
    }

    #[test]
    fn test_single_site_no_noise() {
        let params = ScapeSimParams {
            utr_length: 2000.0,
            weights: vec![0.0, 1.0], // no noise
            alphas: vec![500.0],
            betas: vec![30.0],
            n_fragments: 2000,
            n_cells: 10,
            junction_prob: 0.3,
            seed: 42,
            ..Default::default()
        };
        let result = run_em_on_sim(&params);

        eprintln!(
            "single_site_no_noise: weights={:?}, n_iter={}",
            result.weights, result.n_iter
        );
        assert!(
            result.weights[1] > 0.9,
            "APA weight should be >0.9, got {}",
            result.weights[1]
        );
    }

    #[test]
    fn test_two_sites_balanced() {
        let params = ScapeSimParams {
            utr_length: 3000.0,
            weights: vec![0.0, 0.5, 0.5],
            alphas: vec![500.0, 1500.0],
            betas: vec![30.0, 30.0],
            n_fragments: 3000,
            n_cells: 10,
            junction_prob: 0.3,
            seed: 123,
            ..Default::default()
        };
        let result = run_em_on_sim(&params);

        eprintln!(
            "two_sites_balanced: weights={:?}, n_iter={}",
            result.weights, result.n_iter
        );
        // Both APA weights should be in [0.3, 0.7]
        assert!(
            result.weights[1] > 0.3 && result.weights[1] < 0.7,
            "site 1 weight should be ~0.5, got {}",
            result.weights[1]
        );
        assert!(
            result.weights[2] > 0.3 && result.weights[2] < 0.7,
            "site 2 weight should be ~0.5, got {}",
            result.weights[2]
        );
    }

    #[test]
    fn test_two_sites_with_noise() {
        let params = ScapeSimParams {
            utr_length: 3000.0,
            weights: vec![0.1, 0.45, 0.45],
            alphas: vec![500.0, 1500.0],
            betas: vec![30.0, 30.0],
            n_fragments: 5000,
            n_cells: 10,
            junction_prob: 0.3,
            seed: 456,
            ..Default::default()
        };
        let result = run_em_on_sim(&params);

        eprintln!(
            "two_sites_with_noise: weights={:?}, n_iter={}",
            result.weights, result.n_iter
        );
        // Noise weight should be detectable but not dominant
        assert!(
            result.weights[0] > 0.01 && result.weights[0] < 0.25,
            "noise weight should be in [0.01, 0.25], got {}",
            result.weights[0]
        );
        assert!(
            result.weights[1] > 0.25,
            "site 1 weight should be >0.25, got {}",
            result.weights[1]
        );
        assert!(
            result.weights[2] > 0.25,
            "site 2 weight should be >0.25, got {}",
            result.weights[2]
        );
    }

    #[test]
    fn test_unbalanced_weights() {
        let params = ScapeSimParams {
            utr_length: 3000.0,
            weights: vec![0.05, 0.7, 0.25],
            alphas: vec![500.0, 1500.0],
            betas: vec![30.0, 30.0],
            n_fragments: 5000,
            n_cells: 10,
            junction_prob: 0.3,
            seed: 789,
            ..Default::default()
        };
        let result = run_em_on_sim(&params);

        eprintln!(
            "unbalanced_weights: weights={:?}, n_iter={}",
            result.weights, result.n_iter
        );
        // Site 1 (weight=0.7) should have larger estimated weight than site 2 (weight=0.25)
        assert!(
            result.weights[1] > result.weights[2],
            "site 1 ({}) should have larger weight than site 2 ({})",
            result.weights[1],
            result.weights[2]
        );
    }

    #[test]
    fn test_bic_selects_true_sites() {
        // Simulate 2 true sites, then add a spurious 3rd candidate.
        // BIC selection should prefer K=2 over K=3.
        let params = ScapeSimParams {
            utr_length: 3000.0,
            weights: vec![0.05, 0.5, 0.45],
            alphas: vec![500.0, 1500.0],
            betas: vec![30.0, 30.0],
            n_fragments: 3000,
            n_cells: 10,
            junction_prob: 0.3,
            seed: 321,
            ..Default::default()
        };
        let (fragments, _) = simulate_fragments(&params);
        let lik_params = LikelihoodParams {
            mu_f: params.mu_f,
            sigma_f: params.sigma_f,
            theta_step: 10,
            max_polya: params.max_polya,
            min_polya: params.min_polya,
        };
        let (theta_lik_matrix, theta_grid, cluster_counts, n_for_bic) =
            cluster_for_em(&fragments, &params, &lik_params);

        // 2 true sites + 1 spurious site at position 2500
        let alpha_arr = vec![500.0, 1500.0, 2500.0];
        let beta_arr = vec![30.0, 30.0, 30.0];
        // Order: site 0 (true), site 1 (true), site 2 (spurious)
        let site_order = vec![0, 1, 2];

        // Baseline EM behavior: disable skirt and post-EM merge so these
        // tests cover the non-robust path. Robust path has its own test.
        let em_params = EmParams {
            max_iter: 200,
            tol: 1e-6,
            min_weight: 0.005,
            skirt_eta: 0.0,
            skirt_mult: 0.0,
            merge_beta_mult: 0.0,
            max_sites: 0,
        };

        let site_data = SiteModelData {
            alpha_arr: &alpha_arr,
            beta_arr: &beta_arr,
            theta_lik_matrix: &theta_lik_matrix,
            theta_grid: &theta_grid,
            cluster_counts: &cluster_counts,
            n_for_bic,
            utr_length: params.utr_length,
            max_polya: params.max_polya,
        };
        let result = select_sites_by_bic(&site_data, &em_params, &site_order);

        eprintln!(
            "bic_select: selected {} sites, alphas={:?}, weights={:?}, bic={:.1}",
            result.alphas.len(),
            result.alphas,
            result.weights,
            result.bic
        );

        // BIC should select 2 sites (not 3) since the spurious site adds no signal
        assert!(
            result.alphas.len() <= 3,
            "should select at most 3 sites, got {}",
            result.alphas.len()
        );
        // The two true sites should have non-trivial weight
        assert!(
            result.weights[1] > 0.2,
            "site 1 weight should be >0.2, got {}",
            result.weights[1]
        );
    }

    #[test]
    fn test_bic_single_candidate() {
        // Single candidate should just run fixed_inference
        let params = ScapeSimParams {
            utr_length: 2000.0,
            weights: vec![0.0, 1.0],
            alphas: vec![500.0],
            betas: vec![30.0],
            n_fragments: 1000,
            n_cells: 5,
            junction_prob: 0.3,
            seed: 42,
            ..Default::default()
        };
        let (fragments, _) = simulate_fragments(&params);
        let lik_params = LikelihoodParams {
            mu_f: params.mu_f,
            sigma_f: params.sigma_f,
            theta_step: 10,
            max_polya: params.max_polya,
            min_polya: params.min_polya,
        };
        let (theta_lik_matrix, theta_grid, cluster_counts, n_for_bic) =
            cluster_for_em(&fragments, &params, &lik_params);

        // Baseline EM behavior: disable skirt and post-EM merge so these
        // tests cover the non-robust path. Robust path has its own test.
        let em_params = EmParams {
            max_iter: 200,
            tol: 1e-6,
            min_weight: 0.005,
            skirt_eta: 0.0,
            skirt_mult: 0.0,
            merge_beta_mult: 0.0,
            max_sites: 0,
        };

        let site_data = SiteModelData {
            alpha_arr: &[500.0],
            beta_arr: &[30.0],
            theta_lik_matrix: &theta_lik_matrix,
            theta_grid: &theta_grid,
            cluster_counts: &cluster_counts,
            n_for_bic,
            utr_length: params.utr_length,
            max_polya: params.max_polya,
        };
        let result = select_sites_by_bic(&site_data, &em_params, &[0]);

        assert_eq!(result.alphas.len(), 1);
        assert!(result.weights[1] > 0.8, "single site should dominate");
    }

    #[test]
    fn test_post_em_merge_collapses_close_sites() {
        // Two true sites at 500 and 1500; offer BIC a third spurious candidate
        // very close to 500 (within merge tolerance) and verify the post-EM
        // merge collapses it.
        let params = ScapeSimParams {
            utr_length: 3000.0,
            weights: vec![0.05, 0.5, 0.45],
            alphas: vec![500.0, 1500.0],
            betas: vec![30.0, 30.0],
            n_fragments: 3000,
            n_cells: 10,
            junction_prob: 0.3,
            seed: 4242,
            ..Default::default()
        };
        let (fragments, _) = simulate_fragments(&params);
        let lik_params = LikelihoodParams {
            mu_f: params.mu_f,
            sigma_f: params.sigma_f,
            theta_step: 10,
            max_polya: params.max_polya,
            min_polya: params.min_polya,
        };
        let (theta_lik_matrix, theta_grid, cluster_counts, n_for_bic) =
            cluster_for_em(&fragments, &params, &lik_params);

        // 530 sits 30bp from the true 500 site → within 2*beta=60 merge tol.
        let alpha_arr = vec![500.0, 530.0, 1500.0];
        let beta_arr = vec![30.0, 30.0, 30.0];
        let site_order = vec![0, 1, 2];

        let em_params = EmParams {
            max_iter: 200,
            tol: 1e-6,
            min_weight: 0.005,
            skirt_eta: 0.05,
            skirt_mult: 3.0,
            merge_beta_mult: 2.0,
            max_sites: 0,
        };
        let site_data = SiteModelData {
            alpha_arr: &alpha_arr,
            beta_arr: &beta_arr,
            theta_lik_matrix: &theta_lik_matrix,
            theta_grid: &theta_grid,
            cluster_counts: &cluster_counts,
            n_for_bic,
            utr_length: params.utr_length,
            max_polya: params.max_polya,
        };
        let result = select_sites_by_bic(&site_data, &em_params, &site_order);

        // After the merge, no two surviving alphas should be within tol of each other.
        let live: Vec<f32> = result
            .alphas
            .iter()
            .zip(result.weights.iter().skip(1))
            .filter(|(_, &w)| w > 0.0)
            .map(|(&a, _)| a)
            .collect();
        for (i, &a) in live.iter().enumerate() {
            for &b in &live[i + 1..] {
                assert!(
                    (a - b).abs() >= 2.0 * 30.0,
                    "live sites at {} and {} are within merge tolerance; merge should have collapsed them",
                    a,
                    b
                );
            }
        }
        assert!(
            result.bic.is_finite(),
            "merged result bic should be finite, got {}",
            result.bic
        );
    }
}
