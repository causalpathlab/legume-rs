use crate::mixture::em::{fixed_em_weighted, EmParams, GmmResult};
use crate::mixture::kernel_smooth::{find_modes, gaussian_kernel_smooth};

/// Parameters for per-gene mixture model.
///
/// Components are called **bandwidth-first**: the signal-weighted site pileup is
/// Gaussian-smoothed at `bandwidth` and its modes become component centres
/// (number of components = number of modes). Mixing weights are then fit with a
/// fixed-component EM. This replaces the old gene-length-relative GMM + BIC-over-K
/// scan, whose resolution scaled with gene length rather than the modality's
/// intrinsic spatial scale.
#[derive(Clone)]
pub struct MixtureParams {
    /// Minimum distinct positions per gene to attempt a fit
    pub min_sites: usize,
    /// Optional safety cap on the number of components (peaks). 0 = no cap.
    pub max_k: usize,
    /// Gaussian smoothing bandwidth (nt). 0 = derive a per-gene fallback from
    /// the gene's own site spacing (direct callers / tests); the pipeline
    /// resolves a global per-modality value via
    /// [`crate::editing::bandwidth::estimate_bandwidth`] before calling.
    pub bandwidth: f32,
    /// Drop genes whose fit yields a single active component. A lone
    /// component carries no relative/differential signal (its per-cell
    /// count is just the gene total), so this prunes uninformative rows
    /// from the mixture matrix and the component annotations.
    pub drop_single_component: bool,
    /// EM parameters
    pub em_params: EmParams,
}

impl Default for MixtureParams {
    fn default() -> Self {
        Self {
            min_sites: 3,
            max_k: 5,
            bandwidth: 0.0,
            drop_single_component: false,
            em_params: EmParams {
                max_iter: 200,
                tol: 1e-6,
                min_weight: 0.01,
            },
        }
    }
}

/// Annotation for one mixture component
pub struct MixtureComponentAnnotation {
    /// Gene name
    pub gene_name: Box<str>,
    /// Component index within gene
    pub component_idx: usize,
    /// Learned mean position (5'-relative, strand-aware, in nt)
    pub mu: f32,
    /// Learned standard deviation
    pub sigma: f32,
    /// Mixing weight
    pub pi: f32,
    /// Gene length in nt (denominator for normalized position u = mu / gene_length)
    pub gene_length: f32,
}

/// Result of per-gene mixture model
pub struct GeneMixtureResult {
    /// Best K (number of Gaussian components, not including noise)
    #[cfg(test)]
    pub best_k: usize,
    /// GMM result for best K
    pub gmm: GmmResult,
    /// Per-(cell_index, component) weighted count
    pub cell_component_counts: Vec<(usize, usize, f32)>,
}

/// Weighted observation for the mixture model: unique (cell, position) with weight.
///
/// `count` is the EM weight (and the value distributed to the output sparse matrix).
/// Under `MixtureWeightMode::Converted` it equals the raw converted-read count;
/// under `MixtureWeightMode::Posterior` it is the Beta-posterior regularized
/// effective count `n · (c+α)/(n+α+β)` and therefore generally fractional.
pub struct WeightedObservation {
    /// Index of the cell in the cell list
    pub cell_idx: usize,
    /// Genomic position of the modification
    pub position: f32,
    /// Weight at this (cell, position)
    pub count: f32,
}

/// Run per-gene GMM model selection over K=1..max_k, pick best by BIC.
///
/// * `observations` - weighted observations (unique per cell+position)
/// * `gene_length` - length of the gene for uniform noise component
/// * `params` - mixture parameters
///
/// Returns None if fewer than min_sites distinct positions.
pub fn fit_gene_mixture(
    observations: &[WeightedObservation],
    gene_length: f32,
    params: &MixtureParams,
) -> Option<GeneMixtureResult> {
    if observations.is_empty() {
        return None;
    }

    // Unique site positions (rounded to nt) with summed signal weight.
    let mut by_pos: rustc_hash::FxHashMap<i64, f32> = rustc_hash::FxHashMap::default();
    for o in observations {
        *by_pos.entry(o.position.round() as i64).or_insert(0.0) += o.count.max(0.0);
    }
    if by_pos.len() < params.min_sites {
        return None;
    }
    let mut sites: Vec<(f32, f32)> = by_pos.iter().map(|(&p, &w)| (p as f32, w)).collect();
    sites.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let xs: Vec<f32> = sites.iter().map(|&(p, _)| p).collect();
    let ys: Vec<f32> = sites.iter().map(|&(_, w)| w).collect();

    // Effective bandwidth: the resolved global value (params.bandwidth > 0) or a
    // per-gene fallback derived from this gene's own site spacing.
    let bandwidth = if params.bandwidth > 0.0 {
        params.bandwidth
    } else {
        fallback_bandwidth(&xs)
    };

    // Smooth the signal-weighted site pileup at `bandwidth` and read off modes as
    // component centres. Evaluating only at the (sparse) site positions keeps
    // this O(S^2) in the number of sites, not O(gene_length).
    let smoothed = gaussian_kernel_smooth(&xs, &ys, &xs, bandwidth);

    // find_modes returns only interior maxima; pad with zero sentinels so a peak
    // at the first/last site is detected too. Carry the mode density alongside
    // the centre position for the optional safety cap.
    let mut padded = Vec::with_capacity(smoothed.len() + 2);
    padded.push(0.0);
    padded.extend_from_slice(&smoothed);
    padded.push(0.0);
    let mut centers: Vec<(f32, f32)> = find_modes(&padded)
        .into_iter()
        .map(|i| (xs[i - 1], smoothed[i - 1]))
        .collect();

    // Degenerate (flat) profile → no interior mode: fall back to the single
    // signal-weighted centroid as one component.
    if centers.is_empty() {
        let wsum: f32 = ys.iter().sum();
        let centroid = if wsum > 0.0 {
            xs.iter().zip(ys.iter()).map(|(&x, &w)| x * w).sum::<f32>() / wsum
        } else {
            xs[xs.len() / 2]
        };
        centers.push((centroid, wsum));
    }

    // Optional safety cap: keep the highest-density centres.
    if params.max_k > 0 && centers.len() > params.max_k {
        centers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        centers.truncate(params.max_k);
        centers.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    }

    let centers: Vec<f32> = centers.into_iter().map(|(p, _)| p).collect();
    let k = centers.len();
    let n_total = k + 1; // component 0 = uniform noise
    let n_obs = observations.len();

    // Component log-likelihoods: col 0 = uniform noise over the gene body, cols
    // 1..=k = Gaussian(centre, bandwidth) at each observation position.
    let noise_ll = if gene_length > 0.0 {
        -gene_length.ln()
    } else {
        f32::NEG_INFINITY
    };
    let mut cll = vec![0.0_f32; n_obs * n_total];
    for (n, o) in observations.iter().enumerate() {
        let base = n * n_total;
        cll[base] = noise_ll;
        for (j, &c) in centers.iter().enumerate() {
            cll[base + 1 + j] = gaussian_log_pdf(o.position, c, bandwidth);
        }
    }

    // Fit mixing weights only (centres and σ = bandwidth are fixed). The
    // per-observation signal weight enters as multiplicity.
    let obs_weights: Vec<f32> = observations.iter().map(|o| o.count).collect();
    let fe = fixed_em_weighted(
        &cll,
        n_total,
        Some(&obs_weights),
        Some(n_obs),
        k, // free params = K mixing weights (sum-to-1)
        &params.em_params,
    );

    // Hard-assign each observation to argmax over {noise, components} and
    // accumulate weighted counts per (cell, component). Component 0 = noise.
    let mut counts: rustc_hash::FxHashMap<(usize, usize), f32> = rustc_hash::FxHashMap::default();
    for (n, o) in observations.iter().enumerate() {
        let row = &fe.gamma[n * n_total..(n + 1) * n_total];
        let best = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(c, _)| c)
            .unwrap_or(0);
        *counts.entry((o.cell_idx, best)).or_insert(0.0) += o.count;
    }
    let cell_component_counts: Vec<(usize, usize, f32)> = counts
        .into_iter()
        .map(|((cell, comp), cnt)| (cell, comp, cnt))
        .collect();

    #[cfg(test)]
    let n_active = fe.weights.iter().skip(1).filter(|&&w| w > 0.0).count();
    let gmm = GmmResult {
        weights: fe.weights,
        mus: centers,
        sigmas: vec![bandwidth; k],
        gamma: Vec::new(), // per-obs γ already consumed into cell_component_counts
        bic: fe.bic,
    };

    Some(GeneMixtureResult {
        #[cfg(test)]
        best_k: n_active,
        gmm,
        cell_component_counts,
    })
}

/// Log PDF of a univariate Gaussian.
fn gaussian_log_pdf(x: f32, mu: f32, sigma: f32) -> f32 {
    if sigma <= 0.0 {
        return f32::NEG_INFINITY;
    }
    let z = (x - mu) / sigma;
    -0.5 * z * z - sigma.ln() - 0.5 * std::f32::consts::TAU.ln()
}

/// Per-gene fallback bandwidth from the gene's own sorted site positions: the
/// median nearest-neighbour gap, clamped to a sane nt range. Used only when the
/// pipeline has not resolved a global per-modality bandwidth (e.g. direct
/// callers and unit tests).
fn fallback_bandwidth(sorted_positions: &[f32]) -> f32 {
    if sorted_positions.len() < 2 {
        return 25.0;
    }
    let mut gaps: Vec<f32> = sorted_positions.windows(2).map(|w| w[1] - w[0]).collect();
    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = gaps[gaps.len() / 2];
    median.clamp(10.0, 200.0)
}

#[cfg(test)]
mod tests;
