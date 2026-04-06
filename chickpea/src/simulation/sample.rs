use crate::common::*;
use rand::prelude::*;
use rand_distr::Poisson;
use rayon::prelude::*;
use std::borrow::Cow;

/// Sample topic proportions [K × N] with hard assignments softened by PVE.
///
/// Follows data-beans convention: each cell gets a dominant topic,
/// then mixes with uniform background proportional to (1 - pve).
pub fn sample_topic_proportions(k: usize, n: usize, pve_topic: f32, rng: &mut impl Rng) -> Mat {
    let pve = pve_topic.clamp(0.0, 1.0);

    let runif = rand_distr::Uniform::new(0, k).unwrap();
    let bg = (1.0 - pve) / (k - 1).max(1) as f32;
    let mut theta = Mat::zeros(k, n);

    for j in 0..n {
        let dominant = runif.sample(rng);
        for i in 0..k {
            theta[(i, j)] = if i == dominant { pve + bg } else { bg };
        }
    }

    theta
}

/// Sample dictionary β[D × K] via row-wise softmax of random logits.
pub fn sample_dictionary(d: usize, k: usize, rng: &mut impl Rng) -> Mat {
    let normal = rand_distr::Normal::new(0.0f32, 1.0).unwrap();

    let mut beta = Mat::zeros(d, k);
    for i in 0..d {
        for j in 0..k {
            beta[(i, j)] = normal.sample(rng);
        }
    }

    // In-place row-wise softmax
    for i in 0..d {
        let max_val: f32 = (0..k)
            .map(|j| beta[(i, j)])
            .fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0f32;
        for j in 0..k {
            beta[(i, j)] = (beta[(i, j)] - max_val).exp();
            exp_sum += beta[(i, j)];
        }
        for j in 0..k {
            beta[(i, j)] /= exp_sum;
        }
    }

    beta
}

/// Generate per-cell depth multipliers: depth * exp(N(0, σ²)).
pub fn sample_cell_depths(
    n: usize,
    depth: usize,
    cell_sd_log_depth: f32,
    rng: &mut impl Rng,
) -> Vec<f32> {
    let normal = rand_distr::Normal::new(0.0f32, cell_sd_log_depth).unwrap();
    let depth_f = depth as f32;
    (0..n).map(|_| depth_f * normal.sample(rng).exp()).collect()
}

/// Sample gene-topic effects γ[G × K] from LogNormal(0, σ²).
pub fn sample_gene_topic_effects(
    g: usize,
    k: usize,
    gene_topic_sd: f32,
    rng: &mut impl Rng,
) -> Mat {
    let normal = rand_distr::Normal::new(0.0f32, gene_topic_sd).unwrap();
    let mut gamma = Mat::zeros(g, k);
    for i in 0..g {
        for j in 0..k {
            gamma[(i, j)] = normal.sample(rng).exp();
        }
    }
    gamma
}

/// Sample sparse indicator matrix M[G × P].
///
/// Returns (gene_indices, peak_indices) where each pair represents
/// one entry M[gene, peak] = 1.
pub fn sample_indicator_matrix(
    n_genes: usize,
    n_peaks: usize,
    n_linked_genes: usize,
    n_causal_per_gene: usize,
    rng: &mut impl Rng,
) -> (Vec<usize>, Vec<usize>) {
    let mut genes = Vec::new();
    let mut peaks = Vec::new();

    let mut gene_indices: Vec<usize> = (0..n_genes).collect();
    gene_indices.shuffle(rng);
    let linked_genes = &gene_indices[..n_linked_genes.min(n_genes)];

    for &gi in linked_genes {
        let n_causal = n_causal_per_gene.min(n_peaks);
        let sampled = rand::seq::index::sample(rng, n_peaks, n_causal);
        for pi in sampled.into_iter() {
            genes.push(gi);
            peaks.push(pi);
        }
    }

    (genes, peaks)
}

/// Build derived RNA dictionary W[G × K] = M × β_atac.
///
/// W[g, k] = Σ_r M[g,r] * β_atac[r, k]
pub fn build_derived_dictionary(
    indicator_genes: &[usize],
    indicator_peaks: &[usize],
    beta_atac: &Mat,
    n_genes: usize,
) -> Mat {
    let k = beta_atac.ncols();
    let mut w = Mat::zeros(n_genes, k);

    for (&gi, &pi) in indicator_genes.iter().zip(indicator_peaks.iter()) {
        for kk in 0..k {
            w[(gi, kk)] += beta_atac[(pi, kk)];
        }
    }

    w
}

/// Sample Poisson counts with vectorized rates and rayon parallelism.
///
/// Pre-computes effective dictionary eff = γ ⊙ β, then parallelises
/// over cells: rate_j = depth_j * eff * θ_j (matrix-vector multiply).
pub fn sample_poisson_counts(
    beta_dk: &Mat,
    theta_kn: &Mat,
    cell_depths: &[f32],
    gamma_gk: Option<&Mat>,
    rseed: u64,
) -> Vec<(u64, u64, f32)> {
    // Pre-compute effective dictionary (avoids per-element branch in hot loop)
    let eff_dk: Cow<Mat> = match gamma_gk {
        Some(gamma) => Cow::Owned(gamma.component_mul(beta_dk)),
        None => Cow::Borrowed(beta_dk),
    };

    cell_depths
        .par_iter()
        .enumerate()
        .flat_map_iter(|(j, &depth_j)| {
            let mut rng = StdRng::seed_from_u64(rseed.wrapping_add(j as u64));
            let rates: Vec<f32> = (&*eff_dk * theta_kn.column(j) * depth_j)
                .iter()
                .copied()
                .collect();

            rates.into_iter().enumerate().filter_map(move |(i, rate)| {
                if rate > 1e-6 {
                    let count = sample_poisson_safe(rate, &mut rng);
                    if count > 0.0 {
                        return Some((i as u64, j as u64, count));
                    }
                }
                None
            })
        })
        .collect()
}

fn sample_poisson_safe(rate: f32, rng: &mut impl Rng) -> f32 {
    if rate > 700.0 {
        rate.round()
    } else {
        Poisson::new(rate as f64)
            .map(|d| d.sample(rng) as f32)
            .unwrap_or(0.0)
    }
}
