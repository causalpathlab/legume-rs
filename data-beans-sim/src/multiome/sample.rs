//! Sampler primitives for paired ATAC + RNA simulation.
//!
//! Ported verbatim from `chickpea/src/simulation/sample.rs` (the original
//! `sim-link` simulator) with `Mat` substituted for `nalgebra::DMatrix<f32>`
//! so this module pulls in no chickpea-internal types.

use matrix_util::traits::MatOps;
use nalgebra::DMatrix;
use rand::prelude::*;
use rand_distr::Poisson;
use rayon::prelude::*;
use std::borrow::Cow;

pub type Mat = DMatrix<f32>;

/// Sample dictionary β[D × K] via column-wise softmax of random logits, so
/// each topic (column) is a probability distribution over features and
/// `Σ_d β[d, k] = 1`. With column-stochastic θ this gives the rate model
/// `λ = depth · β · θ` an expected library size of `depth` per cell.
///
/// The seeded N(0,1) fill stays scalar (preserves seed reproducibility);
/// the softmax delegates to `normalize_exp_logits_columns_inplace`
/// (numerically-stable log-sum-exp from `matrix-util`).
pub fn sample_dictionary(d: usize, k: usize, rng: &mut impl Rng) -> Mat {
    let normal = rand_distr::Normal::new(0.0f32, 1.0).unwrap();
    let buf: Vec<f32> = (0..d * k).map(|_| normal.sample(rng)).collect();
    let mut beta = Mat::from_vec(d, k, buf);
    beta.normalize_exp_logits_columns_inplace();
    beta
}

/// Generate per-cell depth multipliers `depth · exp(σ·N(0,1) − σ²/2)`.
/// The `−σ²/2` centering makes `E[depth_j] = depth` so the supplied `depth`
/// is the expected library size, not biased by the log-normal mean drift.
pub fn sample_cell_depths(
    n: usize,
    depth: usize,
    cell_sd_log_depth: f32,
    rng: &mut impl Rng,
) -> Vec<f32> {
    let sigma = cell_sd_log_depth;
    let center = 0.5 * sigma * sigma;
    let normal = rand_distr::Normal::new(0.0f32, sigma).unwrap();
    let depth_f = depth as f32;
    (0..n)
        .map(|_| depth_f * (normal.sample(rng) - center).exp())
        .collect()
}

/// Sample nested topic proportions [K_total, N] with hierarchical assignments.
///
/// Each cell gets a coarse topic (PVE-controlled), then a subtype within it.
/// Returns (θ_full[K_total, N], θ_coarse[K, N]) where K_total = K × K_sub.
/// θ_coarse is the marginalization: θ_coarse[k,n] = Σ_s θ_full[k*K_sub+s, n].
///
/// When K_sub = 1, θ_full == θ_coarse.
pub fn sample_nested_topic_proportions(
    k: usize,
    k_sub: usize,
    n: usize,
    pve_coarse: f32,
    pve_sub: f32,
    rseed: u64,
) -> (Mat, Mat) {
    let k_total = k * k_sub;
    let pve_c = pve_coarse.clamp(0.0, 1.0);
    let pve_s = pve_sub.clamp(0.0, 1.0);
    let bg_coarse = (1.0 - pve_c) / (k - 1).max(1) as f32;
    let bg_sub = (1.0 - pve_s) / (k_sub - 1).max(1) as f32;

    let cells: Vec<(Vec<f32>, Vec<f32>)> = (0..n)
        .into_par_iter()
        .map(|j| {
            let mut rng = StdRng::seed_from_u64(rseed.wrapping_add(j as u64));
            let coarse_dist = rand_distr::Uniform::new(0, k).unwrap();
            let sub_dist = rand_distr::Uniform::new(0, k_sub).unwrap();

            let dom_k = coarse_dist.sample(&mut rng);
            let dom_s = sub_dist.sample(&mut rng);

            let mut full_col = vec![0.0f32; k_total];
            let mut coarse_col = vec![0.0f32; k];

            for kk in 0..k {
                let coarse_weight = if kk == dom_k {
                    pve_c + bg_coarse
                } else {
                    bg_coarse
                };
                coarse_col[kk] = coarse_weight;

                for s in 0..k_sub {
                    let sub_weight = if kk == dom_k {
                        if s == dom_s {
                            pve_s + bg_sub
                        } else {
                            bg_sub
                        }
                    } else {
                        1.0 / k_sub as f32
                    };
                    full_col[kk * k_sub + s] = coarse_weight * sub_weight;
                }
            }

            (full_col, coarse_col)
        })
        .collect();

    let mut theta_full = Mat::zeros(k_total, n);
    let mut theta_coarse = Mat::zeros(k, n);
    for (j, (full_col, coarse_col)) in cells.into_iter().enumerate() {
        for (i, v) in full_col.into_iter().enumerate() {
            theta_full[(i, j)] = v;
        }
        for (i, v) in coarse_col.into_iter().enumerate() {
            theta_coarse[(i, j)] = v;
        }
    }

    (theta_full, theta_coarse)
}

/// Marginalize β_ext[P, K×K_sub] → β_atac[P, K] by averaging over subtypes
/// (uniform `P(s | k) = 1/K_sub`). With column-stochastic β_ext, β_atac's
/// columns also sum to 1.
///
/// Vectorized via nalgebra column-views — each output column is a sum of
/// `K_sub` contiguous source columns, scaled by `1/K_sub`.
pub fn marginalize_dictionary(beta_ext: &Mat, k: usize, k_sub: usize) -> Mat {
    let p = beta_ext.nrows();
    let inv_k_sub = 1.0_f32 / k_sub.max(1) as f32;
    let mut beta = Mat::zeros(p, k);
    for kk in 0..k {
        let mut acc = beta.column_mut(kk);
        for s in 0..k_sub {
            acc += beta_ext.column(kk * k_sub + s);
        }
        acc.scale_mut(inv_k_sub);
    }
    beta
}

/// Sample gene-topic effects γ[G × K] from LogNormal(0, σ²).
pub fn sample_gene_topic_effects(
    g: usize,
    k: usize,
    gene_topic_sd: f32,
    rng: &mut impl Rng,
) -> Mat {
    let normal = rand_distr::Normal::new(0.0f32, gene_topic_sd).unwrap();
    let buf: Vec<f32> = (0..g * k).map(|_| normal.sample(rng).exp()).collect();
    Mat::from_vec(g, k, buf)
}

/// Sample sparse indicator matrix M[G × P] with nearby causal peaks.
///
/// Causal peaks are selected from the same chromosome as the gene,
/// within a local window. This ensures distance-based cis-windows
/// can recover the true links.
///
/// Layout: gene i → chr (i % n_chr) + 1, position ~ i / n_chr.
///         peak j → chr (j % n_chr) + 1, position ~ j / n_chr.
///
/// Returns (gene_indices, peak_indices) where each pair represents
/// one entry M[gene, peak] = 1.
pub fn sample_indicator_matrix(
    n_genes: usize,
    n_peaks: usize,
    n_linked_genes: usize,
    n_causal_per_gene: usize,
    n_chromosomes: usize,
    rng: &mut impl Rng,
) -> (Vec<usize>, Vec<usize>) {
    let mut genes = Vec::new();
    let mut peaks = Vec::new();

    let mut chr_peaks: Vec<Vec<usize>> = vec![Vec::new(); n_chromosomes];
    for p in 0..n_peaks {
        chr_peaks[p % n_chromosomes].push(p);
    }

    let mut gene_indices: Vec<usize> = (0..n_genes).collect();
    gene_indices.shuffle(rng);
    let linked_genes = &gene_indices[..n_linked_genes.min(n_genes)];

    for &gi in linked_genes {
        let chr = gi % n_chromosomes;
        let peaks_on_chr = &chr_peaks[chr];
        if peaks_on_chr.is_empty() {
            continue;
        }

        let gene_pos = gi / n_chromosomes;

        let center = gene_pos.min(peaks_on_chr.len() - 1);

        let half_window = (n_causal_per_gene * 3).max(10).min(peaks_on_chr.len() / 2);
        let lo = center.saturating_sub(half_window);
        let hi = (center + half_window + 1).min(peaks_on_chr.len());
        let window_size = hi - lo;

        let n_causal = n_causal_per_gene.min(window_size);
        let sampled = rand::seq::index::sample(rng, window_size, n_causal);
        for idx in sampled.into_iter() {
            genes.push(gi);
            peaks.push(peaks_on_chr[lo + idx]);
        }
    }

    (genes, peaks)
}

/// Build derived RNA dictionary W[G × K] = M × β.
pub fn build_derived_dictionary(
    indicator_genes: &[usize],
    indicator_peaks: &[usize],
    beta_dk: &Mat,
    n_genes: usize,
) -> Mat {
    let k = beta_dk.ncols();
    let mut w = Mat::zeros(n_genes, k);

    for (&gi, &pi) in indicator_genes.iter().zip(indicator_peaks.iter()) {
        for kk in 0..k {
            w[(gi, kk)] += beta_dk[(pi, kk)];
        }
    }

    w
}

/// Sample Poisson counts with vectorized rates and rayon parallelism.
///
/// Pre-computes effective dictionary `eff = γ ⊙ β`, then parallelises over
/// cells:
///   `rate_j = depth_j · δ_{:, b(j)} · eff · θ_j`     (when `delta_db` set),
///   `rate_j = depth_j ·                  eff · θ_j`   (otherwise).
/// `delta_db` is `exp(log δ)` in mean space; the caller is responsible for
/// applying any PVE / log-normal centering before exponentiating.
pub fn sample_poisson_counts(
    beta_dk: &Mat,
    theta_kn: &Mat,
    cell_depths: &[f32],
    gamma_gk: Option<&Mat>,
    delta_db: Option<&Mat>,
    batch_membership: Option<&[usize]>,
    rseed: u64,
) -> Vec<(u64, u64, f32)> {
    let eff_dk: Cow<Mat> = match gamma_gk {
        Some(gamma) => Cow::Owned(gamma.component_mul(beta_dk)),
        None => Cow::Borrowed(beta_dk),
    };

    cell_depths
        .par_iter()
        .enumerate()
        .flat_map_iter(|(j, &depth_j)| {
            let mut rng = StdRng::seed_from_u64(rseed.wrapping_add(j as u64));
            let mut rate_vec = &*eff_dk * theta_kn.column(j) * depth_j;
            if let (Some(delta), Some(memb)) = (delta_db, batch_membership) {
                let b = memb[j];
                rate_vec.component_mul_assign(&delta.column(b));
            }

            let mut out = Vec::new();
            for (i, &rate) in rate_vec.iter().enumerate() {
                if rate > 1e-6 {
                    let count = sample_poisson_safe(rate, &mut rng);
                    if count > 0.0 {
                        out.push((i as u64, j as u64, count));
                    }
                }
            }
            out
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
