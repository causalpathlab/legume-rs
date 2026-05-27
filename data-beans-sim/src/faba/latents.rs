//! Latent-variable draws for the `faba` simulator. Pure sampling — no
//! Poisson, no I/O. Outputs every ground-truth quantity downstream
//! Poisson rates and the harness need.

use log::info;
use nalgebra::DMatrix;
use rand::Rng;
use rand_distr::{Distribution, Normal, Uniform};

use super::{default_substrate_weights, FabaArgs, MODALITIES, N_SUBSTRATE_FEATURES};
use crate::core::sample_lognormal_dictionary;
use crate::multiome::sample_nested_topic_proportions;

/// Ground-truth latents for the faba simulator. One owner; passed by
/// reference everywhere downstream so we never duplicate big matrices.
pub struct Latents {
    /// Substrate score, [G × S].
    pub s_g: DMatrix<f32>,
    /// Substrate weight matrix per modality, [M × S].
    pub w_m: DMatrix<f32>,
    /// Per-modality intercept tuned so E[φ_{g,m}] ≈ π_meas[m], [M].
    pub intercept_m: Vec<f32>,
    /// Substrate mask, per modality, length-G vec of bool.
    pub phi: Vec<Vec<bool>>,
    /// Writer/editor program coupling, [M × K_prog]. Sparse (Bernoulli π_A).
    pub a_mk: DMatrix<f32>,
    /// Gene response to programs, [G × K_prog]. Sparse (Bernoulli π_z).
    pub z_gk: DMatrix<f32>,
    /// Per-gene baseline expression, [G].
    pub beta_g: Vec<f32>,
    /// Topic dictionary for mRNA pool, [G × K_topic]. Natural space (E[β]=1).
    pub beta_topic_gk: DMatrix<f32>,
    /// Per-(g, m) base intercept on the modification rate, [G × M].
    pub base_gm: DMatrix<f32>,
    /// Per-(g, m) mixture weights. One DMatrix per modality, [C_m × G].
    pub alpha_per_mod: Vec<DMatrix<f32>>,
    /// Cell-state topic proportions (shared with writer/editor activity), [K_topic × N].
    pub theta_kn: DMatrix<f32>,
    /// Gene labels.
    pub gene_names: Vec<Box<str>>,
    /// Cell labels (shared across all per-modality matrices).
    pub cell_names: Vec<Box<str>>,
}

impl Latents {
    pub fn detail(&self, m: usize, c: usize) -> Box<str> {
        match m {
            0 => match c {
                0 => "spliced".into(),
                1 => "unspliced".into(),
                _ => panic!("count modality has 2 components, got c={}", c),
            },
            _ => format!("component_{}", c).into(),
        }
    }
}

/// Top-level: sample every latent.
pub fn sample_all(args: &FabaArgs, pi_meas: &[f32], rng: &mut impl Rng) -> anyhow::Result<Latents> {
    let g = args.n_genes;
    let n = args.n_cells;
    let m = MODALITIES.len();
    let s = N_SUBSTRATE_FEATURES;
    let k = args.k_topics;
    let c_mod = args.components_per_modifier;

    anyhow::ensure!(k >= 1, "k-topics must be ≥ 1");
    anyhow::ensure!(c_mod >= 1, "components-per-modifier must be ≥ 1");

    let normal01 = Normal::new(0.0_f32, 1.0_f32).unwrap();

    ///////////////////////////////
    // s_g (substrate score) //
    ///////////////////////////////
    let s_g = DMatrix::<f32>::from_fn(g, s, |_, _| normal01.sample(rng));

    /////////////////////////////////
    // w_m (substrate weights) //
    /////////////////////////////////
    let mut w_m = DMatrix::<f32>::zeros(m, s);
    for (mi, row) in default_substrate_weights().iter().enumerate() {
        for (si, &v) in row.iter().enumerate() {
            w_m[(mi, si)] = v;
        }
    }

    ////////////////////////////////////////////////////////
    // intercept_m via binary search to hit π_meas[m] //
    ////////////////////////////////////////////////////////
    let mut intercept_m = vec![0.0_f32; m];
    let mut phi: Vec<Vec<bool>> = vec![vec![false; g]; m];
    let u01 = Uniform::new(0.0_f32, 1.0_f32)?;
    for mi in 0..m {
        if mi == 0 || pi_meas[mi] >= 1.0 {
            intercept_m[mi] = f32::INFINITY;
            phi[mi] = vec![true; g];
            continue;
        }
        if pi_meas[mi] <= 0.0 {
            intercept_m[mi] = f32::NEG_INFINITY;
            phi[mi] = vec![false; g];
            continue;
        }
        let scores: Vec<f32> = (0..g)
            .map(|gi| {
                let mut acc = 0.0_f32;
                for si in 0..s {
                    acc += s_g[(gi, si)] * w_m[(mi, si)];
                }
                acc
            })
            .collect();
        let b = tune_intercept(&scores, pi_meas[mi]);
        intercept_m[mi] = b;
        let mut hits = 0usize;
        for gi in 0..g {
            let p = sigmoid(scores[gi] + b);
            let draw: f32 = u01.sample(rng);
            let on = draw < p;
            phi[mi][gi] = on;
            hits += on as usize;
        }
        info!(
            "  substrate '{}': target {:.3}, intercept {:+.3} → realised {:.3} ({}/{})",
            MODALITIES[mi],
            pi_meas[mi],
            b,
            hits as f32 / g as f32,
            hits,
            g
        );
    }

    /////////////////////////////////////////
    // A_{m, k} writer/editor coupling //
    /////////////////////////////////////////
    // K_prog ≡ K_topic by design (user choice 2026-05-27): the writer/editor
    // axis IS the cell-state axis, so A_{m, k} couples topic k to modality m.
    // Only the modifier rows (mi ≥ 1) carry draws — count has no
    // modification machinery, so its row stays all-zero. Drawing only
    // for modifier rows (rather than for all M and then zeroing row 0)
    // keeps the seed-stable draw order from the pre-refactor version.
    let mut a_mk = DMatrix::<f32>::zeros(m, k);
    let a_modifier = spike_slab_matrix(m - 1, k, args.pi_a, args.sigma_a, &u01, rng);
    for mi in 1..m {
        for ki in 0..k {
            a_mk[(mi, ki)] = a_modifier[(mi - 1, ki)];
        }
    }

    ////////////////////////////////
    // z_{g, k} gene response //
    ////////////////////////////////
    let z_gk = spike_slab_matrix(g, k, args.pi_z, args.sigma_z, &u01, rng);

    ///////////////////////////////
    // β_g per-gene baseline //
    ///////////////////////////////
    let beta_normal = Normal::new(0.0_f32, args.sigma_beta).unwrap();
    let beta_g: Vec<f32> = (0..g).map(|_| beta_normal.sample(rng)).collect();

    /////////////////////////
    // β_topic [G × K] //
    /////////////////////////
    let beta_topic_gk = sample_lognormal_dictionary(g, k, args.pve_topic, args.beta_scale, rng);

    //////////////////////////////////////////
    // base_{g, m} per-(g, m) intercept //
    //////////////////////////////////////////
    let base_normal = Normal::new(0.0_f32, args.sigma_base).unwrap();
    let base_gm = DMatrix::<f32>::from_fn(g, m, |_, _| base_normal.sample(rng));

    ////////////////////////////////////////
    // α mixture weights per modality //
    ////////////////////////////////////////
    let mut alpha_per_mod: Vec<DMatrix<f32>> = Vec::with_capacity(m);
    for mi in 0..m {
        let c = if mi == 0 { 2 } else { c_mod };
        let mut alpha = DMatrix::<f32>::zeros(c, g);
        for gi in 0..g {
            let raws = sample_dirichlet(c, args.alpha_mix, rng);
            for (ci, v) in raws.into_iter().enumerate() {
                alpha[(ci, gi)] = v;
            }
        }
        alpha_per_mod.push(alpha);
    }

    ////////////////////////////////////
    // θ_{k, n} cell-state topics //
    ////////////////////////////////////
    let (theta_kn, _) =
        sample_nested_topic_proportions(k, 1, n, args.pve_topic, 1.0, rng.next_u64());

    ///////////////////////////
    // gene / cell names //
    ///////////////////////////
    let gene_names: Vec<Box<str>> = (0..g)
        .map(|i| format!("gene_{}", i).into_boxed_str())
        .collect();
    let cell_names: Vec<Box<str>> = (0..n)
        .map(|i| format!("cell_{}", i).into_boxed_str())
        .collect();

    Ok(Latents {
        s_g,
        w_m,
        intercept_m,
        phi,
        a_mk,
        z_gk,
        beta_g,
        beta_topic_gk,
        base_gm,
        alpha_per_mod,
        theta_kn,
        gene_names,
        cell_names,
    })
}

/// Pick a held-out subset of substrate-positive (g, m) pairs. Only
/// modifier modalities (m ≥ 1) are eligible — count is always fully
/// emitted.
pub fn sample_held_out(
    phi: &[Vec<bool>],
    held_out_frac: f32,
    rng: &mut impl Rng,
) -> Vec<Vec<bool>> {
    let m = phi.len();
    let g = if m > 0 { phi[0].len() } else { 0 };
    let mut out = vec![vec![false; g]; m];
    if held_out_frac <= 0.0 {
        return out;
    }
    let u01 = Uniform::new(0.0_f32, 1.0_f32).expect("unif");
    for mi in 1..m {
        for gi in 0..g {
            if phi[mi][gi] {
                let u: f32 = u01.sample(rng);
                if u < held_out_frac {
                    out[mi][gi] = true;
                }
            }
        }
        let n_held = out[mi].iter().filter(|x| **x).count();
        let n_sub = phi[mi].iter().filter(|x| **x).count();
        info!(
            "  held-out '{}': {}/{} substrate-positive genes ({:.1}%)",
            MODALITIES[mi],
            n_held,
            n_sub,
            100.0 * n_held as f32 / n_sub.max(1) as f32
        );
    }
    out
}

/// Binary-search the intercept `b` so that `mean(sigmoid(scores + b)) ≈ target`.
/// Robust to extreme score distributions; ~40 iterations to ~1e-4 tolerance
/// on `target`.
fn tune_intercept(scores: &[f32], target: f32) -> f32 {
    if scores.is_empty() {
        return 0.0;
    }
    let mut lo = -50.0_f32;
    let mut hi = 50.0_f32;
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        let m: f32 = scores.iter().map(|&s| sigmoid(s + mid)).sum::<f32>() / scores.len() as f32;
        if m < target {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo).abs() < 1e-4 {
            break;
        }
    }
    0.5 * (lo + hi)
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Sample a Dirichlet(α · 1_k). For α < 1 the draws are sharply
/// unequal (most mass on one component); for α = 1 they're
/// uniform-on-simplex. Delegates to `rand_distr::Dirichlet`.
fn sample_dirichlet(k: usize, alpha: f32, rng: &mut impl Rng) -> Vec<f32> {
    let alphas = vec![alpha as f64; k];
    let dist = rand_distr::multi::Dirichlet::new(&alphas).expect("Dirichlet: alpha > 0, k > 0");
    dist.sample(rng).into_iter().map(|x| x as f32).collect()
}

/// Spike-and-slab matrix: each entry is `Bernoulli(π) · N(0, σ²)`.
/// Shared between the writer/editor coupling `A` and the gene response
/// `z` since both have the same generative form.
fn spike_slab_matrix(
    rows: usize,
    cols: usize,
    pi: f32,
    sigma: f32,
    u01: &Uniform<f32>,
    rng: &mut impl Rng,
) -> DMatrix<f32> {
    let normal = Normal::new(0.0_f32, sigma).expect("normal");
    let mut out = DMatrix::<f32>::zeros(rows, cols);
    for r in 0..rows {
        for c in 0..cols {
            if u01.sample(rng) < pi {
                out[(r, c)] = normal.sample(rng);
            }
        }
    }
    out
}
