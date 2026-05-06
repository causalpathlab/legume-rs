use indicatif::ParallelProgressIterator;
use log::info;
use matrix_util::dmatrix_io::*;
use matrix_util::dmatrix_util::row_membership_matrix;
use matrix_util::traits::*;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal, Poisson, Uniform};
use rayon::prelude::*;

/// Sample topic proportions `[K, N]` with hard assignments softened by PVE.
///
/// Each cell is hard-assigned to a topic `k*`, then mixed with a uniform
/// background:
///   `θ[k*] = pve_topic + (1 - pve_topic)/K`,
///   `θ[k]  =             (1 - pve_topic)/K`   for `k ≠ k*`.
/// Columns sum to `1` for every value of `pve_topic ∈ [0, 1]`. At
/// `pve_topic = 0` every column is uniform `1/K` (no topic structure);
/// at `pve_topic = 1` every column is one-hot.
pub(crate) fn sample_theta_kn(
    kk: usize,
    nn: usize,
    pve_topic: f32,
    rng: &mut impl rand::Rng,
) -> anyhow::Result<DMatrix<f32>> {
    let runif = Uniform::new(0, kk)?;
    let k_membership: Vec<usize> = (0..nn).map(|_| runif.sample(rng)).collect();
    let mut theta_kn: DMatrix<f32> = row_membership_matrix(k_membership)?.transpose();

    let pve_topic = pve_topic.clamp(0., 1.);
    if kk > 1 && pve_topic < 1. {
        let p_background = (1.0 - pve_topic) / kk as f32;
        let theta_null = DMatrix::<f32>::from_element(kk, nn, p_background);
        theta_kn = (theta_kn * pve_topic) + theta_null;
    }

    Ok(theta_kn)
}

/// Sample a log-normal dictionary `β[D, K]` with explicit topic-PVE
/// decomposition:
///
/// ```text
/// log β_{g,k} = σ_β · [√π_topic · u_{g,k} + √(1−π_topic) · v_g] − σ_β² / 2
/// ```
///
/// with `u_{g,k}, v_g ~ N(0, 1)` iid. By construction:
///   - `Var(log β_{g,k}) = σ_β²` independent of `π_topic`,
///   - `E[β_{g,k}] = 1` (multiplicative-perturbation interpretation, so a
///     downstream `(depth/G)·β·θ` makes `depth` an emergent library size),
///   - `π_topic = 0` ⇒ β purely per-gene (no topic structure),
///   - `π_topic = 1` ⇒ β purely per-(gene, topic) (no shared baseline).
pub(crate) fn sample_lognormal_dictionary(
    dd: usize,
    kk: usize,
    pve_topic: f32,
    beta_scale: f32,
    rng: &mut impl rand::Rng,
) -> DMatrix<f32> {
    let pve = pve_topic.clamp(0.0, 1.0);
    let a_topic = pve.sqrt();
    let a_invariant = (1.0 - pve).sqrt();
    let center = 0.5 * beta_scale * beta_scale;
    let normal = Normal::new(0.0_f32, 1.0_f32).expect("standard normal");

    let v: Vec<f32> = (0..dd).map(|_| normal.sample(rng)).collect();

    let mut beta = DMatrix::<f32>::zeros(dd, kk);
    for g in 0..dd {
        for k in 0..kk {
            let u: f32 = normal.sample(rng);
            let log_b = beta_scale * (a_topic * u + a_invariant * v[g]) - center;
            beta[(g, k)] = log_b.exp();
        }
    }
    beta
}

/// Sample batch-effect log-shifts `log δ ∈ ℝ^{D × B}` with explicit
/// log-space variance decomposition:
///
/// ```text
/// log δ_{g, b} = √π_batch · z_{g, b} + √(1 − π_batch) · w_g
/// ```
///
/// `z` iid N(0, 1) per (gene, batch), z-scored per column to unit variance,
/// scaled by √π_batch. `w` iid N(0, 1) per gene, z-scored to unit variance,
/// scaled by √(1 − π_batch). By construction `Var(log δ_{g, b}) = 1`
/// independent of `π_batch`. Caller exponentiates if mean-space δ is needed.
pub(crate) fn sample_log_batch_effects(
    dd: usize,
    bb: usize,
    pve_batch: f32,
    rng: &mut impl rand::Rng,
) -> DMatrix<f32> {
    let pve = pve_batch.clamp(0.0, 1.0);
    let normal = Normal::new(0.0_f32, 1.0_f32).expect("standard normal");

    let mut ln_delta_db = DMatrix::from_fn(dd, bb, |_, _| normal.sample(rng));
    ln_delta_db.scale_columns_inplace();
    ln_delta_db *= pve.sqrt();

    let mut ln_null_d = DMatrix::from_fn(dd, 1, |_, _| normal.sample(rng));
    ln_null_d.scale_columns_inplace();
    ln_null_d *= (1.0 - pve).sqrt();

    for col in 0..ln_delta_db.ncols() {
        let mut col_mut = ln_delta_db.column_mut(col);
        col_mut += &ln_null_d.column(0);
    }

    ln_delta_db
}

/// Inject housekeeping genes (first `n_housekeeping` rows) with high,
/// approximately-uniform-across-topics expression. Each housekeeping gene
/// `g` gets a single log-normal draw `β_g ~ LN(log(fold·median(β)), σ_hk²)`
/// shared across all `K` topics — the per-gene baseline is correlated, the
/// per-topic noise is gone (these are by-design topic-invariant genes).
pub(crate) fn inject_housekeeping(
    beta_dk: &mut DMatrix<f32>,
    n_housekeeping: usize,
    fold: f32,
    rng: &mut impl rand::Rng,
) {
    if n_housekeeping == 0 || n_housekeeping >= beta_dk.nrows() {
        return;
    }
    let kk = beta_dk.ncols();
    // Pivot: per-gene mean (across topics) — robust enough for a baseline.
    let mean_val = beta_dk.mean().max(1e-30);
    let hk_mean = mean_val * fold;
    const HK_SIGMA: f32 = 0.3;
    let log_mean = hk_mean.ln() - 0.5 * HK_SIGMA * HK_SIGMA;
    let normal = Normal::new(0.0_f32, 1.0_f32).expect("standard normal");
    for g in 0..n_housekeeping {
        let u: f32 = normal.sample(rng);
        let val = (log_mean + HK_SIGMA * u).exp();
        for k in 0..kk {
            beta_dk[(g, k)] = val;
        }
    }
    info!(
        "injected {} housekeeping genes (LN(log {:.4}, {:.2}), fold={:.1}× mean {:.4})",
        n_housekeeping, hk_mean, HK_SIGMA, fold, mean_val
    );
}

/// Sample Poisson count triplets from
/// `Y(g,j) ~ Poisson( λ_scale · δ(g,B(j)) · Σ_k β(g,k) θ(k,j) )`.
///
/// `lambda_scale` should be `depth / G` so that `E[Σ_g λ_{g,j}] ≈ depth`
/// when `E[β] = E[δ] = 1` (the log-normal+PVE samplers in this module
/// guarantee this). Library size is **emergent**, not enforced — there is
/// no per-cell rescaling.
pub(crate) fn sample_poisson_triplets(
    beta_dk: &DMatrix<f32>,
    theta_kn: &DMatrix<f32>,
    delta_db: Option<&DMatrix<f32>>,
    batch_membership: &[usize],
    lambda_scale: f32,
    rseed: u64,
    seed_offset: u64,
) -> Vec<(u64, u64, f32)> {
    let nn = theta_kn.ncols();
    let eps = 1e-8_f32;
    let threshold = 0.5_f32;
    let has_batch = delta_db.is_some();
    theta_kn
        .column_iter()
        .enumerate()
        .par_bridge()
        .progress_count(nn as u64)
        .map(|(j, theta_j)| {
            let mut rng = rand::rngs::StdRng::seed_from_u64(rseed + seed_offset + j as u64);

            let lambda_j = if has_batch {
                let b = batch_membership[j];
                (beta_dk * theta_j).component_mul(&delta_db.unwrap().column(b))
            } else {
                beta_dk * theta_j
            };

            lambda_j
                .iter()
                .enumerate()
                .filter_map(|(i, &l_ij)| {
                    let l_ij = (l_ij * lambda_scale).max(eps);
                    if let Ok(rpois) = Poisson::new(l_ij) {
                        let y_ij: f32 = rpois.sample(&mut rng);
                        if y_ij > threshold {
                            Some((i as u64, j as u64, y_ij))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        })
        .flatten()
        .collect()
}

pub struct SimArgs {
    pub rows: usize,
    pub cols: usize,
    pub depth: usize,
    pub factors: usize,
    pub batches: usize,
    /// Log-normal scale parameter `σ_β` for the dictionary. Replaces the
    /// old `overdisp` knob. Higher = more variable expression across genes
    /// and topics. Default 1.0.
    pub beta_scale: f32,
    pub pve_topic: f32,
    pub pve_batch: f32,
    pub rseed: u64,
    /// If set, generate a hierarchical gene dictionary using stick-breaking
    /// gates on a binary tree of this depth. K = 2^(depth-1) leaf topics.
    /// Overrides `factors` when set.
    pub hierarchical_depth: Option<usize>,
    /// Number of housekeeping genes with high uniform expression across all
    /// topics. These genes get a single log-normal value broadcast across
    /// all topics, simulating ubiquitous genes (B2M, EEF1A1, ribosomal).
    pub n_housekeeping: usize,
    /// Expression fold-change of housekeeping genes relative to the mean
    /// topic-specific gene. Default: 10.0
    pub housekeeping_fold: f32,
}

pub struct SimOut {
    pub ln_delta_db: DMatrix<f32>,
    pub beta_dk: DMatrix<f32>,
    pub theta_kn: DMatrix<f32>,
    pub batch_membership: Vec<usize>,
    pub triplets: Vec<(u64, u64, f32)>,
    /// If hierarchical: all node probabilities [D, num_nodes] (1-indexed by column)
    pub hierarchy_node_probs: Option<DMatrix<f32>>,
}

/// Generate a simulated dataset with a log-normal factored model.
///
/// ```text
/// log β_{g,k} = σ_β · [√π_topic · u_{g,k} + √(1−π_topic) · v_g] − σ_β² / 2
/// log δ_{g,b} = √π_batch · z_{g,b} + √(1−π_batch) · w_g
/// λ_{g,j}    = (depth / G) · δ_{g, B(j)} · Σ_k β_{g,k} θ_{k,j}
/// y_{g,j}    ~ Poisson(λ_{g,j})
/// ```
///
/// `depth` is the **expected** library size (emergent — no per-cell
/// rescaling). `pve_topic` and `pve_batch` are independent variance shares
/// in log space — both can be 1 simultaneously.
pub fn generate_factored_poisson_gamma_data(args: &SimArgs) -> anyhow::Result<SimOut> {
    let nn = args.cols;
    let dd = args.rows;
    let kk = if let Some(depth) = args.hierarchical_depth {
        1usize << (depth - 1)
    } else {
        args.factors
    };
    let bb = args.batches;
    let rseed = args.rseed;
    let beta_scale = args.beta_scale;
    let pve_topic = args.pve_topic.clamp(0., 1.);
    let pve_batch = args.pve_batch.clamp(0., 1.);

    let mut rng = rand::rngs::StdRng::seed_from_u64(rseed);

    // 1. batch membership matrix
    let runif = Uniform::new(0, bb).expect("unif [0 .. bb)");
    let batch_membership: Vec<usize> = (0..nn).map(|_| runif.sample(&mut rng)).collect();

    let ln_delta_db = sample_log_batch_effects(dd, bb, pve_batch, &mut rng);
    let delta_db = ln_delta_db.map(|x| x.exp());
    info!("simulated batch effects (log-variance decomposition)");

    // 3. dictionary β
    let (mut beta_dk, hierarchy_node_probs) = if let Some(tree_depth) = args.hierarchical_depth {
        let (beta, node_probs) =
            generate_hierarchical_dictionary(dd, tree_depth, beta_scale, &mut rng);
        info!(
            "generated hierarchical dictionary: depth={}, K={} leaves, {} nodes",
            tree_depth,
            beta.ncols(),
            node_probs.ncols()
        );
        (beta, Some(node_probs))
    } else {
        (
            sample_lognormal_dictionary(dd, kk, pve_topic, beta_scale, &mut rng),
            None,
        )
    };

    // 4. housekeeping injection (log-normal, topic-invariant per gene)
    inject_housekeeping(
        &mut beta_dk,
        args.n_housekeeping,
        args.housekeeping_fold,
        &mut rng,
    );

    let theta_kn = sample_theta_kn(kk, nn, pve_topic, &mut rng)?;

    // 5. emergent-library Poisson sampling
    let lambda_scale = (args.depth as f32) / (dd as f32);
    let delta_ref = if bb > 1 { Some(&delta_db) } else { None };
    let triplets = sample_poisson_triplets(
        &beta_dk,
        &theta_kn,
        delta_ref,
        &batch_membership,
        lambda_scale,
        rseed,
        0,
    );

    info!(
        "sampled Poisson data with {} non-zero elements (λ_scale = depth/G = {:.4})",
        triplets.len(),
        lambda_scale,
    );

    Ok(SimOut {
        ln_delta_db,
        beta_dk,
        theta_kn,
        batch_membership,
        triplets,
        hierarchy_node_probs,
    })
}

/// Generate a hierarchical gene dictionary using stick-breaking gates on a
/// binary tree, mirroring the `HierarchicalTopicDecoder` structure.
///
/// * `dd` - number of genes (D)
/// * `tree_depth` - tree depth (>= 2), K = 2^(depth-1) leaf topics
/// * `beta_scale` - log-normal scale `σ_β` for the root distribution; gate
///   logits use a fixed `N(0, 1)` scale (sharper subtrees come from larger
///   `σ_β` driving the root, not from this gate scale).
/// * `rng` - random number generator
///
/// Returns `(beta_dk, node_probs_d_by_numnodes)`:
/// - `beta_dk`: [D, K] leaf dictionary (each row sums to ~1)
/// - `node_probs`: [D, num_nodes] all node probabilities (col 0 = root, etc.)
///
/// Note: the tree's stick-breaking already encodes topic structure by
/// construction, so `pve_topic` does NOT additionally blend β here — only
/// θ blends. (Flat mode applies the explicit log-space topic-PVE blend.)
pub(crate) fn generate_hierarchical_dictionary(
    dd: usize,
    tree_depth: usize,
    beta_scale: f32,
    rng: &mut impl rand::Rng,
) -> (DMatrix<f32>, DMatrix<f32>) {
    assert!(tree_depth >= 2, "Tree depth must be at least 2");

    let num_leaves = 1usize << (tree_depth - 1); // K = 2^(depth-1)
    let num_nodes = (1usize << tree_depth) - 1; // 2^depth - 1
    let num_internal = num_leaves - 1;

    let normal = Normal::new(0.0_f32, 1.0_f32).expect("standard normal");

    // 1. Root distribution: log-normal then normalized to probability simplex
    let center = 0.5 * beta_scale * beta_scale;
    let root_raw: Vec<f32> = (0..dd)
        .map(|_| (beta_scale * normal.sample(rng) - center).exp())
        .collect();
    let root_sum: f32 = root_raw.iter().sum::<f32>().max(1e-30);
    let root_prob: Vec<f32> = root_raw.iter().map(|&x| x / root_sum).collect();

    // 2. Gate logits ~ N(0, 1) → sigmoid gives gates in (0, 1).
    let gate_logits: Vec<Vec<f32>> = (0..num_internal)
        .map(|_| (0..dd).map(|_| normal.sample(rng)).collect())
        .collect();

    // 3. Top-down propagation: compute node probabilities (1-indexed: 0 = unused).
    let mut node_probs: Vec<Vec<f32>> = vec![vec![0.0; dd]; num_nodes + 1];
    node_probs[1] = root_prob;

    for h in 1..num_leaves {
        let gate_idx = h - 1;
        let left_child = 2 * h;
        let right_child = 2 * h + 1;

        let mut left = vec![0.0_f32; dd];
        let mut right = vec![0.0_f32; dd];

        for d in 0..dd {
            let gate = sigmoid(gate_logits[gate_idx][d]);
            left[d] = node_probs[h][d] * gate;
            right[d] = node_probs[h][d] * (1.0 - gate);
        }

        node_probs[left_child] = left;
        node_probs[right_child] = right;
    }

    // 4. Extract leaf dictionary [D, K]
    let mut beta_dk = DMatrix::<f32>::zeros(dd, num_leaves);
    for k in 0..num_leaves {
        let leaf_node = num_leaves + k;
        for d in 0..dd {
            beta_dk[(d, k)] = node_probs[leaf_node][d];
        }
    }

    // 5. Pack all node probabilities into [D, num_nodes]
    let mut all_node_probs = DMatrix::<f32>::zeros(dd, num_nodes);
    for h in 0..num_nodes {
        for d in 0..dd {
            all_node_probs[(d, h)] = node_probs[h + 1][d];
        }
    }

    (beta_dk, all_node_probs)
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
