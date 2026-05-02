use log::info;
use matrix_util::dmatrix_io::*;
use matrix_util::traits::*;
use rand::seq::index::sample;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal, Uniform};

use super::core::{generate_hierarchical_dictionary, sample_poisson_triplets, sample_theta_kn};

pub struct MultimodalSimArgs {
    /// Number of features (genes), shared across modalities
    pub rows: usize,
    /// Number of cells/samples
    pub cols: usize,
    /// Expected depth (non-zero count) per cell, one per modality.
    /// Length defines M (number of modalities).
    pub depth_per_modality: Vec<usize>,
    /// Number of latent topics
    pub factors: usize,
    /// Number of batches
    pub batches: usize,
    /// Scale of base logits W_base ~ N(0, base_scale)
    pub base_scale: f32,
    /// Scale of non-zero delta entries ~ N(0, delta_scale)
    pub delta_scale: f32,
    /// Number of non-zero delta genes per topic (spike-and-slab)
    pub n_delta_features: usize,
    /// PVE for topic structure
    pub pve_topic: f32,
    /// PVE for batch effects
    pub pve_batch: f32,
    /// Random seed
    pub rseed: u64,
    /// Whether batch effects are shared across modalities
    pub shared_batch_effects: bool,
    /// Optional hierarchical tree depth for W_base
    pub hierarchical_depth: Option<usize>,
    /// Overdispersion for hierarchical dictionary
    pub overdisp: f32,
    /// Number of housekeeping genes
    pub n_housekeeping: usize,
    /// Housekeeping fold change
    pub housekeeping_fold: f32,
}

pub struct MultimodalSimOut {
    /// Base logits [K, D] (before softmax)
    pub w_base_kd: DMatrix<f32>,
    /// Delta logits per non-reference modality [K, D] x (M-1)
    pub w_delta_kd: Vec<DMatrix<f32>>,
    /// Effective dictionaries (after softmax) [D, K] x M
    pub beta_dk: Vec<DMatrix<f32>>,
    /// Spike masks [K, D] with 0/1 entries, x (M-1)
    pub spike_mask_kd: Vec<DMatrix<f32>>,
    /// Shared topic proportions [K, N]
    pub theta_kn: DMatrix<f32>,
    /// Batch effects [D, B] per modality (always length M)
    pub ln_delta_db: Vec<DMatrix<f32>>,
    /// Batch membership for each cell
    pub batch_membership: Vec<usize>,
    /// Triplets per modality
    pub triplets: Vec<Vec<(u64, u64, f32)>>,
}

/// Generate batch effects [D, B], reusing the pattern from simulate.rs.
fn generate_batch_effects(
    dd: usize,
    bb: usize,
    pve_batch: f32,
    rng: &mut impl rand::Rng,
) -> DMatrix<f32> {
    if bb <= 1 {
        return DMatrix::<f32>::zeros(dd, bb.max(1));
    }
    let pve_batch = pve_batch.clamp(0., 1.);
    let normal = Normal::new(0.0f32, 1.0).expect("standard normal");

    let mut ln_delta_db = DMatrix::from_fn(dd, bb, |_, _| normal.sample(rng));
    ln_delta_db.scale_columns_inplace();
    ln_delta_db *= pve_batch.sqrt();

    let mut ln_null_d = DMatrix::from_fn(dd, 1, |_, _| normal.sample(rng));
    ln_null_d.scale_columns_inplace();
    ln_null_d *= (1.0 - pve_batch).clamp(0., 1.).sqrt();

    for col in 0..ln_delta_db.ncols() {
        let mut col_mut = ln_delta_db.column_mut(col);
        col_mut += &ln_null_d.column(0);
    }

    ln_delta_db
}

pub fn generate_multimodal_data(args: &MultimodalSimArgs) -> anyhow::Result<MultimodalSimOut> {
    let dd = args.rows;
    let nn = args.cols;
    let kk = if let Some(depth) = args.hierarchical_depth {
        1usize << (depth - 1)
    } else {
        args.factors
    };
    let bb = args.batches;
    let mm = args.depth_per_modality.len();
    let eps = 1e-8_f32;

    anyhow::ensure!(mm >= 1, "need at least 1 modality (depth_per_modality)");
    anyhow::ensure!(bb >= 1, "batches must be >= 1");
    anyhow::ensure!(
        args.n_delta_features <= dd,
        "n_delta_features ({}) must be <= rows ({})",
        args.n_delta_features,
        dd
    );

    let mut rng = rand::rngs::StdRng::seed_from_u64(args.rseed);

    // 1. Batch membership
    let runif_batch = Uniform::new(0, bb)?;
    let batch_membership: Vec<usize> = (0..nn).map(|_| runif_batch.sample(&mut rng)).collect();

    // 2. Batch effects (always M entries; shared = replicate the same matrix)
    let ln_delta_db: Vec<DMatrix<f32>> = if args.shared_batch_effects {
        let shared = generate_batch_effects(dd, bb, args.pve_batch, &mut rng);
        vec![shared; mm]
    } else {
        (0..mm)
            .map(|_| generate_batch_effects(dd, bb, args.pve_batch, &mut rng))
            .collect()
    };

    // 3. W_base [K, D]
    let w_base_kd = if let Some(tree_depth) = args.hierarchical_depth {
        let (beta_dk, _node_probs) =
            generate_hierarchical_dictionary(dd, tree_depth, args.overdisp, &mut rng);
        info!(
            "hierarchical base dictionary: depth={}, K={} leaves",
            tree_depth,
            beta_dk.ncols()
        );
        // Convert simplex probabilities to logits: log(beta) then transpose to [K, D]
        let mut logits_kd = DMatrix::<f32>::zeros(kk, dd);
        for k in 0..kk {
            for d in 0..dd {
                logits_kd[(k, d)] = beta_dk[(d, k)].max(eps).ln();
            }
        }
        logits_kd
    } else {
        let normal = Normal::new(0.0f32, args.base_scale)?;
        DMatrix::from_fn(kk, dd, |_, _| normal.sample(&mut rng))
    };

    // 4. Housekeeping: set first n_housekeeping columns of W_base to uniform high values
    let mut w_base_kd = w_base_kd;
    if args.n_housekeeping > 0 && args.n_housekeeping < dd {
        let hk_val = args.housekeeping_fold * args.base_scale;
        for d in 0..args.n_housekeeping {
            for k in 0..kk {
                w_base_kd[(k, d)] = hk_val;
            }
        }
        info!(
            "injected {} housekeeping genes with logit value {:.2}",
            args.n_housekeeping, hk_val
        );
    }

    // 5. W_delta_m [K, D] for each non-reference modality
    let normal_delta = Normal::new(0.0f32, args.delta_scale)?;
    let mut w_delta_kd: Vec<DMatrix<f32>> = Vec::with_capacity(mm.saturating_sub(1));
    let mut spike_mask_kd: Vec<DMatrix<f32>> = Vec::with_capacity(mm.saturating_sub(1));

    for _ in 1..mm {
        let mut delta = DMatrix::<f32>::zeros(kk, dd);
        let mut mask = DMatrix::<f32>::zeros(kk, dd);

        for k in 0..kk {
            // Sample n_delta_features gene indices without replacement
            let chosen = sample(&mut rng, dd, args.n_delta_features);
            for d in chosen {
                delta[(k, d)] = normal_delta.sample(&mut rng);
                mask[(k, d)] = 1.0;
            }
        }

        w_delta_kd.push(delta);
        spike_mask_kd.push(mask);
    }

    info!(
        "generated {} delta matrices: {} non-zero features per topic out of {}",
        w_delta_kd.len(),
        args.n_delta_features,
        dd
    );

    // 6. Effective dictionaries via column-wise softmax
    //    logits [K, D] → transpose to [D, K] → softmax each column (over D)
    let mut beta_dk: Vec<DMatrix<f32>> = Vec::with_capacity(mm);

    // Reference modality (m=0): softmax(W_base)
    beta_dk.push(w_base_kd.transpose().normalize_exp_logits_columns());

    // Non-reference modalities: softmax(W_base + W_delta_m)
    for delta in &w_delta_kd {
        let logits = &w_base_kd + delta;
        beta_dk.push(logits.transpose().normalize_exp_logits_columns());
    }

    // 7. Shared theta [K, N]
    let theta_kn = sample_theta_kn(kk, nn, args.pve_topic, &mut rng)?;

    // 8. Poisson counts per modality
    let rseed = args.rseed;
    let mut triplets: Vec<Vec<(u64, u64, f32)>> = Vec::with_capacity(mm);

    // Pre-compute exp(batch effects) to avoid redundant work in the modality loop
    let delta_exp: Vec<DMatrix<f32>> = ln_delta_db.iter().map(|m| m.map(|x| x.exp())).collect();

    for (m, &depth_m) in args.depth_per_modality.iter().enumerate() {
        let delta_ref = if bb > 1 { Some(&delta_exp[m]) } else { None };
        let seed_offset = (m as u64) * (nn as u64);

        let trips = sample_poisson_triplets(
            &beta_dk[m],
            &theta_kn,
            delta_ref,
            &batch_membership,
            depth_m,
            rseed,
            seed_offset,
        );

        info!(
            "modality {}: {} non-zero triplets (depth={})",
            m,
            trips.len(),
            depth_m
        );
        triplets.push(trips);
    }

    Ok(MultimodalSimOut {
        w_base_kd,
        w_delta_kd,
        beta_dk,
        spike_mask_kd,
        theta_kn,
        ln_delta_db,
        batch_membership,
        triplets,
    })
}
