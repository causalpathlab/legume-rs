#![allow(dead_code)]

use indicatif::ParallelProgressIterator;
use log::info;
use matrix_util::dmatrix_io::*;
use matrix_util::mtx_io::write_mtx_triplets;
use matrix_util::traits::*;
use matrix_util::{common_io::write_lines, dmatrix_util::row_membership_matrix};
use nalgebra::ComplexField;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Normal, Poisson, Uniform};
use rayon::prelude::*;

/// Sample topic proportions [K, N] with hard assignments softened by PVE.
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
        let denom = (kk - 1) as f32;
        let p_background = (1.0 - pve_topic) / denom;
        let theta_null = DMatrix::<f32>::from_element(kk, nn, p_background);
        theta_kn = (theta_kn * pve_topic) + theta_null;
    }

    Ok(theta_kn)
}

/// Sample Poisson count triplets from `Y(i,j) ~ Poisson(depth * delta(i,B(j)) * sum_k beta(i,k) * theta(k,j))`.
///
/// CNV effects should be pre-multiplied into `beta_dk` before calling.
pub(crate) fn sample_poisson_triplets(
    beta_dk: &DMatrix<f32>,
    theta_kn: &DMatrix<f32>,
    delta_db: Option<&DMatrix<f32>>,
    batch_membership: &[usize],
    depth: usize,
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

            let tot = lambda_j.sum();
            let scale = (depth as f32) / tot;

            lambda_j
                .iter()
                .enumerate()
                .filter_map(|(i, &l_ij)| {
                    let l_ij = (l_ij * scale).max(eps);
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
    pub overdisp: f32,
    pub pve_topic: f32,
    pub pve_batch: f32,
    pub rseed: u64,
    /// If set, generate a hierarchical gene dictionary using stick-breaking
    /// gates on a binary tree of this depth. K = 2^(depth-1) leaf topics.
    /// Overrides `factors` when set.
    pub hierarchical_depth: Option<usize>,
    /// Number of housekeeping genes with high uniform expression across all topics.
    /// These genes get a large, equal dictionary value in every topic, simulating
    /// ubiquitous genes like B2M, EEF1A1, ribosomal proteins.
    pub n_housekeeping: usize,
    /// Expression fold-change of housekeeping genes relative to the mean
    /// topic-specific gene. Default: 10.0
    pub housekeeping_fold: f32,
    /// CNV simulation: number of chromosomes to distribute genes across.
    /// Set to 0 to disable CNV simulation. Default: 0.
    pub n_chromosomes: usize,
    /// CNV simulation: expected number of CNV events per chromosome.
    /// Default: 0.5
    pub cnv_events_per_chr: f32,
    /// CNV simulation: mean block size as fraction of genes per chromosome.
    /// Default: 0.15
    pub cnv_block_frac: f32,
    /// CNV simulation: fold-change for gain events (e.g., 2.0 = 2x expression).
    /// Default: 2.0
    pub cnv_gain_fold: f32,
    /// CNV simulation: fold-change for loss events (e.g., 0.5 = half expression).
    /// Default: 0.5
    pub cnv_loss_fold: f32,
}

pub struct SimOut {
    pub ln_delta_db: DMatrix<f32>,
    pub beta_dk: DMatrix<f32>,
    pub theta_kn: DMatrix<f32>,
    pub batch_membership: Vec<usize>,
    pub triplets: Vec<(u64, u64, f32)>,
    /// If hierarchical: all node probabilities [D, num_nodes] (1-indexed by column)
    pub hierarchy_node_probs: Option<DMatrix<f32>>,
    /// CNV: per-gene chromosome assignment (length D). None if n_chromosomes == 0.
    pub gene_chromosomes: Option<Vec<Box<str>>>,
    /// CNV: per-gene position within chromosome (length D). None if n_chromosomes == 0.
    pub gene_positions: Option<Vec<u64>>,
    /// CNV: per-gene CN state (0=loss, 1=neutral, 2=gain). None if n_chromosomes == 0.
    pub cnv_states: Option<Vec<u8>>,
}

/// CNV block simulation output.
pub(crate) struct CnvSimOut {
    pub(crate) cnv_multiplier: Vec<f32>,
    pub(crate) cnv_states: Vec<u8>,
    pub(crate) chromosomes: Vec<Box<str>>,
    pub(crate) positions: Vec<u64>,
}

pub(crate) struct CnvSimParams {
    pub(crate) n_genes: usize,
    pub(crate) n_chr: usize,
    pub(crate) events_per_chr: f32,
    pub(crate) block_frac: f32,
    pub(crate) gain_fold: f32,
    pub(crate) loss_fold: f32,
}

/// Generate CNV blocks: assign genes to chromosomes with positions,
/// then place contiguous gain/loss events.
pub(crate) fn sample_cnv_blocks(params: &CnvSimParams, rng: &mut impl rand::Rng) -> CnvSimOut {
    let dd = params.n_genes;
    let n_chr = params.n_chr;
    let genes_per_chr = dd / n_chr;
    let spacing = 10_000u64; // bp between genes

    let mut chromosomes = Vec::with_capacity(dd);
    let mut positions = Vec::with_capacity(dd);
    let mut cnv_states = vec![1u8; dd]; // neutral by default
    let mut cnv_multiplier = vec![1.0f32; dd];

    let poisson_events =
        Poisson::new(params.events_per_chr as f64).unwrap_or_else(|_| Poisson::new(0.5).unwrap());

    for chr_idx in 0..n_chr {
        let chr_name: Box<str> = format!("chr{}", chr_idx + 1).into();
        let chr_start = chr_idx * genes_per_chr;
        let chr_end = if chr_idx == n_chr - 1 {
            dd
        } else {
            (chr_idx + 1) * genes_per_chr
        };
        let chr_len = chr_end - chr_start;

        for g in chr_start..chr_end {
            chromosomes.push(chr_name.clone());
            positions.push((g - chr_start) as u64 * spacing);
        }

        // Sample number of CNV events on this chromosome
        let n_events = poisson_events.sample(rng) as usize;
        let block_size = ((chr_len as f32 * params.block_frac) as usize).max(3);

        for _ in 0..n_events {
            let start = rng.random_range(0..chr_len);
            let end = (start + block_size).min(chr_len);
            let is_gain: bool = rng.random_bool(0.5);
            let (state, mult) = if is_gain {
                (2u8, params.gain_fold)
            } else {
                (0u8, params.loss_fold)
            };

            for g in (chr_start + start)..(chr_start + end) {
                cnv_states[g] = state;
                cnv_multiplier[g] = mult;
            }
        }
    }

    info!(
        "CNV simulation: {} genes, {} chromosomes, {} gain, {} loss",
        dd,
        n_chr,
        cnv_states.iter().filter(|&&s| s == 2).count(),
        cnv_states.iter().filter(|&&s| s == 0).count(),
    );

    CnvSimOut {
        cnv_multiplier,
        cnv_states,
        chromosomes,
        positions,
    }
}

/// Generate a simulated dataset with a factored gamma model
/// * `args`: SimulateArgs
/// * `mtx_file`: output data mtx file (.gz recommended)
/// * `dict_file`: true dictionary file
/// * `prop_file`: true proportion file
/// * `ln_batch_file`: log batch effect file
/// * `batch_file`: true batch membership file
///
/// ```text
/// Y(i,j) ~ Poisson( delta(i, B(j)) * sum_k beta(i,k) * theta(k,j) )
/// ```
///
pub fn generate_factored_poisson_gamma_data_mtx(
    args: &SimArgs,
    mtx_file: &str,
    dict_file: &str,
    prop_file: &str,
    ln_batch_file: &str,
    batch_file: &str,
) -> anyhow::Result<()> {
    let sim = generate_factored_poisson_gamma_data(args)?;

    let batch_out: Vec<Box<str>> = sim
        .batch_membership
        .iter()
        .map(|&x| Box::from(x.to_string()))
        .collect();

    write_lines(&batch_out, batch_file)?;
    info!("batch membership: {:?}", &batch_file);

    sim.ln_delta_db.to_tsv(ln_batch_file)?;
    sim.theta_kn.transpose().to_tsv(prop_file)?;
    sim.beta_dk.to_tsv(dict_file)?;

    info!(
        "wrote parameter files:\n{:?},\n{:?},\n{:?}",
        ln_batch_file, &dict_file, &prop_file
    );

    let mut triplets = sim.triplets;

    info!(
        "sampled Poisson data with {} non-zero elements",
        triplets.len()
    );

    info!("sorting these triplets...");
    triplets.sort_by_key(|&(row, _, _)| row);
    triplets.sort_by_key(|&(_, col, _)| col);

    info!("writing them down to {}", mtx_file);

    let nn = args.cols;
    let dd = args.rows;
    write_mtx_triplets(&triplets, dd, nn, mtx_file)?;
    Ok(())
}

/// Generate a simulated dataset with a factored gamma model
/// * `args`: SimulateArgs
/// * `mtx_file`: output data mtx file (.gz recommended)
/// * `dict_file`: true dictionary file
/// * `prop_file`: true proportion file
/// * `ln_batch_file`: log batch effect file
/// * `batch_file`: true batch membership file
///
/// ```text
/// Y(i,j) ~ Poisson( delta(i, B(j)) * sum_k beta(i,k) * theta(k,j) )
/// ```
///
pub fn generate_factored_poisson_gamma_data(args: &SimArgs) -> anyhow::Result<SimOut> {
    let nn = args.cols;
    let dd = args.rows;
    // When hierarchical, K is determined by tree depth
    let kk = if let Some(depth) = args.hierarchical_depth {
        1usize << (depth - 1)
    } else {
        args.factors
    };
    let bb = args.batches;
    let nnz = args.depth;
    let rseed = args.rseed;
    let overdisp = args.overdisp;
    let pve_topic = args.pve_topic.clamp(0., 1.);
    let pve_batch = args.pve_batch.clamp(0., 1.);

    let mut rng = rand::rngs::StdRng::seed_from_u64(rseed);

    // 1. batch membership matrix
    let runif = Uniform::new(0, bb).expect("unif [0 .. bb)");
    let batch_membership: Vec<usize> = (0..nn).map(|_| runif.sample(&mut rng)).collect();

    // 2. batch effect matrix
    let mut ln_delta_db = DMatrix::<f32>::rnorm(dd, bb);
    ln_delta_db.scale_columns_inplace();
    ln_delta_db *= pve_batch.clamp(0., 1.).sqrt();
    let mut ln_null_d = DMatrix::<f32>::rnorm(dd, 1);
    ln_null_d.scale_columns_inplace();
    ln_null_d *= (1.0 - pve_batch).clamp(0., 1.).sqrt();

    for col in 0..ln_delta_db.ncols() {
        let mut ln_delta_d = ln_delta_db.column_mut(col);
        ln_delta_d += &ln_null_d.column(0);
    }

    let delta_db = ln_delta_db.map(|x| x.exp());
    info!("simulated batch effects");

    // 3. factorization model
    let (beta_dk, hierarchy_node_probs) = if let Some(tree_depth) = args.hierarchical_depth {
        let (beta, node_probs) =
            generate_hierarchical_dictionary(dd, tree_depth, overdisp, &mut rng);
        info!(
            "generated hierarchical dictionary: depth={}, K={} leaves, {} nodes",
            tree_depth,
            beta.ncols(),
            node_probs.ncols()
        );
        (beta, Some(node_probs))
    } else {
        let (a, b) = (1. / overdisp, (kk as f32).sqrt() * overdisp);
        let mut beta_dk = DMatrix::<f32>::rgamma(dd, kk, (a, b));

        if kk > 1 && pve_topic < 1. {
            let beta_null = DMatrix::<f32>::rgamma(dd, 1, (a, b))
                .scale((1.0 - pve_topic).clamp(0., 1.).unscale(kk as f32).sqrt());
            for k in 0..kk {
                let x = beta_dk.column(k).scale(pve_topic.clamp(0., 1.).sqrt()) + &beta_null;
                beta_dk.column_mut(k).copy_from(&x);
            }
        }
        (beta_dk, None)
    };

    // Inject housekeeping genes: first n_housekeeping rows get high expression
    // across all topics, drawn from Gamma with mean = housekeeping_fold * mean_beta
    let beta_dk = if args.n_housekeeping > 0 && args.n_housekeeping < dd {
        let mean_val = beta_dk.mean();
        let hk_mean = mean_val * args.housekeeping_fold;
        // Gamma(shape, rate) with mean = shape/rate = hk_mean, CV ~ 1/sqrt(shape)
        let hk_shape = 2.0_f32; // moderate variability across housekeeping genes
        let hk_rate = hk_shape / hk_mean;
        let hk_dist = rand_distr::Gamma::new(hk_shape, 1.0 / hk_rate).expect("housekeeping gamma");
        let mut beta = beta_dk;
        for g in 0..args.n_housekeeping {
            let base = hk_dist.sample(&mut rng);
            for k in 0..kk {
                beta[(g, k)] = base;
            }
        }
        info!(
            "injected {} housekeeping genes: Gamma(shape={:.1}, mean={:.4}) (fold={:.1}x mean {:.4})",
            args.n_housekeeping, hk_shape, hk_mean, args.housekeeping_fold, mean_val
        );
        beta
    } else {
        beta_dk
    };

    let theta_kn = sample_theta_kn(kk, nn, pve_topic, &mut rng)?;

    // 4. CNV simulation (optional)
    let cnv_out = if args.n_chromosomes > 0 {
        Some(sample_cnv_blocks(
            &CnvSimParams {
                n_genes: dd,
                n_chr: args.n_chromosomes,
                events_per_chr: args.cnv_events_per_chr,
                block_frac: args.cnv_block_frac,
                gain_fold: args.cnv_gain_fold,
                loss_fold: args.cnv_loss_fold,
            },
            &mut rng,
        ))
    } else {
        None
    };
    // Pre-multiply CNV into dictionary if present
    let beta_dk = if let Some(ref cnv) = cnv_out {
        let mut beta = beta_dk;
        for i in 0..dd {
            let scale = cnv.cnv_multiplier[i];
            for k in 0..kk {
                beta[(i, k)] *= scale;
            }
        }
        beta
    } else {
        beta_dk
    };

    // 5. putting them all together
    let delta_ref = if bb > 1 { Some(&delta_db) } else { None };
    let triplets = sample_poisson_triplets(
        &beta_dk,
        &theta_kn,
        delta_ref,
        &batch_membership,
        nnz,
        rseed,
        0,
    );

    info!(
        "sampled Poisson data with {} non-zero elements",
        triplets.len()
    );

    let (gene_chromosomes, gene_positions, cnv_states) = match cnv_out {
        Some(cnv) => (
            Some(cnv.chromosomes),
            Some(cnv.positions),
            Some(cnv.cnv_states),
        ),
        None => (None, None, None),
    };

    Ok(SimOut {
        ln_delta_db,
        beta_dk,
        theta_kn,
        batch_membership,
        triplets,
        hierarchy_node_probs,
        gene_chromosomes,
        gene_positions,
        cnv_states,
    })
}

/// Generate a hierarchical gene dictionary using stick-breaking gates on a
/// binary tree, mirroring the `HierarchicalTopicDecoder` structure.
///
/// * `dd` - number of genes (D)
/// * `tree_depth` - tree depth (>= 2), K = 2^(depth-1) leaf topics
/// * `overdisp` - overdispersion parameter (controls gate sharpness)
/// * `rng` - random number generator
///
/// Returns `(beta_dk, node_probs_d_by_numnodes)`:
/// - `beta_dk`: [D, K] leaf dictionary (each column sums to ~1)
/// - `node_probs`: [D, num_nodes] all node probabilities (col 0 = root, etc.)
pub(crate) fn generate_hierarchical_dictionary(
    dd: usize,
    tree_depth: usize,
    overdisp: f32,
    rng: &mut impl rand::Rng,
) -> (DMatrix<f32>, DMatrix<f32>) {
    assert!(tree_depth >= 2, "Tree depth must be at least 2");

    let num_leaves = 1usize << (tree_depth - 1); // K = 2^(depth-1)
    let num_nodes = (1usize << tree_depth) - 1; // 2^depth - 1
    let num_internal = num_leaves - 1;

    // 1. Root distribution: Gamma-drawn then normalized to probability simplex
    let (a, b) = (1. / overdisp, (num_leaves as f32).sqrt() * overdisp);
    let root_raw = DMatrix::<f32>::rgamma(dd, 1, (a, b));
    let root_sum: f32 = root_raw.sum();
    let root_prob: Vec<f32> = root_raw.iter().map(|&x| x / root_sum).collect();

    // 2. Gate logits: Normal(0, overdisp_scale) → sigmoid gives gates in (0,1)
    // Higher overdisp → sharper gates → more distinct subtrees
    let gate_scale = (overdisp / 10.0).clamp(0.1, 5.0);
    let normal = Normal::new(0.0f32, gate_scale).expect("normal distribution");
    let gate_logits: Vec<Vec<f32>> = (0..num_internal)
        .map(|_| (0..dd).map(|_| normal.sample(rng)).collect())
        .collect();

    // 3. Top-down propagation: compute node probabilities
    // node_probs[h] is a Vec<f32> of length dd, 1-indexed (index 0 = node 1)
    let mut node_probs: Vec<Vec<f32>> = vec![vec![0.0; dd]; num_nodes + 1];
    node_probs[1] = root_prob;

    for h in 1..num_leaves {
        let gate_idx = h - 1; // gate_logits index
        let left_child = 2 * h;
        let right_child = 2 * h + 1;

        let mut left = vec![0.0f32; dd];
        let mut right = vec![0.0f32; dd];

        for d in 0..dd {
            let gate = sigmoid(gate_logits[gate_idx][d]);
            left[d] = node_probs[h][d] * gate;
            right[d] = node_probs[h][d] * (1.0 - gate);
        }

        node_probs[left_child] = left;
        node_probs[right_child] = right;
    }

    // 4. Extract leaf dictionary: [D, K]
    let mut beta_dk = DMatrix::<f32>::zeros(dd, num_leaves);
    for k in 0..num_leaves {
        let leaf_node = num_leaves + k; // 1-indexed leaf
        for d in 0..dd {
            beta_dk[(d, k)] = node_probs[leaf_node][d];
        }
    }

    // 5. Pack all node probabilities into [D, num_nodes] matrix
    let mut all_node_probs = DMatrix::<f32>::zeros(dd, num_nodes);
    for h in 0..num_nodes {
        for d in 0..dd {
            all_node_probs[(d, h)] = node_probs[h + 1][d]; // 1-indexed → 0-indexed column
        }
    }

    (beta_dk, all_node_probs)
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
