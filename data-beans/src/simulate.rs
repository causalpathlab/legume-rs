#![allow(dead_code)]

use indicatif::ParallelProgressIterator;
use log::info;
use matrix_util::dmatrix_io::*;
use matrix_util::mtx_io::write_mtx_triplets;
use matrix_util::traits::*;
use matrix_util::{common_io::write_lines, dmatrix_util::row_membership_matrix};
use nalgebra::ComplexField;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal, Poisson, Uniform};
use rayon::prelude::*;

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
    let threshold = 0.5_f32;
    let eps = 1e-8;

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

    let runif = Uniform::new(0, kk)?;
    let k_membership: Vec<usize> = (0..nn).map(|_| runif.sample(&mut rng)).collect();

    let mut theta_kn: DMatrix<f32> = row_membership_matrix(k_membership)?.transpose();

    if kk > 1 && pve_topic < 1. {
        let denom = (kk - 1) as f32;
        let p_background = (1.0 - pve_topic) / denom;
        let theta_null = DMatrix::<f32>::from_element(kk, nn, p_background);
        theta_kn = (theta_kn * pve_topic) + theta_null;
    }

    // let (a, b) = (1. / overdisp, (kk as f32).sqrt() * overdisp);
    // let theta_kn = DMatrix::<f32>::rgamma(kk, nn, (a, b));

    // 4. putting them all together
    // let mut triplets = vec![];

    let triplets = theta_kn
        .column_iter()
        .enumerate()
        .par_bridge()
        .progress_count(nn as u64)
        .map(|(j, theta_j)| {
            let mut rng = rand::rngs::StdRng::seed_from_u64(rseed + j as u64);
            let b = batch_membership[j]; // batch index

            let lambda_j = if bb > 1 {
                (&beta_dk * theta_j).component_mul(&delta_db.column(b))
            } else {
                &beta_dk * theta_j
            };

            let tot = lambda_j.sum();
            let scale = (nnz as f32) / tot;

            lambda_j
                .iter()
                .enumerate()
                .filter_map(|(i, &l_ij)| {
                    let l_ij = (l_ij * scale).max(eps);
                    if let Ok(rpois) = Poisson::new(l_ij) {
                        let y_ij = rpois.sample(&mut rng);
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
        .collect::<Vec<_>>();

    info!(
        "sampled Poisson data with {} non-zero elements",
        triplets.len()
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
/// * `overdisp` - overdispersion parameter (controls gate sharpness)
/// * `rng` - random number generator
///
/// Returns `(beta_dk, node_probs_d_by_numnodes)`:
/// - `beta_dk`: [D, K] leaf dictionary (each column sums to ~1)
/// - `node_probs`: [D, num_nodes] all node probabilities (col 0 = root, etc.)
fn generate_hierarchical_dictionary(
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
