//! `senna fne` (Feature Network Embedding) — continuous Miller-Griffiths-
//! Jordan latent-feature link-prediction model trained on an explicit
//! feature-feature edge list (BioGRID / STRING / KEGG / synthetic-lethality
//! / regulatory).
//!
//! Sibling of `senna bge`. Where `bge` builds a bipartite (cell × feature)
//! graph internally from expression counts, `fne` *consumes* a graph and
//! emits per-feature embeddings alone — no cells involved. The result is
//! a first-class input to `senna {bge, masked-topic}`
//! via `--freeze-feature-embedding`, so cells can train on a gene-relation
//! space derived purely from a curated network.
//!
//! Model. Each node g gets a continuous latent vector E_g ∈ ℝ^H, a learned
//! per-dim diagonal gate γ ∈ ℝ^H, and a per-node bias b_g ∈ ℝ. Edge score
//!
//!   s(i, j) = (E_i ⊙ γ) · E_j + b_i + b_j
//!
//! is symmetric by construction (γ diagonal). With γ ≡ 1 this reduces to
//! plain inner-product embedding (DeepWalk / LINE family). Initial γ = 1.
//!
//! Loss. Per-positive aggregation: for each observed edge (i, j),
//!
//!   ℓ(i, j) = log σ(s_pos) + Σ_k log σ(-s_{neg_k})
//!
//! Negative pairs (i, k) are drawn with k ~ degree^α (node2vec convention,
//! α = 0.75). Self-loops rejected; true-positive collisions ignored — α
//! makes the false-negative rate negligible on sparse graphs.
//!
//! Outputs match the shapes the existing `--freeze-feature-embedding`
//! loader expects (`feature_embedding.parquet` + optional
//! `feature_bias.parquet`), so an `fne` run can be plugged straight into
//! downstream cell-side training.

use crate::embed_common::*;
use auxiliary_data::feature_names::FeatureNameKind;
use candle_core::{Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use graph_embedding_util::stop::setup_stop_handler;
use matrix_util::common_io::read_lines_of_words_delim;
use matrix_util::membership::detect_delimiter;
use matrix_util::pair_graph::FeaturePairGraph;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rustc_hash::FxHashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Args, Debug)]
pub struct FneArgs {
    #[arg(
        required = true,
        value_delimiter = ',',
        help = "Feature-feature edge list(s) (TSV/CSV; two columns per line)",
        long_help = "One or more positional paths, comma-separated or space-\
                     separated. Each file is whitespace/comma/tab-delimited; \
                     every line is a pair of feature names. Lines starting \
                     with `#` are skipped; self-loops and duplicates are \
                     dropped silently. When multiple files are given, the \
                     node set is the union of canonical names across all \
                     files and the edge set is the deduplicated union of \
                     pair-edges — handy for combining BioGRID + STRING + \
                     KEGG into a single graph."
    )]
    networks: Vec<Box<str>>,

    #[arg(
        long,
        short,
        required = true,
        help = "Output prefix",
        long_help = "Produces:\n  \
                     {out}.feature_embedding.parquet  D × H per-feature embeddings\n  \
                     {out}.feature_bias.parquet       D × 1 per-feature bias\n  \
                     {out}.gamma.parquet              H × 1 per-dim gate\n  \
                     {out}.log_likelihood.parquet     epoch × 1 loss trace\n  \
                     {out}.senna.json                 run manifest"
    )]
    out: Box<str>,

    #[arg(long, default_value_t = 32, help = "Embedding dimension H")]
    embedding_dim: usize,

    #[arg(short = 'i', long, default_value_t = 200, help = "Training epochs")]
    epochs: usize,

    #[arg(long, default_value_t = 100, help = "Batches per epoch")]
    batches_per_epoch: usize,

    #[arg(long, default_value_t = 1024, help = "Positive edges per minibatch")]
    batch_size: usize,

    #[arg(long, default_value_t = 5, help = "Negative samples per positive")]
    num_negatives: usize,

    #[arg(
        long,
        alias = "lr",
        default_value_t = 0.01,
        help = "AdamW learning rate"
    )]
    learning_rate: f64,

    #[arg(
        long,
        default_value_t = 0.0,
        help = "AdamW decoupled weight decay (applied to E, γ, b). 0 = off."
    )]
    weight_decay: f64,

    #[arg(
        long,
        default_value_t = 0.75,
        help = "Negative sampling exponent α: q(k) ∝ degree(k)^α. \
                node2vec/word2vec default 0.75."
    )]
    neg_alpha: f32,

    #[arg(
        long,
        default_value_t = '_',
        help = "Delimiter for canonical gene-name matching. The last token after \
                splitting on this char is the canonical name, so `ENSG00000_TGFB1` \
                and `TGFB1` merge into a single node."
    )]
    feature_name_delim: char,

    #[arg(
        long,
        default_value_t = false,
        help = "Disable fuzzy gene-name matching (use exact row-name match)"
    )]
    feature_name_exact: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Allow prefix matching during the node→index resolution pass"
    )]
    feature_name_prefix_match: bool,

    #[arg(long, default_value_t = ComputeDevice::Cpu, value_enum, help = "Compute device")]
    device: ComputeDevice,

    #[arg(long, default_value_t = 0, help = "Device ordinal (for cuda/metal)")]
    device_no: usize,

    #[arg(long, default_value_t = 1, help = "RNG seed")]
    seed: u64,
}

/// Trained-model bundle used by the public training function and by
/// the synthetic-graph unit test.
struct TrainedFne {
    e_feat: Tensor, // [N, H]
    b_feat: Tensor, // [N]
    gamma: Tensor,  // [H]
    feature_names: Vec<Box<str>>,
    loss_trace: Vec<f32>,
}

pub fn fit_fne(args: &FneArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;

    let name_kind = if args.feature_name_exact {
        FeatureNameKind::Exact
    } else {
        FeatureNameKind::Gene {
            delim: args.feature_name_delim,
        }
    };

    let feature_names = discover_features_across_files(&args.networks, &name_kind)?;
    info!(
        "fne: {} unique nodes discovered across {} edge file(s)",
        feature_names.len(),
        args.networks.len()
    );

    let graph = load_merged_graph(
        &args.networks,
        feature_names,
        args.feature_name_prefix_match,
        if args.feature_name_exact {
            None
        } else {
            Some(args.feature_name_delim)
        },
    )?;
    anyhow::ensure!(
        graph.num_edges() > 0,
        "fne: 0 usable edges after name resolution across {} file(s)",
        args.networks.len()
    );

    let stop = setup_stop_handler();
    let dev = args.device.to_device(args.device_no)?;
    let trained = train_fne(
        &graph,
        &TrainConfig {
            embedding_dim: args.embedding_dim,
            epochs: args.epochs,
            batches_per_epoch: args.batches_per_epoch,
            batch_size: args.batch_size,
            num_negatives: args.num_negatives,
            learning_rate: args.learning_rate,
            weight_decay: args.weight_decay,
            neg_alpha: args.neg_alpha,
            seed: args.seed,
            dev: &dev,
            stop: stop.clone(),
        },
    )?;

    write_outputs(&trained, &args.out)?;

    let input: Vec<String> = args.networks.iter().map(|s| s.to_string()).collect();
    crate::run_manifest::write_run_manifest(&crate::run_manifest::RunDescription {
        kind: crate::run_manifest::RunKind::Fne,
        prefix: &args.out,
        data_input: &input,
        data_batch: &[],
        data_input_null: &[],
        dictionary_suffix: None,
        has_model: false,
        has_cell_proj: false,
        pb_gene_suffix: None,
        pb_latent_suffix: None,
        dictionary_empirical_suffix: None,
        feature_embedding_suffix: Some("feature_embedding.parquet"),
        default_colour_by: "cluster",
        has_latent: false,
        has_cell_to_pb: false,
    })?;

    if stop.load(Ordering::SeqCst) {
        info!(
            "Stopped early — outputs reflect partial training (epoch {} of {} requested)",
            trained.loss_trace.len(),
            args.epochs
        );
    } else {
        info!(
            "Done — outputs at {}.{{feature_embedding,feature_bias,gamma,log_likelihood}}.parquet",
            args.out
        );
    }
    Ok(())
}

////////////////////////////////////////////////////////////////////////
// Edge-list discovery: union of canonical node names across the file //
////////////////////////////////////////////////////////////////////////

fn discover_features_across_files(
    paths: &[Box<str>],
    name_kind: &FeatureNameKind,
) -> anyhow::Result<Vec<Box<str>>> {
    let mut set: FxHashSet<Box<str>> = FxHashSet::default();
    for path in paths {
        let delim = detect_delimiter(path);
        let read_out = read_lines_of_words_delim(path, delim, -1)?;
        for line in &read_out.lines {
            if line.len() < 2 {
                continue;
            }
            if line[0].starts_with('#') {
                continue;
            }
            set.insert(name_kind.canonicalize(&line[0]));
            set.insert(name_kind.canonicalize(&line[1]));
        }
    }
    let mut names: Vec<Box<str>> = set.into_iter().collect();
    names.sort();
    anyhow::ensure!(
        !names.is_empty(),
        "fne: 0 feature names across {} edge file(s)",
        paths.len()
    );
    Ok(names)
}

/// Load each path through `FeaturePairGraph::from_edge_list` against the
/// shared canonical node set, then merge edges into a single graph.
/// Per-file dedup happens inside `from_edge_list`; cross-file dedup is a
/// HashSet pass here so the merged graph carries each undirected pair
/// exactly once regardless of how many files mentioned it.
fn load_merged_graph(
    paths: &[Box<str>],
    feature_names: Vec<Box<str>>,
    allow_prefix: bool,
    delim: Option<char>,
) -> anyhow::Result<FeaturePairGraph> {
    let n_features = feature_names.len();
    let mut all_edges: FxHashSet<(usize, usize)> = FxHashSet::default();
    let mut per_file_counts: Vec<usize> = Vec::with_capacity(paths.len());
    for path in paths {
        let g = FeaturePairGraph::from_edge_list(path, feature_names.clone(), allow_prefix, delim)?;
        per_file_counts.push(g.feature_edges.len());
        for e in g.feature_edges {
            all_edges.insert(e);
        }
    }
    let mut feature_edges: Vec<(usize, usize)> = all_edges.into_iter().collect();
    feature_edges.sort_unstable();
    if paths.len() > 1 {
        info!(
            "fne: merged edge set has {} unique pairs (per-file: {:?})",
            feature_edges.len(),
            per_file_counts
        );
    }
    Ok(FeaturePairGraph {
        feature_names,
        n_features,
        feature_edges,
    })
}

////////////////////////////////////////////////////////////////////////
// Training                                                            //
////////////////////////////////////////////////////////////////////////

struct TrainConfig<'a> {
    embedding_dim: usize,
    epochs: usize,
    batches_per_epoch: usize,
    batch_size: usize,
    num_negatives: usize,
    learning_rate: f64,
    weight_decay: f64,
    neg_alpha: f32,
    seed: u64,
    dev: &'a Device,
    /// First Ctrl-C sets this; the loop checks at minibatch + epoch
    /// boundaries and finalizes outputs from the current parameter
    /// state. Caller installs the SIGINT handler before training.
    stop: Arc<AtomicBool>,
}

fn train_fne(graph: &FeaturePairGraph, config: &TrainConfig<'_>) -> anyhow::Result<TrainedFne> {
    let n = graph.num_features();
    let h = config.embedding_dim;
    let edges: &[(usize, usize)] = &graph.feature_edges;
    anyhow::ensure!(!edges.is_empty(), "fne: graph has zero edges");

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, config.dev);
    let e_feat = vb.get_with_hints(
        (n, h),
        "e_feat",
        candle_nn::Init::Randn {
            mean: 0.0,
            stdev: 0.1,
        },
    )?;
    let gamma = vb.get_with_hints((h,), "gamma", candle_nn::Init::Const(1.0))?;
    let b_feat = vb.get_with_hints((n,), "b_feat", candle_nn::Init::Const(0.0))?;

    let degrees = graph.feature_degrees();
    let neg_weights: Vec<f32> = degrees
        .iter()
        .map(|&d| ((d as f32).max(1.0)).powf(config.neg_alpha))
        .collect();
    let neg_picker =
        WeightedIndex::new(&neg_weights).map_err(|e| anyhow::anyhow!("neg weights: {e}"))?;

    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: config.learning_rate,
            weight_decay: config.weight_decay,
            ..Default::default()
        },
    )?;

    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut loss_trace = Vec::with_capacity(config.epochs);
    let n_edges = edges.len();

    info!(
        "fne train: N={n}, H={h}, edges={n_edges}, batch={}×{} per epoch, K={} negs, lr={}, α={}",
        config.batches_per_epoch,
        config.batch_size,
        config.num_negatives,
        config.learning_rate,
        config.neg_alpha
    );

    'epochs: for epoch in 0..config.epochs {
        let mut epoch_loss = 0f32;
        let mut n_steps = 0usize;
        for _ in 0..config.batches_per_epoch {
            if config.stop.load(Ordering::Relaxed) {
                break;
            }
            let (i_idx, j_pos, j_neg_flat) = sample_batch(
                edges,
                n_edges,
                config.batch_size,
                config.num_negatives,
                &neg_picker,
                &mut rng,
            );
            let loss = step(
                &e_feat,
                &gamma,
                &b_feat,
                &i_idx,
                &j_pos,
                &j_neg_flat,
                config.num_negatives,
                config.dev,
            )?;
            opt.backward_step(&loss)?;
            epoch_loss += loss.to_scalar::<f32>()?;
            n_steps += 1;
        }
        let avg = epoch_loss / n_steps.max(1) as f32;
        loss_trace.push(avg);
        if epoch == 0 || (epoch + 1) % 10 == 0 || epoch + 1 == config.epochs {
            info!("epoch {}/{}: loss={:.4}", epoch + 1, config.epochs, avg);
        }
        if config.stop.load(Ordering::SeqCst) {
            info!(
                "Stopping early at epoch {}/{} — finalizing outputs",
                epoch + 1,
                config.epochs
            );
            break 'epochs;
        }
    }

    Ok(TrainedFne {
        e_feat: e_feat.detach(),
        b_feat: b_feat.detach(),
        gamma: gamma.detach(),
        feature_names: graph.feature_names.clone(),
        loss_trace,
    })
}

/// Draw `B` uniform positive edges + `B*K` degree^α negatives. Returns
/// `(i_idx, j_pos, j_neg_flat)` as `Vec<u32>` ready to lift to tensors.
/// Self-loops (`k == i`) are resampled; true-positive collisions are
/// ignored (negligible at α=0.75 on sparse graphs).
fn sample_batch(
    edges: &[(usize, usize)],
    n_edges: usize,
    batch_size: usize,
    num_negatives: usize,
    neg_picker: &WeightedIndex<f32>,
    rng: &mut StdRng,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let b = batch_size;
    let k = num_negatives;
    let mut i_idx = Vec::with_capacity(b);
    let mut j_pos = Vec::with_capacity(b);
    let mut j_neg = Vec::with_capacity(b * k);
    for _ in 0..b {
        let e = rng.random_range(0..n_edges);
        let (u, v) = edges[e];
        // Random direction so the loss sees both (u, v) and (v, u);
        // since the score is symmetric under γ-diagonal scoring, this
        // is purely an i-side bias spreader for b_feat.
        let (i, j) = if rng.random::<bool>() { (u, v) } else { (v, u) };
        i_idx.push(i as u32);
        j_pos.push(j as u32);
        for _ in 0..k {
            loop {
                let kn = neg_picker.sample(rng);
                if kn != i {
                    j_neg.push(kn as u32);
                    break;
                }
            }
        }
    }
    (i_idx, j_pos, j_neg)
}

#[allow(clippy::too_many_arguments)]
fn step(
    e_feat: &Tensor,
    gamma: &Tensor,
    b_feat: &Tensor,
    i_idx: &[u32],
    j_pos: &[u32],
    j_neg_flat: &[u32],
    k: usize,
    dev: &Device,
) -> anyhow::Result<Tensor> {
    let b = i_idx.len();
    let bk = j_neg_flat.len();
    let i_t = Tensor::from_slice(i_idx, b, dev)?;
    let jp_t = Tensor::from_slice(j_pos, b, dev)?;
    let jn_t = Tensor::from_slice(j_neg_flat, bk, dev)?;

    let e_i_pos = e_feat.index_select(&i_t, 0)?; // [B, H]
    let e_j_pos = e_feat.index_select(&jp_t, 0)?; // [B, H]
    let b_i_pos = b_feat.index_select(&i_t, 0)?; // [B]
    let b_j_pos = b_feat.index_select(&jp_t, 0)?; // [B]

    // For negatives, repeat each positive's i K times so per-row lookup
    // is uniform. Cheaper than 3D broadcasting and keeps the loss shape
    // [B*K] mirror-able with positives.
    let mut i_rep: Vec<u32> = Vec::with_capacity(bk);
    for &i in i_idx {
        for _ in 0..k {
            i_rep.push(i);
        }
    }
    let i_rep_t = Tensor::from_slice(&i_rep, bk, dev)?;
    let e_i_neg = e_feat.index_select(&i_rep_t, 0)?; // [B*K, H]
    let e_j_neg = e_feat.index_select(&jn_t, 0)?; // [B*K, H]
    let b_i_neg = b_feat.index_select(&i_rep_t, 0)?; // [B*K]
    let b_j_neg = b_feat.index_select(&jn_t, 0)?; // [B*K]

    let g = gamma.unsqueeze(0)?; // [1, H]

    let pos_dot = (&e_i_pos * &e_j_pos)?
        .broadcast_mul(&g)? // [B, H]
        .sum(1)?; // [B]
    let pos_score = ((&pos_dot + &b_i_pos)? + &b_j_pos)?; // [B]

    let neg_dot = (&e_i_neg * &e_j_neg)?.broadcast_mul(&g)?.sum(1)?; // [B*K]
    let neg_score = ((&neg_dot + &b_i_neg)? + &b_j_neg)?; // [B*K]

    let pos_term = log_sigmoid(&pos_score)?; // [B]
    let neg_logit = neg_score.neg()?;
    let neg_term = log_sigmoid(&neg_logit)?; // [B*K]
    let neg_term_per_pos = neg_term.reshape((b, k))?.sum(1)?; // [B]
    let per_pos = (&pos_term + &neg_term_per_pos)?; // [B]
    let loss = per_pos.mean(0)?.neg()?;
    Ok(loss)
}

/// Numerically stable `log σ(x) = x - softplus(x) = -softplus(-x)`.
/// Local copy so this file stays self-contained.
fn log_sigmoid(x: &Tensor) -> candle_core::Result<Tensor> {
    let zero = x.zeros_like()?;
    let m = x.minimum(&zero)?;
    let lse = ((x.neg()?.abs()?).neg()?.exp()? + 1.0)?.log()?;
    m - lse
}

////////////////////////////////////////////////////////////////////////
// Output writers                                                      //
////////////////////////////////////////////////////////////////////////

fn write_outputs(trained: &TrainedFne, out_prefix: &str) -> anyhow::Result<()> {
    let h = trained.e_feat.dim(1)?;
    let n = trained.e_feat.dim(0)?;

    // feature_embedding.parquet — [N, H] keyed by feature name.
    let e_host = nalgebra::DMatrix::<f32>::from_tensor(&trained.e_feat)?;
    let h_cols: Vec<Box<str>> = (0..h).map(|i| format!("h{i}").into_boxed_str()).collect();
    e_host.to_parquet_with_names(
        &format!("{out_prefix}.feature_embedding.parquet"),
        (Some(&trained.feature_names), Some("gene")),
        Some(&h_cols),
    )?;

    // feature_bias.parquet — [N, 1] keyed by feature name.
    let b_host_2d = nalgebra::DMatrix::<f32>::from_tensor(&trained.b_feat.unsqueeze(1)?)?;
    b_host_2d.to_parquet_with_names(
        &format!("{out_prefix}.feature_bias.parquet"),
        (Some(&trained.feature_names), Some("gene")),
        Some(&[Box::from("bias")]),
    )?;

    // gamma.parquet — [H, 1] keyed by dim name h0..h{H-1}.
    let gamma_host_2d = nalgebra::DMatrix::<f32>::from_tensor(&trained.gamma.unsqueeze(1)?)?;
    gamma_host_2d.to_parquet_with_names(
        &format!("{out_prefix}.gamma.parquet"),
        (Some(&h_cols), Some("h")),
        Some(&[Box::from("gamma")]),
    )?;

    // log_likelihood.parquet — [epochs, 1] keyed by epoch index.
    let loss_mat = nalgebra::DMatrix::<f32>::from_column_slice(
        trained.loss_trace.len(),
        1,
        &trained.loss_trace,
    );
    let epoch_names: Vec<Box<str>> = (0..trained.loss_trace.len())
        .map(|i| i.to_string().into_boxed_str())
        .collect();
    loss_mat.to_parquet_with_names(
        &format!("{out_prefix}.log_likelihood.parquet"),
        (Some(&epoch_names), Some("epoch")),
        Some(&[Box::from("loss")]),
    )?;

    info!(
        "Saved {} features × {} dims to {out_prefix}.feature_embedding.parquet",
        n, h
    );
    Ok(())
}

////////////////////////////////////////////////////////////////////////
// Tests                                                               //
////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    /// Two 4-cliques {0,1,2,3} and {4,5,6,7} plus a bridge (3, 4).
    /// After 300 epochs the model should rank within-clique pairs
    /// higher than cross-clique pairs and the bridge edge higher than
    /// generic cross-clique pairs.
    #[test]
    fn two_cliques_plus_bridge_separates_communities() -> anyhow::Result<()> {
        let n = 8;
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for i in 0..4 {
            for j in (i + 1)..4 {
                edges.push((i, j));
            }
        }
        for i in 4..8 {
            for j in (i + 1)..8 {
                edges.push((i, j));
            }
        }
        edges.push((3, 4));

        let names: Vec<Box<str>> = (0..n).map(|i| format!("g{i}").into_boxed_str()).collect();
        let graph = FeaturePairGraph {
            feature_names: names,
            n_features: n,
            feature_edges: edges,
        };

        let dev = Device::Cpu;
        let trained = train_fne(
            &graph,
            &TrainConfig {
                embedding_dim: 8,
                epochs: 300,
                batches_per_epoch: 20,
                batch_size: 32,
                num_negatives: 4,
                learning_rate: 0.05,
                weight_decay: 0.0,
                neg_alpha: 0.75,
                seed: 42,
                dev: &dev,
                stop: Arc::new(AtomicBool::new(false)),
            },
        )?;

        // Compute pairwise scores on the trained model.
        let e_host = nalgebra::DMatrix::<f32>::from_tensor(&trained.e_feat)?;
        let b_host: Vec<f32> = trained.b_feat.to_vec1()?;
        let gamma_host: Vec<f32> = trained.gamma.to_vec1()?;
        let score = |i: usize, j: usize| -> f32 {
            let mut dot = 0f32;
            for h in 0..e_host.ncols() {
                dot += gamma_host[h] * e_host[(i, h)] * e_host[(j, h)];
            }
            dot + b_host[i] + b_host[j]
        };

        // Within-clique pair score vs. a generic cross-clique pair.
        let within_a = score(0, 1);
        let within_b = score(5, 6);
        let cross = score(0, 5); // (clique-A node, clique-B node), no bridge
        let bridge = score(3, 4);

        // 1. Within-clique > cross-clique on both sides.
        assert!(
            within_a > cross,
            "within-A {within_a} should exceed cross {cross}"
        );
        assert!(
            within_b > cross,
            "within-B {within_b} should exceed cross {cross}"
        );
        // 2. Bridge edge > generic cross-clique pair.
        assert!(
            bridge > cross,
            "bridge {bridge} should exceed cross {cross}"
        );
        // 3. Sanity: loss decreased over training.
        let first = trained.loss_trace[0];
        let last = trained.loss_trace[trained.loss_trace.len() - 1];
        assert!(
            last < first,
            "loss should decrease: first={first}, last={last}"
        );
        Ok(())
    }

    #[test]
    fn log_sigmoid_matches_reference() -> anyhow::Result<()> {
        let dev = Device::Cpu;
        let xs = Tensor::from_slice(&[-3.0f32, -0.5, 0.0, 0.5, 3.0], 5, &dev)?;
        let ours: Vec<f32> = log_sigmoid(&xs)?.to_vec1()?;
        let expected: Vec<f32> = [-3.0f32, -0.5, 0.0, 0.5, 3.0]
            .iter()
            .map(|&x: &f32| -(1.0 + (-x).exp()).ln())
            .collect();
        for (a, e) in ours.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-5, "log_sigmoid: {a} vs {e}");
        }
        Ok(())
    }
}
