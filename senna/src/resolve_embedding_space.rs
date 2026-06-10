//! `senna resolve-embedding-space` (alias `rest`) — **R**esolve **E**mbedding
//! **S**pace for **T**opic-models.
//!
//! The mirror of `senna bge`'s retired `--freeze-feature-embedding` with the
//! roles flipped: instead of freezing a gene embedding and learning cells,
//! this **freezes the cell topic proportions θ** from a finished `senna topic`
//! / `masked-topic` run and **learns a shared cell+gene H-dimensional
//! embedding** against the raw counts. The point is to recast a *good* topic
//! result into a Euclidean metric space where genes, topic-archetypes, and
//! cells coexist — so `senna annotate-by-projection` (marker→type annotation by
//! projecting both into one inner-product space) can consume it, which it
//! cannot do for a raw topic run (whose β is multinomial loadings, not an
//! embedding).
//!
//! Model. Frozen constant θ ∈ ℝ^{N×K} (topic proportions). Learn α ∈ ℝ^{K×H}
//! (topic embedding), ρ ∈ ℝ^{D×H} (gene embedding), b ∈ ℝ^D (gene bias). The
//! cell embedding is derived, frozen-θ: **Z = θ·α ∈ ℝ^{N×H}**. H defaults to K
//! but is a free knob (may exceed K — that is the count-based payoff over a
//! closed-form SVD of β).
//!
//! Objective. Bipartite cell–gene NCE (same family as `bge` / `fne`,
//! partition-free):
//!
//!   score(cell c, gene g) = (θ_c·α)·ρ_g + b_g
//!   ℓ = log σ(score_pos) + Σ_neg log σ(-score_neg)
//!
//! positives are observed (cell, gene) counts (sampled ∝ count); negatives are
//! genes drawn ∝ marginal^α (node2vec convention). AdamW over {α, ρ, b}. By
//! construction ρ·Zᵀ = the θ-weighted gene affinity per cell, so a marker gene
//! sits near the cells expressing it.

use crate::embed_common::*;
use auxiliary_data::data_loading::{read_data_on_shared_rows, ReadSharedRowsArgs};
use candle_core::{Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use candle_util::loss::log_sigmoid;
use graph_embedding_util::stop::setup_stop_handler;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rustc_hash::FxHashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Args, Debug)]
pub struct RestArgs {
    #[arg(
        short = 'f',
        long = "from",
        required = true,
        help = "Topic-family run manifest ({prefix}.senna.json) to freeze θ from",
        long_help = "Path to the `senna.json` manifest of a finished topic-family run \
                     (`senna topic` / `masked-topic` / `joint-topic`). Two things are read \
                     from it:\n  \
                     - outputs.latent — the frozen θ (stored as log θ, exp'd internally); \
                     cells are matched to the counts by barcode.\n  \
                     - data.input / data.batch — the count files ρ trains against (override \
                     with --data-files / --batch-files).\n\
                     Embedding-family runs (bge/fne) are rejected: their latent is already an \
                     H-space embedding, not topic proportions."
    )]
    from: Box<str>,

    #[arg(
        short = 'o',
        long = "out",
        help = "Output prefix (default: --from with .senna.json stripped)",
        long_help = "Output prefix for every artifact. Defaults to `--from` with a trailing \
                     `.senna.json` (or `.json`) removed. Writes:\n  \
                     {out}.feature_embedding.parquet  ρ      gene × H\n  \
                     {out}.cell_embedding.parquet     Z=θ·α  cell × H\n  \
                     {out}.latent.parquet             Z      cell × H (the latent annotate-by-projection reads)\n  \
                     {out}.topic_embedding.parquet    α      topic × H\n  \
                     {out}.feature_bias.parquet       b      gene × 1\n  \
                     {out}.log_likelihood.parquet     per-epoch NCE loss\n  \
                     {out}.senna.json                 run manifest (kind=resolve-embedding-space)"
    )]
    out: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 0,
        help = "Embedding dimension H (0 = K, the topic count)",
        long_help = "Dimensionality H of the shared cell+gene space. 0 (default) uses K, the \
                     number of topics in θ. Because training is against the counts (not a \
                     closed-form SVD of β), H may be set LARGER than K — the extra dimensions \
                     let ρ capture per-gene structure beyond the K topic axes. H < K \
                     compresses; at H = K the geometry is close to a metric recast of the \
                     topic dictionary."
    )]
    embedding_dim: usize,

    #[arg(
        short = 'i',
        long,
        default_value_t = 200,
        help = "Training epochs",
        long_help = "Number of epochs, each running --batches-per-epoch minibatches. Ctrl-C \
                     stops early and finalizes outputs from the current parameters."
    )]
    epochs: usize,

    #[arg(
        long,
        default_value_t = 100,
        help = "Minibatches per epoch",
        long_help = "Minibatches drawn per epoch. Each is --batch-size positive (cell,gene) \
                     edges, so one epoch visits ≈ batches_per_epoch × batch_size edges sampled \
                     with replacement ∝ count."
    )]
    batches_per_epoch: usize,

    #[arg(
        long,
        default_value_t = 1024,
        help = "Positive (cell,gene) edges per minibatch",
        long_help = "Positive edges per minibatch. Each is an observed (cell, gene) count, \
                     sampled ∝ count; the cell contributes Z_c = θ_c·α and the gene its ρ_g. \
                     Larger batches give smoother gradients at higher per-step cost."
    )]
    batch_size: usize,

    #[arg(
        long,
        default_value_t = 5,
        help = "Negative genes per positive",
        long_help = "Negatives drawn per positive edge. For each positive (cell, gene), this \
                     many genes are sampled ∝ marginal^(--neg-alpha) and pushed down for that \
                     cell. More negatives sharpen the embedding at higher per-step cost."
    )]
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
        help = "AdamW weight decay (0 = off)",
        long_help = "Decoupled AdamW weight decay applied uniformly to α, ρ, and b. 0 \
                     (default) = plain Adam; mild values (1e-4..1e-2) shrink the embedding."
    )]
    weight_decay: f64,

    #[arg(
        long,
        default_value_t = 0.75,
        help = "Negative-sampling exponent α: q(g) ∝ marginal(g)^α",
        long_help = "Exponent on the per-gene count marginal in the negative-gene sampler: \
                     q(g) ∝ marginal(g)^α. The node2vec/word2vec default 0.75 down-weights \
                     ubiquitous genes relative to proportional sampling (α=1); α=0 samples \
                     genes uniformly."
    )]
    neg_alpha: f32,

    #[arg(
        long,
        value_delimiter = ',',
        help = "Override count files (default: inherit from --from)",
        long_help = "Comma-separated count matrices (zarr/h5) to train ρ against. When \
                     omitted, the files in the --from manifest's data.input are used (resolved \
                     against the manifest directory). Override when those paths have moved or \
                     you want a different cell set; cells are still matched to θ by barcode, \
                     and any count cell absent from θ is dropped."
    )]
    data_files: Option<Vec<Box<str>>>,

    #[arg(
        short = 'b',
        long,
        value_delimiter = ',',
        help = "Override batch files (default: inherit from --from)",
        long_help = "Per-cell batch label files, one per count file; inherited from the \
                     --from manifest when omitted. Batch labels drive no correction here (θ \
                     already encodes the batch-aware topic structure) — they are carried \
                     through for manifest provenance only."
    )]
    batch_files: Option<Vec<Box<str>>>,

    #[arg(
        long,
        help = "Cells per block for the streaming edge pass (default 1024)",
        long_help = "Block size for the single streaming pass that extracts positive edges \
                     from the counts: cells are read (read_columns_csc) in chunks of this \
                     many, so larger blocks raise peak RAM per read. With --preload-data the \
                     whole matrix is already resident, so this only bounds the per-block CSC."
    )]
    block_size: Option<usize>,

    #[arg(
        long,
        default_value_t = false,
        help = "Preload all sparse columns into memory first",
        long_help = "Load every backend column into memory before the edge-extraction pass. \
                     Strongly recommended on rotational/slow disks, where the streaming reads \
                     are otherwise I/O-bound."
    )]
    preload_data: bool,

    #[arg(long, default_value_t = ComputeDevice::Cpu, value_enum, help = "Compute device")]
    device: ComputeDevice,

    #[arg(long, default_value_t = 0, help = "Device ordinal (for cuda/metal)")]
    device_no: usize,

    #[arg(long, default_value_t = 1, help = "RNG seed")]
    seed: u64,
}

/// Pre-built, file-free training inputs so the unit test can construct them
/// directly (mirrors how `train_fne` takes a `FeaturePairGraph`).
struct RestTrainInputs {
    /// θ aligned to the kept count cells: [N, K] (row-stochastic, frozen).
    theta_aligned: Mat,
    /// Positive-edge gene ids, [E].
    edge_gene: Vec<u32>,
    /// Positive-edge cell ids (row in `theta_aligned`), [E].
    edge_cell: Vec<u32>,
    /// Positive-edge weights (counts), [E].
    edge_w: Vec<f32>,
    /// Per-gene marginal Σ count, [D] — drives the marginal^α negatives.
    gene_marginal: Vec<f64>,
    n_genes: usize,
}

struct RestConfig<'a> {
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
    /// First Ctrl-C sets this; the loop finalizes outputs from the current
    /// parameter state. Caller installs the SIGINT handler before training.
    stop: Arc<AtomicBool>,
}

/// Trained host-side bundle, shared by the public path and the unit test.
struct TrainedRest {
    rho: Mat,         // [D, H] gene embedding
    alpha: Mat,       // [K, H] topic embedding
    z: Mat,           // [N, H] cell embedding = θ·α
    b_gene: Vec<f32>, // [D]
    loss_trace: Vec<f32>,
}

/// Resolve a file-list axis (counts or batches): a CLI override wins as-is;
/// otherwise the manifest's recorded paths are resolved against the manifest
/// directory. Shared by the count and batch axes.
fn inherit_paths(cli: Option<&[Box<str>]>, manifest_rel: &[String], dir: &Path) -> Vec<Box<str>> {
    match cli {
        Some(paths) => paths.to_vec(),
        None => manifest_rel
            .iter()
            .map(|p| {
                crate::run_manifest::resolve(dir, p)
                    .to_string_lossy()
                    .into_owned()
                    .into_boxed_str()
            })
            .collect(),
    }
}

pub fn resolve_embedding_space(args: &RestArgs) -> anyhow::Result<()> {
    let out = match args.out.as_deref() {
        Some(o) => o.to_string(),
        None => crate::run_manifest::derive_out_prefix(&args.from),
    };
    mkdir_parent(&out)?;

    //////////////////////////////////////////////////
    // Frozen topic side: θ + the source count files //
    //////////////////////////////////////////////////
    let (manifest, dir) = crate::run_manifest::RunManifest::load(Path::new(args.from.as_ref()))?;
    anyhow::ensure!(
        manifest.kind.is_topic_family(),
        "resolve-embedding-space needs a topic-family --from run (topic / masked-topic / \
         joint-topic) whose latent is log θ; got kind={}.",
        manifest.kind
    );
    let theta_rel = manifest
        .outputs
        .latent
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--from manifest has no outputs.latent (θ)"))?;
    let theta_path = crate::run_manifest::resolve(&dir, theta_rel)
        .to_string_lossy()
        .into_owned();
    let theta_mat = Mat::from_parquet(&theta_path)?;
    let theta_cell_names = theta_mat.rows;
    let k = theta_mat.mat.ncols();
    anyhow::ensure!(k > 0, "θ ({theta_path}) has zero columns");
    let theta_full = theta_mat.mat.map(|x| x.exp()); // log θ → θ
    let h = if args.embedding_dim == 0 {
        k
    } else {
        args.embedding_dim
    };

    let data_files = inherit_paths(args.data_files.as_deref(), &manifest.data.input, &dir);
    anyhow::ensure!(
        !data_files.is_empty(),
        "no count files: the --from manifest has an empty data.input and --data-files was not given"
    );
    let batch_resolved = inherit_paths(args.batch_files.as_deref(), &manifest.data.batch, &dir);
    let batch_files = (!batch_resolved.is_empty()).then_some(batch_resolved);
    let input_for_manifest: Vec<String> = data_files.iter().map(|s| s.to_string()).collect();
    let batch_for_manifest: Vec<String> = batch_files
        .as_ref()
        .map(|v| v.iter().map(|s| s.to_string()).collect())
        .unwrap_or_default();

    /////////////////////////////
    // Counts + axis alignment //
    /////////////////////////////
    let loaded = read_data_on_shared_rows(ReadSharedRowsArgs {
        data_files,
        batch_files,
        preload: args.preload_data,
        ..Default::default()
    })?;
    let data_vec = loaded.data;
    let gene_names = data_vec.row_names()?;
    let count_cell_names = data_vec.column_names()?;
    let d = data_vec.num_rows();
    info!(
        "counts: {} genes × {} cells; θ: {} cells × {} topics; H={}",
        d,
        data_vec.num_columns(),
        theta_cell_names.len(),
        k,
        h
    );

    // Keep count cells present in θ, IN COUNT-CELL ORDER. `theta_aligned`,
    // `kept_cols`, and `kept_names` are all in that one shared order.
    let theta_index: FxHashMap<&str, usize> = theta_cell_names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_ref(), i))
        .collect();
    let mut kept_cols: Vec<usize> = Vec::new();
    let mut theta_rows: Vec<usize> = Vec::new();
    let mut kept_names: Vec<Box<str>> = Vec::new();
    for (col, bc) in count_cell_names.iter().enumerate() {
        if let Some(&trow) = theta_index.get(bc.as_ref()) {
            kept_cols.push(col);
            theta_rows.push(trow);
            kept_names.push(bc.clone());
        }
    }
    let n = kept_cols.len();
    anyhow::ensure!(
        n > 0,
        "0 cells shared between θ ({} cells) and counts ({} cells) — barcode mismatch?",
        theta_cell_names.len(),
        count_cell_names.len()
    );
    if n < count_cell_names.len() {
        info!("aligned {n}/{} count cells to θ", count_cell_names.len());
    }
    let theta_aligned = Mat::from_fn(n, k, |i, j| theta_full[(theta_rows[i], j)]);

    // Positive edges + gene marginals, streamed in blocks of cells.
    let block = args.block_size.unwrap_or(1024).max(1);
    let mut edge_gene: Vec<u32> = Vec::new();
    let mut edge_cell: Vec<u32> = Vec::new();
    let mut edge_w: Vec<f32> = Vec::new();
    let mut gene_marginal = vec![0f64; d];
    for (chunk_idx, chunk) in kept_cols.chunks(block).enumerate() {
        let base = chunk_idx * block;
        let csc = data_vec.read_columns_csc(chunk.iter().copied())?;
        for (gene, loc, &val) in csc.triplet_iter() {
            if val <= 0.0 {
                continue;
            }
            edge_gene.push(gene as u32);
            edge_cell.push((base + loc) as u32);
            edge_w.push(val);
            gene_marginal[gene] += val as f64;
        }
    }
    anyhow::ensure!(
        !edge_gene.is_empty(),
        "no nonzero counts among the {n} aligned cells"
    );
    info!(
        "{} positive (cell,gene) edges over {} genes",
        edge_gene.len(),
        d
    );

    //////////////
    // Training //
    //////////////
    let stop = setup_stop_handler();
    let dev = args.device.to_device(args.device_no)?;
    let trained = train_rest(
        &RestTrainInputs {
            theta_aligned,
            edge_gene,
            edge_cell,
            edge_w,
            gene_marginal,
            n_genes: d,
        },
        &RestConfig {
            embedding_dim: h,
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

    write_outputs(&trained, &gene_names, &kept_names, &out)?;

    crate::run_manifest::write_run_manifest(&crate::run_manifest::RunDescription {
        kind: crate::run_manifest::RunKind::ResolveEmbeddingSpace,
        prefix: &out,
        data_input: &input_for_manifest,
        data_batch: &batch_for_manifest,
        data_input_null: &[],
        dictionary_suffix: None,
        has_model: false,
        has_cell_proj: false,
        pb_gene_suffix: None,
        pb_latent_suffix: None,
        dictionary_empirical_suffix: None,
        feature_embedding_suffix: Some("feature_embedding.parquet"),
        default_colour_by: "cluster",
        has_latent: true,
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
            "Done — {out}.{{feature_embedding,cell_embedding,latent,topic_embedding}}.parquet. \
             Next: `senna annotate-by-projection --from {out}.senna.json --markers <markers.tsv>`."
        );
    }
    Ok(())
}

fn train_rest(inputs: &RestTrainInputs, config: &RestConfig<'_>) -> anyhow::Result<TrainedRest> {
    let RestTrainInputs {
        theta_aligned,
        edge_gene,
        edge_cell,
        edge_w,
        gene_marginal,
        n_genes,
    } = inputs;
    let d = *n_genes;
    let k = theta_aligned.ncols();
    let n = theta_aligned.nrows();
    let h = config.embedding_dim;
    anyhow::ensure!(!edge_gene.is_empty(), "rest: zero positive edges");
    let n_edges = edge_gene.len();

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, config.dev);
    let alpha = vb.get_with_hints(
        (k, h),
        "alpha",
        candle_nn::Init::Randn {
            mean: 0.0,
            stdev: 0.1,
        },
    )?;
    let rho = vb.get_with_hints(
        (d, h),
        "rho",
        candle_nn::Init::Randn {
            mean: 0.0,
            stdev: 0.1,
        },
    )?;
    let b_gene = vb.get_with_hints((d,), "b_gene", candle_nn::Init::Const(0.0))?;

    // Frozen θ as a constant device tensor [N, K]. `.contiguous()` because
    // `to_tensor` of a column-major nalgebra matrix is non-contiguous and
    // `index_select` (the per-batch cell gather) requires contiguous sources.
    let theta_t = theta_aligned.to_tensor(config.dev)?.contiguous()?;

    let pos_picker =
        WeightedIndex::new(edge_w).map_err(|e| anyhow::anyhow!("positive weights: {e}"))?;
    let neg_weights: Vec<f32> = gene_marginal
        .iter()
        .map(|&m| (m as f32).max(1.0).powf(config.neg_alpha))
        .collect();
    let neg_picker =
        WeightedIndex::new(&neg_weights).map_err(|e| anyhow::anyhow!("negative weights: {e}"))?;

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

    info!(
        "rest train: N={n}, D={d}, K={k}, H={h}, edges={n_edges}, {}×{} per epoch, {} negs, lr={}, α={}",
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
            let (ci, gp, gn) = sample_batch(
                edge_cell,
                edge_gene,
                config.batch_size,
                config.num_negatives,
                &pos_picker,
                &neg_picker,
                &mut rng,
            );
            let loss = step(
                &theta_t,
                &alpha,
                &rho,
                &b_gene,
                &ci,
                &gp,
                &gn,
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

    let rho_host = Mat::from_tensor(&rho)?;
    let alpha_host = Mat::from_tensor(&alpha)?;
    let z = theta_aligned * &alpha_host;
    let b_host: Vec<f32> = b_gene.to_vec1()?;

    Ok(TrainedRest {
        rho: rho_host,
        alpha: alpha_host,
        z,
        b_gene: b_host,
        loss_trace,
    })
}

/// Draw `B` positives (∝ count) + `B*K` negatives (∝ marginal^α). Returns
/// `(cell_idx, gene_pos, gene_neg_flat)` as `Vec<u32>`. Negative collisions
/// with the positive gene are rare and harmless (ignored, like fne).
fn sample_batch(
    edge_cell: &[u32],
    edge_gene: &[u32],
    batch_size: usize,
    num_negatives: usize,
    pos_picker: &WeightedIndex<f32>,
    neg_picker: &WeightedIndex<f32>,
    rng: &mut StdRng,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let b = batch_size;
    let k = num_negatives;
    let mut ci = Vec::with_capacity(b);
    let mut gp = Vec::with_capacity(b);
    let mut gn = Vec::with_capacity(b * k);
    for _ in 0..b {
        let e = pos_picker.sample(rng);
        ci.push(edge_cell[e]);
        gp.push(edge_gene[e]);
        for _ in 0..k {
            gn.push(neg_picker.sample(rng) as u32);
        }
    }
    (ci, gp, gn)
}

#[allow(clippy::too_many_arguments)]
fn step(
    theta_t: &Tensor, // [N, K]
    alpha: &Tensor,   // [K, H]
    rho: &Tensor,     // [D, H]
    b_gene: &Tensor,  // [D]
    ci: &[u32],       // [B] cell ids
    gp: &[u32],       // [B] positive gene ids
    gn: &[u32],       // [B*K] negative gene ids
    k: usize,
    dev: &Device,
) -> anyhow::Result<Tensor> {
    let b = ci.len();
    let bk = gn.len();
    let ci_t = Tensor::from_slice(ci, b, dev)?;
    let gp_t = Tensor::from_slice(gp, b, dev)?;
    let gn_t = Tensor::from_slice(gn, bk, dev)?;

    // Z_b = θ_b · α  [B, H] — computed once and reused for the negatives
    // (which share the same B cells), so the negative side is a broadcast
    // dot, not a second θ-gather + matmul.
    let z_b = theta_t.index_select(&ci_t, 0)?.matmul(alpha)?;
    let h = z_b.dim(1)?;
    let rho_pos = rho.index_select(&gp_t, 0)?; // [B, H]
    let b_pos = b_gene.index_select(&gp_t, 0)?; // [B]
    let pos_score = ((&z_b * &rho_pos)?.sum(1)? + &b_pos)?; // [B]

    // Negatives reuse z_b: score_neg[i,j] = z_b[i] · ρ_{neg[i,j]} + b_{neg[i,j]}.
    let rho_neg = rho.index_select(&gn_t, 0)?.reshape((b, k, h))?; // [B, K, H]
    let b_neg = b_gene.index_select(&gn_t, 0)?.reshape((b, k))?; // [B, K]
    let neg_dot = z_b.unsqueeze(1)?.broadcast_mul(&rho_neg)?.sum(2)?; // [B, K]
    let neg_score = (neg_dot + b_neg)?; // [B, K]

    let pos_term = log_sigmoid(&pos_score)?; // [B]
    let neg_per_pos = log_sigmoid(&neg_score.neg()?)?.sum(1)?; // [B]
    let per_pos = (&pos_term + &neg_per_pos)?; // [B]
    let loss = per_pos.mean(0)?.neg()?;
    Ok(loss)
}

////////////////////////////////////////////////////////////////////////
// Output writers                                                      //
////////////////////////////////////////////////////////////////////////

fn write_outputs(
    trained: &TrainedRest,
    gene_names: &[Box<str>],
    cell_names: &[Box<str>],
    out: &str,
) -> anyhow::Result<()> {
    let h = trained.rho.ncols();
    let h_cols = axis_id_names("h", h);

    // ρ — gene × H (the per-gene embedding annotate-by-projection projects markers into).
    trained.rho.to_parquet_with_names(
        &format!("{out}.feature_embedding.parquet"),
        (Some(gene_names), Some("gene")),
        Some(&h_cols),
    )?;

    // Z — cell × H, written as BOTH cell_embedding and latent (annotate-by-projection
    // reads outputs.latent as the cell embedding).
    trained.z.to_parquet_with_names(
        &format!("{out}.cell_embedding.parquet"),
        (Some(cell_names), Some("cell")),
        Some(&h_cols),
    )?;
    trained.z.to_parquet_with_names(
        &format!("{out}.latent.parquet"),
        (Some(cell_names), Some("cell")),
        Some(&h_cols),
    )?;

    // α — topic × H (the K topic-archetype positions in the shared space).
    let topic_names = axis_id_names("T", trained.alpha.nrows());
    trained.alpha.to_parquet_with_names(
        &format!("{out}.topic_embedding.parquet"),
        (Some(&topic_names), Some("topic")),
        Some(&h_cols),
    )?;

    // Diagnostics: per-gene bias + per-epoch loss trace.
    let b_2d = Mat::from_column_slice(trained.b_gene.len(), 1, &trained.b_gene);
    b_2d.to_parquet_with_names(
        &format!("{out}.feature_bias.parquet"),
        (Some(gene_names), Some("gene")),
        Some(&[Box::from("bias")]),
    )?;
    let loss_mat = Mat::from_column_slice(trained.loss_trace.len(), 1, &trained.loss_trace);
    let epoch_names: Vec<Box<str>> = (0..trained.loss_trace.len())
        .map(|i| i.to_string().into_boxed_str())
        .collect();
    loss_mat.to_parquet_with_names(
        &format!("{out}.log_likelihood.parquet"),
        (Some(&epoch_names), Some("epoch")),
        Some(&[Box::from("loss")]),
    )?;

    info!(
        "Saved ρ [{}×{h}] and Z [{}×{h}] to {out}.{{feature,cell}}_embedding.parquet",
        trained.rho.nrows(),
        trained.z.nrows()
    );
    Ok(())
}

////////////////////////////////////////////////////////////////////////
// Tests                                                               //
////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic block recovery: K=2 topics, D=6 genes (0,1,2 mark topic 0;
    /// 3,4,5 mark topic 1), N=40 cells (first 20 in topic 0). θ ≈ one-hot;
    /// each cell emits counts only on its topic's marker genes. After
    /// training, each topic's cells must score their OWN markers higher than
    /// the other topic's markers in the shared space (Z_c · ρ_g).
    #[test]
    fn topic_blocks_recover_marker_cell_affinity() -> anyhow::Result<()> {
        let (k, d, n) = (2usize, 6usize, 40usize);
        let mut theta = Mat::zeros(n, k);
        for i in 0..n {
            let t = usize::from(i >= n / 2);
            theta[(i, t)] = 0.9;
            theta[(i, 1 - t)] = 0.1;
        }
        let mut edge_cell = Vec::new();
        let mut edge_gene = Vec::new();
        let mut edge_w = Vec::new();
        let mut gene_marginal = vec![0f64; d];
        for i in 0..n {
            let markers: [u32; 3] = if i < n / 2 { [0, 1, 2] } else { [3, 4, 5] };
            for &g in &markers {
                edge_cell.push(i as u32);
                edge_gene.push(g);
                edge_w.push(5.0);
                gene_marginal[g as usize] += 5.0;
            }
        }

        let inputs = RestTrainInputs {
            theta_aligned: theta,
            edge_gene,
            edge_cell,
            edge_w,
            gene_marginal,
            n_genes: d,
        };
        let dev = Device::Cpu;
        let trained = train_rest(
            &inputs,
            &RestConfig {
                embedding_dim: 4,
                epochs: 200,
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

        let score = |c: usize, g: usize| -> f32 { trained.z.row(c).dot(&trained.rho.row(g)) };
        let mean_score = |cells: std::ops::Range<usize>, genes: [usize; 3]| -> f32 {
            let mut s = 0f32;
            let mut cnt = 0usize;
            for c in cells {
                for &g in &genes {
                    s += score(c, g);
                    cnt += 1;
                }
            }
            s / cnt as f32
        };

        let t0_own = mean_score(0..20, [0, 1, 2]);
        let t0_other = mean_score(0..20, [3, 4, 5]);
        let t1_own = mean_score(20..40, [3, 4, 5]);
        let t1_other = mean_score(20..40, [0, 1, 2]);

        assert!(
            t0_own > t0_other,
            "topic-0 cells should score own markers higher: {t0_own} vs {t0_other}"
        );
        assert!(
            t1_own > t1_other,
            "topic-1 cells should score own markers higher: {t1_own} vs {t1_other}"
        );
        assert!(
            trained.loss_trace.last().unwrap() < trained.loss_trace.first().unwrap(),
            "loss should decrease: {:?} → {:?}",
            trained.loss_trace.first(),
            trained.loss_trace.last()
        );
        Ok(())
    }
}
