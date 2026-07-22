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
//!   score(cell c, gene g) = (`θ_c·α)·ρ_g` + `b_g`
//!   ℓ = log `σ(score_pos)` + `Σ_neg` log σ(-score_neg)
//!
//! positives are observed (cell, gene) counts (sampled ∝ count); negatives are
//! genes drawn ∝ marginal^α (node2vec convention). `AdamW` over {α, ρ, b}. By
//! construction ρ·Zᵀ = the θ-weighted gene affinity per cell, so a marker gene
//! sits near the cells expressing it.
//!
//! Post-hoc co-embedding. Like `senna bge`, the learned ρ fans out off the
//! K-archetype cell simplex (cells live in the convex hull of α), so a joint
//! UMAP separates genes from cells. The SIMBA `si.tl.embed` transform
//! ([`graph_embedding_util::feature_coembedding`]) is applied post-hoc: each
//! gene is re-placed at the softmax-over-cells weighted average of the *cell*
//! embeddings Z, landing it on the cell manifold. This overrides the written
//! `{out}.feature_embedding.parquet` (the one `annotate-by-projection` reads);
//! the raw ρ is the disjoint off-manifold cloud and is not written. Cells are
//! the reference and are unchanged, and training is untouched.

use crate::embed_common::*;
use auxiliary_data::data_loading::{read_data_on_shared_rows, ReadSharedRowsArgs};
use candle_core::Device;
use graph_embedding_util as ge;
use graph_embedding_util::fit::resolve_embedding::{
    train_rest, RestConfig, RestTrainInputs, TrainedRest,
};
use graph_embedding_util::stop::setup_stop_handler;
use rustc_hash::FxHashMap;
use std::path::Path;
use std::sync::atomic::Ordering;

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
                     {out}.feature_embedding.parquet  co-embed  gene × H (ρ re-embedded onto the cell manifold; annotate reads this)\n  \
                     {out}.cell_embedding.parquet     Z=θ·α  cell × H (the cell side annotate-by-projection reads)\n  \
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

    ///////////////////////////////////////////////////
    // Frozen topic side: θ + the source count files //
    ///////////////////////////////////////////////////
    let (manifest, dir) = crate::run_manifest::RunManifest::load(Path::new(args.from.as_ref()))?;
    // `latent_is_log_simplex`, not `is_topic_family`: this command `exp()`s the
    // stored latent as θ just below, so it needs a log-θ latent specifically.
    // `masked-vae` is topic-family but stores a raw Gaussian z (softmax, not
    // exp), so it must be rejected here rather than silently mis-`exp`ed.
    anyhow::ensure!(
        manifest.kind.latent_is_log_simplex(),
        "resolve-embedding-space needs a --from run whose latent is log θ \
         (topic / masked-topic / joint-topic); got kind={}.",
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
    let theta_full = theta_mat.mat.map(f32::exp); // log θ → θ
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
    let input_for_manifest: Vec<String> = data_files
        .iter()
        .map(std::string::ToString::to_string)
        .collect();
    let batch_for_manifest: Vec<String> = batch_files
        .as_ref()
        .map(|v| v.iter().map(std::string::ToString::to_string).collect())
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
            gene_marginal[gene] += f64::from(val);
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

    // Post-hoc SIMBA-style co-embedding: re-place each gene ρ onto the frozen
    // cell manifold Z = θ·α (gene = softmax-over-cells weighted average of the
    // cells), mirroring `senna bge`. Without it ρ fans out off the K-archetype
    // cell simplex, so a joint UMAP separates genes from cells; the co-embed
    // lands genes on the cell manifold, which is what `annotate-by-projection`
    // (reading feature_embedding) wants. Overrides {out}.feature_embedding.parquet
    // (the raw off-manifold ρ is not written). Cells are the reference and are
    // unchanged. Post-hoc — training above is untouched. Run on CPU over the
    // finished Z/ρ (Leiden cluster + blocked softmax-matmul pass).
    let cpu = Device::Cpu;
    let z_t = trained.z.to_tensor(&cpu)?;
    let rho_t = trained.rho.to_tensor(&cpu)?;
    let (_labels, target_eff) = ge::cell_clusters(&z_t, Some(k))?;
    ge::write_feature_coembedding(&out, &z_t, &rho_t, &gene_names, target_eff)?;

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
        cell_embedding_suffix: Some("cell_embedding.parquet"),
        default_colour_by: "cluster",
        // Z is the cell table and lives in cell_embedding; this run emits no
        // log θ of its own (θ is the frozen *input*), so there is no latent.
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
            "Done — {out}.{{feature_embedding,cell_embedding,topic_embedding}}.parquet. \
             Next: `senna annotate-by-enrichment --from {out}.senna.json --markers <markers.tsv>`."
        );
    }
    Ok(())
}

////////////////////
// Output writers //
////////////////////

fn write_outputs(
    trained: &TrainedRest,
    gene_names: &[Box<str>],
    cell_names: &[Box<str>],
    out: &str,
) -> anyhow::Result<()> {
    let h = trained.rho.ncols();
    let h_cols = axis_id_names("h", h);

    // NOTE: {out}.feature_embedding.parquet (the co-embed, genes on the cell
    // manifold) is written by the SIMBA co-embedding step in
    // `resolve_embedding_space`, not here. This writer owns the cell/topic side.

    // Z — cell × H. Written ONLY as cell_embedding: `latent` is reserved for
    // log θ across every senna run, and this command's θ is the frozen input,
    // not an output. plot / layout / clustering find Z through
    // `RunOutputs::geometry_latent`, which prefers cell_embedding.
    trained.z.to_parquet_with_names(
        &format!("{out}.cell_embedding.parquet"),
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
        "Saved Z [{}×{h}] to {out}.cell_embedding.parquet and α [{}×{h}] to topic_embedding.parquet",
        trained.z.nrows(),
        trained.alpha.nrows()
    );
    Ok(())
}

