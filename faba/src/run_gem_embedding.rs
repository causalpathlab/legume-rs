//! Entry point for `faba gem` (alias `gem-embedding`).
//!
//! Genes-only joint embedding: each row `{gene}/count/{spliced|unspliced}` is an
//! independent feature sharing the cell axis, and a gene's two tracks embed
//! identically as `β_g` (β-sharing) via the per-gene factorization. Driven
//! straight through the shared `graph_embedding_util` engine — the bilinear
//! score `e_feat·e_cell + b_feat + b_cell`, phase-1 multilevel-pseudobulk
//! training + phase-2 analytical per-cell projection. Cell **identity** is
//! resolved by the SPLICED edges (mature mRNA = current state) and written **raw**
//! (`{out}.latent.parquet`, magnitude kept); the same phase-2 pass fits an analytic
//! velocity increment `δ` to the unspliced edges (identity held fixed) and writes
//! it **raw** (`{out}.velocity.parquet`, ‖δ‖ = speed). Everything is the model's
//! actual MAP estimate — no post-hoc unit-norm, no aggregation. The nascent state
//! is just `θ + δ` = latent + velocity (derivable). Per-gene velocity, if wanted, is
//! the in-model `δ_g` (`--delta-l2`). No softmax co-embedding is written (see the
//! NOTE in `run_gem_genes_bge`: not every gene can be co-embedded).
//!
//! NOTE — `latent` is **raw** (its norm carries library size), so cluster / UMAP it
//! with **cosine** distance, or L2-normalize the rows first; plain Euclidean would be
//! dominated by the depth axis. (Only the gem/splice path stores raw; `senna bge`
//! still writes the L2 direction.)

use anyhow::Context;
use data_beans::sparse_io_vector::ColumnAlignment;
use graph_embedding_util::data::UnifiedData;
use graph_embedding_util::stop::setup_stop_handler;
use graph_embedding_util::{load_unified_data, FeatureNameKind, LoadUnifiedArgs};
use log::info;
use matrix_util::common_io::{basename, mkdir_parent};
use rayon::ThreadPoolBuilder;
use rustc_hash::FxHashMap;

use faba::gem::args::GemArgs;
use faba::gem::sample_id::{file_sample_id, longest_common_underscore_suffix};

pub fn run_gem_embedding(args: &GemArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;
    validate_args(args)?;

    let n_threads = if args.runtime.threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        args.runtime.threads
    };
    ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()
        .ok(); // ignore error if pool already initialised
    info!(
        "rayon thread pool: {} threads",
        rayon::current_num_threads()
    );

    let feature_kind = if args.collapse.feature_name_exact {
        FeatureNameKind::Exact
    } else {
        FeatureNameKind::Gene {
            delim: args.collapse.feature_name_delim,
        }
    };

    let batch_files: Option<&[Box<str>]> = if args.collapse.ignore_batch {
        if args.batch_files.is_some() {
            info!("--ignore-batch: dropping batch labels; treating all cells as one batch");
        }
        None
    } else {
        args.batch_files.as_deref()
    };

    run_gem_genes_bge(args, feature_kind, batch_files)
}

/// Load the `--genes` files into one `UnifiedData`, tagging each file's
/// barcodes with its `@sample` id when there is >1 file and no explicit
/// `--batch-files` (so batch identity survives the Union merge).
fn load_modality(
    files: &[Box<str>],
    strip: &str,
    do_tag: bool,
    batch_files: Option<&[Box<str>]>,
    feature_kind: FeatureNameKind,
    preload: bool,
) -> anyhow::Result<UnifiedData> {
    let mut data_files: Vec<Box<str>> = Vec::with_capacity(files.len());
    let mut sample_ids: Vec<Box<str>> = Vec::with_capacity(files.len());
    for f in files {
        sample_ids.push(file_sample_id(f, strip)?);
        data_files.push(f.clone());
    }
    let per_file_barcode_suffix: Option<Vec<Option<Box<str>>>> = if do_tag {
        Some(sample_ids.into_iter().map(Some).collect())
    } else {
        None
    };
    load_unified_data(LoadUnifiedArgs {
        data_files,
        batch_files: batch_files.map(<[Box<str>]>::to_vec),
        feature_kind: Some(feature_kind),
        preload,
        column_alignment: ColumnAlignment::Union,
        per_file_barcode_suffix,
        ..Default::default()
    })
}

/// Genes-only joint embedding over the shared `graph_embedding_util` engine.
/// Writes the standard geu outputs `{out}.{latent,dictionary,feature_bias,
/// cell_bias}.parquet` (latent = raw spliced θ) and `{out}.velocity.parquet` (raw
/// velocity increment δ). No softmax co-embedding; no nascent/driver post-hoc.
fn run_gem_genes_bge(
    args: &GemArgs,
    feature_kind: FeatureNameKind,
    batch_files: Option<&[Box<str>]>,
) -> anyhow::Result<()> {
    use graph_embedding_util as ge;

    // Genes load (+ per-file `@sample` tag for batch identity when >1 file and
    // no explicit --batch-files). The sample-id strip is the explicit
    // `--genes-sample-strip`, else the longest common `_`-suffix across the
    // genes basenames.
    let do_tag = batch_files.is_none() && args.genes.len() > 1;
    let genes_strip: Box<str> = if !args.collapse.genes_sample_strip.is_empty() {
        args.collapse.genes_sample_strip.clone()
    } else if do_tag {
        let genes_bn: Vec<Box<str>> = args
            .genes
            .iter()
            .map(|f| basename(f))
            .collect::<anyhow::Result<_>>()?;
        let s = longest_common_underscore_suffix(&genes_bn);
        if !s.is_empty() {
            info!("auto-strip: --genes-sample-strip = {:?}", s.as_ref());
        }
        s
    } else {
        "".into()
    };
    if do_tag {
        info!("tagging barcodes with per-file @sample id for batch identity");
    }
    let mut unified = load_modality(
        &args.genes,
        &genes_strip,
        do_tag,
        batch_files,
        feature_kind,
        args.runtime.preload_data,
    )
    .context("load genes backend")?;
    info!(
        "genes loaded: {} features × {} cells, {} batch(es)",
        unified.n_features(),
        unified.n_cells(),
        unified.n_batches()
    );

    // Optional gene-level HVG feature filter (like `senna bge`): select the top-N
    // most variable GENES and drop the rest — dropping BOTH the spliced and
    // unspliced rows of a dropped gene together so the β-sharing factorization
    // stays aligned. `subset_features` narrows the dictionary/co-embed (removing
    // the low-detection "empty" genes that pile at the co-embed centre); the
    // uniform `hvg_weights` over the survivors then restricts the pb projection /
    // membership to those genes too. `None` (n_hvg = 0) keeps every gene.
    let hvg_weights: Option<Vec<f32>> = if args.collapse.n_hvg > 0 {
        use data_beans_alg::hvg::select_hvg_by_stats;
        use data_beans_alg::sparse_streaming::streaming_sparse_running_stats;
        use matrix_util::traits::RunningStatOps;
        // Select the top-N most variable GENES (not rows): compute per-row running
        // stats, POOL a gene's spliced + unspliced tracks onto one gene entry, and
        // rank genes by NB dispersion-trend excess. `--n-hvg N` keeps exactly the N
        // most variable genes (both tracks of each together) — pooling spliced +
        // unspliced total mirrors the pipeline's CR-style gene filter. Ranking on rows
        // instead would return well under N genes (the two correlated tracks of a gene
        // both rank high and collapse to one gene on dedup).
        let (row_gene, genes) = intern_gene_keys(&unified.feature_names);
        let n_genes = genes.len();
        let stat = streaming_sparse_running_stats(unified.count_backend(), None, "HVG")
            .context("HVG streaming stats")?;
        let (means, vars) = (stat.mean(), stat.variance());
        // Pooled gene stats: mean is exact (E[s+u]=E[s]+E[u]); var sums the tracks (a
        // lower bound ignoring cross-track covariance — fine for ranking).
        let mut gmean = vec![0f32; n_genes];
        let mut gvar = vec![0f32; n_genes];
        for (r, (&m, &v)) in means.iter().zip(vars.iter()).enumerate() {
            gmean[row_gene[r] as usize] += m;
            gvar[row_gene[r] as usize] += v;
        }
        let keep_genes: rustc_hash::FxHashSet<usize> =
            select_hvg_by_stats(&gmean, &gvar, args.collapse.n_hvg)
                .into_iter()
                .collect();
        let keep_rows: Vec<usize> = (0..row_gene.len())
            .filter(|&r| keep_genes.contains(&(row_gene[r] as usize)))
            .collect();
        let before = unified.n_features();
        unified.subset_features(&keep_rows);
        info!(
            "HVG filter (--n-hvg {}): {} genes → {} kept ({} → {} feature rows)",
            args.collapse.n_hvg,
            n_genes,
            keep_genes.len(),
            before,
            unified.n_features()
        );
        Some(vec![1.0f32; unified.n_features()])
    } else {
        None
    };

    // Per-gene β-sharing factorization. Each row `{gene}/count/{spliced|unspliced}`
    // maps to its gene, so a gene's spliced and unspliced tracks embed identically
    // as `β_g`. The splice deviation is recovered as the phase-2 velocity increment
    // δ on the CELL axis (identity θ held fixed, δ fit to the unspliced edges).
    // `gene_names` is id-ordered (labels the per-gene δ_g dictionary when
    // `--delta-l2 > 0`); the factor's `row_to_gene` shares the same ids.
    let (factor, gene_names) = build_splice_factor(&unified.feature_names);
    let n_genes = gene_names.len();
    info!(
        "β-sharing factor: {} genes from {} count rows ({} unspliced rows); \
         splice δ → cell-axis velocity increment",
        n_genes,
        unified.feature_names.len(),
        factor.unspliced_rows.iter().filter(|&&b| b).count(),
    );

    let dev = args
        .runtime
        .device
        .to_device(args.runtime.device_no)
        .context("candle device init")?;
    info!("compute device = {:?}", dev);
    let stop = setup_stop_handler();

    let cfg = ge::FitConfig {
        embedding_dim: args.model.embedding_dim,
        num_levels: args.collapse.num_levels,
        sort_dim: args.collapse.sort_dim,
        knn_pb_samples: args.collapse.knn_pb,
        num_opt_iter: args.collapse.num_opt_iter,
        proj_dim: args.collapse.proj_dim,
        hvg_weights,
        // geu's multilevel collapse requires a refine spec (it surfaces the
        // per-level cell→pb maps phase-2 needs). Use geu's defaults — same as a
        // `senna bge` run without `--no-refine`.
        refine: Some(ge::RefineParams::default()),
        epochs: args.train.epochs,
        batches_per_epoch: args.train.batches_per_epoch,
        batch_size: args.train.batch_size,
        num_negatives: 4,
        learning_rate: args.train.learning_rate,
        seed: args.runtime.seed,
        device: dev,
        block_size: None,
        fisher_weights_cache: Some(format!("{}.fisher_weights.parquet", args.out).into_boxed_str()),
        feature_network: None,
        stop: Some(stop),
        feature_embedding_l2: 0.0,
        weight_decay: 0.0,
        max_grad_norm: args.train.max_grad_norm,
        cell_weight_mult: None,
        phase1_cells_per_pb: args.collapse.phase1_cells_per_pb,
        feat_factor: Some(factor),
        delta_l2: args.model.delta_l2,
    };
    let out = ge::fit(&mut unified, cfg).context("ge::fit (genes bge)")?;

    // NOTE: NO softmax co-embedding is written. (1) The gene↔cell co-embedding
    // (`{out}.feature_embedding.parquet`) is dropped: cell-type identity is carried by
    // a *few* high-contrast marker genes — a sparse, heavy-tailed distribution of gene
    // norms ‖β_g‖ — and a softmax barycenter that drops every gene onto the cell
    // manifold has to flatten that contrast, so the great majority of genes land with
    // no meaningful cell location: NOT EVERY GENE CAN BE CO-EMBEDDED. Gene↔cell
    // co-embedding and sharp cell clusters are the same degree of freedom pulling
    // opposite ways — we keep the sharp clusters. (2) No velocity "driver" co-embed
    // either: a per-gene velocity readout, if wanted, is the in-model δ_g (`--delta-l2`
    // → `{out}.delta_dictionary.parquet`), not a post-hoc average. The gene dictionary
    // (β_g) is still written by `save_outputs`.
    let cpu = candle_util::candle_core::Device::Cpu;
    ge::save_outputs(
        &out.model,
        &ge::OutputContext {
            feature_names: &unified.feature_names,
            barcodes: &unified.barcodes,
            cell_keep_idx: None,
        },
        &args.out,
    )
    .context("save outputs")?;

    // Per-gene splice offset δ_g (`--delta-l2 > 0`): the nascent loading
    // (unspliced e_f = β_g + δ_g). Read the trained `delta` Var from the varmap
    // and save it row-labeled by gene — genes with large ‖δ_g‖ carry a distinct
    // nascent/velocity program; the L2 ridge shrinks the rest toward 0.
    if args.model.delta_l2 > 0.0 {
        let vars = out.varmap.data().lock().unwrap();
        if let Some(delta) = vars.get("delta") {
            let d_t = delta.as_tensor().to_device(&cpu)?;
            // genes whose ‖δ_g‖ is above ~0 (the L2 ridge shrinks but does NOT
            // sparsify, so this count is typically most genes — it is a coverage
            // readout, not a sparsity one).
            let per_gene_max: Vec<f32> = d_t.abs()?.max(1)?.to_vec1()?;
            let nz = per_gene_max.iter().filter(|&&x| x > 1e-6).count();
            ge::save_embedding(
                &format!("{}.delta_dictionary.parquet", args.out),
                &d_t,
                &gene_names,
                "gene",
            )
            .context("save δ_g dictionary")?;
            info!(
                "wrote {}.delta_dictionary.parquet (δ_g; {}/{} genes with nonzero offset)",
                args.out, nz, n_genes
            );
        }
    }

    // Splice output (β-sharing): the identity `latent` above is the RAW spliced θ.
    // On the cell axis we also emit the RAW velocity increment δ (`velocity.parquet`),
    // same cell×H layout / barcodes as the latent — magnitude = speed, direction =
    // velocity, no post-hoc unit-norm. The nascent state is just θ + δ = latent +
    // velocity (derivable, not written). Per-gene velocity, if wanted, is the in-model
    // δ_g (`--delta-l2`, written to delta_dictionary.parquet) — not a post-hoc average.
    let h = args.model.embedding_dim;
    let n = unified.barcodes.len();
    if let Some(velocity) = &out.cell_velocity {
        let vel_t = candle_util::candle_core::Tensor::from_vec(velocity.clone(), (n, h), &cpu)?;
        ge::save_embedding(
            &format!("{}.velocity.parquet", args.out),
            &vel_t,
            &unified.barcodes,
            "cell",
        )
        .context("save velocity")?;
        info!(
            "wrote {}.velocity.parquet (raw cell velocity increment δ; ‖δ‖ = speed)",
            args.out
        );
    }

    info!(
        "done (gem — raw spliced identity θ + raw velocity increment δ over the bge engine) — prefix '{}'",
        args.out
    );
    Ok(())
}

/// Split a gem feature row `{gene}/count/{spliced|unspliced}` into its gene key and
/// whether it is the unspliced track. Rows not matching that shape fall back to
/// `(whole name, spliced)` — defensive; genes-only input is all count rows.
fn split_count_row(name: &str) -> (&str, bool) {
    match name.rsplit_once("/count/") {
        Some((gene, suffix)) => (gene, suffix == "unspliced"),
        None => (name, false),
    }
}

/// Intern each row's gene key (see [`split_count_row`]) to a dense gene id. Returns
/// `(row_to_gene, gene_names)`, `gene_names[gid]` the first-seen key for gene `gid`
/// (id order). Single source of the β-sharing gene-identity map — used by the HVG
/// gene filter (pre-subset) and [`build_splice_factor`] (post-subset).
fn intern_gene_keys(feature_names: &[Box<str>]) -> (Vec<u32>, Vec<Box<str>>) {
    let mut gene_ids: FxHashMap<Box<str>, u32> = FxHashMap::default();
    let mut row_to_gene = Vec::with_capacity(feature_names.len());
    let mut gene_names: Vec<Box<str>> = Vec::new();
    for name in feature_names {
        let gene = split_count_row(name).0;
        // Borrow-first: only allocate a `Box<str>` key on a genuinely new gene.
        let gid = match gene_ids.get(gene) {
            Some(&gid) => gid,
            None => {
                let gid = gene_ids.len() as u32;
                gene_ids.insert(gene.into(), gid);
                gene_names.push(gene.into());
                gid
            }
        };
        row_to_gene.push(gid);
    }
    (row_to_gene, gene_names)
}

/// Build the per-gene β-sharing feature factorization + the id-ordered gene names.
/// Each row is `{gene}/count/{spliced|unspliced}`; rows sharing a `{gene}` key map to
/// one gene id (so both tracks embed as `β_g`), and the `unspliced` rows are flagged
/// so phase 2 can split each cell's edges (identity θ from spliced, velocity increment
/// δ from unspliced).
fn build_splice_factor(
    feature_names: &[Box<str>],
) -> (graph_embedding_util::FeatFactorSpec, Vec<Box<str>>) {
    let (row_to_gene, gene_names) = intern_gene_keys(feature_names);
    let unspliced_rows: Vec<bool> = feature_names.iter().map(|n| split_count_row(n).1).collect();
    (
        graph_embedding_util::FeatFactorSpec {
            row_to_gene,
            unspliced_rows,
        },
        gene_names,
    )
}

fn validate_args(args: &GemArgs) -> anyhow::Result<()> {
    anyhow::ensure!(
        args.model.embedding_dim > 0,
        "--embedding-dim must be > 0 (got {})",
        args.model.embedding_dim
    );
    Ok(())
}
