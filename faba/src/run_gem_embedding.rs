//! Entry point for `faba gem` (alias `gem-embedding`).
//!
//! Genes-only joint embedding: each row `{gene}/count/{spliced|unspliced}` is an
//! independent feature sharing the cell axis, and a gene's two tracks embed
//! identically as `β_g` (β-sharing) via the per-gene factorization. Driven
//! straight through the shared `graph_embedding_util` engine — the bilinear
//! score `e_feat·e_cell + b_feat + b_cell`, phase-1 multilevel-pseudobulk
//! training + phase-2 analytical per-cell projection, then the SIMBA feature
//! co-embedding. The splice deviation is recovered post-hoc on the CELL axis by
//! the dual phase-2 projection (`{out}.axis_delta.parquet`).

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
use faba::gem::m6a::{M6aArmBuilder, M6aData, M6aParams};
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
/// cell_bias}.parquet`, the co-embedded `{out}.feature_embedding.parquet`, and
/// the dual-projection cell-axis splice δ `{out}.axis_delta.parquet`.
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
        feature_kind.clone(),
        args.runtime.preload_data,
    )
    .context("load genes backend")?;
    info!(
        "genes loaded: {} features × {} cells, {} batch(es)",
        unified.n_features(),
        unified.n_cells(),
        unified.n_batches()
    );

    // Per-gene β-sharing factorization. Each row `{gene}/count/{spliced|unspliced}`
    // maps to its gene, so a gene's spliced and unspliced tracks embed identically
    // as `β_g`. The splice deviation is recovered post-hoc on the CELL axis by the
    // dual phase-2 projection — there it is identifiable, whereas a gene-side δ_g
    // would trade off against an equal cell-axis shift.
    let factor = build_splice_factor(&unified.feature_names);
    let n_genes = factor
        .row_to_gene
        .iter()
        .copied()
        .max()
        .map_or(0, |m| m as usize + 1);
    info!(
        "β-sharing factor: {} genes from {} count rows ({} unspliced rows); \
         splice δ → cell axis (dual projection)",
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

    // Optional m6A co-embedding arm: load the methylated (M) / unmethylated (U)
    // backends, gene-pool, and hand a builder to the generic geu aux seam.
    // `m6a_gene_names` is kept so the trained `w_g`/`a_g` can be saved (and
    // matched to β_g) after the fit consumes the builder.
    let mut m6a_gene_names: Option<Vec<Box<str>>> = None;
    let m6a_aux: Option<Box<dyn ge::AuxArmBuilder>> = if let (Some(conv), Some(unconv)) =
        (args.m6a_converted.as_ref(), args.m6a_unconverted.as_ref())
    {
        let data = load_m6a_data(
            conv,
            unconv,
            &args.collapse.m6a_sample_strip,
            do_tag,
            batch_files,
            feature_kind,
            args.runtime.preload_data,
        )
        .context("load m6A modality")?;
        info!(
            "m6A: {} genes across {} cells loaded (binomial arm: λ={}, κ={}, N0={})",
            data.gene_names.len(),
            data.per_cell.len(),
            args.m6a_lambda,
            args.m6a_kappa,
            args.m6a_n0,
        );
        m6a_gene_names = Some(data.gene_names.clone());
        let params = M6aParams {
            lambda: args.m6a_lambda,
            n0: args.m6a_n0,
            kappa: args.m6a_kappa,
            cov_min: args.m6a_cov_min,
            batch_size: args.train.batch_size,
            refine_weight: args.m6a_refine_weight,
        };
        Some(Box::new(M6aArmBuilder::new(data, params)))
    } else {
        None
    };

    let cfg = ge::FitConfig {
        embedding_dim: args.model.embedding_dim,
        num_levels: args.collapse.num_levels,
        sort_dim: args.collapse.sort_dim,
        knn_pb_samples: args.collapse.knn_pb,
        num_opt_iter: args.collapse.num_opt_iter,
        proj_dim: args.collapse.proj_dim,
        hvg_weights: None,
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
        aux: m6a_aux,
    };
    let out = ge::fit(&mut unified, cfg).context("ge::fit (genes bge)")?;

    // SIMBA co-embedding (one Leiden clustering → eff-cells temperature target,
    // same as bge) + the standard geu outputs.
    let cpu = candle_util::candle_core::Device::Cpu;

    // m6A model outputs: the trained free gene vectors `w_g` and gene baselines
    // `a_g` live in the shared varmap (geu doesn't save them, since it's
    // modality-agnostic). Persist them here so the m6A dictionary is inspectable
    // and `cos(w_g, β_g)` (methylation-vs-expression decoupling) is computable.
    if let Some(gene_names) = m6a_gene_names.as_ref() {
        let vars = out.varmap.data().lock().unwrap();
        if let Some(w) = vars.get("m6a_w_feat") {
            let w_t = w.as_tensor().to_device(&cpu)?;
            ge::save_embedding(
                &format!("{}.m6a_dictionary.parquet", args.out),
                &w_t,
                gene_names,
                "gene",
            )
            .context("save m6A w_g dictionary")?;
        }
        if let Some(a) = vars.get("m6a_a_feat") {
            let a_t = a
                .as_tensor()
                .to_device(&cpu)?
                .reshape((gene_names.len(), 1))?;
            ge::save_embedding(
                &format!("{}.m6a_gene_bias.parquet", args.out),
                &a_t,
                gene_names,
                "gene",
            )
            .context("save m6A a_g gene bias")?;
        }
        info!(
            "wrote {}.m6a_dictionary.parquet (w_g, {} genes) + .m6a_gene_bias.parquet (a_g)",
            args.out,
            gene_names.len()
        );
    }
    let e_feat_cpu = out.model.e_feat.to_device(&cpu)?;
    let e_cell_cpu = out.model.e_cell.to_device(&cpu)?;
    let (_labels, target_eff) =
        ge::cell_clusters(&e_cell_cpu, args.qc.num_topics).context("cell clusters")?;
    ge::write_feature_coembedding(
        &args.out,
        &e_cell_cpu,
        &e_feat_cpu,
        &unified.feature_names,
        target_eff,
    )
    .context("feature co-embedding")?;
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

    // Dual-projection axis δ (cell-side splice deviation, `δ_cell = dir(φ)−dir(θ)`),
    // same `cell × H` layout / barcodes as the latent. Present iff β-sharing was on.
    if let Some(delta) = &out.axis_delta {
        let h = args.model.embedding_dim;
        let delta_t = candle_util::candle_core::Tensor::from_vec(
            delta.clone(),
            (unified.barcodes.len(), h),
            &cpu,
        )?;
        ge::save_embedding(
            &format!("{}.axis_delta.parquet", args.out),
            &delta_t,
            &unified.barcodes,
            "cell",
        )
        .context("save axis δ")?;
        info!("wrote {}.axis_delta.parquet (cell-axis splice δ)", args.out);
    }

    info!(
        "done (gem — genes-only β-sharing + cell-axis splice δ over the bge engine) — prefix '{}'",
        args.out
    );
    Ok(())
}

/// Build the per-gene β-sharing feature factorization from the genes feature
/// axis. Each row is `{gene}/count/{spliced|unspliced}`; rows sharing a `{gene}`
/// key map to one gene id (so both tracks embed as `β_g`), and the `unspliced`
/// rows are flagged so phase 2 can split each cell's edges for the dual axis-δ
/// projection. Rows that don't match the `…/count/…` shape fall back to their
/// own gene tagged spliced (defensive; genes-only input is all count rows).
fn build_splice_factor(feature_names: &[Box<str>]) -> graph_embedding_util::FeatFactorSpec {
    let mut gene_ids: FxHashMap<Box<str>, u32> = FxHashMap::default();
    let mut row_to_gene = Vec::with_capacity(feature_names.len());
    let mut unspliced_rows = Vec::with_capacity(feature_names.len());
    for name in feature_names {
        let s = name.as_ref();
        let (gene, is_unspliced) = match s.rsplit_once("/count/") {
            Some((g, suffix)) => (g, suffix == "unspliced"),
            None => (s, false),
        };
        // Borrow-first: only allocate a `Box<str>` key on a genuinely new gene
        // (the unspliced row of an already-seen gene would otherwise allocate a
        // key that's immediately dropped).
        let gid = match gene_ids.get(gene) {
            Some(&gid) => gid,
            None => {
                let gid = gene_ids.len() as u32;
                gene_ids.insert(gene.into(), gid);
                gid
            }
        };
        row_to_gene.push(gid);
        unspliced_rows.push(is_unspliced);
    }
    graph_embedding_util::FeatFactorSpec {
        row_to_gene,
        unspliced_rows,
    }
}

/// Load + gene-pool the DART m6A modality into an [`M6aData`]. Methylated (M)
/// reads come from the `--m6a-converted` backends, unmethylated (U) from
/// `--m6a-unconverted`; both are `@sample`-tagged with the same scheme as the
/// genes load so their cells align by `barcode@sample`. Site rows
/// `{gene}/m6A/{chr}:{pos}` are pooled to genes (split on `/m6A/`), and per
/// (barcode, gene) we accumulate `(M, U)`.
fn load_m6a_data(
    conv_files: &[Box<str>],
    unconv_files: &[Box<str>],
    m6a_strip: &str,
    do_tag: bool,
    batch_files: Option<&[Box<str>]>,
    feature_kind: FeatureNameKind,
    preload: bool,
) -> anyhow::Result<M6aData> {
    // Converted files strip e.g. `_m6a_converted`; the unconverted partner is
    // the same stem with `converted`→`unconverted`, so both map to one sample id.
    let unconv_strip = m6a_strip.replace("converted", "unconverted");
    let mut conv = load_modality(
        conv_files,
        m6a_strip,
        do_tag,
        batch_files,
        feature_kind.clone(),
        preload,
    )
    .context("load m6A converted (methylated)")?;
    let mut unconv = load_modality(
        unconv_files,
        &unconv_strip,
        do_tag,
        batch_files,
        feature_kind,
        preload,
    )
    .context("load m6A unconverted (unmethylated)")?;
    conv.materialize_cell_triplets()?;
    unconv.materialize_cell_triplets()?;

    let mut gene_ids: FxHashMap<Box<str>, u32> = FxHashMap::default();
    let mut gene_names: Vec<Box<str>> = Vec::new();
    let conv_fg = feature_to_gene(&conv.feature_names, &mut gene_ids, &mut gene_names);
    let unconv_fg = feature_to_gene(&unconv.feature_names, &mut gene_ids, &mut gene_names);

    // (barcode, gene) → (M, U). Two passes: converted fills M, unconverted U.
    // `cell_entry` clones the barcode key only on a genuine miss (the common
    // case is a repeat cell), so the per-triplet loop doesn't allocate a fresh
    // `Box<str>` for every read.
    type CellAcc = FxHashMap<Box<str>, FxHashMap<u32, (f32, f32)>>;
    fn cell_entry<'a>(acc: &'a mut CellAcc, bc: &str) -> &'a mut FxHashMap<u32, (f32, f32)> {
        if !acc.contains_key(bc) {
            acc.insert(bc.into(), FxHashMap::default());
        }
        acc.get_mut(bc).unwrap()
    }
    let mut acc: CellAcc = FxHashMap::default();
    for t in &conv.triplets {
        let bc = &conv.barcodes[t.cell as usize];
        let gene = conv_fg[t.feature as usize];
        cell_entry(&mut acc, bc).entry(gene).or_insert((0.0, 0.0)).0 += t.count;
    }
    for t in &unconv.triplets {
        let bc = &unconv.barcodes[t.cell as usize];
        let gene = unconv_fg[t.feature as usize];
        cell_entry(&mut acc, bc).entry(gene).or_insert((0.0, 0.0)).1 += t.count;
    }
    let per_cell: FxHashMap<Box<str>, Vec<(u32, f32, f32)>> = acc
        .into_iter()
        .map(|(bc, genes)| {
            let edges = genes.into_iter().map(|(g, (m, u))| (g, m, u)).collect();
            (bc, edges)
        })
        .collect();
    Ok(M6aData {
        gene_names,
        per_cell,
    })
}

/// Map each m6A site feature `{gene}/m6A/{chr}:{pos}` to a dense gene id,
/// growing `gene_ids` / `gene_names`. Rows without the `/m6A/` shape fall back
/// to their own full name as the gene (defensive).
fn feature_to_gene(
    feature_names: &[Box<str>],
    gene_ids: &mut FxHashMap<Box<str>, u32>,
    gene_names: &mut Vec<Box<str>>,
) -> Vec<u32> {
    feature_names
        .iter()
        .map(|name| {
            let gene = name
                .as_ref()
                .rsplit_once("/m6A/")
                .map_or(name.as_ref(), |(g, _)| g);
            if let Some(&id) = gene_ids.get(gene) {
                id
            } else {
                let id = gene_names.len() as u32;
                gene_ids.insert(gene.into(), id);
                gene_names.push(gene.into());
                id
            }
        })
        .collect()
}

fn validate_args(args: &GemArgs) -> anyhow::Result<()> {
    anyhow::ensure!(
        args.model.embedding_dim > 0,
        "--embedding-dim must be > 0 (got {})",
        args.model.embedding_dim
    );
    if let Some(k) = args.qc.num_topics {
        anyhow::ensure!(k >= 2, "--num-topics must be ≥ 2 (got {k})");
    }
    anyhow::ensure!(
        args.m6a_converted.is_some() == args.m6a_unconverted.is_some(),
        "--m6a-converted and --m6a-unconverted must be given together (the binomial \
         arm needs both the methylated and unmethylated reads)"
    );
    if let (Some(c), Some(u)) = (args.m6a_converted.as_ref(), args.m6a_unconverted.as_ref()) {
        anyhow::ensure!(
            c.len() == u.len(),
            "--m6a-converted ({}) and --m6a-unconverted ({}) must list the same number of files",
            c.len(),
            u.len()
        );
        anyhow::ensure!(
            args.m6a_n0 >= 0.0,
            "--m6a-n0 must be ≥ 0 (got {})",
            args.m6a_n0
        );
        anyhow::ensure!(
            args.m6a_kappa >= 0.0,
            "--m6a-kappa must be ≥ 0 (got {})",
            args.m6a_kappa
        );
    }
    Ok(())
}
