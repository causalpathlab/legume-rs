//! Entry point for `senna gem` (alias `gem-embedding`).
//!
//! Genes-only joint embedding, driven straight through the shared
//! `graph_embedding_util` engine via `senna bge`'s driver
//! ([`crate::bge::driver::fit_embed_family`]): rows = features (no modality
//! split, no per-gene factorization), the bilinear score
//! `e_feat·e_cell + b_feat + b_cell`, phase-1 multilevel-pseudobulk training +
//! phase-2 analytical per-cell projection, and the same output set `senna
//! bge` writes. gem is, for now, bge run over every row of a gene-count
//! matrix; a later task reintroduces spliced/unspliced as explicit tracks on
//! top of this same driver.

use anyhow::Context;
use data_beans::sparse_io_vector::ColumnAlignment;
use graph_embedding_util::data::UnifiedData;
use graph_embedding_util::{load_unified_data, FeatureNameKind, LoadUnifiedArgs};
use log::info;
use matrix_util::common_io::{basename, mkdir_parent};
use rayon::ThreadPoolBuilder;

use crate::bge::driver::{fit_embed_family, EmbedPlan};
use crate::gem::args::GemArgs;
use crate::gem::sample_id::{file_sample_id, longest_common_underscore_suffix};

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

    let batch_files = crate::senna_input::effective_batch_files(
        args.collapse.ignore_batch,
        args.batch_files.as_deref(),
    );

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

/// Genes-only joint embedding over the shared `graph_embedding_util` engine,
/// via the driver `senna bge` also runs through. Writes the same output set
/// `senna bge` does (`{out}.{cell_embedding,feature_embedding,dictionary,
/// feature_loading,*_bias,pb_embedding,pb_batch}.parquet`, `{out}.senna.json`,
/// and — unless `--skip-etm` is added to gem's own surface — the ETM tables).
fn run_gem_genes_bge(
    args: &GemArgs,
    feature_kind: FeatureNameKind,
    batch_files: Option<&[Box<str>]>,
) -> anyhow::Result<()> {
    // Genes load (+ per-file `@sample` tag for batch identity when >1 file and
    // no explicit --batch-files). The sample-id strip is the explicit
    // `--genes-sample-strip`, else the longest common `_`-suffix across the
    // genes basenames.
    let genes = args.genes()?;
    let do_tag = batch_files.is_none() && genes.len() > 1;
    let genes_strip: Box<str> = if !args.collapse.genes_sample_strip.is_empty() {
        args.collapse.genes_sample_strip.clone()
    } else if do_tag {
        let genes_bn: Vec<Box<str>> = genes
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
    let data_files: Vec<Box<str>> = genes.to_vec();
    let unified = load_modality(
        genes,
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

    // Optional gene-level HVG feature filter. NOTE this is NOT what `--n-hvg` does
    // in `senna bge`, and the difference is deliberate at both ends: bge keeps the
    // full feature axis and uses the selection only to WEIGHT its random
    // projection, while here it also only weights (never drops) — but the
    // selection itself ranks GENES, pooling a gene's spliced + unspliced rows
    // together first (`gem::hvg::build_gene_index`), so both tracks of a
    // selected gene carry projection weight together.
    //
    // `--must-train-features` force-includes a curated panel on top of that cut, at
    // the GENE level (so both splice tracks of a kept gene come along). Loaded only
    // when the HVG cut is on.
    let hvg_on = args.collapse.n_hvg > 0;
    let selection_on = hvg_on;
    // `--markers` is force-trained alongside `--must-train-features`. The annotators read
    // only the TRAINED feature rows, so a marker off the trained axis is absent from the
    // panel rather than merely down-weighted — naming the panel here is what keeps the genes
    // the calls are made on and the genes the model fit the same set.
    //
    // The panel is kept separately as well as unioned in, so the coverage log below can say
    // what share of the trained axis it is *without* re-reading the file.
    let explicit = data_beans_alg::hvg::load_must_train(
        args.collapse.must_train_features.as_deref(),
        selection_on,
    )?;
    let panel =
        data_beans_alg::hvg::load_must_train(args.collapse.markers.as_deref(), selection_on)?;
    let parts: Vec<&data_beans_alg::hvg::MustTrainFeatures> = [explicit.as_ref(), panel.as_ref()]
        .into_iter()
        .flatten()
        .collect();
    let must_train =
        (!parts.is_empty()).then(|| data_beans_alg::hvg::MustTrainFeatures::union(&parts));

    // Per-ROW projection weights over the FULL feature axis, matching what `senna bge`
    // does. `None` when `--n-hvg 0`.
    let mut hvg_row_weights: Option<Vec<f32>> = None;

    if hvg_on {
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
        let (row_gene, genes) = crate::gem::hvg::build_gene_index(&unified.feature_names)?;
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
        let mut selected = select_hvg_by_stats(&gmean, &gvar, args.collapse.n_hvg);

        // Force-include, resolved against the GENE keys (not the count rows), so a
        // `CD8A` panel entry keeps `CD8A/count/spliced` AND `CD8A/count/unspliced`.
        // The index lowercases and hash-indexes the whole gene vocabulary, so build it once
        // and resolve both lists (the force-train union, then the panel) against it.
        let gene_index = data_beans::utilities::name_matching::GeneIndex::build(&genes);
        let forced = if let Some(must_train) = must_train.as_ref() {
            let forced = must_train.resolve_with(&gene_index);
            let added = data_beans_alg::hvg::union_indices(&mut selected, &forced);
            info!(
                "force-train: {added} gene(s) added on top of the HVG cut \
                 ({} of the {} matched were already HVGs)",
                forced.len() - added,
                forced.len()
            );
            added
        } else {
            0
        };

        let keep_genes: rustc_hash::FxHashSet<usize> = selected.into_iter().collect();

        // What share of the trained axis IS the marker panel? Worth saying out loud, because
        // it is the price of forcing the panel in: the embedding is now partly built to
        // separate the very compartments the panel will later be used to call, so a
        // downstream "the markers agree with the clusters" check is a check on the grouping,
        // not an independent confirmation. Small share ⇒ the axis still has its own opinion.
        if let Some(panel) = panel.as_ref() {
            let on_axis = panel
                .resolve_quiet_with(&gene_index)
                .into_iter()
                .filter(|g| keep_genes.contains(g))
                .count();
            info!(
                "--markers: {on_axis} panel gene(s) on the trained axis = {:.0}% of its {} \
                 gene(s). The embedding is trained to separate what the panel will later \
                 call, so read `annotate-by-projection`'s agreement as a check on the grouping, not an \
                 independent one.",
                100.0 * on_axis as f32 / keep_genes.len().max(1) as f32,
                keep_genes.len()
            );
        }
        // WEIGHT, do not subset — the `senna bge` semantics. Non-selected genes get
        // projection weight 0, so they sit out the basis the pseudobulk partition is built
        // from, but they stay on the feature axis: still trained, still in the
        // dictionary and the co-embedding.
        let mut w = vec![0.0f32; unified.n_features()];
        for (r, slot) in w.iter_mut().enumerate() {
            if keep_genes.contains(&(row_gene[r] as usize)) {
                *slot = 1.0;
            }
        }
        let weighted_rows = w.iter().filter(|&&x| x > 0.0).count();
        hvg_row_weights = Some(w);
        info!(
            "HVG weighting (--n-hvg {}): {} of {} genes selected ({} HVG + {} force-kept) \
             → {} of {} feature rows carry the projection; every gene still trains",
            args.collapse.n_hvg,
            keep_genes.len(),
            n_genes,
            keep_genes.len() - forced,
            forced,
            weighted_rows,
            unified.n_features()
        );
    }

    fit_embed_family(EmbedPlan {
        kind: crate::run_manifest::RunKind::Gem,
        knobs: args.knobs(),
        unified,
        data_files,
        multiome: None,
        hvg_weights: hvg_row_weights,
        tracks: None,
        offset_l2: 0.0,
        pb_reference: None,
        init_from: None,
        train_args: crate::run_manifest::record_train_args(args)?,
        after_fit: None,
    })
}

fn validate_args(args: &GemArgs) -> anyhow::Result<()> {
    // Fail on an ambiguous / empty gene spec before any I/O.
    args.genes()?;
    anyhow::ensure!(
        args.model.embedding_dim > 0,
        "--embedding-dim must be > 0 (got {})",
        args.model.embedding_dim
    );
    Ok(())
}
