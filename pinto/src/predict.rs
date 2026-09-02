//! `pinto predict` — apply a trained `cage` run to a new sample.
//!
//! What crosses samples is the gene side and the community dictionary; the
//! geometry is local and is rebuilt from the new sample's own coordinates:
//!
//! 1. The new data goes through the same preprocessing as `cage` — coordinates,
//!    batch labels, the cell-cell KNN graph (spatial under `-c`, expression
//!    otherwise) — with the same flags, so the pairs are what `cage` would have
//!    built on this sample.
//! 2. The frozen gene embedding `{model}.feature_embedding.parquet` is aligned
//!    to the new sample's gene axis by name. A gene the model never saw gets
//!    no dictionary row: it is dropped from the partition rather than seeded
//!    (this is inference, not a warm start — inventing a row would let a gene
//!    the model knows nothing about pull on every pair's latent).
//! 3. Every cell pair is projected onto that dictionary by the same Poisson MAP
//!    `cage` uses after training (`pair_projection`), giving a pair latent in
//!    the trained space.
//! 4. Each pair is assigned to the nearest trained link community by cosine
//!    against the training centroids — the mean L2-normalized pair latent of
//!    each community, recomputed here from `{model}.latent.parquet` and
//!    `{model}.link_community.parquet` so no new artifact is needed.
//! 5. A cell's propensity is its incident-edge fraction per community, exactly
//!    the definition `cage` / `lc` / `dsvd` publish, and the cell embedding is
//!    the propensity-weighted centroid readout `cage` writes.
//!
//! Outputs mirror `cage`'s inference tables: `{out}.{coord_pairs, latent,
//! link_community, propensity, gene_community, cell_embedding}.parquet` and a
//! `{out}.pinto.json` manifest, so `pinto plot` and `pinto annotate` read a
//! predicted sample as they would a fitted one.

use crate::cell_activity_graph_embedding::args::GeneNameMode;
use crate::cell_activity_graph_embedding::pair_projection::{
    project_pairs, PairBatchDivisor, PairLatent, PairProjectionArgs, PairScore, ProjectionArgs,
};
use crate::link_community::outputs::write_partition_outputs;
use crate::util::cell_pairs::SrtCellPairs;
use crate::util::common::*;
use crate::util::metadata::{create_cage_metadata, RunInputs};
use crate::util::srt_pipeline::{
    preprocess_srt, GeneAxisMode, SrtPreprocessConfig, SrtPreprocessed,
};
use auxiliary_data::frozen_features::{load_frozen_feature_host, FrozenLoadArgs};
use clap::Args;
use graph_embedding_util::embedding_col_names;
use log::info;
use matrix_util::common_io::mkdir_parent;
use matrix_util::traits::IoOps;
use rayon::prelude::*;
use std::path::Path;

#[derive(Args, Debug)]
pub struct PredictArgs {
    #[command(flatten)]
    pub common: crate::util::input::SrtInputArgs,

    #[arg(
        long,
        required = true,
        help = "Trained `pinto cage` prefix (reads its feature_embedding, latent, link_community)",
        long_help = "Trained `pinto cage` run prefix. Reads:\n  \
                     {model}.feature_embedding.parquet  gene × D frozen dictionary\n  \
                     {model}.latent.parquet             training pair latent (E × D)\n  \
                     {model}.link_community.parquet     training pair → community\n\
                     The last two give the community centroids the new pairs are assigned to."
    )]
    pub model: Box<str>,

    #[arg(
        long,
        default_value_t = GeneNameMode::Auto,
        value_enum,
        help = "Gene-name canonicalization for matching the model's genes to the new sample"
    )]
    pub gene_name_mode: GeneNameMode,

    #[arg(
        long,
        default_value_t = 0.0,
        value_name = "FRACTION",
        help = "Refuse to predict below this share of the model's genes (0 = no gate)",
        long_help = "Gene coverage is always reported.\n\
                     This turns it into a hard floor.\n\
                     \n\
                     The share is of the MODEL's genes, not this sample's,\n\
                     so a whole-transcriptome query is not penalized\n\
                     for carrying genes the model never had.\n\
                     \n\
                     Zero mapped genes is always refused — that is a naming failure."
    )]
    pub min_gene_overlap: f32,

    #[arg(
        long,
        help = "Drop genes the model never saw instead of initializing them through its modules",
        long_help = "Genes of this sample that the model never saw are placed through the model's\n\
                     learned modules (needs {model}.module_membership.parquet and\n\
                     {model}.module_dictionary.parquet): membership averaged over the closest\n\
                     matched genes by count profile over pseudobulks of this sample, row =\n\
                     membership times the module dictionary, with their own counts on the\n\
                     partition axis and provenance in {out}.gene_embedding_init.parquet.\n\
                     This flag restores the historical drop: no row, not on the axis."
    )]
    pub no_init_genes: bool,

    #[arg(
        long,
        default_value_t = graph_embedding_util::transfer::DEFAULT_INIT_NEIGHBOURS,
        value_name = "K",
        help = "membership init: matched genes whose memberships are averaged"
    )]
    pub gene_init_neighbours: usize,

    #[arg(
        long,
        default_value_t = graph_embedding_util::transfer::DEFAULT_SIMILARITY_FLOOR,
        value_name = "S",
        help = "membership init: below this best profile similarity a gene takes the diffuse prior"
    )]
    pub gene_init_similarity_floor: f32,

    #[arg(
        long,
        default_value_t = 1.0,
        help = "Ridge on the pair latent (as in cage)",
        hide = true
    )]
    pub pair_ridge: f32,

    #[arg(
        long,
        default_value_t = 300,
        help = "Adam steps per pair (as in cage)",
        hide = true
    )]
    pub pair_steps: usize,

    #[arg(
        long,
        default_value_t = 512,
        help = "Genes sampled per step for the projection log-partition; 0 = all",
        hide = true
    )]
    pub pair_gene_sample: usize,

    #[arg(
        long,
        default_value_t = 8192,
        help = "Cell pairs per projection read block",
        hide = true
    )]
    pub pair_block: usize,

    #[arg(
        long,
        help = "Skip NB Fisher-info weighting of the gene_community table",
        hide = true
    )]
    pub no_fisher_weights: bool,

    /// Restrict the agreement correlations to these features (one name per line).
    ///
    /// Off by default because a pair-level correlation sorts the gene axis once
    /// per pair, and a sample has far more pairs than cells. Pass the same file
    /// here and to `senna predict --eval-features` and the two commands'
    /// `spearman` / `pearson_log1p` columns are the same measurement.
    #[arg(
        long,
        value_name = "FILE",
        help = "Score the agreement correlations on these features only (one name per line)"
    )]
    pub eval_features: Option<Box<str>>,

    /// Training data, read once for the per-gene totals the frozen dictionary
    /// needs.
    ///
    /// Those totals are not only the null: they become `b_g`, the per-gene log
    /// abundance that is HALF of the pair log-rate `b_g + <e_g, e_uv>`. Taken
    /// from the query, as they were before this flag existed, the prediction is
    /// anchored on the very data it is scored against and the likelihood is not
    /// held out at all -- a dictionary that had learned nothing would still
    /// score well on `b_g` alone. Pass the training half and both the rate and
    /// the null come from data the model was actually fitted on.
    ///
    /// Same flag, same meaning as `senna predict --null-from`; pass the same
    /// training half to both and the two commands' scores are comparable.
    #[arg(
        long,
        value_name = "FILE",
        num_args = 1..,
        help = "Training data supplying the per-gene totals (b_g) and the null; pass the train half"
    )]
    pub null_from: Option<Vec<Box<str>>>,
}

/// Per-pair community labels from `{model}.link_community.parquet`, **unfiltered**.
///
/// Deliberately not `plot::load::read_link_community`: that is a display loader and
/// drops every row whose `edge_kind` is not spatial, while `{model}.latent.parquet`
/// carries one row per pair including the expression-augmented ones. Pairing the two
/// positionally after that filter either aborts on a self-consistent model — the
/// counts disagree — or, worse, builds every centroid from the wrong pairs.
fn training_communities(path: &str) -> anyhow::Result<Vec<i64>> {
    crate::plot::load::read_link_community_labels(Path::new(path))
}

/// Community centroids from a training run: the mean of the member pairs' L2-normalized
/// latents, renormalized, `[K × D]`.
///
/// The renormalization is what `assign_to_centroids` needs (it compares cosines), and it
/// is a real difference from `cage`'s own `propensity_weighted_cell_embedding`, which
/// leaves centroid length alone so a diffuse community counts for less. Only the
/// assignment uses these; the cell-embedding readout below rescales back.
fn training_centroids(model: &str) -> anyhow::Result<(Mat, Vec<f32>)> {
    let latent = Mat::from_parquet(&format!("{model}.latent.parquet"))?.mat;
    let communities = training_communities(&format!("{model}.link_community.parquet"))?;
    anyhow::ensure!(
        communities.len() == latent.nrows(),
        "{model}: latent has {} pairs but link_community lists {} — the two files are \
         from different runs",
        latent.nrows(),
        communities.len()
    );
    let k = communities
        .iter()
        .copied()
        .filter(|&c| c >= 0)
        .max()
        .map_or(0, |m| m as usize + 1);
    anyhow::ensure!(k > 0, "{model}: link_community has no communities");
    let d = latent.ncols();
    let mut centroids = Mat::zeros(k, d);
    let mut counts = vec![0f32; k];
    for (e, &c) in communities.iter().enumerate() {
        if c < 0 {
            continue;
        }
        let mut row = latent.row(e).into_owned();
        let n = row.norm();
        if n > 0.0 {
            row /= n;
        }
        let mut dst = centroids.row_mut(c as usize);
        dst += row;
        counts[c as usize] += 1.0;
    }
    for c in 0..k {
        let mut row = centroids.row_mut(c);
        let nrm = row.norm();
        if nrm > 0.0 {
            row /= nrm;
        }
    }
    let live = counts.iter().filter(|&&n| n > 0.0).count();
    anyhow::ensure!(
        live > 0,
        "{model}: every community centroid is empty or degenerate"
    );
    if live < k {
        // An empty community keeps an all-zero row, which scores exactly 0.0 cosine
        // against every pair and so beats any genuinely anti-correlated centroid — the
        // "empty type is a magnet" failure already fixed once in `annotate`.
        // `assign_to_centroids` refuses to assign to a zero row instead.
        log::warn!(
            "{model}: {} of {k} communities have no pairs in the training latent; \
             they cannot be predicted into",
            k - live
        );
    }
    info!("Training centroids: {live} live of {k} communities in a {d}-dim pair latent");
    Ok((centroids, counts))
}

/// Nearest live centroid by cosine, for every (already L2-normalized) pair latent.
///
/// `None` where the pair cannot be assigned: a pair with no counts on any model gene
/// projects to the origin and is equidistant from everything, and a community with no
/// training pairs has a zero centroid that would otherwise capture every pair whose
/// cosine to all live centroids is negative. Both abstain rather than silently landing
/// in community 0.
fn assign_to_centroids(latent_nk: &Mat, centroids_kd: &Mat, counts: &[f32]) -> Vec<Option<usize>> {
    (0..latent_nk.nrows())
        .into_par_iter()
        .map(|e| {
            let row = latent_nk.row(e);
            if row.norm() <= 0.0 {
                return None;
            }
            let mut best: Option<(usize, f32)> = None;
            for (k, &n) in counts.iter().enumerate() {
                if n <= 0.0 {
                    continue;
                }
                let s: f32 = row.dot(&centroids_kd.row(k));
                if !s.is_finite() {
                    continue;
                }
                if best.is_none_or(|(_, b)| s > b) {
                    best = Some((k, s));
                }
            }
            best.map(|(k, _)| k)
        })
        .collect()
}

/// Returns the per-cell propensity and the cell names — the same table
/// written to `{out}.propensity.parquet` — so a caller that consumes the
/// prediction (`pinto impute`) need not round-trip it through parquet.
pub fn predict_cage(args: &PredictArgs) -> anyhow::Result<(Mat, Vec<Box<str>>)> {
    let c = &args.common;
    mkdir_parent(&c.out)?;
    anyhow::ensure!(!c.data_files.is_empty(), "predict: no data files given");

    let peek_names = data_beans::convert::try_open_or_convert(&c.data_files[0])?.row_names()?;
    let feature_kind = args.gene_name_mode.resolve_kind(&peek_names);

    let SrtPreprocessed {
        data_vec,
        coordinates,
        coordinate_names,
        batch_membership,
        batch_effects,
        graph,
        knn,
        edge_source,
        gene_axis,
        row_stats,
        gene_weights,
        n_cells,
        n_rows: _,
        cell_proj,
        ..
    } = preprocess_srt(SrtPreprocessConfig {
        common: c,
        fisher_weights: !args.no_fisher_weights,
        batch_effects: true,
        gene_axis: GeneAxisMode::Strict,
        cell_projection: true,
        feature_kind: Some(feature_kind.clone()),
    })?;
    let gene_axis = gene_axis.expect("GeneAxisMode::Strict must yield Some");
    let cell_names = data_vec.column_names()?;
    let gene_names: Vec<Box<str>> = gene_axis.gene_names().to_vec();
    let n_genes = gene_axis.n_genes();

    // The frozen gene side, on THIS sample's gene axis. Unmatched genes keep a
    // zero row and a zero total, which `PairDictionary` reads as "not on the
    // partition axis" — dropped, not invented.
    let host = load_frozen_feature_host(FrozenLoadArgs {
        dictionary_path: &format!("{}.feature_embedding.parquet", args.model),
        bias_path: None,
        target_feature_names: &gene_names,
        name_kind: feature_kind.clone(),
    })?;
    let n_matched = host.keep_target_indices.len();
    // The MODEL's full feature count, from the dictionary file. NOT
    // `e_feat.nrows()`, which is the count AFTER the intersection and therefore
    // always equals `n_matched` — a coverage built from it is identically 1 and
    // `--min-gene-overlap` can never fire.
    let n_model = host.n_src;
    // The share of the MODEL's genes this sample carries — the same denominator
    // senna's identically-named flag uses. Dividing by the query's gene count
    // instead would refuse a whole-transcriptome sample containing every panel gene.
    let coverage = n_matched as f32 / n_model.max(1) as f32;
    info!(
        "Gene alignment: {n_matched} of the model's {n_model} genes are present here \
         ({:.1}% coverage; this sample has {n_genes} genes, D = {})",
        100.0 * coverage,
        host.h
    );
    anyhow::ensure!(
        n_matched > 0,
        "predict: none of the {n_genes} genes match the model's dictionary — check \
         --gene-name-mode"
    );
    anyhow::ensure!(
        coverage >= args.min_gene_overlap,
        "predict: only {:.1}% of the model's genes are present, below --min-gene-overlap {:.1}%",
        100.0 * coverage,
        100.0 * args.min_gene_overlap
    );
    let mut e_full = Mat::zeros(n_genes, host.h);
    for (i, &g) in host.keep_target_indices.iter().enumerate() {
        e_full.row_mut(g).copy_from(&host.e_feat.row(i));
    }
    // Which genes are on the partition axis: every matched gene, plus — unless
    // `--no-init-genes` — the genes the model never saw, placed through its
    // modules by the same loader cage trains from.
    let mut on_axis: Vec<bool> = vec![false; n_genes];
    for &g in &host.keep_target_indices {
        on_axis[g] = true;
    }
    if !args.no_init_genes && n_matched < n_genes {
        use crate::cell_activity_graph_embedding::pretrained;
        // Profiles over pseudobulks of THIS sample: k-means on its random
        // projection, then per-gene sums per cluster.
        let build_profiles = || -> anyhow::Result<Mat> {
            let proj = cell_proj
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("predict: no cell projection to cluster on"))?;
            let cells_by_dim = proj.proj.transpose(); // [n_cells × k]
            let n_pb = graph_embedding_util::transfer::pseudobulk_count(n_cells);
            let (_, labels) =
                matrix_util::principal_graph::kmeans_centroids_seeded(&cells_by_dim, n_pb, 20, 0);
            let row_profiles = crate::link_community::profiles::coarsen_cell_expression_dense(
                &data_vec, &labels, n_pb,
            )?;
            Ok(gene_axis
                .pool_rows_opt(&row_profiles)
                .unwrap_or(row_profiles))
        };
        let pre = pretrained::load_pretrained_gene_embedding(pretrained::PretrainedArgs {
            dictionary_path: &format!("{}.feature_embedding.parquet", args.model),
            bias_path: None,
            gene_names: &gene_names,
            name_kind: feature_kind.clone(),
            gene_profiles: &build_profiles,
            membership_init: Some(graph_embedding_util::transfer::AlignKnobs {
                k: args.gene_init_neighbours,
                similarity_floor: args.gene_init_similarity_floor,
            }),
        })?;
        let mut n_init = 0usize;
        for (g, r) in pre.records.iter().enumerate() {
            if r.init == pretrained::InitKind::Membership {
                e_full.row_mut(g).copy_from(&pre.e_gene.row(g));
                on_axis[g] = true;
                n_init += 1;
            }
        }
        if n_init > 0 {
            pretrained::write_init_report(&c.out, &pre.records)?;
            info!(
                "{n_init} genes the model never saw were initialized through its modules and \
                 join the partition axis (see {}.gene_embedding_init.parquet)",
                c.out
            );
        } else {
            info!(
                "no module tables beside {}.feature_embedding.parquet: genes the model never \
                 saw stay off the partition axis",
                args.model
            );
        }
    }
    let mut gene_totals = match args.null_from.as_deref() {
        Some(files) => training_gene_totals(files, c, &feature_kind, &gene_names)?,
        None => {
            log::warn!(
                "no --null-from: per-gene totals come from the QUERY, so the pair log-rate \
                 is anchored on the data being scored and llik is not held out. Pass the \
                 training half"
            );
            let row_totals: Vec<f64> = match row_stats {
                Some(st) => st.sum().iter().map(|&x| f64::from(x)).collect(),
                None => {
                    crate::link_community::profiles::compute_row_totals(&data_vec, c.block_size)?
                }
            };
            gene_axis.pool_totals(&row_totals)
        }
    };
    for (g, t) in gene_totals.iter_mut().enumerate() {
        if !on_axis[g] {
            *t = 0.0;
        }
    }

    let eval_features: Option<Vec<Box<str>>> = match args.eval_features.as_deref() {
        // The same reader `senna predict --eval-features` uses. The help on both
        // commands tells the user to pass ONE file to both; parsing it two ways
        // meant a two-column panel scored senna on the right genes and matched
        // nothing here.
        Some(path) => Some(
            matrix_util::common_io::read_name_list(path)
                .map_err(|e| anyhow::anyhow!("reading --eval-features {path}: {e}"))?,
        ),
        None => None,
    };

    let (centroids, centroid_counts) = training_centroids(&args.model)?;
    anyhow::ensure!(
        centroids.ncols() == host.h,
        "{}: the pair latent is {}-dim but the gene dictionary is {}-dim",
        args.model,
        centroids.ncols(),
        host.h
    );

    // Pairs on the new sample's own graph.
    let srt_cell_pairs = SrtCellPairs::with_graph(
        &data_vec,
        &coordinates,
        &graph,
        edge_source.as_deref(),
        Some(&batch_membership),
    );
    srt_cell_pairs.write_coord_pairs(&c.out, &coordinate_names)?;
    let fine_edges: Vec<(u32, u32)> = srt_cell_pairs
        .inner
        .pairs()
        .iter()
        .map(|&(i, j)| (i as u32, j as u32))
        .collect();
    let n_edges = fine_edges.len();
    info!("{n_cells} cells, {n_genes} genes, {n_edges} pairs");
    anyhow::ensure!(
        n_edges > 0,
        "predict: this sample yielded no cell pairs, so there is nothing to predict — \
         check --coord and the neighbourhood flags (-k / --knn-expr)"
    );

    let batch_id_of: HashMap<Box<str>, u32> = {
        let mut uniq: Vec<Box<str>> = batch_membership.to_vec();
        uniq.sort();
        uniq.dedup();
        uniq.into_iter()
            .enumerate()
            .map(|(i, b)| (b, i as u32))
            .collect()
    };
    let batch_of_cell: Vec<u32> = batch_membership.iter().map(|b| batch_id_of[b]).collect();
    let pair_batch = batch_effects.as_ref().map(|delta| PairBatchDivisor {
        delta,
        batch_of_cell: &batch_of_cell,
    });

    let PairLatent {
        latent,
        bias: _,
        scores,
    } = project_pairs(
        &data_vec,
        &fine_edges,
        &e_full,
        pair_batch,
        &PairProjectionArgs {
            projection: ProjectionArgs {
                ridge: args.pair_ridge,
                steps: args.pair_steps,
                gene_sample: args.pair_gene_sample,
            },
            seed: c.seed,
            pair_block: args.pair_block,
            eval_features: eval_features.clone(),
        },
        &gene_axis,
        &gene_totals,
    )?;

    // Composition only, as in cage: L2-normalize each pair before the cut.
    let mut latent_kn = latent.transpose();
    latent_kn.normalize_columns_inplace();
    let latent_nk = latent_kn.transpose();
    latent_nk.to_parquet_with_names(
        &(c.out.to_string() + ".latent.parquet"),
        (None, Some("cell_pair")),
        Some(&embedding_col_names(latent_nk.ncols())),
    )?;

    write_predictive(&c.out, &scores)?;

    let assigned = assign_to_centroids(&latent_nk, &centroids, &centroid_counts);
    let k = centroids.nrows();
    let n_abstain = assigned.iter().filter(|a| a.is_none()).count();
    // An unassignable pair is dropped from the partition rather than parked in a
    // community: it contributes to no cell's propensity, which is the honest reading
    // of "this pair carries no evidence".
    let (kept_edges, labels): (Vec<(usize, usize)>, Vec<usize>) = fine_edges
        .iter()
        .zip(&assigned)
        .filter_map(|(&(u, v), a)| a.map(|k| ((u as usize, v as usize), k)))
        .unzip();
    {
        let mut hist = vec![0usize; k];
        for &l in &labels {
            hist[l] += 1;
        }
        let used = hist.iter().filter(|&&n| n > 0).count();
        info!(
            "Assigned {} of {n_edges} pairs to {used} of {k} trained communities (largest {}); \
             {n_abstain} abstained",
            labels.len(),
            hist.iter().max().copied().unwrap_or(0)
        );
    }
    anyhow::ensure!(
        !labels.is_empty(),
        "predict: no pair could be assigned to a trained community — check that the \
         model and this sample share genes (see the coverage line above)"
    );

    let (propensity, _gene_community) = write_partition_outputs(
        &c.out,
        &kept_edges,
        &labels,
        n_cells,
        k,
        &cell_names,
        &data_vec,
        gene_weights.as_deref(),
        &gene_axis,
        c.block_size,
        // The edge-kind column is per-EDGE and `kept_edges` is a subset, so it can only
        // be forwarded when nothing abstained.
        (n_abstain == 0)
            .then_some(srt_cell_pairs.edge_kind.as_deref())
            .flatten(),
    )?;

    // The same readout cage writes: propensity-weighted community centroids.
    let e_cell = &propensity * &centroids;
    e_cell.to_parquet_with_names(
        &(c.out.to_string() + ".cell_embedding.parquet"),
        (Some(&cell_names), Some("cell")),
        Some(&embedding_col_names(e_cell.ncols())),
    )?;

    let coord_file_str = c.coord_files_joined();
    let mut meta = create_cage_metadata(
        &RunInputs {
            prefix: &c.out,
            data_files: &c.data_files,
            coord_file: coord_file_str.as_deref(),
            coord_columns: &coordinate_names,
            n_cells,
            n_genes,
            n_edges,
            graph: (&knn).into(),
            k,
        },
        batch_effects.is_some(),
        None,
    );
    meta.command = "predict".to_string();
    let meta_path = std::path::PathBuf::from(format!("{}.pinto.json", c.out));
    meta.write(&meta_path)?;
    info!("Wrote {}", meta_path.display());
    info!("Done");
    Ok((propensity, cell_names))
}

/// Per-pair held-out scores, under the same column names `senna predict` uses
/// for the same quantities.
///
/// The `eval_` prefix is not decoration. senna's `predictive.parquet` also has a
/// bare `llik` / `llik_per_count`, and those are the *backend's own*
/// decoder-dependent likelihood, which must not be compared across families. The
/// `eval_` columns are the multinomial over the scored genes, which is what both
/// commands agree on. Writing pinto's comparable number under the bare name
/// would put two different estimands in one column of two files that share a
/// filename — a benchmark reading `llik_per_count` from both would rank an NB
/// density against multinomial nats/count and get a plausible wrong answer.
///
/// `eval_llik_per_count` is nats per observed count, invariant to how many cells
/// were pooled into the profile, so a pair's value sits on the same axis as a
/// cell's.
fn write_predictive(out: &str, scores: &[PairScore]) -> anyhow::Result<()> {
    let n = scores.len();
    let mut pred = Mat::zeros(n, 6);
    let (mut llik, mut null, mut count) = (0f64, 0f64, 0f64);
    for (i, s) in scores.iter().enumerate() {
        pred[(i, 0)] = s.llik;
        pred[(i, 1)] = s.total;
        // NaN, not 0: a pair with no counts has no per-count score, and 0 nats
        // per count reads as a PERFECT prediction to anything averaging this.
        pred[(i, 2)] = if s.total > 0.0 {
            s.llik / s.total
        } else {
            f32::NAN
        };
        pred[(i, 3)] = s.agreement.spearman;
        pred[(i, 4)] = s.agreement.pearson_log1p;
        // The floor, per pair and in the same units — so a row is readable on its
        // own rather than only against the aggregate on the log line.
        pred[(i, 5)] = if s.total > 0.0 {
            s.null_llik / s.total
        } else {
            f32::NAN
        };
        llik += f64::from(s.llik);
        null += f64::from(s.null_llik);
        count += f64::from(s.total);
    }
    let cols: Vec<Box<str>> = vec![
        "eval_llik".into(),
        "eval_count".into(),
        "eval_llik_per_count".into(),
        "spearman".into(),
        "pearson_log1p".into(),
        "eval_null_llik_per_count".into(),
    ];
    pred.to_parquet_with_names(
        &(out.to_string() + ".predictive.parquet"),
        (None, Some("cell_pair")),
        Some(&cols),
    )?;
    if count > 0.0 {
        info!(
            "Predictive: llik/count {:.4} vs abundance null {:.4} (gain {:+.4}) over {:.0} counts",
            llik / count,
            null / count,
            (llik - null) / count,
            count
        );
    }
    info!("Wrote {out}.predictive.parquet");
    Ok(())
}

/// Per-gene totals from the training half, on this sample's gene axis.
///
/// The training data gets its own [`GeneAxis`], so a channelized training matrix
/// pools the same way this one does before the two are matched by name. A gene
/// the training half never saw keeps a zero total, which `PairDictionary` reads
/// as "not on the partition axis" and drops -- the same treatment a gene with no
/// counts here gets, and the honest one: the model has no abundance for it.
fn training_gene_totals(
    files: &[Box<str>],
    common: &crate::util::input::SrtInputArgs,
    feature_kind: &auxiliary_data::feature_names::FeatureNameKind,
    target_genes: &[Box<str>],
) -> anyhow::Result<Vec<f64>> {
    // The query's row names were canonicalized on the way in, by the same
    // `--gene-name-mode` this uses. Without applying it here the two sides are
    // spelled differently and nothing matches — which is exactly the failure this
    // hit first, and it surfaced as "shares no expressed gene" rather than as a
    // naming problem.
    let mut loaded = data_beans::sparse_io_vector::SparseIoVec::new();
    if let Some(canon) = feature_kind.clone().into_canonicalizer() {
        loaded = loaded
            .with_row_canonicalizer(move |name| canon(name))
            .expect("with_row_canonicalizer on empty SparseIoVec");
    }
    for f in files {
        let mut data = data_beans::convert::try_open_or_convert(f)?;
        if common.preload_data {
            data.preload_columns()?;
        }
        loaded.push(std::sync::Arc::from(data), None)?;
    }
    let axis = crate::util::gene_axis::GeneAxis::resolve(&loaded.row_names()?)?;
    let per_row = crate::link_community::profiles::compute_row_totals(&loaded, common.block_size)?;
    let per_gene = axis.pool_totals(&per_row);

    let total_of: HashMap<&str, f64> = axis
        .gene_names()
        .iter()
        .map(std::convert::AsRef::as_ref)
        .zip(per_gene.iter().copied())
        .collect();
    let out: Vec<f64> = target_genes
        .iter()
        .map(|g| total_of.get(g.as_ref()).copied().unwrap_or(0.0))
        .collect();

    let matched = out.iter().filter(|&&t| t > 0.0).count();
    anyhow::ensure!(
        matched > 0,
        "--null-from shares no expressed gene with this sample after canonicalizing names \
         with --gene-name-mode; check that it points at the training data for this model"
    );
    info!(
        "Training totals: {matched} of {} genes carry counts in the training half ({} cells)",
        target_genes.len(),
        loaded.num_columns()
    );
    Ok(out)
}
