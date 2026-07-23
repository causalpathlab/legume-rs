//! `faba annotate --mode enrichment` — marker-set annotation of a
//! `faba gem-encoder` / `gem-topic` topic model, routed through the topic
//! dictionary instead of an embedding metric.
//!
//! # Why a second mode at all
//!
//! Projection mode (the default, see the module doc above) calls a cell by the
//! Euclidean nearest marker centroid, which forms `⟨z_c, ρ_g⟩` — a cell↔gene
//! inner product. **For a topic model that quantity is not a metric.** The
//! decoder's dictionary is
//!
//! ```text
//! β = softmax_g( b_g + ⟨α_t, ρ_g⟩ )
//! ```
//!
//! so it depends only on gene-to-gene DIFFERENCES: the per-gene bias `b_g` can
//! absorb the level, and the absolute cell↔gene direction is a **gauge freedom
//! the likelihood never pins**. Two fits with identical likelihood can put the
//! gene cloud on opposite sides of the cell cloud, and the nearest-centroid call
//! flips with them.
//!
//! Enrichment mode routes the call through `β` (the dictionary) and `θ` (the
//! latent) — the two things a topic model actually estimates — and never forms a
//! cell↔gene inner product. It asks, per factor, whether a cell type's marker
//! panel is over-represented at the top of that factor's gene ranking (a
//! weighted KS walk over simplex specificity), calibrates that with two nulls,
//! and carries the surviving factor×type edges to cells through `θ`. Nothing in
//! that chain has a scale or a direction to get wrong.
//!
//! # What it reads
//!
//! `faba annotate` deliberately has no run manifest, so the four tables are read
//! by prefix from a `faba gem-encoder` run:
//!
//! | file | shape | role |
//! |---|---|---|
//! | `{from}.dictionary.parquet` | `[G, K]` | **log β** (mature) — the group profile |
//! | `{from}.latent.parquet` | `[N, K]` | **log θ** per cell |
//! | `{from}.pb_gene.parquet` | `[G, P]` | pseudobulk gene profile |
//! | `{from}.pb_latent.parquet` | `[P, K]` | **log θ** per pseudobulk |
//!
//! Three of the four are on the **log** scale and are exponentiated on the way
//! in. That is not a formatting detail: `SpecificityMode::Simplex` clamps at
//! `max(0.0)`, and log β is negative everywhere, so handing it the file as
//! written would zero the entire profile and leave every gene ranking
//! degenerate — with no error anywhere, just a run that annotates nothing.
//!
//! The two pseudobulk tables are not optional decoration: the correlation-
//! preserving null in [`enrichment::annotate`] recomputes
//! `β̃ = pb_gene · pb_membership[π]` under shuffled pseudobulk labels, which is
//! what lets it destroy the PB↔topic coupling while leaving the gene-gene
//! correlation intact.
//!
//! This is a **single-pass** call. There is no spliced/velocity split here.
//! Projection mode's velocity track pairs `δ_g` against the per-cell velocity
//! increment; enrichment has no such pair to work with. It needs a `(β, θ)`
//! couple — a gene ranking per factor *and* a membership on the simplex to carry
//! the surviving edges to cells — and the velocity is a **displacement**, not a
//! membership. (`{from}.dictionary_nascent.parquet` does exist, but a nascent
//! program is not the velocity, and reading it would just be a second identity
//! call on a noisier track.)

use anyhow::{Context, Result};
use candle_util::decoder::gem_etm::Track;
use enrichment::consensus::{Abstain, UNASSIGNED};
use enrichment::marker_bootstrap::{ClusterBootstrap, EnrichmentBootstrapConfig};
use enrichment::{annotate, AnnotateConfig, AnnotateOutputs, GroupInputs, Mat, SpecificityMode};
use graph_embedding_util::type_annotation::{parse_and_match_markers, write_label_tsvs};
use log::{info, warn};
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::traits::IoOps;
use rustc_hash::FxHashMap;
use std::io::Write;
use std::path::Path;

use super::run::AnnotateArgs;

/// Outputs land at `{out}.enrichment.*`, never at `{out}.{track}.*`, so running
/// both modes at one prefix leaves both sets of results intact.
const TAG: &str = "enrichment";

/// How many mismatched gene ids to name before truncating the error.
const MAX_LISTED: usize = 10;

/////////////////
// entry point //
/////////////////

/// Read the topic model, build the marker matrix on its gene axis, run the
/// bipartite enrichment core, and write `{out}.enrichment.*`.
pub fn run(prefix: &str, out: &str, args: &AnnotateArgs) -> Result<()> {
    let model = load_topic_model(prefix)?;
    let (markers_gc, celltype_names) = build_markers(args, &model.gene_names)?;

    info!(
        "annotate --mode enrichment: dictionary [{} genes x {} factors], cells [{} x {}], \
         pseudobulk [{} genes x {} PB], markers [{} genes x {} types]",
        model.dictionary_gk.nrows(),
        model.dictionary_gk.ncols(),
        model.cell_theta_nk.nrows(),
        model.cell_theta_nk.ncols(),
        model.pb_gene_gp.nrows(),
        model.pb_gene_gp.ncols(),
        markers_gc.nrows(),
        markers_gc.ncols(),
    );

    let config = AnnotateConfig {
        // `Simplex` is the specificity that matches a softmax dictionary: β's
        // columns already sum to 1 over genes, so a gene's evidence for a factor
        // is its share there against its share everywhere else. Same choice
        // `senna annotate-by-enrichment` makes for the same reason.
        specificity: SpecificityMode::Simplex,
        // faba's `--num-perm` is one knob for "how much calibration", and both
        // nulls are calibration. Splitting it into two flags would add a
        // decision nobody has the information to make.
        num_row_randomization: args.num_perm,
        num_sample_perm: args.num_perm,
        // `pb_membership_pk`'s rows ARE the pseudobulk samples, so the
        // permutation shuffles them directly with no inner stratification.
        batch_labels: None,
        fdr_alpha: args.fdr_alpha,
        q_softmax_temperature: args.q_temperature,
        min_confidence: 0.0,
        seed: args.seed,
        min_markers: args.min_markers,
        stratify_null: true,
        // ON by default, exactly as projection mode assembles it: a single pass
        // over one marker panel always returns a winner, and returns it with no
        // error bar. `boot_num_draws` has no faba flag, so it keeps the library
        // default (100 — the moments only need ~10% accuracy on the SD).
        bootstrap: (!args.no_bootstrap_markers && args.n_boot > 0).then_some(
            EnrichmentBootstrapConfig {
                n_boot: args.n_boot,
                abstain: if args.abstain_separable {
                    Abstain::Separable(args.abstain_alpha)
                } else {
                    Abstain::Support(args.min_support)
                },
                set_coverage: args.set_coverage,
                max_set_size: args.max_set_size,
                ..Default::default()
            },
        ),
    };

    let group = GroupInputs {
        profile_gk: model.dictionary_gk,
        pb_gene_gp: model.pb_gene_gp,
        pb_membership_pk: model.pb_theta_pk,
        cell_membership_nk: model.cell_theta_nk,
        gene_names: model.gene_names,
        cell_names: model.cell_names,
    };

    let outputs = annotate(&group, &markers_gc, &celltype_names, &config)?;
    write_outputs(out, &outputs, &model.topic_names, &celltype_names)?;
    Ok(())
}

/////////////////////////
// the four gem tables //
/////////////////////////

/// The `faba gem-encoder` tables enrichment mode annotates, already reconciled
/// onto one gene axis and off the log scale.
struct TopicModel {
    /// `[G, K]` mature dictionary β, already `exp()`-ed off log β — simplex
    /// columns over the gene axis.
    dictionary_gk: Mat,
    /// `[G, P]` pseudobulk gene profile, **rows reordered onto the dictionary's
    /// gene axis** (see [`align_gene_axis`]).
    pb_gene_gp: Mat,
    /// `[P, K]` pseudobulk θ, already `exp()`-ed off log θ.
    pb_theta_pk: Mat,
    /// `[N, K]` cell θ, already `exp()`-ed off log θ.
    cell_theta_nk: Mat,
    /// Gene ids on the dictionary's axis, track suffix stripped.
    gene_names: Vec<Box<str>>,
    cell_names: Vec<Box<str>>,
    /// The dictionary parquet's own column names (`T0..T{K-1}`) — reused so the
    /// factor axis is labelled the same here as it is in the gem-encoder run.
    topic_names: Vec<Box<str>>,
}

fn load_topic_model(prefix: &str) -> Result<TopicModel> {
    let dict_path = required(prefix, "dictionary.parquet")?;
    let latent_path = required(prefix, "latent.parquet")?;
    let pb_gene_path = required(prefix, "pb_gene.parquet")?;
    let pb_latent_path = required(prefix, "pb_latent.parquet")?;

    let dict = DMatrix::<f32>::from_parquet(&dict_path)
        .with_context(|| format!("reading topic dictionary {dict_path}"))?;
    let latent = DMatrix::<f32>::from_parquet(&latent_path)
        .with_context(|| format!("reading cell latent {latent_path}"))?;
    let pb_gene = DMatrix::<f32>::from_parquet(&pb_gene_path)
        .with_context(|| format!("reading pseudobulk gene profile {pb_gene_path}"))?;
    let pb_latent = DMatrix::<f32>::from_parquet(&pb_latent_path)
        .with_context(|| format!("reading pseudobulk latent {pb_latent_path}"))?;

    // The dictionary is keyed by bare gene id, `pb_gene` by gem feature row
    // (`{gene}/count/spliced`) — two different call sites in `gem_encoder::run`.
    // Normalize both, then reorder pb_gene onto the dictionary's axis.
    let gene_names: Vec<Box<str>> = dict.rows.iter().map(|r| strip_track_suffix(r)).collect();
    let pb_gene_keys: Vec<Box<str>> = pb_gene.rows.iter().map(|r| strip_track_suffix(r)).collect();
    let order = align_gene_axis(&gene_names, &pb_gene_keys, &dict_path, &pb_gene_path)?;
    let pb_gene_gp = pb_gene.mat.select_rows(&order);

    anyhow::ensure!(
        latent.mat.ncols() == dict.mat.ncols(),
        "{latent_path} has {} factors but {dict_path} has {} — they are not from the same \
         `faba gem-encoder` run",
        latent.mat.ncols(),
        dict.mat.ncols()
    );
    anyhow::ensure!(
        pb_latent.mat.ncols() == dict.mat.ncols(),
        "{pb_latent_path} has {} factors but {dict_path} has {} — they are not from the same \
         `faba gem-encoder` run",
        pb_latent.mat.ncols(),
        dict.mat.ncols()
    );
    anyhow::ensure!(
        pb_latent.mat.nrows() == pb_gene_gp.ncols(),
        "{pb_latent_path} has {} pseudobulk rows but {pb_gene_path} has {} columns",
        pb_latent.mat.nrows(),
        pb_gene_gp.ncols()
    );

    Ok(TopicModel {
        dictionary_gk: exp_log_beta(&dict.mat, &dict_path),
        pb_gene_gp,
        pb_theta_pk: exp_log_theta(&pb_latent.mat, &pb_latent_path),
        cell_theta_nk: exp_log_theta(&latent.mat, &latent_path),
        gene_names,
        cell_names: latent.rows,
        topic_names: dict.cols,
    })
}

/// Resolve `{prefix}.{suffix}`, failing with the file's provenance rather than a
/// bare "no such file" from the parquet reader.
fn required(prefix: &str, suffix: &str) -> Result<String> {
    let path = format!("{prefix}.{suffix}");
    anyhow::ensure!(
        Path::new(&path).exists(),
        "--mode enrichment needs {path}, which is not there.\n\
         All four of {{prefix}}.dictionary.parquet, .latent.parquet, .pb_gene.parquet and \
         .pb_latent.parquet come from a `faba gem-encoder` (`gem-topic`) run — point \
         `-f/--from` at that run's `-o` prefix.\n\
         A `faba gem` embedding run writes none of them; annotate one of those with \
         `--mode projection`."
    );
    Ok(path)
}

/// Normalize a gene-axis row key to the bare gene id the marker TSV names.
///
/// `{out}.dictionary.parquet` is written with plain gene ids while
/// `{out}.pb_gene.parquet` is written with gem feature rows
/// (`{gene}/count/spliced`) — different call sites, one axis. Stripping the
/// mature suffix wherever it appears puts the dictionary, the pseudobulk profile
/// and the marker panel on the same keys.
fn strip_track_suffix(name: &str) -> Box<str> {
    let suffix = Track::Mature.row_suffix();
    Box::from(name.strip_suffix(suffix).unwrap_or(name))
}

/// Row permutation putting `pb_genes` into `dict_genes` order.
///
/// The two tables are written in the same loop order today, but that is an
/// implementation detail of one function and nothing on disk records it — a
/// positional zip would silently pair gene *i*'s β with gene *j*'s pseudobulk
/// profile the day either write changes, and every enrichment score downstream
/// would still look perfectly well-formed. So match by name and refuse to
/// proceed when the axes disagree.
fn align_gene_axis(
    dict_genes: &[Box<str>],
    pb_genes: &[Box<str>],
    dict_path: &str,
    pb_path: &str,
) -> Result<Vec<usize>> {
    let index: FxHashMap<&str, usize> = pb_genes
        .iter()
        .enumerate()
        .map(|(i, g)| (g.as_ref(), i))
        .collect();

    let mut order = Vec::with_capacity(dict_genes.len());
    let mut missing: Vec<&str> = Vec::new();
    for gene in dict_genes {
        match index.get(gene.as_ref()) {
            Some(&i) => order.push(i),
            None => missing.push(gene.as_ref()),
        }
    }

    anyhow::ensure!(
        missing.is_empty(),
        "{} of {} genes in {dict_path} have no row in {pb_path} (e.g. {}{}). The two tables must \
         come from the same `faba gem-encoder` run — a stale `pb_gene.parquet` left beside a newer \
         dictionary is the usual cause. Re-run `faba gem-encoder` at this prefix.",
        missing.len(),
        dict_genes.len(),
        missing
            .iter()
            .take(MAX_LISTED)
            .copied()
            .collect::<Vec<_>>()
            .join(", "),
        if missing.len() > MAX_LISTED {
            format!(", … and {} more", missing.len() - MAX_LISTED)
        } else {
            String::new()
        }
    );
    Ok(order)
}

/// `exp()` the `[G, K]` dictionary back onto the simplex.
///
/// `{out}.dictionary.parquet` is **log β** — `GemEtmDecoder::get_dictionary` is
/// `log_softmax_g(b_g + ⟨α_t, ρ_g⟩)`, and gem-encoder's own internal consumer
/// calls the same tensor `log_s`. Every entry is therefore ≤ 0.
///
/// Handing that to `SpecificityMode::Simplex` **silently destroys the run**: it
/// computes `β_g,k.max(0.0) / Σ_k' β_g,k'.max(0.0)`, so a wholly negative matrix
/// clamps to zero, every row hits the `s <= 0.0` skip, and every factor's gene
/// ranking becomes an arbitrary tie. Nothing errors — the enrichment scores come
/// back well-formed and mean nothing. Hence the column-sum check below: after
/// `exp()`, each factor's column is a distribution over genes and must sum to 1.
fn exp_log_beta(log_beta: &Mat, path: &str) -> Mat {
    let beta = log_beta.map(f32::exp);
    let k = beta.ncols();
    if k > 0 {
        let mean_sum: f32 = (0..k)
            .map(|j| beta.column(j).iter().sum::<f32>())
            .sum::<f32>()
            / k as f32;
        if !(0.5..=2.0).contains(&mean_sum) {
            warn!(
                "exp() of {path} has mean COLUMN sum {mean_sum:.3}, not ~1 — this file does not \
                 look like log β. `faba gem-encoder` writes log_softmax over genes, so each \
                 factor's column should exponentiate to a distribution. Enrichment scores \
                 computed on it will be well-formed and meaningless; check the run that produced \
                 this prefix."
            );
        }
    }
    beta
}

/// `exp()` a `[·, K]` log-θ table back onto the simplex.
///
/// `{out}.gem.json` states the contract explicitly — `"latent": "log-theta"` —
/// so θ is `exp()`, **not** a softmax.
///
/// The distinction matters precisely because on a row that honours the contract
/// the two agree exactly (`Σ exp(log θ) = 1`, so the softmax denominator is 1).
/// Softmax is therefore never a *fix*; it is a *concealer*. It renormalizes away
/// exactly the discrepancy that tells you the file is not log θ:
///
/// * runs written before that contract stored **raw logits** under the same
///   shape, and nothing but the row sums distinguishes them;
/// * a pseudobulk that observed no cells is left all-zero by
///   `write_pseudobulk_tables`, which `exp()`s to an all-ones row — visibly not
///   a distribution.
///
/// Under `exp()` both show up as a row sum away from 1 and get reported. Under a
/// softmax both come back as a perfectly plausible θ, and the annotation is
/// quietly built on it.
fn exp_log_theta(log_theta: &Mat, path: &str) -> Mat {
    let theta = log_theta.map(f32::exp);
    let rows = theta.nrows();
    if rows > 0 {
        let mean_sum: f32 = (0..rows)
            .map(|i| theta.row(i).iter().sum::<f32>())
            .sum::<f32>()
            / rows as f32;
        if !(0.5..=2.0).contains(&mean_sum) {
            warn!(
                "exp() of {path} has mean row sum {mean_sum:.3}, not ~1 — this file does not look \
                 like log θ. Runs produced before 2026-07-21 stored RAW LOGITS under the same \
                 shape and the two are indistinguishable except here; check that \
                 {{prefix}}.gem.json carries `\"latent\": \"log-theta\"`, and re-run \
                 `faba gem-encoder` if it does not. Annotation continues on exp() as written."
            );
        }
    }
    theta
}

///////////////////
// marker matrix //
///////////////////

/// Build the `[G, C]` marker matrix on the dictionary's gene axis.
///
/// Same reader projection mode uses ([`parse_and_match_markers`]), so `--no-idf`
/// and `--min-panel-coverage` mean the same thing in both modes and a panel that
/// warns in one warns in the other. Its `(gene_index, weight)` lists are simply
/// scattered into the dense matrix `enrichment::annotate` wants; a zero entry is
/// a miss for the KS walk, a positive one is an IDF-weighted hit.
fn build_markers(args: &AnnotateArgs, gene_names: &[Box<str>]) -> Result<(Mat, Vec<Box<str>>)> {
    let (celltype_names, type_markers) = parse_and_match_markers(
        &args.markers,
        gene_names,
        !args.no_idf,
        args.min_panel_coverage,
    )?;
    let mut markers_gc = Mat::zeros(gene_names.len(), celltype_names.len());
    for (ci, markers) in type_markers.iter().enumerate() {
        for &(gi, w) in markers {
            markers_gc[(gi as usize, ci)] = w;
        }
    }
    Ok((markers_gc, celltype_names))
}

/////////////
// outputs //
/////////////

/// Write `{out}.enrichment.*`, mirroring `senna annotate-by-enrichment`'s file
/// set with `cluster` renamed to `topic` — on this path the group axis is the
/// model's factors, not a Leiden partition.
fn write_outputs(
    out: &str,
    outputs: &AnnotateOutputs,
    topic_names: &[Box<str>],
    celltype_names: &[Box<str>],
) -> Result<()> {
    let prefix = format!("{out}.{TAG}");

    let annotation_path = format!("{prefix}.annotation.parquet");
    outputs.cell_annotation_nc.to_parquet_with_names(
        &annotation_path,
        (
            Some(
                &outputs
                    .argmax_labels
                    .iter()
                    .map(|l| Box::from(l.cell_name.as_ref()))
                    .collect::<Vec<Box<str>>>(),
            ),
            Some("cell"),
        ),
        Some(celltype_names),
    )?;
    info!("wrote {annotation_path}");

    // The SHARED per-cell writer (also emits `membership.tsv`), so this mode and
    // projection mode hand downstream tools an identical contract.
    {
        let cells: Vec<Box<str>> = outputs
            .argmax_labels
            .iter()
            .map(|l| Box::from(l.cell_name.as_ref()))
            .collect();
        let labels: Vec<Box<str>> = outputs
            .argmax_labels
            .iter()
            .map(|l| Box::from(l.label.as_ref()))
            .collect();
        let probs: Vec<f32> = outputs.argmax_labels.iter().map(|l| l.confidence).collect();
        write_label_tsvs(&prefix, &cells, &labels, &probs)?;
    }

    let named = |mat: &Mat, suffix: &str| -> Result<()> {
        let path = format!("{prefix}.{suffix}");
        mat.to_parquet_with_names(
            &path,
            (Some(topic_names), Some("topic")),
            Some(celltype_names),
        )?;
        info!("wrote {path}");
        Ok(())
    };
    named(&outputs.q_kc, "topic_celltype_q.parquet")?;
    named(&outputs.es_kc, "topic_celltype_es.parquet")?;
    named(
        &outputs.es_restandardized_kc,
        "topic_celltype_es_std.parquet",
    )?;
    named(&outputs.pvalue_kc, "topic_celltype_p.parquet")?;
    named(&outputs.qvalue_kc, "topic_celltype_q_values.parquet")?;
    // Correlation-preserving sample-permutation z — graded, unlike the pooled
    // count p. Absent when `--num-perm 0`.
    if let Some(pz) = &outputs.perm_z_kc {
        named(pz, "topic_celltype_perm_z.parquet")?;
    }

    if let Some(boot) = &outputs.bootstrap {
        write_bootstrap_outputs(&prefix, boot, topic_names, celltype_names)?;
    }
    Ok(())
}

/// What the marker bootstrap learned, keyed by **topic** — never by cell.
///
/// When the bootstrap runs, its consensus REPLACES the point estimate inside
/// `enrichment::annotate`, so the per-cell `confidence` shipped in `argmax.tsv`
/// is a resampling frequency for the cell's leading topic. Without these two
/// files nothing on disk records what that frequency was built from.
fn write_bootstrap_outputs(
    prefix: &str,
    boot: &ClusterBootstrap,
    topic_names: &[Box<str>],
    celltype_names: &[Box<str>],
) -> Result<()> {
    let k = topic_names.len();
    let c = boot.c;
    let width = c + 1; // the trailing `unassigned` column

    let support_path = format!("{prefix}.topic_celltype_support.parquet");
    let mut support = Mat::zeros(k, width);
    for kk in 0..k {
        for j in 0..width {
            support[(kk, j)] = boot.consensus.post[kk * width + j];
        }
    }
    let mut support_cols: Vec<Box<str>> = celltype_names.to_vec();
    support_cols.push(Box::from(enrichment::UNASSIGNED_LABEL));
    support.to_parquet_with_names(
        &support_path,
        (Some(topic_names), Some("topic")),
        Some(&support_cols),
    )?;
    info!("wrote {support_path}");

    let qc_path = format!("{prefix}.topic_qc.tsv");
    let mut f = std::fs::File::create(&qc_path).with_context(|| format!("creating {qc_path}"))?;
    writeln!(
        f,
        "topic\tconsensus_label\tlabel_set\tsupport\tset_support\tentropy\tdecision_gap\tn_draws"
    )?;
    for (kk, tname) in topic_names.iter().enumerate() {
        // Canonical celltype order, NOT support order: a label's position should
        // not shift between runs because two shares swapped by 0.01.
        let mut set: Vec<usize> = boot.consensus.label_set[kk].clone();
        set.sort_unstable();
        let set_str = if set.is_empty() {
            String::from("-")
        } else {
            set.iter()
                .map(|&t| celltype_names[t].to_string())
                .collect::<Vec<_>>()
                .join("/")
        };
        let label = boot.consensus.label[kk];
        writeln!(
            f,
            "{tname}\t{}\t{set_str}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}",
            if label == UNASSIGNED {
                enrichment::UNASSIGNED_LABEL
            } else {
                &celltype_names[label]
            },
            boot.consensus.support[kk],
            boot.consensus.set_support[kk],
            boot.consensus.entropy[kk],
            boot.decision_gap[kk],
            boot.n_draws,
        )?;
    }
    info!("wrote {qc_path}");

    let called = boot
        .consensus
        .label
        .iter()
        .filter(|&&t| t != UNASSIGNED)
        .count();
    info!(
        "marker bootstrap: {called}/{k} topics called over {} resamples \
         (mean topic_label_support {:.2}); {} celltype(s) unusable",
        boot.n_draws,
        boot.consensus.support.iter().sum::<f32>() / (k.max(1) as f32),
        boot.usable.iter().filter(|&&u| !u).count(),
    );
    Ok(())
}

#[cfg(test)]
#[path = "by_enrichment_tests.rs"]
mod tests;
