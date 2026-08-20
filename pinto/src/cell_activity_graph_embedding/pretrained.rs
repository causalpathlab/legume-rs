//! Pre-trained gene-embedding ingestion for `cage`.
//!
//! Aligns an external `gene x H` dictionary (a raw feature embedding written
//! by another training run) to this run's gene axis. Matched genes take their
//! dictionary row verbatim; a gene with no dictionary row is seeded from the
//! matched gene whose count profile it resembles most, so it starts near a
//! plausible relative rather than at noise, and stays trainable.
//!
//! The heavy lifting — parquet read, per-side name canonicalization, bias
//! pairing, target-order alignment — is
//! [`auxiliary_data::frozen_features::load_frozen_feature_host`]. This module
//! adds what `cage` needs on top: rejection of co-embed artifacts, expansion
//! from the matched subset back to the full gene axis, profile-neighbor
//! seeding, and an auditable per-gene record of where every row came from.

use crate::util::common::Mat;
use auxiliary_data::feature_names::FeatureNameKind;
use auxiliary_data::feature_rows::parse_feature_row;
use auxiliary_data::frozen_features::{load_frozen_feature_host, FrozenLoadArgs};
use candle_util::candle_core::{Tensor, Var};
use log::{info, warn};
use rayon::prelude::*;

/// Where a gene's initial embedding row came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitKind {
    /// The dictionary had a row for this gene; taken verbatim.
    Matched,
    /// No dictionary row; seeded from the closest matched gene's row
    /// (or from the matched-row mean when the gene's profile is all zero,
    /// in which case `neighbor_gene` is `None`).
    Neighbor,
}

impl InitKind {
    pub fn label(self) -> &'static str {
        match self {
            InitKind::Matched => "matched",
            InitKind::Neighbor => "neighbor",
        }
    }
}

/// One row of the `{out}.gene_embedding_init.parquet` audit table.
pub struct InitRecord {
    pub gene: Box<str>,
    pub init: InitKind,
    /// The matched gene an unmatched one was seeded from. `None` for matched
    /// genes and for mean-seeded genes.
    pub neighbor_gene: Option<Box<str>>,
    /// Profile cosine to `neighbor_gene`; `NaN` where no neighbor was used.
    pub cosine: f32,
}

/// The aligned, fully populated gene side, `[n_genes x h]`, rows in the run's
/// gene order. `records` is parallel to the gene axis and is the single
/// source of truth for which rows came from the dictionary.
pub struct PretrainedGeneEmbedding {
    pub e_gene: Mat,
    pub b_gene: Vec<f32>,
    pub records: Vec<InitRecord>,
}

impl PretrainedGeneEmbedding {
    /// The dictionary's embedding width.
    pub fn h(&self) -> usize {
        self.e_gene.ncols()
    }

    /// `1.0` where the row came from the dictionary, `0.0` where it was
    /// seeded — the freeze mask, derived from `records` so the two can
    /// never disagree.
    pub fn frozen_row_mask(&self) -> Vec<f32> {
        self.records
            .iter()
            .map(|r| {
                if r.init == InitKind::Matched {
                    1.0
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// How many rows came from the dictionary.
    pub fn n_matched(&self) -> usize {
        self.records
            .iter()
            .filter(|r| r.init == InitKind::Matched)
            .count()
    }
}

pub struct PretrainedArgs<'a> {
    /// Path to the `gene x H` dictionary parquet; row column 0 is the gene name.
    pub dictionary_path: &'a str,
    /// Optional `[D, 1]` per-gene bias parquet; zeros when absent.
    pub bias_path: Option<&'a str>,
    /// The run's gene axis, already final.
    pub gene_names: &'a [Box<str>],
    /// Canonicalization applied to both sides before matching.
    pub name_kind: FeatureNameKind,
    /// Produces `[n_genes x P]` per-gene count profiles (any pooling; only
    /// row directions matter). Called at most once, and only when some gene
    /// has no dictionary row — an all-matched dictionary never pays for it.
    pub gene_profiles: &'a dyn Fn() -> anyhow::Result<Mat>,
}

/// Load, align, and fill. See the module doc for the contract; every path
/// through this function leaves `e_gene` fully populated and `records`
/// parallel to `gene_names`.
pub fn load_pretrained_gene_embedding(
    args: PretrainedArgs<'_>,
) -> anyhow::Result<PretrainedGeneEmbedding> {
    let n_genes = args.gene_names.len();
    anyhow::ensure!(n_genes > 0, "empty gene axis");

    // A row name in the channelized `{gene}/{modality}/...` grammar means a
    // channelized or co-embed artifact, which is not a dictionary. Catch it
    // by name — a names-only column read, not a full matrix decode — before
    // alignment would quietly match nothing. The grammar itself is
    // single-sourced in `auxiliary_data::feature_rows`.
    let dict_names = matrix_util::parquet::read_parquet_string_column(args.dictionary_path, 0)?;
    let offending: Vec<&str> = dict_names
        .iter()
        .filter(|r| parse_feature_row(r).is_some())
        .map(|r| r.as_ref())
        .take(3)
        .collect();
    anyhow::ensure!(
        offending.is_empty(),
        "{} does not look like a gene x H dictionary: row names carry the \
         channelized row grammar (e.g. {}). Point --gene-embedding at a raw \
         feature embedding (a topic model's feature_embedding.parquet or an \
         embedding run's feature_loading.parquet), not at a co-embedding \
         output.",
        args.dictionary_path,
        offending.join(", ")
    );

    let host = load_frozen_feature_host(FrozenLoadArgs {
        dictionary_path: args.dictionary_path,
        bias_path: args.bias_path,
        target_feature_names: args.gene_names,
        name_kind: args.name_kind,
    })?;
    let h = host.h;
    let n_matched = host.keep_target_indices.len();
    anyhow::ensure!(n_matched > 0, "no gene of this run matched the dictionary");

    // Expand the matched subset back onto the full axis.
    let mut e_gene = Mat::zeros(n_genes, h);
    let mut b_gene = vec![0.0f32; n_genes];
    let mut matched = vec![false; n_genes];
    for (k, &g) in host.keep_target_indices.iter().enumerate() {
        e_gene.row_mut(g).copy_from(&host.e_feat.row(k));
        b_gene[g] = host.b_feat[k];
        matched[g] = true;
    }
    let matched_idx: Vec<usize> = (0..n_genes).filter(|&g| matched[g]).collect();
    let unmatched_idx: Vec<usize> = (0..n_genes).filter(|&g| !matched[g]).collect();

    // Closest matched gene by profile cosine, per unmatched gene. The
    // profiles (a full pass over the data at the caller) are built only when
    // this branch is reached at all.
    let neighbor_of: Vec<Option<(usize, f32)>> = if unmatched_idx.is_empty() {
        Vec::new()
    } else {
        let prof = (args.gene_profiles)()?;
        anyhow::ensure!(
            prof.nrows() == n_genes,
            "gene_profiles rows ({}) != gene axis ({})",
            prof.nrows(),
            n_genes
        );
        let norm = |g: usize| -> f32 { prof.row(g).iter().map(|v| v * v).sum::<f32>().sqrt() };
        // Matched norms once, not once per unmatched gene: recomputing them
        // inside the search doubles its arithmetic.
        let matched_norms: Vec<f32> = matched_idx.iter().map(|&m| norm(m)).collect();
        unmatched_idx
            .par_iter()
            .map(|&g| {
                let ng = norm(g);
                if ng == 0.0 {
                    return None;
                }
                matched_idx
                    .iter()
                    .zip(matched_norms.iter())
                    .filter_map(|(&m, &nm)| {
                        if nm == 0.0 {
                            return None;
                        }
                        let dot: f32 = prof
                            .row(g)
                            .iter()
                            .zip(prof.row(m).iter())
                            .map(|(a, b)| a * b)
                            .sum();
                        Some((m, dot / (ng * nm)))
                    })
                    .max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)))
            })
            .collect()
    };

    // Mean of the matched rows, the fallback seed for a gene with no usable
    // profile. `host.e_feat` is exactly the matched rows, so its column means
    // are the answer.
    let mean_row = host.e_feat.row_mean();

    // Fill the seeded rows and build every record where its case is decided,
    // in one pass over the gene axis.
    let mut neighbor_at = vec![None; n_genes];
    for (&g, nb) in unmatched_idx.iter().zip(neighbor_of.iter()) {
        neighbor_at[g] = Some(*nb);
    }
    let mut mean_seeded = 0usize;
    let records: Vec<InitRecord> = args
        .gene_names
        .iter()
        .enumerate()
        .map(|(g, gene)| match neighbor_at[g] {
            None => InitRecord {
                gene: gene.clone(),
                init: InitKind::Matched,
                neighbor_gene: None,
                cosine: f32::NAN,
            },
            Some(Some((m, cos))) => {
                let src = e_gene.row(m).into_owned();
                e_gene.row_mut(g).copy_from(&src);
                InitRecord {
                    gene: gene.clone(),
                    init: InitKind::Neighbor,
                    neighbor_gene: Some(args.gene_names[m].clone()),
                    cosine: cos,
                }
            }
            Some(None) => {
                e_gene.row_mut(g).copy_from(&mean_row);
                mean_seeded += 1;
                InitRecord {
                    gene: gene.clone(),
                    init: InitKind::Neighbor,
                    neighbor_gene: None,
                    cosine: f32::NAN,
                }
            }
        })
        .collect();

    let unused = dict_names.len().saturating_sub(n_matched);
    info!(
        "Pre-trained gene embedding: {} matched, {} neighbor-seeded ({} of those from the matched mean), {} dictionary rows unused",
        n_matched,
        unmatched_idx.len(),
        mean_seeded,
        unused
    );
    if n_matched < n_genes / 2 {
        warn!(
            "fewer than half the genes matched the dictionary ({n_matched}/{n_genes}); \
             check --gene-name-mode if this is unexpected"
        );
    }

    Ok(PretrainedGeneEmbedding {
        e_gene,
        b_gene,
        records,
    })
}

/// Write the audit table: one row per gene, in gene-axis order.
pub fn write_init_report(out_prefix: &str, records: &[InitRecord]) -> anyhow::Result<()> {
    let genes: Vec<Box<str>> = records.iter().map(|r| r.gene.clone()).collect();
    let init: Vec<Box<str>> = records.iter().map(|r| r.init.label().into()).collect();
    let neighbor: Vec<Box<str>> = records
        .iter()
        .map(|r| r.neighbor_gene.clone().unwrap_or_else(|| "".into()))
        .collect();
    let cosine: Vec<f32> = records.iter().map(|r| r.cosine).collect();

    matrix_util::parquet::write_named_table(
        &format!("{out_prefix}.gene_embedding_init.parquet"),
        "gene",
        &genes,
        &[
            ("init".into(), matrix_util::parquet::Column::Str(&init)),
            (
                "neighbor_gene".into(),
                matrix_util::parquet::Column::Str(&neighbor),
            ),
            ("cosine".into(), matrix_util::parquet::Column::F32(&cosine)),
        ],
    )
}

/// The freeze state for a run whose dictionary rows must not move: the fixed
/// copy, the frozen-row mask, and the registered Var the restore writes
/// through. Bundled so the two same-typed tensors cannot be swapped at a call
/// site.
pub struct FrozenGene {
    fixed: Tensor,
    keep_mask: Tensor,
    var: Var,
}

impl FrozenGene {
    pub fn new(fixed: Tensor, keep_mask: Tensor, var: Var) -> Self {
        Self {
            fixed,
            keep_mask,
            var,
        }
    }

    /// Put the frozen rows back after an optimizer step. See
    /// [`candle_util::frozen_features::restore_frozen_rows`] for why a
    /// post-step restore rather than a gradient mask.
    pub fn restore(&self) -> anyhow::Result<()> {
        candle_util::frozen_features::restore_frozen_rows(&self.var, &self.fixed, &self.keep_mask)?;
        Ok(())
    }
}
