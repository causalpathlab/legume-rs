//! Score cells against a frozen `senna bge` gene embedding.
//!
//! Not the counterpart of [`crate::topic::masked_artifact`], despite both being "how you open
//! a model": that module *declares which files must exist* and validates them, because a
//! masked model is a checkpoint plus six parquets that several writers can drift apart. A bge
//! run has no checkpoint at all (`has_model: false`), so there is nothing to declare — the
//! whole model on the gene side is `(ρ, b_feat)`, and the only real work is finding them and
//! then scoring against them. Hence a scorer, in the bge module, rather than an artifact
//! declaration at the crate root.
//!
//! **ρ lives in more than one place, and only one resolver knows the rules.**
//! [`crate::run_manifest::resolve_feature_loading_for`] exists because three consumers each
//! probed for ρ independently and each broke differently. Go through it. Note it returns
//! `(ρ_path, bias_path)` and `deconvolve` discards the second — a probe needs both, since
//! `(ρ, b_feat)` is exactly the frozen side that [`project_cells`] and
//! [`graph_embedding_util::posterior::multinomial_ll`] consume.
//!
//! `--skip-etm` is **not** required. It used to be, before `feature_loading.parquet` always
//! carried raw signed ρ; it now does on both paths, which is why `deconvolve` documents
//! itself as working with or without the flag.

use crate::run_manifest::{self, ArtifactScale, RunKind, RunManifest};
use crate::topic::eval::{build_gene_remap_with, QueryNameOpts};
use anyhow::Context;
use auxiliary_data::data_loading::{read_data_on_shared_rows, ReadSharedRowsArgs};
use data_beans::sparse_io_vector::SparseIoVec;
use graph_embedding_util::cell_projection::project_cells;
// `multinomial_ll` is not in `posterior`'s re-export list (only `poisson_ll` is), so it comes
// from the module directly. It is the one we want — see `BgeModel::score`.
use graph_embedding_util::posterior::lnpdf::multinomial_ll;
use graph_embedding_util::posterior::{FrozenSide, NodeTerm};
use log::info;
use matrix_util::traits::IoOps;
use nalgebra::DMatrix;
use std::path::Path;

/// Ridge on the per-cell Poisson MAP solve.
///
/// A projection, not a fit: every cell is solved independently against a frozen ρ, so the
/// ridge only keeps the solve conditioned when a cell has few counts. Mild is right —
/// heavier shrinkage would pull sparse cells toward the origin and make them look like
/// poor fits for a reason that has nothing to do with the model.
const PROJECTION_RIDGE: f64 = 1.0;

/// An opened `senna bge` model: the frozen feature side, and the gene axis it lives on.
pub struct BgeEmbedding {
    /// ρ as **row-major** `[D, H]`. `project_cells` and `FrozenSide` both want that layout;
    /// nalgebra stores column-major, so this is transposed once at load rather than per cell.
    pub rho: Vec<f32>,
    pub b_feat: Vec<f32>,
    pub gene_names: Vec<Box<str>>,
    pub h: usize,
}

pub struct BgeFit {
    pub data_vec: SparseIoVec,
    pub llik: Vec<f32>,
    pub total: Vec<f32>,
}

impl BgeEmbedding {
    /// Open from a run prefix or a `{run}.senna.json` path.
    pub fn open(from: &str) -> anyhow::Result<Self> {
        let (manifest, dir) = RunManifest::load(Path::new(from))?;
        anyhow::ensure!(
            manifest.kind == RunKind::Bge,
            "{from} is a '{}' run; this reader is for `senna bge` output",
            manifest.kind
        );

        let (rho_path, bias_path) = run_manifest::resolve_feature_loading_for(&manifest, &dir)?;
        let rho = DMatrix::<f32>::from_parquet(&rho_path)
            .with_context(|| format!("reading per-gene loading ρ {rho_path}"))?;
        // Catches a manifest that points at a log-simplex β instead of signed ρ — the exact
        // confusion `--skip-etm` used to create.
        ArtifactScale::ensure(&rho.mat, ArtifactScale::Signed, &rho_path)?;

        let bias_path = bias_path.ok_or_else(|| {
            anyhow::anyhow!(
                "{from}: found ρ at {rho_path} but no matching feature_bias.parquet. The \
                 per-gene bias is half of the frozen side — scoring without it would silently \
                 use b_feat = 0 and misrank every cell."
            )
        })?;
        let bias = DMatrix::<f32>::from_parquet(&bias_path)
            .with_context(|| format!("reading per-gene bias {bias_path}"))?;
        anyhow::ensure!(
            bias.rows == rho.rows,
            "gene axes disagree: ρ has {} genes, bias has {}",
            rho.rows.len(),
            bias.rows.len()
        );

        let (d, h) = (rho.mat.nrows(), rho.mat.ncols());
        let mut rho_rm = Vec::with_capacity(d * h);
        for i in 0..d {
            for k in 0..h {
                rho_rm.push(rho.mat[(i, k)]);
            }
        }
        info!("bge model: {d} genes, H={h} (ρ {rho_path})");

        Ok(Self {
            rho: rho_rm,
            b_feat: bias.mat.column(0).iter().copied().collect(),
            gene_names: rho.rows,
            h,
        })
    }

    /// Per-cell predictive fit of `files` under this frozen embedding.
    ///
    /// Two steps per cell: project it onto the frozen side (Poisson MAP, `project_cells`),
    /// then score it there. The score is [`multinomial_ll`], **not** `poisson_ll`, because it
    /// profiles the per-cell intercept `b_a` out analytically — so it is depth-invariant *by
    /// construction*, which is what the topic paths approximate by hand with `llik / total`.
    /// It is also the estimand bge's own phase-1 trains under.
    pub fn score(
        &self,
        files: &[Box<str>],
        preload: bool,
        block: usize,
    ) -> anyhow::Result<BgeFit> {
        let loaded = read_data_on_shared_rows(ReadSharedRowsArgs {
            data_files: files.to_vec(),
            preload,
            ..Default::default()
        })?;
        let data_vec = loaded.data;
        info!(
            "Query data: {} features × {} cells",
            data_vec.num_rows(),
            data_vec.num_columns()
        );

        let new_genes = data_vec.row_names()?;
        let qopts = QueryNameOpts {
            kind: crate::masked_topic::FeatureNameKindArg::Exact.resolve_or_gene(),
            suffix_delim: None,
            keep_suffix: None,
        };
        let remap = build_gene_remap_with(&self.gene_names, &new_genes, &qopts);
        let n_model = self.gene_names.len();
        anyhow::ensure!(
            remap.n_mapped * 10 >= n_model,
            "too few genes overlap: {}/{n_model} mapped",
            remap.n_mapped
        );

        // The exact normalizer: every model gene, so `partition_scale = 1`.
        let partition: Vec<u32> = (0..n_model as u32).collect();
        let side = FrozenSide {
            e: &self.rho,
            b: &self.b_feat,
            h: self.h,
        };

        let ntot = data_vec.num_columns();
        let mut llik = Vec::with_capacity(ntot);
        let mut total = Vec::with_capacity(ntot);
        // Blocked because `project_cells` takes every cell at once; unblocked, a large query
        // would hold its whole sparse matrix in `per_cell` on top of the backend.
        for lb in (0..ntot).step_by(block) {
            let ub = (lb + block).min(ntot);
            let csc = data_vec.read_columns_csc(lb..ub)?;
            let per_cell: Vec<Vec<(u32, f32)>> = (0..csc.ncols())
                .map(|j| {
                    csc.col(j)
                        .row_indices()
                        .iter()
                        .zip(csc.col(j).values())
                        .filter_map(|(&i, &v)| remap.new_to_train[i].map(|t| (t as u32, v)))
                        .collect()
                })
                .collect();

            let (e_cell, _b_cell) = project_cells(
                &self.rho,
                &self.b_feat,
                &per_cell,
                self.h,
                PROJECTION_RIDGE,
                None,
            );
            for (c, pos) in per_cell.iter().enumerate() {
                let node = NodeTerm::new(pos, &partition, 1.0);
                llik.push(multinomial_ll(
                    &e_cell[c * self.h..(c + 1) * self.h],
                    &node,
                    &side,
                ));
                total.push(pos.iter().map(|&(_, v)| v).sum::<f32>());
            }
        }

        Ok(BgeFit {
            data_vec,
            llik,
            total,
        })
    }
}
