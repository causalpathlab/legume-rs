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
//! `(ρ, b_feat)` is exactly the frozen side that [`FrozenProjector`] and
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
use graph_embedding_util::fit::{FrozenProjectionArgs, FrozenProjector, PROJECTION_RIDGE_SGD};
// `multinomial_ll` is not in `posterior`'s re-export list (only `poisson_ll` is), so it comes
// from the module directly. It is the one we want — see `BgeModel::score`.
use crate::embed_common::Mat;
use crate::logging::new_progress_bar;
use candle_util::candle_core::Device;
use graph_embedding_util::posterior::lnpdf::multinomial_ll;
use graph_embedding_util::posterior::{FrozenSide, NodeTerm};
use log::info;
use matrix_util::traits::IoOps;
use nalgebra::DMatrix;
use rayon::prelude::*;
use std::path::Path;

/// An opened `senna bge` model: the frozen feature side, and the gene axis it lives on.
pub struct BgeEmbedding {
    /// ρ as **row-major** `[D, H]`. The SGD projection and `FrozenSide` both want that layout;
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
    /// `[N, H]` per-cell latent — the Poisson-MAP projection onto the frozen ρ.
    ///
    /// The projection has always produced this and the scorer used to drop it on the
    /// floor, so a bge model could score new cells but not *place* them: every caller
    /// wanting held-out coordinates had to re-run the whole projection itself. It is the
    /// same object `{prefix}.cell_embedding.parquet` holds for the training cells.
    pub latent: Mat,
}

impl BgeEmbedding {
    /// Open from a run prefix or a `{run}.senna.json` path.
    ///
    /// Both, because the two callers disagree: `probe --model` is documented as a run
    /// *prefix* (every other family resolves `{prefix}.model.json` from one), while
    /// `deconvolve --from` is handed the manifest path itself. `RunManifest::load` reads a
    /// file, so a prefix alone used to fail with a bare "No such file or directory" naming a
    /// path the user never typed.
    pub fn open(from: &str) -> anyhow::Result<Self> {
        let direct = Path::new(from);
        let suffixed = std::path::PathBuf::from(format!("{from}.senna.json"));
        let manifest_path = if direct.is_file() {
            direct.to_path_buf()
        } else if suffixed.is_file() {
            suffixed
        } else {
            anyhow::bail!(
                "{from}: no run manifest here. Looked for `{from}` and `{from}.senna.json` — \
                 `senna bge -o {from}` writes the second."
            );
        };
        let (manifest, dir) = RunManifest::load(&manifest_path)?;
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
        // nalgebra stores column-major, so the transpose's buffer *is* row-major of the
        // original — no hand-rolled loop, and no second place stating the layout invariant.
        let rho_rm = rho.mat.transpose().as_slice().to_vec();
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
    /// Two steps per cell: project it onto the frozen side (block Poisson SGD),
    /// then score it there. The score is [`multinomial_ll`], **not** `poisson_ll`, because it
    /// profiles the per-cell intercept `b_a` out analytically — so it is depth-invariant *by
    /// construction*, which is what the topic paths approximate by hand with `llik / total`.
    /// It is also the estimand bge's own phase-1 trains under.
    ///
    /// ⚠️ **This is a much weaker novelty detector than the topic families, structurally.** A
    /// cell here is fitted with a *free* `H`-dimensional vector (H = 128 by default); in a
    /// topic model it is confined to the `K`-simplex over fixed topics — 4 free parameters at
    /// K = 5. With that much per-cell freedom a genuinely novel cell simply finds somewhere in
    /// ℝ^H to sit and reconstructs well. Measured on a held-out-topic batch that `topic` and
    /// `vae` both flagged at 100%, bge flagged 2.4% (p = 0.97) while calibrating correctly
    /// against itself at 5.1%. So a COVERED verdict from bge means "the embedding can represent
    /// these cells", **not** "the model has seen this biology" — read it as a floor, and prefer
    /// a topic-family probe when the question is novelty.
    pub fn score(
        &self,
        files: &[Box<str>],
        preload: bool,
        block: usize,
        qopts: &QueryNameOpts,
        dev: &Device,
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
        let mut remap = build_gene_remap_with(&self.gene_names, &new_genes, qopts);
        let n_model = self.gene_names.len();
        // The caller supplies the floor. `probe` passes 0 — a thin panel is exactly what
        // it exists to score — while `predict` passes `--min-gene-overlap`, which used to
        // be dropped here along with every `--feature-name-*` flag.
        crate::topic::eval::ensure_gene_coverage(&remap, qopts.min_overlap, "--feature-name-kind")?;
        // After the coverage gate, not before: the hidden genes are deliberately
        // withheld, so counting them as missing coverage would refuse every
        // ablated run. Driven from `qopts` rather than a separate parameter, so
        // this module no longer reaches back into the `predict` subcommand.
        if let Some(hide) = qopts.hide.as_deref() {
            crate::topic::eval::hide_features(&mut remap, &new_genes, hide)?;
        }

        // The exact normalizer: every model gene, so `partition_scale = 1`.
        let partition: Vec<u32> = (0..n_model as u32).collect();
        let side = FrozenSide {
            e: &self.rho,
            b: &self.b_feat,
            h: self.h,
        };

        // The whole per-dictionary setup — the transposed design, the live-feature scan,
        // the null normalizer, the learning rate — happens once, here, instead of on
        // every projection call.
        let projector = FrozenProjector::new(&FrozenProjectionArgs {
            feat: &self.rho,
            b_feat: &self.b_feat,
            h: self.h,
            lambda: f64::from(PROJECTION_RIDGE_SGD),
            dev,
        })?;

        let ntot = data_vec.num_columns();
        let mut llik = Vec::with_capacity(ntot);
        let mut total = Vec::with_capacity(ntot);
        let mut latent = Mat::zeros(ntot, self.h);
        // ONE bar for the query. The projector advances it as its blocks step, so this
        // side never increments it: nesting a second bar under it (one per projection
        // call) is what the group loop used to do.
        let bar = new_progress_bar(ntot as u64).with_message("scoring");

        // Columns are READ in `block`-sized slabs — that bound is the reader's — but they
        // are PROJECTED in groups the engine sizes.
        //
        // The two are not the same thing and used to be conflated. `block` is
        // `--minibatch-size`, default 500; the projection engine sizes its internal blocks
        // from an activation budget of its own and takes thousands of cells at a time.
        // Handing it 500 gave it exactly one under-sized block per call, so its per-step
        // matmul ran on a short `M`. The fix used to be a byte budget guessed here, in
        // nonzeros — a currency the engine never sees, aimed at a block size this crate
        // cannot read. [`FrozenProjector::group_nodes`] is that number in the engine's own
        // terms, and cutting the group EXACTLY there is what keeps every block full.
        let group_nodes = projector.group_nodes();
        let mut group = EdgeGroup::default();

        let mut lb = 0usize;
        while lb < ntot {
            let group_lb = lb;
            // Accumulate slabs until the group is full. The inner loop's exit condition
            // IS the flush condition, so a part-group at the end needs no special case.
            while lb < ntot && group.len() < group_nodes {
                let ub = (lb + block).min(group_lb + group_nodes).min(ntot);
                let csc = data_vec.read_columns_csc(lb..ub)?;
                // One walk per cell yields both the remapped entries and the cell's
                // total; the total used to be a third pass over the same nonzeros.
                // Parallel per cell, matching the `multinomial_ll` pass below — the
                // walks are independent, and an indexed range keeps them in cell order.
                let (per_cell, totals): (Vec<Vec<(u32, f32)>>, Vec<f32>) = (0..csc.ncols())
                    .into_par_iter()
                    .map(|j| {
                        let col = csc.col(j);
                        let pos: Vec<(u32, f32)> = col
                            .row_indices()
                            .iter()
                            .zip(col.values())
                            .filter_map(|(&i, &v)| remap.new_to_train[i].map(|t| (t as u32, v)))
                            .collect();
                        let tot = pos.iter().map(|&(_, v)| v).sum::<f32>();
                        (pos, tot)
                    })
                    .unzip();
                drop(csc);

                group.extend(&per_cell);
                total.extend(totals);
                lb = ub;
            }

            let n_group = group.len();
            let e_cell = project_group(&projector, &group, &bar)?;
            // `multinomial_ll` walks the whole gene axis twice per cell for the
            // normalizer, which makes it the dominant cost here — and it sat serial
            // directly below `project_cells`, which is already rayon-parallel. Everything
            // it reads is shared and immutable, so the map parallelizes as-is and stays in
            // order. `NodeTerm` borrows an interleaved slice, so the split arrays are
            // re-zipped into a per-cell temporary — a few thousand entries against a
            // normalizer that touches the whole gene axis twice.
            llik.par_extend((0..n_group).into_par_iter().map(|c| {
                let (feat, count) = group.cell(c);
                let pos: Vec<(u32, f32)> =
                    feat.iter().copied().zip(count.iter().copied()).collect();
                let node = NodeTerm::new(&pos, &partition, 1.0);
                multinomial_ll(&e_cell[c * self.h..(c + 1) * self.h], &node, &side)
            }));
            // `e_cell` is row-major `[n_group, H]`; `Mat` is column-major, so this is a
            // transposing copy rather than a memcpy.
            for c in 0..n_group {
                for j in 0..self.h {
                    latent[(group_lb + c, j)] = e_cell[c * self.h + j];
                }
            }
            group.clear();
        }
        bar.finish_and_clear();

        Ok(BgeFit {
            data_vec,
            llik,
            total,
            latent,
        })
    }
}

/// Project one group of cells onto the frozen side, returning `θ` row-major `[n, H]`.
///
/// The SAME block SGD the training run used for its own phase 2. The Newton/IRLS
/// alternative fits a cell's observed features only and lets a ridge stand in for the
/// log-partition, so it lands in a different space — and a train/test comparison whose
/// two halves were projected differently measures the estimator gap as much as the model.
fn project_group(
    projector: &FrozenProjector,
    group: &EdgeGroup,
    bar: &indicatif::ProgressBar,
) -> anyhow::Result<Vec<f32>> {
    let nodes: Vec<(u32, &[u32], &[f32])> = (0..group.len())
        .map(|j| {
            let (feat, count) = group.cell(j);
            (j as u32, feat, count)
        })
        .collect();
    Ok(projector.project(&nodes, group.len(), bar)?.theta)
}

/// One projection group's remapped edges, flat and split.
///
/// Flat because the SGD entry point wants `(&[u32], &[f32])` per cell: held as a
/// `Vec` per cell it would be two allocations per cell and a whole second copy of the
/// group's nonzeros, which mattered little when a "group" was one `--minibatch-size`
/// slab and does not when it is tens of thousands of cells.
#[derive(Default)]
struct EdgeGroup {
    /// `offsets[i]..offsets[i + 1]` bounds cell `i`; always starts with one `0`.
    offsets: Vec<usize>,
    feat: Vec<u32>,
    count: Vec<f32>,
}

impl EdgeGroup {
    fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    fn cell(&self, i: usize) -> (&[u32], &[f32]) {
        let (s, e) = (self.offsets[i], self.offsets[i + 1]);
        (&self.feat[s..e], &self.count[s..e])
    }

    fn extend(&mut self, per_cell: &[Vec<(u32, f32)>]) {
        if self.offsets.is_empty() {
            self.offsets.push(0);
        }
        for pos in per_cell {
            self.feat.extend(pos.iter().map(|&(f, _)| f));
            self.count.extend(pos.iter().map(|&(_, c)| c));
            self.offsets.push(self.feat.len());
        }
    }

    fn clear(&mut self) {
        self.offsets.clear();
        self.feat.clear();
        self.count.clear();
    }
}
