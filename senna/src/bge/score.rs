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
    /// The learned-module tables `(π [D × M], μ [M × H])` when the run trained them —
    /// what lets an unseen gene be initialized through its co-expression rather than
    /// dropped. `None` for a plain run, where an unseen gene falls back to its
    /// neighbours' row average.
    pub modules: Option<(DMatrix<f32>, DMatrix<f32>)>,
}

/// How `predict` treats the new data's genes the model never saw.
#[derive(Clone, Copy, Debug)]
pub struct InitOpts {
    /// `false` drops unseen genes, the pre-alignment behaviour.
    pub enabled: bool,
    /// Neighbours whose membership is averaged for an unseen gene.
    pub k: usize,
    /// Below this best cosine similarity an unseen gene takes the diffuse prior.
    pub similarity_floor: f32,
    /// Re-project every cell with the initialized genes as observations (pass 2).
    /// Off, they are scored from the pass-1 latent and never move it.
    pub in_fit: bool,
}

impl InitOpts {
    pub const OFF: Self = Self {
        enabled: false,
        k: 10,
        similarity_floor: 0.2,
        in_fit: false,
    };
}

/// What the alignment produced, for the writers.
pub struct InitOutcome {
    pub alignment: graph_embedding_util::transfer::GeneAlignment,
    /// Row names of the NEW data (the alignment's `union_to_new` indexes these).
    pub new_gene_names: Vec<Box<str>>,
    /// New-data rows that were initialized, in alignment order.
    pub unseen_rows: Vec<usize>,
    /// Per-cell score on the initialized genes.
    pub scores: Vec<super::transfer::InitScore>,
    /// Pseudobulks the profiles were formed over.
    pub n_clusters: usize,
    /// Whether pass 2 ran, i.e. whether `latent` saw the initialized genes.
    pub in_fit: bool,
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
    /// `[N]` per-cell intercept fitted alongside the latent.
    pub b_cell: Vec<f32>,
    /// The gene-axis alignment, when the query carried genes the model never saw
    /// and initialization was on.
    pub init: Option<InitOutcome>,
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

        // Module tables, when the run trained them. Rows must be the dictionary's genes
        // in the dictionary's order: the alignment indexes π by training row.
        let modules = match (
            manifest.outputs.module_membership.as_deref(),
            manifest.outputs.module_dictionary.as_deref(),
        ) {
            (Some(pi_rel), Some(mu_rel)) => {
                let pi_path = run_manifest::resolve(&dir, pi_rel)
                    .to_string_lossy()
                    .into_owned();
                let mu_path = run_manifest::resolve(&dir, mu_rel)
                    .to_string_lossy()
                    .into_owned();
                let pi = DMatrix::<f32>::from_parquet(&pi_path)
                    .with_context(|| format!("reading module membership {pi_path}"))?;
                let mu = DMatrix::<f32>::from_parquet(&mu_path)
                    .with_context(|| format!("reading module dictionary {mu_path}"))?;
                anyhow::ensure!(
                    pi.rows == rho.rows,
                    "module membership genes disagree with the dictionary ({} vs {} rows)",
                    pi.rows.len(),
                    rho.rows.len()
                );
                anyhow::ensure!(
                    mu.mat.ncols() == h && mu.mat.nrows() == pi.mat.ncols(),
                    "module dictionary is {}×{} but π has {} modules and H={h}",
                    mu.mat.nrows(),
                    mu.mat.ncols(),
                    pi.mat.ncols()
                );
                info!("bge model carries {} learned gene modules", pi.mat.ncols());
                Some((pi.mat, mu.mat))
            }
            _ => None,
        };

        Ok(Self {
            rho: rho_rm,
            b_feat: bias.mat.column(0).iter().copied().collect(),
            gene_names: rho.rows,
            h,
            modules,
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
        self.score_with_init(files, preload, block, qopts, InitOpts::OFF, dev)
    }

    /// [`Self::score`] with the gene-axis alignment: genes the model never saw are
    /// initialized through the modules (see `super::transfer`) instead of dropped.
    /// With `init.enabled == false`, or a query with no unseen genes, this is
    /// byte-identical to [`Self::score`].
    pub fn score_with_init(
        &self,
        files: &[Box<str>],
        preload: bool,
        block: usize,
        qopts: &QueryNameOpts,
        init: InitOpts,
        dev: &Device,
    ) -> anyhow::Result<BgeFit> {
        use super::transfer::{profiles_by_cluster, score_initialized, union_remap, unseen_rows};
        use graph_embedding_util::transfer::{
            align_gene_axis, moment_matched_bias, AlignInputs, GeneStatus, ModuleTables,
        };

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
        // The remap BEFORE the hide pass: the alignment must see a withheld model gene as
        // matched (its row is kept, its counts are not), never as unseen.
        let before_hide = remap.new_to_train.clone();
        // After the coverage gate, not before: the hidden genes are deliberately
        // withheld, so counting them as missing coverage would refuse every
        // ablated run. Driven from `qopts` rather than a separate parameter, so
        // this module no longer reaches back into the `predict` subcommand.
        if let Some(hide) = qopts.hide.as_deref() {
            crate::topic::eval::hide_features(&mut remap, &new_genes, hide)?;
        }
        let hidden_rows: std::collections::HashSet<usize> = (0..new_genes.len())
            .filter(|&r| before_hide[r].is_some() && remap.new_to_train[r].is_none())
            .collect();
        let unseen: Vec<usize> = if init.enabled {
            unseen_rows(&before_hide, &hidden_rows)
        } else {
            Vec::new()
        };

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

        // Pass 1: every cell on the matched genes.
        let mut pass = project_all(ProjectAll {
            data_vec: &data_vec,
            remap: &remap.new_to_train,
            projector: &projector,
            side: &side,
            partition: &partition,
            n_model,
            block,
            h: self.h,
        })?;

        let mut init_out: Option<InitOutcome> = None;
        if !unseen.is_empty() {
            let n_cells = pass.latent.nrows();
            let n_new = new_genes.len();
            // Pseudobulks for the profiles: a clustering of the pass-1 latents, so a
            // profile is a gene's expression across the cell states this model sees.
            let n_clusters = (n_cells / 50).clamp(8, 256).min(n_cells.max(1));
            let (_, labels) = matrix_util::principal_graph::kmeans_centroids_seeded(
                &pass.latent,
                n_clusters,
                20,
                0,
            );
            // One more pass over the columns: profiles over ALL new rows, plus each
            // cell's counts on the unseen genes for the bias and the score.
            let mut unseen_local = vec![u32::MAX; n_new];
            for (i, &r) in unseen.iter().enumerate() {
                unseen_local[r] = i as u32;
            }
            let mut obs: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n_cells];
            let mut totals = vec![0f32; unseen.len()];
            let mut profiles = DMatrix::<f32>::zeros(n_new, n_clusters);
            let mut lb = 0usize;
            while lb < n_cells {
                let ub = (lb + block).min(n_cells);
                let csc = data_vec.read_columns_csc(lb..ub)?;
                // Raw CSC arrays sliced by offset: a column view would borrow a local.
                let (offsets, rows_all, vals_all) =
                    (csc.col_offsets(), csc.row_indices(), csc.values());
                profiles_by_cluster(
                    &mut profiles,
                    &labels,
                    (0..csc.ncols()).map(|j| {
                        let (s0, s1) = (offsets[j], offsets[j + 1]);
                        (lb + j, &rows_all[s0..s1], &vals_all[s0..s1])
                    }),
                );
                for j in 0..csc.ncols() {
                    let c = lb + j;
                    let col = csc.col(j);
                    for (&row, &v) in col.row_indices().iter().zip(col.values()) {
                        let li = unseen_local[row];
                        if li != u32::MAX {
                            obs[c].push((li, v));
                            totals[li as usize] += v;
                        }
                    }
                }
                lb = ub;
            }

            let rho_dm = DMatrix::<f32>::from_row_slice(n_model, self.h, &self.rho);
            let modules = self
                .modules
                .as_ref()
                .map(|(pi, mu)| ModuleTables { pi, mu });
            let mut alignment = align_gene_axis(&AlignInputs {
                rho: &rho_dm,
                b_feat: &self.b_feat,
                modules,
                new_to_train: &before_hide,
                profiles_new: Some(&profiles),
                k: init.k,
                similarity_floor: init.similarity_floor,
            });
            let init_idx = alignment.with_status(GeneStatus::Initialized);
            anyhow::ensure!(
                init_idx.len() == unseen.len()
                    && init_idx
                        .iter()
                        .zip(&unseen)
                        .all(|(&g, &r)| alignment.union_to_new[g] == Some(r)),
                "alignment and unseen-row bookkeeping disagree"
            );
            let init_rows = alignment.rows.select_rows(init_idx.iter());
            // A gene the query never expresses has no scale to match; give it the
            // smallest trained bias rather than 0, which would make it the most
            // abundant gene in the partition.
            let fallback = self.b_feat.iter().copied().fold(f32::INFINITY, f32::min);
            let init_bias =
                moment_matched_bias(&init_rows, &pass.latent, &pass.b_cell, &totals, fallback);
            for (i, &g) in init_idx.iter().enumerate() {
                alignment.bias[g] = init_bias[i];
            }
            let n_diffuse = init_idx
                .iter()
                .filter(|&&g| alignment.provenance[g].as_ref().is_some_and(|p| p.diffuse))
                .count();
            info!(
                "gene-axis alignment: {} matched, {} missing, {} initialized ({} diffuse) over \
                 {n_clusters} pseudobulks, k={} floor={}",
                alignment.with_status(GeneStatus::Matched).len(),
                alignment.with_status(GeneStatus::Missing).len(),
                init_idx.len(),
                n_diffuse,
                init.k,
                init.similarity_floor
            );

            // Pass 2: the initialized genes become observations. The comparable score
            // still normalizes over the model's genes only.
            if init.in_fit {
                let union_rm = alignment.rows.transpose().as_slice().to_vec();
                let projector2 = FrozenProjector::new(&FrozenProjectionArgs {
                    feat: &union_rm,
                    b_feat: &alignment.bias,
                    h: self.h,
                    lambda: f64::from(PROJECTION_RIDGE_SGD),
                    dev,
                })?;
                let remap2 = union_remap(&remap.new_to_train, &unseen, n_model);
                info!(
                    "pass 2: re-projecting every cell with the {} initialized genes observed",
                    unseen.len()
                );
                pass = project_all(ProjectAll {
                    data_vec: &data_vec,
                    remap: &remap2,
                    projector: &projector2,
                    side: &side,
                    partition: &partition,
                    n_model,
                    block,
                    h: self.h,
                })?;
            }

            // The null for the initialized genes is the query's own composition over
            // them: there is no training marginal for a gene the model never saw.
            let tot: f32 = totals.iter().sum::<f32>().max(1e-12);
            let null_comp: Vec<f32> = totals.iter().map(|&t| t / tot).collect();
            let scores = score_initialized(
                &init_rows,
                &init_bias,
                &pass.latent,
                &pass.b_cell,
                &obs,
                &null_comp,
            );
            init_out = Some(InitOutcome {
                alignment,
                new_gene_names: new_genes.clone(),
                unseen_rows: unseen,
                scores,
                n_clusters,
                in_fit: init.in_fit,
            });
        }

        Ok(BgeFit {
            data_vec,
            llik: pass.llik,
            total: pass.total,
            latent: pass.latent,
            b_cell: pass.b_cell,
            init: init_out,
        })
    }
}

/// One projection sweep over the query: every cell placed against `projector`
/// through `remap`, scored by [`multinomial_ll`] on the model's own genes.
struct ProjectAll<'a> {
    data_vec: &'a SparseIoVec,
    /// New-data row → row of `projector`'s dictionary.
    remap: &'a [Option<usize>],
    projector: &'a FrozenProjector<'a>,
    side: &'a FrozenSide<'a>,
    partition: &'a [u32],
    /// Rows below this belong to the model's own genes and enter the score;
    /// rows at or above it are initialized genes, observed by the projector but
    /// not scored here.
    n_model: usize,
    block: usize,
    h: usize,
}

struct ProjectAllOut {
    latent: Mat,
    b_cell: Vec<f32>,
    llik: Vec<f32>,
    total: Vec<f32>,
}

fn project_all(a: ProjectAll<'_>) -> anyhow::Result<ProjectAllOut> {
    let ntot = a.data_vec.num_columns();
    let mut llik = Vec::with_capacity(ntot);
    let mut total = Vec::with_capacity(ntot);
    let mut b_cell = Vec::with_capacity(ntot);
    let mut latent = Mat::zeros(ntot, a.h);
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
    let group_nodes = a.projector.group_nodes();
    let mut group = EdgeGroup::default();

    let mut lb = 0usize;
    while lb < ntot {
        let group_lb = lb;
        // Accumulate slabs until the group is full. The inner loop's exit condition
        // IS the flush condition, so a part-group at the end needs no special case.
        while lb < ntot && group.len() < group_nodes {
            let ub = (lb + a.block).min(group_lb + group_nodes).min(ntot);
            let csc = a.data_vec.read_columns_csc(lb..ub)?;
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
                        .filter_map(|(&i, &v)| a.remap[i].map(|t| (t as u32, v)))
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
        let proj = project_group(a.projector, &group, &bar)?;
        let e_cell = proj.theta;
        b_cell.extend_from_slice(&proj.b_node);
        // `multinomial_ll` walks the whole gene axis twice per cell for the
        // normalizer, which makes it the dominant cost here — and it sat serial
        // directly below `project_cells`, which is already rayon-parallel. Everything
        // it reads is shared and immutable, so the map parallelizes as-is and stays in
        // order. `NodeTerm` borrows an interleaved slice, so the split arrays are
        // re-zipped into a per-cell temporary — a few thousand entries against a
        // normalizer that touches the whole gene axis twice. Initialized genes (rows at
        // or past `n_model`) are left out of the score: it stays over the model's genes.
        let n_model = a.n_model as u32;
        llik.par_extend((0..n_group).into_par_iter().map(|c| {
            let (feat, count) = group.cell(c);
            let pos: Vec<(u32, f32)> = feat
                .iter()
                .copied()
                .zip(count.iter().copied())
                .filter(|&(f, _)| f < n_model)
                .collect();
            let node = NodeTerm::new(&pos, a.partition, 1.0);
            multinomial_ll(&e_cell[c * a.h..(c + 1) * a.h], &node, a.side)
        }));
        // `e_cell` is row-major `[n_group, H]`; `Mat` is column-major, so this is a
        // transposing copy rather than a memcpy.
        for c in 0..n_group {
            for j in 0..a.h {
                latent[(group_lb + c, j)] = e_cell[c * a.h + j];
            }
        }
        group.clear();
    }
    bar.finish_and_clear();

    Ok(ProjectAllOut {
        latent,
        b_cell,
        llik,
        total,
    })
}

fn project_group(
    projector: &FrozenProjector,
    group: &EdgeGroup,
    bar: &indicatif::ProgressBar,
) -> anyhow::Result<graph_embedding_util::fit::FrozenProjection> {
    let nodes: Vec<(u32, &[u32], &[f32])> = (0..group.len())
        .map(|j| {
            let (feat, count) = group.cell(j);
            (j as u32, feat, count)
        })
        .collect();
    projector.project(&nodes, group.len(), bar)
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
