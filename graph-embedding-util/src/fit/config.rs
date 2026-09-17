use crate::model::JointEmbedModel;
use candle_util::candle_core::Device;
use candle_util::candle_nn::VarMap;
use data_beans_alg::refine_multilevel::RefineParams;

/// Stratification exponent for cell-axis positive sampling: outer pick
/// is `q(c) ∝ degree(c)^alpha_cell` within each batch. `0.5` gives
/// rare/shallow cells real coverage without starving deeply sequenced
/// cells.
pub(crate) const DEFAULT_STRATIFY_ALPHA_CELL: f32 = 0.5;

/// Fraction of `epochs` the module membership is held at its warm start when
/// [`GeneModuleConfig::warmup_epochs`] is not given.
pub(crate) const MODULE_WARMUP_FRAC: f64 = 0.25;

/// Row structure of the feature axis: every row belongs to one TRACK (a
/// (modality, channel) pair) and names one GENE. Track 0 is the base track.
///
/// Invariants, all checked by [`TrackSpec::validate`]:
/// - `track_of_row.len() == gene_of_row.len() == n_features`;
/// - every track id is `< tracks.len()`;
/// - the gene ids are dense `0..n_genes` (every id in the range is used);
/// - track 0 is non-empty and is a count track;
/// - within one track a gene appears at most once (a track holds at most one
///   row per gene);
/// - on a ONE-track spec, `gene_of_row` is the identity `0..n_features` — a
///   one-track axis is the plain gene axis, and the trainer's per-gene tables
///   are read back as feature rows, so any other ordering would silently
///   permute them.
#[derive(Clone, Debug)]
pub struct TrackSpec {
    /// row → track id, `len == n_features`, dense `0..n_tracks`, 0 = base.
    pub track_of_row: Vec<u32>,
    /// row → gene id, `len == n_features`, dense `0..n_genes`.
    pub gene_of_row: Vec<u32>,
    /// Per track, in id order.
    pub tracks: Vec<TrackInfo>,
}

/// What one track is: its name and whether it carries counts.
#[derive(Clone, Debug)]
pub struct TrackInfo {
    /// e.g. "count/spliced".
    pub name: Box<str>,
    /// A count track gets a distilled encoder in phase 2 (track 0 always is one).
    pub is_count: bool,
}

/// Name of the single track [`TrackSpec::base`] builds.
const BASE_TRACK_NAME: &str = "base";

impl TrackSpec {
    /// The one-track axis every row of which is its own gene — what a feature
    /// axis of plain genes is.
    #[must_use]
    pub fn base(n_features: usize) -> Self {
        Self {
            track_of_row: vec![0; n_features],
            gene_of_row: (0..n_features as u32).collect(),
            tracks: vec![TrackInfo {
                name: BASE_TRACK_NAME.into(),
                is_count: true,
            }],
        }
    }

    #[must_use]
    pub fn n_tracks(&self) -> usize {
        self.tracks.len()
    }

    /// One past the largest gene id — `0` on an empty axis. Meaningful only on
    /// a spec that [`Self::validate`] accepted, where the ids are dense.
    #[must_use]
    pub fn n_genes(&self) -> usize {
        self.gene_of_row
            .iter()
            .copied()
            .max()
            .map_or(0, |g| g as usize + 1)
    }

    /// The one-track axis: row = gene, nothing to compose. Requires the gene
    /// ids to BE the row ids — the trainer's per-gene tables are read back as
    /// feature rows, so a permuted one-track spec is not the base axis.
    #[must_use]
    pub fn is_base(&self) -> bool {
        self.n_tracks() == 1 && self.gene_ids_are_the_identity()
    }

    /// `gene_of_row[i] == i` for every row.
    fn gene_ids_are_the_identity(&self) -> bool {
        self.gene_of_row
            .iter()
            .enumerate()
            .all(|(row, &g)| g as usize == row)
    }

    /// Row ids belonging to track `t`, ascending.
    #[must_use]
    pub fn rows_of_track(&self, t: usize) -> Vec<u32> {
        self.track_of_row
            .iter()
            .enumerate()
            .filter(|&(_, &tr)| tr as usize == t)
            .map(|(row, _)| row as u32)
            .collect()
    }

    /// Track ids carrying counts, ascending. Always contains `0` on a
    /// validated spec.
    #[must_use]
    pub fn count_tracks(&self) -> Vec<usize> {
        self.tracks
            .iter()
            .enumerate()
            .filter(|&(_, t)| t.is_count)
            .map(|(t, _)| t)
            .collect()
    }

    /// Check every invariant listed on the struct against a feature axis of
    /// `n_features` rows.
    pub fn validate(&self, n_features: usize) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.track_of_row.len() == n_features,
            "track_of_row has {} entries but the feature axis has {n_features} rows",
            self.track_of_row.len()
        );
        anyhow::ensure!(
            self.gene_of_row.len() == n_features,
            "gene_of_row has {} entries but the feature axis has {n_features} rows",
            self.gene_of_row.len()
        );
        let n_t = self.tracks.len();
        if let Some(&bad) = self.track_of_row.iter().find(|&&t| t as usize >= n_t) {
            anyhow::bail!("track id {bad} is out of range for {n_t} tracks");
        }
        // Dense gene ids: n_genes = max + 1, and every id below it is used.
        let n_g = self.n_genes();
        let mut gene_seen = vec![false; n_g];
        for &g in &self.gene_of_row {
            gene_seen[g as usize] = true;
        }
        if let Some(missing) = gene_seen.iter().position(|&s| !s) {
            anyhow::bail!("gene ids are not dense: {missing} of {n_g} is never used");
        }
        anyhow::ensure!(n_t > 0, "a track spec needs at least the base track");
        anyhow::ensure!(
            self.track_of_row.contains(&0),
            "the base track (0) has no rows"
        );
        anyhow::ensure!(
            self.tracks[0].is_count,
            "the base track must be a count track"
        );
        // At most one row per (track, gene).
        let mut seen = vec![false; n_t * n_g];
        for (row, (&t, &g)) in self.track_of_row.iter().zip(&self.gene_of_row).enumerate() {
            let slot = &mut seen[t as usize * n_g + g as usize];
            anyhow::ensure!(
                !*slot,
                "row {row} repeats gene {g} within track {t}: a track holds at most one row per gene"
            );
            *slot = true;
        }
        // A one-track axis IS the gene axis: `fit` reads the trainer's per-gene
        // tables back as feature rows, so any other ordering would permute them
        // with every shape still valid. Reject it here, at the boundary.
        anyhow::ensure!(
            n_t > 1 || self.gene_ids_are_the_identity(),
            "a one-track spec must name gene `i` on row `i`"
        );
        Ok(())
    }
}

/// Hyperparameter / configuration bundle for [`fit`]. Constructed by
/// each caller from its own CLI arguments — this crate doesn't import
/// `clap`.
pub struct FitConfig {
    pub embedding_dim: usize,
    /// Batch labels to anchor the cross-batch counterfactual on — a prior
    /// run's carried pseudobulks. Maps to [`MultilevelParams::anchor_batches`]
    /// (greedy batch correction: new batches corrected toward the anchor
    /// frame, the frame never re-adjusted).
    pub anchor_batches: Option<Vec<Box<str>>>,
    /// Batch labels whose columns are mixtures over cell states — maps to
    /// [`MultilevelParams::bulk_batches`]. Greedy: they are corrected toward
    /// the non-bulk (cell) frame and never serve as its counterfactual.
    pub bulk_batches: Option<Vec<Box<str>>>,
    /// Carry the finest collapse level (posterior + cell → pb membership) out
    /// on [`FitOutput::finest_collapse`], retaining its sufficient statistics
    /// even under the memory-lean calibration. `senna bge --emit-pb-reference`
    /// serializes it as the next round's carried reference.
    pub emit_finest_collapse: bool,
    /// Number of multilevel-collapse levels (coarse → fine). Maps
    /// directly to [`MultilevelParams::num_levels`].
    pub num_levels: usize,
    /// Binary-tree partition depth at the finest level — at most
    /// `2^sort_dim + 1` pseudobulk leaves. Maps to
    /// [`MultilevelParams::sort_dim`].
    pub sort_dim: usize,
    /// In-batch k-NN used when merging cells into pseudobulk samples.
    /// Maps to [`MultilevelParams::knn_pb_samples`].
    pub knn_pb_samples: usize,
    /// Coordinate-descent iterations for the per-batch δ correction
    /// inside the collapse. Maps to [`MultilevelParams::num_opt_iter`].
    pub num_opt_iter: usize,
    /// Target rank of the random-projection sketch that seeds batch
    /// correction and the multilevel collapse.
    pub proj_dim: usize,
    pub epochs: usize,
    /// `None` = auto: one weighted pass per epoch over the largest axis.
    /// `Some(n)` = fixed step budget.
    pub batches_per_epoch: Option<usize>,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub seed: u64,
    pub device: Device,
    /// Streaming block size for column-block I/O. `None` falls back to
    /// `matrix_util::utils::default_block_size(n_features)` which
    /// clamps to 100 for large feature counts — that's tiny on
    /// rotational disks. Pass `Some(1024)` or higher when you have
    /// the RAM, especially without `--preload-data`.
    pub block_size: Option<usize>,
    /// Optional per-row HVG weights for the random projection (length =
    /// full feature axis). When `Some(w)`, the RP uses
    /// `project_columns_weighted` with these weights — uninformative
    /// genes are down-weighted but still contribute to the sketch and
    /// every downstream pass. When `None`, falls back to plain batch-
    /// corrected RP (every gene weight = 1).
    pub hvg_weights: Option<Vec<f32>>,
    /// BBKNN + DC-Poisson refinement on the multi-level pseudobulk
    /// partition. `Some(RefineParams::default())` enables it (parity
    /// with senna topic / svd / postprocess); `None` falls back to the
    /// raw hash partition. Setting `num_gibbs == 0 && num_greedy == 0`
    /// inside `Some(..)` is equivalent to disabling.
    pub refine: Option<RefineParams>,
    /// `AdamW` decoupled weight decay applied uniformly to every parameter
    /// (the shared `E_feat`, `b_feat`, and every per-axis head). Post-
    /// step shrinkage; doesn't enter the backward graph. `0.0` disables.
    pub weight_decay: f64,
    /// Phase-1 cell-axis mode (`k`). Controls only what shapes `E_feat` in
    /// phase 1; phase 2 always analytically projects *every* cell against the
    /// fixed feature side, so the full per-cell embedding is unaffected.
    /// - `k == 0`: suppress the cell axis entirely (pure-pb — `E_feat` shaped
    ///   by pb aggregates only; fastest). `senna gem`'s default; `senna bge`
    ///   injects a moderate `k`, which measured better than either extreme.
    /// - `1 ≤ k < n_cells`: keep ≤`k` cells per pb-sample at EVERY collapse
    ///   level (union), shrinking the phase-1 step budget
    ///   (`Σ active_cells / batch_size`) while keeping rare/shallow cells
    ///   visible to the shared feature dictionary.
    /// - `k ≥ n_cells`: no pb-sample exceeds `k`, so subsampling is a no-op —
    ///   every cell shapes `E_feat` (legacy all-cells behaviour; slowest).
    pub phase1_cells_per_pb: usize,
    /// Hierarchical phase 1: units per optimizer step.
    pub hier_units_per_step: usize,
    /// Hierarchical phase 1: modules drawn per unit per step for the gene-level term.
    pub hier_modules_per_unit: usize,
    /// Gene modules. The hierarchical phase 1 reads only `n_modules` and
    /// `parent`: `M` sizes its hard gene partition and a parent seeds it
    /// (`senna update`). The remaining fields configure the learned mixed-
    /// membership layer ([`crate::model::FeatModules`]) that `pinto cage`
    /// trains directly; `fit()` never builds that layer.
    pub gene_modules: Option<GeneModuleConfig>,
    /// Row structure of the feature axis. `None` = every row is its own gene
    /// ([`TrackSpec::base`], built inside [`fit`]) — what `senna bge` runs.
    pub tracks: Option<TrackSpec>,
    /// Ridge on the per-track offsets (`Δ^t_m`, `u^t_g · V^t`), keeping the
    /// non-base tracks close to the base model. Inert at one track.
    pub offset_l2: f32,
    /// Rank of every non-base track's per-gene offset, `δ^t_g = u^t_g · V^t`:
    /// a track moves its genes inside one shared `offset_rank`-dimensional
    /// subspace. Its own number, checked against `embedding_dim` (`1..=H`) on
    /// a tracked axis; inert at one track.
    pub offset_rank: usize,
    /// Gene rows of the dictionary given before the fit (a `senna fne`
    /// embedding, say): phase 1 starts from them, and under `freeze` pins them
    /// and trains only the rest — the unit side, every bias, and the rows of
    /// genes not listed. On a tracked axis these are the BASE rows (ids on the
    /// gene axis); every track's offset trains on top.
    pub preset_features: Option<crate::PresetRows>,
    /// Given offsets on non-base tracks, by gene (see [`crate::PresetOffsets`]);
    /// empty for none. Requires `preset_features` under a pinning mode.
    pub preset_offsets: Vec<crate::PresetOffsets>,
}

/// Caller-facing configuration of the learned gene modules.
#[derive(Clone, Debug)]
pub struct GeneModuleConfig {
    /// Number of modules `M`.
    pub n_modules: usize,
    /// Epochs the warm-start membership is held before it trains. `None` = a
    /// quarter of the epochs, at least one.
    pub warmup_epochs: Option<usize>,
    /// Per-step probability that a feature is hidden when the module counts are
    /// pooled (`0` = off).
    pub gene_dropout: f32,
    /// Weight of the exact cell–module term relative to the NCE.
    pub lambda_module: f32,
    /// Weight of the load-balance prior `KL(π̄ ‖ Uniform)`.
    pub lambda_balance: f32,
    /// Ridge on the per-feature residual `r_g` — the module model's only per-row
    /// table, so this replaces `feature_embedding_l2`.
    pub residual_l2: f32,
    /// Units (cells or pseudobulks) pooled for the exact term per step per axis.
    pub units_per_step: usize,
    /// Share of a feature's warm-start membership on its k-means module.
    pub init_own_mass: f32,
    /// A parent run's module tables to warm-start from (`senna update`): matched
    /// features carry the parent's membership, unmatched ones are initialized
    /// through the parent's modules from their profile neighbours, and `μ` starts
    /// at the parent's. Overrides `n_modules` with the parent's `M`. `None` = the
    /// k-means warm start from this fit's own pseudobulks.
    pub parent: Option<ParentModulesOwned>,
}

/// A parent run's module tables, carried on the config for the warm start
/// (`senna update`).
#[derive(Clone, Debug)]
pub struct ParentModulesOwned {
    /// Parent composed rows `[D_parent × H]`.
    pub rho: nalgebra::DMatrix<f32>,
    /// Parent membership `[D_parent × M]`.
    pub pi: nalgebra::DMatrix<f32>,
    /// Parent module dictionary `[M × H]`.
    pub mu: nalgebra::DMatrix<f32>,
    /// For each feature of this fit's axis, the parent row it matched.
    pub row_to_parent: Vec<Option<usize>>,
    /// Neighbourhood for the unmatched features' initialization.
    pub knobs: crate::transfer::AlignKnobs,
}

impl GeneModuleConfig {
    /// Epochs the warm-start membership is held: the explicit count, else a
    /// quarter of the epochs, at least one and at most all of them. The one
    /// definition every trainer with a module model uses.
    #[must_use]
    pub fn warmup_epochs_for(&self, epochs: usize) -> usize {
        self.warmup_epochs
            .unwrap_or_else(|| ((epochs as f64) * MODULE_WARMUP_FRAC).ceil() as usize)
            .clamp(usize::from(epochs > 0), epochs)
    }
}

/// Trained model + its `VarMap`. The varmap is exposed so callers can
/// save checkpoints or re-run inference; the current caller (`senna
/// gbe`) only consumes `model`, so it sits unused but kept alive.
pub struct FitOutput {
    pub model: JointEmbedModel,
    /// The finest collapse level and its cell → pb membership, present iff
    /// [`FitConfig::emit_finest_collapse`] was set. The membership indexes
    /// the global cell ids of the `UnifiedData` the fit ran on.
    pub finest_collapse: Option<(data_beans_alg::collapse_data::CollapsedOut, Vec<usize>)>,
    pub varmap: VarMap,
    /// Un-normalized baseline MAP per-cell projection norm from phase 2 (`0`
    /// for cells with no observed features / when phase 2 was skipped). The
    /// empty-droplet cell QC reads this: empties solve to ≈0, real cells far
    /// above. The stored latent (`model.e_cell`) is the L2 *direction*; this
    /// norm is the un-normalized magnitude it was divided by.
    pub cell_nrms: Vec<f32>,
    /// The phase-1 pseudobulk embeddings per collapse level (coarsest → finest),
    /// each with its pseudobulks' batches — the geometry the feature side was
    /// trained against, for batch diagnostics. In the cells' frame (shifted by
    /// the phase-2 gauge with them), so they co-plot with `model.e_cell`.
    pub pb_embeddings: Vec<super::pb_readout::PbLevelEmbedding>,
    /// The distilled encoders phase 2 placed the cells with, when phase 2 was
    /// given distillation targets; `None` when the block SGD placed them. One
    /// per COUNT track of the feature axis.
    pub cell_encoder: Option<super::projection::CellEncoders>,
    /// Per-batch gene fold `log δ_gb` phase 2 divided each batch's cell counts by;
    /// `None` on single-batch data.
    pub batch_gene_fold: Option<super::batch_fold::BatchGeneFold>,
    /// Per-cell intercept of each non-base track (tracks `1..T`, `[n_cells]`
    /// each); empty on a one-track axis.
    pub track_intercepts: Vec<Vec<f32>>,
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod config_tests;
