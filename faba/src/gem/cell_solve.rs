//! Phase-2 for `faba gem`: project every cell onto the **frozen** feature
//! dictionary, in parallel. This is the gem adapter; the actual per-cell
//! Poisson-MAP IRLS solver is the shared, model-agnostic
//! [`graph_embedding_util::cell_projection`] (also used by `senna bge`).
//!
//! The gem-specific work here is turning the trained `GemModel` + the
//! cell-axis pools into the two inputs the solver wants: the frozen
//! feature embeddings `e_f` / `b_f`, and each cell's observed
//! `(feature, count)` list.

use super::common::candle_core;
use anyhow::{Context, Result};
use candle_core::Tensor;
use data_beans::sparse_io_vector::SparseIoVec;
use nalgebra::DMatrix;
use rustc_hash::FxHashMap;

use super::args::GemArgs;
use super::feature_table::{BackendRowMap, RowStratum};
use super::model::{FeatureRows, GemModel};
use super::pseudobulk::{AxisPools, GenesResidual};
use super::sampling::COUNT_BIAS_MODALITY;

/// One distinct feature-row identity — the `embed_and_bias_rows` inputs that
/// all collapse to the same frozen `e_f` / `b_f`. Mirrors how the sampler
/// builds positives (`sampling::push_agg_positive` / `push_component_positive`):
/// AGG → β_g; count-comp → splice modality + shared `COUNT_BIAS_MODALITY`
/// bias; modifier-comp → its own modality + region + bias.
pub(crate) struct Identity {
    pub gene: u32,
    pub q_modality: u32,
    pub region: u32,
    pub is_agg: bool,
    pub bias_modality: u32,
}

/// Keep the per-feature divisor's log well inside the solver's `s` clamp (±30);
/// a fold-factor of e^4 ≈ 55× is already far beyond anything real.
const DIVISOR_LOG_CLAMP: f64 = 4.0;

/// Phase-2 batch correction: divide each cell's genes counts by the finest-level
/// `μ_residual[gene, pb(cell)]` (its batch fold-factor) before the Poisson-MAP
/// projection, so `e_cell` fits the de-batched signal — the analytic twin of
/// `senna topic`'s `AdjMethod::Residual` divisor. Built only when the collapse
/// fit a `μ_residual` (>1 batch); a no-op otherwise. Satellite (modifier)
/// features are not corrected (they have no genes-backend fold-factor).
pub(crate) struct BatchDivisor<'a> {
    mu_residual: &'a DMatrix<f32>, // finest [spliced_row × n_pb]
    gene_to_row: FxHashMap<u32, usize>,
    cell_to_finest_pb: &'a [usize], // cell → finest pb id (μ_residual column)
}

impl<'a> BatchDivisor<'a> {
    /// `genes_residual` is the finest-level fold-factor; `cell_to_finest_pb` is
    /// the finest entry of `cell_to_pb_per_level` (coarsest-first ⇒ `.last()`).
    pub(crate) fn new(genes_residual: &'a GenesResidual, cell_to_finest_pb: &'a [usize]) -> Self {
        let mut gene_to_row = FxHashMap::default();
        for (row, &g) in genes_residual.resid_row_to_gene.iter().enumerate() {
            gene_to_row.insert(g, row);
        }
        Self {
            mu_residual: &genes_residual.mu_residual,
            gene_to_row,
            cell_to_finest_pb,
        }
    }

    /// The clamped fold-factor `μ_residual[row, pb]` a count is divided by; 1.0
    /// (no-op) on an out-of-range column or non-positive value. The shared core
    /// of both the pooled (`genes_divisor`) and streaming (`divide_column`) paths.
    fn clamped_factor(&self, row: usize, pb: usize) -> f32 {
        if pb >= self.mu_residual.ncols() {
            return 1.0;
        }
        let d = self.mu_residual[(row, pb)];
        if d > 0.0 {
            let hi = DIVISOR_LOG_CLAMP.exp() as f32;
            d.clamp(hi.recip(), hi)
        } else {
            1.0
        }
    }

    /// The clamped fold-factor `μ_residual[gene, pb(cell)]` to divide a count by;
    /// 1.0 (no-op) when the gene/cell is unmapped. Used by the pooled path (the
    /// streaming path divides per cell via [`Self::divide_column`]).
    fn genes_divisor(&self, gene: u32, cell: usize) -> f32 {
        let (Some(&row), Some(&pb)) = (
            self.gene_to_row.get(&gene),
            self.cell_to_finest_pb.get(cell),
        ) else {
            return 1.0;
        };
        self.clamped_factor(row, pb)
    }

    /// Map each genes-backend row to its `μ_residual` row (`None` ⇒ no
    /// fold-factor ⇒ divisor 1.0). `backend_gene[row]` is the row's gene id (the
    /// streaming caller derives it). This lets [`Self::divide_column`] read the
    /// small `μ_residual` directly instead of broadcasting it into a dense
    /// `[n_backend_rows × n_pb]` divisor.
    pub(crate) fn backend_to_resid(&self, backend_gene: &[Option<u32>]) -> Vec<Option<usize>> {
        backend_gene
            .iter()
            .map(|g| g.and_then(|g| self.gene_to_row.get(&g).copied()))
            .collect()
    }

    /// Self-normalizing **sparse** divide of one cell's genes column by its batch
    /// fold-factor, read straight from `μ_residual` (no dense broadcast).
    /// Numerically identical to matrix-util's
    /// `adjust_by_division_of_selected_inplace` (DMatrix variant): a per-cell
    /// scale `λ = Σx/Σd` with divisor 1.0 off-gene (keeping depth for `b_cell`).
    /// `pb` is the cell's finest pseudobulk; `(rows, vals)` are the column's
    /// nonzero backend rows + counts; `emit(backend_row, divided_value)` is
    /// called once per nonzero.
    fn divide_column<F: FnMut(usize, f32)>(
        &self,
        resid_row: &[Option<usize>],
        pb: usize,
        rows: &[usize],
        vals: &[f32],
        mut emit: F,
    ) {
        let factor = |row: usize| resid_row[row].map_or(1.0, |rr| self.clamped_factor(rr, pb));
        let dsum: f32 = rows.iter().map(|&r| factor(r)).sum();
        let xsum: f32 = vals.iter().copied().sum();
        let scale = if dsum > 0.0 { xsum / dsum } else { 1.0 };
        for (&row, &v) in rows.iter().zip(vals) {
            emit(row, v / (factor(row) * scale));
        }
    }
}

/// Solve `e_cell` (and `b_cell`) by projecting every cell onto the frozen
/// feature side, then overwrite the model's `e_cell` / `b_cell` vars.
/// Returns the pre-L2-normalisation norm for each cell (one entry per model
/// cell, ordered by cell id 0..n_cells).  A near-zero norm signals that the
/// IRLS had nothing to fit — the cell's expressed genes were all near the
/// β=0 init — and is used downstream for cell prior-score QC.
pub(crate) fn solve_cell_embeddings(
    model: &mut GemModel,
    pools: &AxisPools,
    args: &GemArgs,
    batch_divisor: Option<&BatchDivisor>,
) -> Result<Vec<f32>> {
    let n_cells = model.n_cells;
    let h = model.embedding_dim;
    if n_cells == 0 {
        return Ok(vec![]);
    }

    // 1. Distinct identities + per-cell (identity, count) lists (genes counts
    //    divided by their batch fold-factor when `batch_divisor` is set).
    let (ids, per_cell) = collect_identities(pools, n_cells, batch_divisor);
    // 2. Frozen feature embeddings (one batched device pass → CPU).
    let (frozen_e, frozen_b) = embed_identities(model, &ids, h)?;
    // 3. Parallel per-cell Poisson-MAP projection (shared solver), with a
    //    per-cell progress bar over the rayon fan-out.
    let lambda = args.train.phase2_ridge.max(0.0) as f64;
    let pb = graph_embedding_util::progress::new_progress_bar(n_cells as u64);
    pb.set_message("cell projection");
    // gem scores with a per-cell bias (b_cell), so keep the fitted b_c.
    let (e_flat, b_flat) = graph_embedding_util::cell_projection::project_cells(
        &frozen_e,
        &frozen_b,
        &per_cell,
        h,
        lambda,
        Some(&pb),
    );
    pb.finish_and_clear();
    // 4. L2-normalise (depth → b_cell) + write back into the model vars.
    finalize_e_cell(model, e_flat, b_flat)
}

/// L2-normalise each solved `e_cell` row (depth correction — the Poisson-MAP
/// matches absolute counts, so the *norm* leaks sequencing depth and would
/// otherwise dominate ~82% of the variance and collapse downstream archetypal
/// topics; depth stays in the unpenalized `b_cell`). Captures the pre-norm
/// magnitudes (`cell_nrms`: near-zero = the IRLS fit nothing, a dead-gene-
/// region cell — used for cell QC), then writes `e_cell`/`b_cell` back into
/// the model vars + cached fields. Shared by the pooled and streaming solvers.
pub(crate) fn finalize_e_cell(
    model: &mut GemModel,
    mut e_flat: Vec<f32>,
    b_flat: Vec<f32>,
) -> Result<Vec<f32>> {
    let n_cells = model.n_cells;
    let h = model.embedding_dim;
    let mut cell_nrms: Vec<f32> = Vec::with_capacity(n_cells);
    for c in 0..n_cells {
        let row = &mut e_flat[c * h..(c + 1) * h];
        let nrm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        cell_nrms.push(nrm);
        if nrm > 0.0 {
            for v in row.iter_mut() {
                *v /= nrm;
            }
        }
    }
    let e_t = Tensor::from_vec(e_flat, (n_cells, h), &model.dev)?;
    let b_t = Tensor::from_vec(b_flat, n_cells, &model.dev)?;
    {
        let vars = model.varmap.data().lock().unwrap();
        vars.get("e_cell")
            .context("e_cell var missing")?
            .set(&e_t)?;
        vars.get("b_cell")
            .context("b_cell var missing")?
            .set(&b_t)?;
    }
    model.e_cell = e_t;
    model.b_cell = b_t;
    Ok(cell_nrms)
}

// ─────────────────────────────────────────────────────────────────────────
// Streaming phase-2: project cells without ever materialising the per-cell
// `AxisPools` (~one entry per (gene, cell) ≈ tens of GB at 700k cells). The
// frozen identity→embedding dictionary is cell-independent, so it's built once
// from the row maps; cells are then streamed from the backend in chunks, each
// chunk's `(identity, count)` lists reconstructed on the fly and projected.
// Reproduces `collect_identities` exactly: one AGG entry per (gene, cell)
// (spliced+unspliced summed), one count entry per (gene, splice, cell), one
// modifier entry per (gene, component, cell).
// ─────────────────────────────────────────────────────────────────────────

/// One satellite modality backend, ready to stream its modifier mass into the
/// matched genes cells.
pub struct SatStream<'a> {
    pub backend: &'a SparseIoVec,
    pub row_map: &'a BackendRowMap,
    pub feature_to_backend_row: &'a [usize],
    /// `genes_cell_to_sat_cols[genes_cell]` = the satellite columns matched to
    /// that genes cell (inverse of `SatelliteLink.col_to_genes_cell`; unmatched
    /// satellite columns are simply absent → donate nothing).
    pub genes_cell_to_sat_cols: Vec<Vec<u32>>,
}

/// Everything streaming phase-2 needs in lieu of `pb.cell_pools`.
pub struct CellStreamCtx<'a> {
    pub genes_backend: &'a SparseIoVec,
    pub genes_row_map: &'a BackendRowMap,
    pub feature_to_backend_row: &'a [usize],
    pub satellites: Vec<SatStream<'a>>,
    /// `cell_columns[cell_id]` = the genes-backend column for model cell
    /// `cell_id`. Identity (`0..n_cells`) after the up-front mask+subset;
    /// `live_cell_old_ids` in the refine pass (backend keeps its N_old layout).
    /// `n_cells` == `cell_columns.len()`. Satellite reverse indices are keyed
    /// by `cell_id`, not the backend column.
    pub cell_columns: Vec<usize>,
}

/// Inverse of `feature_to_backend_row`: backend compact row → unified feature
/// id (`u32::MAX` for rows not in the unified space). Mirrors
/// `UnifiedData::materialize_cell_triplets`.
fn backend_to_unified(feature_to_backend_row: &[usize], n_backend_rows: usize) -> Vec<u32> {
    let mut b2u = vec![u32::MAX; n_backend_rows];
    for (uid, &brow) in feature_to_backend_row.iter().enumerate() {
        if brow < n_backend_rows {
            b2u[brow] = uid as u32;
        }
    }
    b2u
}

/// Interned-identity key: `(gene, q_modality, region, is_agg)`.
type IdentityKey = (u32, u32, u32, bool);
/// Identity key → dense identity-id lookup.
type IdentityIndex = FxHashMap<IdentityKey, u32>;

/// Build the bounded identity universe from the row maps (cell-independent) and
/// a `(gene, q_modality, region, is_agg) → identity-id` lookup. Same keys and
/// insertion logic as `collect_identities`/`intern`.
fn build_identity_universe(ctx: &CellStreamCtx) -> (Vec<Identity>, IdentityIndex) {
    let mut ids: Vec<Identity> = Vec::new();
    let mut map: IdentityIndex = FxHashMap::default();
    // Intern every identity a row could yield (same keys as `collect_identities`).
    let mut scan = |rm: &BackendRowMap| {
        for uid in 0..rm.stratum.len() {
            let (Some(stratum), Some(g)) = (rm.stratum[uid], rm.gene[uid]) else {
                continue;
            };
            match stratum {
                RowStratum::CountComp => {
                    let m = rm.modality[uid].unwrap_or(0);
                    intern(&mut ids, &mut map, count_identity(g, m));
                    intern(&mut ids, &mut map, agg_identity(g));
                }
                RowStratum::ModifierComp => {
                    if let Some(m) = rm.modality[uid] {
                        let r = rm.region[uid].unwrap_or(0);
                        intern(&mut ids, &mut map, modifier_identity(g, m, r));
                    }
                }
                RowStratum::Site => {}
            }
        }
    };
    scan(ctx.genes_row_map);
    for sat in &ctx.satellites {
        scan(sat.row_map);
    }
    (ids, map)
}

fn agg_identity(g: u32) -> Identity {
    Identity {
        gene: g,
        q_modality: 0,
        region: 0,
        is_agg: true,
        bias_modality: 0,
    }
}
fn count_identity(g: u32, m: u32) -> Identity {
    Identity {
        gene: g,
        q_modality: m,
        region: 0,
        is_agg: false,
        bias_modality: COUNT_BIAS_MODALITY,
    }
}
fn modifier_identity(g: u32, m: u32, r: u32) -> Identity {
    Identity {
        gene: g,
        q_modality: m,
        region: r,
        is_agg: false,
        bias_modality: m,
    }
}

/// Map one streamed `(backend row → uid, value)` into the chunk's per-cell
/// lists, mirroring `aggregate_pools::accumulate` + `collect_identities`: a
/// CountComp row feeds both its splice identity and the gene's AGG sum; a
/// ModifierComp row feeds its (modality, region) identity. Shared by the genes
/// and satellite stream callbacks so the mapping can't drift between them.
/// Per-chunk streaming accumulator: each local cell's `(identity, count)`
/// entries, plus the `(local_cell, gene)` → summed-count map for the AGG rows
/// (spliced+unspliced merged). Owns both so the `for_each_triplet` closures can
/// borrow it without entangling the surrounding cell-axis arrays.
struct CellAccum {
    per_cell: Vec<Vec<(u32, f32)>>,
    agg_acc: FxHashMap<(u32, u32), f32>,
}

impl CellAccum {
    /// Route one streamed `(row, local_cell, value)` triplet (via its backend
    /// row map `rm`) into the per-cell identity list / AGG sum.
    fn accumulate(
        &mut self,
        rm: &BackendRowMap,
        uid: usize,
        local: usize,
        v: f32,
        id_of: &impl Fn(&IdentityKey) -> u32,
    ) {
        let (Some(stratum), Some(g)) = (rm.stratum[uid], rm.gene[uid]) else {
            return;
        };
        match stratum {
            RowStratum::CountComp => {
                let m = rm.modality[uid].unwrap_or(0);
                self.per_cell[local].push((id_of(&(g, m, 0, false)), v));
                *self.agg_acc.entry((local as u32, g)).or_insert(0.0) += v;
            }
            RowStratum::ModifierComp => {
                if let Some(m) = rm.modality[uid] {
                    let r = rm.region[uid].unwrap_or(0);
                    self.per_cell[local].push((id_of(&(g, m, r, false)), v));
                }
            }
            RowStratum::Site => {}
        }
    }
}

/// Streaming counterpart of [`solve_cell_embeddings`]: identical result, but
/// never materialises the per-cell pool.
pub(crate) fn solve_cell_embeddings_streaming(
    model: &mut GemModel,
    ctx: &CellStreamCtx,
    args: &GemArgs,
    batch_divisor: Option<&BatchDivisor>,
) -> Result<Vec<f32>> {
    let n_cells = ctx.cell_columns.len();
    let h = model.embedding_dim;
    if n_cells == 0 {
        return Ok(vec![]);
    }

    // 1. Cell-independent identity dictionary + frozen embeddings.
    let (ids, key_to_id) = build_identity_universe(ctx);
    let (frozen_e, frozen_b) = embed_identities(model, &ids, h)?;
    let id_of = |k: &IdentityKey| *key_to_id.get(k).expect("identity in universe");
    let lambda = args.train.phase2_ridge.max(0.0) as f64;

    // 2. Backend-row → unified-feature-id inverses (built once).
    let g_b2u = backend_to_unified(ctx.feature_to_backend_row, ctx.genes_backend.num_rows());
    let sat_b2u: Vec<Vec<u32>> = ctx
        .satellites
        .iter()
        .map(|s| backend_to_unified(s.feature_to_backend_row, s.backend.num_rows()))
        .collect();

    // Sparse batch-divide map (built once): each genes-backend row → its
    // `μ_residual` row, so the divide reads the small fold-factor matrix directly
    // (no dense `[backend_rows × n_pb]` broadcast). `None` ⇒ no correction (≤1 batch).
    let resid_row: Option<Vec<Option<usize>>> = batch_divisor.map(|bo| {
        let backend_gene: Vec<Option<u32>> = (0..ctx.genes_backend.num_rows())
            .map(|r| {
                let uid = g_b2u[r];
                if uid == u32::MAX {
                    None
                } else {
                    ctx.genes_row_map.gene[uid as usize]
                }
            })
            .collect();
        bo.backend_to_resid(&backend_gene)
    });

    // 3. Chunk over cells. Per chunk: stream genes + satellite triplets, build
    //    per-cell (identity, count) lists, project, scatter back.
    let nnz = ctx.genes_backend.num_non_zeros().unwrap_or(0);
    let avg = (nnz / n_cells.max(1)).max(1);
    let chunk_cells = (8_000_000 / avg).clamp(1, n_cells);
    let slab = chunk_cells.min(1 << 14);

    let mut e_flat = vec![0f32; n_cells * h];
    let mut b_flat = vec![0f32; n_cells];

    // One bar across every chunk (sized to the full cell count), incremented
    // per solved cell inside `project_cells`.
    let pb = graph_embedding_util::progress::new_progress_bar(n_cells as u64);
    pb.set_message("cell projection");

    for chunk_start in (0..n_cells).step_by(chunk_cells) {
        let chunk_end = (chunk_start + chunk_cells).min(n_cells);
        let nlocal = chunk_end - chunk_start;
        // The genes-backend columns for this chunk's cells (local index == cell
        // id − chunk_start, i.e. the `out_col` the backend reports).
        let cols = &ctx.cell_columns[chunk_start..chunk_end];

        // AGG sums spliced+unspliced per (local_cell, gene); count/modifier
        // entries are one-per-backend-row (pushed directly, never merged).
        let mut accum = CellAccum {
            per_cell: vec![Vec::new(); nlocal],
            agg_acc: FxHashMap::default(),
        };

        let rm = ctx.genes_row_map;
        if let (Some(bo), Some(resid_row)) = (batch_divisor, resid_row.as_ref()) {
            // Divide mode: read each slab's genes counts as a CSC, then divide each
            // cell's column by its batch fold-factor (sparse — straight from
            // `μ_residual`, per-cell self-normalizing scale) and accumulate the
            // divided values. Slab-bounded like `for_each_triplet`.
            for slab_start in (0..nlocal).step_by(slab) {
                let slab_end = (slab_start + slab).min(nlocal);
                let scols = &cols[slab_start..slab_end];
                let yy = ctx.genes_backend.read_columns_csc(scols.iter().copied())?;
                for (k, col) in yy.col_iter().enumerate() {
                    let local = slab_start + k;
                    let cell_pb = bo.cell_to_finest_pb[chunk_start + local];
                    bo.divide_column(
                        resid_row,
                        cell_pb,
                        col.row_indices(),
                        col.values(),
                        |row, v| {
                            let uid = g_b2u[row];
                            if uid != u32::MAX {
                                accum.accumulate(rm, uid as usize, local, v, &id_of);
                            }
                        },
                    );
                }
            }
        } else {
            ctx.genes_backend
                .for_each_triplet(cols.iter().copied(), slab, |row, out_col, v| {
                    let uid = g_b2u[row as usize];
                    if uid != u32::MAX {
                        accum.accumulate(rm, uid as usize, out_col as usize, v, &id_of);
                    }
                })?;
        }

        for (si, sat) in ctx.satellites.iter().enumerate() {
            // Gather this chunk's satellite columns + their local-cell routing.
            // Reverse index is keyed by cell id (= chunk_start + local), not the
            // genes-backend column.
            let mut sat_cols: Vec<usize> = Vec::new();
            let mut sat_local: Vec<usize> = Vec::new();
            for k in 0..nlocal {
                for &sc in &sat.genes_cell_to_sat_cols[chunk_start + k] {
                    sat_cols.push(sc as usize);
                    sat_local.push(k);
                }
            }
            if sat_cols.is_empty() {
                continue;
            }
            let rm = sat.row_map;
            let b2u = &sat_b2u[si];
            sat.backend
                .for_each_triplet(sat_cols.iter().copied(), slab, |row, out_col, v| {
                    let uid = b2u[row as usize];
                    if uid != u32::MAX {
                        accum.accumulate(rm, uid as usize, sat_local[out_col as usize], v, &id_of);
                    }
                })?;
        }

        for ((local, g), sum) in accum.agg_acc {
            accum.per_cell[local as usize].push((id_of(&(g, 0, 0, true)), sum));
        }

        let (e_chunk, b_chunk) = graph_embedding_util::cell_projection::project_cells(
            &frozen_e,
            &frozen_b,
            &accum.per_cell,
            h,
            lambda,
            Some(&pb),
        );
        e_flat[chunk_start * h..chunk_end * h].copy_from_slice(&e_chunk);
        b_flat[chunk_start..chunk_end].copy_from_slice(&b_chunk);
    }
    pb.finish_and_clear();

    finalize_e_cell(model, e_flat, b_flat)
}

pub(crate) fn intern(ids: &mut Vec<Identity>, map: &mut IdentityIndex, id: Identity) -> u32 {
    let key = (id.gene, id.q_modality, id.region, id.is_agg);
    if let Some(&x) = map.get(&key) {
        return x;
    }
    let x = ids.len() as u32;
    map.insert(key, x);
    ids.push(id);
    x
}

/// Walk the cell-axis pools once: intern each distinct feature identity and
/// append `(identity, count)` to its cell's list. When `batch_divisor` is set,
/// the genes strata (AGG + count-comp) are divided by their batch fold-factor
/// `μ_residual[gene, pb(cell)]` — a plain per-element divide (this pooled path
/// is the rare cell-axis case; the streaming path adds a per-cell self-normalizing
/// scale via [`BatchDivisor::divide_column`]). Satellites are not corrected.
fn collect_identities(
    pools: &AxisPools,
    n_cells: usize,
    batch_divisor: Option<&BatchDivisor>,
) -> (Vec<Identity>, Vec<Vec<(u32, f32)>>) {
    let mut ids: Vec<Identity> = Vec::new();
    let mut map: IdentityIndex = FxHashMap::default();
    let mut per_cell: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n_cells];

    // Genes-strata batch divide by `μ_residual[gene, pb(cell)]` (identity for ≤1
    // batch, where `batch_divisor` is `None`).
    let divide = |gene: u32, cell: usize, c: f32| {
        batch_divisor.map_or(c, |bo| c / bo.genes_divisor(gene, cell))
    };

    // AGG anchor: e_f = β_g (gate masked), bias = b_agg[g]. Genes backend.
    for i in 0..pools.agg.len() {
        let cell = pools.agg.axis_ids[i] as usize;
        if cell >= n_cells {
            continue;
        }
        let gene = pools.agg.gene_ids[i];
        let count = divide(gene, cell, pools.agg.counts[i]);
        let idx = intern(
            &mut ids,
            &mut map,
            Identity {
                gene,
                q_modality: 0,
                region: 0,
                is_agg: true,
                bias_modality: 0,
            },
        );
        per_cell[cell].push((idx, count));
    }

    // Count-comp: splice modality (≥1), region 0, shared count bias slot.
    // Genes backend.
    for i in 0..pools.count_comp.len() {
        let cell = pools.count_comp.axis_ids[i] as usize;
        if cell >= n_cells {
            continue;
        }
        let gene = pools.count_comp.gene_ids[i];
        let count = divide(gene, cell, pools.count_comp.counts[i]);
        let idx = intern(
            &mut ids,
            &mut map,
            Identity {
                gene,
                q_modality: pools.count_comp.modality_ids[i],
                region: pools.count_comp.region_ids[i],
                is_agg: false,
                bias_modality: COUNT_BIAS_MODALITY,
            },
        );
        per_cell[cell].push((idx, count));
    }

    // Modifier-comp: its own modality + transcript-position region + bias.
    for (m, pool) in pools.modifier_comp_per_modality.iter().enumerate() {
        let m = m as u32;
        for i in 0..pool.len() {
            let cell = pool.axis_ids[i] as usize;
            if cell >= n_cells {
                continue;
            }
            let idx = intern(
                &mut ids,
                &mut map,
                Identity {
                    gene: pool.gene_ids[i],
                    q_modality: m,
                    region: pool.region_ids[i],
                    is_agg: false,
                    bias_modality: m,
                },
            );
            per_cell[cell].push((idx, pool.counts[i]));
        }
    }

    (ids, per_cell)
}

/// Compute the frozen `(e_f, b_f)` for each identity via the model's
/// `embed_and_bias_rows`, in chunks, then bring them to the CPU. Returns
/// `(e [n_id × h] row-major, b [n_id])`.
pub(crate) fn embed_identities(
    model: &GemModel,
    ids: &[Identity],
    h: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    const CHUNK: usize = 65536;
    let n = ids.len();
    let mut e = vec![0f32; n * h];
    let mut b = vec![0f32; n];
    let mut off = 0;
    for chunk in ids.chunks(CHUNK) {
        let gene: Vec<u32> = chunk.iter().map(|x| x.gene).collect();
        let q_mod: Vec<u32> = chunk.iter().map(|x| x.q_modality).collect();
        let region: Vec<u32> = chunk.iter().map(|x| x.region).collect();
        let bias_mod: Vec<u32> = chunk.iter().map(|x| x.bias_modality).collect();
        let is_agg: Vec<bool> = chunk.iter().map(|x| x.is_agg).collect();
        let (e_t, b_t) = model.embed_and_bias_rows(&FeatureRows {
            gene_for_rho: &gene,
            gene_for_z: &gene,
            modality_for_q: &q_mod,
            region_for_delta: &region,
            gene_for_bias: &gene,
            modality_for_bias: &bias_mod,
            is_agg: &is_agg,
        })?;
        let e_rows = e_t.to_vec2::<f32>()?;
        let b_vals = b_t.to_vec1::<f32>()?;
        for (j, row) in e_rows.iter().enumerate() {
            e[(off + j) * h..(off + j + 1) * h].copy_from_slice(row);
        }
        b[off..off + chunk.len()].copy_from_slice(&b_vals);
        off += chunk.len();
    }
    Ok((e, b))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gr(mu: DMatrix<f32>, row_to_gene: Vec<u32>) -> GenesResidual {
        GenesResidual {
            mu_residual: mu,
            resid_row_to_gene: row_to_gene,
        }
    }

    #[test]
    fn genes_divisor_is_clamped_mu_residual() {
        // 2 spliced rows × 2 finest pbs of fold-factors; cell 0 → pb1, cell 1 → pb0.
        let g = gr(
            DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 0.5, 1.0]),
            vec![10, 20],
        );
        let cell_to_finest_pb = [1usize, 0usize];
        let bo = BatchDivisor::new(&g, &cell_to_finest_pb);
        assert!((bo.genes_divisor(10, 0) - 2.0).abs() < 1e-6); // μ[(0,1)]
        assert!((bo.genes_divisor(20, 1) - 0.5).abs() < 1e-6); // μ[(1,0)]
        assert_eq!(bo.genes_divisor(999, 0), 1.0); // unknown gene → no-op
        assert_eq!(bo.genes_divisor(10, 5), 1.0); // out-of-range cell → no-op
                                                  // wild fold-factor is clamped to e^4.
        let g2 = gr(DMatrix::from_row_slice(1, 1, &[1.0e9]), vec![7]);
        let bo2 = BatchDivisor::new(&g2, &[0usize]);
        assert!((bo2.genes_divisor(7, 0) - (DIVISOR_LOG_CLAMP as f32).exp()).abs() < 1.0);
    }

    #[test]
    fn divide_column_is_self_normalizing_sparse() {
        // 2 spliced rows × 2 pbs of fold-factors; genes 10, 20.
        let g = gr(
            DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 0.5, 1.0]),
            vec![10, 20],
        );
        let bo = BatchDivisor::new(&g, &[1usize]); // one cell → pb1
                                                   // backend rows: 0→gene10, 1→gene20, 2→non-gene (divisor 1.0).
        let resid_row = bo.backend_to_resid(&[Some(10), Some(20), None]);
        assert_eq!(resid_row, vec![Some(0), Some(1), None]);

        // Column counts at pb1 ⇒ divisors d = [μ(0,1), μ(1,1), 1.0] = [2, 1, 1].
        // Σx = 20, Σd = 4 ⇒ scale = 5, so xᵢ / (dᵢ·scale) = [0.4, 1.2, 2.0] —
        // matching matrix-util's `adjust_by_division_of_selected_inplace`.
        let rows = [0usize, 1, 2];
        let vals = [4.0f32, 6.0, 10.0];
        let mut got = [0f32; 3];
        bo.divide_column(&resid_row, 1, &rows, &vals, |row, v| got[row] = v);

        let want = [0.4f32, 1.2, 2.0];
        for i in 0..3 {
            assert!(
                (got[i] - want[i]).abs() < 1e-5,
                "row {i}: {} vs {}",
                got[i],
                want[i]
            );
        }
    }
}
