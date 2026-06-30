//! faba-local m6A co-embedding arm for `faba gem`.
//!
//! DART m6A is observed per (gene `g`, cell `c`) as **methylated** reads `M`
//! (the C→T converted counts) and **unmethylated** reads `U` (unconverted);
//! coverage `N = M + U`. We model the methylation rate with a coverage-
//! conditioned binomial whose *negatives are real* (the unmethylated reads), so
//! no noise sampler is needed:
//!
//! ```text
//! z_gc = w_g · e_cell + a_g + b_c       (shares e_cell with expression)
//! ℓ    = ω · [ -(p̂ logσ(z) + (1-p̂) logσ(-z)) ]      p̂ = M/(M+U),  ω = N/(N+N0)
//! ```
//!
//! `w_g` is a **free** per-gene vector (NOT anchored to the expression `β_g`);
//! the only tie to expression is the shared `e_cell`, so `cos(w_g, β_g)` reads
//! out how decoupled a gene's methylation is from its expression. The arm plugs
//! into the generic `graph_embedding_util` seams — [`ge::LossArm`] (phase-1
//! co-training against the shared pb `e_cell`) and [`ge::PerCellAuxTerm`]
//! (phase-2 joint per-cell MAP) — so geu stays modality-agnostic.

use anyhow::Context;
use candle_util::candle_core::{DType, Device, Tensor};
use candle_util::candle_nn::{init, VarBuilder, VarMap};
use candle_util::loss::log_sigmoid;
use graph_embedding_util as ge;
use graph_embedding_util::JointEmbedModel;
use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::weighted::WeightedIndex;
use rand_distr::{Distribution, StandardNormal};
use rustc_hash::FxHashMap;

/// VarMap names for the m6A feature side (registered in the *shared* varmap so
/// the single AdamW trains them alongside `E_feat` / `e_cell`).
const W_VAR: &str = "m6a_w_feat";
const A_VAR: &str = "m6a_a_feat";

/// Per-gene methylated/unmethylated counts for one cell: `(gene_id, M, U)`.
type CellEdges = Vec<(u32, f32, f32)>;

/// Gene-pooled m6A counts, loaded + aligned by barcode. `per_cell[barcode]` is
/// that cell's `(gene_id, M, U)` list (genes with any M or U coverage). Built
/// by [`crate::run_gem_embedding`]'s loader from the converted/unconverted
/// backends; gene ids index `gene_names`.
pub struct M6aData {
    pub gene_names: Vec<Box<str>>,
    pub per_cell: FxHashMap<Box<str>, CellEdges>,
}

impl M6aData {
    pub fn n_genes(&self) -> usize {
        self.gene_names.len()
    }
}

/// Hyperparameters for the binomial arm (CLI-tunable; see `GemArgs`).
#[derive(Clone, Copy)]
pub struct M6aParams {
    /// Phase-1 arm mixing weight λ (how much m6A shapes the shared `e_cell`).
    pub lambda: f32,
    /// Coverage stabilizer in `ω = N/(N+N0)` — down-weights shallow genes/cells.
    pub n0: f32,
    /// Phase-2 per-cell joint-solve weight κ (expression is raw Poisson, so
    /// this typically needs to be ≫ 1 to make m6A visible).
    pub kappa: f64,
    /// Drop pb edges whose pooled coverage `N` is ≤ this.
    pub cov_min: f32,
    /// Per-step minibatch size for the arm's edge sampler.
    pub batch_size: usize,
    /// M4: weight of the m6A block appended to `proj_kn` for joint membership
    /// (0 = off; the partition stays expression-only).
    pub refine_weight: f32,
}

/// Builds the m6A arms (phase 1) and per-cell term (phase 2) for `FitConfig.aux`.
/// Owns the loaded [`M6aData`]; geu invokes it at the two hook points.
pub struct M6aArmBuilder {
    data: M6aData,
    params: M6aParams,
}

impl M6aArmBuilder {
    pub fn new(data: M6aData, params: M6aParams) -> Self {
        Self { data, params }
    }
}

/// Population std of a slice (0 if empty / degenerate).
fn std_of(xs: &[f32]) -> f32 {
    let n = xs.len();
    if n == 0 {
        return 0.0;
    }
    let mean = xs.iter().map(|&x| f64::from(x)).sum::<f64>() / n as f64;
    let var = xs
        .iter()
        .map(|&x| (f64::from(x) - mean).powi(2))
        .sum::<f64>()
        / n as f64;
    var.sqrt() as f32
}

/// Map the unified expression barcodes to their cell ids (`barcode → cell`).
fn barcode_index(barcodes: &[Box<str>]) -> FxHashMap<&str, usize> {
    barcodes
        .iter()
        .enumerate()
        .map(|(i, b)| (b.as_ref(), i))
        .collect()
}

/// Register (or fetch) the shared m6A feature side `w_g [G,H]` (randn) + `a_g [G]`
/// (zeros) in `varmap`, returning live Tensor handles.
fn alloc_feature_side(
    varmap: &VarMap,
    n_genes: usize,
    h: usize,
    dev: &Device,
) -> anyhow::Result<(Tensor, Tensor)> {
    let vs = VarBuilder::from_varmap(varmap, DType::F32, dev);
    let w = vs.get_with_hints(
        (n_genes, h),
        W_VAR,
        init::Init::Randn {
            mean: 0.0,
            stdev: 0.1,
        },
    )?;
    let a = vs.get_with_hints(n_genes, A_VAR, init::Init::Const(0.0))?;
    Ok((w, a))
}

/// Read a trained Var back out of the varmap as a flat `f32` buffer.
fn read_var(varmap: &VarMap, name: &str) -> anyhow::Result<Vec<f32>> {
    let data = varmap.data().lock().unwrap();
    let var = data
        .get(name)
        .with_context(|| format!("m6A var {name} missing from varmap"))?;
    Ok(var.as_tensor().flatten_all()?.to_vec1::<f32>()?)
}

impl ge::AuxArmBuilder for M6aArmBuilder {
    fn augment_projection(
        &mut self,
        ctx: ge::AuxProjCtx,
    ) -> anyhow::Result<Option<nalgebra::DMatrix<f32>>> {
        if self.params.refine_weight <= 0.0 {
            return Ok(None); // M4 off — partition stays expression-only
        }
        let n = ctx.barcodes.len();
        let g = self.data.n_genes();
        let k = ctx.proj.nrows().min(g).max(1);
        if g == 0 {
            return Ok(None);
        }
        let bc_idx = barcode_index(ctx.barcodes);

        // Per-gene mean rate over covered cells — centering removes the gene
        // baseline so the feature is the cell-specific methylation *deviation*.
        let mut sum = vec![0f64; g];
        let mut cnt = vec![0u32; g];
        for (bc, edges) in &self.data.per_cell {
            if !bc_idx.contains_key(bc.as_ref()) {
                continue;
            }
            for &(gene, m, u) in edges {
                let cov = m + u;
                if cov > 0.0 {
                    sum[gene as usize] += f64::from(m / cov);
                    cnt[gene as usize] += 1;
                }
            }
        }
        let gene_mean: Vec<f32> = (0..g)
            .map(|i| {
                if cnt[i] > 0 {
                    (sum[i] / f64::from(cnt[i])) as f32
                } else {
                    0.0
                }
            })
            .collect();

        // Seeded Gaussian projection P [k × G] (row-major: p[j*G + gene]).
        let mut rng = StdRng::seed_from_u64(0x6d36_4153);
        let scale_p = 1.0 / (k as f32).sqrt();
        let mut p = vec![0f32; k * g];
        for v in p.iter_mut() {
            let z: f32 = StandardNormal.sample(&mut rng);
            *v = z * scale_p;
        }

        // proj_m6a [k × n] in nalgebra column-major (col c, row j → c*k + j).
        let mut out = vec![0f32; k * n];
        let n0 = self.params.n0;
        for (bc, edges) in &self.data.per_cell {
            let Some(&c) = bc_idx.get(bc.as_ref()) else {
                continue;
            };
            let base = c * k;
            for &(gene, m, u) in edges {
                let cov = m + u;
                if cov <= 0.0 {
                    continue;
                }
                let omega = cov / (cov + n0);
                let feat = omega * (m / cov - gene_mean[gene as usize]);
                if feat == 0.0 {
                    continue;
                }
                let prow = gene as usize;
                for j in 0..k {
                    out[base + j] += p[j * g + prow] * feat;
                }
            }
        }

        // Scale the block so its row-std = refine_weight × expression row-std.
        let expr_std = std_of(ctx.proj.as_slice());
        let nz: Vec<f32> = out.iter().copied().filter(|&x| x != 0.0).collect();
        let m6a_std = std_of(&nz).max(1e-8);
        let s = self.params.refine_weight * expr_std / m6a_std;
        for v in out.iter_mut() {
            *v *= s;
        }
        let n_cells_cov = self
            .data
            .per_cell
            .keys()
            .filter(|b| bc_idx.contains_key(b.as_ref()))
            .count();
        log::info!(
            "m6A membership (M4): +{k} proj rows, weight {:.2} ({} genes covered, {n_cells_cov} cells)",
            self.params.refine_weight,
            cnt.iter().filter(|&&c| c > 0).count(),
        );
        Ok(Some(nalgebra::DMatrix::from_vec(k, n, out)))
    }

    fn build_arms(&mut self, ctx: ge::AuxArmCtx) -> anyhow::Result<Vec<Box<dyn ge::LossArm>>> {
        let h = ctx.embedding_dim;
        let g = self.data.n_genes();
        let (w_g, a_g) = alloc_feature_side(ctx.varmap, g, h, ctx.device)?;

        // Resolve each m6A cell to its expression cell id once (cells absent
        // from the expression axis are dropped; cells with no m6A simply never
        // appear here).
        let bc_idx = barcode_index(ctx.barcodes);
        let resolved: Vec<(usize, &CellEdges)> = self
            .data
            .per_cell
            .iter()
            .filter_map(|(bc, edges)| bc_idx.get(bc.as_ref()).map(|&c| (c, edges)))
            .collect();
        log::info!(
            "m6A arm: {}/{} m6A cells matched to the expression axis ({} genes)",
            resolved.len(),
            self.data.per_cell.len(),
            g,
        );

        // Both `cell_to_pb_per_level` and `level_pb_models` are coarsest-first,
        // so level `i` lines up directly.
        let mut arms: Vec<Box<dyn ge::LossArm>> = Vec::new();
        for (level, model) in ctx.level_pb_models.iter().enumerate() {
            let cell_to_pb = &ctx.cell_to_pb_per_level[level];
            let n_pb = model.e_cell.dim(0)?;
            // Aggregate (M, U) per (pb, gene) for this level.
            let mut acc: FxHashMap<(u32, u32), (f32, f32)> = FxHashMap::default();
            for &(cell, edges) in &resolved {
                let pb = cell_to_pb[cell] as u32;
                for &(gene, m, u) in edges {
                    let e = acc.entry((pb, gene)).or_insert((0.0, 0.0));
                    e.0 += m;
                    e.1 += u;
                }
            }
            // Flatten to edge arrays, keeping only covered pb edges.
            let (mut pbs, mut genes, mut phat, mut omega) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for ((pb, gene), (m, u)) in acc {
                let n = m + u;
                if n <= self.params.cov_min {
                    continue;
                }
                pbs.push(pb);
                genes.push(gene);
                phat.push(m / n);
                omega.push(n / (n + self.params.n0));
            }
            if genes.is_empty() {
                continue;
            }
            let picker = WeightedIndex::new(&omega)
                .context("m6A arm: degenerate edge weights (all-zero ω)")?;
            arms.push(Box::new(M6aArm {
                w_g: w_g.clone(),
                a_g: a_g.clone(),
                e_cell: model.e_cell.clone(),
                b_cell: model.b_cell.clone(),
                pbs,
                genes,
                phat,
                omega,
                picker,
                lambda: self.params.lambda,
                batch_size: self.params.batch_size,
                label: format!("m6a_l{level}").into_boxed_str(),
            }));
            log::info!("m6A arm level {level}: {n_pb} pb, edges built");
        }
        Ok(arms)
    }

    fn build_cell_term(
        &mut self,
        ctx: ge::AuxCellCtx,
    ) -> anyhow::Result<Option<Box<dyn ge::PerCellAuxTerm>>> {
        let h = ctx.embedding_dim;
        let g = self.data.n_genes();
        let w_g = read_var(ctx.varmap, W_VAR)?;
        let a_g = read_var(ctx.varmap, A_VAR)?;
        anyhow::ensure!(
            w_g.len() == g * h && a_g.len() == g,
            "m6A var shape mismatch"
        );

        // Per-expression-cell edge lists, indexed by cell id. `build_cell_term`
        // is the last hook and the builder is dropped right after, so move the
        // edge vecs out of `self.data.per_cell` rather than cloning.
        let bc_idx = barcode_index(ctx.barcodes);
        let mut per_cell: Vec<CellEdges> = vec![Vec::new(); ctx.n_cells];
        for (bc, edges) in std::mem::take(&mut self.data.per_cell) {
            if let Some(&c) = bc_idx.get(bc.as_ref()) {
                per_cell[c] = edges;
            }
        }
        let n_with = per_cell.iter().filter(|e| !e.is_empty()).count();
        log::info!(
            "m6A phase-2 term: {n_with} cells with m6A edges → joint Poisson+binomial per-cell solve (κ={})",
            self.params.kappa
        );
        Ok(Some(Box::new(M6aCellTerm {
            w_g,
            a_g,
            h,
            per_cell,
            kappa: self.params.kappa,
            n0: f64::from(self.params.n0),
        })))
    }
}

/// One phase-1 m6A loss arm at a single pb level. Scores gene-pooled pb
/// methylation rates against that level's shared pb `e_cell` (a clone whose
/// storage tracks training updates).
struct M6aArm {
    w_g: Tensor,    // [G, H], shared across levels
    a_g: Tensor,    // [G]
    e_cell: Tensor, // [n_pb, H] — this level's shared pb cell embedding
    b_cell: Tensor, // [n_pb]
    pbs: Vec<u32>,
    genes: Vec<u32>,
    phat: Vec<f32>,
    omega: Vec<f32>,
    picker: WeightedIndex<f32>,
    lambda: f32,
    batch_size: usize,
    label: Box<str>,
}

impl ge::LossArm for M6aArm {
    fn step_loss(&self, rng: &mut StdRng, dev: &Device) -> anyhow::Result<Option<Tensor>> {
        let n = self.genes.len();
        if n == 0 {
            return Ok(None);
        }
        let b = self.batch_size.min(n).max(1);
        let (mut g_idx, mut p_idx, mut ph, mut om) = (
            Vec::with_capacity(b),
            Vec::with_capacity(b),
            Vec::with_capacity(b),
            Vec::with_capacity(b),
        );
        for _ in 0..b {
            let e = self.picker.sample(rng);
            g_idx.push(self.genes[e]);
            p_idx.push(self.pbs[e]);
            ph.push(self.phat[e]);
            om.push(self.omega[e]);
        }
        let wsum: f32 = om.iter().sum::<f32>().max(1e-8);
        let g_t = Tensor::from_vec(g_idx, b, dev)?;
        let p_t = Tensor::from_vec(p_idx, b, dev)?;
        let w_b = self.w_g.index_select(&g_t, 0)?; // [b,H]
        let a_b = self.a_g.index_select(&g_t, 0)?; // [b]
        let e_b = self.e_cell.index_select(&p_t, 0)?; // [b,H]
        let bc_b = self.b_cell.index_select(&p_t, 0)?; // [b]
                                                       // z = w·e + a + b_c (same bilinear scorer as the expression arm).
        let z = JointEmbedModel::score_diag(&w_b, &e_b, &a_b, &bc_b)?; // [b]
        let pos = log_sigmoid(&z)?;
        let neg = log_sigmoid(&z.neg()?)?;
        let phat_t = Tensor::from_vec(ph, b, dev)?;
        let om_t = Tensor::from_vec(om, b, dev)?;
        // per_edge = ω · [ -(p̂·logσ(z) + (1-p̂)·logσ(-z)) ]
        let one_minus = phat_t.affine(-1.0, 1.0)?; // 1 - p̂
        let ce = (phat_t.mul(&pos)? + one_minus.mul(&neg)?)?; // [b]
        let per = ce.neg()?.mul(&om_t)?; // [b]
        let loss = (per.sum_all()? / f64::from(wsum))?; // scalar, Σω-normalized
        Ok(Some(loss))
    }

    fn lambda(&self) -> f32 {
        self.lambda
    }

    fn label(&self) -> &str {
        &self.label
    }
}

/// Phase-2 per-cell binomial term: folds the m6A likelihood into a cell's joint
/// MAP solve, sharing `e_c` with the Poisson expression term. `θ` layout is
/// `[e_c (h); b_c_expr (1); b_c_m6a (1)]` — `n_extra == 1` (a dedicated m6A
/// per-cell intercept, NOT the expression library-size bias).
struct M6aCellTerm {
    w_g: Vec<f32>, // [G*H] row-major (frozen, trained)
    a_g: Vec<f32>, // [G]
    h: usize,
    per_cell: Vec<CellEdges>, // indexed by cell id
    kappa: f64,
    n0: f64,
}

impl ge::PerCellAuxTerm for M6aCellTerm {
    fn n_extra(&self) -> usize {
        1 // dedicated b_c^m6a
    }

    fn accumulate(
        &self,
        cell: usize,
        theta: &DVector<f64>,
        h: usize,
        extra_offset: usize,
        grad: &mut DVector<f64>,
        hess: &mut DMatrix<f64>,
    ) {
        debug_assert_eq!(h, self.h, "m6A term embedding dim mismatch");
        let edges = &self.per_cell[cell];
        if edges.is_empty() {
            return;
        }
        let b_m6a = theta[extra_offset];
        for &(gene, m, u) in edges {
            let n = m + u;
            if n <= 0.0 {
                continue;
            }
            let p_hat = f64::from(m / n);
            let omega = f64::from(n) / (f64::from(n) + self.n0);
            let coef = self.kappa * omega;
            let wg = &self.w_g[gene as usize * self.h..(gene as usize + 1) * self.h];
            // z = w_g·e_c + a_g + b_c^m6a
            let mut z = f64::from(self.a_g[gene as usize]) + b_m6a;
            for (k, &wgk) in wg.iter().enumerate() {
                z += theta[k] * f64::from(wgk);
            }
            let sig = 1.0 / (1.0 + (-z).exp());
            let resid = p_hat - sig; // gradient of the CE in z
            let wvar = coef * sig * (1.0 - sig); // GN curvature
                                                 // grad/Hess over w̃ = [w_g (h); 0 (b_expr slot); 1 (b_m6a slot)].
            for (a, &wga) in wg.iter().enumerate() {
                let wga = f64::from(wga);
                grad[a] += coef * resid * wga;
                let row = wvar * wga;
                for (bb, &wgb) in wg.iter().enumerate().skip(a) {
                    hess[(a, bb)] += row * f64::from(wgb);
                }
                hess[(a, extra_offset)] += row; // cross with the b_m6a column (1)
            }
            grad[extra_offset] += coef * resid;
            hess[(extra_offset, extra_offset)] += wvar;
        }
    }

    fn extra_ridge(&self) -> f64 {
        // Weak prior on b_c^m6a so low-coverage cells stay finite.
        1.0
    }
}
