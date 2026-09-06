//! The pseudobulk tree: grow the finest partition from the top nodes by
//! residual bisection to a leaf target per level.
//!
//! The marginal hash (`binary_sort_columns`) signs the leading components of
//! a random sketch of the counts. Those components follow the mass shared
//! within a lineage, so the low `coarse_bits` bits name a lineage node while the
//! remaining high bits split that node along directions that are still
//! mass-driven. A within-node program of modest fold change over a small
//! share of the counts rarely flips such a bit, and the leaves then mix the
//! states the program separates.
//!
//! Here the high bits are rewritten per node. Every cell is standardised
//! against its node's profile (Pearson residuals, per batch), the leading
//! residual component is signed into a left and a right side, residuals are
//! recomputed inside each side against the side's own profile, and the
//! bisection recurses to the requested depth. Every split keeps what
//! differentiates its branches: the loadings of the component that defined
//! it, the weighted left-versus-right log fold change per gene, a strength
//! verdict against the Marchenko-Pastur noise edge, and a two-group Poisson
//! likelihood ratio. The low bits are preserved exactly and the bit layout
//! is unchanged, so every downstream level mask still yields a nested
//! partition.

use super::*;
use crate::nb_dispersion::DispersionTrend;
use nalgebra_sparse::CscMatrix;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use serde::{Deserialize, Serialize};

////////////////
// Parameters //
////////////////

/// Seed for the landmark draw on large nodes.
pub const DEFAULT_RESIDUAL_BITS_SEED: u64 = 0x5245_5349_4442_4954;

/// Genes (by total residual variance) that enter the variance-explained
/// summary of a partition.
const VE_TOP_GENES: usize = 200;

/// What to do with a split whose leading residual component sits at the
/// noise edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BelowEdge {
    /// Apply the sign split anyway and flag it. Keeps the leaf count of the
    /// marginal code, where every split is applied.
    Keep,
    /// Leave the sub-node as a leaf.
    Stop,
}

#[derive(Clone, Debug)]
pub struct PbTreeParams {
    /// A sub-node with fewer cells is a leaf.
    pub min_cells_to_split: usize,
    /// A gene enters a node's residual only if its count over the node's
    /// cells reaches this floor.
    pub min_node_gene_count: f32,
    /// Genes kept per node, by residual variance.
    pub max_genes: usize,
    /// A split carries signal when its leading singular value exceeds the
    /// noise edge by this relative margin.
    pub edge_margin: f32,
    pub below_edge: BelowEdge,
    /// Above this many cells the component is fitted on landmarks.
    pub max_cells_exact: usize,
    pub num_landmarks: usize,
    /// Contrast genes stored per side of a split.
    pub top_contrast_genes: usize,
    pub seed: u64,
    /// Reassign cells between the top nodes by likelihood before the
    /// growth (see `reassign_cells`). `None` keeps the marginal nodes.
    pub reassign_cells: Option<super::ReassignCellsParams>,
}

impl Default for PbTreeParams {
    fn default() -> Self {
        Self {
            min_cells_to_split: 8,
            min_node_gene_count: 5.0,
            max_genes: 2000,
            edge_margin: 0.10,
            below_edge: BelowEdge::Keep,
            max_cells_exact: 512,
            num_landmarks: 512,
            top_contrast_genes: 20,
            seed: DEFAULT_RESIDUAL_BITS_SEED,
            reassign_cells: None,
        }
    }
}

/////////////
// Records //
/////////////

/// One gene's role in a split contrast. `lfc` is right over left.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContrastGene {
    pub gene: usize,
    pub loading: f32,
    pub lfc: f32,
    pub weight: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SplitRecord {
    /// The root (top node) this split lives under.
    pub root: usize,
    /// 1-based depth below the node.
    pub depth: usize,
    /// High bits fixed above this split, i.e. the branch prefix.
    pub path: usize,
    pub n_cells: usize,
    pub n_left: usize,
    pub n_right: usize,
    pub n_genes: usize,
    pub n_left_per_batch: Vec<usize>,
    pub n_right_per_batch: Vec<usize>,
    pub s1: f32,
    pub s2: f32,
    pub sigma: f32,
    pub mp_edge: f32,
    pub passes_edge: bool,
    /// Two-group Poisson log-likelihood ratio of the split against the
    /// merged node, over the node's genes.
    pub llr_split: f64,
    /// Residual variance explained by the split over the random expectation.
    pub ve_ratio: f32,
    /// Whether the split was written into the codes.
    pub applied: bool,
    /// Genes the loadings refer to (not serialised).
    #[serde(skip)]
    pub genes: Vec<usize>,
    /// Loadings of the defining component over `genes` (not serialised).
    #[serde(skip)]
    pub loadings: Vec<f32>,
    /// Genes up on the right side.
    pub pos: Vec<ContrastGene>,
    /// Genes up on the left side.
    pub neg: Vec<ContrastGene>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RootRecord {
    pub root: usize,
    pub n_cells: usize,
    pub n_genes: usize,
    /// Residual variance explained by the residual leaves over the random
    /// expectation, on the node-level residual.
    pub ve_ratio_residual: f32,
    /// The same statistic for the marginal high bits of the same cells.
    pub ve_ratio_marginal: f32,
    pub splits: Vec<SplitRecord>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PbTree {
    /// Width of the coarsest level's field in the packed codes.
    pub coarse_bits: usize,
    pub depth: usize,
    pub num_cells: usize,
    pub num_batches: usize,
    pub edge_margin: f32,
    /// Cells moved between top nodes by the reassignment pass before the
    /// growth (0 when the pass is off).
    pub reassigned_cells: usize,
    pub roots: Vec<RootRecord>,
    /// Finest leaf code to the finest pb ids holding its cells, filled after
    /// refinement.
    pub leaf_to_finest_pb: Vec<(usize, Vec<usize>)>,
}

//////////////////////
// Node-level input //
//////////////////////

/// One coarse node's cells: counts, batch and depth in the node's local
/// column order.
pub(crate) struct NodeBlock {
    /// genes x cells
    pub csc: CscMatrix<f32>,
    pub batch: Vec<usize>,
    pub depth: Vec<f32>,
    pub num_batches: usize,
}

impl NodeBlock {
    pub(crate) fn new(csc: CscMatrix<f32>, batch: Vec<usize>, num_batches: usize) -> Self {
        let depth: Vec<f32> = (0..csc.ncols())
            .map(|j| csc.col(j).values().iter().sum())
            .collect();
        Self {
            csc,
            batch,
            depth,
            num_batches: num_batches.max(1),
        }
    }

    fn ngenes(&self) -> usize {
        self.csc.nrows()
    }
}

/// Per-batch gene rates of a (sub)node, with a pooled fallback for batches
/// that hold too few cells to carry their own profile.
pub(crate) struct NodeProfiles {
    pub pooled: Vec<f64>,
    per_batch: Vec<Vec<f64>>,
    pub batch_depth: Vec<f64>,
    pub node_sum: Vec<f64>,
    node_sum_sq: Vec<f64>,
    pub total_depth: f64,
    pub num_cells: usize,
}

impl NodeProfiles {
    pub(crate) fn new(block: &NodeBlock, local: &[usize], min_cells: usize) -> Self {
        let d = block.ngenes();
        let nb = block.num_batches;
        struct Acc {
            sum_b: Vec<Vec<f64>>,
            depth_b: Vec<f64>,
            cells_b: Vec<usize>,
            sum_sq: Vec<f64>,
        }
        let fresh = || Acc {
            sum_b: vec![vec![0f64; d]; nb],
            depth_b: vec![0f64; nb],
            cells_b: vec![0usize; nb],
            sum_sq: vec![0f64; d],
        };
        let acc = local
            .par_iter()
            .fold(fresh, |mut a, &c| {
                let b = block.batch[c];
                let col = block.csc.col(c);
                for (&g, &v) in col.row_indices().iter().zip(col.values()) {
                    a.sum_b[b][g] += f64::from(v);
                    a.sum_sq[g] += f64::from(v) * f64::from(v);
                }
                a.depth_b[b] += f64::from(block.depth[c]);
                a.cells_b[b] += 1;
                a
            })
            .reduce(fresh, |mut a, o| {
                for (x, y) in a.sum_b.iter_mut().zip(&o.sum_b) {
                    for (p, q) in x.iter_mut().zip(y) {
                        *p += q;
                    }
                }
                for (x, y) in a.depth_b.iter_mut().zip(&o.depth_b) {
                    *x += y;
                }
                for (x, y) in a.cells_b.iter_mut().zip(&o.cells_b) {
                    *x += y;
                }
                for (x, y) in a.sum_sq.iter_mut().zip(&o.sum_sq) {
                    *x += y;
                }
                a
            });
        let Acc {
            sum_b,
            depth_b,
            cells_b,
            sum_sq: node_sum_sq,
        } = acc;
        let node_sum: Vec<f64> = (0..d).map(|g| sum_b.iter().map(|s| s[g]).sum()).collect();
        let total_depth: f64 = depth_b.iter().sum();
        let pooled: Vec<f64> = node_sum
            .iter()
            .map(|&s| {
                if total_depth > 0.0 {
                    s / total_depth
                } else {
                    0.0
                }
            })
            .collect();
        let per_batch: Vec<Vec<f64>> = (0..nb)
            .map(|b| {
                if cells_b[b] >= min_cells && depth_b[b] > 0.0 {
                    sum_b[b].iter().map(|&s| s / depth_b[b]).collect()
                } else {
                    pooled.clone()
                }
            })
            .collect();
        Self {
            pooled,
            per_batch,
            batch_depth: depth_b,
            node_sum,
            node_sum_sq,
            total_depth,
            num_cells: local.len(),
        }
    }

    #[inline]
    fn rate(&self, b: usize, g: usize) -> f64 {
        self.per_batch[b][g]
    }
}

#[inline]
fn clipped(x: f64, clip: f32) -> f32 {
    let c = f64::from(clip);
    x.clamp(-c, c) as f32
}

/// Mean squared Pearson residual per gene over `local`, against `prof`.
/// Nonzero counts are clipped; zero counts contribute their exact
/// expectation.
pub(crate) fn residual_variance(
    block: &NodeBlock,
    local: &[usize],
    prof: &NodeProfiles,
    clip: f32,
) -> Vec<f64> {
    let d = block.ngenes();
    let nb = block.num_batches;
    let fresh = || (vec![0f64; d], vec![vec![0f64; d]; nb]);
    let (sq, nz_depth) = local
        .par_iter()
        .fold(fresh, |(mut sq, mut nz), &c| {
            let b = block.batch[c];
            let n_c = f64::from(block.depth[c]);
            let col = block.csc.col(c);
            for (&g, &v) in col.row_indices().iter().zip(col.values()) {
                let mu = n_c * prof.rate(b, g);
                if mu > 0.0 {
                    let r = f64::from(clipped((f64::from(v) - mu) / mu.sqrt(), clip));
                    sq[g] += r * r;
                    nz[b][g] += n_c;
                }
            }
            (sq, nz)
        })
        .reduce(fresh, |(mut sq, mut nz), (sq2, nz2)| {
            for (x, y) in sq.iter_mut().zip(&sq2) {
                *x += y;
            }
            for (a, b) in nz.iter_mut().zip(&nz2) {
                for (x, y) in a.iter_mut().zip(b) {
                    *x += y;
                }
            }
            (sq, nz)
        });
    let n = local.len().max(1) as f64;
    (0..d)
        .map(|g| {
            let zeros: f64 = (0..nb)
                .map(|b| prof.rate(b, g) * (prof.batch_depth[b] - nz_depth[b][g]).max(0.0))
                .sum();
            (sq[g] + zeros) / n
        })
        .collect()
}

/// Dense Pearson residuals of `rows` over `genes` against `prof`.
pub(crate) fn residual_dense(
    block: &NodeBlock,
    rows: &[usize],
    genes: &[usize],
    prof: &NodeProfiles,
    clip: f32,
) -> DMatrix<f32> {
    let d = block.ngenes();
    let mut gidx = vec![usize::MAX; d];
    for (j, &g) in genes.iter().enumerate() {
        gidx[g] = j;
    }
    let p = genes.len();
    let flat: Vec<Vec<f32>> = rows
        .par_iter()
        .map(|&c| {
            let b = block.batch[c];
            let n_c = f64::from(block.depth[c]);
            let mut row = vec![0f32; p];
            for (j, &g) in genes.iter().enumerate() {
                let mu = n_c * prof.rate(b, g);
                row[j] = if mu > 0.0 {
                    clipped(-mu.sqrt(), clip)
                } else {
                    0.0
                };
            }
            let col = block.csc.col(c);
            for (&g, &v) in col.row_indices().iter().zip(col.values()) {
                let j = gidx[g];
                if j == usize::MAX {
                    continue;
                }
                let mu = n_c * prof.rate(b, g);
                row[j] = if mu > 0.0 {
                    clipped((f64::from(v) - mu) / mu.sqrt(), clip)
                } else {
                    0.0
                };
            }
            row
        })
        .collect();
    DMatrix::<f32>::from_fn(rows.len(), p, |i, j| flat[i][j])
}

/// Projection of every cell's residual row onto `v1`, without forming the
/// dense residual: the zero entries of a cell are handled in closed form.
fn residual_scores(
    block: &NodeBlock,
    local: &[usize],
    genes: &[usize],
    prof: &NodeProfiles,
    v1: &[f32],
    clip: f32,
) -> Vec<f32> {
    let d = block.ngenes();
    let nb = block.num_batches;
    let mut gidx = vec![usize::MAX; d];
    for (j, &g) in genes.iter().enumerate() {
        gidx[g] = j;
    }
    let s_b: Vec<f64> = (0..nb)
        .map(|b| {
            genes
                .iter()
                .enumerate()
                .map(|(j, &g)| prof.rate(b, g).sqrt() * f64::from(v1[j]))
                .sum()
        })
        .collect();
    local
        .par_iter()
        .map(|&c| {
            let b = block.batch[c];
            let n_c = f64::from(block.depth[c]);
            let col = block.csc.col(c);
            let mut acc = 0f64;
            for (&g, &v) in col.row_indices().iter().zip(col.values()) {
                let j = gidx[g];
                if j == usize::MAX {
                    continue;
                }
                let mu = n_c * prof.rate(b, g);
                if mu > 0.0 {
                    let r = f64::from(clipped((f64::from(v) - mu) / mu.sqrt(), clip));
                    acc += (r + mu.sqrt()) * f64::from(v1[j]);
                }
            }
            (acc - n_c.sqrt() * s_b[b]) as f32
        })
        .collect()
}

//////////////////////
// Split statistics //
//////////////////////

/// Largest singular value of an iid noise matrix of the same shape.
pub(crate) fn mp_edge(n: usize, p: usize, sigma2: f64) -> f32 {
    (sigma2.sqrt() * ((n as f64).sqrt() + (p as f64).sqrt())) as f32
}

/// Two-group Poisson profile log-likelihood ratio of splitting into `sum_l`
/// and `sum_r` against the merged composition.
pub(crate) fn two_group_llr(sum_l: &[f64], sum_r: &[f64]) -> f64 {
    let nl: f64 = sum_l.iter().sum();
    let nr: f64 = sum_r.iter().sum();
    if nl <= 0.0 || nr <= 0.0 {
        return 0.0;
    }
    let f = |y: f64, n: f64| if y > 0.0 { y * (y / n).ln() } else { 0.0 };
    sum_l
        .iter()
        .zip(sum_r)
        .map(|(&l, &r)| f(l, nl) + f(r, nr) - f(l + r, nl + nr))
        .sum()
}

/// Variance explained by a partition of the rows of `r`, over the random
/// expectation `(groups - 1) / (rows - 1)`, averaged across the `top_q`
/// columns with the largest total variance. 1.0 for a trivial partition.
pub(crate) fn ve_ratio(r: &DMatrix<f32>, labels: &[usize], top_q: usize) -> f32 {
    let n = r.nrows();
    if n < 2 {
        return 1.0;
    }
    let (compact, k) = crate::dc_poisson::compact_labels(labels);
    if k < 2 {
        return 1.0;
    }
    let expected = (k - 1) as f64 / (n - 1) as f64;
    let mut counts = vec![0f64; k];
    for &z in &compact {
        counts[z] += 1.0;
    }
    let mut per_gene: Vec<(f64, f64)> = (0..r.ncols())
        .into_par_iter()
        .map(|j| {
            let col = r.column(j);
            let mean = col.iter().map(|&x| f64::from(x)).sum::<f64>() / n as f64;
            let ss_total: f64 = col.iter().map(|&x| (f64::from(x) - mean).powi(2)).sum();
            let mut gsum = vec![0f64; k];
            for (i, &x) in col.iter().enumerate() {
                gsum[compact[i]] += f64::from(x);
            }
            let ss_between: f64 = (0..k)
                .filter(|&z| counts[z] > 0.0)
                .map(|z| counts[z] * (gsum[z] / counts[z] - mean).powi(2))
                .sum();
            (ss_total, ss_between)
        })
        .collect();
    per_gene.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let top: Vec<f64> = per_gene
        .iter()
        .take(top_q)
        .filter(|(t, _)| *t > 0.0)
        .map(|(t, b)| b / t / expected)
        .collect();
    if top.is_empty() {
        1.0
    } else {
        (top.iter().sum::<f64>() / top.len() as f64) as f32
    }
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let m = v.len() / 2;
    if v.len() % 2 == 1 {
        v[m]
    } else {
        0.5 * (v[m - 1] + v[m])
    }
}

/// Leading component of `r` (rows x columns): loadings over columns
/// oriented so the largest magnitude loading is positive, plus the top two
/// singular values, from the crate-wide randomised SVD.
fn top_component(r: &DMatrix<f32>) -> Option<(Vec<f32>, f32, f32)> {
    if r.nrows() < 2 || r.ncols() < 2 {
        return None;
    }
    let (_, s, v) = r.rsvd(2).ok()?;
    if s.is_empty() || v.ncols() == 0 {
        return None;
    }
    let mut order: Vec<usize> = (0..s.len()).collect();
    order.sort_by(|&a, &b| s[b].partial_cmp(&s[a]).unwrap_or(std::cmp::Ordering::Equal));
    let s1 = s[order[0]];
    let s2 = order.get(1).map_or(0.0, |&i| s[i]);
    if !s1.is_finite() || s1 <= 0.0 {
        return None;
    }
    let mut v1: Vec<f32> = v.column(order[0]).iter().copied().collect();
    let argmax = v1
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
        .map_or(0, |(i, _)| i);
    if v1[argmax] < 0.0 {
        v1.iter_mut().for_each(|x| *x = -*x);
    }
    Some((v1, s1, s2))
}

///////////////////
// Node handling //
///////////////////

struct SubNode {
    prof: NodeProfiles,
    genes: Vec<usize>,
    sigma2: f64,
}

/// Profile of `local`, gene selection by residual variance, and the robust
/// noise level of the selected residuals.
fn prepare(block: &NodeBlock, local: &[usize], params: &PbTreeParams) -> SubNode {
    let prof = NodeProfiles::new(block, local, params.min_cells_to_split);
    let clip = (local.len() as f32).sqrt();
    let var = residual_variance(block, local, &prof, clip);
    let mut cand: Vec<(usize, f64)> = (0..block.ngenes())
        .filter(|&g| prof.node_sum[g] >= f64::from(params.min_node_gene_count))
        .map(|g| (g, var[g]))
        .collect();
    cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    cand.truncate(params.max_genes);
    cand.sort_by_key(|x| x.0);
    let mut vars: Vec<f64> = cand.iter().map(|x| x.1).collect();
    let sigma2 = median(&mut vars).max(f64::EPSILON);
    SubNode {
        prof,
        genes: cand.into_iter().map(|x| x.0).collect(),
        sigma2,
    }
}

/// Rows the component is fitted on: every cell up to the exact cap, else a
/// seeded landmark subset.
fn svd_rows(local: &[usize], params: &PbTreeParams, seed: u64) -> Vec<usize> {
    if local.len() <= params.max_cells_exact || params.num_landmarks >= local.len() {
        return local.to_vec();
    }
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut pool = local.to_vec();
    let k = params.num_landmarks.max(2);
    for i in 0..k {
        let j = rng.random_range(i..pool.len());
        pool.swap(i, j);
    }
    pool.truncate(k);
    pool
}

fn side_sums(block: &NodeBlock, cells: &[usize], genes: &[usize]) -> Vec<f64> {
    let d = block.ngenes();
    let mut gidx = vec![usize::MAX; d];
    for (j, &g) in genes.iter().enumerate() {
        gidx[g] = j;
    }
    cells
        .par_iter()
        .fold(
            || vec![0f64; genes.len()],
            |mut acc, &c| {
                let col = block.csc.col(c);
                for (&g, &v) in col.row_indices().iter().zip(col.values()) {
                    let j = gidx[g];
                    if j != usize::MAX {
                        acc[j] += f64::from(v);
                    }
                }
                acc
            },
        )
        .reduce(
            || vec![0f64; genes.len()],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    *x += y;
                }
                a
            },
        )
}

/// Contrast genes of a split: the strongest loadings on either side with
/// their post-split log fold change (right over left) and a dispersion-trend
/// weight fitted on the node's own counts.
fn contrast(
    sub: &SubNode,
    loadings: &[f32],
    sum_l: &[f64],
    sum_r: &[f64],
    k: usize,
) -> (Vec<ContrastGene>, Vec<ContrastGene>) {
    let n = sub.prof.num_cells.max(1) as f64;
    let means: Vec<f32> = sub
        .genes
        .iter()
        .map(|&g| (sub.prof.node_sum[g] / n) as f32)
        .collect();
    let vars: Vec<f32> = sub
        .genes
        .iter()
        .zip(&means)
        .map(|(&g, &m)| (sub.prof.node_sum_sq[g] / n - f64::from(m) * f64::from(m)).max(0.0) as f32)
        .collect();
    let trend = DispersionTrend::fit(&means, &vars);
    let total: f64 = sub
        .genes
        .iter()
        .map(|&g| sub.prof.node_sum[g])
        .sum::<f64>()
        .max(1.0);
    let avg_s = (sub.prof.total_depth / n) as f32;
    let nl: f64 = sum_l.iter().sum::<f64>().max(1.0);
    let nr: f64 = sum_r.iter().sum::<f64>().max(1.0);
    let make = |j: usize| ContrastGene {
        gene: sub.genes[j],
        loading: loadings[j],
        lfc: (((sum_r[j] + 0.5) / nr).ln() - ((sum_l[j] + 0.5) / nl).ln()) as f32,
        weight: trend.fisher_weight(
            (sub.prof.node_sum[sub.genes[j]] / total) as f32,
            avg_s,
            means[j],
        ),
    };
    let mut order: Vec<usize> = (0..loadings.len()).collect();
    order.sort_by(|&a, &b| {
        loadings[b]
            .partial_cmp(&loadings[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let pos: Vec<ContrastGene> = order
        .iter()
        .take(k)
        .filter(|&&j| loadings[j] > 0.0)
        .map(|&j| make(j))
        .collect();
    let neg: Vec<ContrastGene> = order
        .iter()
        .rev()
        .take(k)
        .filter(|&&j| loadings[j] < 0.0)
        .map(|&j| make(j))
        .collect();
    (pos, neg)
}

//////////////////////////////
// Frontier: target counts  //
//////////////////////////////

/// One root of the tree: a top-level node with its cells (global ids) and
/// their counts in the same order.
pub(crate) struct RootBlock {
    pub root: usize,
    pub cells: Vec<usize>,
    pub block: NodeBlock,
}

/// A proposed split of one leaf, computed independently of every other leaf.
struct Proposal {
    left: Vec<usize>,
    right: Vec<usize>,
    record: SplitRecord,
    /// Whether the split may be written: both sides populated and the
    /// below-edge policy satisfied.
    applicable: bool,
    /// Ordering key for the budget: the two-group likelihood ratio.
    gain: f64,
}

struct Leaf {
    root_idx: usize,
    /// Block-local cell indices.
    cells: Vec<usize>,
    depth: usize,
    path: usize,
    proposal: Option<Proposal>,
}

/// Split proposal for one leaf; `None` when the leaf is too small or has no
/// usable genes or component.
fn propose(root: &RootBlock, leaf: &Leaf, params: &PbTreeParams) -> Option<Proposal> {
    let block = &root.block;
    let local = &leaf.cells;
    if local.len() < params.min_cells_to_split {
        return None;
    }
    let sub = prepare(block, local, params);
    if sub.genes.len() < 2 {
        return None;
    }
    let clip = (local.len() as f32).sqrt();
    let seed = params.seed
        ^ (root.root as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ ((leaf.depth as u64) << 48)
        ^ (leaf.path as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    let rows = svd_rows(local, params, seed);
    let r = residual_dense(block, &rows, &sub.genes, &sub.prof, clip);
    let (v1, s1, s2) = top_component(&r)?;
    let scores = residual_scores(block, local, &sub.genes, &sub.prof, &v1, clip);
    let mut right: Vec<usize> = Vec::new();
    let mut left: Vec<usize> = Vec::new();
    for (&c, &s) in local.iter().zip(&scores) {
        if s > 0.0 {
            right.push(c);
        } else {
            left.push(c);
        }
    }
    let edge = mp_edge(rows.len(), sub.genes.len(), sub.sigma2);
    let min_side = (params.min_cells_to_split / 2).max(1);
    let balanced = left.len().min(right.len()) >= min_side;
    let passes = balanced && s1 > (1.0 + params.edge_margin) * edge;
    let applicable = balanced && (passes || params.below_edge == BelowEdge::Keep);
    let sum_l = side_sums(block, &left, &sub.genes);
    let sum_r = side_sums(block, &right, &sub.genes);
    let nb = block.num_batches;
    let per_batch = |cells: &[usize]| {
        let mut v = vec![0usize; nb];
        for &c in cells {
            v[block.batch[c]] += 1;
        }
        v
    };
    let mut side_of_local = vec![0usize; block.csc.ncols()];
    for &c in &right {
        side_of_local[c] = 1;
    }
    let row_side: Vec<usize> = rows.iter().map(|&c| side_of_local[c]).collect();
    let llr = two_group_llr(&sum_l, &sum_r);
    let (pos, neg) = contrast(&sub, &v1, &sum_l, &sum_r, params.top_contrast_genes);
    let record = SplitRecord {
        root: root.root,
        depth: leaf.depth + 1,
        path: leaf.path,
        n_cells: local.len(),
        n_left: left.len(),
        n_right: right.len(),
        n_genes: sub.genes.len(),
        n_left_per_batch: per_batch(&left),
        n_right_per_batch: per_batch(&right),
        s1,
        s2,
        sigma: sub.sigma2.sqrt() as f32,
        mp_edge: edge,
        passes_edge: passes,
        llr_split: llr,
        ve_ratio: ve_ratio(&r, &row_side, VE_TOP_GENES),
        applied: false,
        genes: sub.genes,
        loadings: v1,
        pos,
        neg,
    };
    Some(Proposal {
        left,
        right,
        record,
        applicable,
        gain: llr,
    })
}

/// The nested partitions produced by [`bisect_to_target`], coarse to fine,
/// plus the recorded tree.
pub(crate) struct TargetTree {
    /// Per level (coarse first), a leaf label per global cell;
    /// `usize::MAX` for cells that belong to no root.
    pub labels_per_level: Vec<Vec<usize>>,
    pub tree: PbTree,
}

/// Grow the tree from `roots` until each of `level_targets` (coarse to
/// fine, leaf counts) is met, snapshotting the partition at every target.
///
/// Each round proposes a split for every leaf in parallel, then applies
/// proposals in decreasing evidence until the next target is reached. A
/// leaf below the cell floor, or one whose split fails the below-edge
/// policy, stays a leaf; when no leaf can split, the remaining levels
/// repeat the last partition. Deterministic given the seed, independent of
/// the thread count.
pub(crate) fn bisect_to_target(
    roots: &[RootBlock],
    num_cells: usize,
    level_targets: &[usize],
    params: &PbTreeParams,
) -> TargetTree {
    let mut leaves: Vec<Leaf> = roots
        .iter()
        .enumerate()
        .map(|(i, r)| Leaf {
            root_idx: i,
            cells: (0..r.cells.len()).collect(),
            depth: 0,
            path: 0,
            proposal: None,
        })
        .collect();
    let mut label: Vec<usize> = vec![usize::MAX; num_cells];
    for (i, r) in roots.iter().enumerate() {
        for &c in &r.cells {
            label[c] = i;
        }
    }
    leaves.par_iter_mut().for_each(|leaf| {
        leaf.proposal = propose(&roots[leaf.root_idx], leaf, params);
    });
    let mut splits: Vec<Vec<SplitRecord>> = vec![Vec::new(); roots.len()];
    let mut levels: Vec<Vec<usize>> = Vec::with_capacity(level_targets.len());
    let mut next = 0usize;
    let mut alive: Vec<bool> = vec![true; leaves.len()];
    // One bar over the whole growth: leaves reached against the finest target.
    let final_target = level_targets.last().copied().unwrap_or(0) as u64;
    let prog_bar =
        matrix_util::progress::new_progress_bar(final_target).with_message("pb tree leaves");
    prog_bar.set_position(leaves.len() as u64);

    loop {
        let n_leaves = alive.iter().filter(|&&a| a).count();
        while next < level_targets.len() && n_leaves >= level_targets[next] {
            levels.push(label.clone());
            next += 1;
        }
        if next >= level_targets.len() {
            break;
        }
        let mut cand: Vec<(f64, usize)> = leaves
            .iter()
            .enumerate()
            .filter(|(i, l)| alive[*i] && l.proposal.as_ref().is_some_and(|p| p.applicable))
            .map(|(i, l)| (l.proposal.as_ref().map_or(0.0, |p| p.gain), i))
            .collect();
        if cand.is_empty() {
            while next < level_targets.len() {
                levels.push(label.clone());
                next += 1;
            }
            break;
        }
        cand.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        let budget = level_targets[next] - n_leaves;
        prog_bar.set_message(format!(
            "pb tree leaves, level {}/{} toward {}",
            next + 1,
            level_targets.len(),
            level_targets[next]
        ));
        let mut fresh: Vec<usize> = Vec::new();
        for &(_, i) in cand.iter().take(budget) {
            let Leaf {
                root_idx,
                depth,
                path,
                proposal,
                ..
            } = std::mem::replace(
                &mut leaves[i],
                Leaf {
                    root_idx: 0,
                    cells: Vec::new(),
                    depth: 0,
                    path: 0,
                    proposal: None,
                },
            );
            let mut p = proposal.expect("candidate has a proposal");
            p.record.applied = true;
            splits[root_idx].push(p.record);
            alive[i] = false;
            let root = &roots[root_idx];
            for (side, cells) in [(0usize, p.left), (1usize, p.right)] {
                let id = leaves.len();
                for &c in &cells {
                    label[root.cells[c]] = id;
                }
                leaves.push(Leaf {
                    root_idx,
                    cells,
                    depth: depth + 1,
                    path: (path << 1) | side,
                    proposal: None,
                });
                alive.push(true);
                fresh.push(id);
            }
            prog_bar.inc(1);
        }
        let mut new_leaves: Vec<(usize, Leaf)> = fresh
            .iter()
            .map(|&id| {
                (
                    id,
                    std::mem::replace(
                        &mut leaves[id],
                        Leaf {
                            root_idx: 0,
                            cells: Vec::new(),
                            depth: 0,
                            path: 0,
                            proposal: None,
                        },
                    ),
                )
            })
            .collect();
        new_leaves.par_iter_mut().for_each(|(_, leaf)| {
            leaf.proposal = propose(&roots[leaf.root_idx], leaf, params);
        });
        for (id, leaf) in new_leaves {
            leaves[id] = leaf;
        }
    }

    prog_bar.finish_and_clear();

    // Proposals that were never applied are still worth reading.
    for (i, leaf) in leaves.iter_mut().enumerate() {
        if alive[i] {
            if let Some(p) = leaf.proposal.take() {
                splits[leaf.root_idx].push(p.record);
            }
        }
    }
    let max_depth = splits
        .iter()
        .flatten()
        .filter(|s| s.applied)
        .map(|s| s.depth)
        .max()
        .unwrap_or(0);
    let num_batches = roots.first().map_or(1, |r| r.block.num_batches);
    let nodes: Vec<RootRecord> = roots
        .iter()
        .zip(splits)
        .map(|(r, mut sp)| {
            sp.sort_by_key(|s| (s.depth, s.path));
            RootRecord {
                root: r.root,
                n_cells: r.cells.len(),
                n_genes: sp.first().map_or(0, |s| s.n_genes),
                ve_ratio_residual: 1.0,
                ve_ratio_marginal: 1.0,
                splits: sp,
            }
        })
        .collect();
    let tree = PbTree {
        coarse_bits: 0,
        depth: max_depth,
        num_cells,
        num_batches,
        edge_margin: params.edge_margin,
        reassigned_cells: 0,
        roots: nodes,
        leaf_to_finest_pb: Vec::new(),
    };
    TargetTree {
        labels_per_level: levels,
        tree,
    }
}

/// Variance explained by two partitions of a root's cells on the root-level
/// residual: the tree's finest labels and a reference labelling (the
/// marginal code). Written into the node records.
pub(crate) fn node_ve_ratios(
    roots: &[RootBlock],
    finest: &[usize],
    reference: &[usize],
    params: &PbTreeParams,
    tree: &mut PbTree,
) {
    let ratios: Vec<(f32, f32)> = roots
        .par_iter()
        .map(|r| {
            if r.cells.len() < params.min_cells_to_split {
                return (1.0, 1.0);
            }
            let local: Vec<usize> = (0..r.cells.len()).collect();
            let sub = prepare(&r.block, &local, params);
            if sub.genes.len() < 2 {
                return (1.0, 1.0);
            }
            let rows = svd_rows(&local, params, params.seed ^ r.root as u64);
            let m = residual_dense(
                &r.block,
                &rows,
                &sub.genes,
                &sub.prof,
                (local.len() as f32).sqrt(),
            );
            let a: Vec<usize> = rows.iter().map(|&c| finest[r.cells[c]]).collect();
            let b: Vec<usize> = rows.iter().map(|&c| reference[r.cells[c]]).collect();
            (
                ve_ratio(&m, &a, VE_TOP_GENES),
                ve_ratio(&m, &b, VE_TOP_GENES),
            )
        })
        .collect();
    for (node, (res, marg)) in tree.roots.iter_mut().zip(ratios) {
        node.ve_ratio_residual = res;
        node.ve_ratio_marginal = marg;
    }
}

/////////////////////
// Prefix packing  //
/////////////////////

fn bits_for(k: usize) -> usize {
    (usize::BITS as usize - k.max(2).saturating_sub(1).leading_zeros() as usize).max(1)
}

/// Pack nested partitions (coarse to fine) into one code per cell whose low
/// bits name the coarsest group and each further field the child index
/// within the parent, so masking the code to a level's width reproduces
/// that level's partition. Returns the codes and the level widths,
/// finest-first as the collapse expects. Cells labelled `usize::MAX` get 0.
pub(crate) fn pack_levels(levels: &[Vec<usize>]) -> (Vec<usize>, Vec<usize>) {
    let n = levels.first().map_or(0, Vec::len);
    let mut codes = vec![0usize; n];
    let mut widths: Vec<usize> = Vec::with_capacity(levels.len());
    let mut shift = 0usize;
    let mut parent_codes: Vec<usize> = vec![0; n];
    for (li, level) in levels.iter().enumerate() {
        let active: Vec<usize> = level
            .iter()
            .map(|&l| if l == usize::MAX { 0 } else { l })
            .collect();
        let field: Vec<usize>;
        let width: usize;
        if li == 0 {
            let (compact, k) = crate::dc_poisson::compact_labels(&active);
            width = bits_for(k);
            field = compact;
        } else {
            // child index of each fine group within its parent code
            let mut child_index: HashMap<(usize, usize), usize> = HashMap::default();
            let mut children_of: HashMap<usize, usize> = HashMap::default();
            let mut idx = vec![0usize; n];
            for c in 0..n {
                let key = (parent_codes[c], active[c]);
                let next = children_of.entry(parent_codes[c]).or_insert(0);
                let e = child_index.entry(key).or_insert_with(|| {
                    let v = *next;
                    *next += 1;
                    v
                });
                idx[c] = *e;
            }
            let max_children = children_of.values().copied().max().unwrap_or(1);
            width = bits_for(max_children);
            field = idx;
        }
        for c in 0..n {
            codes[c] |= field[c] << shift;
        }
        shift += width;
        widths.push(shift);
        parent_codes.copy_from_slice(&codes);
    }
    widths.reverse();
    (codes, widths)
}

/////////////////////////
// Driver over a store //
/////////////////////////

/// Build the tree over `data_vec` from a top-level partition `node_of_cell`
/// (`usize::MAX` = not a root member) to the leaf targets, and return the
/// packed codes, their level widths (finest-first) and the tree.
pub(crate) fn build_tree(
    data_vec: &SparseIoVec,
    node_of_cell: &[usize],
    reference: &[usize],
    level_targets: &[usize],
    params: &PbTreeParams,
) -> anyhow::Result<(Vec<usize>, Vec<usize>, PbTree)> {
    let n = data_vec.num_columns();
    anyhow::ensure!(
        node_of_cell.len() == n,
        "node labels for {} of {n} cells",
        node_of_cell.len()
    );
    let col_to_batch = data_vec.get_batch_membership(0..n);
    let num_batches = data_vec.num_batches().max(1);
    let mut by_node: HashMap<usize, Vec<usize>> = HashMap::default();
    for (c, &nd) in node_of_cell.iter().enumerate() {
        if nd != usize::MAX {
            by_node.entry(nd).or_default().push(c);
        }
    }
    let mut nodes: Vec<(usize, Vec<usize>)> = by_node.into_iter().collect();
    nodes.sort_by_key(|x| x.0);
    let roots: Vec<RootBlock> = nodes
        .into_par_iter()
        .map(|(root, cells)| {
            let csc = data_vec.read_columns_csc(cells.iter().copied())?;
            let batch: Vec<usize> = cells.iter().map(|&c| col_to_batch[c]).collect();
            Ok(RootBlock {
                root,
                cells,
                block: NodeBlock::new(csc, batch, num_batches),
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let mut out = bisect_to_target(&roots, n, level_targets, params);
    let finest = out
        .labels_per_level
        .last()
        .cloned()
        .unwrap_or_else(|| vec![usize::MAX; n]);
    node_ve_ratios(&roots, &finest, reference, params, &mut out.tree);
    let (codes, widths) = pack_levels(&out.labels_per_level);
    out.tree.coarse_bits = *widths.last().unwrap_or(&0);
    log_tree(&out.tree, level_targets, &out.labels_per_level);
    Ok((codes, widths, out.tree))
}

fn log_tree(tree: &PbTree, targets: &[usize], levels: &[Vec<usize>]) {
    let mut per_depth: Vec<(usize, usize)> = vec![(0, 0); tree.depth];
    let mut ratios: Vec<f64> = Vec::new();
    let mut ve_res: Vec<f64> = Vec::new();
    let mut ve_marg: Vec<f64> = Vec::new();
    let mut applied = 0usize;
    let mut total = 0usize;
    for node in &tree.roots {
        ve_res.push(f64::from(node.ve_ratio_residual));
        ve_marg.push(f64::from(node.ve_ratio_marginal));
        for s in &node.splits {
            total += 1;
            applied += usize::from(s.applied);
            if s.applied && s.depth >= 1 && s.depth <= tree.depth {
                per_depth[s.depth - 1].1 += 1;
                per_depth[s.depth - 1].0 += usize::from(s.passes_edge);
            }
            if s.mp_edge > 0.0 {
                ratios.push(f64::from(s.s1 / s.mp_edge));
            }
        }
    }
    let by_depth: Vec<String> = per_depth
        .iter()
        .enumerate()
        .map(|(d, (a, n))| format!("d{} {}/{}", d + 1, a, n))
        .collect();
    let reached: Vec<usize> = levels
        .iter()
        .map(|l| {
            l.iter()
                .filter(|&&x| x != usize::MAX)
                .collect::<std::collections::HashSet<_>>()
                .len()
        })
        .collect();
    info!(
        "pb tree: {} roots, max depth {}, {} splits proposed, {} applied; leaves per level {:?} for targets {:?}; \
         applied splits above the noise edge by depth: {}; median s1/edge {:.2}; \
         variance-explained ratio residual {:.2} vs marginal {:.2} (median over roots, 1.0 = random)",
        tree.roots.len(),
        tree.depth,
        total,
        applied,
        reached,
        targets,
        by_depth.join(", "),
        median(&mut ratios),
        median(&mut ve_res),
        median(&mut ve_marg),
    );
}

#[cfg(test)]
#[path = "pb_tree_tests.rs"]
mod tests;
