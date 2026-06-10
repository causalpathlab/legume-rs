//! Marker-set cell-type annotation by projection onto a **frozen feature
//! embedding** — model-agnostic across `senna bge`, `faba gem`, `pinto
//! cage`, and (via its adapter) topic models.
//!
//! The bipartite/freeze-feature pattern these embeddings share: a cell type
//! is just another node defined by its marker genes, so embed it in the same
//! feature space the cells live in, then annotate by a cosine score there:
//!
//! ```text
//! e_T = L2-normalize( Σ_{f ∈ markers(T)} w_f · feature_emb[f] )   (signature direction)
//! score(c, T) = ⟨ ê_cell[c], e_T ⟩                                (cosine)
//! posterior(c, ·) = softmax_T score(c, ·) / temperature
//! ```
//!
//! **Why the marker centroid (not a Poisson-MAP projection).** The cells were
//! placed by fitting *graded* counts across many features, which pins their
//! direction. A marker set is presence-style and roughly flat, so a per-node
//! Poisson-MAP would let the free intercept absorb the level and the ridge
//! shrink the direction to noise — it degenerates. The weighted centroid of
//! the marker feature embeddings is the natural, degeneracy-free operator,
//! and it's exactly comparable to `e_cell`: because the model scores cell `c`
//! on gene `f` by `⟨e_f, e_cell⟩`, a cell expressing gene `f` has `e_cell`
//! aligned with `e_f` — so the marker-β centroid *is* the direction that
//! type's cells occupy. Because `e_cell` is direction-only (depth lives in a
//! discarded per-cell bias), the score is depth- and batch-invariant.
//!
//! **Significance (z-score).** A raw cosine is a similarity, not a tested
//! enrichment. With `n_perm > 0` we calibrate each cell→type affinity against
//! random gene sets of the *same size and weight multiset*: draw `n_perm`
//! permuted "types" per real type (gene identities shuffled, weights kept),
//! form their centroid signatures, and standardize the observed cosine
//! against the **moments** of the null cosines: `z = (obs − μ) / σ`. The null
//! cosines are ~Gaussian (a fixed unit cell vs near-random unit signatures in
//! high dimension), so a normal upper-tail p (`p = ½ erfc(z/√2) = pnorm(-z)`)
//! is well-calibrated and — unlike an empirical rank — has no `1/(n_perm+1)`
//! floor. We report the continuous `z`; downstream takes the normal tail.

use anyhow::{Context, Result};
use data_beans::utilities::name_matching::{idf_weight, GeneIndex};
use log::info;
use matrix_util::common_io::{read_lines_of_words_delim, ReadLinesOut};
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::traits::IoOps;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

/// Per-type `(feature_index, weight)` marker lists — one inner `Vec` per type.
type MarkerSets = Vec<Vec<(u32, f32)>>;

/// Tunables for the annotation routines.
pub struct AnnotateProjConfig {
    /// Softmax temperature for the per-cell posterior (lower → sharper).
    pub temperature: f32,
    /// Permutation draws per type for the null (0 disables z-scores).
    pub n_perm: usize,
    /// Deterministic RNG seed for the permutation null.
    pub seed: u64,
}

impl Default for AnnotateProjConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            n_perm: 200,
            seed: 42,
        }
    }
}

/// Result of [`annotate_by_projection`]. All matrices are row-major.
pub struct AnnotateProjOutputs {
    pub n_cells: usize,
    pub n_types: usize,
    /// `[N × C]` per-cell softmax posterior over types.
    pub posterior_nc: Vec<f32>,
    /// `[C × H]` L2-normalized type signature embeddings (drop-in plot anchors).
    pub type_emb_ch: Vec<f32>,
    /// Per cell: `(argmax type index, posterior at that type)`.
    pub argmax: Vec<(usize, f32)>,
    /// `[N × C]` null-standardized z-scores `(obs − μ_null) / σ_null`, or
    /// `None` when `n_perm == 0`. The continuous significance score; the
    /// normal upper tail `pnorm(-z)` recovers a p-value with no empirical floor.
    pub zscore_nc: Option<Vec<f32>>,
}

//////////////////////////////
// High-level entry point
//////////////////////////////

/// End-to-end annotation from in-memory embeddings: parse + match the marker
/// TSV against `gene_names`, project every type, score every cell, and write
/// `{out_prefix}.{posterior,zscore,type_embedding}.parquet`. The thin per-tool
/// adapters (gem / cage / bge) only load the two embedding matrices from their
/// own manifest and call this.
///
/// * `feature_emb` `[G × H]`, `gene_names` len `G`.
/// * `cell_emb` `[N × H]`, `cell_names` len `N`.
/// * `out_prefix` — full prefix incl. tool infix, e.g. `…/run.gem_annot`.
#[allow(clippy::too_many_arguments)]
pub fn annotate_embeddings(
    feature_emb: &DMatrix<f32>,
    gene_names: &[Box<str>],
    cell_emb: &DMatrix<f32>,
    cell_names: &[Box<str>],
    markers_path: &str,
    out_prefix: &str,
    use_idf: bool,
    cfg: &AnnotateProjConfig,
) -> Result<()> {
    let g = feature_emb.nrows();
    let h = feature_emb.ncols();
    let n = cell_emb.nrows();
    anyhow::ensure!(
        cell_emb.ncols() == h,
        "embedding dim mismatch: features H={h}, cells H={}",
        cell_emb.ncols()
    );
    anyhow::ensure!(gene_names.len() == g, "gene_names len != feature rows");
    anyhow::ensure!(cell_names.len() == n, "cell_names len != cell rows");
    info!("annotate: β [{g} × {h}], cells [{n} × {h}]");

    let (type_names, type_markers) = parse_and_match_markers(markers_path, gene_names, use_idf)?;
    anyhow::ensure!(
        type_names.len() >= 2,
        "need ≥ 2 cell types with matched markers, found {}",
        type_names.len()
    );
    let matched: usize = type_markers.iter().map(Vec::len).sum();
    info!(
        "markers: {} types, {} matched (gene, type) entries",
        type_names.len(),
        matched
    );

    let beta_flat = row_major(feature_emb);
    let cell_flat = row_major(cell_emb);
    let res = annotate_by_projection(&beta_flat, g, &cell_flat, n, &type_markers, h, cfg);

    write_annotation_outputs(out_prefix, cell_names, &type_names, h, &res)?;
    log_label_histogram(&type_names, &res.argmax, n);
    Ok(())
}

//////////////////////////////
// Core compute (pure, slice-based)
//////////////////////////////

/// Annotate cells against marker-defined types by signature projection.
///
/// * `feature_emb` — row-major `[n_features × h]` frozen feature embedding.
/// * `cell_emb` — row-major `[n_cells × h]` cell embedding (normalized
///   defensively here, so an un-normalized input still works).
/// * `type_markers[t]` — type `t`'s `(feature_index, weight)` list.
pub fn annotate_by_projection(
    feature_emb: &[f32],
    n_features: usize,
    cell_emb: &[f32],
    n_cells: usize,
    type_markers: &[Vec<(u32, f32)>],
    h: usize,
    cfg: &AnnotateProjConfig,
) -> AnnotateProjOutputs {
    let n_types = type_markers.len();

    // 1. Type signatures = L2-normalized weighted marker centroids.
    let type_emb_ch = type_signatures(feature_emb, n_features, type_markers, h);

    // 2. Unit-normalize the cell embedding (gem/bge already are; cheap guard).
    let mut cell_u = cell_emb.to_vec();
    l2_normalize_rows(&mut cell_u, n_cells, h);

    // 3. Cosine score + softmax posterior + argmax, per cell (parallel rows).
    let inv_temp = 1.0 / cfg.temperature.max(1e-6);
    let mut posterior_nc = vec![0f32; n_cells * n_types];
    let argmax: Vec<(usize, f32)> = posterior_nc
        .par_chunks_mut(n_types.max(1))
        .enumerate()
        .map(|(n, post)| {
            let cu = &cell_u[n * h..(n + 1) * h];
            for c in 0..n_types {
                post[c] = dot(cu, &type_emb_ch[c * h..(c + 1) * h]);
            }
            softmax_inplace(post, inv_temp);
            let mut best = 0usize;
            for c in 1..n_types {
                if post[c] > post[best] {
                    best = c;
                }
            }
            (best, post.get(best).copied().unwrap_or(0.0))
        })
        .collect();

    // 4. Permutation null → per-cell, per-type z-scores.
    let zscore_nc = (cfg.n_perm > 0 && n_types > 0).then(|| {
        permutation_zscores(
            feature_emb,
            n_features,
            &cell_u,
            n_cells,
            type_markers,
            &type_emb_ch,
            h,
            cfg,
        )
    });

    AnnotateProjOutputs {
        n_cells,
        n_types,
        posterior_nc,
        type_emb_ch,
        argmax,
        zscore_nc,
    }
}

/// `[C × H]` L2-normalized weighted centroid of each type's marker feature
/// embeddings (parallel over types). Empty types get a zero row.
fn type_signatures(
    feature_emb: &[f32],
    n_features: usize,
    type_markers: &[Vec<(u32, f32)>],
    h: usize,
) -> Vec<f32> {
    let n_types = type_markers.len();
    let mut out = vec![0f32; n_types * h];
    out.par_chunks_mut(h)
        .zip(type_markers.par_iter())
        .for_each(|(row, markers)| {
            for &(gi, w) in markers {
                let gi = gi as usize;
                if gi >= n_features {
                    continue;
                }
                let ef = &feature_emb[gi * h..(gi + 1) * h];
                for (r, &e) in row.iter_mut().zip(ef) {
                    *r += w * e;
                }
            }
            let nrm = row.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if nrm > 0.0 {
                for v in row.iter_mut() {
                    *v /= nrm;
                }
            }
        });
    out
}

/// `[N × C]` null-standardized z-scores. For each type we draw `n_perm`
/// random gene sets of the same size (gene identity shuffled, weights kept);
/// each draw's normalized centroid is a null signature. A cell's observed
/// cosine is standardized against the moments of its `n_perm` null cosines.
#[allow(clippy::too_many_arguments)]
fn permutation_zscores(
    feature_emb: &[f32],
    n_features: usize,
    cell_u: &[f32],
    n_cells: usize,
    type_markers: &[Vec<(u32, f32)>],
    type_emb_ch: &[f32],
    h: usize,
    cfg: &AnnotateProjConfig,
) -> Vec<f32> {
    let n_types = type_markers.len();
    let n_perm = cfg.n_perm;

    // C·n_perm null signatures, built in parallel. Each (type, perm) is a
    // deterministic seeded draw; the random gene set is accumulated straight
    // into its centroid row — no intermediate marker-list is materialized.
    let mut null_emb = vec![0f32; n_types * n_perm * h];
    null_emb
        .par_chunks_mut(h)
        .enumerate()
        .for_each(|(idx, row)| {
            let (t, p) = (idx / n_perm, idx % n_perm);
            let markers = &type_markers[t];
            let m = markers.len().min(n_features);
            let mut rng = SmallRng::seed_from_u64(
                cfg.seed ^ ((t as u64) << 32) ^ (p as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            );
            let drawn = rand::seq::index::sample(&mut rng, n_features, m).into_vec();
            for (&gidx, &(_, w)) in drawn.iter().zip(markers.iter()) {
                let ef = &feature_emb[gidx * h..(gidx + 1) * h];
                for (r, &e) in row.iter_mut().zip(ef) {
                    *r += w * e;
                }
            }
            let nrm = row.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if nrm > 0.0 {
                for v in row.iter_mut() {
                    *v /= nrm;
                }
            }
        });

    // Per cell (parallel): standardize observed cosine against null moments.
    let mut zscore_nc = vec![0f32; n_cells * n_types];
    zscore_nc
        .par_chunks_mut(n_types)
        .enumerate()
        .for_each(|(n, zr)| {
            let cu = &cell_u[n * h..(n + 1) * h];
            for t in 0..n_types {
                let obs = dot(cu, &type_emb_ch[t * h..(t + 1) * h]) as f64;
                // Online mean/variance over the n_perm null cosines.
                let (mut mean, mut m2) = (0f64, 0f64);
                for p in 0..n_perm {
                    let v = t * n_perm + p;
                    let s = dot(cu, &null_emb[v * h..(v + 1) * h]) as f64;
                    let delta = s - mean;
                    mean += delta / (p as f64 + 1.0);
                    m2 += delta * (s - mean);
                }
                let sd = (m2 / (n_perm as f64).max(1.0)).sqrt().max(1e-6);
                zr[t] = ((obs - mean) / sd) as f32;
            }
        });
    zscore_nc
}

//////////////////////////////
// Marker parsing + matching
//////////////////////////////

/// Parse a marker TSV and match its genes to `gene_names`, returning the
/// sorted type vocabulary and per-type `(gene_index, weight)` lists.
///
/// Matching is exact-first (lowercased full name or its last `_`-segment
/// symbol), falling back to the shared `flexible_name_match`. Weights are IDF
/// — `ln(C / df_gene)` — unless `use_idf` is false (then unit), so markers
/// shared across many types are down-weighted (a ubiquitous gene → IDF 0,
/// dropped so it can't anchor a type).
fn parse_and_match_markers(
    markers_path: &str,
    gene_names: &[Box<str>],
    use_idf: bool,
) -> Result<(Vec<Box<str>>, MarkerSets)> {
    let pairs = read_marker_tsv(markers_path)?;

    // Type vocabulary (sorted, stable).
    let mut type_names: Vec<Box<str>> = pairs.iter().map(|(_, t)| t.clone()).collect();
    type_names.sort();
    type_names.dedup();
    let type_idx: FxHashMap<&str, usize> = type_names
        .iter()
        .enumerate()
        .map(|(i, t)| (t.as_ref(), i))
        .collect();

    // Shared three-tier gene matcher (exact → symbol → flexible fallback).
    let index = GeneIndex::build(gene_names);

    let c = type_names.len();
    let mut membership: Vec<FxHashSet<u32>> = vec![FxHashSet::default(); c];
    // Memoize matches by gene string: the same gene recurs across types, and
    // the flexible fallback is an O(genes) scan — resolve each distinct gene
    // at most once.
    let mut match_cache: FxHashMap<&str, Option<usize>> = FxHashMap::default();
    let mut unmatched = 0usize;
    for (gene, ct) in &pairs {
        let Some(&ti) = type_idx.get(ct.as_ref()) else {
            continue;
        };
        match *match_cache
            .entry(gene.as_ref())
            .or_insert_with(|| index.match_gene(gene))
        {
            Some(gi) => {
                membership[ti].insert(gi as u32);
            }
            None => unmatched += 1,
        }
    }
    if unmatched > 0 {
        info!("{unmatched} marker pairs had no matching gene (dropped)");
    }

    // IDF: df_gene = #types containing the gene.
    let mut df: FxHashMap<u32, usize> = FxHashMap::default();
    for genes in &membership {
        for &gi in genes {
            *df.entry(gi).or_insert(0) += 1;
        }
    }
    let use_idf = use_idf && c >= 2;
    let type_markers: Vec<Vec<(u32, f32)>> = membership
        .iter()
        .map(|genes| {
            genes
                .iter()
                .filter_map(|&gi| {
                    if use_idf {
                        let w = idf_weight(c, df.get(&gi).copied().unwrap_or(1));
                        (w > 0.0).then_some((gi, w)) // drop ubiquitous (IDF 0)
                    } else {
                        Some((gi, 1.0))
                    }
                })
                .collect()
        })
        .collect();

    Ok((type_names, type_markers))
}

/// Parse a marker TSV/CSV into `(gene, celltype)` pairs via the shared,
/// gz-aware `read_lines_of_words_delim` (tab/comma — matching senna's marker
/// reader). Takes the first two tokens per line, skips a `gene`/`symbol`
/// header and `#` comments, and maps spaces in cell-type names → `_`.
fn read_marker_tsv(path: &str) -> Result<Vec<(Box<str>, Box<str>)>> {
    let ReadLinesOut { lines, .. } = read_lines_of_words_delim(path, &['\t', ','][..], -1)
        .with_context(|| format!("reading markers {path}"))?;
    let out: Vec<(Box<str>, Box<str>)> = lines
        .into_iter()
        .filter_map(|words| {
            let gene = words.first()?.trim();
            let ct = words.get(1)?.trim();
            let gl = gene.to_lowercase();
            if gene.is_empty()
                || gene.starts_with('#')
                || ct.is_empty()
                || gl == "gene"
                || gl == "symbol"
            {
                return None;
            }
            Some((Box::from(gene), Box::from(ct.replace(' ', "_"))))
        })
        .collect();
    anyhow::ensure!(!out.is_empty(), "no marker pairs parsed from {path}");
    Ok(out)
}

//////////////////////////////
// Output writing
//////////////////////////////

/// Write `{out_prefix}.{posterior,zscore,type_embedding}.parquet`. The argmax
/// label is the row-max of the posterior, so it isn't materialized; `pvalue`
/// is `pnorm(-zscore)`, so it isn't either.
fn write_annotation_outputs(
    out_prefix: &str,
    cell_names: &[Box<str>],
    type_names: &[Box<str>],
    h: usize,
    res: &AnnotateProjOutputs,
) -> Result<()> {
    let (n, c) = (res.n_cells, res.n_types);

    let posterior = DMatrix::<f32>::from_row_iterator(n, c, res.posterior_nc.iter().copied());
    let post_path = format!("{out_prefix}.posterior.parquet");
    posterior
        .to_parquet_with_names(
            &post_path,
            (Some(cell_names), Some("cell")),
            Some(type_names),
        )
        .with_context(|| format!("writing {post_path}"))?;
    info!("wrote {post_path}");

    if let Some(z) = res.zscore_nc.as_ref() {
        let zmat = DMatrix::<f32>::from_row_iterator(n, c, z.iter().copied());
        let z_path = format!("{out_prefix}.zscore.parquet");
        zmat.to_parquet_with_names(&z_path, (Some(cell_names), Some("cell")), Some(type_names))
            .with_context(|| format!("writing {z_path}"))?;
        info!("wrote {z_path} (p = pnorm(-z))");
    }

    let dim_names: Vec<Box<str>> = (0..h)
        .map(|j| format!("dim_{j}").into_boxed_str())
        .collect();
    let type_emb = DMatrix::<f32>::from_row_iterator(c, h, res.type_emb_ch.iter().copied());
    let te_path = format!("{out_prefix}.type_embedding.parquet");
    type_emb
        .to_parquet_with_names(
            &te_path,
            (Some(type_names), Some("cell_type")),
            Some(&dim_names),
        )
        .with_context(|| format!("writing {te_path}"))?;
    info!("wrote {te_path}");
    Ok(())
}

/// Log a per-type argmax-count histogram (a quick console sanity check).
fn log_label_histogram(type_names: &[Box<str>], argmax: &[(usize, f32)], n_cells: usize) {
    let mut counts = vec![0usize; type_names.len()];
    for &(t, _) in argmax {
        counts[t] += 1;
    }
    let mut order: Vec<usize> = (0..type_names.len()).collect();
    order.sort_by(|&a, &b| counts[b].cmp(&counts[a]));
    info!("annotation summary ({n_cells} cells, argmax):");
    for t in order {
        if counts[t] > 0 {
            info!("  {:24} {:6}", type_names[t], counts[t]);
        }
    }
}

//////////////////////////////
// Small numeric helpers
//////////////////////////////

/// Flatten a column-major `DMatrix` into a row-major `Vec<f32>` (parallel rows).
fn row_major(m: &DMatrix<f32>) -> Vec<f32> {
    let (r, c) = (m.nrows(), m.ncols());
    let mut v = vec![0f32; r * c];
    v.par_chunks_mut(c.max(1)).enumerate().for_each(|(i, row)| {
        for (j, slot) in row.iter_mut().enumerate() {
            *slot = m[(i, j)];
        }
    });
    v
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

/// L2-normalize each row of a row-major `[rows × cols]` buffer in place.
fn l2_normalize_rows(buf: &mut [f32], rows: usize, cols: usize) {
    buf.par_chunks_mut(cols.max(1)).take(rows).for_each(|row| {
        let nrm = row.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if nrm > 0.0 {
            for v in row.iter_mut() {
                *v /= nrm;
            }
        }
    });
}

/// In-place softmax of `xs` scaled by `inv_temp` (numerically stable).
fn softmax_inplace(xs: &mut [f32], inv_temp: f32) {
    if xs.is_empty() {
        return;
    }
    let mut mx = f32::NEG_INFINITY;
    for &x in xs.iter() {
        mx = mx.max(x * inv_temp);
    }
    let mut sum = 0f32;
    for x in xs.iter_mut() {
        let e = (*x * inv_temp - mx).exp();
        *x = e;
        sum += e;
    }
    if sum > 0.0 {
        for x in xs.iter_mut() {
            *x /= sum;
        }
    }
}
