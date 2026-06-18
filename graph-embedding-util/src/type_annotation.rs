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
//! ```
//!
//! **Two layers (fine + coarse).** A low-dim embedding cannot separate more
//! distinguishable directions than it has room for, so a fine marker set with
//! many nested types (e.g. CD8 Naive/Effector/Memory) over-types: the
//! signatures collapse onto a common cone and every per-cell score is flat.
//! We fix this **from the cells, not the markers**: cluster the cells in the
//! embedding space ([`matrix_util::clustering::leiden_clustering`], cosine
//! kNN + Leiden, community count automatic), then merge fine types that peak
//! on the same community into one coarse group, named by the lexical
//! commonality of its members. The coarse layer is what the cells can
//! actually resolve; the fine layer is kept alongside for subtype detail.
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
use matrix_util::common_io::{read_lines_of_words_delim, write_lines, ReadLinesOut};
use matrix_util::dmatrix_io::DMatrix;
use matrix_util::parquet::{write_named_table, Column};
use matrix_util::traits::IoOps;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

/// Per-type `(feature_index, weight)` marker lists — one inner `Vec` per type.
type MarkerSets = Vec<Vec<(u32, f32)>>;

/// Tunables for the annotation routines.
pub struct AnnotateProjConfig {
    /// Permutation draws per type for the null (0 disables z/p, falls back
    /// to raw cosine scores).
    pub n_perm: usize,
    /// Deterministic RNG seed for the permutation null and clustering.
    pub seed: u64,
    /// k for the cell kNN graph used by the coarsening clusterer.
    pub knn: usize,
    /// Leiden modularity resolution for cell clustering (higher → more,
    /// smaller communities → more coarse groups).
    pub resolution: f64,
    /// Enable cell-grounded coarsening (cluster cells, merge over-split
    /// types). When false, the coarse layer mirrors the fine layer.
    pub coarsen: bool,
}

impl Default for AnnotateProjConfig {
    fn default() -> Self {
        Self {
            n_perm: 200,
            seed: 42,
            knn: 30,
            resolution: 1.0,
            coarsen: true,
        }
    }
}

/// Result of [`annotate_by_projection`]. All matrices are row-major.
///
/// The annotation is two-layer: a *fine* layer (one marker-defined type per
/// column) and a *coarse* layer whose groups are **cell communities** —
/// fine types that collapse onto the same community of cells are merged and
/// named by the lexical commonality of their members. Each per-cell
/// community id indexes directly into the coarse arrays.
pub struct AnnotateProjOutputs {
    pub n_cells: usize,
    /// Number of fine types.
    pub n_types: usize,
    /// Number of coarse groups (= number of cell communities).
    pub n_coarse: usize,
    /// `[C × H]` L2-normalized fine type signature embeddings (plot anchors).
    pub type_emb_ch: Vec<f32>,
    /// `[K × H]` L2-normalized coarse signature embeddings.
    pub coarse_emb_kh: Vec<f32>,
    /// `[K]` lexical label per coarse group.
    pub coarse_names: Vec<Box<str>>,
    /// `[C]` fine type → coarse group index.
    pub coarse_of_fine: Vec<usize>,
    /// `[N]` cell → community (= coarse group) index.
    pub community: Vec<usize>,
    /// `[N × C]` per-cell fine score: z-score if `n_perm > 0`, else cosine.
    pub fine_z: Vec<f32>,
    /// `[N × K]` per-cell coarse score, same scale as `fine_z`.
    pub coarse_z: Vec<f32>,
    /// `[N × C]` fine p-values `pnorm(-z)`, or `None` when `n_perm == 0`.
    pub fine_p: Option<Vec<f32>>,
    /// `[N × K]` coarse p-values, or `None` when `n_perm == 0`.
    pub coarse_p: Option<Vec<f32>>,
    /// `[N]` argmax fine type per cell.
    pub fine_label: Vec<usize>,
    /// `[K × C]` mean fine score of cells in each community (for profiling).
    pub enrich: Vec<f32>,
    /// Per community: the ranked lineage-defining fine types (the same set
    /// used for the label), most-significant first.
    pub community_members: Vec<Vec<usize>>,
}

//////////////////////////////
// High-level entry point
//////////////////////////////

/// End-to-end annotation from in-memory embeddings: parse + match the marker
/// TSV against `gene_names`, project every type, cluster + coarsen, and write
/// `{out_prefix}.{annot,community_profile,type_map,type_embedding,coarse_embedding}.parquet`.
/// The thin per-tool adapters (gem / cage / bge) only load the two embedding
/// matrices from their own manifest and call this.
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
    let res = annotate_by_projection(
        &beta_flat,
        g,
        &cell_flat,
        n,
        &type_markers,
        &type_names,
        h,
        cfg,
    )?;

    write_annotation_outputs(out_prefix, cell_names, &type_names, h, &res)?;
    log_label_histogram(&res);
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
#[allow(clippy::too_many_arguments)]
pub fn annotate_by_projection(
    feature_emb: &[f32],
    n_features: usize,
    cell_emb: &[f32],
    n_cells: usize,
    type_markers: &[Vec<(u32, f32)>],
    type_names: &[Box<str>],
    h: usize,
    cfg: &AnnotateProjConfig,
) -> Result<AnnotateProjOutputs> {
    let n_types = type_markers.len();
    anyhow::ensure!(n_types >= 1, "annotate_by_projection needs ≥ 1 type");

    // 1. Fine signatures + unit-normalized cells (gem/bge already are).
    let type_emb_ch = type_signatures(feature_emb, n_features, type_markers, h);
    let mut cell_u = cell_emb.to_vec();
    l2_normalize_rows(&mut cell_u, n_cells, h);

    // Marker pool: the union of every type's marker genes. The permutation
    // null draws from THIS pool (a label shuffle), not the whole genome, so
    // each null type is "another type's markers of the same size".
    let marker_pool = marker_gene_pool(type_markers, n_features);

    // Score a marker set (its signatures) against every cell — used identically
    // for the fine and the coarse layer.
    let score = |markers: &[Vec<(u32, f32)>], emb: &[f32]| {
        type_scores(
            feature_emb,
            &marker_pool,
            &cell_u,
            n_cells,
            markers,
            emb,
            h,
            cfg,
        )
    };

    // 2. Fine per-cell scores (permutation z, else cosine) + p-values.
    let (fine_z, fine_p) = score(type_markers, &type_emb_ch);
    let fine_label = argmax_rows(&fine_z, n_cells, n_types);

    // 3. Cluster cells in the embedding space → communities. These ARE the
    // coarse groups: types that collapse onto the same community get merged.
    let do_coarsen = cfg.coarsen && n_cells >= 2 && n_types >= 2;
    let (community, n_comm) = if do_coarsen {
        let cell_mat = DMatrix::<f32>::from_row_iterator(n_cells, h, cell_u.iter().copied());
        let knn = cfg.knn.clamp(1, n_cells - 1);
        let labels = matrix_util::clustering::leiden_clustering(
            &cell_mat,
            knn,
            cfg.resolution,
            None,
            Some(cfg.seed),
            true, // cosine: bge/gem embeddings are dot-product trained
        )?;
        let k = labels.iter().copied().max().map_or(0, |m| m + 1).max(1);
        (labels, k)
    } else {
        // Coarse layer mirrors fine: each cell's "community" is its fine argmax.
        (fine_label.clone(), n_types)
    };

    // 4. Community × fine-type enrichment (mean fine score over each
    // community), column-centered so a marker set that is high *everywhere*
    // (common-mode aligned) doesn't name every community — what selects a
    // community's lineage is its enrichment RELATIVE to the other communities.
    let mut enrich = community_enrichment(&fine_z, n_cells, n_types, &community, n_comm);
    center_columns(&mut enrich, n_comm, n_types);

    // A community's identity is the fine types ENRICHED in it (its top few),
    // ranked with a √(marker-set size) weight so a tiny high-IDF set doesn't
    // win on label-shuffle noise. Shared by the label, the coarse markers, and
    // the community profile so all three agree.
    let mk_weight: Vec<f32> = type_markers
        .iter()
        .map(|m| (m.len() as f32).sqrt())
        .collect();
    let community_members = top_enriched_members(&enrich, &mk_weight, n_comm, n_types, 6);

    // 5. Merge map + lexical names + coarse re-scoring.
    let (coarse_of_fine, coarse_names, coarse_emb_kh, coarse_z, coarse_p) = if do_coarsen {
        // `coarse_of_fine` assigns each fine type to its peak community (the
        // type_map merge record).
        let mut sizes = vec![0usize; n_comm];
        for &kk in &community {
            sizes[kk] += 1;
        }
        let coarse_of_fine = build_merge_map(&enrich, &sizes, n_comm, n_types);
        let coarse_names: Vec<Box<str>> = community_members
            .iter()
            .map(|m| lexical_label(m, type_names))
            .collect();
        let coarse_markers = coarse_markers_from_groups(&community_members, type_markers);
        let coarse_emb_kh = type_signatures(feature_emb, n_features, &coarse_markers, h);
        // Same marker pool for the coarse null — coarse markers ⊆ the pool.
        let (coarse_z, coarse_p) = score(&coarse_markers, &coarse_emb_kh);
        (
            coarse_of_fine,
            coarse_names,
            coarse_emb_kh,
            coarse_z,
            coarse_p,
        )
    } else {
        (
            (0..n_types).collect(),
            type_names.to_vec(),
            type_emb_ch.clone(),
            fine_z.clone(),
            fine_p.clone(),
        )
    };

    Ok(AnnotateProjOutputs {
        n_cells,
        n_types,
        n_coarse: n_comm,
        type_emb_ch,
        coarse_emb_kh,
        coarse_names,
        coarse_of_fine,
        community,
        fine_z,
        coarse_z,
        fine_p,
        coarse_p,
        fine_label,
        enrich,
        community_members,
    })
}

/// Per-cell, per-type score matrix `[N × n_types]`. With a permutation null
/// (`n_perm > 0`) this is the null-standardized z-score and we also return
/// `p = pnorm(-z)`; otherwise it is the raw cosine and `p` is `None`.
#[allow(clippy::too_many_arguments)]
fn type_scores(
    feature_emb: &[f32],
    marker_pool: &[u32],
    cell_u: &[f32],
    n_cells: usize,
    type_markers: &[Vec<(u32, f32)>],
    type_emb: &[f32],
    h: usize,
    cfg: &AnnotateProjConfig,
) -> (Vec<f32>, Option<Vec<f32>>) {
    let n_types = type_markers.len();
    if cfg.n_perm > 0 && n_types > 0 {
        let z = permutation_zscores(
            feature_emb,
            marker_pool,
            cell_u,
            n_cells,
            type_markers,
            type_emb,
            h,
            cfg,
        );
        let p: Vec<f32> = z.par_iter().map(|&v| pnorm_upper(v)).collect();
        (z, Some(p))
    } else {
        let mut s = vec![0f32; n_cells * n_types];
        s.par_chunks_mut(n_types.max(1))
            .enumerate()
            .for_each(|(n, row)| {
                let cu = &cell_u[n * h..(n + 1) * h];
                for (t, slot) in row.iter_mut().enumerate() {
                    *slot = dot(cu, &type_emb[t * h..(t + 1) * h]);
                }
            });
        (s, None)
    }
}

/// Argmax type per row of an `[n × c]` row-major score matrix.
fn argmax_rows(score: &[f32], n: usize, c: usize) -> Vec<usize> {
    if c == 0 {
        return vec![0; n];
    }
    (0..n)
        .map(|i| {
            let row = &score[i * c..(i + 1) * c];
            let mut best = 0;
            for j in 1..c {
                if row[j] > row[best] {
                    best = j;
                }
            }
            best
        })
        .collect()
}

/// `[n_comm × n_types]` mean fine score of the cells in each community —
/// the cell-grounded confusability that drives merging.
fn community_enrichment(
    fine_z: &[f32],
    n_cells: usize,
    n_types: usize,
    community: &[usize],
    n_comm: usize,
) -> Vec<f32> {
    let width = n_comm * n_types;
    // Parallel scatter-accumulate: each cell chunk folds into its own
    // per-community (sum, count), then the partials are reduced.
    let (sum, cnt) = (0..n_cells)
        .into_par_iter()
        .fold(
            || (vec![0f64; width], vec![0usize; n_comm]),
            |(mut sum, mut cnt), c| {
                let k = community[c];
                cnt[k] += 1;
                let row = &fine_z[c * n_types..(c + 1) * n_types];
                for (s, &v) in sum[k * n_types..(k + 1) * n_types].iter_mut().zip(row) {
                    *s += v as f64;
                }
                (sum, cnt)
            },
        )
        .reduce(
            || (vec![0f64; width], vec![0usize; n_comm]),
            |(mut as_, mut ac), (bs, bc)| {
                for (a, b) in as_.iter_mut().zip(bs) {
                    *a += b;
                }
                for (a, b) in ac.iter_mut().zip(bc) {
                    *a += b;
                }
                (as_, ac)
            },
        );
    let mut enrich = vec![0f32; width];
    for k in 0..n_comm {
        let d = cnt[k].max(1) as f64;
        for t in 0..n_types {
            enrich[k * n_types + t] = (sum[k * n_types + t] / d) as f32;
        }
    }
    enrich
}

/// Subtract each fine type's cross-community mean from its `[n_comm × n_types]`
/// enrichment column, so the score reflects how distinctive a type is to a
/// community rather than its absolute (common-mode) level.
fn center_columns(enrich: &mut [f32], n_comm: usize, n_types: usize) {
    if n_comm == 0 {
        return;
    }
    for t in 0..n_types {
        let mut mean = 0f64;
        for k in 0..n_comm {
            mean += enrich[k * n_types + t] as f64;
        }
        let mean = (mean / n_comm as f64) as f32;
        for k in 0..n_comm {
            enrich[k * n_types + t] -= mean;
        }
    }
}

/// Assign each fine type to the community where it is most enriched — the
/// type_map merge record (fine type → coarse group). Centered enrichment is
/// weighted by `√(community size)`: the mean over a community's cells has
/// standard error `∝ 1/√n`, so a tiny community's noise-inflated enrichment
/// would otherwise grab most types. The weight makes types prefer large,
/// reliably-enriched communities.
fn build_merge_map(enrich: &[f32], sizes: &[usize], n_comm: usize, n_types: usize) -> Vec<usize> {
    let w: Vec<f32> = sizes.iter().map(|&n| (n as f32).sqrt()).collect();
    (0..n_types)
        .map(|t| {
            let mut best = 0;
            for k in 1..n_comm {
                if enrich[k * n_types + t] * w[k] > enrich[best * n_types + t] * w[best] {
                    best = k;
                }
            }
            best
        })
        .collect()
}

/// Per community, the up-to-`max_n` most-enriched fine types with positive
/// (above-null) enrichment — the lineage that defines the community's name
/// and coarse marker set. Always returns at least the single top type so no
/// community is left nameless.
///
/// Ranking is `enrich · weight[t]` where `weight[t] = √|markers_t|`: a tiny
/// high-IDF marker set (e.g. 3-gene Platelet) is noisy under the label-shuffle
/// null and would otherwise top ambiguous communities on variance alone; the
/// √-size weight favours lineages backed by more markers. (Selection only —
/// the `enrich > 0` gate still uses the raw centered value.)
fn top_enriched_members(
    enrich: &[f32],
    weight: &[f32],
    n_comm: usize,
    n_types: usize,
    max_n: usize,
) -> Vec<Vec<usize>> {
    let score = |k: usize, t: usize| enrich[k * n_types + t] * weight[t];
    (0..n_comm)
        .map(|k| {
            let mut order: Vec<usize> = (0..n_types).collect();
            order.sort_by(|&a, &b| {
                score(k, b)
                    .partial_cmp(&score(k, a))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut sel: Vec<usize> = order
                .iter()
                .copied()
                .take(max_n)
                .filter(|&t| enrich[k * n_types + t] > 0.0)
                .collect();
            if sel.is_empty() {
                sel.push(order[0]);
            }
            sel
        })
        .collect()
}

/// Name a coarse group by the tokens shared by a **majority** of its member
/// type names (split on space/`_`, numeric tokens dropped), kept in the
/// representative (most-enriched, first) member's order: `{Naive B, Memory B,
/// pre B}` → `B`; `{CD8 Naive, CD8 Effector_1, CD8 Memory}` → `CD8`. A strict
/// intersection is brittle — one off-lineage member in a large community
/// would wipe the shared token — so we keep tokens present in ≥ 60% of
/// members (and ≥ 2). Falls back to the representative's full name when no
/// token clears the bar.
fn lexical_label(members: &[usize], type_names: &[Box<str>]) -> Box<str> {
    let tok = |s: &str| -> Vec<String> {
        s.split([' ', '_'])
            .filter(|x| !x.is_empty() && !x.chars().all(|c| c.is_ascii_digit()))
            .map(str::to_string)
            .collect()
    };
    match members {
        [] => Box::from("NA"),
        [only] => type_names[*only].clone(),
        [first, rest @ ..] => {
            let m = 1 + rest.len();
            let thresh = (((m as f64) * 0.6).ceil() as usize).max(2);
            // document frequency of each token across members (deduped per member)
            let mut df: FxHashMap<String, usize> = FxHashMap::default();
            for &t in members {
                let mut seen: FxHashSet<String> = FxHashSet::default();
                for w in tok(&type_names[t]) {
                    if seen.insert(w.clone()) {
                        *df.entry(w).or_insert(0) += 1;
                    }
                }
            }
            // label = representative's tokens that clear the majority bar, in order
            let label: Vec<String> = tok(&type_names[*first])
                .into_iter()
                .filter(|w| df.get(w).copied().unwrap_or(0) >= thresh)
                .collect();
            if label.is_empty() {
                type_names[*first].clone()
            } else {
                label.join(" ").into_boxed_str()
            }
        }
    }
}

/// Coarse marker set per group = union of member `(gene, weight)` lists,
/// deduped by gene keeping the max weight.
fn coarse_markers_from_groups(
    members: &[Vec<usize>],
    type_markers: &[Vec<(u32, f32)>],
) -> Vec<Vec<(u32, f32)>> {
    members
        .iter()
        .map(|grp| {
            let mut map: FxHashMap<u32, f32> = FxHashMap::default();
            for &t in grp {
                for &(g, w) in &type_markers[t] {
                    let e = map.entry(g).or_insert(f32::NEG_INFINITY);
                    if w > *e {
                        *e = w;
                    }
                }
            }
            map.into_iter().collect()
        })
        .collect()
}

/// Normal upper-tail p-value `pnorm(-z) = ½·erfc(z/√2)`. The permutation
/// null cosines are ~Gaussian in high dimension, so this is well-calibrated
/// with no `1/(n_perm+1)` empirical floor.
fn pnorm_upper(z: f32) -> f32 {
    use statrs::function::erf::erfc;
    (0.5 * erfc(z as f64 / std::f64::consts::SQRT_2)) as f32
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

/// Union of every type's marker gene indices (sorted, unique, in range) — the
/// universe the label-shuffle null samples from.
fn marker_gene_pool(type_markers: &[Vec<(u32, f32)>], n_features: usize) -> Vec<u32> {
    let mut set: FxHashSet<u32> = FxHashSet::default();
    for markers in type_markers {
        for &(g, _) in markers {
            if (g as usize) < n_features {
                set.insert(g);
            }
        }
    }
    let mut pool: Vec<u32> = set.into_iter().collect();
    pool.sort_unstable();
    pool
}

/// `[N × C]` null-standardized z-scores under a **label-shuffle** null. For
/// each type we draw `n_perm` random gene sets of the same size from the
/// `marker_pool` (the union of every type's marker genes), keeping the type's
/// own weights; each draw's normalized centroid is a null signature. Sampling
/// from the marker pool — not the whole genome — makes the null "another
/// type's markers of the same size" rather than "random background genes", so
/// it cancels the common-mode shared by markers as a class and tests whether
/// THIS type's markers specifically align with the cell. A cell's observed
/// cosine is standardized against the moments of its `n_perm` null cosines.
#[allow(clippy::too_many_arguments)]
fn permutation_zscores(
    feature_emb: &[f32],
    marker_pool: &[u32],
    cell_u: &[f32],
    n_cells: usize,
    type_markers: &[Vec<(u32, f32)>],
    type_emb_ch: &[f32],
    h: usize,
    cfg: &AnnotateProjConfig,
) -> Vec<f32> {
    let n_types = type_markers.len();
    let n_perm = cfg.n_perm;
    let pool_n = marker_pool.len();

    // C·n_perm null signatures, built in parallel. Each (type, perm) is a
    // deterministic seeded draw of marker-pool genes; the set is accumulated
    // straight into its centroid row — no intermediate marker-list is built.
    let mut null_emb = vec![0f32; n_types * n_perm * h];
    null_emb
        .par_chunks_mut(h)
        .enumerate()
        .for_each(|(idx, row)| {
            let (t, p) = (idx / n_perm, idx % n_perm);
            let markers = &type_markers[t];
            let m = markers.len().min(pool_n);
            let mut rng = SmallRng::seed_from_u64(
                cfg.seed ^ ((t as u64) << 32) ^ (p as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            );
            // shuffle labels: m random genes from the marker pool, T's weights
            let drawn = rand::seq::index::sample(&mut rng, pool_n, m);
            for (pool_i, &(_, w)) in drawn.iter().zip(markers.iter()) {
                let gidx = marker_pool[pool_i] as usize;
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
                let obs = f64::from(dot(cu, &type_emb_ch[t * h..(t + 1) * h]));
                // Online mean/variance over the n_perm null cosines.
                let (mut mean, mut m2) = (0f64, 0f64);
                for p in 0..n_perm {
                    let v = t * n_perm + p;
                    let s = f64::from(dot(cu, &null_emb[v * h..(v + 1) * h]));
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

/// Write the tidy annotation tables:
/// * `{prefix}.annot.parquet` — one row per cell: community, coarse + fine
///   label, score (z), and p-value for each layer.
/// * `{prefix}.community_profile.parquet` — one row per community.
/// * `{prefix}.type_map.parquet` — fine → coarse merge record.
/// * `{prefix}.{type,coarse}_embedding.parquet` — signature plot anchors.
fn write_annotation_outputs(
    out_prefix: &str,
    cell_names: &[Box<str>],
    type_names: &[Box<str>],
    h: usize,
    res: &AnnotateProjOutputs,
) -> Result<()> {
    let (n, c, k) = (res.n_cells, res.n_types, res.n_coarse);
    let nan = f32::NAN;

    ////////////////////////////
    // per-cell annotation table
    ////////////////////////////
    let mut community = Vec::with_capacity(n);
    let mut coarse_label = Vec::with_capacity(n);
    let mut coarse_z = Vec::with_capacity(n);
    let mut coarse_p = Vec::with_capacity(n);
    let mut fine_label = Vec::with_capacity(n);
    let mut fine_z = Vec::with_capacity(n);
    let mut fine_p = Vec::with_capacity(n);
    for cell in 0..n {
        let kk = res.community[cell];
        let ff = res.fine_label[cell];
        community.push(kk as i32);
        coarse_label.push(res.coarse_names[kk].clone());
        coarse_z.push(res.coarse_z[cell * k + kk]);
        coarse_p.push(res.coarse_p.as_ref().map_or(nan, |p| p[cell * k + kk]));
        fine_label.push(type_names[ff].clone());
        fine_z.push(res.fine_z[cell * c + ff]);
        fine_p.push(res.fine_p.as_ref().map_or(nan, |p| p[cell * c + ff]));
    }
    // BH q-values across the N per-cell calls of each layer (FDR over the
    // selected-label p-values); NaN-filled when the null was skipped.
    let coarse_q = if res.coarse_p.is_some() {
        enrichment::fdr::bh_fdr(&coarse_p)
    } else {
        vec![nan; n]
    };
    let fine_q = if res.fine_p.is_some() {
        enrichment::fdr::bh_fdr(&fine_p)
    } else {
        vec![nan; n]
    };
    let annot_path = format!("{out_prefix}.annot.parquet");
    write_named_table(
        &annot_path,
        "cell",
        cell_names,
        &[
            (Box::from("community"), Column::I32(&community)),
            (Box::from("coarse_label"), Column::Str(&coarse_label)),
            (Box::from("coarse_z"), Column::F32(&coarse_z)),
            (Box::from("coarse_p"), Column::F32(&coarse_p)),
            (Box::from("coarse_q"), Column::F32(&coarse_q)),
            (Box::from("fine_label"), Column::Str(&fine_label)),
            (Box::from("fine_z"), Column::F32(&fine_z)),
            (Box::from("fine_p"), Column::F32(&fine_p)),
            (Box::from("fine_q"), Column::F32(&fine_q)),
        ],
    )
    .with_context(|| format!("writing {annot_path}"))?;
    info!("wrote {annot_path}");

    ////////////////////////////
    // cell → coarse-label membership TSV
    ////////////////////////////
    // A plain `cell<TAB>coarse_label` file (no header — `Membership::from_file`
    // reads with `hdr_line = -1`, so a header would be a stray non-matching
    // entry). Feeds `data-beans stat -s row -g` and `faba gem-summary` to
    // group any count matrix by cell type.
    let membership_lines: Vec<Box<str>> = cell_names
        .iter()
        .zip(coarse_label.iter())
        .map(|(cell, label)| format!("{cell}\t{label}").into_boxed_str())
        .collect();
    let membership_path = format!("{out_prefix}.membership.tsv");
    write_lines(&membership_lines, &membership_path)
        .with_context(|| format!("writing {membership_path}"))?;
    info!("wrote {membership_path}");

    ////////////////////////////
    // community profile table
    ////////////////////////////
    let mut comm_sizes = vec![0i32; k];
    for &kk in &res.community {
        comm_sizes[kk] += 1;
    }
    let comm_names: Vec<Box<str>> = (0..k).map(|i| i.to_string().into_boxed_str()).collect();
    let comm_label: Vec<Box<str>> = res.coarse_names.clone();
    // Show the SAME ranked lineage set that produced the label (weighted
    // ordering), displaying each member's raw centered enrichment.
    let top_fine: Vec<Box<str>> = (0..k)
        .map(|kk| {
            res.community_members[kk]
                .iter()
                .take(5)
                .map(|&t| format!("{}({:.1})", type_names[t], res.enrich[kk * c + t]))
                .collect::<Vec<_>>()
                .join(",")
                .into_boxed_str()
        })
        .collect();
    let profile_path = format!("{out_prefix}.community_profile.parquet");
    write_named_table(
        &profile_path,
        "community",
        &comm_names,
        &[
            (Box::from("n_cells"), Column::I32(&comm_sizes)),
            (Box::from("coarse_label"), Column::Str(&comm_label)),
            (Box::from("top_fine_types"), Column::Str(&top_fine)),
        ],
    )
    .with_context(|| format!("writing {profile_path}"))?;
    info!("wrote {profile_path}");

    ////////////////////////////
    // fine → coarse merge map
    ////////////////////////////
    let mut group_members: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (t, &kk) in res.coarse_of_fine.iter().enumerate() {
        group_members[kk].push(t);
    }
    let map_coarse: Vec<Box<str>> = (0..c)
        .map(|t| res.coarse_names[res.coarse_of_fine[t]].clone())
        .collect();
    let map_members: Vec<Box<str>> = (0..c)
        .map(|t| {
            group_members[res.coarse_of_fine[t]]
                .iter()
                .map(|&m| type_names[m].as_ref())
                .collect::<Vec<_>>()
                .join(",")
                .into_boxed_str()
        })
        .collect();
    let map_path = format!("{out_prefix}.type_map.parquet");
    write_named_table(
        &map_path,
        "fine_type",
        type_names,
        &[
            (Box::from("coarse_label"), Column::Str(&map_coarse)),
            (Box::from("members"), Column::Str(&map_members)),
        ],
    )
    .with_context(|| format!("writing {map_path}"))?;
    info!("wrote {map_path}");

    ////////////////////////////
    // signature plot anchors (fine + coarse)
    ////////////////////////////
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

    let coarse_emb = DMatrix::<f32>::from_row_iterator(k, h, res.coarse_emb_kh.iter().copied());
    let ce_path = format!("{out_prefix}.coarse_embedding.parquet");
    coarse_emb
        .to_parquet_with_names(
            &ce_path,
            (Some(&res.coarse_names), Some("cell_type")),
            Some(&dim_names),
        )
        .with_context(|| format!("writing {ce_path}"))?;
    info!("wrote {ce_path}");

    Ok(())
}

/// Log a per-coarse-label cell-count histogram (a quick console sanity check).
fn log_label_histogram(res: &AnnotateProjOutputs) {
    let mut counts: FxHashMap<&str, usize> = FxHashMap::default();
    for &kk in &res.community {
        *counts.entry(res.coarse_names[kk].as_ref()).or_insert(0) += 1;
    }
    let mut ranked: Vec<(&str, usize)> = counts.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1));
    info!(
        "annotation summary ({} cells, {} communities → {} coarse labels):",
        res.n_cells,
        res.n_coarse,
        ranked.len()
    );
    for (label, n) in ranked {
        info!("  {label:24} {n:6}");
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

#[cfg(test)]
mod tests;
