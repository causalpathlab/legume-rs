//! Donor-private CNV clone calling from inferCNV profiles.
//!
//! Default engine: [`CloneEngine::Bayes`] — two-stage posterior malignancy +
//! peaky donor-mix components ([`crate::clone_bayes`]). Legacy mixture path
//! (`--engine mixture`) sketches cells, k-means clusters, scores donor
//! enclosure × segmental structure, then BIC-gates clusters.
//!
//! Stratum `0` is the mixable bucket. A false clone hard-partitions collapse
//! and under-integrates batch δ — costly; prefer the Bayes gate without
//! `--ref`.

use crate::clone_bayes::{
    bayes_config_from_clone_call, call_clones_bayes, load_burden_from_cells_table,
};
use crate::kmeans_init::select_kmeans_k;
use data_beans::sparse_data_visitors::styled_progress_bar;
use data_beans::sparse_io_vector::SparseIoVec;
use genomic_data::coordinates::{parse_peak_coordinates, PeakCoord};
use matrix_util::clustering::{Kmeans, KmeansArgs};
use matrix_util::parquet::{read_table_columns, write_named_table, Column};
use nalgebra::DMatrix;
use nalgebra_sparse::CscMatrix;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};
use rustc_hash::FxHashMap;

/// Clone-calling engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum CloneEngine {
    /// Two-stage Bayesian malignancy + peaky donor-mix gate (default).
    #[default]
    Bayes,
    /// Legacy cluster-score Gaussian mixture.
    Mixture,
}

/// Knobs for [`call_clones`].
#[derive(Debug, Clone)]
pub struct CloneCallConfig {
    /// Same as [`crate::cell_profile::CellProfileConfig::bin_size`]: `0` =
    /// classic inferCNV (gene / interval rows → one sketch dim per chromosome);
    /// `> 0` = average into fixed genomic tiles of this many bp (prefer
    /// `1_000_000` with inferCNV's large-cohort guidance).
    pub bin_size: i64,
    /// Finite K_max (Bayes components / mixture k-means K).
    pub k_max: usize,
    /// Which gate: Bayes (default) or legacy mixture.
    pub engine: CloneEngine,
    /// Donor-mix purity floor. Bayes default ~0.8; mixture `None` = mixture only.
    pub min_purity: Option<f32>,
    /// Clusters / components smaller than this cannot be clones.
    pub min_cells: usize,
    /// Optional segmental-CN z floor on top of the mixture (None = mixture only).
    /// Genomic roughness of the chromosome sketch, not tissue spatial z.
    pub spatial_z: Option<f32>,
    /// Random subsets drawn for the segmental-CN null (mixture only).
    pub n_perm: usize,
    pub seed: u64,
    pub kmeans_iter: usize,
    /// Bayes Gibbs sweeps after burn-in (posterior samples).
    pub n_sweeps: usize,
    /// Bayes burn-in sweeps discarded before posterior averages.
    pub n_burnin: usize,
    /// Stratum 0 if `p_malig <` this (Bayes).
    pub p_malig_threshold: f32,
}

impl Default for CloneCallConfig {
    fn default() -> Self {
        Self {
            bin_size: 0,
            k_max: 8,
            engine: CloneEngine::Bayes,
            min_purity: None,
            min_cells: 50,
            spatial_z: None,
            n_perm: 32,
            seed: 1,
            kmeans_iter: 200,
            n_sweeps: 100,
            n_burnin: 40,
            p_malig_threshold: 0.5,
        }
    }
}

/// Per-cluster diagnostics before the mixture gate.
#[derive(Debug, Clone, Copy)]
pub struct ClusterScore {
    pub n_cells: usize,
    pub purity: f32,
    /// −log10(hypergeometric enrichment p) of the majority donor.
    pub enrich: f32,
    pub spatial_score: f32,
    pub spatial_z: f32,
    /// Scalar the mixture sees: donor enclosure × spatial structure.
    pub clone_score: f32,
}

/// One cell's clone-call row.
#[derive(Debug, Clone)]
pub struct CloneRow {
    pub cell: Box<str>,
    pub donor: Box<str>,
    pub cluster: usize,
    pub stratum: usize,
    pub purity: f32,
    pub spatial_score: f32,
    pub spatial_z: f32,
    /// Mean |CNV| burden (from `{prefix}.cells.parquet` or recomputed).
    pub burden: f32,
    /// Posterior malignancy `E[m_i]` (Bayes); 0/1 proxy on mixture path.
    pub p_malig: f32,
}

/// Donor tag: suffix after the last `@` (data-beans disjoint `@basename`),
/// otherwise the whole name.
#[must_use]
pub fn donor_of(cell: &str) -> &str {
    cell.rsplit_once('@').map(|(_, d)| d).unwrap_or(cell)
}

/// Per-cell mean of the profile within each chromosome, genome order of first
/// appearance. Rows with empty/`NA` chromosome names are dropped.
///
/// Returns `(sketch [C × N], chromosome names)`.
pub fn chromosome_sketch(
    profiles: &DMatrix<f32>,
    chr_of_row: &[impl AsRef<str>],
) -> (DMatrix<f32>, Vec<Box<str>>) {
    assert_eq!(profiles.nrows(), chr_of_row.len());
    let mut chr_index: FxHashMap<Box<str>, usize> = FxHashMap::default();
    let mut names: Vec<Box<str>> = Vec::new();
    let mut row_chr: Vec<Option<usize>> = Vec::with_capacity(chr_of_row.len());
    for name in chr_of_row {
        let raw = name.as_ref();
        if raw.is_empty() || raw.eq_ignore_ascii_case("NA") {
            row_chr.push(None);
            continue;
        }
        let key: Box<str> = raw.into();
        let idx = *chr_index.entry(key.clone()).or_insert_with(|| {
            let i = names.len();
            names.push(key);
            i
        });
        row_chr.push(Some(idx));
    }
    let c = names.len();
    let n = profiles.ncols();
    if c == 0 || n == 0 {
        return (DMatrix::zeros(c, n), names);
    }
    let mut counts = vec![0f32; c];
    for &idx in &row_chr {
        if let Some(i) = idx {
            counts[i] += 1.0;
        }
    }
    let inv: Vec<f32> = counts
        .iter()
        .map(|&k| if k > 0.0 { 1.0 / k } else { 0.0 })
        .collect();
    let sketch = matrix_util::dmatrix_util::build_columns_par(c, n, |j, col| {
        col.fill(0.0);
        let src = profiles.column(j);
        for (r, &idx) in row_chr.iter().enumerate() {
            if let Some(i) = idx {
                col[i] += src[r];
            }
        }
        for (v, w) in col.iter_mut().zip(&inv) {
            *v *= w;
        }
    });
    (sketch, names)
}

/// Mean within each sketch bin for a CSC block (rows = genomic intervals).
pub fn sketch_csc(
    csc: &CscMatrix<f32>,
    row_bin: &[Option<usize>],
    n_bins: usize,
    inv_bins: &[f32],
) -> DMatrix<f32> {
    let n = csc.ncols();
    matrix_util::dmatrix_util::build_columns_par(n_bins, n, |j, col| {
        col.fill(0.0);
        let column = csc.col(j);
        for (&r, &v) in column.row_indices().iter().zip(column.values()) {
            if let Some(i) = row_bin.get(r).copied().flatten() {
                col[i] += v;
            }
        }
        for (v, w) in col.iter_mut().zip(inv_bins) {
            *v *= w;
        }
    })
}

/// Map each genomic-interval row onto a sketch bin: whole chromosome when
/// `bin_size ≤ 0`, else the `bin_size`-bp tile containing the interval midpoint.
fn compact_sketch_bins(
    coords: &[Option<PeakCoord>],
    bin_size: i64,
) -> (Vec<Option<usize>>, Vec<Box<str>>, Vec<f32>) {
    // Key: (chr, tile_start); tile_start = -1 means the whole chromosome.
    let mut index: FxHashMap<(Box<str>, i64), usize> = FxHashMap::default();
    let mut names: Vec<Box<str>> = Vec::new();
    let mut row_bin = Vec::with_capacity(coords.len());
    let mut counts: Vec<f32> = Vec::new();
    for c in coords {
        let Some(coord) = c.as_ref() else {
            row_bin.push(None);
            continue;
        };
        if coord.chr.is_empty() || coord.chr.eq_ignore_ascii_case("NA") {
            row_bin.push(None);
            continue;
        }
        let tile = if bin_size <= 0 {
            -1
        } else {
            let mid = (coord.start + coord.end) / 2;
            mid.max(0).div_euclid(bin_size) * bin_size
        };
        let idx = *index.entry((coord.chr.clone(), tile)).or_insert_with(|| {
            let i = names.len();
            names.push(if tile < 0 {
                coord.chr.clone()
            } else {
                format!("{}:{}-{}", coord.chr, tile, tile + bin_size).into()
            });
            counts.push(0.0);
            i
        });
        counts[idx] += 1.0;
        row_bin.push(Some(idx));
    }
    let inv: Vec<f32> = counts
        .iter()
        .map(|&k| if k > 0.0 { 1.0 / k } else { 0.0 })
        .collect();
    (row_bin, names, inv)
}

fn ln_choose(n: usize, k: usize) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    let k = k.min(n - k);
    (0..k)
        .map(|i| ((n - k + 1 + i) as f64).ln() - ((i + 1) as f64).ln())
        .sum()
}

fn hypergeom_sf(k: usize, draws: usize, k_pop: usize, n_pop: usize) -> f64 {
    let k_max = draws.min(k_pop);
    if k > k_max {
        return 0.0;
    }
    if draws > n_pop || k_pop > n_pop {
        return 0.0;
    }
    if k == 0 {
        return 1.0;
    }
    let n_other = n_pop - k_pop;
    if draws > n_other && k < draws - n_other {
        return 1.0;
    }
    let mut log_p = ln_choose(k_pop, k) + ln_choose(n_other, draws - k) - ln_choose(n_pop, draws);
    let mut acc = log_p.exp();
    for x in k..k_max {
        let num = (k_pop - x) as f64 * (draws - x) as f64;
        let den = (x + 1) as f64 * (n_pop - k_pop - draws + x + 1) as f64;
        if den <= 0.0 {
            break;
        }
        log_p += num.ln() - den.ln();
        let p = log_p.exp();
        if !p.is_finite() {
            break;
        }
        acc += p;
        if acc >= 1.0 {
            return 1.0;
        }
    }
    acc.clamp(0.0, 1.0)
}

fn mean_l1(sketch: &DMatrix<f32>, members: &[usize]) -> f32 {
    let c = sketch.nrows();
    if members.is_empty() || c == 0 {
        return 0.0;
    }
    let inv = 1.0 / (members.len() as f32);
    let mut acc = vec![0f32; c];
    for &j in members {
        for i in 0..c {
            acc[i] += sketch[(i, j)];
        }
    }
    acc.iter().map(|v| (v * inv).abs()).sum::<f32>() / (c as f32)
}

/// Score one k-means cluster (no keep/drop decision).
pub fn score_cluster(
    members: &[usize],
    sketch: &DMatrix<f32>,
    donor_of_cell: &[usize],
    n_per_donor: &[usize],
    cfg: &CloneCallConfig,
    rng: &mut impl RngExt,
) -> ClusterScore {
    let n = members.len();
    if n == 0 {
        return ClusterScore {
            n_cells: 0,
            purity: 0.0,
            enrich: 0.0,
            spatial_score: 0.0,
            spatial_z: 0.0,
            clone_score: 0.0,
        };
    }
    let mut counts = vec![0usize; n_per_donor.len()];
    for &j in members {
        counts[donor_of_cell[j]] += 1;
    }
    let (maj_d, maj_n) = counts
        .iter()
        .copied()
        .enumerate()
        .max_by_key(|(_, c)| *c)
        .unwrap_or((0, 0));
    let purity = maj_n as f32 / n as f32;
    let p_enrich = hypergeom_sf(maj_n, n, n_per_donor[maj_d], donor_of_cell.len()).max(1e-300);
    let enrich = (-p_enrich.ln() / std::f64::consts::LN_10) as f32;

    let obs = mean_l1(sketch, members);
    let n_cells_total = sketch.ncols();
    let mut pool: Vec<usize> = (0..n_cells_total).collect();
    let mut nulls = Vec::with_capacity(cfg.n_perm);
    for _ in 0..cfg.n_perm {
        pool.shuffle(rng);
        nulls.push(mean_l1(sketch, &pool[..n.min(n_cells_total)]));
    }
    let mu = nulls.iter().sum::<f32>() / (nulls.len().max(1) as f32);
    let var = nulls.iter().map(|x| (x - mu).powi(2)).sum::<f32>() / (nulls.len().max(1) as f32);
    let sd = var.sqrt().max(1e-6);
    let z = (obs - mu) / sd;
    // Enclosure × structure. Enrichment down-weights chance purity in a
    // donor-imbalanced cohort; softplus(z) keeps flat clusters near zero.
    let spatial_pos = if z > 20.0 { z } else { (1.0 + z.exp()).ln() };
    let clone_score = purity * enrich.max(0.0) * spatial_pos;
    ClusterScore {
        n_cells: n,
        purity,
        enrich,
        spatial_score: obs,
        spatial_z: z,
        clone_score,
    }
}

/// BIC-select a Gaussian mixture on `clone_score` over `K = 1..=n_eligible`
/// (eligible = clusters with ≥ `min_cells`) and return which belong to a
/// non-background component.
///
/// K=1 wins ⇒ nothing is a clone. For K≥2 the lowest-mean component is the
/// mixable background; clusters whose score is nearest another component are
/// kept (optional purity / segmental-z floors can only reject, never invent).
pub fn select_clones_by_mixture(
    scores: &[ClusterScore],
    cfg: &CloneCallConfig,
) -> (Vec<bool>, Option<(f32, f32)>) {
    let mut keep = vec![false; scores.len()];
    let eligible: Vec<usize> = scores
        .iter()
        .enumerate()
        .filter(|(_, s)| s.n_cells >= cfg.min_cells)
        .map(|(i, _)| i)
        .collect();
    if eligible.len() < 2 {
        return (keep, None);
    }
    let values: Vec<f32> = eligible.iter().map(|&i| scores[i].clone_score).collect();
    let k_range: Vec<usize> = (1..=eligible.len()).collect();
    let (k, means, _vars, _weights) = select_kmeans_k(&values, &k_range);
    let lo = means.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = means.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    log::info!(
        "clone-score mixture: BIC-selected K={k} of 1..{}; means=[{}]",
        eligible.len(),
        means
            .iter()
            .map(|m| format!("{m:.3}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    if k <= 1 {
        return (keep, None);
    }
    let low_idx = means
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    for &i in &eligible {
        let s = &scores[i];
        let nearest = means
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                (*a - s.clone_score)
                    .abs()
                    .partial_cmp(&(*b - s.clone_score).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(j, _)| j)
            .unwrap_or(low_idx);
        let mut ok = nearest != low_idx;
        if let Some(p) = cfg.min_purity {
            ok &= s.purity >= p;
        }
        if let Some(z) = cfg.spatial_z {
            ok &= s.spatial_z >= z;
        }
        keep[i] = ok;
    }
    (keep, Some((lo, hi)))
}

/// Cluster cells on a chromosome sketch and map failures to stratum 0.
pub fn call_clones_from_sketch(
    sketch: &DMatrix<f32>,
    cell_names: &[Box<str>],
    cfg: &CloneCallConfig,
) -> Vec<CloneRow> {
    let n = sketch.ncols();
    assert_eq!(n, cell_names.len());
    let donors: Vec<Box<str>> = cell_names.iter().map(|c| donor_of(c).into()).collect();
    let mut donor_id: FxHashMap<Box<str>, usize> = FxHashMap::default();
    let mut donor_of_cell = Vec::with_capacity(n);
    let mut n_per_donor: Vec<usize> = Vec::new();
    for d in &donors {
        let id = *donor_id.entry(d.clone()).or_insert_with(|| {
            n_per_donor.push(0);
            n_per_donor.len() - 1
        });
        n_per_donor[id] += 1;
        donor_of_cell.push(id);
    }

    let k = cfg.k_max.max(1).min(n.max(1));
    let labels = if k <= 1 || sketch.nrows() == 0 {
        vec![0usize; n]
    } else {
        sketch.kmeans_columns(KmeansArgs {
            num_clusters: k,
            max_iter: cfg.kmeans_iter,
        })
    };

    let mut members: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (j, &lab) in labels.iter().enumerate() {
        members[lab.min(k - 1)].push(j);
    }

    let mut rng = SmallRng::seed_from_u64(cfg.seed);
    let mut scores = Vec::with_capacity(k);
    for mem in &members {
        scores.push(score_cluster(
            mem,
            sketch,
            &donor_of_cell,
            &n_per_donor,
            cfg,
            &mut rng,
        ));
    }
    let (cluster_keep, mix) = select_clones_by_mixture(&scores, cfg);
    if let Some((lo, hi)) = mix {
        log::info!("clone-score mixture background..clone means {lo:.3} .. {hi:.3}");
    } else {
        log::info!("clone-score mixture prefers a single component → no clones");
    }

    for (lab, s) in scores.iter().enumerate() {
        if s.n_cells == 0 {
            continue;
        }
        log::info!(
            "CNV cluster {lab}: n={}, purity={:.3}, enrich={:.1}, spatial={:.4} z={:.2}, score={:.3} → {}",
            s.n_cells,
            s.purity,
            s.enrich,
            s.spatial_score,
            s.spatial_z,
            s.clone_score,
            if cluster_keep[lab] { "clone" } else { "bucket 0" }
        );
    }

    // Stable remap of kept clusters → 1..K by descending size.
    let mut kept: Vec<usize> = cluster_keep
        .iter()
        .enumerate()
        .filter(|(_, k)| **k)
        .map(|(i, _)| i)
        .collect();
    kept.sort_by_key(|&i| std::cmp::Reverse(members[i].len()));
    let mut remap = vec![0usize; k];
    for (new_id, &old) in kept.iter().enumerate() {
        remap[old] = new_id + 1;
    }

    (0..n)
        .map(|j| {
            let lab = labels[j].min(k - 1);
            let s = &scores[lab];
            CloneRow {
                cell: cell_names[j].clone(),
                donor: donors[j].clone(),
                cluster: lab,
                stratum: remap[lab],
                purity: s.purity,
                spatial_score: s.spatial_score,
                spatial_z: s.spatial_z,
                burden: 0.0,
                p_malig: if remap[lab] > 0 { 1.0 } else { 0.0 },
            }
        })
        .collect()
}

/// Stream a CNV backend (rows = `chr:start-end`) into a genomic sketch and
/// call clones. Resolves per-cell burden from the `{prefix}.cells.parquet`
/// hints, else mean |col| fused into the sketch pass.
pub fn call_clones(data: &SparseIoVec, cfg: &CloneCallConfig) -> anyhow::Result<Vec<CloneRow>> {
    call_clones_with_burden(data, cfg, None, &[])
}

/// Like [`call_clones`], with optional precomputed burden and
/// `{prefix}.cells.parquet` paths to try (first hit wins) before the fused
/// mean |col|.
pub fn call_clones_with_burden(
    data: &SparseIoVec,
    cfg: &CloneCallConfig,
    burden: Option<&[f32]>,
    cells_table_hints: &[&str],
) -> anyhow::Result<Vec<CloneRow>> {
    anyhow::ensure!(data.num_columns() > 0, "no cells to call clones on");
    anyhow::ensure!(
        cfg.engine != CloneEngine::Bayes || cfg.n_sweeps > 0,
        "--n-sweeps must be >= 1 for the Bayes engine (no posterior samples otherwise)"
    );
    let row_names = data.row_names()?;
    let coords = parse_peak_coordinates(&row_names);
    let (row_bin, names, inv) = compact_sketch_bins(&coords, cfg.bin_size);
    anyhow::ensure!(
        !names.is_empty(),
        "CNV backend rows are not genomic intervals (`chr:start-end`)"
    );
    let n = data.num_columns();
    let n_rows = data.num_rows().max(1);
    let inv_rows = 1.0 / n_rows as f32;
    let c = names.len();
    let mut sketch = DMatrix::<f32>::zeros(c, n);
    // Fuse mean |CNV| while sketching so we never re-read the backend for burden.
    let mut fused_burden = vec![0f32; n];
    let blocks = matrix_util::utils::generate_minibatch_intervals(n, 0, Some(512));
    let bar = styled_progress_bar(blocks.len() as u64, "clone sketch blocks");
    for (lb, ub) in blocks {
        let csc = data.read_columns_csc(lb..ub)?;
        let block = sketch_csc(&csc, &row_bin, c, &inv);
        sketch.columns_mut(lb, ub - lb).copy_from(&block);
        for (offset, dest) in fused_burden[lb..ub].iter_mut().enumerate() {
            let s: f32 = csc.col(offset).values().iter().map(|v| v.abs()).sum();
            *dest = s * inv_rows;
        }
        bar.inc(1);
    }
    bar.finish_and_clear();
    log::info!(
        "clone sketch: {} bins × {} cells (bin_size={})",
        c,
        n,
        cfg.bin_size
    );
    let cell_names = data.column_names()?;

    let burden_owned = if let Some(b) = burden {
        anyhow::ensure!(b.len() == n, "burden length {} != n_cells {}", b.len(), n);
        None
    } else {
        let mut loaded = None;
        let mut tried: Vec<&str> = Vec::new();
        for path in cells_table_hints {
            if path.is_empty() || tried.contains(path) {
                continue;
            }
            tried.push(path);
            if let Some(b) = load_burden_from_cells_table(path, &cell_names, &fused_burden)? {
                log::info!("loaded burden from {path}");
                loaded = Some(b);
                break;
            }
        }
        if loaded.is_none() {
            log::info!("using mean |CNV| fused from the sketch pass");
        }
        Some(loaded.unwrap_or(fused_burden))
    };
    let burden_slice: &[f32] = burden.unwrap_or_else(|| burden_owned.as_deref().unwrap());

    match cfg.engine {
        CloneEngine::Bayes => {
            let bcfg = bayes_config_from_clone_call(cfg);
            Ok(call_clones_bayes(&sketch, &cell_names, burden_slice, &bcfg))
        }
        CloneEngine::Mixture => {
            let mut rows = call_clones_from_sketch(&sketch, &cell_names, cfg);
            for (r, &b) in rows.iter_mut().zip(burden_slice) {
                r.burden = b;
            }
            Ok(rows)
        }
    }
}

/// Write `{out}.clones.parquet`: one row per cell with `cell`, `donor`,
/// `cluster`, `stratum`, `purity`, `spatial_score`, `spatial_z`, `burden`,
/// `p_malig`.
pub fn write_clone_table(rows: &[CloneRow], path: &str) -> anyhow::Result<()> {
    let cell: Vec<Box<str>> = rows.iter().map(|r| r.cell.clone()).collect();
    let donor: Vec<Box<str>> = rows.iter().map(|r| r.donor.clone()).collect();
    let cluster: Vec<i32> = rows.iter().map(|r| r.cluster as i32).collect();
    let stratum: Vec<i32> = rows.iter().map(|r| r.stratum as i32).collect();
    let purity: Vec<f32> = rows.iter().map(|r| r.purity).collect();
    let spatial_score: Vec<f32> = rows.iter().map(|r| r.spatial_score).collect();
    let spatial_z: Vec<f32> = rows.iter().map(|r| r.spatial_z).collect();
    let burden: Vec<f32> = rows.iter().map(|r| r.burden).collect();
    let p_malig: Vec<f32> = rows.iter().map(|r| r.p_malig).collect();
    write_named_table(
        path,
        "cell",
        &cell,
        &[
            ("donor".into(), Column::Str(&donor)),
            ("cluster".into(), Column::I32(&cluster)),
            ("stratum".into(), Column::I32(&stratum)),
            ("purity".into(), Column::F32(&purity)),
            ("spatial_score".into(), Column::F32(&spatial_score)),
            ("spatial_z".into(), Column::F32(&spatial_z)),
            ("burden".into(), Column::F32(&burden)),
            ("p_malig".into(), Column::F32(&p_malig)),
        ],
    )
}

/// Read a table written by [`write_clone_table`].
pub fn read_clone_table(path: &str) -> anyhow::Result<Vec<CloneRow>> {
    let (strs, nums) = read_table_columns(
        path,
        &["cell", "donor"],
        &[
            "cluster",
            "stratum",
            "purity",
            "spatial_score",
            "spatial_z",
            "burden",
            "p_malig",
        ],
    )?;
    let n = strs[0].len();
    let rows = (0..n)
        .map(|i| {
            let cluster = nums[0][i];
            let stratum = nums[1][i];
            anyhow::ensure!(
                cluster >= 0.0 && stratum >= 0.0,
                "{path}: row {i}: negative cluster/stratum"
            );
            Ok(CloneRow {
                cell: strs[0][i].clone(),
                donor: strs[1][i].clone(),
                cluster: cluster as usize,
                stratum: stratum as usize,
                purity: nums[2][i] as f32,
                spatial_score: nums[3][i] as f32,
                spatial_z: nums[4][i] as f32,
                burden: nums[5][i] as f32,
                p_malig: nums[6][i] as f32,
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(rows)
}

/// Map a clone table onto `cell_names` (collapse column order). Missing
/// cells become stratum 0.
pub fn align_strata_to_cells(
    rows: &[CloneRow],
    cell_names: &[Box<str>],
) -> anyhow::Result<Vec<usize>> {
    let mut by_cell: FxHashMap<&str, usize> = FxHashMap::default();
    for r in rows {
        by_cell.insert(r.cell.as_ref(), r.stratum);
    }
    let mut missing = 0usize;
    let out: Vec<usize> = cell_names
        .iter()
        .map(|c| {
            by_cell.get(c.as_ref()).copied().unwrap_or_else(|| {
                missing += 1;
                0
            })
        })
        .collect();
    if missing > 0 {
        log::warn!(
            "{} / {} cells missing from the clone table; treating as stratum 0",
            missing,
            cell_names.len()
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngExt;

    #[test]
    fn sketch_buckets_match_infercnv_bin_size() {
        let coords = [
            Some(PeakCoord {
                chr: "chr1".into(),
                start: 100,
                end: 200,
            }),
            Some(PeakCoord {
                chr: "chr1".into(),
                start: 1_500_000,
                end: 1_500_100,
            }),
            Some(PeakCoord {
                chr: "chr2".into(),
                start: 10,
                end: 20,
            }),
            None,
        ];
        let (row_chr, names_chr, _) = compact_sketch_bins(&coords, 0);
        assert_eq!(names_chr, vec!["chr1".into(), "chr2".into()]);
        assert_eq!(row_chr, vec![Some(0), Some(0), Some(1), None]);

        let (row_mb, names_mb, _) = compact_sketch_bins(&coords, 1_000_000);
        assert_eq!(
            names_mb,
            vec![
                "chr1:0-1000000".into(),
                "chr1:1000000-2000000".into(),
                "chr2:0-1000000".into(),
            ]
        );
        assert_eq!(row_mb, vec![Some(0), Some(1), Some(2), None]);
    }

    #[test]
    fn donor_of_uses_at_suffix() {
        assert_eq!(donor_of("AAACCTG@AML001"), "AML001");
        assert_eq!(donor_of("plain"), "plain");
        assert_eq!(donor_of("a@b@c"), "c");
    }

    #[test]
    fn chromosome_sketch_means_per_chr() {
        // 4 rows: chr1, chr1, chr2, chr2. One cell: values 1,3,10,30 → chr1=2, chr2=20.
        let m = DMatrix::from_column_slice(4, 1, &[1.0, 3.0, 10.0, 30.0]);
        let (s, names) = chromosome_sketch(&m, &["chr1", "chr1", "chr2", "chr2"]);
        assert_eq!(names, vec!["chr1".into(), "chr2".into()]);
        assert!((s[(0, 0)] - 2.0).abs() < 1e-6);
        assert!((s[(1, 0)] - 20.0).abs() < 1e-6);
    }

    fn synthetic_sketch(n_flat: usize, n_clone: usize) -> (DMatrix<f32>, Vec<Box<str>>) {
        // 3 chromosomes. Flat cells ~ N(0, 0.02); clone cells chr0 = -1.
        let n = n_flat + n_clone;
        let mut rng = SmallRng::seed_from_u64(7);
        let mut data = Vec::with_capacity(3 * n);
        let mut names = Vec::with_capacity(n);
        for i in 0..n_flat {
            for _ in 0..3 {
                data.push(0.02 * (rng.random::<f32>() - 0.5));
            }
            names.push(format!("c{i}@Ctl").into_boxed_str());
        }
        for i in 0..n_clone {
            data.push(-1.0 + 0.02 * (rng.random::<f32>() - 0.5));
            data.push(0.02 * (rng.random::<f32>() - 0.5));
            data.push(0.02 * (rng.random::<f32>() - 0.5));
            names.push(format!("t{i}@AML").into_boxed_str());
        }
        (DMatrix::from_vec(3, n, data), names)
    }

    #[test]
    fn donor_private_structured_cluster_becomes_a_clone() {
        let (sketch, names) = synthetic_sketch(80, 80);
        let rows = call_clones_from_sketch(
            &sketch,
            &names,
            &CloneCallConfig {
                engine: CloneEngine::Mixture,
                k_max: 2,
                min_cells: 20,
                min_purity: None,
                spatial_z: None,
                n_perm: 24,
                seed: 3,
                kmeans_iter: 50,
                ..Default::default()
            },
        );
        let n_clone = rows.iter().filter(|r| r.stratum > 0).count();
        let aml_cloned = rows
            .iter()
            .filter(|r| r.donor.as_ref() == "AML" && r.stratum > 0)
            .count();
        let ctl_cloned = rows
            .iter()
            .filter(|r| r.donor.as_ref() == "Ctl" && r.stratum > 0)
            .count();
        assert!(n_clone >= 50, "expected a clone, got {n_clone} cells");
        assert!(aml_cloned >= 50, "AML cells should form the clone");
        assert!(
            ctl_cloned < 10,
            "Control cells should stay in bucket 0, got {ctl_cloned}"
        );
    }

    #[test]
    fn shared_shift_across_donors_dumps_to_bucket_zero() {
        // Same chr0 = -1 in BOTH donors → not donor-enclosing.
        let n = 80;
        let mut data = Vec::new();
        let mut names = Vec::new();
        for d in ["A", "B"] {
            for i in 0..n {
                data.extend_from_slice(&[-1.0f32, 0.0, 0.0]);
                names.push(format!("c{i}@{d}").into_boxed_str());
            }
        }
        let sketch = DMatrix::from_vec(3, 2 * n, data);
        let rows = call_clones_from_sketch(
            &sketch,
            &names,
            &CloneCallConfig {
                engine: CloneEngine::Mixture,
                k_max: 2,
                min_cells: 20,
                min_purity: None,
                spatial_z: None,
                n_perm: 16,
                seed: 1,
                kmeans_iter: 50,
                ..Default::default()
            },
        );
        assert!(
            rows.iter().all(|r| r.stratum == 0),
            "shared pattern must dump to stratum 0"
        );
    }

    #[test]
    fn align_missing_cells_default_to_zero() {
        let rows = vec![CloneRow {
            cell: "a".into(),
            donor: "d".into(),
            cluster: 0,
            stratum: 2,
            purity: 1.0,
            spatial_score: 1.0,
            spatial_z: 3.0,
            burden: 0.1,
            p_malig: 0.9,
        }];
        let names: Vec<Box<str>> = ["a", "b"].map(Into::into).to_vec();
        let s = align_strata_to_cells(&rows, &names).unwrap();
        assert_eq!(s, vec![2, 0]);
    }

    #[test]
    fn clone_table_roundtrip_keeps_burden_and_p_malig() {
        let dir = std::env::temp_dir();
        let path = dir.join("cnv_clone_roundtrip.parquet");
        let path = path.to_str().unwrap();
        let rows = vec![CloneRow {
            cell: "a@D".into(),
            donor: "D".into(),
            cluster: 1,
            stratum: 1,
            purity: 0.91,
            spatial_score: 0.12,
            spatial_z: 2.5,
            burden: 0.234567,
            p_malig: 0.8765,
        }];
        write_clone_table(&rows, path).unwrap();
        let got = read_clone_table(path).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].cell.as_ref(), "a@D");
        assert_eq!(got[0].stratum, 1);
        assert!((got[0].burden - 0.234567).abs() < 1e-6);
        assert!((got[0].p_malig - 0.8765).abs() < 1e-4);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn hypergeom_sf_known_values() {
        // P(X >= 5) for Hypergeometric(N=10, K=5, n=5) = P(X=5) = 1/C(10,5)*C(5,5)*C(5,0)
        let p = hypergeom_sf(5, 5, 5, 10);
        let want = (ln_choose(5, 5) + ln_choose(5, 0) - ln_choose(10, 5)).exp();
        assert!((p - want).abs() < 1e-9, "{p} vs {want}");
        assert!((hypergeom_sf(0, 5, 5, 10) - 1.0).abs() < 1e-12);
    }
}
