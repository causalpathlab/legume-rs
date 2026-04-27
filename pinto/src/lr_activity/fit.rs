//! Pseudobulk + propensity-stratified sample-permutation LR activity test.
//!
//! Cells are collapsed into pseudobulk samples = (batch × propensity-bin),
//! where the propensity bin is the sign-LSH binary-sort code from an
//! SVD'd random projection of gene expression (within-batch centred).
//! Each cell carries soft community membership in two roles: `p_send[i, c]`
//! is the fraction of i's incident edges in community c on which i acts
//! as sender; `p_recv[i, c]` is the receiver-side analogue. Per (community,
//! sample) we accumulate role-weighted gene sums for the LR genes, giving
//! sender and receiver pseudobulk profiles per sample.
//!
//! For each (batch, community, LR pair) the statistic is a **weighted
//! Spearman** rank correlation between sender-pseudobulk L and
//! receiver-pseudobulk R across the samples in that batch (sample weight
//! = sqrt(send_weight · recv_weight)). Spearman over rank-transformed
//! values is used instead of weighted Pearson to suppress saturation when
//! one or two samples carry most of the community-level mass.
//!
//! The null is sample-level permutation of L within propensity-stratified
//! buckets — shuffles are restricted to samples sharing the top
//! `shuffle_stratify_dim` bits of the propensity code, so the cell-type
//! marginal is preserved across permutations. R and per-sample weights are
//! held fixed. One-sided positive p (parametric Gaussian tail of z plus
//! empirical 1/(n+1)-floored permutation p); BH on the parametric p
//! within batch.

use crate::lr_activity::args::SrtLrActivityArgs;
use crate::lr_activity::io::*;
use crate::lr_activity::outputs::{
    bh_qvalues, write_lr_activity, write_lr_activity_json, LrActivityRow, StratumEntry,
};
use crate::util::common::*;
use data_beans::convert::try_open_or_convert;
use data_beans_alg::random_projection::{binary_sort_columns, RandProjOps};
use matrix_util::membership::GeneIndexResolver;
use nalgebra::DMatrix;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use special::Error as SpecialError;

/// One-sided upper-tail Gaussian p-value: P(Z >= z) for standard normal.
#[inline]
fn one_sided_p_z(z: f32) -> f32 {
    // erfc(z/√2) / 2; numerically stable for large |z|.
    let z = z as f64;
    let p = 0.5 * SpecialError::compl_error(z / std::f64::consts::SQRT_2);
    (p as f32).clamp(0.0, 1.0)
}

/// Pseudo-batch label written when no per-edge batch is on file.
pub const BATCH_LABEL_ALL: &str = "all";
/// Pseudo-batch label for pooled-across-batches rows emitted alongside per-batch rows.
pub const BATCH_LABEL_META: &str = "pooled";

/// Pseudocount added to per-(community, sample) role weights when forming
/// the pseudobulk mean. Keeps low-evidence samples from blowing up.
const PRIOR_PSEUDOCOUNT: f32 = 1.0;
/// Floor to avoid div-by-zero when a sample has no presence in a community.
const EPS: f32 = 1e-8;
/// Don't compute a statistic when the stratum has fewer than this many
/// samples — correlation + permutation are too noisy below this.
const MIN_SAMPLES_PER_STRATUM: usize = 4;

pub fn fit_srt_lr_activity(args: &SrtLrActivityArgs) -> anyhow::Result<()> {
    let c = &args.common;

    //////////////////////////////////////////////////////
    // 1. Load expression + resolve gene index map
    //////////////////////////////////////////////////////
    info!("Loading expression data...");
    let data_vec = load_expr_data(c)?;

    let gene_names = data_vec.row_names()?;
    let cell_names = data_vec.column_names()?;
    let n_cells = data_vec.num_columns();
    let n_genes = data_vec.num_rows();

    let gene_resolver =
        GeneIndexResolver::build(&gene_names, args.gene_delimiter, args.gene_allow_prefix);
    let cell_to_col: HashMap<Box<str>, usize> = cell_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();

    //////////////////////////////////////////////////////
    // 2. Parse LR pairs, resolve genes
    //////////////////////////////////////////////////////
    info!("Reading LR pairs from {}...", &args.lr_pairs);
    let raw_pairs = read_lr_pairs(&args.lr_pairs)?;
    let mut resolved_pairs: Vec<(Box<str>, Box<str>, usize, usize)> = Vec::new();
    let mut missing = 0usize;
    for (l, r) in raw_pairs {
        match (gene_resolver.resolve(&l), gene_resolver.resolve(&r)) {
            (Some(li), Some(ri)) => resolved_pairs.push((l, r, li, ri)),
            _ => missing += 1,
        }
    }
    if missing > 0 {
        info!("Skipped {} LR pairs with unresolved gene names", missing);
    }
    anyhow::ensure!(
        !resolved_pairs.is_empty(),
        "no LR pairs resolved against expression row names"
    );
    info!("Resolved {} LR pairs", resolved_pairs.len());

    //////////////////////////////////////////////////////
    // 3. Read edges + batches from prior `pinto lc` run
    //////////////////////////////////////////////////////
    let lc_edges_path = format!("{}.link_community.parquet", &args.lc_prefix);
    let coord_pairs_path = format!("{}.coord_pairs.parquet", &args.lc_prefix);
    info!("Reading edge assignments from {}", &lc_edges_path);
    let mut edge_records = read_link_community(&lc_edges_path)?;
    info!("Attaching per-edge batch from {}", &coord_pairs_path);
    attach_batch_from_coord_pairs(&mut edge_records, &coord_pairs_path)?;

    let mut edges: Vec<(usize, usize, u32, Option<Box<str>>)> =
        Vec::with_capacity(edge_records.len());
    let mut unresolved = 0usize;
    for e in edge_records {
        match (
            cell_to_col.get(&e.left_cell).copied(),
            cell_to_col.get(&e.right_cell).copied(),
        ) {
            (Some(i), Some(j)) => edges.push((i, j, e.community, e.batch)),
            _ => unresolved += 1,
        }
    }
    if unresolved > 0 {
        info!(
            "Dropped {} edges whose cell names are not in the expression data",
            unresolved
        );
    }
    anyhow::ensure!(
        !edges.is_empty(),
        "no edges resolved against expression data"
    );
    let n_communities = (edges.iter().map(|e| e.2).max().unwrap_or(0) as usize) + 1;
    info!(
        "{} cells, {} edges, {} communities",
        n_cells,
        edges.len(),
        n_communities
    );

    //////////////////////////////////////////////////////
    // 4. Per-gene total counts (filter LR pairs)
    //////////////////////////////////////////////////////
    info!("Computing per-gene total counts...");
    const SUM_CHUNK: usize = 512;
    let mut gene_sum = vec![0.0f32; n_genes];
    for start in (0..n_genes).step_by(SUM_CHUNK) {
        let end = (start + SUM_CHUNK).min(n_genes);
        let mat = data_vec.read_rows_ndarray(start..end)?;
        for r in 0..(end - start) {
            gene_sum[start + r] = mat.row(r).iter().sum();
        }
    }
    let pre_filter_n = resolved_pairs.len();
    let real_pairs: Vec<(Box<str>, Box<str>, usize, usize)> = resolved_pairs
        .into_iter()
        .filter(|(_, _, li, ri)| {
            gene_sum[*li] >= args.min_gene_count && gene_sum[*ri] >= args.min_gene_count
        })
        .collect();
    if pre_filter_n - real_pairs.len() > 0 {
        info!(
            "Dropped {} LR pairs whose L or R has < {} total counts",
            pre_filter_n - real_pairs.len(),
            args.min_gene_count
        );
    }
    anyhow::ensure!(
        !real_pairs.is_empty(),
        "no LR pairs survive --min-gene-count={}",
        args.min_gene_count
    );

    //////////////////////////////////////////////////////
    // 5. Cell→community soft membership (sender / receiver)
    //////////////////////////////////////////////////////
    info!("Building cell→community soft membership...");
    let (p_send, p_recv) = build_role_memberships(&edges, n_cells, n_communities);

    //////////////////////////////////////////////////////
    // 6. Per-cell batch label (modal incident batch)
    //////////////////////////////////////////////////////
    let cell_batch = derive_cell_batch_labels(&edges, n_cells);

    //////////////////////////////////////////////////////
    // 7. Random projection + propensity binary-sort
    //////////////////////////////////////////////////////
    info!(
        "Random projection (dim {}) + propensity binary-sort...",
        args.propensity_dim
    );
    let proj = data_vec.project_columns_with_batch_correction(
        args.propensity_dim,
        c.block_size,
        Some(&cell_batch),
    )?;
    let propbin = binary_sort_columns(&proj.proj, args.propensity_dim)?;
    let (sample_id_per_cell, sample_batch_label, sample_propbin) =
        assign_samples(&cell_batch, &propbin);
    let n_samples = sample_batch_label.len();
    info!(
        "{} pseudobulk samples across {} batches",
        n_samples,
        sample_batch_label
            .iter()
            .map(|b| b.as_ref())
            .collect::<HashSet<_>>()
            .len()
    );

    //////////////////////////////////////////////////////
    // 8. Read just the LR-gene rows (dense, small)
    //////////////////////////////////////////////////////
    let mut lr_genes: Vec<usize> = Vec::new();
    let mut gene_to_local: HashMap<usize, usize> = HashMap::default();
    for &(_, _, li, ri) in &real_pairs {
        for g in [li, ri] {
            if let std::collections::hash_map::Entry::Vacant(e) = gene_to_local.entry(g) {
                e.insert(lr_genes.len());
                lr_genes.push(g);
            }
        }
    }
    info!("Reading {} LR gene rows from backend...", lr_genes.len());
    let x_lr = data_vec.read_rows_dmatrix(lr_genes.iter().copied())?;

    //////////////////////////////////////////////////////
    // 9. Collapse into per-(community, sample) pseudobulk
    //////////////////////////////////////////////////////
    info!(
        "Collapsing into pseudobulk: {} comm × {} samples × {} LR genes...",
        n_communities,
        n_samples,
        lr_genes.len()
    );
    let collapse = collapse_role_pseudobulk(
        &x_lr,
        &sample_id_per_cell,
        &p_send,
        &p_recv,
        n_communities,
        n_samples,
    );

    //////////////////////////////////////////////////////
    // 10. Per-stratum scoring (per-batch and pooled)
    //////////////////////////////////////////////////////
    let mut rows: Vec<LrActivityRow> = Vec::new();
    let mut strata: Vec<StratumEntry> = Vec::new();

    let mut unique_batches: Vec<Box<str>> = sample_batch_label
        .iter()
        .cloned()
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    unique_batches.sort();

    let n_real_batches = unique_batches
        .iter()
        .filter(|b| b.as_ref() != BATCH_LABEL_ALL)
        .count();

    let base_seed = c.seed;

    for batch_label in &unique_batches {
        let samples_in_batch: Vec<usize> = (0..n_samples)
            .filter(|&s| sample_batch_label[s].as_ref() == batch_label.as_ref())
            .collect();
        if samples_in_batch.len() < MIN_SAMPLES_PER_STRATUM {
            continue;
        }
        for community in 0..(n_communities as u32) {
            if !community_present(&collapse, community as usize, &samples_in_batch) {
                continue;
            }
            let stratum_id = strata.len();
            strata.push(stratum_entry(
                batch_label,
                community as i32,
                &edges,
                &cell_names,
                Some(batch_label.as_ref()),
                community,
            ));
            let mut br = score_pairs_for_stratum(
                batch_label,
                community,
                &samples_in_batch,
                &sample_propbin,
                &real_pairs,
                &gene_to_local,
                &gene_names,
                &collapse,
                args.n_permutations,
                base_seed.wrapping_add(community as u64 * 1_000_003),
                stratum_id,
                /*shuffle_within_batch=*/ false,
                &sample_batch_label,
                args.shuffle_stratify_dim,
            );
            rows.append(&mut br);
        }
    }

    if n_real_batches > 1 {
        let all_samples: Vec<usize> = (0..n_samples).collect();
        let meta_label: Box<str> = BATCH_LABEL_META.into();
        for community in 0..(n_communities as u32) {
            if !community_present(&collapse, community as usize, &all_samples) {
                continue;
            }
            let stratum_id = strata.len();
            strata.push(stratum_entry(
                &meta_label,
                community as i32,
                &edges,
                &cell_names,
                None,
                community,
            ));
            let mut mr = score_pairs_for_stratum(
                &meta_label,
                community,
                &all_samples,
                &sample_propbin,
                &real_pairs,
                &gene_to_local,
                &gene_names,
                &collapse,
                args.n_permutations,
                base_seed
                    .wrapping_add(0xDEAD_BEEF)
                    .wrapping_add(community as u64),
                stratum_id,
                /*shuffle_within_batch=*/ true,
                &sample_batch_label,
                args.shuffle_stratify_dim,
            );
            rows.append(&mut mr);
        }
    }

    //////////////////////////////////////////////////////
    // 11. BH within each batch stratum
    //////////////////////////////////////////////////////
    apply_bh_per_batch(&mut rows);

    //////////////////////////////////////////////////////
    // 12. Write output
    //////////////////////////////////////////////////////
    let out_path = format!("{}.lr_activity.parquet", &c.out);
    info!("Writing {} rows to {}", rows.len(), out_path);
    write_lr_activity(&out_path, &rows)?;

    if args.emit_json {
        let json_path = format!("{}.lr_activity.json", &c.out);
        let upstream_meta_path = format!("{}.metadata.json", &args.lc_prefix);
        let upstream_meta =
            crate::util::metadata::PintoMetadata::read(std::path::Path::new(&upstream_meta_path))
                .ok();
        write_lr_activity_json(
            &json_path,
            args.lc_prefix.as_ref(),
            upstream_meta.as_ref().map(|_| upstream_meta_path.as_str()),
            &rows,
            &strata,
            args.json_q_threshold,
        )?;
        info!("Wrote {}", json_path);

        if let Some(mut meta) = upstream_meta {
            meta.outputs.lr_activity = Some(json_path.clone());
            let _ = meta.write(std::path::Path::new(&upstream_meta_path));
        }
    }

    Ok(())
}

fn load_expr_data(c: &crate::util::input::SrtInputArgs) -> anyhow::Result<SparseIoVec> {
    anyhow::ensure!(!c.data_files.is_empty(), "empty data files");
    let attach_data_name = c.data_files.len() > 1;
    let mut data_vec = SparseIoVec::new();
    for data_file in c.data_files.iter() {
        info!("Importing data file: {}", data_file);
        let mut data = try_open_or_convert(data_file)?;
        if c.preload_data {
            info!("Preloading columns for {}", data_file);
            data.preload_columns()?;
        }
        let data_name = attach_data_name
            .then(|| matrix_util::common_io::basename(data_file))
            .transpose()?;
        data_vec.push(Arc::from(data), data_name)?;
    }
    Ok(data_vec)
}

/// Per-cell soft community membership matrices for the two roles.
/// Row-normalised so each cell's per-role mass sums to 1 (over communities
/// that the cell touches in that role).
fn build_role_memberships(
    edges: &[(usize, usize, u32, Option<Box<str>>)],
    n_cells: usize,
    n_communities: usize,
) -> (DMatrix<f32>, DMatrix<f32>) {
    let mut p_send = DMatrix::<f32>::zeros(n_cells, n_communities);
    let mut p_recv = DMatrix::<f32>::zeros(n_cells, n_communities);
    for &(i, j, k, _) in edges {
        p_send[(i, k as usize)] += 1.0;
        p_recv[(j, k as usize)] += 1.0;
    }
    for i in 0..n_cells {
        let s = p_send.row(i).sum();
        if s > 0.0 {
            p_send.row_mut(i).scale_mut(1.0 / s);
        }
        let r = p_recv.row(i).sum();
        if r > 0.0 {
            p_recv.row_mut(i).scale_mut(1.0 / r);
        }
    }
    (p_send, p_recv)
}

/// Modal batch among edges incident to each cell. Cells with no batched
/// edges fall back to `BATCH_LABEL_ALL` (also the default for runs where
/// `coord_pairs.parquet` carried no batch columns).
fn derive_cell_batch_labels(
    edges: &[(usize, usize, u32, Option<Box<str>>)],
    n_cells: usize,
) -> Vec<Box<str>> {
    let mut counts: Vec<HashMap<Box<str>, usize>> =
        (0..n_cells).map(|_| HashMap::default()).collect();
    for (i, j, _k, b_opt) in edges {
        if let Some(b) = b_opt {
            *counts[*i].entry(b.clone()).or_insert(0) += 1;
            *counts[*j].entry(b.clone()).or_insert(0) += 1;
        }
    }
    counts
        .into_iter()
        .map(|m| {
            m.into_iter()
                .max_by_key(|(_, c)| *c)
                .map(|(b, _)| b)
                .unwrap_or_else(|| BATCH_LABEL_ALL.into())
        })
        .collect()
}

/// Group cells into pseudobulk samples = unique (batch, propensity-bin)
/// combinations. Returns (sample_id_per_cell, per-sample batch label,
/// per-sample propensity-bin).
fn assign_samples(
    cell_batch: &[Box<str>],
    propbin: &[usize],
) -> (Vec<usize>, Vec<Box<str>>, Vec<usize>) {
    let mut key_to_id: HashMap<(Box<str>, usize), usize> = HashMap::default();
    let mut sample_batch_label: Vec<Box<str>> = Vec::new();
    let mut sample_propbin: Vec<usize> = Vec::new();
    let mut sample_id_per_cell: Vec<usize> = Vec::with_capacity(cell_batch.len());
    for i in 0..cell_batch.len() {
        let key = (cell_batch[i].clone(), propbin[i]);
        let id = if let Some(&id) = key_to_id.get(&key) {
            id
        } else {
            let id = sample_batch_label.len();
            sample_batch_label.push(key.0.clone());
            sample_propbin.push(key.1);
            key_to_id.insert(key, id);
            id
        };
        sample_id_per_cell.push(id);
    }
    (sample_id_per_cell, sample_batch_label, sample_propbin)
}

/// Per-(community, sample) role-weighted pseudobulk for the LR genes.
/// `num_send[c]` is `(n_lr_genes × n_samples)`; `denom_send[c][s]` is the
/// total p_send weight of cells belonging to sample `s`.
struct CollapseOut {
    num_send: Vec<DMatrix<f32>>,
    denom_send: Vec<Vec<f32>>,
    num_recv: Vec<DMatrix<f32>>,
    denom_recv: Vec<Vec<f32>>,
}

fn collapse_role_pseudobulk(
    x_lr: &DMatrix<f32>,
    sample_id: &[usize],
    p_send: &DMatrix<f32>,
    p_recv: &DMatrix<f32>,
    n_communities: usize,
    n_samples: usize,
) -> CollapseOut {
    let n_lr = x_lr.nrows();
    let n_cells = x_lr.ncols();

    // Communities are independent — accumulate each one in its own thread.
    // Inverting the loop order (community-outer, cell-inner) lets axpy run
    // without inter-thread races.
    type CommunityCollapse = (DMatrix<f32>, DMatrix<f32>, Vec<f32>, Vec<f32>);
    let per_community: Vec<CommunityCollapse> = (0..n_communities)
        .into_par_iter()
        .map(|c| {
            let mut num_s = DMatrix::<f32>::zeros(n_lr, n_samples);
            let mut num_r = DMatrix::<f32>::zeros(n_lr, n_samples);
            let mut den_s = vec![0.0f32; n_samples];
            let mut den_r = vec![0.0f32; n_samples];
            for i in 0..n_cells {
                let s = sample_id[i];
                let xi = x_lr.column(i);
                let ps = p_send[(i, c)];
                if ps > 0.0 {
                    num_s.column_mut(s).axpy(ps, &xi, 1.0);
                    den_s[s] += ps;
                }
                let pr = p_recv[(i, c)];
                if pr > 0.0 {
                    num_r.column_mut(s).axpy(pr, &xi, 1.0);
                    den_r[s] += pr;
                }
            }
            (num_s, num_r, den_s, den_r)
        })
        .collect();

    let mut num_send = Vec::with_capacity(n_communities);
    let mut num_recv = Vec::with_capacity(n_communities);
    let mut denom_send = Vec::with_capacity(n_communities);
    let mut denom_recv = Vec::with_capacity(n_communities);
    for (ns, nr, ds, dr) in per_community {
        num_send.push(ns);
        num_recv.push(nr);
        denom_send.push(ds);
        denom_recv.push(dr);
    }
    CollapseOut {
        num_send,
        denom_send,
        num_recv,
        denom_recv,
    }
}

fn community_present(collapse: &CollapseOut, c: usize, samples: &[usize]) -> bool {
    samples
        .iter()
        .any(|&s| collapse.denom_send[c][s] > 0.0 && collapse.denom_recv[c][s] > 0.0)
}

/// Pseudobulk mean: numerator / (denominator + prior). Prior keeps
/// low-evidence samples from blowing up under low denominator.
#[inline]
fn pb_mean(num: f32, denom: f32) -> f32 {
    num / (denom + PRIOR_PSEUDOCOUNT)
}

/// Average ranks (1-based, ties → mean rank). Used to convert pseudobulk
/// expression vectors to a ranks before applying weighted Pearson — gives
/// a weighted Spearman without the `±1` saturation that weighted Pearson
/// hits when one sample carries most of the mass.
fn avg_ranks(v: &[f32]) -> Vec<f32> {
    let n = v.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut rank = vec![0.0f32; n];
    let mut k = 0;
    while k < n {
        let mut j = k + 1;
        while j < n && v[idx[j]] == v[idx[k]] {
            j += 1;
        }
        let avg = ((k + j - 1) as f32) / 2.0 + 1.0;
        for &i in &idx[k..j] {
            rank[i] = avg;
        }
        k = j;
    }
    rank
}

/// Weighted Pearson correlation between `l` and `r` using sample weights `w`.
fn weighted_corr(l: &[f32], r: &[f32], w: &[f32]) -> f32 {
    let mut sw = 0.0f32;
    let mut sl = 0.0f32;
    let mut sr = 0.0f32;
    for k in 0..l.len() {
        sw += w[k];
        sl += w[k] * l[k];
        sr += w[k] * r[k];
    }
    if sw <= EPS {
        return f32::NAN;
    }
    let ml = sl / sw;
    let mr = sr / sw;
    let mut cov = 0.0f32;
    let mut vl = 0.0f32;
    let mut vr = 0.0f32;
    for k in 0..l.len() {
        let dl = l[k] - ml;
        let dr = r[k] - mr;
        cov += w[k] * dl * dr;
        vl += w[k] * dl * dl;
        vr += w[k] * dr * dr;
    }
    if vl <= EPS || vr <= EPS {
        return 0.0;
    }
    cov / (vl.sqrt() * vr.sqrt())
}

#[allow(clippy::too_many_arguments)]
fn score_pairs_for_stratum(
    batch_label: &str,
    community: u32,
    samples_in_stratum: &[usize],
    sample_propbin: &[usize],
    real_pairs: &[(Box<str>, Box<str>, usize, usize)],
    gene_to_local: &HashMap<usize, usize>,
    gene_names: &[Box<str>],
    collapse: &CollapseOut,
    n_perm: usize,
    base_seed: u64,
    stratum_id: usize,
    shuffle_within_batch: bool,
    sample_batch_label: &[Box<str>],
    shuffle_stratify_dim: usize,
) -> Vec<LrActivityRow> {
    let c = community as usize;
    let n_s = samples_in_stratum.len();
    if n_s < MIN_SAMPLES_PER_STRATUM {
        return Vec::new();
    }

    let w_send: Vec<f32> = samples_in_stratum
        .iter()
        .map(|&s| collapse.denom_send[c][s])
        .collect();
    let w_recv: Vec<f32> = samples_in_stratum
        .iter()
        .map(|&s| collapse.denom_recv[c][s])
        .collect();
    let w_pair: Vec<f32> = (0..n_s).map(|k| (w_send[k] * w_recv[k]).sqrt()).collect();

    // Permutation buckets: rows of `bucket_idx` give the position-indices
    // (into the stratum's sample list) that may be shuffled together.
    // Pooled strata always shuffle within batch (to preserve batch-level
    // confounders); per-batch strata are unconstrained on the batch axis.
    // When `shuffle_stratify_dim > 0`, we additionally subgroup by the top
    // bits of the propensity binary code so the cell-population marginal
    // is preserved across permutations.
    let strat_mask: usize = if shuffle_stratify_dim == 0 {
        0
    } else {
        (1usize << shuffle_stratify_dim) - 1
    };
    let bucket_idx: Vec<Vec<usize>> = {
        let mut buckets: HashMap<(Box<str>, usize), Vec<usize>> = HashMap::default();
        for (k, &s) in samples_in_stratum.iter().enumerate() {
            let batch_key: Box<str> = if shuffle_within_batch {
                sample_batch_label[s].clone()
            } else {
                Box::from("_")
            };
            let strat_key = sample_propbin[s] & strat_mask;
            buckets.entry((batch_key, strat_key)).or_default().push(k);
        }
        buckets.into_values().collect()
    };

    real_pairs
        .par_iter()
        .enumerate()
        .filter_map(|(pair_idx, (lname, rname, li, ri))| {
            let l_local = *gene_to_local.get(li)?;
            let r_local = *gene_to_local.get(ri)?;

            let l_vec: Vec<f32> = samples_in_stratum
                .iter()
                .map(|&s| {
                    pb_mean(
                        collapse.num_send[c][(l_local, s)],
                        collapse.denom_send[c][s],
                    )
                })
                .collect();
            let r_vec: Vec<f32> = samples_in_stratum
                .iter()
                .map(|&s| {
                    pb_mean(
                        collapse.num_recv[c][(r_local, s)],
                        collapse.denom_recv[c][s],
                    )
                })
                .collect();
            // Rank-transform once; permutation reshuffles the rank vector
            // (equivalent to re-ranking after a value permutation, since
            // permutation only relabels ranks).
            let l_rank = avg_ranks(&l_vec);
            let r_rank = avg_ranks(&r_vec);
            let t_obs = weighted_corr(&l_rank, &r_rank, &w_pair);
            if !t_obs.is_finite() {
                return None;
            }

            let mut rng =
                SmallRng::seed_from_u64(base_seed.wrapping_add(pair_idx as u64 * 1_000_003));
            let mut shuffled = l_rank.clone();
            let mut nulls: Vec<f32> = Vec::with_capacity(n_perm);
            for _ in 0..n_perm {
                permute_in_buckets(&mut shuffled, &l_rank, &bucket_idx, &mut rng);
                let t = weighted_corr(&shuffled, &r_rank, &w_pair);
                if t.is_finite() {
                    nulls.push(t);
                }
            }
            if nulls.len() < n_perm.div_ceil(2) {
                return None;
            }
            let n = nulls.len() as f32;
            let mu = nulls.iter().sum::<f32>() / n;
            let var = nulls.iter().map(|v| (v - mu).powi(2)).sum::<f32>() / (n - 1.0).max(1.0);
            let sd = var.sqrt().max(1e-8);
            let z = (t_obs - mu) / sd;

            // One-sided: positive correlation = LR pair active.
            let n_ge: usize = nulls.iter().filter(|&&v| v >= t_obs).count();
            let p_emp = ((n_ge + 1) as f32) / ((nulls.len() + 1) as f32);
            let p_z = one_sided_p_z(z);

            Some(LrActivityRow {
                batch: Box::from(batch_label),
                community: community as i32,
                ligand: lname.clone(),
                receptor: rname.clone(),
                ligand_resolved: gene_names[*li].clone(),
                receptor_resolved: gene_names[*ri].clone(),
                n_samples: n_s as i32,
                stat_obs: t_obs,
                null_mean: mu,
                null_sd: sd,
                z,
                p_empirical: p_emp.clamp(0.0, 1.0),
                p_z,
                q_bh: f32::NAN,
                stratum_id,
            })
        })
        .collect()
}

fn permute_in_buckets(out: &mut [f32], src: &[f32], buckets: &[Vec<usize>], rng: &mut SmallRng) {
    out.copy_from_slice(src);
    let mut tmp: Vec<f32> = Vec::new();
    for bucket in buckets {
        if bucket.len() < 2 {
            continue;
        }
        tmp.clear();
        tmp.extend(bucket.iter().map(|&k| out[k]));
        tmp.shuffle(rng);
        for (slot, &k) in bucket.iter().enumerate() {
            out[k] = tmp[slot];
        }
    }
}

fn stratum_entry(
    batch_label: &str,
    community: i32,
    edges: &[(usize, usize, u32, Option<Box<str>>)],
    cell_names: &[Box<str>],
    batch_filter: Option<&str>,
    community_filter: u32,
) -> StratumEntry {
    let edges_named: Vec<(Box<str>, Box<str>)> = edges
        .iter()
        .filter(|(_, _, k, b)| {
            *k == community_filter
                && match batch_filter {
                    Some(bf) => b.as_ref().map(|s| s.as_ref()) == Some(bf),
                    None => true,
                }
        })
        .map(|(i, j, _, _)| {
            let l = cell_names
                .get(*i)
                .cloned()
                .unwrap_or_else(|| format!("cell_{i}").into_boxed_str());
            let r = cell_names
                .get(*j)
                .cloned()
                .unwrap_or_else(|| format!("cell_{j}").into_boxed_str());
            (l, r)
        })
        .collect();
    StratumEntry {
        batch: Box::from(batch_label),
        community,
        edges: edges_named,
    }
}

fn apply_bh_per_batch(rows: &mut [LrActivityRow]) {
    let mut by_batch: HashMap<Box<str>, Vec<usize>> = HashMap::default();
    for (i, r) in rows.iter().enumerate() {
        by_batch.entry(r.batch.clone()).or_default().push(i);
    }
    for (_b, idxs) in by_batch {
        let p: Vec<f32> = idxs.iter().map(|&i| rows[i].p_z).collect();
        let q = bh_qvalues(&p);
        for (k, &i) in idxs.iter().enumerate() {
            rows[i].q_bh = q[k];
        }
    }
}
