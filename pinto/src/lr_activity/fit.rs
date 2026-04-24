//! Orchestration for `pinto lr-activity`.

use crate::lr_activity::args::SrtLrActivityArgs;
use crate::lr_activity::entropy::{normalised_conditional_entropy, quantile_edges};
use crate::lr_activity::io::*;
use crate::lr_activity::matcher::{DecoyTarget, MatcherContext};
use crate::lr_activity::moran::global_moran_per_gene;
use crate::lr_activity::outputs::{bh_qvalues, write_lr_activity, LrActivityRow};
use crate::util::common::*;
use data_beans::convert::try_open_or_convert;
use data_beans_alg::union_find::UnionFind;
use matrix_util::membership::GeneIndexResolver;
use rand::rngs::SmallRng;
use rand::SeedableRng;

/// Pseudo-batch label written for a single-batch run (no batch labels on file).
const BATCH_LABEL_ALL: &str = "__all__";
/// Pseudo-batch label for pooled-across-batches rows emitted in addition to per-batch rows.
const BATCH_LABEL_META: &str = "__meta__";

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
    let mut real_pairs: Vec<(Box<str>, Box<str>, usize, usize)> = Vec::new();
    let mut missing = 0usize;
    for (l, r) in raw_pairs {
        match (gene_resolver.resolve(&l), gene_resolver.resolve(&r)) {
            (Some(li), Some(ri)) => real_pairs.push((l, r, li, ri)),
            _ => missing += 1,
        }
    }
    if missing > 0 {
        info!("Skipped {} LR pairs with unresolved gene names", missing);
    }
    anyhow::ensure!(
        !real_pairs.is_empty(),
        "no LR pairs resolved against expression row names"
    );
    info!("Resolved {} LR pairs", real_pairs.len());

    //////////////////////////////////////////////////////
    // 3. Read edges + batches from prior `pinto lc` run
    //////////////////////////////////////////////////////
    let lc_edges_path = format!("{}.link_community.parquet", &args.lc_prefix);
    let coord_pairs_path = format!("{}.coord_pairs.parquet", &args.lc_prefix);
    info!("Reading edge assignments from {}", &lc_edges_path);
    let mut edge_records = read_link_community(&lc_edges_path)?;
    info!("Attaching per-edge batch from {}", &coord_pairs_path);
    attach_batch_from_coord_pairs(&mut edge_records, &coord_pairs_path)?;

    // Map cell names to column indices. Any edge whose endpoints fall outside
    // the loaded expression is dropped (warn once).
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

    let edge_only: Vec<(usize, usize)> = edges.iter().map(|e| (e.0, e.1)).collect();

    //////////////////////////////////////////////////////
    // 4. Identify gene set: real LR genes + decoy pool
    //////////////////////////////////////////////////////
    let mut gene_needed: HashSet<usize> = HashSet::default();
    for &(_, _, li, ri) in &real_pairs {
        gene_needed.insert(li);
        gene_needed.insert(ri);
    }

    info!("Materialising expression for {} genes...", n_genes);
    // For memory headroom, we materialise the full G × N dense matrix once.
    // Downstream callers slice by gene row index. Accumulate per-gene sums in
    // the same pass so we never stride the dense matrix.
    let mut x_gn = Mat::zeros(n_genes, n_cells);
    let mut gene_sum = vec![0.0f32; n_genes];
    {
        let csc = data_vec.read_columns_csc(0..n_cells)?;
        for (col, row_slice, val_slice) in csc_column_iter(&csc) {
            for (row, val) in row_slice.iter().zip(val_slice.iter()) {
                x_gn[(*row, col)] = *val;
                gene_sum[*row] += *val;
            }
        }
    }
    let gene_mean: Vec<f32> = gene_sum.iter().map(|&s| s / (n_cells as f32)).collect();

    // Real-LR genes collapsed into a set to exclude from decoy pool.
    let mut real_gene_set: HashSet<usize> = HashSet::default();
    let mut known_pairs: HashSet<(usize, usize)> = HashSet::default();
    for &(_, _, li, ri) in &real_pairs {
        real_gene_set.insert(li);
        real_gene_set.insert(ri);
        known_pairs.insert((li, ri));
    }

    // Candidate decoy pool: genes above --min-gene-count, not in real LR set.
    let candidate_genes: Vec<usize> = (0..n_genes)
        .filter(|&g| gene_sum[g] >= args.min_gene_count && !real_gene_set.contains(&g))
        .collect();
    anyhow::ensure!(
        !candidate_genes.is_empty(),
        "candidate decoy pool is empty; lower --min-gene-count"
    );
    info!("Decoy candidate pool: {} genes", candidate_genes.len());

    //////////////////////////////////////////////////////
    // 5. Per-gene Moran's I on the pinto edge graph
    //////////////////////////////////////////////////////
    info!("Computing global Moran's I per gene...");
    let moran_full_dvec = global_moran_per_gene(&x_gn, &edge_only);
    let moran_full: Vec<f32> = (0..n_genes).map(|g| moran_full_dvec[g]).collect();

    //////////////////////////////////////////////////////
    // 6. Matcher context
    //////////////////////////////////////////////////////
    let matcher_means: Vec<f32> = candidate_genes.iter().map(|&g| gene_mean[g]).collect();
    let matcher_moran: Vec<f32> = candidate_genes.iter().map(|&g| moran_full[g]).collect();
    let matcher = MatcherContext::new(candidate_genes.clone(), matcher_means, matcher_moran);

    //////////////////////////////////////////////////////
    // 7. Stratify edges by (batch, community) and compute
    //////////////////////////////////////////////////////
    let mut strata: HashMap<(Option<Box<str>>, u32), Vec<usize>> = HashMap::default();
    for (idx, (_i, _j, k, b)) in edges.iter().enumerate() {
        strata.entry((b.clone(), *k)).or_default().push(idx);
    }

    // Collect results into one flat vector, then BH within each batch stratum.
    let mut rows: Vec<LrActivityRow> = Vec::new();
    let base_seed = c.seed;

    // --- per-batch strata ---
    let mut stratum_keys: Vec<(Option<Box<str>>, u32)> = strata.keys().cloned().collect();
    stratum_keys.sort_by(|a, b| {
        let ka = a.0.as_ref().map(|s| s.as_ref()).unwrap_or("");
        let kb = b.0.as_ref().map(|s| s.as_ref()).unwrap_or("");
        (ka, a.1).cmp(&(kb, b.1))
    });

    for stratum_key in &stratum_keys {
        let edge_ids = &strata[stratum_key];
        let (batch_opt, community) = stratum_key.clone();
        let stratum_edges: Vec<(usize, usize)> = edge_ids.iter().map(|&k| edge_only[k]).collect();
        let kept_ccs = extract_kept_ccs(&stratum_edges, n_cells, args.min_cc_edges);
        if kept_ccs.is_empty() {
            continue;
        }
        let batch_label: Box<str> = batch_opt
            .as_ref()
            .cloned()
            .unwrap_or_else(|| BATCH_LABEL_ALL.into());
        let mut batch_rows = score_pairs_parallel(
            &batch_label,
            community as i32,
            &kept_ccs,
            &real_pairs,
            &gene_mean,
            &moran_full,
            &matcher,
            &known_pairs,
            &real_gene_set,
            &x_gn,
            args,
            base_seed,
        );
        rows.append(&mut batch_rows);
    }

    // --- pooled meta rows across batches, one per (community, LR) ---
    let mut by_community: HashMap<u32, Vec<usize>> = HashMap::default();
    for (idx, e) in edges.iter().enumerate() {
        by_community.entry(e.2).or_default().push(idx);
    }
    let mut community_keys: Vec<u32> = by_community.keys().copied().collect();
    community_keys.sort_unstable();

    let meta_label: Box<str> = BATCH_LABEL_META.into();
    for &community in &community_keys {
        let edge_ids = &by_community[&community];
        let stratum_edges: Vec<(usize, usize)> = edge_ids.iter().map(|&k| edge_only[k]).collect();
        let kept_ccs = extract_kept_ccs(&stratum_edges, n_cells, args.min_cc_edges);
        if kept_ccs.is_empty() {
            continue;
        }
        let mut meta_rows = score_pairs_parallel(
            &meta_label,
            community as i32,
            &kept_ccs,
            &real_pairs,
            &gene_mean,
            &moran_full,
            &matcher,
            &known_pairs,
            &real_gene_set,
            &x_gn,
            args,
            base_seed ^ 0xDEAD_BEEF,
        );
        rows.append(&mut meta_rows);
    }

    //////////////////////////////////////////////////////
    // 8. BH within each batch stratum
    //////////////////////////////////////////////////////
    apply_bh_per_batch(&mut rows);

    //////////////////////////////////////////////////////
    // 9. Write output
    //////////////////////////////////////////////////////
    let out_path = format!("{}.lr_activity.parquet", &c.out);
    info!("Writing {} rows to {}", rows.len(), out_path);
    write_lr_activity(&out_path, &rows)?;

    Ok(())
}

/// Aggregated statistics for one (stratum, LR pair) across its kept CCs.
struct AggStat {
    ce_obs: f32,
    null_mean: f32,
    null_sd: f32,
    z: f32,
    p_empirical: f32,
}

/// For each CC: compute `H(R|L)/H(R)` on the real LR pair and on every decoy
/// pair (sharing the per-CC quantile edges), then aggregate observed and null
/// statistics across CCs with inverse-variance weights.
///
/// Returns `None` if no CC produced a valid statistic.
fn aggregate_cc_statistics(
    ccs: &[Vec<(usize, usize)>],
    li: usize,
    ri: usize,
    decoys: &[(usize, usize)],
    x_gn: &Mat,
    n_bins: usize,
) -> Option<AggStat> {
    let mut per_cc_obs: Vec<f32> = Vec::new();
    let mut per_cc_null: Vec<Vec<f32>> = Vec::new();
    let mut per_cc_weight: Vec<f32> = Vec::new();

    let cap = ccs.iter().map(|es| es.len()).max().unwrap_or(0);
    let mut l_buf: Vec<f32> = Vec::with_capacity(cap);
    let mut r_buf: Vec<f32> = Vec::with_capacity(cap);

    for edges in ccs {
        l_buf.clear();
        r_buf.clear();
        l_buf.extend(edges.iter().map(|&(s, _)| x_gn[(li, s)]));
        r_buf.extend(edges.iter().map(|&(_, t)| x_gn[(ri, t)]));
        let bins_l = quantile_edges(&l_buf, n_bins);
        let bins_r = quantile_edges(&r_buf, n_bins);
        let Some(ce_obs) = normalised_conditional_entropy(&l_buf, &r_buf, &bins_l, &bins_r) else {
            continue;
        };
        let mut nulls: Vec<f32> = Vec::with_capacity(decoys.len());
        for &(dlg, drg) in decoys {
            l_buf.clear();
            r_buf.clear();
            l_buf.extend(edges.iter().map(|&(s, _)| x_gn[(dlg, s)]));
            r_buf.extend(edges.iter().map(|&(_, t)| x_gn[(drg, t)]));
            let bins_l_n = quantile_edges(&l_buf, n_bins);
            let bins_r_n = quantile_edges(&r_buf, n_bins);
            if let Some(ce) = normalised_conditional_entropy(&l_buf, &r_buf, &bins_l_n, &bins_r_n) {
                nulls.push(ce);
            }
        }
        if nulls.len() < 10 {
            continue;
        }
        let n = nulls.len() as f32;
        let mu = nulls.iter().sum::<f32>() / n;
        let var = nulls.iter().map(|v| (v - mu).powi(2)).sum::<f32>() / (n - 1.0).max(1.0);
        let weight = if var > 0.0 { 1.0 / var } else { 0.0 };
        if weight <= 0.0 || !weight.is_finite() {
            continue;
        }
        per_cc_obs.push(ce_obs);
        per_cc_null.push(nulls);
        per_cc_weight.push(weight);
    }

    if per_cc_obs.is_empty() {
        return None;
    }

    let w_sum: f32 = per_cc_weight.iter().sum();
    let ce_obs_agg: f32 = per_cc_obs
        .iter()
        .zip(per_cc_weight.iter())
        .map(|(&o, &w)| o * w)
        .sum::<f32>()
        / w_sum;

    // For the aggregated null, combine per-CC nulls by drawing one value per CC
    // per draw (paired across decoys). Decoys are in the same order for every
    // CC so this matches a block-wise pairing and preserves within-draw
    // correlation across CCs.
    let n_draws = per_cc_null.iter().map(|v| v.len()).min().unwrap_or(0);
    if n_draws == 0 {
        return None;
    }
    let mut agg_null: Vec<f32> = Vec::with_capacity(n_draws);
    for d in 0..n_draws {
        let mut numer = 0.0f32;
        for (nulls, &w) in per_cc_null.iter().zip(per_cc_weight.iter()) {
            numer += nulls[d] * w;
        }
        agg_null.push(numer / w_sum);
    }

    let nd = agg_null.len() as f32;
    let mu = agg_null.iter().sum::<f32>() / nd;
    let var = agg_null.iter().map(|v| (v - mu).powi(2)).sum::<f32>() / (nd - 1.0).max(1.0);
    let sd = var.sqrt().max(1e-8);
    let z = (ce_obs_agg - mu) / sd;

    // One-sided: elevated LR activity ⇒ *lower* CE than null.
    let n_le_or_eq: usize = agg_null.iter().filter(|&&v| v <= ce_obs_agg).count();
    let p = ((n_le_or_eq + 1) as f32) / ((n_draws + 1) as f32);

    Some(AggStat {
        ce_obs: ce_obs_agg,
        null_mean: mu,
        null_sd: sd,
        z,
        p_empirical: p.clamp(0.0, 1.0),
    })
}

/// Iterate a CSC matrix column by column, yielding `(col, row_idx_slice, val_slice)`.
fn csc_column_iter(
    csc: &nalgebra_sparse::CscMatrix<f32>,
) -> impl Iterator<Item = (usize, &[usize], &[f32])> {
    let offsets = csc.col_offsets();
    let row_indices = csc.row_indices();
    let values = csc.values();
    (0..csc.ncols()).map(move |col| {
        let start = offsets[col];
        let end = offsets[col + 1];
        (col, &row_indices[start..end], &values[start..end])
    })
}

/// Assign each node a connected-component id over the given edges, using
/// `UnionFind` from `data-beans-alg`. Nodes not touched by any edge keep
/// singleton components (each with a distinct label).
fn connected_components_from_edges(edges: &[(usize, usize)], n: usize) -> Vec<usize> {
    let mut uf = UnionFind::new(n);
    for &(i, j) in edges {
        uf.union(i, j);
    }
    let mut next = 0usize;
    let mut rep_to_label: HashMap<usize, usize> = HashMap::default();
    (0..n)
        .map(|i| {
            let r = uf.find(i);
            *rep_to_label.entry(r).or_insert_with(|| {
                let l = next;
                next += 1;
                l
            })
        })
        .collect()
}

/// Decompose a stratum's edge list into connected components, drop components
/// below `min_cc_edges`, and return the kept components' edge lists.
fn extract_kept_ccs(
    stratum_edges: &[(usize, usize)],
    n_cells: usize,
    min_cc_edges: usize,
) -> Vec<Vec<(usize, usize)>> {
    let ccs = connected_components_from_edges(stratum_edges, n_cells);
    let mut cc_edges: HashMap<usize, Vec<(usize, usize)>> = HashMap::default();
    for &(i, j) in stratum_edges {
        cc_edges.entry(ccs[i]).or_default().push((i, j));
    }
    cc_edges
        .into_values()
        .filter(|es| es.len() >= min_cc_edges)
        .collect()
}

/// Score every real LR pair against a fixed set of kept connected components
/// in parallel. Each pair gets a deterministic per-pair RNG seeded from
/// `base_seed` and the pair index, so results are reproducible.
#[allow(clippy::too_many_arguments)]
fn score_pairs_parallel(
    batch_label: &str,
    community: i32,
    kept_ccs: &[Vec<(usize, usize)>],
    real_pairs: &[(Box<str>, Box<str>, usize, usize)],
    gene_mean: &[f32],
    moran_full: &[f32],
    matcher: &MatcherContext,
    known_pairs: &HashSet<(usize, usize)>,
    real_gene_set: &HashSet<usize>,
    x_gn: &Mat,
    args: &SrtLrActivityArgs,
    base_seed: u64,
) -> Vec<LrActivityRow> {
    let n_edges_total: usize = kept_ccs.iter().map(|es| es.len()).sum();
    real_pairs
        .par_iter()
        .enumerate()
        .filter_map(|(pair_idx, (lname, rname, li, ri))| {
            let target_l_mean = gene_mean[*li];
            let target_l_moran = moran_full[*li];
            let target_r_mean = gene_mean[*ri];
            let target_r_moran = moran_full[*ri];
            if !target_l_moran.is_finite() || !target_r_moran.is_finite() {
                return None;
            }
            let mut rng = SmallRng::seed_from_u64(
                base_seed
                    .wrapping_add(community as u64 * 1_000_003)
                    .wrapping_add(pair_idx as u64),
            );
            let decoys = matcher.sample_decoys(
                &DecoyTarget {
                    real_l: *li,
                    real_r: *ri,
                    l_mean: target_l_mean,
                    l_moran: target_l_moran,
                    r_mean: target_r_mean,
                    r_moran: target_r_moran,
                    expr_tol: args.expr_tol,
                    moran_tol: args.moran_tol,
                    n_null: args.n_null,
                    scheme: args.null_scheme,
                    known_pairs,
                    exclude_genes: real_gene_set,
                },
                &mut rng,
            );
            let agg = aggregate_cc_statistics(kept_ccs, *li, *ri, &decoys, x_gn, args.n_bins)?;
            Some(LrActivityRow {
                batch: batch_label.to_string().into_boxed_str(),
                community,
                ligand: lname.clone(),
                receptor: rname.clone(),
                n_edges: n_edges_total as i32,
                n_components: kept_ccs.len() as i32,
                ce_obs: agg.ce_obs,
                ce_null_mean: agg.null_mean,
                ce_null_sd: agg.null_sd,
                z: agg.z,
                p_empirical: agg.p_empirical,
                q_bh: f32::NAN,
            })
        })
        .collect()
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

fn apply_bh_per_batch(rows: &mut [LrActivityRow]) {
    let mut by_batch: HashMap<Box<str>, Vec<usize>> = HashMap::default();
    for (i, r) in rows.iter().enumerate() {
        by_batch.entry(r.batch.clone()).or_default().push(i);
    }
    for (_b, idxs) in by_batch {
        let p: Vec<f32> = idxs.iter().map(|&i| rows[i].p_empirical).collect();
        let q = bh_qvalues(&p);
        for (k, &i) in idxs.iter().enumerate() {
            rows[i].q_bh = q[k];
        }
    }
}
