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
//! Per-(community, sample) gene rates are estimated as `Gamma(num + a0,
//! denom + b0)` posteriors with a `Gamma(1, 1)` prior, calibrated via
//! `matrix_param::dmatrix_gamma::GammaMatrix`. The variational log mean
//! `E[log λ] = ψ(a) − log(b)` (rather than `log1p(num/denom)`) is the
//! input to the test: a sample with `num = 0` and small `denom` lands
//! well above one with `num = 0` and large `denom`, breaking the
//! zero-count tie pile-up. NB-Fisher-info gene weights are baked into
//! these log-mean matrices once at the collapse boundary.
//!
//! For each (batch, community, LR pair) the statistic is a **weighted
//! covariance** between the sender-pseudobulk log-mean of L and the
//! receiver-pseudobulk log-mean of R across samples (sample weight =
//! sqrt(send_weight · recv_weight)). Covariance (rather than correlation)
//! preserves L-R magnitude information so pairs separate cleanly under
//! restandardization.
//!
//! The null is sample-level permutation of L within propensity-stratified
//! buckets — shuffles are restricted to samples sharing the top
//! `shuffle_stratify_dim` bits of the propensity code, so the cell-type
//! marginal is preserved across permutations. Per shuffle, a *fresh*
//! log-posterior sample (delta method: `Normal(ψ(a) − log(b), ψ'(a))`) is
//! drawn for both L and R rates and used as the per-permutation log
//! expression. The same draw is shared across all pairs in a given
//! shuffle so cross-pair dependence (and the WY guarantee) is preserved;
//! sparse pseudobulks correctly contribute a wider null than dense ones.
//!
//! Inference layers (per stratum):
//! - `p_empirical` / `p_z` — per-pair permutation diagnostics.
//! - `z_re` / `p_re` — Efron-Tibshirani restandardization: `(stat_obs - μ) / σ`
//!   with `(μ, σ)` = robust (median, 1.4826·MAD) of `stat_obs` across pairs.
//!   Two-sided p; sign restriction (active LR ⇒ positive `z_re`) applied at
//!   the reporting layer. Strata with MAD ≤ 1e-4 are flagged uncalibrated.
//! - `fwer_wy` — Westfall-Young single-step minP (FWER); same shuffle σ_k
//!   applied to every pair so cross-pair dependence is preserved.

use crate::lr_activity::args::SrtLrActivityArgs;
use crate::lr_activity::io::*;
use crate::lr_activity::outputs::{
    pvalue_histogram, write_lr_activity, write_lr_activity_json, LrActivityRow, StratumEntry,
};
use crate::util::common::*;
use data_beans::convert::try_open_or_convert;
use data_beans_alg::random_projection::{binary_sort_columns, RandProjOps};
use matrix_param::dmatrix_gamma::GammaMatrix;
use matrix_param::traits::{CalibrateTarget, Inference, TwoStatParam};
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

/// Gamma posterior hyperparameters for pseudobulk rates. `Gamma(1, 1)` is
/// the standard weak-but-proper prior used elsewhere in pinto (propensity,
/// link-community profiles, dsvd).
const GAMMA_A0: f32 = 1.0;
const GAMMA_B0: f32 = 1.0;
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

    // Sparse-community filter: communities with too few edges can't
    // calibrate (most pseudobulk samples will be empty / have constant L
    // or R, collapsing stat_obs to 0 and breaking restandardization).
    let mut edges_per_community = vec![0usize; n_communities];
    for e in &edges {
        edges_per_community[e.2 as usize] += 1;
    }
    let active_communities: HashSet<u32> = (0..n_communities as u32)
        .filter(|&c| edges_per_community[c as usize] >= args.min_edges_per_community)
        .collect();
    let n_skipped = n_communities - active_communities.len();
    if n_skipped > 0 {
        info!(
            "Skipping {} sparse communities (< {} edges each)",
            n_skipped, args.min_edges_per_community
        );
    }

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

    // Fisher weight is multiplicative on pb_mean *before* log1p — without
    // the non-linear log step it would cancel out of any correlation /
    // covariance and have no effect.
    info!("Computing NB Fisher-info gene weights for LR genes...");
    let fisher_all = compute_nb_fisher_weights(&data_vec, c.block_size)?;
    let fisher_lr: Vec<f32> = lr_genes.iter().map(|&g| fisher_all[g]).collect();
    info!(
        "Fisher w (LR genes): min={:.3e}, mean={:.3e}, max={:.3e}",
        fisher_lr.iter().cloned().fold(f32::INFINITY, f32::min),
        fisher_lr.iter().sum::<f32>() / (fisher_lr.len().max(1) as f32),
        fisher_lr.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
    );

    //////////////////////////////////////////////////////
    // 9. Collapse into per-(community, sample) pseudobulk
    //////////////////////////////////////////////////////
    info!(
        "Collapsing into pseudobulk: {} comm × {} samples × {} LR genes...",
        n_communities,
        n_samples,
        lr_genes.len()
    );
    let mut collapse = collapse_role_pseudobulk(
        &x_lr,
        &sample_id_per_cell,
        &p_send,
        &p_recv,
        n_communities,
        n_samples,
    );

    // Bake Fisher weights into log-mean matrices once (per-gene row scale).
    // Under weighted_cov, per-gene scalars on each side stack as `w_L · w_R`
    // on stat_obs; precomputing here saves an n_pairs × n_samples mul per
    // stratum.
    for c in 0..n_communities {
        apply_gene_weights(&mut collapse.log_mean_send[c], &fisher_lr);
        apply_gene_weights(&mut collapse.log_mean_recv[c], &fisher_lr);
    }

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

    let mut plan: Vec<(Box<str>, u32, Vec<usize>, bool)> = Vec::new();
    for batch_label in &unique_batches {
        let samples_in_batch: Vec<usize> = (0..n_samples)
            .filter(|&s| sample_batch_label[s].as_ref() == batch_label.as_ref())
            .collect();
        if samples_in_batch.len() < MIN_SAMPLES_PER_STRATUM {
            continue;
        }
        for community in 0..(n_communities as u32) {
            if !active_communities.contains(&community) {
                continue;
            }
            if !community_present(&collapse, community as usize, &samples_in_batch) {
                continue;
            }
            plan.push((
                batch_label.clone(),
                community,
                samples_in_batch.clone(),
                false,
            ));
        }
    }
    if n_real_batches > 1 {
        let all_samples: Vec<usize> = (0..n_samples).collect();
        let meta_label: Box<str> = BATCH_LABEL_META.into();
        for community in 0..(n_communities as u32) {
            if !active_communities.contains(&community) {
                continue;
            }
            if !community_present(&collapse, community as usize, &all_samples) {
                continue;
            }
            plan.push((meta_label.clone(), community, all_samples.clone(), true));
        }
    }

    let n_strata = plan.len();
    info!(
        "Scoring {} strata × {} LR pairs ({} permutations each)...",
        n_strata,
        real_pairs.len(),
        args.n_permutations
    );

    for (k, (batch_label, community, samples, is_meta)) in plan.into_iter().enumerate() {
        let stratum_id = strata.len();
        strata.push(stratum_entry(
            &batch_label,
            community as i32,
            &edges,
            &cell_names,
            if is_meta {
                None
            } else {
                Some(batch_label.as_ref())
            },
            community,
        ));
        let seed = if is_meta {
            base_seed
                .wrapping_add(0xDEAD_BEEF)
                .wrapping_add(community as u64)
        } else {
            base_seed.wrapping_add(community as u64 * 1_000_003)
        };
        let t0 = std::time::Instant::now();
        let mut br = score_pairs_for_stratum(
            &batch_label,
            community,
            &samples,
            &sample_propbin,
            &real_pairs,
            &gene_to_local,
            &gene_names,
            &collapse,
            &fisher_lr,
            args.n_permutations,
            seed,
            stratum_id,
            /*shuffle_within_batch=*/ is_meta,
            &sample_batch_label,
            args.shuffle_stratify_dim,
        );
        info!(
            "  [{}/{}] batch={} community={} n_samples={} → {} rows ({:.1}s)",
            k + 1,
            n_strata,
            batch_label,
            community,
            samples.len(),
            br.len(),
            t0.elapsed().as_secs_f32()
        );
        rows.append(&mut br);
    }

    if log::log_enabled!(log::Level::Info) {
        eprintln!();
        eprintln!("{}", pvalue_histogram(&rows, 50));
        eprintln!();
    }

    //////////////////////////////////////////////////////
    // 11. Write output
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
            args.json_fwer_threshold,
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

/// Per-(community, sample) role-weighted pseudobulk for the LR genes,
/// stored as Gamma posterior log-means: `log_mean_send[c][(g, s)] =
/// E[log λ | data] = ψ(num + a0) - log(denom + b0)`. Using the variational
/// log-mean instead of `log1p(num/(denom+1))` breaks the zero-count tie
/// pile-up: a sample with `num = 0` and small `denom` lands far above one
/// with `num = 0` and large `denom` (correctly less confident in λ ≈ 0).
struct CollapseOut {
    log_mean_send: Vec<DMatrix<f32>>,
    log_mean_recv: Vec<DMatrix<f32>>,
    denom_send: Vec<Vec<f32>>,
    denom_recv: Vec<Vec<f32>>,
    /// Calibrated Gamma posteriors per (community, role) used by the
    /// per-permutation log-posterior draw inside `score_pairs_for_stratum`.
    /// Fisher weights are NOT pre-applied here (gene-row scaling is
    /// stacked on top of each draw inside the perm loop).
    gamma_send: Vec<GammaMatrix>,
    gamma_recv: Vec<GammaMatrix>,
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
    type CommunityCollapse = (
        DMatrix<f32>,
        DMatrix<f32>,
        Vec<f32>,
        Vec<f32>,
        GammaMatrix,
        GammaMatrix,
    );
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
            let ones_col = nalgebra::DVector::<f32>::from_element(n_lr, 1.0);
            let den_s_mat = &ones_col
                * nalgebra::RowDVector::<f32>::from_iterator(n_samples, den_s.iter().copied());
            let den_r_mat = &ones_col
                * nalgebra::RowDVector::<f32>::from_iterator(n_samples, den_r.iter().copied());

            let mut g_s = GammaMatrix::new((n_lr, n_samples), GAMMA_A0, GAMMA_B0);
            g_s.update_stat(&num_s, &den_s_mat);
            g_s.calibrate_with(CalibrateTarget::All);
            let log_mean_s = g_s.posterior_log_mean().clone();

            let mut g_r = GammaMatrix::new((n_lr, n_samples), GAMMA_A0, GAMMA_B0);
            g_r.update_stat(&num_r, &den_r_mat);
            g_r.calibrate_with(CalibrateTarget::All);
            let log_mean_r = g_r.posterior_log_mean().clone();

            (log_mean_s, log_mean_r, den_s, den_r, g_s, g_r)
        })
        .collect();

    let mut log_mean_send = Vec::with_capacity(n_communities);
    let mut log_mean_recv = Vec::with_capacity(n_communities);
    let mut denom_send = Vec::with_capacity(n_communities);
    let mut denom_recv = Vec::with_capacity(n_communities);
    let mut gamma_send = Vec::with_capacity(n_communities);
    let mut gamma_recv = Vec::with_capacity(n_communities);
    for (ls, lr, ds, dr, gs, gr) in per_community {
        log_mean_send.push(ls);
        log_mean_recv.push(lr);
        denom_send.push(ds);
        denom_recv.push(dr);
        gamma_send.push(gs);
        gamma_recv.push(gr);
    }
    CollapseOut {
        log_mean_send,
        log_mean_recv,
        denom_send,
        denom_recv,
        gamma_send,
        gamma_recv,
    }
}

fn community_present(collapse: &CollapseOut, c: usize, samples: &[usize]) -> bool {
    samples
        .iter()
        .any(|&s| collapse.denom_send[c][s] > 0.0 && collapse.denom_recv[c][s] > 0.0)
}

/// Weighted covariance between `l` and `r` using sample weights `w`.
/// Preferred over correlation here: (1) preserves absolute magnitude of
/// L-R coupling so pairs of different scales separate cleanly under
/// restandardization (correlation is bounded and piles up at the median);
/// (2) tied-zero samples (zero-inflated pseudobulks) contribute 0 to the
/// running sum instead of the spurious ±1 inflation that correlation
/// suffers when zero-patterns co-occur. Returns NaN when weights sum to 0.
fn weighted_cov(l: &[f32], r: &[f32], w: &[f32]) -> f32 {
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
    for k in 0..l.len() {
        cov += w[k] * (l[k] - ml) * (r[k] - mr);
    }
    cov / sw
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
    fisher_lr: &[f32],
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

    // Pre-generate K *shared* sample permutations for this stratum. Same
    // shuffle σ_k applied to every pair → preserves cross-pair dependence
    // (shared genes, batch confounders) for Westfall-Young joint
    // inference. Single seeded RNG for determinism.
    let shared_shuffles: Vec<Vec<usize>> = {
        let mut rng = SmallRng::seed_from_u64(base_seed);
        (0..n_perm)
            .map(|_| {
                let mut perm: Vec<usize> = (0..n_s).collect();
                for bucket in &bucket_idx {
                    if bucket.len() < 2 {
                        continue;
                    }
                    let mut vals: Vec<usize> = bucket.iter().map(|&p| perm[p]).collect();
                    vals.shuffle(&mut rng);
                    for (slot_pos, &out_pos) in bucket.iter().enumerate() {
                        perm[out_pos] = vals[slot_pos];
                    }
                }
                perm
            })
            .collect()
    };

    // Per-pair fixed quantities (l_local, r_local, t_obs).
    struct PairCtx {
        l_local: usize,
        r_local: usize,
        lname: Box<str>,
        rname: Box<str>,
        li: usize,
        ri: usize,
        t_obs: f32,
    }
    let pair_ctx: Vec<Option<PairCtx>> = real_pairs
        .par_iter()
        .map(|(lname, rname, li, ri)| -> Option<PairCtx> {
            let l_local = *gene_to_local.get(li)?;
            let r_local = *gene_to_local.get(ri)?;
            let l_vec: Vec<f32> = samples_in_stratum
                .iter()
                .map(|&s| collapse.log_mean_send[c][(l_local, s)])
                .collect();
            let r_vec: Vec<f32> = samples_in_stratum
                .iter()
                .map(|&s| collapse.log_mean_recv[c][(r_local, s)])
                .collect();
            let t_obs = weighted_cov(&l_vec, &r_vec, &w_pair);
            if !t_obs.is_finite() {
                return None;
            }
            Some(PairCtx {
                l_local,
                r_local,
                lname: lname.clone(),
                rname: rname.clone(),
                li: *li,
                ri: *ri,
                t_obs,
            })
        })
        .collect();

    // Build per-pair null vectors `t_perm[i]`. Per shuffle (sequential),
    // draw a fresh log-rate sample (delta method) for both roles, apply
    // Fisher, then compute t_perm for all pairs in parallel against the
    // same draw + same shuffle. Sharing the draw across pairs preserves
    // cross-pair dependence and the WY guarantee.
    let t_perm_per_pair: Vec<Vec<f32>> = {
        let mut t_per_pair: Vec<Vec<f32>> = vec![Vec::with_capacity(n_perm); pair_ctx.len()];
        for sigma in &shared_shuffles {
            let mut log_s = collapse.gamma_send[c]
                .posterior_log_sample()
                .expect("posterior_log_sample (send) failed");
            let mut log_r = collapse.gamma_recv[c]
                .posterior_log_sample()
                .expect("posterior_log_sample (recv) failed");
            apply_gene_weights(&mut log_s, fisher_lr);
            apply_gene_weights(&mut log_r, fisher_lr);
            let t_k: Vec<f32> = pair_ctx
                .par_iter()
                .map(|pc_opt| match pc_opt {
                    None => f32::NAN,
                    Some(pc) => {
                        let l_perm: Vec<f32> = samples_in_stratum
                            .iter()
                            .map(|&s| log_s[(pc.l_local, sigma[s])])
                            .collect();
                        let r_perm: Vec<f32> = samples_in_stratum
                            .iter()
                            .map(|&s| log_r[(pc.r_local, s)])
                            .collect();
                        weighted_cov(&l_perm, &r_perm, &w_pair)
                    }
                })
                .collect();
            for (i, t) in t_k.into_iter().enumerate() {
                t_per_pair[i].push(t);
            }
        }
        t_per_pair
    };

    // Per-pair: aggregate t_perm into stats and build the row.
    let pair_results: Vec<(LrActivityRow, Vec<f32>)> = pair_ctx
        .into_par_iter()
        .zip(t_perm_per_pair.into_par_iter())
        .filter_map(|(pc_opt, t_perm)| {
            let pc = pc_opt?;
            // Single pass over t_perm collects all per-pair null moments.
            let mut n_finite = 0usize;
            let mut sum = 0.0f32;
            let mut sumsq = 0.0f32;
            let mut n_gt = 0u32;
            let mut n_eq = 0u32;
            for &v in &t_perm {
                if !v.is_finite() {
                    continue;
                }
                n_finite += 1;
                sum += v;
                sumsq += v * v;
                if v > pc.t_obs {
                    n_gt += 1;
                } else if v == pc.t_obs {
                    n_eq += 1;
                }
            }
            if n_finite < n_perm.div_ceil(2) {
                return None;
            }
            let n_f = n_finite as f32;
            let mu = sum / n_f;
            let var = (sumsq - n_f * mu * mu) / (n_f - 1.0).max(1.0);
            let sd_raw = var.max(0.0).sqrt();

            // Degenerate null: shuffles produce a (near-)constant statistic.
            // Flag NaN for per-pair stats; t_perm stays so this pair can
            // still (not) contribute to the WY min-p downstream.
            let degenerate = !sd_raw.is_finite() || sd_raw < 1e-6;

            let p_emp = if degenerate {
                f32::NAN
            } else {
                (n_gt as f32 + 0.5 * n_eq as f32 + 0.5) / (n_f + 1.0)
            };

            let (z, p_z) = if degenerate {
                (f32::NAN, f32::NAN)
            } else {
                let zv = (pc.t_obs - mu) / sd_raw;
                (zv, one_sided_p_z(zv))
            };

            let row = LrActivityRow {
                batch: Box::from(batch_label),
                community: community as i32,
                ligand: pc.lname.clone(),
                receptor: pc.rname.clone(),
                ligand_resolved: gene_names[pc.li].clone(),
                receptor_resolved: gene_names[pc.ri].clone(),
                n_samples: n_s as i32,
                stat_obs: pc.t_obs,
                null_mean: mu,
                null_sd: sd_raw,
                z,
                p_empirical: p_emp,
                p_z,
                z_re: f32::NAN,
                p_re: f32::NAN,
                fwer_wy: f32::NAN,
                stratum_id,
            };
            Some((row, t_perm))
        })
        .collect();

    let (mut rows, perms_per_pair): (Vec<LrActivityRow>, Vec<Vec<f32>>) =
        pair_results.into_iter().unzip();

    // Efron-Tibshirani restandardization: re-center / re-scale stat_obs
    // against the across-pair empirical null bulk in this stratum, using
    // robust moments (median, MAD). Bypasses the per-pair permutation σ
    // calibration issues. Degenerate pairs (constant L or R → t_obs = 0,
    // null_sd = 0; flagged via NaN `z` upstream) are excluded both from
    // moment estimation (else MAD collapses to 0 and σ_emp pegs at the
    // floor) and from receiving z_re / p_re.
    let mut stats: Vec<f32> = rows
        .iter()
        .filter(|r: &&LrActivityRow| r.stat_obs.is_finite() && r.z.is_finite())
        .map(|r| r.stat_obs)
        .collect();
    if !stats.is_empty() {
        let cmp = |a: &f32, b: &f32| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);
        let mid = stats.len() / 2;
        let (_, med_ref, _) = stats.select_nth_unstable_by(mid, cmp);
        let med = *med_ref;
        let mut abs_dev: Vec<f32> = stats.iter().map(|&v| (v - med).abs()).collect();
        let mid = abs_dev.len() / 2;
        let (_, mad_ref, _) = abs_dev.select_nth_unstable_by(mid, cmp);
        let mad = *mad_ref;
        // If MAD is too small, the empirical null has insufficient spread
        // to calibrate against — σ_emp would shrink toward the floor and
        // tiny |stat_obs - med| differences produce extreme z_re values.
        // Common in sparse/small communities where most pairs collapse to
        // stat_obs ≈ 0. Leave the stratum's z_re / p_re NaN; WY skips it.
        if mad > 1e-4 {
            // 1.4826 is the standard normal-consistent MAD scaling.
            let sigma_emp = 1.4826 * mad;
            for r in rows.iter_mut() {
                if r.stat_obs.is_finite() && r.z.is_finite() {
                    let zr = (r.stat_obs - med) / sigma_emp;
                    r.z_re = zr;
                    // Two-sided p so the null is properly Uniform(0,1)
                    // regardless of any residual skew in the stat_obs
                    // distribution; Storey's π₀ estimator stays calibrated.
                    // Sign restriction (active LR = positive z_re) is
                    // applied at the reporting layer.
                    let p_two = 2.0 * one_sided_p_z(zr.abs());
                    r.p_re = p_two.min(1.0);
                }
            }
        }
    }

    // Westfall-Young single-step minP: `min_p[k] = min_i p_perm[k, i]` is
    // the null distribution of "the most significant pair under shuffle
    // k". Adjusted p = (1 + #{k : min_p[k] ≤ p_obs[i]}) / (K + 1).
    // Degenerate pairs don't enter min_p — they can't be most significant.
    if !perms_per_pair.is_empty() && !perms_per_pair[0].is_empty() {
        let k_perm = perms_per_pair[0].len();
        let k_perm_f = k_perm as f32;
        let mut min_p: Vec<f32> = vec![1.0f32; k_perm];
        let mut order: Vec<usize> = Vec::with_capacity(k_perm);
        for (row, t_perm) in rows.iter().zip(perms_per_pair.iter()) {
            if !row.stat_obs.is_finite() || !row.z.is_finite() {
                continue;
            }
            order.clear();
            order.extend(0..k_perm);
            order.sort_by(|&a, &b| {
                t_perm[b]
                    .partial_cmp(&t_perm[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for (pos, &k) in order.iter().enumerate() {
                let p_perm = (pos + 1) as f32 / k_perm_f;
                if p_perm < min_p[k] {
                    min_p[k] = p_perm;
                }
            }
        }

        for (row, t_perm) in rows.iter_mut().zip(perms_per_pair.iter()) {
            if !row.stat_obs.is_finite() || !row.z.is_finite() {
                continue;
            }
            let n_ge = t_perm
                .iter()
                .filter(|&&v| v.is_finite() && v >= row.stat_obs)
                .count();
            let p_obs = (n_ge as f32 + 1.0) / (k_perm_f + 1.0);
            let n_le = min_p.iter().filter(|&&v| v <= p_obs).count();
            let fwer_wy = (n_le as f32 + 1.0) / (k_perm_f + 1.0);
            row.fwer_wy = fwer_wy.min(1.0);
        }
    }
    rows
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
