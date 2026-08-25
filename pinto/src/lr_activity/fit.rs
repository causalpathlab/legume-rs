//! Pseudobulk + propensity-stratified sample-permutation LR activity test.
//!
//! Cells are collapsed into pseudobulk samples = (batch × propensity-bin),
//! where the propensity bin is the sign-LSH binary-sort code from an
//! SVD'd random projection of gene expression (within-batch centred).
//! Each cell carries soft membership over the link communities (the
//! fraction of its within-community edge instances falling in each), and
//! per (community, sample) we accumulate membership-weighted gene sums for
//! the LR genes, giving one pseudobulk profile per sample per community.
//! Strata ARE link communities: every within-community edge is enumerated
//! in both orientations, so every statistic here is symmetric in the pair
//! and no endpoint plays a privileged role. (An earlier directional design
//! paired communities into a -> b strata; it was removed after its arm
//! measured statistically inert and its roles proved to descend from the
//! edge list's arbitrary serialization order — see `orientation`.)
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
//! covariance** between the pseudobulk log-mean of L and of R across
//! samples (sample weight = the membership mass behind the sample).
//! Covariance (rather than correlation) preserves L-R magnitude
//! information so pairs separate cleanly under restandardization.
//!
//! The null is sample-level permutation of L within propensity-stratified
//! buckets — shuffles are restricted to samples sharing the top
//! `shuffle_stratify_dim` bits of the propensity code, so the cell-type
//! marginal is preserved across permutations. Per shuffle, a *fresh*
//! log-posterior sample (delta method: `Normal(ψ(a) − log(b), ψ'(a))`) is
//! drawn and used as the per-permutation log expression for both genes.
//! The same draw is shared across all pairs in a given shuffle so
//! cross-pair dependence (and the WY guarantee) is preserved; sparse
//! pseudobulks correctly contribute a wider null than dense ones.
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
use crate::lr_activity::orientation::CommunityStrata;
use crate::lr_activity::outputs::{
    pvalue_histogram, write_lr_activity, write_lr_activity_json, LrActivityRow, StratumEntry,
};
use crate::util::common::*;
use crate::util::gene_axis::GeneAxis;
use data_beans::convert::try_open_or_convert;
use data_beans_alg::gene_weighting::fisher_weights_from_stats;
use data_beans_alg::random_projection::{binary_sort_columns, RandProjOps};
use matrix_param::dmatrix_gamma::GammaMatrix;
use matrix_param::traits::{CalibrateTarget, Inference, TwoStatParam};
use matrix_util::common_io::mkdir_parent;
use matrix_util::membership::GeneIndexResolver;
use matrix_util::rand_util::mix_seed;
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
    let c = args;
    mkdir_parent(&c.out)?;

    /////////////////////////////////////////////////
    // 1. Load expression + resolve gene index map //
    /////////////////////////////////////////////////
    info!("Loading expression data...");
    let data_vec = load_expr_data(c)?;

    let row_names = data_vec.row_names()?;
    let cell_names = data_vec.column_names()?;
    let n_cells = data_vec.num_columns();

    // Ligand and receptor names name GENES, so they are resolved against the
    // gene axis. Against raw row names they cannot resolve at all on a
    // splice-channelized matrix: the resolver aliases on `--gene-delimiter`
    // (default `_`) and never on `/`, so `GENE1` does not reach
    // `GENE1/count/spliced` and every pair is dropped.
    let gene_axis = GeneAxis::resolve_or_identity(&row_names)?;
    let gene_names: Vec<Box<str>> = gene_axis.gene_names().to_vec();
    let n_genes = gene_axis.n_genes();
    if gene_axis.is_channelized() {
        info!(
            "Feature axis: {} rows carry splice channels over {} genes; a \
             ligand or receptor is scored on its gene, both tracks summed",
            row_names.len(),
            n_genes
        );
    }

    let gene_resolver =
        GeneIndexResolver::build(&gene_names, args.gene_delimiter, args.gene_allow_prefix);
    if gene_resolver.alias_collisions() > 0 {
        warn!(
            "{} gene-name aliases are shared by more than one row (duplicated \
             symbols, most likely); an LR pair naming one resolves to a single \
             arbitrary row. Name genes by their full row name to disambiguate.",
            gene_resolver.alias_collisions()
        );
    }
    let cell_to_col: HashMap<Box<str>, usize> = cell_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();

    //////////////////////////////////////
    // 2. Parse LR pairs, resolve genes //
    //////////////////////////////////////
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
        // warn, not info: at the default log level an info line is invisible,
        // and a pairs file in the wrong naming convention would silently run
        // on whatever fraction happened to match.
        warn!(
            "Skipped {} of {} LR pairs with unresolved gene names; \
             check the pair file's naming against the matrix row names \
             (--gene-delimiter aliases compound names)",
            missing,
            missing + resolved_pairs.len()
        );
    }
    anyhow::ensure!(
        !resolved_pairs.is_empty(),
        "no LR pairs resolved against expression row names"
    );
    info!("Resolved {} LR pairs", resolved_pairs.len());

    ///////////////////////////////////////////////////////
    // 3. Read edges + batches from prior `pinto lc` run //
    ///////////////////////////////////////////////////////
    let lc_edges_path = format!("{}.link_community.parquet", &args.lc_prefix);
    let coord_pairs_path = format!("{}.coord_pairs.parquet", &args.lc_prefix);
    info!("Reading edge assignments from {}", &lc_edges_path);
    let mut edge_records = read_link_community(&lc_edges_path)?;
    info!("Attaching per-edge batch from {}", &coord_pairs_path);
    attach_batch_from_coord_pairs(&mut edge_records, &coord_pairs_path)?;

    // Two lists with different jobs. `anchor_edges` is everything, and fixes
    // which community each cell belongs to. `edges` is the physically adjacent
    // subset, and is what actually gets tested: a directional test on a pair
    // that is merely expression-similar has no estimand, since the two cells
    // never touch. See `orientation` for why the anchor may be wider.
    let mut anchor_edges: Vec<(usize, usize, u32, Option<Box<str>>)> =
        Vec::with_capacity(edge_records.len());
    let mut edges: Vec<(usize, usize, u32, Option<Box<str>>)> =
        Vec::with_capacity(edge_records.len());
    let mut unresolved = 0usize;
    for e in edge_records {
        match (
            cell_to_col.get(&e.left_cell).copied(),
            cell_to_col.get(&e.right_cell).copied(),
        ) {
            (Some(i), Some(j)) => {
                let edge = (i, j, e.community, e.batch);
                if e.is_spatial {
                    edges.push(edge.clone());
                }
                anchor_edges.push(edge);
            }
            _ => unresolved += 1,
        }
    }
    let n_expression = anchor_edges.len() - edges.len();
    if n_expression > 0 {
        info!(
            "{} of {} pairs are expression-similar rather than adjacent: they set \
             each cell's community but are not themselves tested",
            n_expression,
            anchor_edges.len()
        );
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
    // Over the anchor list: a community that exists only on expression pairs
    // still has to be a valid stratum label.
    let n_communities = (anchor_edges.iter().map(|e| e.2).max().unwrap_or(0) as usize) + 1;

    // One stratum per link community; edges bridging two communities sit
    // out. See `orientation` for why the directional design was removed.
    let strata_map = CommunityStrata::from_edge_modes(&anchor_edges, &edges, n_cells);
    let n_strata_total = strata_map.n_strata();
    info!(
        "{} cells, {} edges, {} communities -> {} within-community strata",
        n_cells,
        edges.len(),
        n_communities,
        n_strata_total,
    );

    // Sparse-stratum filter: strata with too few edges can't calibrate (most
    // pseudobulk samples will be empty / have constant L or R, collapsing
    // stat_obs to 0 and breaking restandardization).
    let active_strata: HashSet<u32> = (0..n_strata_total as u32)
        .filter(|&s| strata_map.edges_in(s as usize) >= args.min_edges_per_community)
        .collect();
    let n_skipped = n_strata_total - active_strata.len();
    if n_skipped > 0 {
        info!(
            "Skipping {} sparse strata (< {} edge instances each)",
            n_skipped, args.min_edges_per_community
        );
    }

    ////////////////////////////////////////////////
    // 4. Per-gene total counts (filter LR pairs) //
    ////////////////////////////////////////////////
    // One pass, two consumers. `sum()` on the gene-axis statistics IS the
    // per-gene total the count filter needs, and the same statistics give the
    // NB precisions further down, so a separate totals pass would be a second
    // full read of the matrix for a number already in hand.
    info!("Computing per-gene statistics...");
    let (_, gene_stats) = gene_axis.running_stats(&data_vec, c.block_size, "NB-Fisher")?;
    let gene_sum: Vec<f32> = gene_stats.sum().to_vec();
    let fisher_all = fisher_weights_from_stats(&gene_stats, n_cells);
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

    //////////////////////////////////////////////////
    // 8. Read just the LR-gene rows (dense, small) //
    //////////////////////////////////////////////////
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
    // `lr_genes` are gene ids; the backend is read by row, so expand each gene
    // to the rows carrying it and fold them back afterwards. On a matrix
    // without channels this is the identity and the read is unchanged.
    let mut rows_to_read: Vec<usize> = Vec::with_capacity(lr_genes.len());
    let mut row_owner: Vec<usize> = Vec::with_capacity(lr_genes.len());
    for r in 0..row_names.len() {
        if let Some(&local) = gene_to_local.get(&gene_axis.gene_of_row(r)) {
            rows_to_read.push(r);
            row_owner.push(local);
        }
    }
    info!(
        "Reading {} rows for {} LR genes from backend...",
        rows_to_read.len(),
        lr_genes.len()
    );
    let x_rows = data_vec.read_rows_dmatrix(rows_to_read.iter().copied())?;
    // Column-outer and parallel, because `DMatrix` is column-major: a row-outer
    // walk strides by `nrows` and takes a cache miss per element.
    let x_lr = build_columns_par(lr_genes.len(), n_cells, |col, dst| {
        let src = x_rows.column(col);
        for (r, &local) in row_owner.iter().enumerate() {
            dst[local] += src[r];
        }
    });

    ///////////////////////////////////////////////////////////
    // Descriptive mode: per-batch edge scores, then done.   //
    ///////////////////////////////////////////////////////////
    if args.edge_scores_only {
        use crate::lr_activity::edge_scores::*;

        let distinct: HashSet<&str> = edges.iter().filter_map(|e| e.3.as_deref()).collect();
        // A label set that is entirely numeric is the fingerprint of a
        // coord_pairs table from before batch labels were exported as
        // strings: the numbers are a spatial offset, not identities, and a
        // score matrix keyed by them cannot be joined to anything.
        if !distinct.is_empty() && distinct.iter().all(|b| b.parse::<f64>().is_ok()) {
            warn!(
                "batch labels {:?} are all numeric; this looks like a run from \
                 before batch labels were exported. The scores will be keyed by \
                 a meaningless offset; re-run `pinto lc` to get identifiable \
                 batch labels.",
                distinct
            );
        }

        info!("Computing per-cell log1p depth...");
        let log_depth = per_cell_log1p_depth(&data_vec, c.block_size)?;

        info!(
            "Scoring {} LR pairs on the edges of {} strata...",
            real_pairs.len(),
            strata_map.n_strata()
        );
        let (score_rows, n_straddling) = compute_edge_scores(&EdgeScoresInput {
            edges: &edges,
            strata: &strata_map,
            pairs: &real_pairs,
            gene_to_local: &gene_to_local,
            x_lr: &x_lr,
            log_depth: &log_depth,
        });
        if n_straddling > 0 {
            info!(
                "Dropped {} straddling edges (endpoint batch labels differ); \
                 a contact belonging to no single batch scores in none",
                n_straddling
            );
        }
        let n_unmeasurable = score_rows.iter().filter(|r| r.log_or.is_nan()).count();
        if n_unmeasurable > 0 {
            info!(
                "{} of {} rows are prior dominated (no co-detection observed, \
                 none expected): log_or and log_or_se are NaN there, not zero",
                n_unmeasurable,
                score_rows.len()
            );
        }
        let out_path = format!("{}.lr_scores.parquet", &c.out);
        write_edge_scores(&c.out, &score_rows)?;
        info!("Wrote {} score rows to {}", score_rows.len(), out_path);

        // Back-fill the upstream manifest, the way the test path does for
        // its JSON sidecar, so a downstream reader can find this table
        // from the prefix alone rather than guessing the name.
        let upstream_meta_path = format!("{}.pinto.json", &args.lc_prefix);
        if let Ok(mut meta) =
            crate::util::metadata::PintoMetadata::read(std::path::Path::new(&upstream_meta_path))
        {
            meta.outputs.lr_scores = Some(out_path.clone());
            if let Err(e) = meta.write(std::path::Path::new(&upstream_meta_path)) {
                warn!("could not record the score table in {upstream_meta_path}: {e}");
            }
        }
        return Ok(());
    }

    //////////////////////////////////////////
    // 5. Cell→community soft membership    //
    //////////////////////////////////////////
    info!("Building cell→community soft membership...");
    let p_member = strata_map.memberships(&edges, n_cells);

    ////////////////////////////////////////////////////
    // 6. Per-cell batch label (modal incident batch) //
    ////////////////////////////////////////////////////
    let cell_batch = derive_cell_batch_labels(&edges, n_cells);

    ///////////////////////////////////////////////////
    // 7. Random projection + propensity binary-sort //
    ///////////////////////////////////////////////////
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

    // Fisher weight is multiplicative on pb_mean *before* log1p — without
    // the non-linear log step it would cancel out of any correlation /
    // covariance and have no effect.
    // Per GENE, off the statistics computed above. A Fisher weight is a
    // function of abundance and mean and is not additive, so no arithmetic on
    // two rows' weights gives the pooled gene's weight.
    let fisher_lr: Vec<f32> = lr_genes.iter().map(|&g| fisher_all[g]).collect();
    info!(
        "Fisher w (LR genes): min={:.3e}, mean={:.3e}, max={:.3e}",
        fisher_lr.iter().cloned().fold(f32::INFINITY, f32::min),
        fisher_lr.iter().sum::<f32>() / (fisher_lr.len().max(1) as f32),
        fisher_lr.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
    );

    /////////////////////////////////////////////////////////
    // 9. Collapse into per-(community, sample) pseudobulk //
    /////////////////////////////////////////////////////////
    info!(
        "Collapsing into pseudobulk: {} strata × {} samples × {} LR genes...",
        n_strata_total,
        n_samples,
        lr_genes.len()
    );
    let mut collapse = collapse_pseudobulk(
        &x_lr,
        &sample_id_per_cell,
        &p_member,
        n_strata_total,
        n_samples,
        &active_strata,
    );

    // Bake Fisher weights into log-mean matrices once (per-gene row scale).
    // Under weighted_cov, per-gene scalars on each side stack as `w_L · w_R`
    // on stat_obs; precomputing here saves an n_pairs × n_samples mul per
    // stratum.
    for c in 0..n_strata_total {
        if !active_strata.contains(&(c as u32)) {
            continue;
        }
        apply_gene_weights(&mut collapse.log_mean[c], &fisher_lr);
    }

    ////////////////////////////////////////////////////
    // 10. Per-stratum scoring (per-batch and pooled) //
    ////////////////////////////////////////////////////
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
        for stratum in 0..(n_strata_total as u32) {
            if !active_strata.contains(&stratum) {
                continue;
            }
            if !stratum_present(&collapse, stratum as usize, &samples_in_batch) {
                continue;
            }
            plan.push((
                batch_label.clone(),
                stratum,
                samples_in_batch.clone(),
                false,
            ));
        }
    }
    if n_real_batches > 1 {
        let all_samples: Vec<usize> = (0..n_samples).collect();
        let meta_label: Box<str> = BATCH_LABEL_META.into();
        for stratum in 0..(n_strata_total as u32) {
            if !active_strata.contains(&stratum) {
                continue;
            }
            if !stratum_present(&collapse, stratum as usize, &all_samples) {
                continue;
            }
            plan.push((meta_label.clone(), stratum, all_samples.clone(), true));
        }
    }

    let n_strata = plan.len();
    info!(
        "Scoring {} strata × {} LR pairs ({} permutations each)...",
        n_strata,
        real_pairs.len(),
        args.n_permutations
    );

    // Sequential deliberately. `score_pairs_for_stratum` parallelises over
    // permutations and, measured on a full-database run, already holds ~83%
    // of all cores; a second parallel axis here would chase the remainder
    // while multiplying the in-flight posterior draws, which are what set
    // peak memory.
    for (k, (batch_label, stratum, samples, is_meta)) in plan.into_iter().enumerate() {
        let stratum_id = strata.len();
        strata.push(stratum_entry(
            &batch_label,
            stratum as usize,
            &strata_map,
            &edges,
            &cell_names,
            if is_meta {
                None
            } else {
                Some(batch_label.as_ref())
            },
        ));
        let seed = if is_meta {
            base_seed
                .wrapping_add(0xDEAD_BEEF)
                .wrapping_add(stratum as u64)
        } else {
            base_seed.wrapping_add(stratum as u64 * 1_000_003)
        };
        let t0 = std::time::Instant::now();
        let mut br = score_pairs_for_stratum(
            &batch_label,
            stratum,
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
            &strata_map,
        );
        info!(
            "  [{}/{}] batch={} stratum={} n_samples={} → {} rows ({:.1}s)",
            k + 1,
            n_strata,
            batch_label,
            strata_map.label(stratum as usize),
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

    //////////////////////
    // 11. Write output //
    //////////////////////
    let out_path = format!("{}.lr_activity.parquet", &c.out);
    info!("Writing {} rows to {}", rows.len(), out_path);
    write_lr_activity(&out_path, &rows)?;

    if args.emit_json {
        let json_path = format!("{}.lr_activity.json", &c.out);
        let upstream_meta_path = format!("{}.pinto.json", &args.lc_prefix);
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

fn load_expr_data(c: &SrtLrActivityArgs) -> anyhow::Result<SparseIoVec> {
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
/// Per-stratum pseudobulk of the LR genes over (batch, propensity-bin)
/// samples, weighted by each cell's soft membership in the stratum's link
/// community, stored as Gamma posterior log-means: `log_mean[c][(g, s)] =
/// E[log λ | data] = ψ(num + a0) - log(denom + b0)`. Using the variational
/// log-mean instead of `log1p(num/(denom+1))` breaks the zero-count tie
/// pile-up: a sample with `num = 0` and small `denom` lands far above one
/// with `num = 0` and large `denom` (correctly less confident in λ ≈ 0).
///
/// One plane per stratum: with both orientations of every within-community
/// edge enumerated, the old sender/receiver roles coincide exactly.
struct CollapseOut {
    log_mean: Vec<DMatrix<f32>>,
    denom: Vec<Vec<f32>>,
    /// Calibrated Gamma posteriors per community used by the per-permutation
    /// log-posterior draw inside `score_pairs_for_stratum`. Fisher weights
    /// are NOT pre-applied here.
    gamma: Vec<GammaMatrix>,
}

fn collapse_pseudobulk(
    x_lr: &DMatrix<f32>,
    sample_id: &[usize],
    p_member: &DMatrix<f32>,
    n_strata: usize,
    n_samples: usize,
    active: &HashSet<u32>,
) -> CollapseOut {
    let n_lr = x_lr.nrows();
    let n_cells = x_lr.ncols();

    // Communities are independent — accumulate each one in its own thread.
    type StratumCollapse = (DMatrix<f32>, Vec<f32>, GammaMatrix);
    let per_stratum: Vec<StratumCollapse> = (0..n_strata)
        .into_par_iter()
        .map(|c| {
            // A stratum the sparsity filter already rejected is never scored;
            // building its entry anyway would retain dense planes and a full
            // digamma/trigamma calibration for the whole run. Return an empty
            // entry and keep the vectors indexed by stratum id.
            if !active.contains(&(c as u32)) {
                return (
                    DMatrix::zeros(0, 0),
                    Vec::new(),
                    GammaMatrix::new((0, 0), GAMMA_A0, GAMMA_B0),
                );
            }
            let mut num = DMatrix::<f32>::zeros(n_lr, n_samples);
            let mut den = vec![0.0f32; n_samples];
            for i in 0..n_cells {
                let s = sample_id[i];
                let pm = p_member[(i, c)];
                if pm > 0.0 {
                    num.column_mut(s).axpy(pm, &x_lr.column(i), 1.0);
                    den[s] += pm;
                }
            }
            let ones_col = nalgebra::DVector::<f32>::from_element(n_lr, 1.0);
            let den_mat = &ones_col
                * nalgebra::RowDVector::<f32>::from_iterator(n_samples, den.iter().copied());

            let mut g = GammaMatrix::new((n_lr, n_samples), GAMMA_A0, GAMMA_B0);
            g.update_stat(&num, &den_mat);
            g.calibrate_with(CalibrateTarget::All);
            let log_mean = g.posterior_log_mean().clone();
            (log_mean, den, g)
        })
        .collect();

    let mut log_mean = Vec::with_capacity(n_strata);
    let mut denom = Vec::with_capacity(n_strata);
    let mut gamma = Vec::with_capacity(n_strata);
    for (lm, d, g) in per_stratum {
        log_mean.push(lm);
        denom.push(d);
        gamma.push(g);
    }
    CollapseOut {
        log_mean,
        denom,
        gamma,
    }
}

fn stratum_present(collapse: &CollapseOut, c: usize, samples: &[usize]) -> bool {
    samples.iter().any(|&s| collapse.denom[c][s] > 0.0)
}

/// Weighted covariance between `l` and `r` using sample weights `w`.
/// Preferred over correlation here: (1) preserves absolute magnitude of
/// L-R coupling so pairs of different scales separate cleanly under
/// restandardization (correlation is bounded and piles up at the median);
/// (2) tied-zero samples (zero-inflated pseudobulks) contribute 0 to the
/// running sum instead of the spurious ±1 inflation that correlation
/// suffers when zero-patterns co-occur. Returns NaN when weights sum to 0.
pub(crate) fn weighted_cov(l: &[f32], r: &[f32], w: &[f32]) -> f32 {
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

/// Robust median of a slice; `None` when empty. NaN-tolerant ordering.
fn robust_median(v: &[f32]) -> Option<f32> {
    if v.is_empty() {
        return None;
    }
    let mut buf = v.to_vec();
    let mid = buf.len() / 2;
    let cmp = |a: &f32, b: &f32| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);
    let (_, m, _) = buf.select_nth_unstable_by(mid, cmp);
    Some(*m)
}

/// The Efron-Tibshirani restandardization scale for a stratum, or `None`
/// when the stratum cannot calibrate one.
///
/// `z_re` asks whether a pair stands out against the OTHER pairs in its
/// stratum, so the cross-pair spread must be commensurate with a single
/// pair's own permutation noise. When every pair sits in one tight clump the
/// clump's width is a shared noise floor, and dividing by it manufactures
/// astronomical significance for trivial values: a real run produced
/// `z_re = 12.3, p_re = 1e-34` on a pair whose own permutation p was 0.42.
///
/// The guard is scale-RELATIVE — `1.4826 * MAD >= 0.1 * median(null_sd)` —
/// never an absolute floor, which its predecessor was: an absolute floor
/// both passes the failure above (its MAD sat just over the constant) and
/// wrongly flags healthy strata measured in small units. The 0.1 sits an
/// order of magnitude below every healthy stratum measured (0.32-1.98 on
/// the reference run) and an order above the failure case (~0.02).
pub(crate) fn restandardization_scale(stats: &[f32], null_sds: &[f32]) -> Option<f32> {
    let med = robust_median(stats)?;
    let abs_dev: Vec<f32> = stats.iter().map(|&v| (v - med).abs()).collect();
    let mad = robust_median(&abs_dev)?;
    // 1.4826 is the standard normal-consistent MAD scaling.
    let sigma_emp = 1.4826 * mad;
    let noise = robust_median(null_sds)?;
    (sigma_emp.is_finite() && noise.is_finite() && sigma_emp >= 0.1 * noise).then_some(sigma_emp)
}

/// The weighted covariance, or `None` when the pair carries no testable
/// hypothesis in this stratum.
///
/// `None` when the statistic is non-finite (no usable weight at all) or lands
/// on bit-exact zero. Exact zero is not a measured null: it means no weighted
/// sample deviated on both sides at once (a constant side, or two genes
/// detected in disjoint samples), or the few real contributions cancelled to
/// the last bit, which is what near-floor genes produce in practice. Such a
/// pair cannot reject at any threshold, so scoring it would pay multiplicity
/// in the Westfall-Young family, drag the robust spread that scales every
/// other pair's `z_re`, and ship a row that reads as a measured null rather
/// than as an absent measurement.
///
/// An earlier guard keyed on the observed statistic AND the null spread both
/// being zero, which never fired: the null is built from fresh posterior
/// draws that fluctuate even where the observed posterior mean is flat.
///
/// A merely SMALL statistic stays in, deliberately — that is a magnitude
/// question, and a threshold here would be a silent power cut rather than an
/// exclusion of the untestable.
pub(crate) fn testable_weighted_cov(l: &[f32], r: &[f32], w: &[f32]) -> Option<f32> {
    let t = weighted_cov(l, r, w);
    if t.is_finite() && t != 0.0 {
        Some(t)
    } else {
        None
    }
}

#[allow(clippy::too_many_arguments)]
fn score_pairs_for_stratum(
    batch_label: &str,
    stratum: u32,
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
    strata_map: &CommunityStrata,
) -> Vec<LrActivityRow> {
    let c = stratum as usize;
    let n_s = samples_in_stratum.len();
    if n_s < MIN_SAMPLES_PER_STRATUM {
        return Vec::new();
    }

    let w_pair: Vec<f32> = samples_in_stratum
        .iter()
        .map(|&s| collapse.denom[c][s])
        .collect();

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
    // Each stored shuffle is a permutation of GLOBAL sample ids, in stratum
    // position order. The buckets shuffle positions, and the final map takes
    // every position through `samples_in_stratum`, so consumers gather from
    // the rate matrices directly and no position-space index survives to be
    // confused with a global one (which has been a live bug here once).
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
                perm.into_iter().map(|p| samples_in_stratum[p]).collect()
            })
            .collect()
    };

    // Per-pair fixed quantities. Untestable pairs (see
    // `testable_weighted_cov`) are compacted away here, so nothing downstream
    // carries a placeholder for them: not the permutation loop, not the WY
    // family, not the restandardization moments.
    struct PairCtx {
        l_local: usize,
        r_local: usize,
        lname: Box<str>,
        rname: Box<str>,
        li: usize,
        ri: usize,
        t_obs: f32,
        /// Product of the two per-gene Fisher weights. The observed side has
        /// them baked into the log-mean matrices; under the bilinear
        /// `weighted_cov` the same scaling reaches a permuted statistic as
        /// this one multiply, instead of rescaling a whole draw matrix per
        /// permutation.
        w_lr: f32,
    }
    let pair_ctx: Vec<PairCtx> = real_pairs
        .par_iter()
        .filter_map(|(lname, rname, li, ri)| -> Option<PairCtx> {
            let l_local = *gene_to_local.get(li)?;
            let r_local = *gene_to_local.get(ri)?;
            let l_vec: Vec<f32> = samples_in_stratum
                .iter()
                .map(|&s| collapse.log_mean[c][(l_local, s)])
                .collect();
            let r_vec: Vec<f32> = samples_in_stratum
                .iter()
                .map(|&s| collapse.log_mean[c][(r_local, s)])
                .collect();
            let t_obs = testable_weighted_cov(&l_vec, &r_vec, &w_pair)?;
            let w_lr = fisher_lr[l_local] * fisher_lr[r_local];
            Some(PairCtx {
                l_local,
                r_local,
                lname: lname.clone(),
                rname: rname.clone(),
                li: *li,
                ri: *ri,
                t_obs,
                w_lr,
            })
        })
        .collect();

    // Build per-pair null vectors `t_perm[i]`.
    //
    // Parallel over PERMUTATIONS, sequential over pairs within each. The
    // Westfall-Young guarantee constrains sharing WITHIN a permutation —
    // every pair must see the same shuffle and the same posterior draw — and
    // says nothing about the order permutations are computed in. Successive
    // permutations carry no state either: the draw seed is a pure function of
    // `k`, and the shuffles were all materialised up front.
    //
    // This is the better axis to parallelise on: far more permutations than
    // cores, so the granularity is coarse, where the old shape re-entered a
    // parallel region once per permutation (the draws themselves were
    // internally parallel, so the win is fewer region entries and coarser
    // work items, not idle cores recovered). Draw memory is bounded by the
    // worker count, since only in-flight draws are alive.
    //
    // `collect` on an indexed parallel iterator preserves order, which the WY
    // step below relies on: `min_p[k]` must refer to the same permutation for
    // every pair.

    let t_perm_per_pair: Vec<Vec<f32>> = {
        let t_per_k: Vec<Vec<f32>> = shared_shuffles
            .par_iter()
            .enumerate()
            .map(|(k, sigma)| {
                // Seeded per (stratum, permutation, role) through the shared
                // avalanche mixer, so nearby permutation indices map to
                // well-separated streams. (The shuffles beside this were
                // already deterministic; the posterior draw once was not, and
                // `--seed` did not reach the null.)
                let draw_seed = mix_seed(base_seed, k as u64);
                // One draw serves both genes: entries of the posterior sample
                // are independent across (gene, sample) cells, so the ligand
                // and receptor rows carry independent noise from one matrix.
                let log_draw = collapse.gamma[c]
                    .posterior_log_sample(draw_seed)
                    .expect("posterior_log_sample failed");
                // No Fisher rescale of the draw matrices here: `weighted_cov`
                // is bilinear, so the per-gene weights reach each statistic as
                // the precomputed `w_lr` scalar below — the same algebra the
                // observed side uses by baking the weights into the log-means.
                pair_ctx
                    .iter()
                    .map(|pc| {
                        let l_perm: Vec<f32> =
                            sigma.iter().map(|&g| log_draw[(pc.l_local, g)]).collect();
                        let r_obs: Vec<f32> = samples_in_stratum
                            .iter()
                            .map(|&s| log_draw[(pc.r_local, s)])
                            .collect();
                        pc.w_lr * weighted_cov(&l_perm, &r_obs, &w_pair)
                    })
                    .collect()
            })
            .collect();
        // Transpose [k][pair] -> [pair][k]; the k order is what `min_p`
        // indexes, so it must survive.
        // NOT `vec![Vec::with_capacity(..); n]`: that clones the prototype,
        // and a Vec clone copies contents (none) rather than capacity, so
        // every slot but the last would start at zero and regrow ~10 times.
        let mut t_per_pair: Vec<Vec<f32>> = (0..pair_ctx.len())
            .map(|_| Vec::with_capacity(n_perm))
            .collect();
        for t_k in t_per_k {
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
        .filter_map(|(pc, t_perm)| {
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

            // A null with no spread cannot standardize anything: flag NaN
            // for the per-pair stats rather than divide by (near-)zero.
            // This guards the NULL axis only — an untestable OBSERVED
            // statistic never gets this far (`testable_weighted_cov`
            // compacted those pairs away before the permutation loop). No
            // magnitude floor here for the same reason as there: a small
            // real spread is still a distribution.
            let degenerate = !sd_raw.is_finite() || sd_raw <= 0.0;

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
                // The REAL link community id — no longer a stratum index
                // hiding under this name.
                community: strata_map.community(c) as i32,
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
    // against the across-pair empirical bulk in this stratum, using robust
    // moments (median, MAD). Untestable pairs never got a row, so every row
    // with a finite z participates. The scale is guarded relative to the
    // pairs' own permutation noise — see `restandardization_scale`.
    let finite: Vec<&LrActivityRow> = rows
        .iter()
        .filter(|r| r.stat_obs.is_finite() && r.z.is_finite())
        .collect();
    let stats: Vec<f32> = finite.iter().map(|r| r.stat_obs).collect();
    let null_sds: Vec<f32> = finite.iter().map(|r| r.null_sd).collect();
    match restandardization_scale(&stats, &null_sds) {
        Some(sigma_emp) => {
            let med = robust_median(&stats).expect("stats nonempty when scale exists");
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
        None => {
            if !stats.is_empty() {
                warn!(
                    "stratum {}: cross-pair spread is too small against the pairs' own \
                     permutation noise to calibrate restandardization; z_re and p_re \
                     withheld (NaN) for its {} pairs",
                    stratum_id,
                    stats.len()
                );
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
    stratum: usize,
    strata_map: &CommunityStrata,
    edges: &[(usize, usize, u32, Option<Box<str>>)],
    cell_names: &[Box<str>],
    batch_filter: Option<&str>,
) -> StratumEntry {
    let name_of = |c: usize| {
        cell_names
            .get(c)
            .cloned()
            .unwrap_or_else(|| format!("cell_{c}").into_boxed_str())
    };
    // Each edge appears once per orientation in the stratum; keep the
    // un-flipped instances so the sidecar lists every participating pair
    // exactly once, with no direction claimed.
    let edges_named: Vec<(Box<str>, Box<str>)> = strata_map
        .oriented(stratum)
        .iter()
        .filter(|&&(_, flipped)| !flipped)
        .filter(|&&(e, _)| {
            let b = &edges[e as usize].3;
            match batch_filter {
                // Edges left unbatched (no `batch` column in coord_pairs)
                // are matched by the synthetic `BATCH_LABEL_ALL` stratum
                // — `derive_cell_batch_labels` already gave those cells
                // that label, so the per-edge filter has to agree.
                Some(bf) => match b.as_ref().map(|s| s.as_ref()) {
                    Some(eb) => eb == bf,
                    None => bf == BATCH_LABEL_ALL,
                },
                None => true,
            }
        })
        .map(|&(e, _)| {
            let (i, j, _, _) = edges[e as usize];
            (name_of(i), name_of(j))
        })
        .collect();
    StratumEntry {
        batch: Box::from(batch_label),
        community: strata_map.community(stratum) as i32,
        edges: edges_named,
    }
}
