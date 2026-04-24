//! `cocoa spatial-diff` — per-topic spatial differential expression.
//!
//! Stratify cells by θ_k (HIGH/LOW/DROP), match HIGH to spatial-kNN LOW
//! neighbors, and report log τ_high − log τ_low per (gene, topic) with
//! hybrid permutation + CLT calibration.
//!
//! Outputs:
//! - `{out}.spatial_diff.tsv.gz` — gene, topic, contrast_log, null_mean, null_sd, z, pval
//! - `{out}.spatial_diff_indiv.tsv.gz` — per-individual log-fold-change (with `--indv-files`)

use crate::common::*;
use crate::input::*;
use crate::spatial_match::*;
use crate::stat::z_to_pvalue;

use clap::Parser;
use data_beans_alg::gene_weighting::{apply_gene_weights, compute_nb_fisher_weights};
use matrix_param::dmatrix_gamma::GammaMatrix;
use matrix_param::traits::{CalibrateTarget, Inference, TwoStatParam};
use matrix_util::common_io::write_lines;
use matrix_util::knn_graph::KnnGraph;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rustc_hash::FxHashMap as HashMap;
use std::collections::BTreeSet;

#[derive(Parser, Debug, Clone)]
pub struct SpatialDiffArgs {
    #[arg(required = true, help = "Sparse count files (.zarr or .h5).")]
    data_files: Vec<Box<str>>,

    #[arg(
        short = 'i',
        long,
        value_delimiter = ',',
        help = "Per-cell individual membership files (comma-separated)."
    )]
    indv_files: Option<Vec<Box<str>>>,

    #[arg(
        short = 'r',
        long = "topic-proportion-files",
        value_delimiter = ',',
        required = true,
        help = "Cell × topic propensity parquet files (e.g. pinto's .propensity.parquet)."
    )]
    topic_proportion_files: Vec<Box<str>>,

    #[arg(
        long = "topic-proportion-value",
        default_value = "prob",
        help = "Topic proportion scale (`prob` or `logit`)."
    )]
    topic_proportion_value: TopicValue,

    #[arg(
        long = "coords-file",
        required = true,
        help = "TSV of cell coords: `cell<TAB>x<TAB>y[<TAB>z]` or `x<TAB>y[<TAB>z]` (row-order)."
    )]
    coords_file: Box<str>,

    #[arg(long = "spatial-knn", default_value_t = DEFAULT_SPATIAL_KNN, help = "Spatial kNN fan-out.")]
    spatial_knn: usize,

    #[arg(
        long = "spatial-radius",
        help = "Optional hard radius cap on kNN edges."
    )]
    spatial_radius: Option<f32>,

    #[arg(long = "topic-high-quantile", default_value_t = DEFAULT_HIGH_Q, help = "HIGH θ_k quantile.")]
    topic_high_quantile: f32,

    #[arg(long = "topic-low-quantile", default_value_t = DEFAULT_LOW_Q, help = "LOW θ_k quantile.")]
    topic_low_quantile: f32,

    #[arg(long, default_value_t = 1.0, help = "Gamma prior shape a0.")]
    a0: f32,

    #[arg(long, default_value_t = 1.0, help = "Gamma prior rate b0.")]
    b0: f32,

    #[arg(long, default_value_t = 1000, help = "Block size for sparse reads.")]
    block_size: usize,

    #[arg(
        long = "no-adjust-housekeeping",
        default_value_t = false,
        help = "Disable NB-Fisher housekeeping gene adjustment (default ON)."
    )]
    no_adjust_housekeeping: bool,

    #[arg(short, long = "out", required = true, help = "Output prefix.")]
    out: Box<str>,

    #[arg(long, default_value_t = false, help = "Preload all columns.")]
    preload_data: bool,

    #[arg(
        long = "n-permutations",
        default_value_t = 0,
        help = "Permutation reps for hybrid permutation+CLT null (0 disables)."
    )]
    n_permutations: usize,

    #[arg(long, default_value_t = 42, help = "Random seed for permutation RNG.")]
    rseed: u64,
}

struct TopicStat {
    y_high: Mat,
    size_high: DVec,
    y_low_matched: Mat,
    size_low_matched: DVec,
}

impl TopicStat {
    fn new(n_genes: usize, n_indv: usize) -> Self {
        Self {
            y_high: Mat::zeros(n_genes, n_indv),
            size_high: DVec::zeros(n_indv),
            y_low_matched: Mat::zeros(n_genes, n_indv),
            size_low_matched: DVec::zeros(n_indv),
        }
    }

    fn n_genes(&self) -> usize {
        self.y_high.nrows()
    }

    fn n_indv(&self) -> usize {
        self.y_high.ncols()
    }

    fn scale_rows(&mut self, w: &[f32]) {
        apply_gene_weights(&mut self.y_high, w);
        apply_gene_weights(&mut self.y_low_matched, w);
    }
}

struct PermCtx<'a> {
    data: &'a SparseIoVec,
    graph: &'a KnnGraph,
    strata: &'a [Stratum],
    cell_to_indv_idx: &'a [usize],
    n_indv: usize,
    theta_k: &'a [f32],
    radius: Option<f32>,
    block_size: usize,
    gene_weights: Option<&'a [f32]>,
    a0: f32,
    b0: f32,
    n_genes: usize,
}

pub fn run_cocoa_spatial_diff(args: SpatialDiffArgs) -> anyhow::Result<()> {
    if args.topic_low_quantile >= args.topic_high_quantile {
        return Err(anyhow::anyhow!(
            "--topic-low-quantile must be < --topic-high-quantile"
        ));
    }

    let data = read_input_data(InputDataArgs {
        data_files: args.data_files,
        indv_files: args.indv_files.clone(),
        topic_assignment_files: None,
        topic_proportion_files: Some(args.topic_proportion_files),
        exposure_assignment_file: None,
        preload_data: args.preload_data,
        topic_value: args.topic_proportion_value,
    })?;

    let n_cells = data.sparse_data.num_columns();
    let n_genes = data.sparse_data.num_rows();
    let n_topics = data.cell_topic.ncols();
    let gene_names = data.sparse_data.row_names()?;
    let column_names = data.sparse_data.column_names()?;

    info!(
        "Loaded {} cells × {} genes, {} topics",
        n_cells, n_genes, n_topics
    );

    let coords = read_coords_file(&args.coords_file, &column_names, n_cells)?;
    info!("Spatial coords: {} × {}", coords.nrows(), coords.ncols());

    let (cell_to_indv_idx, indv_names): (Vec<usize>, Vec<Box<str>>) =
        build_indv_index(&data.cell_to_indv, args.indv_files.is_some());
    let n_indv = indv_names.len().max(1);
    info!("Using {} individual bins", n_indv);

    let gene_weights: Option<Vec<f32>> = if args.no_adjust_housekeeping {
        None
    } else {
        info!("Computing NB-Fisher housekeeping weights");
        Some(compute_nb_fisher_weights(
            &data.sparse_data,
            Some(args.block_size),
        )?)
    };

    info!(
        "Building spatial kNN graph (k={}, radius={:?})",
        args.spatial_knn, args.spatial_radius
    );
    let graph = build_spatial_graph(&coords, args.spatial_knn, args.block_size)?;
    info!("Spatial graph: {} edges", graph.num_edges());

    type CohortRow = (Box<str>, usize, f32, f32, f32, f32, f32);
    type IndvRow = (Box<str>, usize, Box<str>, f32);
    let mut all_rows: Vec<CohortRow> = Vec::new();
    let mut indv_rows: Vec<IndvRow> = Vec::new();

    let mut perm_rng = rand::rngs::StdRng::seed_from_u64(args.rseed);

    for k in 0..n_topics {
        let theta_k: Vec<f32> = (0..n_cells).map(|i| data.cell_topic[(i, k)]).collect();
        let strata = stratify_topic(&theta_k, args.topic_low_quantile, args.topic_high_quantile);
        let n_high = strata.iter().filter(|&&s| s == Stratum::High).count();
        let n_low = strata.iter().filter(|&&s| s == Stratum::Low).count();
        info!(
            "topic {}/{}: {} HIGH, {} LOW cells",
            k + 1,
            n_topics,
            n_high,
            n_low
        );
        if n_high == 0 || n_low == 0 {
            warn!("  topic {}: empty HIGH or LOW stratum — skipping", k);
            continue;
        }

        let matches = high_to_low_neighbors(&graph, &strata, args.spatial_radius);
        let mut stat = TopicStat::new(n_genes, n_indv);
        accumulate_topic(
            &data.sparse_data,
            &matches,
            &cell_to_indv_idx,
            &theta_k,
            &strata,
            &mut stat,
            args.block_size,
        )?;
        if let Some(w) = gene_weights.as_deref() {
            stat.scale_rows(w);
        }

        let (tau_high, tau_low) = fit_gamma_pair(&stat, args.a0, args.b0);
        let obs_contrast = cohort_contrast_per_gene(&stat, &tau_high, &tau_low);

        let (null_mean, null_sd) = if args.n_permutations > 0 {
            info!("  topic {}: {} permutations", k, args.n_permutations);
            let ctx = PermCtx {
                data: &data.sparse_data,
                graph: &graph,
                strata: &strata,
                cell_to_indv_idx: &cell_to_indv_idx,
                n_indv,
                theta_k: &theta_k,
                radius: args.spatial_radius,
                block_size: args.block_size,
                gene_weights: gene_weights.as_deref(),
                a0: args.a0,
                b0: args.b0,
                n_genes,
            };
            run_permutations(&ctx, args.n_permutations, &mut perm_rng)?
        } else {
            (vec![f32::NAN; n_genes], vec![f32::NAN; n_genes])
        };

        let tau_high_log = tau_high.posterior_log_mean();
        let tau_low_log = tau_low.posterior_log_mean();

        for g in 0..n_genes {
            let c = obs_contrast[g];
            let (nm, ns) = (null_mean[g], null_sd[g]);
            let (z, p) = if ns.is_finite() && ns > 1e-8 {
                let zv = (c - nm) / ns;
                (zv, z_to_pvalue(zv))
            } else {
                (f32::NAN, f32::NAN)
            };
            all_rows.push((gene_names[g].clone(), k, c, nm, ns, z, p));

            if args.indv_files.is_some() {
                for i in 0..n_indv {
                    if stat.size_high[i] > 0.0 && stat.size_low_matched[i] > 0.0 {
                        let diff = tau_high_log[(g, i)] - tau_low_log[(g, i)];
                        indv_rows.push((gene_names[g].clone(), k, indv_names[i].clone(), diff));
                    }
                }
            }
        }
    }

    let out_path = format!("{}.spatial_diff.tsv.gz", args.out);
    let mut lines: Vec<Box<str>> = Vec::with_capacity(all_rows.len() + 1);
    lines.push("gene\ttopic\tcontrast_log\tnull_mean\tnull_sd\tz\tpval".into());
    for (gene, topic, c, nm, ns, z, p) in &all_rows {
        lines.push(format!("{gene}\t{topic}\t{c}\t{nm}\t{ns}\t{z}\t{p}").into());
    }
    write_lines(&lines, &out_path)?;
    info!("Wrote cohort contrasts to {out_path}");

    if args.indv_files.is_some() && !indv_rows.is_empty() {
        let indv_path = format!("{}.spatial_diff_indiv.tsv.gz", args.out);
        let mut ilines: Vec<Box<str>> = Vec::with_capacity(indv_rows.len() + 1);
        ilines.push("gene\ttopic\tindividual\tlog_fold_change".into());
        for (gene, topic, indv, d) in &indv_rows {
            ilines.push(format!("{gene}\t{topic}\t{indv}\t{d}").into());
        }
        write_lines(&ilines, &indv_path)?;
        info!("Wrote individual-level contrasts to {indv_path}");
    }

    info!("Done");
    Ok(())
}

fn build_indv_index(cell_to_indv: &[Box<str>], have_indv: bool) -> (Vec<usize>, Vec<Box<str>>) {
    if !have_indv {
        return (vec![0; cell_to_indv.len()], vec!["cohort".into()]);
    }
    let mut unique: BTreeSet<Box<str>> = BTreeSet::new();
    for s in cell_to_indv {
        if !s.is_empty() && s.as_ref() != "NA" {
            unique.insert(s.clone());
        }
    }
    let names: Vec<Box<str>> = unique.into_iter().collect();
    let name_to_idx: HashMap<Box<str>, usize> = names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();
    let idx: Vec<usize> = cell_to_indv
        .iter()
        .map(|n| name_to_idx.get(n).copied().unwrap_or(usize::MAX))
        .collect();
    (idx, names)
}

fn read_coords_file(path: &str, column_names: &[Box<str>], n_cells: usize) -> anyhow::Result<Mat> {
    use matrix_util::common_io::read_lines_of_words_delim;
    let parsed = read_lines_of_words_delim(path, &['\t', ',', ' '], -1)?;
    let lines = parsed.lines;
    if lines.is_empty() {
        return Err(anyhow::anyhow!("coords file {} is empty", path));
    }

    let first_is_numeric = lines[0][0].as_ref().parse::<f32>().is_ok();
    let ncol = lines[0].len();
    let d = if first_is_numeric { ncol } else { ncol - 1 };
    if d < 2 {
        return Err(anyhow::anyhow!(
            "coords file must have at least 2 coordinate columns"
        ));
    }

    let mut coords = Mat::zeros(n_cells, d);

    if first_is_numeric {
        if lines.len() != n_cells {
            return Err(anyhow::anyhow!(
                "coords file has {} rows but data has {} cells (no cell-name column)",
                lines.len(),
                n_cells
            ));
        }
        for (i, line) in lines.iter().enumerate() {
            for j in 0..d {
                coords[(i, j)] = line[j].parse::<f32>()?;
            }
        }
    } else {
        let name_to_idx: HashMap<Box<str>, usize> = column_names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();
        let mut matched = 0usize;
        for line in lines.iter() {
            if let Some(&idx) = name_to_idx.get(&line[0]) {
                for j in 0..d {
                    coords[(idx, j)] = line[j + 1].parse::<f32>()?;
                }
                matched += 1;
            }
        }
        info!("coords: matched {}/{} cells by name", matched, n_cells);
        if matched == 0 {
            return Err(anyhow::anyhow!(
                "no cell names from coords file matched data columns"
            ));
        }
        if matched < n_cells {
            return Err(anyhow::anyhow!(
                "coords file matched only {}/{} cells by name; remaining cells would have zero coords and bias kNN matching",
                matched,
                n_cells
            ));
        }
    }

    Ok(coords)
}

fn accumulate_topic(
    data: &SparseIoVec,
    matches: &[Vec<(usize, f32)>],
    cell_to_indv_idx: &[usize],
    theta_k: &[f32],
    strata: &[Stratum],
    stat: &mut TopicStat,
    block_size: usize,
) -> anyhow::Result<()> {
    let n_cells = data.num_columns();
    debug_assert_eq!(data.num_rows(), stat.n_genes());

    let high_cells: Vec<usize> = (0..n_cells)
        .filter(|&i| strata.get(i).copied() == Some(Stratum::High) && !matches[i].is_empty())
        .collect();
    if high_cells.is_empty() {
        return Ok(());
    }

    for block in high_cells.chunks(block_size) {
        let y_high_block = data.read_columns_csc(block.iter().cloned())?;

        // Batch-read LOW neighbors for the whole block so each sparse backend
        // call amortizes over many HIGH cells.
        let mut low_ids: Vec<usize> = Vec::new();
        let mut low_offsets: Vec<usize> = Vec::with_capacity(block.len() + 1);
        low_offsets.push(0);
        for &cell_i in block {
            for &(j, _) in &matches[cell_i] {
                low_ids.push(j);
            }
            low_offsets.push(low_ids.len());
        }
        let y_low_block = if low_ids.is_empty() {
            None
        } else {
            Some(data.read_columns_csc(low_ids.iter().cloned())?)
        };

        for (bpos, &cell_i) in block.iter().enumerate() {
            let indv_i = cell_to_indv_idx[cell_i];
            if indv_i == usize::MAX {
                continue;
            }
            let theta_i = theta_k[cell_i];

            if let Some(col) = y_high_block.get_col(bpos) {
                for (&g, &y) in col.row_indices().iter().zip(col.values().iter()) {
                    stat.y_high[(g, indv_i)] += theta_i * y;
                }
            }
            stat.size_high[indv_i] += theta_i;

            let nbrs = &matches[cell_i];
            let weights: Vec<f32> = nbrs
                .iter()
                .map(|&(j, d)| (-d).exp() * (1.0 - theta_k[j]).max(0.0))
                .collect();
            let denom: f32 = weights.iter().sum();
            if denom < 1e-8 {
                continue;
            }
            let lo = low_offsets[bpos];
            let y_low = y_low_block.as_ref().expect("low block present");
            for (p, &w) in weights.iter().enumerate() {
                let scale = (theta_i * w) / denom;
                if let Some(col) = y_low.get_col(lo + p) {
                    for (&g, &y) in col.row_indices().iter().zip(col.values().iter()) {
                        stat.y_low_matched[(g, indv_i)] += scale * y;
                    }
                }
                stat.size_low_matched[indv_i] += scale;
            }
        }
    }

    Ok(())
}

fn cohort_contrast_per_gene(
    stat: &TopicStat,
    tau_high: &GammaMatrix,
    tau_low: &GammaMatrix,
) -> Vec<f32> {
    let hi = tau_high.posterior_log_mean();
    let lo = tau_low.posterior_log_mean();
    let n_genes = stat.n_genes();
    let n_indv = stat.n_indv();
    let mut out = Vec::with_capacity(n_genes);
    for g in 0..n_genes {
        let mut acc = 0f32;
        let mut n_eff = 0usize;
        for i in 0..n_indv {
            if stat.size_high[i] > 0.0 && stat.size_low_matched[i] > 0.0 {
                acc += hi[(g, i)] - lo[(g, i)];
                n_eff += 1;
            }
        }
        out.push(if n_eff > 0 {
            acc / n_eff as f32
        } else {
            f32::NAN
        });
    }
    out
}

fn permute_strata_within_indv(
    strata: &[Stratum],
    cell_to_indv_idx: &[usize],
    n_indv: usize,
    rng: &mut rand::rngs::StdRng,
) -> Vec<Stratum> {
    let mut out = strata.to_vec();
    for i in 0..n_indv {
        let mut positions: Vec<usize> = Vec::new();
        let mut labels: Vec<Stratum> = Vec::new();
        for (c, s) in strata.iter().enumerate() {
            if cell_to_indv_idx.get(c).copied() == Some(i)
                && (*s == Stratum::High || *s == Stratum::Low)
            {
                positions.push(c);
                labels.push(*s);
            }
        }
        labels.shuffle(rng);
        for (p, l) in positions.iter().zip(labels.iter()) {
            out[*p] = *l;
        }
    }
    out
}

fn run_permutations(
    ctx: &PermCtx<'_>,
    n_perm: usize,
    rng: &mut rand::rngs::StdRng,
) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
    let n_genes = ctx.n_genes;
    let mut mean = vec![0f32; n_genes];
    let mut m2 = vec![0f32; n_genes];
    let mut count = vec![0u32; n_genes];

    for rep in 0..n_perm {
        let perm_strata =
            permute_strata_within_indv(ctx.strata, ctx.cell_to_indv_idx, ctx.n_indv, rng);
        let perm_matches = high_to_low_neighbors(ctx.graph, &perm_strata, ctx.radius);
        let mut pstat = TopicStat::new(n_genes, ctx.n_indv);
        accumulate_topic(
            ctx.data,
            &perm_matches,
            ctx.cell_to_indv_idx,
            ctx.theta_k,
            &perm_strata,
            &mut pstat,
            ctx.block_size,
        )?;
        if let Some(w) = ctx.gene_weights {
            pstat.scale_rows(w);
        }
        let (th, tl) = fit_gamma_pair(&pstat, ctx.a0, ctx.b0);
        let c = cohort_contrast_per_gene(&pstat, &th, &tl);
        for g in 0..n_genes {
            if !c[g].is_finite() {
                continue;
            }
            count[g] += 1;
            let n = count[g] as f32;
            let delta = c[g] - mean[g];
            mean[g] += delta / n;
            let delta2 = c[g] - mean[g];
            m2[g] += delta * delta2;
        }
        if (rep + 1) % 50 == 0 {
            info!("    permutation {}/{}", rep + 1, n_perm);
        }
    }

    let mut null_mean = vec![f32::NAN; n_genes];
    let mut null_sd = vec![f32::NAN; n_genes];
    for g in 0..n_genes {
        if count[g] >= 2 {
            null_mean[g] = mean[g];
            null_sd[g] = (m2[g] / (count[g] as f32 - 1.0)).sqrt();
        }
    }
    Ok((null_mean, null_sd))
}

fn fit_gamma_pair(stat: &TopicStat, a0: f32, b0: f32) -> (GammaMatrix, GammaMatrix) {
    let n_g = stat.n_genes();
    let n_i = stat.n_indv();

    let mut tau_high = GammaMatrix::new((n_g, n_i), a0, b0);
    let mut tau_low = GammaMatrix::new((n_g, n_i), a0, b0);

    let denom_high = Mat::from_fn(n_g, n_i, |_g, i| stat.size_high[i].max(1e-8));
    let denom_low = Mat::from_fn(n_g, n_i, |_g, i| stat.size_low_matched[i].max(1e-8));

    tau_high.update_stat(&stat.y_high, &denom_high);
    tau_high.calibrate_with(CalibrateTarget::MeanOnly);
    tau_high.calibrate();

    tau_low.update_stat(&stat.y_low_matched, &denom_low);
    tau_low.calibrate_with(CalibrateTarget::MeanOnly);
    tau_low.calibrate();

    (tau_high, tau_low)
}

#[cfg(test)]
mod recovery_tests {
    use super::*;
    use data_beans::sparse_io::create_sparse_from_ndarray;
    use data_beans::sparse_io_vector::SparseIoVec;
    use nalgebra::DMatrix;
    use ndarray::Array2;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Gamma as GammaDist, Poisson};
    use std::sync::Arc;
    type SparseData = dyn SparseIo<IndexIter = Vec<usize>>;

    #[test]
    fn recovers_spatial_marker_gene_in_correct_topic() -> anyhow::Result<()> {
        const NX: usize = 20;
        const NY: usize = 10;
        const N_GENES: usize = 30;
        const N_TOPICS: usize = 3;
        const MARKER: usize = 0;
        const TARGET_K: usize = 0;
        let n_cells = NX * NY;
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);

        let coords = DMatrix::<f32>::from_fn(n_cells, 2, |i, j| {
            if j == 0 {
                (i / NY) as f32
            } else {
                (i % NY) as f32
            }
        });

        let gamma_unit = GammaDist::new(1.0, 1.0).unwrap();
        let mut theta = Mat::zeros(n_cells, N_TOPICS);
        for i in 0..n_cells {
            let mut row = [0f32; N_TOPICS];
            let mut sum = 0f32;
            for r in row.iter_mut() {
                *r = gamma_unit.sample(&mut rng) as f32;
                sum += *r;
            }
            for (k, &r) in row.iter().enumerate() {
                theta[(i, k)] = r / sum;
            }
        }

        let bg = Poisson::new(0.5).unwrap();
        let hot = Poisson::new(15.0).unwrap();
        let mut counts = Array2::<f32>::zeros((N_GENES, n_cells));
        for i in 0..n_cells {
            for g in 0..N_GENES {
                counts[(g, i)] = bg.sample(&mut rng) as f32;
            }
        }
        let x_median = (NX as f32 - 1.0) * 0.5;
        let mut sorted: Vec<f32> = (0..n_cells).map(|i| theta[(i, TARGET_K)]).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let thr = sorted[(n_cells as f32 * 0.75) as usize];
        for i in 0..n_cells {
            if theta[(i, TARGET_K)] >= thr && coords[(i, 0)] < x_median {
                counts[(MARKER, i)] = hot.sample(&mut rng) as f32;
            }
        }

        let gene_names: Vec<Box<str>> = (0..N_GENES)
            .map(|g| format!("gene{g}").into_boxed_str())
            .collect();
        let cell_names: Vec<Box<str>> = (0..n_cells)
            .map(|i| format!("cell{i}").into_boxed_str())
            .collect();
        let mut sp = create_sparse_from_ndarray(&counts, None, None)?;
        sp.register_row_names_vec(&gene_names);
        sp.register_column_names_vec(&cell_names);
        sp.preload_columns()?;
        let sparse: Arc<SparseData> = Arc::from(sp);
        let mut data = SparseIoVec::new();
        data.push(sparse, Some("batch0".into()))?;

        let graph = build_spatial_graph(&coords, 8, 1000)?;

        let mut top_gene = [0usize; N_TOPICS];
        let mut top_c = [f32::NEG_INFINITY; N_TOPICS];
        let mut marker_c = [f32::NAN; N_TOPICS];

        for k in 0..N_TOPICS {
            let theta_k: Vec<f32> = (0..n_cells).map(|i| theta[(i, k)]).collect();
            let strata = stratify_topic(&theta_k, 0.25, 0.75);
            let matches = high_to_low_neighbors(&graph, &strata, None);
            let mut stat = TopicStat::new(N_GENES, 1);
            accumulate_topic(
                &data,
                &matches,
                &vec![0usize; n_cells],
                &theta_k,
                &strata,
                &mut stat,
                1000,
            )?;
            let (th, tl) = fit_gamma_pair(&stat, 1.0, 1.0);
            let hi = th.posterior_log_mean();
            let lo = tl.posterior_log_mean();
            for g in 0..N_GENES {
                let c = hi[(g, 0)] - lo[(g, 0)];
                if c > top_c[k] {
                    top_c[k] = c;
                    top_gene[k] = g;
                }
                if g == MARKER {
                    marker_c[k] = c;
                }
            }
        }

        assert_eq!(top_gene[TARGET_K], MARKER);
        assert!(marker_c[TARGET_K] > marker_c[1].max(marker_c[2]) + 0.5);
        Ok(())
    }
}
