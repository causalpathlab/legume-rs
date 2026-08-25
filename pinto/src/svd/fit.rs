use crate::link_community::profiles::{
    compute_propensity_and_gene_community_stat, PropensityReportConfig,
};
use crate::util::cell_pairs::*;
use crate::util::common::*;
use crate::util::graph_coarsen::*;
use crate::util::srt_pipeline::{
    preprocess_srt, topology_graph, GeneAxisMode, SrtPreprocessConfig, SrtPreprocessed,
};
use data_beans_alg::cell_pairs::CellPairs;
use data_beans_alg::random_projection::*;

use clap::Parser;
use matrix_param::dmatrix_gamma::GammaMatrix;
use matrix_param::traits::*;
use matrix_util::common_io::mkdir_parent;

#[derive(Parser, Debug, Clone)]
pub struct SrtDeltaSvdArgs {
    #[command(flatten)]
    pub common: crate::util::input::SrtInputArgs,

    #[arg(
        short = 't',
        long,
        default_value_t = 10,
        help = "Number of SVD components for latent pair representation"
    )]
    n_latent_topics: usize,

    /// How the pair latent becomes link communities. Shared verbatim with
    /// `cage` — the k-means fallback width is `--n-latent-topics` here.
    #[command(flatten)]
    edge_clustering: crate::util::edge_clustering::EdgeClusterArgs,
}

/// Input for fused multi-level pair delta visitor.
struct FusedDeltaInput<'a> {
    batch_effect: Option<&'a Mat>,
    pair_to_sample: &'a [usize],
}

/// Accumulated shared/difference statistics per gene per sample.
///
/// For each cell pair (left, right) and each gene g:
///   shared_g = log1p(x_left_g) + log1p(x_right_g)
///   diff_g   = |log1p(x_left_g) - log1p(x_right_g)|
pub(crate) struct PairDeltaCollapsedStat {
    shared_ds: Mat,
    diff_ds: Mat,
    size_s: DVec,
    n_genes: usize,
    n_samples: usize,
}

impl PairDeltaCollapsedStat {
    pub(crate) fn new(n_genes: usize, n_samples: usize) -> Self {
        Self {
            shared_ds: Mat::zeros(n_genes, n_samples),
            diff_ds: Mat::zeros(n_genes, n_samples),
            size_s: DVec::zeros(n_samples),
            n_genes,
            n_samples,
        }
    }

    pub(crate) fn optimize(
        &self,
        hyper_param: Option<(f32, f32)>,
    ) -> anyhow::Result<PairDeltaParameters> {
        let (a0, b0) = hyper_param.unwrap_or((1_f32, 1_f32));
        let shape = (self.n_genes, self.n_samples);

        let mut shared = GammaMatrix::new(shape, a0, b0);
        let mut diff = GammaMatrix::new(shape, a0, b0);

        let size_s = &self.size_s.transpose();
        let sample_size_ds = Mat::from_rows(&vec![size_s.clone(); shape.0]);

        info!("Calibrating pair delta statistics");

        shared.update_stat(&self.shared_ds, &sample_size_ds);
        shared.calibrate();
        diff.update_stat(&self.diff_ds, &sample_size_ds);
        diff.calibrate();

        info!("Resolved pair delta collapsed statistics");

        Ok(PairDeltaParameters { shared, diff })
    }
}

pub(crate) struct PairDeltaParameters {
    pub(crate) shared: GammaMatrix,
    pub(crate) diff: GammaMatrix,
}

/// Cell-pair SVD pipeline with shared/difference channels:
///
/// 1. Load data + coordinates
/// 2. Estimate batch effects
/// 3. Build spatial cell-cell KNN graph
/// 4. Random projection → assign pairs to samples
/// 5. Collapse: compute shared/diff per gene per sample
/// 6. Fit Poisson-Gamma on each channel
/// 7. SVD on vertically stacked [shared; diff] posterior log means
/// 8. Nystrom projection → per-pair latent codes
/// 9. Export dictionary + pair latents
pub fn fit_srt_delta_svd(args: &SrtDeltaSvdArgs) -> anyhow::Result<()> {
    let c = &args.common;
    mkdir_parent(&c.out)?;

    anyhow::ensure!(c.n_pseudobulk > 0, "n_pseudobulk must be > 0");
    anyhow::ensure!(args.n_latent_topics > 0, "n_latent_topics must be > 0");

    // 1-3. Load + KNN + batch effects (no Fisher weights).
    let SrtPreprocessed {
        data_vec,
        coordinates,
        coordinate_names,
        batch_membership,
        batch_effects: batch_db,
        graph,
        spatial_graph,
        edge_source,
        cell_proj: _,
        gene_axis: _,
        row_weights: _,
        row_stats: _,
        gene_weights: _,
        gene_stats: _,
        n_cells,
        n_rows: n_genes,
    } = preprocess_srt(SrtPreprocessConfig {
        common: c,
        fisher_weights: false,
        batch_effects: true,
        gene_axis: GeneAxisMode::Rows,
        // `dsvd` takes its own projection because it asks for a different one:
        // it batch-corrects unconditionally, where preprocessing corrects only
        // when batch effects were estimated. On a single-batch run those two
        // are not the same projection, so sharing would quietly change results.
        //
        // Note this is a request, not a contract: preprocessing still takes one
        // when `--knn-expr` needs it for the expression graph, and that one is
        // dropped here. Combining `--knn-expr` with coordinates therefore costs
        // two projection passes.
        cell_projection: false,
        feature_kind: None,
    })?;
    let has_coords = c.has_coordinates();
    let gene_names = data_vec.row_names()?;

    // Wrap graph with data for pair-level operations
    let srt_cell_pairs = SrtCellPairs::with_graph(
        &data_vec,
        &coordinates,
        &graph,
        edge_source.as_deref(),
        Some(&batch_membership),
    );

    srt_cell_pairs.write_coord_pairs(&c.out, &coordinate_names)?;

    // 4. Per-cell random projection
    info!("Per-cell random projection...");
    let mut cell_proj = data_vec.project_columns_with_batch_correction(
        c.proj_dim,
        c.block_size,
        Some(&batch_membership),
    )?;

    // 5. Graph-constrained coarsening + multi-level assignment
    info!(
        "Graph coarsening + multi-level assignment ({} levels, n_clusters={})...",
        c.num_levels, c.n_pseudobulk
    );

    let batch_ref = batch_db.as_ref();

    let topology = topology_graph(&graph, &spatial_graph);
    let ml = graph_coarsen_multilevel(
        topology,
        &mut cell_proj.proj,
        srt_cell_pairs.inner.pairs(),
        CoarsenConfig {
            n_clusters: c.n_pseudobulk,
            num_levels: c.num_levels,
            refine_iterations: c.refine_iterations,
            seeding: has_coords.then(|| SeedingParams {
                coordinates: &coordinates,
                batch_membership: Some(&batch_membership),
            }),
            modularity_veto: None,
            dc_poisson: None,
        },
    );

    // Only the FINEST level is ever read (the Poisson-Gamma fit below),
    // so only that one is accumulated. The coarser levels cost
    // `n_genes x n_samples` dense accumulators EACH, and every one of
    // them also multiplied the work done while the pass below holds its
    // lock; they were built and dropped unused.
    let finest_samples = *ml
        .all_num_samples
        .last()
        .ok_or(anyhow::anyhow!("no levels"))?;
    let finest_pair_to_sample = ml
        .all_pair_to_sample
        .last()
        .ok_or(anyhow::anyhow!("no levels"))?;
    info!(
        "Accumulating pair-delta statistics at the finest level ({} samples of {:?})",
        finest_samples, ml.all_num_samples
    );
    let mut collapsed = PairDeltaCollapsedStat::new(n_genes, finest_samples);

    let fused_input = FusedDeltaInput {
        batch_effect: batch_ref,
        pair_to_sample: finest_pair_to_sample,
    };

    srt_cell_pairs.inner.visit_pairs_by_block(
        &fused_pair_delta_visitor,
        &fused_input,
        &mut collapsed,
        c.block_size,
    )?;

    // 6. Fit Poisson-Gamma (finest level)
    info!("Fitting Poisson-Gamma model...");
    let collapsed_stat = &collapsed;
    let params = collapsed_stat.optimize(None)?;

    // 7. SVD on [shared; diff] posterior log means
    info!("Randomized SVD ({} components)...", args.n_latent_topics);

    let training_dm = concatenate_vertical(&[
        params.shared.posterior_log_mean().scale_columns(),
        params.diff.posterior_log_mean().scale_columns(),
    ])?;

    let (u_dk, s_k, _) = training_dm.rsvd(args.n_latent_topics)?;
    let basis_dk = nystrom_basis(&u_dk, &s_k);

    // Write dictionary
    let dict_row_names: Vec<Box<str>> = gene_names
        .iter()
        .map(|g| format!("{}@shared", g).into_boxed_str())
        .chain(
            gene_names
                .iter()
                .map(|g| format!("{}@diff", g).into_boxed_str()),
        )
        .collect();

    u_dk.to_parquet_with_names(
        &(c.out.to_string() + ".basis.parquet"),
        (Some(&dict_row_names), Some("gene")),
        None,
    )?;

    // 8. Nystrom projection
    info!("Nystrom projection...");

    let mut proj_kn = Mat::zeros(args.n_latent_topics, srt_cell_pairs.inner.num_pairs());

    let nystrom_input = NystromPairInput {
        basis_shared: basis_dk.rows(0, n_genes).clone_owned(),
        basis_diff: basis_dk.rows(n_genes, n_genes).clone_owned(),
        batch_effect: batch_db,
    };

    srt_cell_pairs.inner.visit_pairs_by_block(
        &nystrom_pair_delta_visitor,
        &nystrom_input,
        &mut proj_kn,
        c.block_size,
    )?;

    // 9. Export
    // L2-normalize each pair's latent vector so downstream clustering
    // is driven by direction rather than magnitude.
    proj_kn.normalize_columns_inplace();

    // One `[E × T]` copy, written out and then clustered — the shared routine
    // takes pairs as rows, so this is the same buffer both times.
    let proj_ne = proj_kn.transpose();
    proj_ne.to_parquet_with_names(
        &(c.out.to_string() + ".latent.parquet"),
        (None, Some("cell_pair")),
        None,
    )?;

    // 10. Propensity + dictionary
    let edges = srt_cell_pairs.inner.pairs();

    // Leiden by default, same as `cage`: the SVD width is chosen for how much
    // variance to keep, which is no reason for the pairs to fall into exactly
    // that many interaction regimes. `pinto prop` still re-cuts the same latent
    // at a fixed K when you want one.
    let n_clusters = compute_propensity_and_gene_community_stat(
        &proj_ne,
        edges,
        &data_vec,
        n_cells,
        &PropensityReportConfig {
            clustering: args.edge_clustering.resolve(c.seed),
            block_size: c.block_size,
            // `dsvd` never resolves a gene axis (it stacks its two channels on
            // the row axis), so its gene-community table stays row-keyed.
            gene_axis: None,
            edge_kind: srt_cell_pairs.edge_kind.as_deref(),
        },
        &c.out,
    )?;

    {
        use crate::util::metadata::{create_dsvd_metadata, RunInputs};
        let coord_file_str = c.coord_files_joined();
        let meta = create_dsvd_metadata(&RunInputs {
            prefix: &c.out,
            data_files: &c.data_files,
            coord_file: coord_file_str.as_deref(),
            coord_columns: &coordinate_names,
            n_cells,
            n_genes: data_vec.num_rows(),
            n_edges: edges.len(),
            k: n_clusters,
        });
        let meta_path = std::path::PathBuf::from(format!("{}.pinto.json", c.out));
        meta.write(&meta_path)?;
        info!("Wrote {}", meta_path.display());
    }

    info!("Done");
    Ok(())
}

/// Fused block-based visitor: read pair data once, accumulate into all levels' stats.
fn fused_pair_delta_visitor(
    bound: (usize, usize),
    data: &CellPairs,
    input: &FusedDeltaInput,
    arc_stats: Arc<Mutex<&mut PairDeltaCollapsedStat>>,
) -> anyhow::Result<()> {
    let (lb, ub) = bound;
    let pairs = &data.pairs()[lb..ub];
    let n_pairs = ub - lb;

    // Two sequential reads, deliberately NOT deduplicated. Endpoints do
    // repeat within a block, but a deduplicated read forces a per-pair
    // index lookup and scattered column access afterwards, and measured
    // slower on a large pair graph than simply reading both sides in
    // order and zipping them.
    let left = pairs.iter().map(|x| x.0);
    let right = pairs.iter().map(|x| x.1);
    let mut y_left = data.data.read_columns_csc(left)?;
    let mut y_right = data.data.read_columns_csc(right)?;

    // batch adjustment
    if let Some(delta_db) = input.batch_effect {
        let left = pairs.iter().map(|x| x.0);
        let right = pairs.iter().map(|x| x.1);
        let left_batches = data.data.get_batch_membership(left);
        y_left.adjust_by_division_of_selected_inplace(delta_db, &left_batches);
        let right_batches = data.data.get_batch_membership(right);
        y_right.adjust_by_division_of_selected_inplace(delta_db, &right_batches);
    }

    // Per-pair SPARSE contributions: a pair touches its nonzero genes, not
    // all of them, and the lock below only pays for what it touches.
    // The two CSC columns arrive with sorted, unique row indices, so the
    // union is a two-pointer merge with no map and no allocation.
    let mut genes: Vec<u32> = Vec::new();
    let mut shared: Vec<f32> = Vec::new();
    let mut diff: Vec<f32> = Vec::new();
    let mut offsets: Vec<usize> = Vec::with_capacity(n_pairs + 1);
    offsets.push(0);

    for (lc, rc) in y_left.col_iter().zip(y_right.col_iter()) {
        let (li, lv) = (lc.row_indices(), lc.values());
        let (ri, rv) = (rc.row_indices(), rc.values());
        let (mut a, mut b) = (0usize, 0usize);
        while a < li.len() || b < ri.len() {
            let take_left = b >= ri.len() || (a < li.len() && li[a] < ri[b]);
            let take_right = a >= li.len() || (b < ri.len() && ri[b] < li[a]);
            if take_left {
                let l = lv[a].ln_1p();
                genes.push(li[a] as u32);
                shared.push(l);
                diff.push(l);
                a += 1;
            } else if take_right {
                let r = rv[b].ln_1p();
                genes.push(ri[b] as u32);
                shared.push(r);
                diff.push(r);
                b += 1;
            } else {
                let l = lv[a].ln_1p();
                let r = rv[b].ln_1p();
                genes.push(li[a] as u32);
                shared.push(l + r);
                diff.push((l - r).abs());
                a += 1;
                b += 1;
            }
        }
        offsets.push(genes.len());
    }

    // Scatter under the lock. Held only for the touched genes of one
    // level, where it used to cover every gene row of every level.
    let mut stats = arc_stats.lock().expect("lock fused delta stats");
    for local_idx in 0..n_pairs {
        let sample = input.pair_to_sample[lb + local_idx];
        let mut col_shared = stats.shared_ds.column_mut(sample);
        for k in offsets[local_idx]..offsets[local_idx + 1] {
            col_shared[genes[k] as usize] += shared[k];
        }
        let mut col_diff = stats.diff_ds.column_mut(sample);
        for k in offsets[local_idx]..offsets[local_idx + 1] {
            col_diff[genes[k] as usize] += diff[k];
        }
        stats.size_s[sample] += 1.0;
    }

    Ok(())
}

/// Shared input for Nystrom pair-delta projection.
struct NystromPairInput {
    basis_shared: Mat,
    basis_diff: Mat,
    batch_effect: Option<Mat>,
}

/// Nystrom projection visitor: project each pair onto the split basis.
///
/// For each pair and each gene present in either cell:
///   shared = log1p(x_left) + log1p(x_right)
///   diff   = |log1p(x_left) - log1p(x_right)|
///   proj  += shared * basis_shared[gene] + diff * basis_diff[gene]
fn nystrom_pair_delta_visitor(
    bound: (usize, usize),
    data: &CellPairs,
    shared_in: &NystromPairInput,
    arc_proj: Arc<Mutex<&mut Mat>>,
) -> anyhow::Result<()> {
    let (lb, ub) = bound;
    let pairs = &data.pairs()[lb..ub];
    let left = pairs.iter().map(|pp| pp.0);
    let right = pairs.iter().map(|pp| pp.1);

    let mut y_left = data.data.read_columns_csc(left)?;
    let mut y_right = data.data.read_columns_csc(right)?;

    // batch adjustment
    if let Some(delta_db) = &shared_in.batch_effect {
        let left = pairs.iter().map(|x| x.0);
        let right = pairs.iter().map(|x| x.1);
        let left_batches = data.data.get_batch_membership(left);
        y_left.adjust_by_division_of_selected_inplace(delta_db, &left_batches);
        let right_batches = data.data.get_batch_membership(right);
        y_right.adjust_by_division_of_selected_inplace(delta_db, &right_batches);
    }

    let n_topics = shared_in.basis_shared.ncols();
    let n_pairs_block = ub - lb;
    let mut local_proj = Mat::zeros(n_topics, n_pairs_block);

    for (pair_idx, (left_col, right_col)) in y_left.col_iter().zip(y_right.col_iter()).enumerate() {
        let right_log: HashMap<usize, f32> = right_col
            .row_indices()
            .iter()
            .zip(right_col.values().iter())
            .map(|(&g, &v)| (g, v.ln_1p()))
            .collect();

        let mut proj_k = DVec::zeros(n_topics);
        let mut left_visited: HashSet<usize> = Default::default();

        for (&gene, &val) in left_col.row_indices().iter().zip(left_col.values().iter()) {
            let log_left = val.ln_1p();
            let log_right = right_log.get(&gene).copied().unwrap_or(0.0);
            let sigma = log_left + log_right;
            let delta = (log_left - log_right).abs();
            proj_k += sigma * &shared_in.basis_shared.row(gene).transpose();
            proj_k += delta * &shared_in.basis_diff.row(gene).transpose();
            left_visited.insert(gene);
        }

        // Right-only genes: log_left = 0 → sigma = log_right, delta = log_right
        for (&gene, _) in right_col
            .row_indices()
            .iter()
            .zip(right_col.values().iter())
        {
            if !left_visited.contains(&gene) {
                let log_right = right_log[&gene];
                proj_k += log_right * &shared_in.basis_shared.row(gene).transpose();
                proj_k += log_right * &shared_in.basis_diff.row(gene).transpose();
            }
        }

        local_proj.column_mut(pair_idx).copy_from(&proj_k);
    }

    let mut proj = arc_proj.lock().expect("lock nystrom pair delta proj");
    proj.columns_range_mut(lb..ub).copy_from(&local_proj);

    Ok(())
}
