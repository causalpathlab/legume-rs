//! Link community model pipeline for spatial transcriptomics.
//!
//! Discovers link communities from spatial cell-cell KNN graphs via a V-cycle
//! through the coarsening pyramid: collapsed Gibbs + greedy at every level
//! (coarsest → finest) with label inheritance between levels, followed by
//! component-EM refinement at full fine resolution.
//!
//! Pipeline:
//!   1. Load data + coordinates
//!   2. Build spatial (or expression) KNN graph
//!   3. Estimate batch effects
//!   4. Multi-level cell coarsening (produces the pyramid)
//!   5. Profile context setup: gene-pair graph + elbow filter + module
//!      collapse, or projection basis with optional gene filtering
//!   6. Build full fine-resolution profiles
//!   7. V-cycle: per-level super-edges → profiles → Gibbs + greedy with
//!      coarse→fine label inheritance; per-level outputs are written inside
//!   8. Outer EM loop on full profiles (E: component-EM + greedy; optional M:
//!      re-cluster gene-pair modules)
//!   9. Final outputs (propensity, gene_topic, link_community, scores, BHC)

use crate::gene_network::graph::*;
use crate::gene_network::pairs::*;
use crate::link_community::cascade::{run_cascade, CascadeConfig, GenePairProfileState};
use crate::link_community::gibbs::{ComponentGibbsArgs, LinkGibbsSampler};
use crate::link_community::model::{LinkCommunityStats, LinkProfileStore};
use crate::link_community::module_cluster::{self, KmeansModules, ModuleClusterer};
use crate::link_community::outputs::{
    link_community_histogram, write_bhc_cut, write_bhc_merges, write_partition_outputs,
    write_score_trace, ScoreEntry,
};
use crate::link_community::profiles::*;
use crate::util::batch_effects::{estimate_and_write_batch_effects, EstimateBatchArgs};
use crate::util::cell_pairs::*;
use crate::util::common::*;
use crate::util::graph_coarsen::*;
use crate::util::input::*;
use data_beans_alg::random_projection::RandProjOps;
use rand::rngs::SmallRng;
use rand::SeedableRng;

// Re-export args so external callers (main.rs) keep using the old path.
pub use crate::link_community::args::SrtLinkCommunityArgs;

/// Main entry point for `pinto lc`.
pub fn fit_srt_link_community(args: &SrtLinkCommunityArgs) -> anyhow::Result<()> {
    let c = &args.common;
    let k = args.n_communities;
    let n_outer_iter = args.n_outer_iter.max(1);

    //////////////////////////////////////////////////////
    // 1. Load data (with or without coordinates)       //
    //////////////////////////////////////////////////////
    info!("Loading data files...");

    let has_coords = c.has_coordinates();

    let SRTData {
        data: mut data_vec,
        mut coordinates,
        mut coordinate_names,
        batches: mut batch_membership,
    } = if has_coords {
        read_data_with_coordinates(c.to_read_args())?
    } else {
        info!("No coordinate files provided — using expression mode");
        read_data_without_coordinates(c.to_read_args())?
    };

    let n_genes = data_vec.num_rows();
    let n_cells = data_vec.num_columns();

    anyhow::ensure!(c.proj_dim > 0, "proj_dim must be > 0");
    anyhow::ensure!(args.n_communities > 0, "n_communities must be > 0");
    anyhow::ensure!(c.knn_spatial > 0, "knn_spatial must be > 0");
    anyhow::ensure!(
        c.knn_spatial < n_cells,
        "knn_spatial ({}) must be < number of cells ({})",
        c.knn_spatial,
        n_cells
    );

    //////////////////////////////////////////////////////
    // 2. Build KNN graph (spatial or expression-based) //
    //////////////////////////////////////////////////////
    let graph;

    if has_coords {
        info!("Building spatial KNN graph (k={})...", c.knn_spatial);
        graph = build_spatial_graph(
            &coordinates,
            SrtCellPairsArgs {
                knn: c.knn_spatial,
                block_size: c.block_size,
                reciprocal: c.reciprocal,
            },
        )?;
    } else {
        info!(
            "Building expression KNN graph (k={}, proj_dim={})...",
            c.knn_spatial, c.proj_dim
        );
        let cell_proj_pre = data_vec.project_columns_with_batch_correction(
            c.proj_dim,
            c.block_size,
            None::<&[Box<str>]>,
        )?;
        let (g, embedding) = build_expression_graph(
            &cell_proj_pre.proj,
            SrtCellPairsArgs {
                knn: c.knn_spatial,
                block_size: c.block_size,
                reciprocal: c.reciprocal,
            },
        )?;
        graph = g;
        coordinates = embedding;
        coordinate_names = vec!["pc_1".into(), "pc_2".into()];
    }

    // Auto-detect batches from connected components (opt-in via --auto-batch)
    if c.auto_batch && c.batch_files.is_none() {
        crate::util::input::auto_batch_from_components(&graph, &mut batch_membership);
    }

    //////////////////////////////////////////////////////
    // 3. Estimate batch effects (skipped single-batch) //
    //////////////////////////////////////////////////////
    let batch_sort_dim = c.proj_dim.min(10);
    let batch_db = estimate_and_write_batch_effects(
        &mut data_vec,
        &batch_membership,
        EstimateBatchArgs {
            proj_dim: c.proj_dim,
            sort_dim: batch_sort_dim,
            block_size: c.block_size,
            batch_knn: c.batch_knn,
            num_levels: c.num_levels,
        },
        &c.out,
    )?;

    // Edge-module clusterer (only used in gene-pair mode when the raw
    // gene-pair dimension exceeds --n-edge-modules).
    let edge_clusterer = KmeansModules {
        n_modules: args.n_edge_modules.max(2),
        max_iter: 100,
    };

    // Gene-pair mode: load external network before borrowing data_vec
    let mut gene_pair_state: Option<(GenePairGraph, DVec)> = None;
    if let Some(ref network_file) = args.gene_network {
        let gene_names = data_vec.row_names()?;
        info!("Loading external gene network from {}...", network_file);
        let gene_graph = GenePairGraph::from_edge_list(
            network_file,
            gene_names,
            args.gene_network_allow_prefix,
            args.gene_network_delimiter,
        )?;
        anyhow::ensure!(
            gene_graph.num_edges() > 0,
            "Gene network matched 0 gene pairs. Check that gene names in {} \
             match the data gene names (use --gene-network-delimiter or \
             --gene-network-allow-prefix for fuzzy matching).",
            network_file
        );
        info!("Gene graph: {} gene pairs", gene_graph.num_edges());
        let gene_means = compute_gene_raw_means(&data_vec, c.block_size)?;
        gene_pair_state = Some((gene_graph, gene_means));
    }

    // Wrap graph with data for pair-level operations
    let srt_cell_pairs = SrtCellPairs::with_graph(&data_vec, &coordinates, graph);

    srt_cell_pairs.to_parquet(
        &(c.out.to_string() + ".coord_pairs.parquet"),
        Some(coordinate_names.clone()),
    )?;

    let edges = &srt_cell_pairs.graph.edges;
    let n_edges = edges.len();
    info!("{} cells, {} edges", n_cells, n_edges);

    //////////////////////////////////////////////////////
    // 4. Multi-level cell coarsening                   //
    //////////////////////////////////////////////////////
    info!(
        "Graph coarsening ({} levels, {} coarse clusters)...",
        c.num_levels, c.n_pseudobulk
    );

    let batch_arg: Option<&[Box<str>]> = if batch_db.is_some() {
        Some(&batch_membership)
    } else {
        None
    };

    let cell_proj =
        data_vec.project_columns_with_batch_correction(c.proj_dim, c.block_size, batch_arg)?;

    let ml = graph_coarsen_multilevel(
        &srt_cell_pairs.graph,
        &mut cell_proj.proj.clone(),
        &srt_cell_pairs.pairs,
        CoarsenConfig {
            n_clusters: c.n_pseudobulk,
            num_levels: c.num_levels,
            refine_iterations: c.refine_iterations,
            seeding: has_coords.then(|| SeedingParams {
                coordinates: &coordinates,
                batch_membership: Some(&batch_membership),
            }),
            modularity_veto: (args.modularity_gamma > 0.0).then_some(ModularityVeto {
                gamma: args.modularity_gamma,
            }),
            dc_poisson: Some(DcPoissonConfig {
                params: data_beans_alg::dc_poisson::RefineParams {
                    num_gibbs: 10,
                    num_greedy: 5,
                    gene_weighting: data_beans_alg::dc_poisson::GeneWeighting::FisherInfoNb,
                    seed: c.seed,
                    gibbs_stagnation: 0.005,
                    profile_source: data_beans_alg::dc_poisson::ProfileSource::Raw,
                },
                data: &data_vec,
                num_genes: n_genes,
            }),
        },
    );

    //////////////////////////////////////////////////////
    // 5. Profile context setup                         //
    //////////////////////////////////////////////////////
    // Two modes:
    //   - Gene-pair: load the user's external gene-gene network, build a
    //     raw coarse-level profile to run elbow filtering, optionally cluster
    //     the remaining gene-pairs into edge modules via K-means.
    //   - Projection: use the cell embedding's random-projection basis, with
    //     optional low-count gene filtering.
    let mut gp_profile_state: Option<GenePairProfileState>;
    let proj_basis: Option<Mat>;

    // Coarsest super-edges are needed once for the elbow filter / module
    // K-means (gene-pair mode). The cascade rebuilds level-0 super-edges
    // internally when it runs.
    let coarsest_labels = &ml.all_cell_labels[0];
    let (coarse_super_edges, _) = build_super_edges(edges, coarsest_labels);
    let n_super = coarse_super_edges.len();
    info!(
        "Coarsest level: {} super-edges from {} fine edges",
        n_super, n_edges
    );

    if let Some((ref mut gene_graph, ref gene_means)) = gene_pair_state {
        let n_gene_pairs = gene_graph.num_edges();

        info!("Building coarse gene-pair profiles for elbow filter...");
        let gene_adj = gene_graph.build_directed_adjacency();
        let raw_profiles = build_edge_profiles_by_gene_pairs(
            &data_vec,
            &coarse_super_edges,
            &gene_adj,
            gene_means,
            n_gene_pairs,
            c.block_size,
        )?;

        // Elbow filter on coarse column sums.
        let col_sums: Vec<f32> = (0..n_gene_pairs)
            .map(|p| {
                let mut s = 0.0f32;
                for e in 0..n_super {
                    s += raw_profiles.profile(e)[p];
                }
                s
            })
            .collect();

        let (threshold, elbow_rank) = elbow_threshold(&col_sums);

        let keep_cols: Vec<usize> = (0..n_gene_pairs)
            .filter(|&p| col_sums[p] > threshold)
            .collect();

        let (filtered_profiles, final_n_gene_pairs) = if keep_cols.len() < n_gene_pairs {
            info!(
                "Elbow threshold: {:.4} (rank {}), kept {}/{} gene pairs",
                threshold,
                elbow_rank,
                keep_cols.len(),
                n_gene_pairs
            );
            gene_graph.filter_edges(&keep_cols);
            (
                filter_profile_columns(&raw_profiles, &keep_cols),
                keep_cols.len(),
            )
        } else {
            (raw_profiles, n_gene_pairs)
        };

        gene_graph.to_parquet(&(c.out.to_string() + ".gene_graph.parquet"))?;

        let gene_adj = gene_graph.build_directed_adjacency();

        // Collapse gene-pair columns into modules if too many.
        let module_collapse = if args.n_edge_modules > 0 && final_n_gene_pairs > args.n_edge_modules
        {
            info!(
                "Clustering {} gene pairs into {} edge modules...",
                final_n_gene_pairs, args.n_edge_modules
            );
            let (_module_profiles, assignments) =
                collapse_profile_columns(&filtered_profiles, &edge_clusterer);
            Some(assignments)
        } else {
            None
        };

        gp_profile_state = Some(GenePairProfileState {
            gene_adj,
            gene_means: gene_means.clone(),
            n_gene_pairs: final_n_gene_pairs,
            module_collapse,
        });
        proj_basis = None;
    } else {
        // Compressed all-gene edge profiles (default): y_e = W^T (x_i + x_j)
        // with W a G × proj_dim Gaussian basis.
        let mut basis = cell_proj.basis.clone();

        if args.min_gene_count > 0.0 {
            info!("Computing gene totals for filtering...");
            let gene_totals = compute_gene_totals(&data_vec, c.block_size)?;
            let n_kept = filter_basis_by_gene_count(&mut basis, &gene_totals, args.min_gene_count);
            info!(
                "Kept {}/{} genes (min_count={:.0})",
                n_kept, n_genes, args.min_gene_count
            );
        }

        gp_profile_state = None;
        proj_basis = Some(basis);
    }

    //////////////////////////////////////////////////////
    // 6. Full fine-resolution profiles                  //
    //////////////////////////////////////////////////////
    let all_edge_indices: Vec<usize> = (0..n_edges).collect();
    let (mut full_profiles, full_raw_gp_profiles) = build_full_profiles_inline(
        &data_vec,
        &all_edge_indices,
        edges,
        &gp_profile_state,
        &proj_basis,
    )?;

    info!(
        "Full profiles: {} edges × {} dims ({:.1} MB)",
        full_profiles.n_edges,
        full_profiles.m,
        (full_profiles.profiles.len() * std::mem::size_of::<f32>()) as f64 / 1_048_576.0
    );

    //////////////////////////////////////////////////////
    // 7. V-cycle cascade through the pyramid            //
    //////////////////////////////////////////////////////
    let cell_names = data_vec.column_names()?;
    let mut sampler = LinkGibbsSampler::new(SmallRng::seed_from_u64(c.seed));

    let cascade_cfg = CascadeConfig {
        k,
        num_gibbs: args.num_gibbs,
        num_greedy: args.num_greedy,
        user_alpha: args.alpha,
        block_size: c.block_size,
        no_level_outputs: args.no_level_outputs,
    };
    let cascade_result = run_cascade(
        &c.out,
        edges,
        &ml.all_cell_labels,
        &data_vec,
        gp_profile_state.as_ref(),
        proj_basis.as_ref(),
        &cascade_cfg,
        &mut sampler,
        &cell_names,
    )?;

    let mut current_labels = cascade_result.fine_labels;
    let mut score_trace: Vec<ScoreEntry> = cascade_result.score_trace;

    //////////////////////////////////////////////////////
    // 8. Outer EM loop: E-step (component-EM + greedy) //
    //    + optional M-step (re-cluster gene-pair mods) //
    //////////////////////////////////////////////////////
    // Alpha for EM: auto-scale from full-resolution sparsity (or user override).
    let alpha: f64 = args.alpha.map_or_else(
        || {
            let full_mean_sf = full_profiles.size_factors.iter().sum::<f32>()
                / full_profiles.n_edges.max(1) as f32;
            let v = (full_mean_sf as f64 / k as f64).max(0.01);
            info!(
                "EM α = {:.4} (mean size factor {:.1} / K={})",
                v, full_mean_sf, k
            );
            v
        },
        |v| v as f64,
    );
    let comp_args = ComponentGibbsArgs {
        graph: &srt_cell_pairs.graph,
        edges,
        k,
        alpha,
    };

    let num_fine_sweeps = args.num_em.unwrap_or((args.num_gibbs / 4).max(5));

    // M-step only meaningful when gene-pair-module collapse is active.
    let has_mstep = gp_profile_state
        .as_ref()
        .is_some_and(|gp| gp.module_collapse.is_some());
    let effective_outer = if has_mstep { n_outer_iter } else { 1 };
    let mut prev_score = f64::NEG_INFINITY;

    for outer_iter in 0..effective_outer {
        if effective_outer > 1 {
            info!(
                "=== Outer iteration {}/{} ===",
                outer_iter + 1,
                effective_outer
            );
        }

        if num_fine_sweeps > 0 {
            info!("EM Gibbs on full edge set ({} sweeps)...", num_fine_sweeps);
            let moves = sampler.run_components_em(
                &mut current_labels,
                &full_profiles,
                &comp_args,
                num_fine_sweeps,
            );
            info!("EM Gibbs: {} moves", moves);
        }

        info!("Greedy finalization ({} max sweeps)...", args.num_greedy);
        let greedy_moves = sampler.run_greedy_by_components(
            &mut current_labels,
            &full_profiles,
            &comp_args,
            args.num_greedy,
        );

        let fine_stats = LinkCommunityStats::from_profiles(&full_profiles, k, &current_labels);
        let score = fine_stats.total_score();
        let total_mass: f64 = fine_stats.size_sum.iter().sum();
        let mi = fine_stats.mutual_information();
        info!(
            "Greedy: {} moves, score={:.4e}, score/mass={:.4e}, MI={:.4} nats",
            greedy_moves,
            score,
            if total_mass > 0.0 {
                score / total_mass
            } else {
                0.0
            },
            mi
        );
        // Final-phase score rows tagged with level = -1, one per outer EM iter.
        score_trace.push(ScoreEntry {
            level: -1,
            sweep: outer_iter as i32,
            score,
            n_edges: fine_stats.n_edges,
            total_mass,
            mutual_information: mi,
        });

        if outer_iter + 1 >= n_outer_iter {
            break;
        }
        if score <= prev_score * 1.001 && prev_score > f64::NEG_INFINITY {
            info!("Outer EM converged (score plateau), stopping early");
            break;
        }
        prev_score = score;

        if let (Some(raw), Some(ref mut gp)) = (&full_raw_gp_profiles, &mut gp_profile_state) {
            if gp.module_collapse.is_some() {
                info!("M-step: recomputing gene-pair modules from community rates...");
                let raw_stats = LinkCommunityStats::from_profiles(raw, k, &current_labels);
                let rate_mat = compute_module_rate_matrix(&raw_stats);
                let mut new_collapse = edge_clusterer.cluster(&rate_mat);
                module_cluster::compact_labels(&mut new_collapse);
                let new_n_mod = new_collapse.iter().copied().max().unwrap_or(0) + 1;
                info!("Re-estimated {} edge modules", new_n_mod);

                full_profiles = raw.collapse_modules(&new_collapse);
                gp.module_collapse = Some(new_collapse);
            }
        }
    }

    // Final stats from current labels (fresh build — no drift).
    let fine_stats = LinkCommunityStats::from_profiles(&full_profiles, k, &current_labels);
    let final_membership = fine_stats.membership.clone();

    if log::log_enabled!(log::Level::Info) {
        eprintln!();
        eprintln!("{}", link_community_histogram(&final_membership, k, 50));
        eprintln!();
    }

    //////////////////////////////////////////////////////
    // 9. Extract and write final outputs               //
    //////////////////////////////////////////////////////
    // BHC-collapsed results (when enabled and non-degenerate) are the
    // authoritative `{out}.*` outputs. The pre-BHC fine partition is
    // written under `{out}.draft.*` so users can still inspect it.
    let bhc_enabled = !args.no_bhc;
    let draft_prefix = if bhc_enabled {
        format!("{}.draft", c.out)
    } else {
        c.out.to_string()
    };
    info!(
        "Writing {} outputs (propensity, gene_topic, link_community) → {}.*",
        if bhc_enabled { "draft" } else { "final" },
        draft_prefix
    );
    write_partition_outputs(
        &draft_prefix,
        edges,
        &final_membership,
        n_cells,
        k,
        &cell_names,
        &data_vec,
        c.block_size,
    )?;

    info!("Writing score trace...");
    write_score_trace(&(c.out.to_string() + ".scores.parquet"), &score_trace)?;

    if bhc_enabled {
        let gamma = args.bhc_gamma.unwrap_or(1.0);
        info!(
            "Running BHC merge (γ={:.4}, bg-empirical) over K={} communities...",
            gamma, k
        );
        let merges = crate::link_community::bhc::bhc_merge(&fine_stats, gamma);
        write_bhc_merges(&(c.out.to_string() + ".bhc.merges.parquet"), &merges)?;

        let labels = crate::link_community::bhc::bhc_cut(&merges, k, args.bhc_cut);
        let n_consensus = labels
            .iter()
            .filter(|&&v| v >= 0)
            .map(|&v| v as usize)
            .max()
            .map_or(0, |m| m + 1);
        info!(
            "BHC redundancy cut (log_bf ≥ {:.3}): {} merged communities",
            args.bhc_cut, n_consensus
        );
        write_bhc_cut(&(c.out.to_string() + ".bhc.cut.parquet"), &labels)?;

        if n_consensus > 0 {
            let consensus_membership: Vec<usize> = final_membership
                .iter()
                .map(|&orig| {
                    let lab = labels[orig];
                    debug_assert!(
                        lab >= 0,
                        "non-empty community {} has no consensus label",
                        orig
                    );
                    lab as usize
                })
                .collect();
            info!(
                "Writing final outputs (propensity, gene_topic, link_community) → {}.*",
                c.out
            );
            write_partition_outputs(
                &c.out,
                edges,
                &consensus_membership,
                n_cells,
                n_consensus,
                &cell_names,
                &data_vec,
                c.block_size,
            )?;
        } else {
            info!(
                "BHC produced no merges; draft outputs at {}.* are the final result",
                draft_prefix
            );
        }
    }

    info!("Done");
    Ok(())
}

/// Build full-resolution edge profiles, optionally keeping raw (un-collapsed)
/// gene-pair profiles for the outer-EM M-step.
///
/// Returns `(collapsed_profiles, Option<raw_gp_profiles>)`.
fn build_full_profiles_inline(
    data: &SparseIoVec,
    edge_indices: &[usize],
    all_edges: &[(usize, usize)],
    gp_state: &Option<GenePairProfileState>,
    proj_basis: &Option<Mat>,
) -> anyhow::Result<(LinkProfileStore, Option<LinkProfileStore>)> {
    if let Some(ref gp) = *gp_state {
        let raw = build_gene_pair_profiles_for_edges(
            data,
            edge_indices,
            all_edges,
            &gp.gene_adj,
            &gp.gene_means,
            gp.n_gene_pairs,
        )?;
        if let Some(ref assignments) = gp.module_collapse {
            let collapsed = raw.collapse_modules(assignments);
            Ok((collapsed, Some(raw)))
        } else {
            Ok((raw, None))
        }
    } else {
        let basis = proj_basis.as_ref().expect("projection basis missing");
        let profiles = build_projection_profiles_for_edges(data, edge_indices, all_edges, basis)?;
        Ok((profiles, None))
    }
}
