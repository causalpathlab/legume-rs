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
//!   5. Profile context setup:
//!        - Gene-network path: SNN-augment → k-core trim → Leiden modules →
//!          `ModulePairBasis` + per-cell module expression (`x_{c,m}`).
//!        - Projection path: Gaussian random-projection basis over genes,
//!          with optional low-count gene filtering.
//!   6. Build full fine-resolution profiles (sparse CSR)
//!   7. V-cycle: per-level super-edges → profiles → Gibbs + greedy with
//!      coarse→fine label inheritance; per-level outputs are written inside
//!   8. Single-pass component-EM + greedy on full profiles
//!   9. Final outputs (propensity, gene_community, link_community, scores,
//!      cosine dictionary merge)

use crate::gene_network::graph::*;
use crate::gene_network::modules::{kcore_trim, leiden_gene_modules};
use crate::link_community::cascade::{run_cascade, CascadeConfig, ModulePairContext, ProfileMode};
use crate::link_community::dict_merge::{cosine_cut, cosine_merge};
use crate::link_community::gibbs::{ComponentGibbsArgs, IncidenceConfig, LinkGibbsSampler};
use crate::link_community::incidence::{fit_log_incidence, pack_propensity_row_major};
use crate::link_community::model::{LinkCommunityStats, LinkProfileStore};
use crate::link_community::outputs::{
    link_community_histogram, write_dict_cut, write_dict_merges, write_partition_outputs,
    write_score_trace, ScoreEntry,
};
use crate::link_community::profiles::*;
use crate::util::cell_pairs::*;
use crate::util::common::*;
use crate::util::graph_coarsen::*;
use crate::util::srt_pipeline::{preprocess_srt, SrtPreprocessConfig, SrtPreprocessed};
use data_beans_alg::random_projection::RandProjOps;
use matrix_util::common_io::mkdir_parent;
use rand::rngs::SmallRng;
use rand::SeedableRng;

// Re-export args so external callers (main.rs) keep using the old path.
pub use crate::link_community::args::SrtLinkCommunityArgs;

/// Main entry point for `pinto lc`.
pub fn fit_srt_link_community(args: &SrtLinkCommunityArgs) -> anyhow::Result<()> {
    let c = &args.common;
    let k = args.n_communities;

    mkdir_parent(&c.out)?;

    anyhow::ensure!(args.n_communities > 0, "n_communities must be > 0");

    //////////////////////////////////////////////////////
    // 1-3. Load + KNN + batch effects + gene weights   //
    //////////////////////////////////////////////////////
    let SrtPreprocessed {
        data_vec,
        coordinates,
        coordinate_names,
        batch_membership,
        batch_effects: batch_db,
        graph,
        gene_weights,
        n_cells,
        n_genes,
    } = preprocess_srt(SrtPreprocessConfig {
        common: c,
        fisher_weights: true,
        batch_effects: true,
        feature_kind: None,
    })?;
    let has_coords = c.has_coordinates();
    let gene_weights = gene_weights.expect("fisher_weights=true must yield Some");

    //////////////////////////////////////////////////////
    // 4-pre. Gene network setup (if provided)           //
    //////////////////////////////////////////////////////
    // Resolve gene modules on the SNN-augmented, k-core-trimmed graph before
    // building the cell-cell KNN coarsening, so the module-pair basis is
    // ready to feed the V-cycle.
    let mut module_ctx: Option<ModulePairContext> = None;
    if let Some(ref network_file) = args.gene_network {
        let gene_names = data_vec.row_names()?;
        info!("Loading external gene network from {}...", network_file);
        let mut gene_graph = GenePairGraph::from_edge_list(
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

        gene_graph.augment_with_snn(args.snn_min_shared);

        let keep = kcore_trim(&gene_graph, args.gene_trim_min_degree);
        let module_of_gene =
            leiden_gene_modules(&gene_graph, &keep, args.gene_modules_resolution, c.seed);

        gene_graph.to_parquet(
            &(c.out.to_string() + ".gene_graph.parquet"),
            ("gene1", "gene2"),
        )?;

        let basis = ModulePairBasis::build(&gene_graph, module_of_gene);
        anyhow::ensure!(
            basis.n_pairs > 0,
            "Gene-network module-pair basis is empty (0 pairs). Try a lower \
             --gene-trim-min-degree or --gene-modules-resolution, or disable \
             --gene-network to use the projection basis."
        );

        info!("Computing per-cell module expression (NB Fisher gene-weighted)...");
        let (module_expr, cell_totals) = build_module_expression(
            &data_vec,
            &basis.module_of_gene,
            basis.n_modules,
            Some(&gene_weights),
            c.block_size,
        )?;

        module_ctx = Some(ModulePairContext {
            basis,
            module_expr,
            cell_totals,
        });
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
                    feature_weighting: data_beans_alg::dc_poisson::FeatureWeighting::FisherInfoNb,
                    seed: c.seed,
                    gibbs_stagnation: 0.005,
                    profile_source: data_beans_alg::dc_poisson::ProfileSource::Raw,
                    ..Default::default()
                },
                data: &data_vec,
                num_genes: n_genes,
            }),
        },
    );

    //////////////////////////////////////////////////////
    // 5. Profile context: module-pair OR projection     //
    //////////////////////////////////////////////////////
    let proj_basis: Option<Mat> = if module_ctx.is_none() {
        // Projection mode: Gaussian random-projection basis with optional
        // low-count gene filtering, then NB Fisher-info gene weighting baked
        // into the basis: basis'[g, m] = w_g · basis[g, m]. Equivalent to
        // projecting w·x instead of x — housekeeping / high-mean-high-
        // dispersion genes get attenuated (w_g → 0), informative genes
        // recover w_g ≈ 1.
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
        apply_gene_weights(&mut basis, &gene_weights);
        Some(basis)
    } else {
        None
    };

    let profile_mode = if let Some(ctx) = module_ctx.as_ref() {
        ProfileMode::ModulePair {
            basis: &ctx.basis,
            module_expr: &ctx.module_expr,
            cell_totals: &ctx.cell_totals,
        }
    } else {
        ProfileMode::Projection {
            basis: proj_basis.as_ref().expect("projection basis must be set"),
        }
    };

    //////////////////////////////////////////////////////
    // 6. Full fine-resolution profiles                  //
    //////////////////////////////////////////////////////
    let all_edge_indices: Vec<usize> = (0..n_edges).collect();
    let full_profiles = build_full_profiles(
        &data_vec,
        &all_edge_indices,
        edges,
        &profile_mode,
        c.block_size,
    )?;

    let density_pct = 100.0 * full_profiles.nnz() as f64
        / (full_profiles.n_edges.max(1) * full_profiles.m.max(1)) as f64;
    info!(
        "Full profiles: {} edges × {} dims — nnz={} ({:.2}% dense, {:.1} MB)",
        full_profiles.n_edges,
        full_profiles.m,
        full_profiles.nnz(),
        density_pct,
        ((full_profiles.values.len() * std::mem::size_of::<f32>())
            + (full_profiles.indices.len() * std::mem::size_of::<u32>())) as f64
            / 1_048_576.0
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
        &profile_mode,
        &cascade_cfg,
        &mut sampler,
        &cell_names,
        Some(&gene_weights),
    )?;

    let mut current_labels = cascade_result.fine_labels;
    let mut score_trace: Vec<ScoreEntry> = cascade_result.score_trace;
    let cascade_level_indices = cascade_result.written_level_indices;

    //////////////////////////////////////////////////////
    // 8. Component-EM + greedy on full profiles        //
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
    // Optional frozen K×K incidence (hybrid: Gibbs for edges, VB for B).
    // Vertex propensity θ and the variational expected log-incidence
    // E_q[log B] = ψ(a + S) − log(b + W) are both computed once from the
    // cascade's fine-resolution labels and held fixed for the duration of
    // EM + greedy. No per-move bookkeeping; the score gains the term
    //   Σ_{k'} (θ_L[k'] + θ_R[k']) · log_B[k, k'].
    let (incidence_propensity, incidence_log_b): (Option<Vec<f64>>, Option<Vec<f64>>) =
        if !args.no_incidence {
            let prop_mat = compute_node_membership(edges, &current_labels, n_cells, k);
            let prop_flat = pack_propensity_row_major(&prop_mat);
            let log_b = fit_log_incidence(
                edges,
                &current_labels,
                &prop_flat,
                k,
                args.incidence_a,
                args.incidence_b,
            );
            info!(
                "Incidence enabled (VB log B, hybrid Gibbs): K={}, Gamma prior (a={:.3}, b={:.3})",
                k, args.incidence_a, args.incidence_b
            );
            (Some(prop_flat), Some(log_b))
        } else {
            (None, None)
        };

    let comp_args = ComponentGibbsArgs {
        graph: &srt_cell_pairs.graph,
        edges,
        k,
        alpha,
        incidence: match (incidence_propensity.as_deref(), incidence_log_b.as_deref()) {
            (Some(p), Some(lb)) => Some(IncidenceConfig {
                propensity: p,
                log_incidence: lb,
            }),
            _ => None,
        },
    };

    let num_fine_sweeps = args.num_em.unwrap_or((args.num_gibbs / 4).max(5));

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
    score_trace.push(ScoreEntry {
        level: -1,
        sweep: 0,
        score,
        n_edges: fine_stats.n_edges,
        total_mass,
        mutual_information: mi,
    });

    let final_membership = fine_stats.membership.clone();

    if log::log_enabled!(log::Level::Info) {
        eprintln!();
        eprintln!("{}", link_community_histogram(&final_membership, k, 50));
        eprintln!();
    }

    //////////////////////////////////////////////////////
    // 9. Extract and write final outputs               //
    //////////////////////////////////////////////////////
    let draft_prefix = format!("{}.draft", c.out);
    info!(
        "Writing draft outputs (propensity, gene_community, link_community) → {}.*",
        draft_prefix
    );
    let (_draft_propensity, draft_gene_community) = write_partition_outputs(
        &draft_prefix,
        edges,
        &final_membership,
        n_cells,
        k,
        &cell_names,
        &data_vec,
        Some(&gene_weights),
        c.block_size,
    )?;

    info!("Writing score trace...");
    write_score_trace(&(c.out.to_string() + ".scores.parquet"), &score_trace)?;

    let mut merge_present_with_consensus = false;
    {
        use matrix_param::traits::Inference;
        info!(
            "Running cosine dictionary merge (average linkage) over K={} communities...",
            k
        );
        let merges = cosine_merge(draft_gene_community.posterior_log_mean());
        write_dict_merges(&(c.out.to_string() + ".dict_merges.parquet"), &merges)?;

        let labels = cosine_cut(&merges, k, args.merge_cut);
        let n_consensus = labels
            .iter()
            .filter(|&&v| v >= 0)
            .map(|&v| v as usize)
            .max()
            .map_or(0, |m| m + 1);
        info!(
            "Dictionary cut (cosine ≥ {:.3}): {} merged communities",
            args.merge_cut, n_consensus
        );
        write_dict_cut(&(c.out.to_string() + ".dict_merges.cut.parquet"), &labels)?;

        if n_consensus > 0 && n_consensus < k {
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
                "Writing final outputs (propensity, gene_community, link_community) → {}.*",
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
                Some(&gene_weights),
                c.block_size,
            )?;
            merge_present_with_consensus = true;
        } else {
            info!(
                "Dictionary merge produced no collapses at cosine ≥ {:.3}; draft outputs at {}.* are the final result",
                args.merge_cut, draft_prefix
            );
        }
    }

    //////////////////////////////////////////////////////
    // 10. Write metadata.json (information-flow root)  //
    //////////////////////////////////////////////////////
    {
        use crate::util::metadata::{create_lc_metadata, RunInputs};
        let coord_file_str = c.coord_files_joined();
        let meta = create_lc_metadata(
            &RunInputs {
                prefix: &c.out,
                data_files: &c.data_files,
                coord_file: coord_file_str.as_deref(),
                coord_columns: &coordinate_names,
                n_cells,
                n_genes: data_vec.num_rows(),
                n_edges: edges.len(),
                k,
            },
            merge_present_with_consensus,
            &cascade_level_indices,
        );
        let meta_path = std::path::PathBuf::from(format!("{}.metadata.json", c.out));
        meta.write(&meta_path)?;
        info!("Wrote {}", meta_path.display());
    }

    info!("Done");
    Ok(())
}

/// Build full-resolution edge profiles via the active profile mode.
fn build_full_profiles(
    data: &SparseIoVec,
    edge_indices: &[usize],
    all_edges: &[(usize, usize)],
    mode: &ProfileMode<'_>,
    block_size: Option<usize>,
) -> anyhow::Result<LinkProfileStore> {
    match *mode {
        ProfileMode::ModulePair {
            basis,
            module_expr,
            cell_totals,
        } => Ok(build_module_pair_profiles_for_edges(
            module_expr,
            cell_totals,
            all_edges,
            edge_indices,
            basis,
        )),
        ProfileMode::Projection { basis } => {
            build_projection_profiles_for_edges(data, edge_indices, all_edges, basis, block_size)
        }
    }
}
