//! The `senna simba` driver: load, cell QC, HVG subset, train, write.

use super::SimbaArgs;
use crate::embed_common::*;
use crate::run_manifest::{record_train_args, write_run_manifest, RunDescription, RunKind};
use candle_util::candle_core::{Device, Tensor};
use data_beans_alg::hvg::select_hvg_streaming;
use graph_embedding_util as ge;
use graph_embedding_util::simba::{
    compare_entities, run_simba, EntityMetrics, SimbaConfig, METRICS_T, N_TOP_CELLS,
};
use log::info;
use matrix_util::common_io::mkdir_parent;

pub fn fit_simba(args: &SimbaArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;
    anyhow::ensure!(
        !args.data_files.is_empty(),
        "no input files: pass one or more zarr/h5 count matrices"
    );
    let unified = ge::load_unified_data(ge::LoadUnifiedArgs {
        data_files: args.data_files.clone(),
        batch_files: args.batch_files.clone(),
        feature_kind: Some(ge::FeatureNameKind::Gene { delim: '_' }),
        preload: args.preload_data,
        column_alignment: data_beans::sparse_io_vector::ColumnAlignment::Disjoint,
        per_file_feature_suffix: None,
        ..Default::default()
    })?;
    let data = unified.count_backend();
    let n_cells = unified.n_cells();

    ////////////////////////////
    // Cell QC (output filter) //
    ////////////////////////////
    // As in bge: every cell is a node of the graph and trains; QC only
    // decides which cells reach the per-cell outputs (and serve as the
    // reference set the genes are co-embedded onto).
    let qc_keep_idx: Option<Vec<usize>> = match args.qc.to_config() {
        Some(cfg) => {
            if cfg.feature_min_cells > 0 {
                log::warn!(
                    "--qc-feature-min-cells is ignored by simba (cell-only QC; the gene set is \
                     the HVG selection)"
                );
            }
            let report =
                data_beans::qc_lib::compute_qc_exempting(data, &cfg, args.block_size, None)?;
            let keep = report.emit_idx_unmasked();
            info!(
                "QC: {} / {} cells kept for output ({} near-empty, {} MAD-outlier dropped)",
                keep.len(),
                n_cells,
                report.near_empty.iter().filter(|&&e| e).count(),
                report.n_cells_dropped,
            );
            Some(keep)
        }
        None => None,
    };

    ////////////////////////////////
    // HVG: the embedded gene set //
    ////////////////////////////////
    // SIMBA builds its graph over the highly variable genes only
    // (`use_highly_variable=True`), so unlike bge the selection hard-subsets
    // the trained axis. `--n-hvg 0` embeds every gene.
    let effective = crate::hvg::resolve_multiome_with_hvg(false, args.data_files.len(), &args.hvg);
    let must_train =
        crate::hvg::load_must_train(effective.must_train_file, effective.selection_on())?;
    let (hvg_rows, gene_names): (Vec<usize>, Vec<Box<str>>) = if effective.selection_on() {
        let sel = select_hvg_streaming(
            data,
            (effective.n_hvg > 0).then_some(effective.n_hvg),
            effective.feature_list_file,
            must_train.as_ref(),
            args.block_size,
        )?;
        anyhow::ensure!(
            sel.selected_indices.len() == sel.selected_names.len(),
            "simba: HVG indices and names disagree"
        );
        info!(
            "simba: embedding {} selected genes of {}",
            sel.selected_indices.len(),
            data.num_rows()
        );
        (sel.selected_indices, sel.selected_names)
    } else {
        info!("simba: embedding all {} genes", data.num_rows());
        ((0..data.num_rows()).collect(), data.row_names()?)
    };

    //////////////
    // Training //
    //////////////
    let cfg = SimbaConfig {
        dim: args.embedding_dim,
        epochs: args.epochs,
        lr: args.learning_rate,
        batch_size: args.batch_size,
        num_batch_negs: args.num_batch_negs,
        num_uniform_negs: args.num_uniform_negs,
        wd: args.weight_decay,
        wd_interval: args.wd_interval,
        eval_fraction: args.eval_fraction,
        n_bins: args.n_bins,
        coembed_t: args.coembed_temp,
        seed: args.seed,
        device: args.device.to_device(args.device_no)?,
    };
    let out = run_simba(data, &hvg_rows, &cfg)?;

    /////////////
    // Outputs //
    /////////////
    let cpu = Device::Cpu;
    let (e_cell, cell_names): (Tensor, Vec<Box<str>>) = match &qc_keep_idx {
        Some(keep) => {
            let idx: Vec<u32> = keep.iter().map(|&i| i as u32).collect();
            let idx = Tensor::from_vec(idx, keep.len(), &cpu)?;
            (
                out.e_cell.index_select(&idx, 0)?,
                keep.iter().map(|&i| unified.barcodes[i].clone()).collect(),
            )
        }
        None => (out.e_cell.clone(), unified.barcodes.clone()),
    };
    anyhow::ensure!(!cell_names.is_empty(), "simba: cell QC kept no cells");
    let prefix = args.out.as_ref();
    ge::save_embedding(
        &format!("{prefix}.cell_embedding.parquet"),
        &e_cell,
        &cell_names,
        "cell",
    )?;
    // Row axis `gene`, as bge labels its raw gene table.
    ge::save_embedding(
        &format!("{prefix}.feature_loading.parquet"),
        &out.e_gene,
        &gene_names,
        "gene",
    )?;
    // SIMBA's `si.tl.embed`: genes onto the (kept) cells at a fixed T.
    let coembed = ge::feature_coembedding_fixed_t(&e_cell, &out.e_gene, cfg.coembed_t)?;
    ge::save_embedding(
        &format!("{prefix}.feature_embedding.parquet"),
        &coembed,
        &gene_names,
        "feature",
    )?;
    info!(
        "Feature co-embedding (SIMBA, T={}) → {prefix}.feature_embedding.parquet",
        cfg.coembed_t
    );

    // SIMBA's `compare_entities` marker metrics.
    let metrics = compare_entities(&e_cell, &out.e_gene, N_TOP_CELLS, METRICS_T)?;
    let score_cols: Vec<Box<str>> = EntityMetrics::COLUMNS
        .iter()
        .map(|c| Box::<str>::from(*c))
        .collect();
    metrics.to_tensor()?.to_parquet_with_names(
        &format!("{prefix}.feature_scores.parquet"),
        (Some(&gene_names), Some("feature")),
        Some(&score_cols),
    )?;

    // The expression levels: one row per relation.
    let n_lv = out.relations.len();
    let mut rows = Vec::with_capacity(5 * n_lv);
    let mut level_names = Vec::with_capacity(n_lv);
    for (r, &level) in out.relations.levels.iter().enumerate() {
        let edges = &out.discretization.bin_edges;
        rows.extend([
            f32::from(level),
            edges[usize::from(level) - 1] as f32,
            edges[usize::from(level)] as f32,
            out.relations.weights[r],
            out.level_counts[r] as f32,
        ]);
        level_names.push(format!("L{level}").into_boxed_str());
    }
    let bin_cols: Vec<Box<str>> = ["level", "lower", "upper", "weight", "n_edges"]
        .iter()
        .map(|c| Box::<str>::from(*c))
        .collect();
    Tensor::from_vec(rows, (n_lv, 5), &cpu)?.to_parquet_with_names(
        &format!("{prefix}.simba_bins.parquet"),
        (Some(&level_names), Some("relation")),
        Some(&bin_cols),
    )?;

    let input: Vec<String> = args.data_files.iter().map(ToString::to_string).collect();
    let batch: Vec<String> = args
        .batch_files
        .as_ref()
        .map(|v| v.iter().map(ToString::to_string).collect())
        .unwrap_or_default();
    write_run_manifest(&RunDescription {
        train_args: Some(record_train_args(args)?),
        kind: RunKind::Simba,
        prefix,
        data_input: &input,
        data_batch: &batch,
        data_input_null: &[],
        dictionary_suffix: None,
        has_model: false,
        has_cell_proj: false,
        pb_gene_suffix: None,
        pb_latent_suffix: None,
        pb_reference_suffix: None,
        dictionary_empirical_suffix: None,
        feature_embedding_suffix: Some("feature_embedding.parquet"),
        feature_loading_suffix: Some("feature_loading.parquet"),
        module_membership_suffix: None,
        module_dictionary_suffix: None,
        softmax_dictionary_suffix: None,
        cell_embedding_suffix: Some("cell_embedding.parquet"),
        default_colour_by: "cluster",
        has_latent: false,
        velocity_suffix: None,
        velocity_factor_suffix: None,
        delta_feature_embedding_suffix: None,
        has_cell_to_pb: false,
    })?;
    info!(
        "simba: {} edges over {} cells × {} genes (per level {:?}); wd {}; final train loss {:.4}/edge → {prefix}.*",
        out.n_edges,
        n_cells,
        gene_names.len(),
        out.level_counts,
        out.wd,
        out.epochs.last().map_or(f64::NAN, |e| e.train_loss)
    );
    Ok(())
}
