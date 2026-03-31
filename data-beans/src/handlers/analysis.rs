use crate::misc::*;
use crate::qc::*;
use crate::simulate;
use crate::sparse_io::*;
use crate::sparse_io_vector::*;

use log::info;
use matrix_util::common_io::*;
use matrix_util::membership::Membership;
use matrix_util::mtx_io;
use matrix_util::traits::IoOps;
use rayon::prelude::*;
use regex::Regex;
use std::sync::Arc;

// Import the argument structs from main.rs
use crate::simulate_multimodal;
use crate::{RunSimulateArgs, RunSimulateMultimodalArgs, RunStatArgs, SparseIoBackend, StatDim};

/// Compute statistics across sparse matrix data
///
/// This function computes row or column statistics across one or more sparse matrix
/// data files. Statistics can be stratified by column groups if a group membership
/// file is provided.
pub fn run_stat(cmd_args: &RunStatArgs) -> anyhow::Result<()> {
    let output = cmd_args.output.clone();
    dirname(&output).as_deref().map(mkdir).transpose()?;

    // to avoid duplicate barcodes in the column names
    let attach_data_name = cmd_args.data_files.len() > 1;

    let mut data = SparseIoVec::new();
    for data_file in cmd_args.data_files.iter() {
        let (backend, data_file) = resolve_backend_file(data_file, None)?;

        let mut this_data = open_sparse_matrix(&data_file, &backend)?;
        if cmd_args.preload {
            info!("Preloading data from {} ...", data_file);
            this_data.preload_columns()?;
        }
        let data_name = attach_data_name.then(|| basename(&data_file)).transpose()?;
        data.push(Arc::from(this_data), data_name)?;
    }

    match cmd_args.stat_dim {
        StatDim::Row => {
            if let Some(column_group_file) = &cmd_args.column_group_file {
                let cols = data.column_names()?;

                // Load membership and match to data columns
                // Use delimiter to extract base barcode for matching
                let membership = Membership::from_file(column_group_file, 0, 1, true)?
                    .with_delimiter(cmd_args.delimiter);
                let (column_membership, stats) = membership.match_keys(&cols);

                info!(
                    "Column matching: {} exact + {} base_key + {} prefix = {}/{} matched",
                    stats.exact,
                    stats.base_key,
                    stats.prefix,
                    stats.total_matched(),
                    stats.total()
                );

                if column_membership.is_empty() {
                    let data_sample: Vec<_> = cols.iter().take(3).collect();
                    let memb_sample = membership.sample_keys(3);
                    info!("Data columns sample: {:?}", data_sample);
                    info!("Membership keys sample: {:?}", memb_sample);
                }

                let unique_groups = membership.unique_groups();
                info!(
                    "Will collect stats for {} groups: {:?}",
                    unique_groups.len(),
                    unique_groups
                );

                let (group_names, group_stats) = collect_stratified_row_stat_across_vec(
                    &data,
                    &column_membership,
                    cmd_args.block_size,
                )?;

                info!(
                    "Collected {} group stats: {:?}",
                    group_names.len(),
                    group_names
                );

                if cmd_args.output.eq_ignore_ascii_case("stdout") {
                    for (g, row_stat) in group_names.iter().zip(group_stats.iter()) {
                        let out: Vec<Box<str>> = row_stat
                            .to_string_vec(&data.row_names()?, "\t")?
                            .into_iter()
                            .map(|s| format!("{}\t{}", g, s).into_boxed_str())
                            .collect();
                        write_lines(&out, &cmd_args.output)?;
                    }
                } else {
                    use matrix_util::sparse_stat::save_grouped_stats_parquet;

                    info!("writing out: {}", cmd_args.output);
                    save_grouped_stats_parquet(
                        &cmd_args.output,
                        &data.row_names()?,
                        &group_names,
                        &group_stats,
                    )?;
                }
            } else {
                let row_stat = collect_row_stat_across_vec(&data, cmd_args.block_size)?;
                row_stat.save(&cmd_args.output, &data.row_names()?, "\t")?;
            }
        }
        StatDim::Column => {
            let select_rows = cmd_args.row_name_pattern.as_ref().map(|pattern| {
                let re = Regex::new(&format!("(?i){}", pattern))
                    .expect("Invalid regex pattern for --row-name-pattern");
                let row_names = data.row_names().expect("couldn't get the row names");
                let selected: Vec<_> = row_names
                    .iter()
                    .enumerate()
                    .filter_map(|(i, name)| if re.is_match(name) { Some(i) } else { None })
                    .collect();
                info!(
                    "Row pattern '{}' matched {}/{} rows",
                    pattern,
                    selected.len(),
                    row_names.len()
                );
                selected
            });

            let col_stat =
                collect_column_stat_across_vec(&data, select_rows.as_deref(), cmd_args.block_size)?;

            col_stat.save(&cmd_args.output, &data.column_names()?, "\t")?;
        }
    };

    Ok(())
}

/// Simulate factored Poisson-Gamma data
///
/// This function generates simulated single-cell RNA-seq data using a factored
/// Poisson-Gamma model. The simulated data includes batch effects and topic
/// (cell type) structure. Output is saved in the specified backend format
/// (.zarr or .h5) along with parameter files.
pub fn run_simulate(cmd_args: &RunSimulateArgs) -> anyhow::Result<()> {
    let output = cmd_args.output.clone();
    mkdir(&output)?;

    let backend = cmd_args.backend.clone();

    let backend_file = match backend {
        SparseIoBackend::HDF5 => output.to_string() + ".h5",
        SparseIoBackend::Zarr => output.to_string() + ".zarr",
    };

    let mtx_file = output.to_string() + ".mtx.gz";
    let row_file = output.to_string() + ".rows.gz";
    let col_file = output.to_string() + ".cols.gz";

    let dict_file = mtx_file.replace(".mtx.gz", ".dict.parquet");
    let prop_file = mtx_file.replace(".mtx.gz", ".prop.parquet");
    let batch_memb_file = mtx_file.replace(".mtx.gz", ".batch.gz");
    let ln_batch_file = mtx_file.replace(".mtx.gz", ".ln_batch.parquet");

    remove_all_files(&vec![
        backend_file.clone().into_boxed_str(),
        mtx_file.clone().into_boxed_str(),
        dict_file.clone().into_boxed_str(),
        prop_file.clone().into_boxed_str(),
        batch_memb_file.clone().into_boxed_str(),
        ln_batch_file.clone().into_boxed_str(),
    ])
    .expect("failed to clean up existing output files");

    let sim_args = simulate::SimArgs {
        rows: cmd_args.rows,
        cols: cmd_args.cols,
        depth: cmd_args.depth,
        factors: cmd_args.factors,
        batches: cmd_args.batches,
        overdisp: cmd_args.overdisp,
        pve_topic: cmd_args.pve_topic,
        pve_batch: cmd_args.pve_batch,
        rseed: cmd_args.rseed,
        hierarchical_depth: cmd_args.hierarchical_depth,
        n_housekeeping: cmd_args.n_housekeeping,
        housekeeping_fold: cmd_args.housekeeping_fold,
        n_chromosomes: 0,
        cnv_events_per_chr: 0.5,
        cnv_block_frac: 0.15,
        cnv_gain_fold: 2.0,
        cnv_loss_fold: 0.5,
    };

    let sim = simulate::generate_factored_poisson_gamma_data(&sim_args)?;
    info!("successfully generated factored Poisson-Gamma data");

    let batch_out: Vec<Box<str>> = sim
        .batch_membership
        .iter()
        .map(|&x| Box::from(x.to_string()))
        .collect();

    write_lines(&batch_out, &batch_memb_file)?;
    info!("batch membership: {:?}", &batch_memb_file);

    let mtx_shape = (sim_args.rows, sim_args.cols, sim.triplets.len());

    let rows: Vec<Box<str>> = (0..cmd_args.rows)
        .map(|i| i.to_string().into_boxed_str())
        .collect();

    let cols: Vec<Box<str>> = (0..cmd_args.cols)
        .map(|i| i.to_string().into_boxed_str())
        .collect();

    sim.ln_delta_db
        .to_parquet_with_names(&ln_batch_file, (Some(&rows), Some("feature")), None)?;
    sim.theta_kn.transpose().to_parquet_with_names(
        &prop_file,
        (Some(&cols), Some("cell")),
        None,
    )?;
    sim.beta_dk
        .to_parquet_with_names(&dict_file, (Some(&rows), Some("feature")), None)?;

    if let Some(ref node_probs) = sim.hierarchy_node_probs {
        let hierarchy_file = mtx_file.replace(".mtx.gz", ".hierarchy.parquet");
        node_probs.to_parquet_with_names(&hierarchy_file, (Some(&rows), Some("feature")), None)?;
        info!("wrote hierarchy node probabilities: {:?}", &hierarchy_file);
    }

    info!(
        "wrote parameter files:\n{:?},\n{:?},\n{:?}",
        &ln_batch_file, &dict_file, &prop_file
    );

    if cmd_args.save_mtx {
        let mut triplets = sim.triplets.clone();
        triplets.par_sort_by_key(|&(row, _, _)| row);
        triplets.par_sort_by_key(|&(_, col, _)| col);

        mtx_io::write_mtx_triplets(&triplets, sim_args.rows, sim_args.cols, &mtx_file)?;
        write_lines(&rows, &row_file)?;
        write_lines(&cols, &col_file)?;

        info!(
            "save mtx, row, and column files:\n{}\n{}\n{}",
            mtx_file, row_file, col_file
        );
    }

    info!("registering triplets ...");

    let mut data = create_sparse_from_triplets(
        &sim.triplets,
        mtx_shape,
        Some(&backend_file),
        Some(&backend),
    )?;

    info!("created sparse matrix: {}", backend_file);

    data.register_row_names_vec(&rows);
    data.register_column_names_vec(&cols);

    info!("done");
    Ok(())
}

/// Run multimodal simulation with shared base + delta dictionaries.
pub fn run_simulate_multimodal(cmd_args: &RunSimulateMultimodalArgs) -> anyhow::Result<()> {
    let output = cmd_args.output.clone();
    mkdir(&output)?;

    let sim_args = simulate_multimodal::MultimodalSimArgs {
        rows: cmd_args.rows,
        cols: cmd_args.cols,
        depth_per_modality: cmd_args.depth.clone(),
        factors: cmd_args.factors,
        batches: cmd_args.batches,
        base_scale: cmd_args.base_scale,
        delta_scale: cmd_args.delta_scale,
        n_delta_features: cmd_args.n_delta_features,
        pve_topic: cmd_args.pve_topic,
        pve_batch: cmd_args.pve_batch,
        rseed: cmd_args.rseed,
        shared_batch_effects: cmd_args.shared_batch_effects,
        hierarchical_depth: cmd_args.hierarchical_depth,
        overdisp: cmd_args.overdisp,
        n_housekeeping: cmd_args.n_housekeeping,
        housekeeping_fold: cmd_args.housekeeping_fold,
    };

    let mm = sim_args.depth_per_modality.len();
    let sim = simulate_multimodal::generate_multimodal_data(&sim_args)?;
    info!("generated multimodal data: {} modalities", mm);

    let rows: Vec<Box<str>> = (0..cmd_args.rows)
        .map(|i| i.to_string().into_boxed_str())
        .collect();
    let cols: Vec<Box<str>> = (0..cmd_args.cols)
        .map(|i| i.to_string().into_boxed_str())
        .collect();

    // Shared outputs
    let prop_file = format!("{}.prop.parquet", output);
    sim.theta_kn.transpose().to_parquet_with_names(
        &prop_file,
        (Some(&cols), Some("cell")),
        None,
    )?;

    let batch_file = format!("{}.batch.gz", output);
    let batch_out: Vec<Box<str>> = sim
        .batch_membership
        .iter()
        .map(|&x| Box::from(x.to_string()))
        .collect();
    write_lines(&batch_out, &batch_file)?;

    // W_base
    let base_file = format!("{}.w_base.parquet", output);
    sim.w_base_kd
        .to_parquet_with_names(&base_file, (None, None), None)?;

    // Per-modality outputs
    let backend = cmd_args.backend.clone();
    for m in 0..mm {
        let suffix = format!(".m{}", m);
        let backend_file = match backend {
            SparseIoBackend::HDF5 => format!("{}{}.h5", output, suffix),
            SparseIoBackend::Zarr => format!("{}{}.zarr", output, suffix),
        };

        let mtx_shape = (cmd_args.rows, cmd_args.cols, sim.triplets[m].len());

        // Dictionary
        let dict_file = format!("{}{}.dict.parquet", output, suffix);
        sim.beta_dk[m].to_parquet_with_names(&dict_file, (Some(&rows), Some("feature")), None)?;

        // Batch effects
        let ln_batch_file = format!("{}{}.ln_batch.parquet", output, suffix);
        sim.ln_delta_db[m].to_parquet_with_names(
            &ln_batch_file,
            (Some(&rows), Some("feature")),
            None,
        )?;

        // Delta (non-reference only)
        if m > 0 {
            let delta_file = format!("{}.w_delta{}.parquet", output, suffix);
            sim.w_delta_kd[m - 1].to_parquet_with_names(&delta_file, (None, None), None)?;

            let mask_file = format!("{}.spike_mask{}.parquet", output, suffix);
            sim.spike_mask_kd[m - 1].to_parquet_with_names(&mask_file, (None, None), None)?;
        }

        // MTX
        if cmd_args.save_mtx {
            let mtx_file = format!("{}{}.mtx.gz", output, suffix);
            let mut triplets = sim.triplets[m].clone();
            triplets.par_sort_by_key(|&(row, col, _)| (col, row));
            mtx_io::write_mtx_triplets(&triplets, cmd_args.rows, cmd_args.cols, &mtx_file)?;
        }

        // Sparse backend
        let mut data = create_sparse_from_triplets(
            &sim.triplets[m],
            mtx_shape,
            Some(&backend_file),
            Some(&backend),
        )?;

        data.register_row_names_vec(&rows);
        data.register_column_names_vec(&cols);
        info!("modality {}: {}", m, backend_file);
    }

    info!("done");
    Ok(())
}
