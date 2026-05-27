//! Entry point for `faba rna-mod-embed` (alias `rmodem`).

use anyhow::Context;
use data_beans::sparse_io_vector::ColumnAlignment;
use graph_embedding_util::{load_unified_data, FeatureNameKind};
use log::info;
use matrix_util::common_io::mkdir_parent;

use crate::rna_mod_embed::args::RnaModEmbedArgs;
use crate::rna_mod_embed::feature_table::FeatureTable;
use crate::rna_mod_embed::manifest::write_outputs;
use crate::rna_mod_embed::model::RnaModEmbedModel;
use crate::rna_mod_embed::pseudobulk::build_pseudobulk;
use crate::rna_mod_embed::train::train;

pub fn run_rna_mod_embed(args: &RnaModEmbedArgs) -> anyhow::Result<()> {
    mkdir_parent(&args.out)?;

    validate_args(args)?;

    // Collect modality files in faba's conventional order: count first,
    // then m6A, A2I, pA. Order seeds modality_id assignment in the
    // FeatureTable (slot 0 is forced to "count" regardless).
    let mut data_files: Vec<Box<str>> = Vec::new();
    data_files.push(args.genes.clone());
    if let Some(p) = args.dartseq.as_ref() {
        data_files.push(p.clone());
    }
    if let Some(p) = args.atoi.as_ref() {
        data_files.push(p.clone());
    }
    if let Some(p) = args.apa.as_ref() {
        data_files.push(p.clone());
    }

    let batch_files: Option<&[Box<str>]> = if args.ignore_batch {
        if args.batch_files.is_some() {
            info!("--ignore-batch: dropping batch labels; treating all cells as one batch");
        }
        None
    } else {
        args.batch_files.as_deref()
    };

    info!(
        "rmodem: loading {} modality file(s) under Union column alignment",
        data_files.len()
    );
    let mut unified = load_unified_data(
        &data_files,
        batch_files,
        FeatureNameKind::Exact,
        /* preload = */ false,
        ColumnAlignment::Union,
    )
    .context("rmodem: load_unified_data")?;

    // `--ignore-batch` semantics: under Union with multiple modality
    // files, `load_unified_data` defaults to auto_modality_batch (one
    // batch per cell-modality-presence pattern). That contradicts
    // --ignore-batch's "one batch" promise — explicitly collapse here.
    if args.ignore_batch {
        let n = unified.n_cells();
        unified.batch_membership = vec![0_u32; n];
        unified.batch_names = vec!["all".into()];
    }

    info!(
        "rmodem: unified {} features × {} cells × {} batches",
        unified.n_features(),
        unified.n_cells(),
        unified.n_batches()
    );

    let table = FeatureTable::build(&unified.feature_names);
    info!(
        "rmodem: feature_table: {} genes, {} modalities, {} count-comp + {} modifier-comp rows",
        table.n_genes(),
        table.n_modalities(),
        table.count_comp_rows.len(),
        table.modifier_comp_rows.len(),
    );
    anyhow::ensure!(
        table.n_genes() > 0,
        "rmodem: no genes parsed from feature axis — check row name convention (`gene/modality/detail`)"
    );

    let n_cells = unified.n_cells();
    let pb = build_pseudobulk(&mut unified, &table, args).context("rmodem: build pseudobulk")?;

    let n_pbs_per_level: Vec<usize> = pb.pb_pools_per_level.iter().map(|l| l.n_units).collect();

    let dev = candle_core::Device::Cpu;
    let mut model = RnaModEmbedModel::new(
        table.n_genes(),
        table.n_modalities(),
        args.n_programs,
        args.embedding_dim,
        n_cells,
        &n_pbs_per_level,
        &dev,
    )
    .context("rmodem: init model")?;

    train(args, &table, &pb, &mut model).context("rmodem: training loop")?;

    write_outputs(&args.out, &table, &pb, &model, &unified).context("rmodem: write outputs")?;

    info!("rmodem: done — prefix '{}'", args.out);
    Ok(())
}

/// Argument-level sanity checks before any I/O or training. Surfaces
/// configuration mistakes (zero-dim model, NaN/out-of-range tempering,
/// stratum-fraction overflow) as a clear `anyhow::Error` rather than a
/// candle panic or silently zero-loss training.
fn validate_args(args: &RnaModEmbedArgs) -> anyhow::Result<()> {
    anyhow::ensure!(
        args.embedding_dim > 0,
        "--embedding-dim must be > 0 (got {})",
        args.embedding_dim
    );
    anyhow::ensure!(
        args.n_programs > 0,
        "--num-programs must be > 0 (got {})",
        args.n_programs
    );
    anyhow::ensure!(
        args.tau.is_finite() && (0.0..=1.0).contains(&args.tau),
        "--tau must be a finite value in [0, 1] (got {})",
        args.tau
    );
    anyhow::ensure!(
        args.tau_modality.is_finite() && (0.0..=1.0).contains(&args.tau_modality),
        "--tau-modality must be a finite value in [0, 1] (got {})",
        args.tau_modality
    );
    anyhow::ensure!(
        args.f_agg.is_finite() && (0.0..=1.0).contains(&args.f_agg),
        "--f-agg must be in [0, 1] (got {})",
        args.f_agg
    );
    anyhow::ensure!(
        args.f_count.is_finite() && (0.0..=1.0).contains(&args.f_count),
        "--f-count must be in [0, 1] (got {})",
        args.f_count
    );
    anyhow::ensure!(
        args.f_agg + args.f_count <= 1.0,
        "--f-agg + --f-count must be ≤ 1.0 (got {} + {} = {}); modifier stratum gets the remainder",
        args.f_agg,
        args.f_count,
        args.f_agg + args.f_count
    );
    // At least one negative-source must be active or training is a no-op:
    // log_softmax over a single positive column is identically zero.
    anyhow::ensure!(
        args.n_rand + args.n_swap_z + args.n_swap_q > 0,
        "at least one of --n-rand, --n-swap-z, --n-swap-q must be > 0 \
         (else NCE loss collapses to zero and AdamW makes no progress)"
    );
    Ok(())
}
