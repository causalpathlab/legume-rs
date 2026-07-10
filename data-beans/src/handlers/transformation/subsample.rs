use crate::hdf5_io::*;
use crate::sparse_io::*;
use crate::zarr_io::{finalize_output, prepare_output};

use clap::Args;
use log::info;
use rand::rngs::SmallRng;
use rand::seq::index::sample;
use rand::SeedableRng;
use rustc_hash::FxHashMap as HashMap;

#[derive(Args, Debug)]
pub struct SubsampleArgs {
    /// data file -- `.zarr`, `.zarr.zip`, or `.h5`
    pub data_file: Box<str>,

    /// number of cells (columns) to keep at random
    #[arg(long)]
    pub cells: Option<usize>,

    /// fraction of cells to keep at random, in (0, 1]; ignored when --cells is set
    #[arg(long)]
    pub cell_frac: Option<f64>,

    /// number of genes/features (rows) to keep at random
    #[arg(long)]
    pub genes: Option<usize>,

    /// fraction of genes to keep at random, in (0, 1]; ignored when --genes is set
    #[arg(long)]
    pub gene_frac: Option<f64>,

    /// RNG seed for reproducible sampling
    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    /// backend for the output file
    #[arg(long, value_enum, default_value = "zarr")]
    pub backend: SparseIoBackend,

    /// output header: {output}.zarr.zip by default; pass --no-zip to keep a {output}.zarr directory
    #[arg(short, long)]
    pub output: Box<str>,

    /// keep a `.zarr` directory instead of producing a `.zarr.zip` archive
    #[arg(long = "no-zip", default_value_t = true, action = clap::ArgAction::SetFalse)]
    pub zip: bool,
}

/// Randomly subsample cells and/or genes into a new, smaller backend.
///
/// Only the selected columns are read, so the cost scales with the *output*
/// size rather than the input — handy for carving quick test/demo datasets
/// out of a large backend. An unset dimension keeps all of its entries.
pub fn run_subsample(args: &SubsampleArgs) -> anyhow::Result<()> {
    let (backend_in, file_in) = resolve_backend_file(&args.data_file, None)?;
    let data = open_sparse_matrix(&file_in, &backend_in)?;

    let nrow = data
        .num_rows()
        .ok_or_else(|| anyhow::anyhow!("backend has no `nrow`"))?;
    let ncol = data
        .num_columns()
        .ok_or_else(|| anyhow::anyhow!("backend has no `ncol`"))?;

    let n_cells = resolve_target(args.cells, args.cell_frac, ncol, "cells")?;
    let n_genes = resolve_target(args.genes, args.gene_frac, nrow, "genes")?;
    if n_cells.is_none() && n_genes.is_none() {
        anyhow::bail!("specify at least one of --cells / --cell-frac / --genes / --gene-frac");
    }

    let mut rng = SmallRng::seed_from_u64(args.seed);

    // Sample ascending index sets so the output preserves the original order.
    let cell_idx = sample_sorted(&mut rng, ncol, n_cells);
    let gene_idx = sample_sorted(&mut rng, nrow, n_genes);

    info!(
        "subsampling to {} genes x {} cells (seed {})",
        gene_idx.len(),
        cell_idx.len(),
        args.seed
    );

    // Read only the selected columns; rows come back as global indices.
    let (_, _, raw) = data.read_triplets_by_columns(cell_idx.clone())?;

    let subset_genes = gene_idx.len() < nrow;
    let triplets: Vec<(u64, u64, f32)> = if subset_genes {
        let old2new: HashMap<u64, u64> = gene_idx
            .iter()
            .enumerate()
            .map(|(new, &old)| (old as u64, new as u64))
            .collect();
        raw.into_iter()
            .filter_map(|(r, c, v)| old2new.get(&r).map(|&nr| (nr, c, v)))
            .collect()
    } else {
        raw
    };

    let out_nrow = gene_idx.len();
    let out_ncol = cell_idx.len();
    let nnz = triplets.len();

    let row_names_all = data.row_names()?;
    let col_names_all = data.column_names()?;
    let out_row_names: Vec<Box<str>> = gene_idx.iter().map(|&i| row_names_all[i].clone()).collect();
    let out_col_names: Vec<Box<str>> = cell_idx.iter().map(|&i| col_names_all[i].clone()).collect();

    let (effective_output, backend_out, file_out) =
        prepare_output(&args.output, args.backend.clone(), args.zip)?;

    let mut out = create_sparse_from_triplets_owned(
        triplets,
        (out_nrow, out_ncol, nnz),
        Some(file_out.as_ref()),
        Some(&backend_out),
    )?;
    out.register_row_names_vec(&out_row_names);
    out.register_column_names_vec(&out_col_names);
    drop(out);

    let final_path = finalize_output(&file_out, &effective_output)?;
    info!(
        "done: {} ({} genes x {} cells, {} non-zeros)",
        final_path, out_nrow, out_ncol, nnz
    );
    Ok(())
}

/// Resolve a target keep-count from an explicit count or a fraction. `None`
/// means "keep all". A count is capped at `total`; a fraction rounds to at
/// least one.
fn resolve_target(
    count: Option<usize>,
    frac: Option<f64>,
    total: usize,
    label: &str,
) -> anyhow::Result<Option<usize>> {
    if let Some(k) = count {
        if k == 0 {
            anyhow::bail!("--{} must be >= 1", label);
        }
        Ok(Some(k.min(total)))
    } else if let Some(f) = frac {
        if !(f > 0.0 && f <= 1.0) {
            anyhow::bail!("--{}-frac must be in (0, 1]", label);
        }
        Ok(Some(((f * total as f64).round() as usize).clamp(1, total)))
    } else {
        Ok(None)
    }
}

/// Draw `target` distinct indices from `0..total` (or all of them when
/// `target` is `None` or `>= total`), returned in ascending order.
fn sample_sorted(rng: &mut SmallRng, total: usize, target: Option<usize>) -> Vec<usize> {
    match target {
        Some(k) if k < total => {
            let mut v = sample(rng, total, k).into_vec();
            v.sort_unstable();
            v
        }
        _ => (0..total).collect(),
    }
}
