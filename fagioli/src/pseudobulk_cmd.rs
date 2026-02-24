use std::sync::Arc;

use clap::Args;
use log::info;
use nalgebra::DMatrix;

use data_beans::convert::try_open_or_convert;
use data_beans::sparse_io_vector::SparseIoVec;
use fagioli::io::cell_annotations::{
    build_onehot_membership, infer_cell_annotations, read_cell_annotations,
    read_membership_proportions,
};
use fagioli::mapping::pseudobulk::{collapse_pseudobulk, Membership};
use matrix_param::io::ParamIo;
use matrix_util::common_io::basename;

#[derive(Args, Debug, Clone)]
pub struct PseudobulkArgs {
    // ── Input ────────────────────────────────────────────────────────────
    #[arg(long, num_args = 1.., help = "Single-cell count matrices (Zarr, HDF5, or mtx; multiple supported)")]
    pub sc_backend_files: Vec<Box<str>>,

    #[arg(
        long,
        help = "Cell annotations TSV: cell_id, individual_id[, cell_type]",
        long_help = "Cell annotations file (TSV or TSV.GZ).\n\
            Columns: cell_id, individual_id, and optionally cell_type.\n\
            If cell_type column is present, hard cell-type assignments are used.\n\
            Use --membership-parquet for soft assignments instead."
    )]
    pub cell_annotations: Option<Box<str>>,

    #[arg(
        long,
        help = "Soft membership proportions (parquet, alternative to cell_type column)"
    )]
    pub membership_parquet: Option<Box<str>>,

    // ── Pseudobulk parameters ────────────────────────────────────────────
    #[arg(long, default_value = "1.0", help = "Gamma prior shape (a0)")]
    pub gamma_a0: f32,

    #[arg(long, default_value = "1.0", help = "Gamma prior rate (b0)")]
    pub gamma_b0: f32,

    // ── Output ───────────────────────────────────────────────────────────
    #[arg(
        short,
        long,
        help = "Output prefix (writes {prefix}.{cell_type}.parquet)"
    )]
    pub output: Box<str>,
}

pub fn pseudobulk(args: &PseudobulkArgs) -> anyhow::Result<()> {
    // 1. Open SC backend(s)
    let attach_data_name = args.sc_backend_files.len() > 1;
    let mut data_vec = SparseIoVec::new();
    for data_file in &args.sc_backend_files {
        info!("Importing data file: {}", data_file);
        let data = try_open_or_convert(data_file)?;
        let data_name = attach_data_name.then(|| basename(data_file)).transpose()?;
        data_vec.push(Arc::from(data), data_name)?;
    }

    let column_names = data_vec.column_names()?;

    // 2. Read cell annotations (or default: all cells → one individual)
    let annotations = if let Some(path) = &args.cell_annotations {
        info!("Reading cell annotations from {}", path);
        read_cell_annotations(path)?
    } else {
        infer_cell_annotations(&column_names)
    };

    // 3. Build membership (soft from parquet, hard from annotations, or default: one cell type)
    let membership = if let Some(parquet_path) = &args.membership_parquet {
        info!("Reading soft membership from {}", parquet_path);
        read_membership_proportions(parquet_path, &column_names)?
    } else if let Some(ann_path) = &args.cell_annotations {
        info!("Building one-hot membership from {}", ann_path);
        build_onehot_membership(ann_path, &column_names)?
    } else {
        info!("No membership provided; treating all cells as one cell type");
        Membership {
            matrix: DMatrix::from_element(column_names.len(), 1, 1.0),
            cell_type_names: vec![Box::from("all")],
        }
    };

    // 4. Collapse pseudobulk
    info!(
        "Collapsing pseudobulk with Poisson-Gamma model (a0={}, b0={})...",
        args.gamma_a0, args.gamma_b0
    );
    let collapsed = collapse_pseudobulk(
        data_vec,
        &annotations,
        &membership,
        args.gamma_a0,
        args.gamma_b0,
    )?;

    // 5. Write output per cell type
    for (ct_idx, ct_name) in collapsed.cell_type_names.iter().enumerate() {
        let out_path = format!("{}.{}.parquet", args.output, ct_name);
        info!("Writing {} to {}", ct_name, out_path);

        collapsed.gamma_params[ct_idx].to_parquet_with_names(
            &out_path,
            (Some(&collapsed.gene_names), Some("gene")),
            Some(&collapsed.individual_ids),
        )?;
    }

    info!(
        "Pseudobulk collapse complete: {} cell types, {} individuals, {} genes",
        collapsed.cell_type_names.len(),
        collapsed.individual_ids.len(),
        collapsed.gene_names.len(),
    );

    Ok(())
}
