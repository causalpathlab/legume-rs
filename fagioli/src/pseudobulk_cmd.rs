use std::sync::Arc;

use clap::Args;
use log::info;
use nalgebra::DMatrix;

use data_beans::convert::try_open_or_convert;
use data_beans::sparse_io_vector::SparseIoVec;
use fagioli::mapping::pseudobulk::{
    build_onehot_membership, collapse_pseudobulk, read_cell_annotations,
    read_membership_proportions, CellAnnotations, Membership,
};
use matrix_param::io::ParamIo;
use matrix_util::common_io::basename;

#[derive(Args, Debug, Clone)]
pub struct PseudobulkArgs {
    /// Single-cell count matrices (Zarr, HDF5, or mtx paths; multiple files supported)
    #[arg(long, num_args = 1..)]
    pub sc_backend_files: Vec<Box<str>>,

    /// Cell annotations file (TSV or TSV.GZ): cell_id, individual_id[, cell_type]
    #[arg(long)]
    pub cell_annotations: Option<Box<str>>,

    /// Soft membership proportions from parquet (alternative to hard cell-type column)
    #[arg(long)]
    pub membership_parquet: Option<Box<str>>,

    /// Gamma prior shape parameter
    #[arg(long, default_value = "1.0")]
    pub gamma_a0: f32,

    /// Gamma prior rate parameter
    #[arg(long, default_value = "1.0")]
    pub gamma_b0: f32,

    /// Output prefix
    #[arg(short, long)]
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
        info!(
            "No cell annotations provided; inferring individuals from cell names (barcode@indiv)"
        );
        let mut individual_to_idx: std::collections::HashMap<Box<str>, usize> =
            std::collections::HashMap::new();
        let mut individual_ids: Vec<Box<str>> = Vec::new();
        let mut cell_to_individual = std::collections::HashMap::new();

        for cell_name in &column_names {
            let indiv: Box<str> = if let Some(pos) = cell_name.rfind('@') {
                Box::from(&cell_name[pos + 1..])
            } else {
                Box::from("all")
            };
            let idx = *individual_to_idx.entry(indiv.clone()).or_insert_with(|| {
                let i = individual_ids.len();
                individual_ids.push(indiv);
                i
            });
            cell_to_individual.insert(cell_name.clone(), idx);
        }

        info!(
            "Inferred {} individuals from cell names",
            individual_ids.len()
        );
        CellAnnotations {
            cell_to_individual,
            individual_ids,
        }
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
