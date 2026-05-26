use super::run_squeeze_if_needed;
use crate::hdf5_io::*;
use crate::sparse_io::*;
use crate::utilities::name_matching::{
    compose_id_name, filter_row_indices_by_type, make_names_unique,
};
use data_beans::zarr_io::*;

use clap::Args;
use log::info;
use matrix_util::common_io::*;

#[derive(Args, Debug)]
pub struct From10xMoleculeArgs {
    #[arg(
        help = "Input 10X molecule_info.h5 file",
        long_help = "Specify the molecule_info.h5 file from Cell Ranger count/multi.\n\
		     Contains per-molecule data: barcode_idx, feature_idx, count, gem_group, etc."
    )]
    pub h5_file: Box<str>,

    #[arg(
        long,
        value_enum,
        default_value = "zarr",
        help = "Backend format for output",
        long_help = "Choose the backend format for the output file."
    )]
    pub backend: SparseIoBackend,

    #[arg(
        short,
        long,
        help = "Output file header or name",
        long_help = "Specify the output file header.\n\
		     The zarr backend produces {output}.zarr.zip by default;\n\
		     pass --no-zip to keep a {output}.zarr directory instead."
    )]
    pub output: Box<str>,

    /// keep a `.zarr` directory instead of producing a `.zarr.zip` archive
    #[arg(long = "no-zip", default_value_t = true, action = clap::ArgAction::SetFalse)]
    pub zip: bool,

    #[arg(
        long,
        default_value = "Gene Expression",
        help = "Library type to include",
        long_help = "Filter molecules to only those from libraries of this type.\n\
		     Common types: 'Gene Expression', 'Antibody Capture', 'CRISPR Guide Capture'.\n\
		     Reads library_info JSON to determine which library indices match."
    )]
    pub library_type: Box<str>,

    #[arg(
        long,
        default_value = "",
        help = "Select row type (feature_type)",
        long_help = "Filter features by type. Rows are included if their type contains this value.\n\
		     Empty (default) keeps all features. 10X uses 'Gene Expression', 'Antibody Capture', etc."
    )]
    pub select_row_type: Box<str>,

    #[arg(
        long,
        default_value = "",
        help = "Remove row type",
        long_help = "Remove rows if their type contains this value. Empty (default) removes nothing."
    )]
    pub remove_row_type: Box<str>,

    #[arg(
        long,
        default_value_t = false,
        help = "Skip pass_filter and include all barcodes",
        long_help = "By default, only barcodes that passed Cell Ranger cell calling are included.\n\
		     Set this flag to include ALL barcodes with at least one molecule."
    )]
    pub no_pass_filter: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Squeeze sparse rows or columns",
        long_help = "Enable squeezing to remove rows and columns with too few non-zeros."
    )]
    pub do_squeeze: bool,

    #[arg(
        long,
        default_value_t = 1,
        help = "Row non-zero cutoff",
        long_help = "Minimum number of non-zero elements required for rows."
    )]
    pub row_nnz_cutoff: usize,

    #[arg(
        long,
        default_value_t = 1,
        help = "Column non-zero cutoff",
        long_help = "Minimum number of non-zero elements required for columns."
    )]
    pub column_nnz_cutoff: usize,

    #[arg(
        long,
        help = "Cells per rayon job for the post-build squeeze pass \
                (omit for auto-scaling by feature count)"
    )]
    pub block_size: Option<usize>,
}
pub fn run_build_from_10x_molecule(args: &From10xMoleculeArgs) -> anyhow::Result<()> {
    let file = hdf5::File::open(args.h5_file.to_string())?;
    info!("Opened molecule_info.h5: {}", args.h5_file);

    let effective_output = apply_zip_flag(&args.output, args.zip);
    let (backend, backend_file) =
        resolve_backend_file(&effective_output, Some(args.backend.clone()))?;

    if std::path::Path::new(backend_file.as_ref()).exists() {
        info!("Removing existing backend file: {}", &backend_file);
        remove_file(&backend_file)?;
    }

    // 1. Read per-molecule arrays. Keep ndarray `Array1` owners — don't
    // `.to_vec()` them, as that doubles peak memory on large molecule files.
    let barcode_idx = file.dataset("barcode_idx")?.read_1d::<u64>()?;
    let feature_idx = file.dataset("feature_idx")?.read_1d::<u32>()?;
    let count = file.dataset("count")?.read_1d::<u32>()?;
    let gem_group = file.dataset("gem_group")?.read_1d::<u16>()?;
    let library_idx = file.dataset("library_idx")?.read_1d::<u16>()?;
    let n_molecules = barcode_idx.len();
    info!("Read {} molecules", n_molecules);

    // 2. Read lookup tables
    let barcodes = read_hdf5_strings(file.dataset("barcodes")?)?;
    let feature_group = file.group("features")?;
    let mut row_ids: Vec<Box<str>> = read_hdf5_strings(feature_group.dataset("id")?)?;
    let mut row_names: Vec<Box<str>> = read_hdf5_strings(feature_group.dataset("name")?)?;
    let mut row_types: Vec<Box<str>> = read_hdf5_strings(feature_group.dataset("feature_type")?)?;

    let n_features = row_ids.len();
    info!("Read {} barcodes, {} features", barcodes.len(), n_features);

    // 3. Parse library_info and filter by library type
    let valid_libraries: rustc_hash::FxHashSet<u16> = {
        let lib_info_ds = file.dataset("library_info")?;
        let lib_info_raw = read_hdf5_strings(lib_info_ds)?;
        let lib_info_json: String = lib_info_raw.iter().map(|s| s.as_ref()).collect();
        let lib_entries: Vec<serde_json::Value> = serde_json::from_str(&lib_info_json)?;

        let mut valid = rustc_hash::FxHashSet::default();
        for entry in &lib_entries {
            if let (Some(lib_id), Some(lib_type)) = (
                entry.get("library_id").and_then(|v| v.as_u64()),
                entry.get("library_type").and_then(|v| v.as_str()),
            ) {
                if lib_type.contains(args.library_type.as_ref()) {
                    valid.insert(lib_id as u16);
                }
            }
        }
        info!(
            "Library type '{}': {} of {} libraries match",
            args.library_type,
            valid.len(),
            lib_entries.len()
        );
        valid
    };

    // 4. Read pass_filter if needed
    let valid_cells: Option<rustc_hash::FxHashSet<(u64, u16)>> = if !args.no_pass_filter {
        let pf = file.dataset("barcode_info/pass_filter")?.read_2d::<u64>()?;
        let mut cells = rustc_hash::FxHashSet::default();
        for row in pf.rows() {
            let bc_idx = row[0];
            let lib_idx = row[1] as u16;
            if valid_libraries.contains(&lib_idx) {
                cells.insert((bc_idx, lib_idx));
            }
        }
        info!("pass_filter: {} valid cells", cells.len());
        Some(cells)
    } else {
        info!("Skipping pass_filter (--no-pass-filter)");
        None
    };

    // 5. Filter molecules and aggregate into triplets
    //    Column key = (barcode_idx, gem_group)
    use rustc_hash::FxHashMap as HashMap;
    use std::collections::BTreeSet;

    let mut col_keys = BTreeSet::new();
    let mut triplet_map: HashMap<(u64, u64), f32> = Default::default();

    {
        // Borrow raw arrays as slices once; avoids repeated bounds-checked
        // indexing through `Array1::Index` in the hot loop.
        let barcode_idx_s = barcode_idx.as_slice().expect("barcode_idx not contiguous");
        let feature_idx_s = feature_idx.as_slice().expect("feature_idx not contiguous");
        let count_s = count.as_slice().expect("count not contiguous");
        let gem_group_s = gem_group.as_slice().expect("gem_group not contiguous");
        let library_idx_s = library_idx.as_slice().expect("library_idx not contiguous");

        for i in 0..n_molecules {
            // Filter by library type
            if !valid_libraries.contains(&library_idx_s[i]) {
                continue;
            }

            // Filter by pass_filter
            if let Some(ref cells) = valid_cells {
                if !cells.contains(&(barcode_idx_s[i], library_idx_s[i])) {
                    continue;
                }
            }

            let col_key = (barcode_idx_s[i], gem_group_s[i]);
            col_keys.insert(col_key);

            // Will remap column index after collecting all keys
            let row = feature_idx_s[i] as u64;
            *triplet_map
                .entry((row, barcode_idx_s[i] * 65536 + gem_group_s[i] as u64))
                .or_insert(0.0) += count_s[i] as f32;
        }
    }

    // Free the raw per-molecule arrays before we materialize the (huge)
    // triplets Vec. These can easily be multiple GB on 10X Aggr outputs.
    drop(barcode_idx);
    drop(feature_idx);
    drop(count);
    drop(gem_group);
    drop(library_idx);
    drop(valid_cells);

    // Build dense column index mapping
    let col_keys_vec: Vec<(u64, u16)> = col_keys.into_iter().collect();
    let col_key_to_idx: HashMap<(u64, u16), u64> = col_keys_vec
        .iter()
        .enumerate()
        .map(|(idx, &key)| (key, idx as u64))
        .collect();
    let ncols = col_keys_vec.len();

    // Build column names: SEQUENCE-GEMGROUP
    let column_names: Vec<Box<str>> = col_keys_vec
        .iter()
        .map(|&(bc_idx, gg)| {
            let bc = barcodes[bc_idx as usize].as_ref();
            format!("{}-{}", bc, gg).into_boxed_str()
        })
        .collect();

    info!("Aggregated into {} columns (cells)", ncols);

    // Convert triplet_map to proper triplets with remapped column indices
    let triplets: Vec<(u64, u64, f32)> = triplet_map
        .into_iter()
        .map(|((row, packed_col), val)| {
            let bc_idx = packed_col / 65536;
            let gg = (packed_col % 65536) as u16;
            let col = col_key_to_idx[&(bc_idx, gg)];
            (row, col, val)
        })
        .collect();

    let nrows = n_features;
    let nnz = triplets.len();
    info!("Built {} triplets in {} x {} matrix", nnz, nrows, ncols);

    // 6. Build backend
    let mut out = create_sparse_from_triplets_owned(
        triplets,
        (nrows, ncols, nnz),
        Some(&backend_file),
        Some(&backend),
    )?;
    info!("Created sparse matrix: {}", backend_file);

    // 7. Register names
    // Composite row names: id_name
    if nrows < row_ids.len() {
        row_ids.truncate(nrows);
    }
    if nrows < row_names.len() {
        row_names.truncate(nrows);
    }

    let mut row_id_names = compose_id_name(row_ids, row_names);
    make_names_unique(&mut row_id_names);

    out.register_row_names_vec(&row_id_names);
    out.register_column_names_vec(&column_names);

    // 8. Filter by feature type
    if nrows < row_types.len() {
        row_types.truncate(nrows);
    }

    let select_rows =
        filter_row_indices_by_type(&row_types, &args.select_row_type, &args.remove_row_type);

    if select_rows.len() < nrows {
        info!(
            "Filtering features: {} -> {} of '{}' type",
            nrows,
            select_rows.len(),
            args.select_row_type
        );
        out.subset_columns_rows(None, Some(&select_rows))?;
    }

    // 9. Squeeze if needed
    run_squeeze_if_needed(
        args.do_squeeze,
        args.row_nnz_cutoff,
        args.column_nnz_cutoff,
        args.block_size,
        &backend_file,
    )?;
    finalize_zarr_output(&backend_file, &effective_output)?;
    info!("done");
    Ok(())
}
