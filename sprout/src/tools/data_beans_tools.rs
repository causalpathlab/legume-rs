use crate::tools::executor::ParamExtractor;
use crate::tools::registry::{ParameterType, ToolDefinition, ToolParameter, ToolRegistry};

/// Register all data-beans tools in the registry
pub fn register_data_beans_tools(registry: &mut ToolRegistry) {
    // Matrix info tool
    registry.register(ToolDefinition {
        name: "show_matrix_info".to_string(),
        description: "Display basic information about a sparse matrix file including dimensions and non-zero count".to_string(),
        parameters: vec![
            ToolParameter {
                name: "data_file".to_string(),
                description: "Path to the data file (.zarr or .h5)".to_string(),
                param_type: ParameterType::FilePath,
                required: true,
                default: None,
            },
        ],
        handler: Box::new(|params| {
            let data_file = params.get_string("data_file")?;

            // Use data_beans to get info
            use data_beans::sparse_io::{open_sparse_matrix, SparseIoBackend};

            let backend = if data_file.ends_with(".zarr") || data_file.ends_with(".zarr.zip") {
                SparseIoBackend::Zarr
            } else {
                SparseIoBackend::HDF5
            };

            let sparse = open_sparse_matrix(&data_file, &backend)?;

            let rows = sparse.num_rows().unwrap_or(0);
            let cols = sparse.num_columns().unwrap_or(0);
            let nnz = sparse.num_non_zeros().unwrap_or(0);

            let info = format!(
                "Matrix Info:\n  Rows: {}\n  Columns: {}\n  Non-zeros: {}",
                rows, cols, nnz
            );

            Ok(info)
        }),
    });

    // List column names tool
    registry.register(ToolDefinition {
        name: "list_column_names".to_string(),
        description: "List column names (cell barcodes) from a sparse matrix".to_string(),
        parameters: vec![
            ToolParameter {
                name: "data_file".to_string(),
                description: "Path to the data file (.zarr or .h5)".to_string(),
                param_type: ParameterType::FilePath,
                required: true,
                default: None,
            },
            ToolParameter {
                name: "limit".to_string(),
                description: "Maximum number of names to return".to_string(),
                param_type: ParameterType::Integer,
                required: false,
                default: Some(serde_json::json!(20)),
            },
        ],
        handler: Box::new(|params| {
            let data_file = params.get_string("data_file")?;
            let limit = params.get_usize_or("limit", 20);

            use data_beans::sparse_io::{open_sparse_matrix, SparseIoBackend};

            let backend = if data_file.ends_with(".zarr") || data_file.ends_with(".zarr.zip") {
                SparseIoBackend::Zarr
            } else {
                SparseIoBackend::HDF5
            };

            let sparse = open_sparse_matrix(&data_file, &backend)?;
            let names = sparse.column_names()?;

            let total = names.len();
            let show_count = limit.min(total);

            let mut result = format!("Column names ({} total, showing first {}):\n", total, show_count);
            for (i, name) in names.iter().take(limit).enumerate() {
                result.push_str(&format!("  {}: {}\n", i, name));
            }

            Ok(result)
        }),
    });

    // List row names tool
    registry.register(ToolDefinition {
        name: "list_row_names".to_string(),
        description: "List row names (gene/feature names) from a sparse matrix".to_string(),
        parameters: vec![
            ToolParameter {
                name: "data_file".to_string(),
                description: "Path to the data file (.zarr or .h5)".to_string(),
                param_type: ParameterType::FilePath,
                required: true,
                default: None,
            },
            ToolParameter {
                name: "limit".to_string(),
                description: "Maximum number of names to return".to_string(),
                param_type: ParameterType::Integer,
                required: false,
                default: Some(serde_json::json!(20)),
            },
        ],
        handler: Box::new(|params| {
            let data_file = params.get_string("data_file")?;
            let limit = params.get_usize_or("limit", 20);

            use data_beans::sparse_io::{open_sparse_matrix, SparseIoBackend};

            let backend = if data_file.ends_with(".zarr") || data_file.ends_with(".zarr.zip") {
                SparseIoBackend::Zarr
            } else {
                SparseIoBackend::HDF5
            };

            let sparse = open_sparse_matrix(&data_file, &backend)?;
            let names = sparse.row_names()?;

            let total = names.len();
            let show_count = limit.min(total);

            let mut result = format!("Row names ({} total, showing first {}):\n", total, show_count);
            for (i, name) in names.iter().take(limit).enumerate() {
                result.push_str(&format!("  {}: {}\n", i, name));
            }

            Ok(result)
        }),
    });

    // List H5 contents tool
    registry.register(ToolDefinition {
        name: "list_h5_contents".to_string(),
        description: "List the contents of an HDF5 file to see its structure".to_string(),
        parameters: vec![
            ToolParameter {
                name: "h5_file".to_string(),
                description: "Path to the HDF5 file".to_string(),
                param_type: ParameterType::FilePath,
                required: true,
                default: None,
            },
        ],
        handler: Box::new(|params| {
            let h5_file = params.get_string("h5_file")?;

            // This would call the list_h5 handler
            // For now, return a placeholder since handlers aren't exported yet
            Ok(format!("Listing contents of HDF5 file: {}\n(Full implementation requires data-beans handler export)", h5_file))
        }),
    });

    // List Zarr contents tool
    registry.register(ToolDefinition {
        name: "list_zarr_contents".to_string(),
        description: "List the contents of a Zarr file to see its structure".to_string(),
        parameters: vec![
            ToolParameter {
                name: "zarr_file".to_string(),
                description: "Path to the Zarr file".to_string(),
                param_type: ParameterType::FilePath,
                required: true,
                default: None,
            },
        ],
        handler: Box::new(|params| {
            let zarr_file = params.get_string("zarr_file")?;

            // This would call the list_zarr handler
            Ok(format!("Listing contents of Zarr file: {}\n(Full implementation requires data-beans handler export)", zarr_file))
        }),
    });

    // TODO: Add more tools as data-beans exports more handlers:
    // - run_statistics
    // - subset_columns
    // - extract_columns
    // - extract_rows
    // - build_from_mtx
    // - merge_backends
    // - squeeze
}
