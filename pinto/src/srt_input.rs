use crate::srt_cell_pairs::connected_components;
use crate::srt_common::*;
use crate::srt_knn_graph::KnnGraph;
use clap::Parser;
use data_beans::convert::try_open_or_convert;

/// Shared CLI arguments for spatial data input across all pinto subcommands.
///
/// Flatten this into each subcommand's args struct with `#[command(flatten)]`.
#[derive(Parser, Debug, Clone)]
pub struct SrtInputArgs {
    #[arg(
        required = true,
        value_delimiter(','),
        help = "Spatial gene expression data files (.zarr or .h5)",
        long_help = "Spatial gene expression data files (.zarr or .h5 format, comma separated).\n\
                       Each file is a genes-by-cells sparse matrix.\n\
                       Multiple files are concatenated column-wise (cells), and each file\n\
                       is treated as a separate batch unless --batch is specified."
    )]
    pub data_files: Vec<Box<str>>,

    #[arg(
        long = "coord",
        short = 'c',
        required = true,
        value_delimiter(','),
        help = "Spatial coordinate files, one per data file",
        long_help = "Spatial coordinate files, one per data file (comma separated).\n\
                       Accepted formats: CSV, TSV, space-delimited text, or .parquet.\n\
                       First column: cell/barcode names (must match data file column names).\n\
                       Subsequent columns: spatial coordinates (x, y, ...).\n\
                       Header row is auto-detected, or use --coord-header-row to specify.\n\
                       Coordinate columns are selected by --coord-column-names or\n\
                       --coord-column-indices."
    )]
    pub coord_files: Vec<Box<str>>,

    #[arg(
        long = "coord-column-indices",
        value_delimiter(','),
        help = "0-based column indices for coordinates in coord files",
        long_help = "0-based column indices for coordinate columns (comma separated).\n\
                       Use when the coord file has extra columns beyond barcode,x,y.\n\
                       Overrides --coord-column-names when both are specified.\n\
                       Example: --coord-column-indices 1,2 for 2nd and 3rd columns."
    )]
    pub coord_columns: Option<Vec<usize>>,

    #[arg(
        long = "coord-column-names",
        value_delimiter(','),
        default_value = "pxl_row_in_fullres,pxl_col_in_fullres",
        help = "Column names for spatial coordinates in coord files",
        long_help = "Column names to select as spatial coordinates (comma separated).\n\
                       Looked up in the header row of each coord file.\n\
                       Default suits 10x Visium tissue_positions files.\n\
                       For generic x,y data, use: --coord-column-names x,y"
    )]
    pub coord_column_names: Vec<Box<str>>,

    #[arg(
        long,
        help = "Header row index in coord files (0 = first line)",
        long_help = "0-based row index of the header in coord files.\n\
                       If omitted, auto-detected by checking whether the first row\n\
                       looks numeric. Set to 0 when the first line is column names."
    )]
    pub coord_header_row: Option<usize>,

    #[arg(
        long,
        short = 'b',
        value_delimiter(','),
        help = "Batch label files, one per data file",
        long_help = "Batch membership files, one per data file (comma separated).\n\
                       Plain text, one batch label per line (one line per cell).\n\
                       Cells with the same label share batch effects.\n\
                       If omitted, each data file is treated as one batch."
    )]
    pub batch_files: Option<Vec<Box<str>>>,

    #[arg(
        long,
        default_value_t = false,
        help = "Auto-detect batches from disconnected spatial graph components"
    )]
    pub auto_batch: bool,

    #[arg(
        long,
        short,
        required = true,
        help = "Output file prefix (e.g., results/my_run)"
    )]
    pub out: Box<str>,

    #[arg(
        long,
        default_value_t = 100,
        help = "Number of cell pairs per parallel block",
        long_help = "Number of cell pairs processed per parallel block.\n\
                       Larger values use more memory but reduce I/O overhead.\n\
                       Decrease if running out of memory."
    )]
    pub block_size: usize,

    #[arg(
        long,
        default_value_t = false,
        help = "Preload all sparse data into memory",
        long_help = "Preload all sparse column data into memory before processing.\n\
                       Faster when data fits in RAM; required for some parallel\n\
                       access patterns. Increases peak memory usage."
    )]
    pub preload_data: bool,

    #[arg(long, default_value_t = 42, help = "Random seed for reproducibility")]
    pub seed: u64,

    #[arg(
        long,
        default_value_t = 5,
        help = "Number of multi-level coarsening levels",
        long_help = "Number of hierarchical coarsening levels (coarse-to-fine).\n\
                       At each level, cells are merged via graph-constrained matching,\n\
                       halving the number of groups. More levels give better initialization\n\
                       for the finest level but take longer. Typical range: 2-10."
    )]
    pub num_levels: usize,

    #[arg(
        long,
        short = 'p',
        default_value_t = 50,
        help = "Random projection dimension for cell embeddings",
        long_help = "Dimension of random projections used for cell embeddings.\n\
                       Cells are projected from gene space (G dims) down to this\n\
                       dimension for KNN graph construction and coarsening.\n\
                       Higher values preserve more signal but cost more memory."
    )]
    pub proj_dim: usize,

    #[arg(
        long = "batch-knn",
        default_value_t = 10,
        help = "KNN for cross-batch matching during batch correction",
        long_help = "Number of nearest neighbours for cross-batch super-cell matching.\n\
                       During batch effect estimation, cells are coarsened into super-cells.\n\
                       Each super-cell finds its K nearest neighbors from other batches via\n\
                       HNSW on centroids. These cross-batch matches provide counterfactual\n\
                       expression estimates for batch effect decomposition.\n\
                       Only used when multiple batches are present."
    )]
    pub batch_knn: usize,

    #[arg(
        long,
        short = 'd',
        default_value_t = 1024,
        help = "Target number of pseudobulk samples at coarsest level",
        long_help = "Target number of pseudobulk cell groups at the coarsest level.\n\
                       Cells are merged via graph-constrained coarsening until roughly\n\
                       this many groups remain. These pseudobulk samples are used for\n\
                       Poisson-Gamma estimation and multi-level refinement.\n\
                       Larger values give finer granularity but slower coarsening."
    )]
    pub n_pseudobulk: usize,

    #[arg(
        short = 'k',
        long,
        default_value_t = 5,
        help = "Spatial KNN: neighbours per cell for cell-pair graph",
        long_help = "Number of nearest neighbours per cell for building the spatial\n\
                       cell-pair graph. Each cell connects to its K closest neighbours\n\
                       in coordinate space (Euclidean distance, HNSW index).\n\
                       The resulting edges define cell pairs for all downstream analysis.\n\
                       Typical range: 3-20. Lower values capture fine local structure;\n\
                       higher values smooth over larger neighbourhoods."
    )]
    pub knn_spatial: usize,

    #[arg(
        long,
        default_value_t = false,
        help = "Use reciprocal (mutual) KNN matching for spatial graph",
        long_help = "Use reciprocal (mutual) KNN matching for the spatial cell graph.\n\
                       By default, union matching is used: an edge (i,j) exists if i is\n\
                       in j's KNN list OR j is in i's KNN list. With reciprocal matching,\n\
                       both must be in each other's KNN list, producing a sparser graph\n\
                       with higher-confidence edges."
    )]
    pub reciprocal: bool,
}

impl SrtInputArgs {
    /// Convert to the internal read args for data loading.
    pub fn to_read_args(&self) -> SRTReadArgs {
        SRTReadArgs {
            data_files: self.data_files.clone(),
            coord_files: self.coord_files.clone(),
            preload_data: self.preload_data,
            coord_columns: self.coord_columns.clone().unwrap_or_default(),
            coord_column_names: self.coord_column_names.clone(),
            batch_files: self.batch_files.clone(),
            header_in_coord: self.coord_header_row,
        }
    }
}

pub struct SRTReadArgs {
    pub data_files: Vec<Box<str>>,
    pub coord_files: Vec<Box<str>>,
    pub preload_data: bool,
    pub coord_columns: Vec<usize>,
    pub coord_column_names: Vec<Box<str>>,
    pub batch_files: Option<Vec<Box<str>>>,
    pub header_in_coord: Option<usize>,
}

pub struct SRTData {
    pub data: SparseIoVec,
    pub coordinates: Mat,
    pub coordinate_names: Vec<Box<str>>,
    pub batches: Vec<Box<str>>,
}

pub fn read_expr_data(data_files: &[Box<str>]) -> anyhow::Result<SparseIoVec> {
    if data_files.is_empty() {
        return Err(anyhow::anyhow!("empty data files"));
    }

    let attach_data_name = data_files.len() > 1;
    let mut data_vec = SparseIoVec::new();

    for data_file in data_files.iter() {
        info!("Importing data file: {}", data_file);
        let data = try_open_or_convert(data_file)?;
        let data_name = attach_data_name.then(|| basename(data_file)).transpose()?;
        data_vec.push(Arc::from(data), data_name)?;
    }

    Ok(data_vec)
}

pub fn read_data_with_coordinates(args: SRTReadArgs) -> anyhow::Result<SRTData> {
    anyhow::ensure!(
        args.coord_files.len() == args.data_files.len(),
        "Number of coordinate files ({}) must match number of data files ({})",
        args.coord_files.len(),
        args.data_files.len()
    );

    let attach_data_name = args.data_files.len() > 1;
    let mut data_vec = SparseIoVec::new();

    for data_file in args.data_files.iter() {
        info!("Importing data file: {}", data_file);

        let mut data = try_open_or_convert(data_file)?;
        let data_name = attach_data_name.then(|| basename(data_file)).transpose()?;

        if args.preload_data {
            data.preload_columns()?;
        }

        data_vec.push(Arc::from(data), data_name)?;
    }

    // check if row names are the same across data
    let row_names = data_vec[0].row_names()?;

    for j in 1..data_vec.len() {
        let row_names_j = data_vec[j].row_names()?;
        if row_names != row_names_j {
            return Err(anyhow::anyhow!("Row names are not the same"));
        }
    }

    let mut coord_vec = Vec::with_capacity(args.coord_files.len());

    let mut coord_column_names = vec![];

    for (i, coord_file) in args.coord_files.iter().enumerate() {
        info!("Reading coordinate file: {}", coord_file);
        let ext = file_ext(coord_file)?;

        let MatWithNames {
            rows: coord_cell_names,
            cols: column_names,
            mat: data,
        } = match ext.as_ref() {
            "parquet" => Mat::from_parquet_with_indices_names(
                coord_file,
                Some(0),
                Some(&args.coord_columns),
                Some(&args.coord_column_names),
            )?,
            _ => {
                let header_row = args.header_in_coord.or_else(|| {
                    detect_header_row(coord_file, &['\t', ',', ' '], &args.coord_column_names)
                });
                Mat::read_data(
                    coord_file,
                    &['\t', ',', ' '],
                    header_row,
                    Some(0),
                    Some(&args.coord_columns),
                    Some(&args.coord_column_names),
                )?
            }
        };

        let data_cell_names = data_vec[i].column_names()?;

        if coord_column_names.is_empty() {
            coord_column_names.extend(column_names);
        } else if coord_column_names != column_names {
            return Err(anyhow::anyhow!(
                "coordinate column names do not match with each other"
            ));
        }

        if data_cell_names == coord_cell_names {
            coord_vec.push(data);
        } else {
            info!("reordering coordinate information");

            let coord_index_map: HashMap<&Box<str>, usize> = coord_cell_names
                .iter()
                .enumerate()
                .map(|(index, name)| (name, index))
                .collect();

            let reordered_indices: Vec<usize> = data_cell_names
                .iter()
                .map(|name| {
                    coord_index_map
                        .get(name)
                        .ok_or_else(|| {
                            anyhow::anyhow!("cell '{}' not found in the file {}", name, coord_file)
                        })
                        .copied()
                })
                .collect::<anyhow::Result<_>>()?;

            coord_vec.push(concatenate_vertical(
                &reordered_indices
                    .iter()
                    .map(|&index| data.row(index))
                    .collect::<Vec<_>>(),
            )?);
        }
    }

    let coord_nk = concatenate_vertical(&coord_vec)?;

    // will incorporate batch label as an additional coordinate
    let mut batch_membership = Vec::with_capacity(data_vec.len());

    if let Some(batch_files) = &args.batch_files {
        if batch_files.len() != args.data_files.len() {
            return Err(anyhow::anyhow!("# batch files != # of data files"));
        }

        for batch_file in batch_files.iter() {
            info!("Reading batch file: {}", batch_file);
            for s in read_lines(batch_file)? {
                batch_membership.push(s.to_string().into_boxed_str());
            }
        }
    } else if data_vec.len() > 1 {
        info!("Each data file will be considered a different batch.");
        for (id, &nn) in data_vec.num_columns_by_data()?.iter().enumerate() {
            batch_membership.extend(vec![id.to_string().into_boxed_str(); nn]);
        }
    } else {
        // Single data file, single batch — uniform label
        let nn = data_vec.num_columns();
        batch_membership.extend(vec!["0".to_string().into_boxed_str(); nn]);
    }

    if batch_membership.len() != data_vec.num_columns() {
        return Err(anyhow::anyhow!(
            "# batch membership {} != # of columns {}",
            batch_membership.len(),
            data_vec.num_columns()
        ));
    }

    // use batch index as another coordinate
    let uniq_batches = batch_membership.par_iter().cloned().collect::<HashSet<_>>();
    let n_batches = uniq_batches.len();
    let coord_nk = if n_batches > 1 {
        info!("attaching {} batch index coordinate(s)", n_batches);
        coord_column_names.push("batch".to_string().into_boxed_str());
        append_batch_coordinate(&coord_nk, &batch_membership)?
    } else {
        coord_nk
    };

    info!(
        "Read {} x {} coordinates",
        coord_nk.nrows(),
        coord_nk.ncols()
    );

    Ok(SRTData {
        data: data_vec,
        coordinates: coord_nk,
        coordinate_names: coord_column_names,
        batches: batch_membership,
    })
}

/// Replace batch membership with connected component labels if the spatial
/// graph has multiple disconnected components (e.g., tissue microarray cores).
///
/// Returns the number of components found.
pub fn auto_batch_from_components(graph: &KnnGraph, batch_membership: &mut Vec<Box<str>>) -> usize {
    let (labels, n_components) = connected_components(graph);
    if n_components > 1 {
        *batch_membership = labels
            .iter()
            .map(|l| format!("cc_{l}").into_boxed_str())
            .collect();
        info!(
            "Auto-detected {} spatial components as batches",
            n_components
        );
    }
    n_components
}

/// Auto-detect whether the first line of a delimited file is a header row
/// by checking if it contains any of the requested column names.
fn detect_header_row(
    file_path: &str,
    delimiters: &[char],
    column_names: &[Box<str>],
) -> Option<usize> {
    if column_names.is_empty() {
        return None;
    }
    let first_line = std::io::BufRead::lines(std::io::BufReader::new(
        std::fs::File::open(file_path).ok()?,
    ))
    .next()?
    .ok()?;
    let tokens: HashSet<&str> = first_line.split(delimiters.as_ref()).collect();
    if column_names
        .iter()
        .any(|name| tokens.contains(name.as_ref()))
    {
        info!("Auto-detected header row in {}", file_path);
        Some(0)
    } else {
        None
    }
}

fn append_batch_coordinate<T>(coords: &Mat, batch_membership: &[T]) -> anyhow::Result<Mat>
where
    T: Sync + Send + Clone + Eq + std::hash::Hash + std::fmt::Debug,
{
    if coords.nrows() != batch_membership.len() {
        return Err(anyhow::anyhow!("incompatible batch membership"));
    }

    let minval = coords.min();
    let maxval = coords.max();
    let width = (maxval - minval).max(1.);

    let uniq_batches = batch_membership.iter().collect::<HashSet<_>>();

    let batch_index = uniq_batches
        .into_iter()
        .enumerate()
        .map(|(k, v)| (v, k))
        .collect::<HashMap<_, _>>();

    let batch_coord = batch_membership
        .iter()
        .map(|k| {
            let b = *batch_index
                .get(k)
                .ok_or_else(|| anyhow::anyhow!("batch key {:?} not found in index", k))?;
            Ok(width * (b as f32))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let bb = Mat::from_vec(coords.nrows(), 1, batch_coord);

    concatenate_horizontal(&[coords.clone(), bb])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_append_batch_coordinate_single_batch() {
        let coords = Mat::from_vec(3, 2, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let batches = vec!["A", "A", "A"];
        let result = append_batch_coordinate(&coords, &batches).unwrap();
        assert_eq!(result.nrows(), 3);
        assert_eq!(result.ncols(), 3); // original 2 + 1 batch
                                       // Single batch → all batch coordinates should be 0.0
        for i in 0..3 {
            assert_eq!(result[(i, 2)], 0.0);
        }
    }

    #[test]
    fn test_append_batch_coordinate_two_batches() {
        let coords = Mat::from_vec(4, 2, vec![0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0]);
        let batches = vec!["A", "A", "B", "B"];
        let result = append_batch_coordinate(&coords, &batches).unwrap();
        assert_eq!(result.ncols(), 3);
        // Batch coordinates should differ between A and B
        let batch_a = result[(0, 2)];
        let batch_b = result[(2, 2)];
        assert_ne!(
            batch_a, batch_b,
            "different batches should have different coordinates"
        );
        // Same batch should have same coordinate
        assert_eq!(result[(0, 2)], result[(1, 2)]);
        assert_eq!(result[(2, 2)], result[(3, 2)]);
    }

    #[test]
    fn test_append_batch_coordinate_scaling() {
        // Width = max - min of all coords
        let coords = Mat::from_vec(2, 1, vec![0.0, 10.0]);
        let batches = vec!["X", "Y"];
        let result = append_batch_coordinate(&coords, &batches).unwrap();
        // Width = 10.0, one batch at 0*width, other at 1*width
        let vals: Vec<f32> = (0..2).map(|i| result[(i, 1)]).collect();
        let (lo, hi) = if vals[0] < vals[1] {
            (vals[0], vals[1])
        } else {
            (vals[1], vals[0])
        };
        assert_eq!(lo, 0.0);
        assert_eq!(hi, 10.0); // 1 * width
    }

    #[test]
    fn test_append_batch_coordinate_mismatch() {
        let coords = Mat::from_vec(2, 1, vec![0.0, 1.0]);
        let batches = vec!["A"]; // wrong length
        assert!(append_batch_coordinate(&coords, &batches).is_err());
    }
}
