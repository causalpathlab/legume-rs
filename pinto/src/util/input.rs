use crate::util::cell_pairs::connected_components;
use crate::util::common::*;
use crate::util::knn_graph::KnnGraph;
use auxiliary_data::feature_names::FeatureNameKind;
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
        long_help = "Spatial gene expression data files, comma separated.\n\
                     Accepted formats are .zarr and .h5.\n\
                     Each file is a genes-by-cells sparse matrix.\n\
                     Multiple files are concatenated column-wise, over cells.\n\
                     Each file is then its own batch unless --batch-files says otherwise."
    )]
    pub data_files: Vec<Box<str>>,

    #[arg(
        long = "coord",
        short = 'c',
        value_delimiter(','),
        help = "Spatial coordinate files, one per data file (recommended)",
        long_help = "Spatial coordinate files, one per data file, comma separated.\n\
                     Recommended for spatial transcriptomics data.\n\
                     Omit them to run in expression mode.\n\
                     The KNN graph then comes from gene expression.\n\
                     \n\
                     Accepted formats: CSV, TSV, space-delimited text, .parquet,\n\
                     or .zarr/.zarr.zip (Xenium cells.zarr.zip).\n\
                     \n\
                     In CSV, TSV and parquet, the first column is cell names.\n\
                     Those names must match the data file's column names.\n\
                     Every later column is a spatial coordinate.\n\
                     \n\
                     In zarr, cell IDs are read from /cell_id.\n\
                     Coordinates come from /cell_summary.\n\
                     Its attributes supply the column names.\n\
                     \n\
                     Pick columns with --coord-column-names or -indices."
    )]
    pub coord_files: Vec<Box<str>>,

    #[arg(
        long = "coord-column-indices",
        value_delimiter(','),
        help = "0-based column indices for coordinates in coord files",
        long_help = "0-based column indices for coordinate columns, comma separated.\n\
                     Use this when the coord file has columns beyond barcode,x,y.\n\
                     It overrides --coord-column-names when both are given. Example:\n\
                     --coord-column-indices 1,2 picks columns 2 and 3."
    )]
    pub coord_columns: Option<Vec<usize>>,

    #[arg(
        long = "coord-column-names",
        value_delimiter(','),
        default_value = "pxl_row_in_fullres,pxl_col_in_fullres,cell_centroid_x,cell_centroid_y,x_centroid,y_centroid",
        help = "Column names for spatial coordinates in coord files",
        long_help = "Column names to select as spatial coordinates, comma separated.\n\
                     Names absent from a given file are skipped, so the default\n\
                     covers the common single-cell and spot layouts at once.\n\
                     \n\
                     Default: pxl_row_in_fullres, pxl_col_in_fullres,\n\
                     cell_centroid_x, cell_centroid_y, x_centroid, y_centroid.\n\
                     \n\
                     Quoting does not matter. A file whose header reads \"x_centroid\"\n\
                     matches the name x_centroid, and the same holds for the cell\n\
                     names in the first column.\n\
                     \n\
                     If none of these names is present, the file is read by POSITION\n\
                     instead, taking columns 4 and 5. That is the tissue_positions\n\
                     layout, and it is only attempted when the file looks like one,\n\
                     meaning it also carries in_tissue, array_row and array_col.\n\
                     Any other named file is rejected with its column list, rather\n\
                     than read positionally: two numeric columns in those positions\n\
                     would otherwise be accepted as coordinates without complaint.\n\
                     Pass this flag with two of the names it printed to resolve it."
    )]
    pub coord_column_names: Vec<Box<str>>,

    #[arg(
        long,
        help = "Header row index in coord files (0 = first line)",
        long_help = "0-based row index of the header in coord files. If omitted,\n\
                     it is auto-detected. Detection checks whether the first row looks numeric.\n\
                     Set it to 0 when the first line holds column names.",
        hide = true
    )]
    pub coord_header_row: Option<usize>,

    #[arg(
        long,
        short = 'b',
        value_delimiter(','),
        help = "Batch label files, one per data file",
        long_help = "Batch membership files, one per data file, comma separated.\n\
                     Each is plain text with one batch label per cell, per line.\n\
                     Cells sharing a label share batch effects. If omitted,\n\
                     each data file is treated as one batch."
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
        help = "Cells per parallel block (omit for auto-scaling by feature count)",
        hide = true
    )]
    pub block_size: Option<usize>,

    #[arg(
        long,
        default_value_t = false,
        help = "Preload all sparse data into memory",
        long_help = "Preload all sparse column data into memory up front.\n\
                     Faster when the data fits in RAM.\n\
                     Some parallel access patterns require it. It raises peak memory usage."
    )]
    pub preload_data: bool,

    #[arg(long, default_value_t = 42, help = "Random seed for reproducibility")]
    pub seed: u64,

    #[arg(
        long,
        default_value_t = 5,
        help = "Number of multi-level coarsening levels",
        long_help = "Number of hierarchical coarsening levels, coarse to fine.\n\
                     Each level merges cells by graph-constrained matching.\n\
                     That halves the number of groups per level.\n\
                     More levels initialize the finest level better. They also take longer.\n\
                     Typical range: 2-10.",
        hide = true
    )]
    pub num_levels: usize,

    #[arg(
        long,
        default_value_t = 5,
        help = "Refinement sweeps per coarsening level (0 to disable)",
        long_help = "Leiden-style local-moving sweeps per coarsening level.\n\
                     They run after the dendrogram cut.\n\
                     A sweep moves nodes to graph-adjacent clusters.\n\
                     A move must improve cosine similarity to the centroid.\n\
                     Moves that would disconnect the source cluster are rejected.\n\
                     Sweeping stops early once no node moves. 5 is usually enough;\n\
                     0 skips refinement.",
        hide = true
    )]
    pub refine_iterations: usize,

    #[arg(
        long,
        short = 'p',
        default_value_t = 200,
        help = "Random projection dimension for cell embeddings",
        long_help = "Dimension of the random projection for cell embeddings.\n\
                     Cells are projected from G gene dimensions down to this one.\n\
                     That projection feeds KNN construction and coarsening,\n\
                     and in `lc` it also sets the width of every edge profile.\n\
                     \n\
                     This is the main quality knob, and it is easy to set too low.\n\
                     A sketch too narrow to separate two cell programs merges them,\n\
                     so the rare profiles that sit between programs get absorbed\n\
                     into whichever neighbouring community is largest.\n\
                     That shows up as speckle in a propensity map\n\
                     rather than as coherent spatial domains.\n\
                     \n\
                     Cost is linear in this value:\n\
                     runtime, peak memory, and the size of the edge profile store.\n\
                     Raise it when domains look speckled or over-merged.\n\
                     Lower it when a large dataset will not fit."
    )]
    pub proj_dim: usize,

    #[arg(
        long = "batch-knn",
        default_value_t = 10,
        help = "KNN for cross-batch matching during batch correction",
        long_help = "Neighbours per pb-sample for cross-batch matching.\n\
                     Batch-effect estimation first coarsens cells into pb-samples.\n\
                     Each pb-sample then finds K neighbours in other batches.\n\
                     The search is HNSW over centroids.\n\
                     Those matches give counterfactual expression estimates.\n\
                     Batch-effect decomposition needs them.\n\
                     This is used only when multiple batches are present.",
        hide = true
    )]
    pub batch_knn: usize,

    #[arg(
        long,
        short = 'd',
        default_value_t = 1024,
        help = "Target number of pseudobulk samples at coarsest level",
        long_help = "Target number of pseudobulk cell groups at the coarsest level.\n\
                     Coarsening merges cells until roughly this many remain.\n\
                     They feed Poisson-Gamma estimation and refinement.\n\
                     Larger values give finer granularity. They also coarsen more slowly."
    )]
    pub n_pseudobulk: usize,

    #[arg(
        short = 'k',
        long,
        default_value_t = 5,
        help = "KNN: neighbours per cell for cell-pair graph",
        long_help = "Neighbours per cell when building the cell-pair graph.\n\
                     Each cell connects to its K closest neighbours.\n\
                     Search runs over an HNSW index in Euclidean distance.\n\
                     \n\
                     With --coord, neighbours live in coordinate space. Without it,\n\
                     they live in expression embedding space.\n\
                     \n\
                     The resulting edges are the cell pairs used downstream. Typical range:\n\
                     3-20 spatial, 10-30 expression."
    )]
    pub knn_spatial: usize,

    #[arg(
        long,
        default_value_t = 0,
        help = "KNN: expression-similar neighbours added to the pair graph (0 = off)",
        long_help = "Neighbours per cell in a second, non-spatial KNN graph built on\n\
                     random-projected expression.\n\
                     Its edges are added to the spatial ones, so the cell pairs\n\
                     downstream are the union of both.\n\
                     \n\
                     Expression-similar pairs are the same cell type by construction.\n\
                     They act as a reference for what a same-type pair looks like,\n\
                     which sharpens those communities and leaves the pairs that bridge\n\
                     two types standing out as the residue.\n\
                     \n\
                     The expression graph is built after batch correction, so it\n\
                     matches cell type rather than batch.\n\
                     \n\
                     This is not symmetric with -k: expression neighbours agree with\n\
                     each other less often, so fewer of their edges collapse together\n\
                     and the same K adds more pairs than the spatial graph holds.\n\
                     Runtime and peak memory both rise with the resulting pair count.\n\
                     \n\
                     Ignored without --coord, where the graph already comes from\n\
                     expression.\n\
                     \n\
                     Off by default. On one high-density section it roughly doubled\n\
                     both runtime and peak memory, and the communities it separated\n\
                     tracked annotated tissue boundaries only weakly. Worth trying\n\
                     when interfaces are the question; not worth paying for otherwise.\n\
                     Try 5 to 20."
    )]
    pub knn_expr: usize,

    #[arg(
        long,
        default_value_t = false,
        help = "Use reciprocal (mutual) KNN matching for spatial graph",
        long_help = "Use reciprocal (mutual) KNN matching for the spatial graph.\n\
                     The default is union matching.\n\
                     There an edge (i,j) exists if i is in j's KNN list, or j is in i's.\n\
                     Reciprocal matching requires both.\n\
                     That yields a sparser graph of higher-confidence edges.",
        hide = true
    )]
    pub reciprocal: bool,

    #[command(flatten)]
    pub qc: data_beans::qc_lib::QcArgs,
}

impl SrtInputArgs {
    /// Whether spatial coordinate files were provided.
    pub fn has_coordinates(&self) -> bool {
        !self.coord_files.is_empty()
    }

    /// Comma-joined string of coordinate file paths, or `None` when the
    /// run was without `--coord`. Convenient for .pinto.json fields.
    pub fn coord_files_joined(&self) -> Option<String> {
        if self.coord_files.is_empty() {
            None
        } else {
            Some(
                self.coord_files
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            )
        }
    }

    /// Convert to the internal read args for data loading. The
    /// `feature_kind` selects how row names get canonicalized for
    /// cross-file / cross-resource matching. `pinto cage` uses
    /// `FeatureNameKind::Gene { delim: '_' }` (or `auto_detect`'d) so
    /// gene names like `ENSG00000105329_TGFB1` register both the full
    /// name and the `TGFB1` suffix as aliases — required for matching
    /// against external gene resources (PPI, marker sets, etc.). `lc`
    /// and `svd` currently pass `FeatureNameKind::Exact` for strict
    /// equality.
    pub fn to_read_args_with_kind(&self, feature_kind: FeatureNameKind) -> SRTReadArgs {
        SRTReadArgs {
            data_files: self.data_files.clone(),
            coord_files: self.coord_files.clone(),
            preload_data: self.preload_data,
            coord_columns: self.coord_columns.clone().unwrap_or_default(),
            coord_column_names: self.coord_column_names.clone(),
            batch_files: self.batch_files.clone(),
            header_in_coord: self.coord_header_row,
            feature_kind,
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
    /// Optional row-name canonicalizer for fuzzy cross-file gene/locus
    /// alignment. Default = `Exact` keeps the strict row-name equality
    /// check in [`read_data_with_coordinates`] (used by `pinto lc` and
    /// friends). When non-Exact, canonicalization runs through
    /// [`SparseIoVec::with_row_canonicalizer`] and the strict check is
    /// skipped — `SparseIoVec`'s intersection logic handles alignment.
    pub feature_kind: FeatureNameKind,
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
    let canonicalize_rows = !args.feature_kind.is_exact();
    if let Some(canon) = args.feature_kind.clone().into_canonicalizer() {
        data_vec = data_vec
            .with_row_canonicalizer(move |name| canon(name))
            .expect("with_row_canonicalizer on empty SparseIoVec");
    }

    for data_file in args.data_files.iter() {
        info!("Importing data file: {}", data_file);

        let mut data = try_open_or_convert(data_file)?;
        let data_name = attach_data_name.then(|| basename(data_file)).transpose()?;

        if args.preload_data {
            data.preload_columns()?;
        }

        data_vec.push(Arc::from(data), data_name)?;
    }

    // Strict row-name equality check is only meaningful for exact-match
    // mode; when canonicalization is enabled, SparseIoVec's intersection
    // already aligns rows and a per-backend equality check would fail on
    // the very inputs the canonicalizer is meant to handle.
    if !canonicalize_rows {
        let row_names = data_vec[0].row_names()?;
        for j in 1..data_vec.len() {
            let row_names_j = data_vec[j].row_names()?;
            if row_names != row_names_j {
                return Err(anyhow::anyhow!("Row names are not the same"));
            }
        }
    }

    let mut coord_vec = Vec::with_capacity(args.coord_files.len());

    let mut coord_column_names = vec![];

    for (i, coord_file) in args.coord_files.iter().enumerate() {
        info!("Reading coordinate file: {}", coord_file);

        let MatWithNames {
            rows: coord_cell_names,
            cols: column_names,
            mat: data,
        } = read_one_coord_file(
            coord_file,
            &args.coord_columns,
            &args.coord_column_names,
            args.header_in_coord,
        )?;

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

/// Load expression data and batch labels without spatial coordinates.
///
/// Use this when no coordinate files are provided (expression mode).
/// Coordinates will be filled in later from expression embeddings.
pub fn read_data_without_coordinates(args: SRTReadArgs) -> anyhow::Result<SRTData> {
    let attach_data_name = args.data_files.len() > 1;
    let mut data_vec = SparseIoVec::new();
    let canonicalize_rows = !args.feature_kind.is_exact();
    if let Some(canon) = args.feature_kind.clone().into_canonicalizer() {
        data_vec = data_vec
            .with_row_canonicalizer(move |name| canon(name))
            .expect("with_row_canonicalizer on empty SparseIoVec");
    }

    for data_file in args.data_files.iter() {
        info!("Importing data file: {}", data_file);

        let mut data = try_open_or_convert(data_file)?;
        let data_name = attach_data_name.then(|| basename(data_file)).transpose()?;

        if args.preload_data {
            data.preload_columns()?;
        }

        data_vec.push(Arc::from(data), data_name)?;
    }

    // Strict row-name equality only in exact-match mode (see
    // `read_data_with_coordinates` for the same rationale).
    if !canonicalize_rows {
        let row_names = data_vec[0].row_names()?;
        for j in 1..data_vec.len() {
            let row_names_j = data_vec[j].row_names()?;
            if row_names != row_names_j {
                return Err(anyhow::anyhow!("Row names are not the same"));
            }
        }
    }

    // Parse batch membership
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

    let n_cells = data_vec.num_columns();

    // Placeholder coordinates (0×0) — will be replaced by expression embeddings
    let coordinates = Mat::zeros(n_cells, 0);
    let coordinate_names = vec![];

    info!("Read {} cells (expression mode, no coordinates)", n_cells);

    Ok(SRTData {
        data: data_vec,
        coordinates,
        coordinate_names,
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

/// Read a single coordinate file with a name-then-index fallback.
///
/// Tries to read the requested coordinate columns by name first. When the
/// caller did NOT pass explicit indices (`user_indices` empty) and the
/// name-based read either errored or returned fewer than 2 columns, retries
/// with conventional 0-based indices: `[0, 1]` for zarr (whose
/// `/cell_summary` array contains only coordinate columns) or `[4, 5]` for
/// text/parquet files (Visium classic `tissue_positions` layout). Fallback
/// columns are labeled `x, y`.
pub fn read_one_coord_file(
    coord_file: &str,
    user_indices: &[usize],
    user_names: &[Box<str>],
    header_in_coord: Option<usize>,
) -> anyhow::Result<MatWithNames<Mat>> {
    let ext = file_ext(coord_file)?;
    let is_zarr = coord_file.contains(".zarr");

    let read_coord = |indices: &[usize], names: &[Box<str>]| -> anyhow::Result<MatWithNames<Mat>> {
        if is_zarr {
            data_beans::zarr_io::read_zarr_coordinates(coord_file, indices, names)
        } else {
            match ext.as_ref() {
                "parquet" => Mat::from_parquet_with_indices_names(
                    coord_file,
                    Some(0),
                    Some(indices),
                    Some(names),
                ),
                _ => {
                    let header_row = header_in_coord
                        .or_else(|| detect_header_row(coord_file, &['\t', ',', ' '], names))
                        .or_else(|| detect_header_row_numeric(coord_file, &['\t', ',', ' ']));
                    Mat::read_data(
                        coord_file,
                        &['\t', ',', ' '],
                        header_row,
                        Some(0),
                        Some(indices),
                        Some(names),
                    )
                }
            }
        }
    };

    let initial = read_coord(user_indices, user_names);
    let needs_fallback = user_indices.is_empty()
        && match &initial {
            Err(_) => true,
            Ok(r) => r.mat.ncols() < 2,
        };

    if needs_fallback {
        // Zarr's `/cell_summary` is coord-only with no barcode column, so
        // start at index 0. Text/parquet fall back to Visium classic's
        // `tissue_positions.{csv,parquet}` layout: barcode, in_tissue,
        // array_row, array_col, pxl_row_in_fullres, pxl_col_in_fullres —
        // the pixel coordinates sit at indices 4 and 5.
        let fallback_indices: Vec<usize> = if is_zarr { vec![0, 1] } else { vec![4, 5] };
        let fallback_names: Vec<Box<str>> = vec!["x".into(), "y".into()];

        // Taking columns 4 and 5 is not a general guess: it is knowledge of one
        // specific layout. So check the file IS that layout before applying it.
        //
        // Applied blindly it produces a silent wrong answer rather than an
        // error. A single-cell vendor cell table, for instance, carries
        // per-cell QC counts at exactly those positions, so every cell would
        // take the same coordinates and the run would report nothing amiss.
        //
        // A file with no header at all is left alone: there are no names to
        // check, and positional reading is the only thing available.
        if !is_zarr {
            if let Ok(found) = peek_delimited_header(coord_file) {
                let has = |want: &str| {
                    found
                        .iter()
                        .any(|f| f.eq_ignore_ascii_case(want))
                };
                // The signature of `tissue_positions.{csv,parquet}`, whose
                // pixel coordinates sit at 4 and 5 whatever they are called.
                let is_known_layout =
                    has("in_tissue") && has("array_row") && has("array_col");
                if !is_known_layout && found.len() > 1 {
                    return Err(anyhow::anyhow!(
                        "coord file '{}' has named columns, but none of them are the \
                         coordinates asked for {:?} and its layout is not one this \
                         recognises. Its columns are {:?}. Pass --coord-column-names \
                         with two of those.",
                        coord_file,
                        user_names,
                        found,
                    ));
                }
            }
        }

        match initial {
            Err(e) => warn!(
                "coord file '{}' could not be read by names {:?} ({}); \
                 falling back to 0-based column indices {:?} as (x, y)",
                coord_file, user_names, e, fallback_indices,
            ),
            Ok(r) => warn!(
                "coord file '{}' matched {} of the requested column names {:?}; \
                 falling back to 0-based column indices {:?} as (x, y)",
                coord_file,
                r.mat.ncols(),
                user_names,
                fallback_indices,
            ),
        }
        let mut r = read_coord(&fallback_indices, &fallback_names)?;
        // Override column names: the underlying readers use indices when both
        // are passed and pull names from the file's header/schema, so the
        // fallback `["x","y"]` hint is otherwise ignored. A stable label is
        // more useful downstream than whatever the file happened to have.
        r.cols = fallback_names;
        Ok(r)
    } else {
        initial
    }
}

/// Detect a header row by checking whether non-zero columns of the first
/// line contain any non-numeric tokens. Used as a second-chance fallback
/// when name-based detection fails (e.g. when reading by index with
/// non-default headers like `barcode,foo,bar`).
fn detect_header_row_numeric(file_path: &str, delimiters: &[char]) -> Option<usize> {
    let first_line = std::io::BufRead::lines(std::io::BufReader::new(
        std::fs::File::open(file_path).ok()?,
    ))
    .next()?
    .ok()?;
    let any_non_numeric_after_col0 = first_line
        .split(delimiters.as_ref())
        .skip(1)
        .any(|t| !t.is_empty() && t.parse::<f64>().is_err());
    if any_non_numeric_after_col0 {
        info!(
            "Auto-detected header row in {} (numeric heuristic)",
            file_path
        );
        Some(0)
    } else {
        None
    }
}

/// Auto-detect whether the first line of a delimited file is a header row
/// by checking if it contains any of the requested column names.
/// The column names on the first line of a delimited file, if it looks like a
/// header rather than data.
///
/// Only used to tell a user which columns their file actually has, so a
/// best-effort read is enough and any failure just means we say nothing.
fn peek_delimited_header(path: &str) -> anyhow::Result<Vec<Box<str>>> {
    use std::io::BufRead;
    let file = std::fs::File::open(path)?;
    let mut first = String::new();
    std::io::BufReader::new(file).read_line(&mut first)?;
    let fields: Vec<Box<str>> = first
        .trim_end_matches(['\n', '\r'])
        .split(['\t', ',', ' '])
        .map(|f| f.trim().trim_matches('"').to_string().into_boxed_str())
        .filter(|f| !f.is_empty())
        .collect();
    // If every field parses as a number this is data, not a header.
    anyhow::ensure!(
        fields.iter().any(|f| f.parse::<f64>().is_err()),
        "first line looks like data, not a header"
    );
    Ok(fields)
}

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

    use std::io::Write as _;

    fn write_csv(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn test_read_one_coord_file_text_match_no_fallback() {
        let f = write_csv(
            "barcode,cell_centroid_x,cell_centroid_y\n\
             cell_a,1.0,2.0\n\
             cell_b,3.0,4.0\n",
        );
        let names: Vec<Box<str>> = vec!["cell_centroid_x".into(), "cell_centroid_y".into()];
        let r = read_one_coord_file(f.path().to_str().unwrap(), &[], &names, None).unwrap();
        assert_eq!(r.mat.ncols(), 2);
        assert_eq!(r.mat.nrows(), 2);
        assert_eq!(r.cols[0].as_ref(), "cell_centroid_x");
        assert_eq!(r.cols[1].as_ref(), "cell_centroid_y");
        assert_eq!(r.rows[0].as_ref(), "cell_a");
    }

    #[test]
    fn test_read_one_coord_file_visium_classic_fallback() {
        // Visium tissue_positions.csv layout: barcode, in_tissue, array_row,
        // array_col, pxl_row_in_fullres, pxl_col_in_fullres. With unmatched
        // requested names the fallback must pick indices [4, 5] (the pixel
        // coords) — NOT [1, 2] (in_tissue, array_row).
        let f = write_csv(
            "barcode,in_tissue,array_row,array_col,my_x,my_y\n\
             cell_a,1,0,0,100.5,200.5\n\
             cell_b,1,1,1,300.5,400.5\n",
        );
        let names: Vec<Box<str>> = vec!["pxl_row_in_fullres".into(), "pxl_col_in_fullres".into()];
        let r = read_one_coord_file(f.path().to_str().unwrap(), &[], &names, None).unwrap();
        assert_eq!(r.mat.ncols(), 2);
        assert_eq!(r.mat[(0, 0)], 100.5);
        assert_eq!(r.mat[(0, 1)], 200.5);
        assert_eq!(r.mat[(1, 0)], 300.5);
        assert_eq!(r.mat[(1, 1)], 400.5);
        assert_eq!(r.cols[0].as_ref(), "x");
        assert_eq!(r.cols[1].as_ref(), "y");
    }

    #[test]
    fn test_read_one_coord_file_user_indices_skip_fallback() {
        let f = write_csv(
            "barcode,foo,bar\n\
             cell_a,10.0,20.0\n",
        );
        let names: Vec<Box<str>> = vec!["nonexistent".into()];
        let r = read_one_coord_file(f.path().to_str().unwrap(), &[1, 2], &names, None).unwrap();
        assert_eq!(r.mat.ncols(), 2);
        // Columns came from indices, so names are "foo","bar" (from header), not "x","y".
        assert_eq!(r.cols[0].as_ref(), "foo");
        assert_eq!(r.cols[1].as_ref(), "bar");
    }

    fn xenium_zarr_path() -> Option<std::path::PathBuf> {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("docs/temp/cells.zarr.zip");
        p.exists().then_some(p)
    }

    #[test]
    fn test_read_one_coord_file_zarr_fallback_to_0_1() {
        // Zarr fixture has cell_centroid_{x,y}; passing only Visium names should
        // error in the underlying reader and trigger fallback to indices [0, 1].
        let Some(p) = xenium_zarr_path() else { return };
        let names: Vec<Box<str>> = vec!["pxl_row_in_fullres".into(), "pxl_col_in_fullres".into()];
        let r = read_one_coord_file(p.to_str().unwrap(), &[], &names, None).unwrap();
        assert_eq!(r.mat.ncols(), 2);
        assert!(r.mat.nrows() > 0);
        assert_eq!(r.cols[0].as_ref(), "x");
        assert_eq!(r.cols[1].as_ref(), "y");
    }

    #[test]
    fn test_read_one_coord_file_zarr_match_no_fallback() {
        let Some(p) = xenium_zarr_path() else { return };
        let names: Vec<Box<str>> = vec!["cell_centroid_x".into(), "cell_centroid_y".into()];
        let r = read_one_coord_file(p.to_str().unwrap(), &[], &names, None).unwrap();
        assert_eq!(r.mat.ncols(), 2);
        assert_eq!(r.cols[0].as_ref(), "cell_centroid_x");
        assert_eq!(r.cols[1].as_ref(), "cell_centroid_y");
    }
}
