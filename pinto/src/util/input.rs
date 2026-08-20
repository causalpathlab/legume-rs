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
                     If none of these names is present, the run stops and prints the\n\
                     file's first line, so you can see what it does have. Nothing is\n\
                     read by position on a guess: any two numeric columns would pass\n\
                     as coordinates, and a table with counts where the coordinates\n\
                     were expected would put every cell in one spot without\n\
                     complaining.\n\
                     \n\
                     Say which columns instead. Use this flag for named ones, or\n\
                     --coord-column-indices for positions. The classic spot layout\n\
                     has no header, and its coordinates are --coord-column-indices 4,5."
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
        default_value_t = 5,
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
                     On by default. It roughly doubles runtime and peak memory,\n\
                     because it roughly doubles the pairs. Set it to 0 for the\n\
                     spatial pairs alone, which is also the way to reproduce a run\n\
                     made before this existed.\n\
                     \n\
                     lc models every pair. cage's chain training only samples pairs\n\
                     that share a super-cell at each chain level and a batch, so\n\
                     distant expression pairs reach its pair outputs and propensity\n\
                     but largely not its chain loss; the dropped counts are logged."
    )]
    pub knn_expr: usize,

    #[arg(
        long,
        value_enum,
        default_value_t = KnnExprScope::Global,
        help = "Whether expression neighbours may cross disconnected tissue",
        long_help = "Where --knn-expr searches for expression neighbours.\n\
                     \n\
                     global (default): anywhere on the slide. Asking for expression\n\
                     neighbours at all means wanting a reference for what a cell type\n\
                     looks like, and the strongest such reference is drawn from every\n\
                     piece of tissue rather than one. Section effects are handled by\n\
                     correction rather than by refusing to look: the projection these\n\
                     neighbours come from is batch-corrected when batches are known,\n\
                     so pair --auto-batch with this on a multi-section slide.\n\
                     \n\
                     within: inside each disconnected piece of the\n\
                     spatial graph. Separate sections, or the cores of a tissue\n\
                     microarray, are usually separate samples, and this keeps every\n\
                     pair inside one of them. Choose it when a pair spanning two\n\
                     samples is not something the model should be shown at all.\n\
                     \n\
                     On a microarray the two differ sharply: searching globally,\n\
                     nearly every neighbour found sits in a different core."
    )]
    pub knn_expr_scope: KnnExprScope,

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

/// Where `--knn-expr` looks for expression neighbours.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum KnnExprScope {
    /// Inside each disconnected piece of the spatial graph.
    Within,
    /// Anywhere on the slide.
    Global,
}

/// Label each disconnected piece of the spatial graph as its own batch.
///
/// Pieces that are fragments of a larger one are folded into it first, so the
/// count returned is the number of BATCHES, which is at most the number of
/// components. Leaves the labels untouched when the graph is connected.
pub fn auto_batch_from_components(
    graph: &KnnGraph,
    coordinates: &Mat,
    batch_membership: &mut Vec<Box<str>>,
    fold_fragments: bool,
) -> usize {
    let (labels, n_components) = connected_components(graph);
    if n_components <= 1 {
        return n_components;
    }
    // Folding reasons about physical geometry: box containment in the
    // coordinate frame, with a slack measured from the graph's own edge
    // lengths. Without real coordinates both halves of that argument are in
    // different, arbitrary units (a 2-D layout vs projection-space
    // distances), so a caller in expression mode asks for no folding and
    // every component stays its own batch.
    if !fold_fragments {
        for (cell, &l) in labels.iter().enumerate() {
            batch_membership[cell] = format!("component_{l}").into_boxed_str();
        }
        return n_components;
    }

    // A connected component stands in for a sample, but it is not one. The
    // spatial graph breaks wherever the tissue does, so a single physical piece
    // arrives as one large component plus a scatter of small ones. Treating each
    // as a batch estimates an effect per fragment, on far too few cells.
    //
    // A fragment is recognised by WHERE it is, not by how small it is: it lies
    // inside the piece it broke off from, while two genuinely different pieces
    // sit far apart. So a component is folded into another when its centre falls
    // within that component's extent, widened by a few cell diameters to cover
    // the gap that split them.
    //
    // The widening is measured, not chosen: the graph's own edge lengths are the
    // cell-to-cell spacing. It only has to be small against the distance between
    // pieces, and on real tissue those differ by more than an order of
    // magnitude, so the exact multiple does not change the answer.
    let spacing = median_edge_length(graph).unwrap_or(0.0);
    let slack = 3.0 * spacing;

    // Not a clamp: `0.clamp(1, 2)` is 1, and the coordinate lookup below would
    // then index a column that is not there.
    if coordinates.ncols() == 0 {
        return n_components;
    }
    // Every coordinate column, deliberately including the per-file batch
    // pseudo-coordinate a multi-file load appends: it is what separates two
    // files sharing one (x, y) frame, and judging containment without it
    // folds them into a single batch and silently skips batch correction.
    let dims = coordinates.ncols();
    let mut lo = vec![f64::INFINITY; n_components * dims];
    let mut hi = vec![f64::NEG_INFINITY; n_components * dims];
    let mut sum = vec![0.0f64; n_components * dims];
    let mut size = vec![0usize; n_components];
    let mut rep = vec![usize::MAX; n_components];
    for (cell, &l) in labels.iter().enumerate() {
        rep[l] = rep[l].min(cell);
        size[l] += 1;
        for d in 0..dims {
            let v = coordinates[(cell, d)] as f64;
            lo[l * dims + d] = lo[l * dims + d].min(v);
            hi[l * dims + d] = hi[l * dims + d].max(v);
            sum[l * dims + d] += v;
        }
    }

    // Fold into the SMALLEST enclosing piece, so a fragment joins the core it
    // sits in rather than any larger box that happens to span it.
    let mut into: Vec<usize> = (0..n_components).collect();
    for c in 0..n_components {
        let mut best: Option<(f64, f64, usize, usize)> = None;
        for other in 0..n_components {
            if other == c || size[other] <= size[c] {
                continue;
            }
            let centre = |d: usize| sum[c * dims + d] / size[c].max(1) as f64;
            let inside = (0..dims).all(|d| {
                centre(d) >= lo[other * dims + d] - slack
                    && centre(d) <= hi[other * dims + d] + slack
            });
            if !inside {
                continue;
            }
            let extent: f64 = (0..dims)
                .map(|d| hi[other * dims + d] - lo[other * dims + d])
                .sum();
            // Ties break on the candidate's own corner, then on its smallest
            // member cell — a stable identity — never on `other`, whose
            // numbering comes from a parallel union-find and varies per run.
            let corner = lo[other * dims];
            let key = (extent, corner, rep[other]);
            if best.is_none_or(|(e, c, r, _)| (extent, corner, rep[other]) < (e, c, r)) {
                best = Some((key.0, key.1, key.2, other));
            }
        }
        if let Some((_, _, _, target)) = best {
            into[c] = target;
        }
    }
    // A fragment may point at a piece that is itself a fragment; follow through.
    for c in 0..n_components {
        let mut t = into[c];
        let mut hops = 0;
        while into[t] != t && hops < n_components {
            t = into[t];
            hops += 1;
        }
        into[c] = t;
    }

    let kept: std::collections::BTreeSet<usize> = into.iter().copied().collect();
    *batch_membership = labels
        .iter()
        .map(|&l| format!("cc_{}", into[l]).into_boxed_str())
        .collect();
    let n_batches = kept.len();
    if n_batches < n_components {
        // Sizes, not just a count: a run that folded a 4000-cell "fragment"
        // into its neighbour has merged two samples, and a count alone hides it.
        let mut folded: Vec<usize> = (0..n_components)
            .filter(|&c| into[c] != c)
            .map(|c| size[c])
            .collect();
        folded.sort_unstable_by(|a, b| b.cmp(a));
        info!(
            "Auto-detected {} spatial components; {} lie inside another and were \
             folded into it, giving {} batches. Folded piece sizes, largest first: {:?}",
            n_components,
            n_components - n_batches,
            n_batches,
            &folded[..folded.len().min(10)]
        );
    } else {
        info!(
            "Auto-detected {} spatial components as batches",
            n_components
        );
    }
    n_batches
}

/// Median length of the graph's edges, which on a spatial graph is the
/// cell-to-cell spacing. `None` when the graph carries no usable lengths.
fn median_edge_length(graph: &KnnGraph) -> Option<f64> {
    let mut d: Vec<f32> = graph
        .distances
        .iter()
        .copied()
        .filter(|x| x.is_finite() && *x > 0.0)
        .collect();
    if d.is_empty() {
        return None;
    }
    let mid = d.len() / 2;
    d.select_nth_unstable_by(mid, f32::total_cmp);
    Some(d[mid] as f64)
}

/// Read a single coordinate file with a name-then-index fallback.
///
/// Coordinates are the columns the caller named or numbered. Nothing is
/// inferred: reading by position is a guess about the file, and a wrong guess
/// does not fail, because any two numeric columns pass as coordinates.
///
/// When neither selector resolves, this errors and quotes the file's first
/// line back, since the caller cannot fix it without seeing what the file
/// holds.
pub fn read_one_coord_file(
    coord_file: &str,
    user_indices: &[usize],
    user_names: &[Box<str>],
    header_in_coord: Option<usize>,
) -> anyhow::Result<MatWithNames<Mat>> {
    let ext = file_ext(coord_file)?;
    let is_zarr = coord_file.contains(".zarr");

    // One index is a typo for two, not a request for 1-D geometry: it also
    // bypasses the too-few-columns fallback below (which only arms itself when
    // NO index was given), so a spatial graph would be built on a line with no
    // error anywhere downstream.
    anyhow::ensure!(
        user_indices.is_empty() || user_indices.len() >= 2,
        "--coord-column-indices selects {} column; coordinates need at least two",
        user_indices.len()
    );

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
        // No guessing. Coordinates are whichever columns the caller named or
        // numbered; if neither resolves, say so and stop.
        //
        // This used to fall back to columns 4 and 5, the positions they occupy
        // in one particular vendor layout. That is a guess about the file, and
        // when it is wrong it does not fail: any two numeric columns are
        // accepted as coordinates, so a table with per-cell counts in those
        // positions puts every cell in the same place and the run looks
        // healthy. A wrong answer that announces nothing is worse than an
        // error, and the caller already has an exact way to say what it meant.
        let mut hint = String::new();
        if !is_zarr {
            if let Ok(first) = first_line_fields(coord_file) {
                hint = format!(" The file's first line reads {first:?}.");
            }
        }
        let matched = initial.as_ref().map_or(0, |r| r.mat.ncols());
        return Err(anyhow::anyhow!(
            "coord file '{}' resolved {matched} of the coordinate columns asked for \
             {:?}, and two are needed.{} \
             Pass --coord-column-names with the names it does have, or \
             --coord-column-indices with their 0-based positions. For the classic \
             spot layout that is --coord-column-indices 4,5.",
            coord_file,
            user_names,
            hint,
        ));
    }

    if let Ok(r) = initial.as_ref() {
        if r.mat.ncols() > 2 {
            warn!(
                "coord file '{}' resolved {} coordinate columns {:?}; the graph will be \
                 built in that many dimensions. If two naming conventions both matched, \
                 pass --coord-column-names with just the pair you mean",
                coord_file,
                r.mat.ncols(),
                r.cols,
            );
        }
    }
    initial
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
    // Unquote with the tokenizer's own rule first: a fully quoted numeric
    // field ("100.5") must read as numeric, or a quoted headerless file gets
    // its first data row swallowed as a header.
    let any_non_numeric_after_col0 = first_line
        .split(delimiters.as_ref())
        .skip(1)
        .map(matrix_util::common_io::unquote_field)
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
/// The first line of a delimited file, split into fields.
///
/// Only ever quoted back to the caller in an error, so they can see what the
/// file actually holds. It makes no claim about whether that line is a header.
fn first_line_fields(path: &str) -> anyhow::Result<Vec<Box<str>>> {
    use std::io::BufRead;
    let mut first = String::new();
    // gz-aware, because a coord file often is; opening it raw would fail here
    // and quietly drop the one hint this error promises.
    matrix_util::common_io::open_buf_reader(path)?.read_line(&mut first)?;
    Ok(first
        .trim_end_matches(['\n', '\r'])
        .split(['\t', ',', ' '])
        .map(|f| f.trim().trim_matches('"').to_string().into_boxed_str())
        .filter(|f| !f.is_empty())
        .collect())
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
    fn unresolved_coordinate_columns_are_reported_with_the_file_s_own_fields() {
        // Reading by position is a guess about the file, and a wrong guess does
        // not fail: any two numeric columns pass as coordinates. So nothing is
        // guessed. The error quotes the first line back, because the caller
        // cannot fix this without knowing what the file actually holds.
        let f = write_csv(
            "barcode,in_tissue,array_row,array_col,my_x,my_y\n\
             cell_a,1,0,0,100.5,200.5\n\
             cell_b,1,1,1,300.5,400.5\n",
        );
        let names: Vec<Box<str>> = vec!["pxl_row_in_fullres".into(), "pxl_col_in_fullres".into()];
        let err = match read_one_coord_file(f.path().to_str().unwrap(), &[], &names, None) {
            Ok(_) => panic!("columns that resolve to nothing must not be guessed at"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("my_x"),
            "the file's own fields guide the fix: {err}"
        );
        assert!(
            err.contains("--coord-column-indices"),
            "and say how to say it: {err}"
        );
    }

    #[test]
    fn a_headerless_positional_file_is_read_when_the_caller_says_which_columns() {
        // The classic spot layout has no header at all, so there are no names
        // to match. Naming the positions is the whole answer.
        let f = write_csv(
            "cell_a,1,0,0,100.5,200.5\n\
             cell_b,1,1,1,300.5,400.5\n",
        );
        let names: Vec<Box<str>> = vec!["pxl_row_in_fullres".into()];
        let r = read_one_coord_file(f.path().to_str().unwrap(), &[4, 5], &names, None).unwrap();
        assert_eq!(r.mat.ncols(), 2);
        assert_eq!(r.mat[(0, 0)], 100.5);
        assert_eq!(r.mat[(0, 1)], 200.5);
        assert_eq!(r.mat[(1, 0)], 300.5);
        assert_eq!(r.rows[0].as_ref(), "cell_a");
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
    fn zarr_coordinates_that_match_no_requested_name_are_reported() {
        // This used to assert a fallback to indices [0, 1]. There is no
        // fallback now, so it asserts the replacement contract. It is skipped
        // without the fixture, which is why the old version stayed green for
        // as long as it did after the behaviour under it changed.
        let Some(p) = xenium_zarr_path() else { return };
        let names: Vec<Box<str>> = vec!["pxl_row_in_fullres".into(), "pxl_col_in_fullres".into()];
        match read_one_coord_file(p.to_str().unwrap(), &[], &names, None) {
            Ok(_) => panic!("names that match nothing must not be guessed at"),
            Err(e) => {
                let m = e.to_string();
                assert!(m.contains("coordinate columns"), "{m}");
            }
        }
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
