use clap::Args;

mod convert;
mod split;
mod squeeze;
mod subsample;
mod subset;

pub use convert::*;
pub use split::*;
pub use squeeze::*;
pub use subsample::*;
pub use subset::*;

#[derive(Args, Debug)]
pub struct ReorderRowsArgs {
    /// Data file -- either `.zarr` or `.h5`
    pub data_file: Box<str>,

    /// Row/feature name file (name per each line; `.tsv.gz` or `.tsv`)
    #[arg(short, long, required = true)]
    pub row_file: Box<str>,

    /// output header
    #[arg(short, long, required = true)]
    pub output: Box<str>,
}

#[derive(Args, Debug)]
pub struct SubsetColumnsArgs {
    /// data file -- either `.zarr` or `.h5`
    pub data_file: Box<str>,

    /// column indices to take: e.g., `0,1,2,3`
    #[arg(short = 'i', long, value_delimiter = ',')]
    pub column_indices: Option<Vec<usize>>,

    /// column name file where each line is a column name
    #[arg(short = 'f', long)]
    pub name_file: Option<Box<str>>,

    /// delimiter for base-key extraction (e.g., '@' to match "ACGT-1@batch" with "ACGT-1")
    #[arg(short = 'd', long, default_value = "@")]
    pub delimiter: char,

    /// enable prefix matching (stored name is prefix of query or vice versa)
    #[arg(long, default_value_t = true)]
    pub allow_prefix: bool,

    /// squeeze
    #[arg(long, default_value_t = false)]
    pub do_squeeze: bool,

    /// minimum number of non-zero cutoff for rows
    #[arg(long, default_value_t = 1)]
    pub row_nnz_cutoff: usize,

    /// minimum number of non-zero cutoff for columns
    #[arg(long, default_value_t = 1)]
    pub column_nnz_cutoff: usize,

    /// output file
    #[arg(short, long, required = true)]
    pub output: Box<str>,
}

#[derive(Args, Debug)]
pub struct SubsetRowsArgs {
    /// data file -- either `.zarr` or `.h5`
    pub data_file: Box<str>,

    /// row indices to take: e.g., `0,1,2,3`
    #[arg(short = 'i', long, value_delimiter = ',')]
    pub row_indices: Option<Vec<usize>>,

    /// row name file where each line is a row name
    #[arg(short = 'f', long)]
    pub name_file: Option<Box<str>>,

    /// delimiter for base-key extraction (e.g., '@' to match "gene@batch" with "gene")
    #[arg(short = 'd', long, default_value = "@")]
    pub delimiter: char,

    /// enable prefix matching (stored name is prefix of query or vice versa)
    #[arg(long, default_value_t = true)]
    pub allow_prefix: bool,

    /// squeeze
    #[arg(long, default_value_t = false)]
    pub do_squeeze: bool,

    /// minimum number of non-zero cutoff for rows
    #[arg(long, default_value_t = 1)]
    pub row_nnz_cutoff: usize,

    /// minimum number of non-zero cutoff for columns
    #[arg(long, default_value_t = 1)]
    pub column_nnz_cutoff: usize,

    /// output file
    #[arg(short, long, required = true)]
    pub output: Box<str>,
}

#[derive(Args, Debug)]
#[command(about)]
pub struct RunSqueezeArgs {
    /// data files -- either `.zarr` or `.h5`
    #[arg(required = true, value_delimiter = ',')]
    pub data_files: Vec<Box<str>>,

    /// number of non-zero cutoff for rows
    #[arg(short, long, default_value = "0")]
    pub row_nnz_cutoff: usize,

    /// number of non-zero cutoff for columns
    #[arg(short, long, default_value = "0")]
    pub column_nnz_cutoff: usize,

    /// Cells per rayon job. Omit for auto-scaling by feature count.
    #[arg(long)]
    pub block_size: Option<usize>,

    /// keep data on the streaming read path instead of preloading it
    ///
    /// `default_value_t = true` with no `SetFalse` action used to make this
    /// flag decorative: there was no way to turn preloading off at all.
    #[arg(
        long = "no-preload",
        alias = "no-preload-data",
        default_value_t = true,
        action = clap::ArgAction::SetFalse,
        help = "Skip preloading; stream reads instead (preloading is the default)",
        long_help = "Skip preloading column data into memory before squeezing.\n\
                     Preloading (the default) is faster but costs 12 bytes per\n\
                     non-zero, and is skipped automatically over the\n\
                     LEGUME_PRELOAD_BUDGET_BYTES budget."
    )]
    pub preload: bool,

    /// show nnz histogram before squeezing
    #[arg(
        long,
        default_value_t = false,
        help = "Show ASCII histogram of row/column nnz distributions",
        long_help = "Display log1p-transformed ASCII histograms.\n\
                     They cover row and column non-zero counts, before squeezing.\n\
                     Use them to pick appropriate cutoff values."
    )]
    pub show_histogram: bool,

    /// save histogram data to files
    #[arg(
        long,
        help = "Output file prefix for saving histogram data",
        long_help = "Save histogram data to {prefix}.row_nnz.txt and {prefix}.col_nnz.txt files.\n\
                     Each file contains nnz counts that can be used for further analysis."
    )]
    pub save_histogram: Option<Box<str>>,

    /// dry run - only show histograms without performing squeeze
    #[arg(
        long,
        default_value_t = false,
        help = "Preview mode - show histograms without squeezing",
        long_help = "Only display histograms and statistics without actually performing the squeeze operation.\n\
                     Useful for determining appropriate cutoff values."
    )]
    pub dry_run: bool,

    /// interactive mode - prompt user after showing histogram
    #[arg(
        short,
        long,
        default_value_t = false,
        help = "Interactive mode - ask for confirmation after showing histogram",
        long_help = "Show histogram and prompt user to proceed, adjust cutoffs, or cancel.\n\
                     Automatically enables --show-histogram."
    )]
    pub interactive: bool,

    /// auto cutoff - apply the k-means-suggested cutoff without prompting
    #[arg(
        long,
        default_value_t = false,
        help = "Apply the k-means-suggested nnz cutoff headlessly (no prompt)",
        long_help = "Resolve row and column cutoffs automatically, then squeeze.\n\
                     The cutoffs come from a 2-means split of log(1+nnz). No prompt is shown.\n\
                     \n\
                     Explicit --row-nnz-cutoff and --column-nnz-cutoff still win,\n\
                     per dimension. So you can pin one axis and auto the other.\n\
                     Combine with --dry-run to preview the cutoffs without writing."
    )]
    pub auto_cutoff: bool,

    /// output file for squeezed data
    #[arg(
        short,
        long,
        help = "Output file for squeezed data",
        long_help = "Save squeezed data to a new file instead of modifying in-place.\n\
                     With multiple inputs,\n\
                     all files will be squeezed and merged into {output}.{backend}.\n\
                     If not specified,\n\
                     modifies files in-place (requires confirmation in interactive mode)."
    )]
    pub output: Option<Box<str>>,

    /// row alignment strategy for merging multiple files
    #[arg(
        long,
        value_enum,
        default_value = "common",
        help = "Row alignment strategy when merging multiple files",
        long_help = "How to align rows across files after squeezing:\n\
                     - common: Keep only rows present in ALL files (intersection)\n\
                     - union: Keep rows present in ANY file (union, fills missing with zeros)"
    )]
    pub row_align: RowAlignMode,
}

#[derive(clap::ValueEnum, Clone, Debug, PartialEq)]
#[clap(rename_all = "lowercase")]
pub enum RowAlignMode {
    Common,
    Union,
}

/// Stream a column selection (with an optional ASCENDING row filter) into a
/// fresh backend, without ever materialising the survivors.
///
/// The shape `split` and `subsample` share: pick columns, optionally keep an
/// ascending subset of rows, write a new file. Both used to read every
/// surviving triplet into one `Vec` and hand it to the sorting triplet writer —
/// 24 B/nnz of residency that OOMs at imaging scale. Here the exact output nnz
/// comes from the resident indptr (or one counting pass when rows are
/// filtered), and survivors flow through the validated streaming writer in
/// byte-budgeted slabs.
///
/// `row_filter`, when given, must be ascending: the renumbering is then
/// monotone, so within-column row order survives and no per-column sort is
/// needed. Both callers sample or select ascending; the debug assert keeps the
/// next caller honest rather than silently writing unsorted columns (which the
/// streaming writer would refuse anyway).
pub(crate) fn stream_column_selection(
    data: &dyn crate::sparse_io::SparseIo<IndexIter = Vec<usize>>,
    selected_columns: &[usize],
    row_filter: Option<&[usize]>,
    out_row_names: &[Box<str>],
    out_col_names: &[Box<str>],
    file_out: &str,
    backend_out: &crate::sparse_io::SparseIoBackend,
) -> anyhow::Result<(usize, usize, usize)> {
    use crate::sparse_io::*;

    let nrow_full = data
        .num_rows()
        .ok_or_else(|| anyhow::anyhow!("backend has no `nrow`"))?;

    // Old-row → new-row, monotone by construction.
    let row_map: Option<Vec<Option<u64>>> = row_filter.map(|keep| {
        debug_assert!(
            keep.windows(2).all(|w| w[0] < w[1]),
            "row filter must ascend"
        );
        let mut map = vec![None; nrow_full];
        for (new, &old) in keep.iter().enumerate() {
            map[old] = Some(new as u64);
        }
        map
    });
    let out_nrow = row_filter.map_or(nrow_full, <[usize]>::len);
    let out_ncol = selected_columns.len();

    // Exact per-output-column nnz. Free from the indptr when every row
    // survives; one counting pass — counts, never entries — otherwise.
    let per_col_nnz: Vec<u64> = match &row_map {
        None => selected_columns
            .iter()
            .map(|&c| {
                data.column_nnz(c)
                    .ok_or_else(|| anyhow::anyhow!("no indptr entry for column {c}"))
            })
            .collect::<anyhow::Result<_>>()?,
        Some(map) => {
            // Block reads here too; counts only, never entries.
            let mut counts = vec![0u64; out_ncol];
            let coarse = matrix_util::utils::generate_minibatch_intervals(out_ncol, 0, Some(8192));
            for (lb, ub) in coarse {
                let (_, _, triplets) =
                    data.read_triplets_by_columns(selected_columns[lb..ub].to_vec())?;
                for (r, c_local, _) in triplets {
                    if map[r as usize].is_some() {
                        counts[lb + c_local as usize] += 1;
                    }
                }
            }
            counts
        }
    };
    let nnz: u64 = per_col_nnz.iter().sum();

    let mut out = create_sparse_streaming_empty(Some(file_out), Some(backend_out))?;
    out.begin_streaming_csc((out_nrow, out_ncol, nnz as usize))?;

    const SLAB_BUDGET_BYTES: usize = 256 << 20;
    let blocks = matrix_util::utils::byte_budget_intervals(&per_col_nnz, SLAB_BUDGET_BYTES, 24);

    let t_stream = std::time::Instant::now();
    let mut nnz_offset = 0u64;
    for (lb, ub) in blocks {
        // ONE block read per slab, never a read per column: a single-column
        // read pays the cached-subset machinery per call, and at hundreds of
        // thousands of columns that made this path slower than the memory wall
        // it replaced. The block read returns LOCAL column ids for the
        // requested set, rows ascending within each column — the writer's
        // invariant already.
        log::info!(
            "slab {lb}..{ub}: reading (t={:.0}s)",
            t_stream.elapsed().as_secs_f32()
        );
        let (_, _, triplets) = data.read_triplets_by_columns(selected_columns[lb..ub].to_vec())?;

        let n_block = ub - lb;
        let mut per_col: Vec<Vec<(u64, f32)>> = vec![Vec::new(); n_block];
        for (r, c_local, x) in triplets {
            let kept = match &row_map {
                None => Some(r),
                Some(map) => map[r as usize],
            };
            if let Some(new_r) = kept {
                per_col[c_local as usize].push((new_r, x));
            }
        }
        let mut local_colptr = Vec::with_capacity(n_block);
        let mut row_indices = Vec::new();
        let mut values = Vec::new();
        for entries in &per_col {
            local_colptr.push(row_indices.len() as u64);
            for &(r, x) in entries {
                row_indices.push(r);
                values.push(x);
            }
        }
        out.append_csc_slab(lb as u64, nnz_offset, &local_colptr, &row_indices, &values)?;
        nnz_offset += values.len() as u64;
    }
    log::info!("slabs done (t={:.0}s)", t_stream.elapsed().as_secs_f32());

    out.finalize_streaming_csc()?;
    out.build_csr_from_csc_streaming()?;
    log::info!(
        "transpose done (t={:.0}s)",
        t_stream.elapsed().as_secs_f32()
    );
    out.register_row_names_vec(out_row_names);
    out.register_column_names_vec(out_col_names);
    Ok((out_nrow, out_ncol, nnz as usize))
}
