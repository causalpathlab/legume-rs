use super::run_squeeze_if_needed;
use crate::hdf5_io::*;
use crate::sparse_io::*;
use data_beans::zarr_io::*;

use clap::Args;
use log::info;
use matrix_util::common_io::*;

#[derive(Args, Debug)]
pub struct FromFragmentsArgs {
    #[arg(
        help = "Fragments TSV file (plain, .gz, or bgzipped .gz/.bgz)",
        long_help = "Per-fragment records, one per line, tab-separated:\n\
                     chr<TAB>start<TAB>end<TAB>barcode[<TAB>count]\n\
                     \n\
                     Lines starting with '#' are skipped (e.g. the cellranger-arc\n\
                     header block). Both plain gzip and bgzip-compressed files\n\
                     are supported (MultiGzDecoder reads concatenated blocks)."
    )]
    pub fragments: Box<str>,

    #[arg(long, value_enum, default_value = "zarr")]
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
        help = "BED file of peaks (chr<TAB>start<TAB>end[...])",
        long_help = "When provided, fragments are aggregated into these regions\n\
                     instead of fixed-width genome tiles. Lines starting with '#',\n\
                     'track', or 'browser' are skipped. Row names are formatted as\n\
                     `chr:start-end`. Peaks may overlap."
    )]
    pub peaks: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = 5000,
        help = "Tile width in bp when --peaks is not provided",
        long_help = "Tile the genome into fixed-width bins on the fly. Bin (i)\n\
                     spans [i*bin_size, (i+1)*bin_size). Each fragment contributes\n\
                     to every bin it overlaps. Set 0 to disable (then --peaks is required)."
    )]
    pub bin_size: u64,

    #[arg(
        long,
        help = "Barcode whitelist file (one barcode per line, plain or .gz)",
        long_help = "When provided, fragments whose barcode is not in this list\n\
                     are skipped and the column order matches the whitelist."
    )]
    pub barcodes: Option<Box<str>>,

    #[arg(
        long,
        default_value_t = false,
        help = "Use the count column (col 5) instead of contributing 1 per fragment",
        long_help = "By default each fragment adds 1 to the (feature, cell) entry.\n\
                     With this flag, the Tn5 insertion count in column 5 is used\n\
                     (missing/unparsable counts fall back to 1)."
    )]
    pub use_count: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Decompress the fragments file in-memory and parse in parallel",
        long_help = "By default the file is streamed line-by-line on a single \
                     thread. With this flag, the whole (decompressed) file is \
                     loaded into a byte buffer once and parsing/aggregation is \
                     split across rayon workers. Recommended for fast SSDs and \
                     for large inputs on machines with enough RAM \
                     (~3-10x of the compressed file size). \
                     Output is identical to the serial path."
    )]
    pub preload_data: bool,

    #[arg(long, default_value_t = false)]
    pub do_squeeze: bool,

    #[arg(long, default_value_t = 1)]
    pub row_nnz_cutoff: usize,

    #[arg(long, default_value_t = 1)]
    pub column_nnz_cutoff: usize,

    #[arg(long)]
    pub block_size: Option<usize>,
}

/// Build backend from a (sc)ATAC/histone fragments TSV file.
///
/// Streams the (potentially bgzipped) fragments file once. Each line is
/// `chr<TAB>start<TAB>end<TAB>barcode[<TAB>count]`. Fragments are
/// aggregated into a (feature x cell) sparse matrix where features are
/// either user-supplied peaks (`--peaks`) or fixed-width genome tiles
/// (`--bin-size`).
pub fn run_build_from_fragments(args: &FromFragmentsArgs) -> anyhow::Result<()> {
    use rustc_hash::FxHashMap as HashMap;
    use std::io::BufRead;

    if args.peaks.is_none() && args.bin_size == 0 {
        anyhow::bail!("must provide either --peaks <bed> or --bin-size > 0");
    }

    let effective_output = apply_zip_flag(&args.output, args.zip);
    let (backend, backend_file) =
        resolve_backend_file(&effective_output, Some(args.backend.clone()))?;

    if std::path::Path::new(backend_file.as_ref()).exists() {
        info!("Removing existing backend file: {}", &backend_file);
        remove_file(&backend_file)?;
    }

    ////////////////////////////////////
    //  1. Build the feature catalog  //
    ////////////////////////////////////
    //
    // Peaks mode: load all peaks up front, sort per-chromosome.
    // Bin mode: discover bins lazily as fragments stream in.
    //
    // Both modes key on the normalized chromosome name (no `chr` prefix) so
    // that a BED using `1` and a fragments file using `chr1` still match.
    // Display row names preserve the input's original convention — for peaks
    // mode that's whatever the BED gave us; for bin mode, the first chr
    // string seen for each canonical key wins.
    let use_peaks = args.peaks.is_some();
    let bin_size = args.bin_size as i64;

    let mut row_names: Vec<Box<str>> = Vec::new();

    // peaks mode lookup state (key: stripped chr)
    let mut peak_chr_idx: HashMap<Box<str>, usize> = Default::default();
    let mut peaks_by_chr: Vec<Vec<(i64, i64, u64)>> = Vec::new();
    let mut peak_max_width: Vec<i64> = Vec::new();

    // bin mode lookup state: stripped_chr -> (bin_index -> row_idx)
    let mut bin_row_idx: HashMap<Box<str>, HashMap<i64, u64>> = Default::default();
    // stripped_chr -> the first display string we saw for it
    let mut bin_display_chr: HashMap<Box<str>, Box<str>> = Default::default();

    if let Some(peaks_file) = args.peaks.as_ref() {
        info!("Loading peaks from {}", peaks_file);
        let buf = open_buf_reader(peaks_file.as_ref())?;
        let mut tmp: HashMap<Box<str>, Vec<(i64, i64, u64)>> = Default::default();
        for line in buf.lines() {
            let line = line?;
            let s = line.trim();
            if s.is_empty()
                || s.starts_with('#')
                || s.starts_with("track")
                || s.starts_with("browser")
            {
                continue;
            }
            let mut it = s.split('\t');
            let chr = match it.next() {
                Some(c) if !c.is_empty() => c,
                _ => continue,
            };
            let start: i64 = match it.next().and_then(|x| x.parse().ok()) {
                Some(v) => v,
                None => continue,
            };
            let end: i64 = match it.next().and_then(|x| x.parse().ok()) {
                Some(v) => v,
                None => continue,
            };
            if end <= start {
                continue;
            }
            let row_idx = row_names.len() as u64;
            row_names.push(format!("{}:{}-{}", chr, start, end).into_boxed_str());
            tmp.entry(chr_key(chr).into())
                .or_default()
                .push((start, end, row_idx));
        }
        let total_peaks = row_names.len();
        for (chr, mut v) in tmp.into_iter() {
            v.sort_unstable_by_key(|&(s, _, _)| s);
            let max_width = v.iter().map(|&(s, e, _)| e - s).max().unwrap_or(0);
            peak_chr_idx.insert(chr, peaks_by_chr.len());
            peaks_by_chr.push(v);
            peak_max_width.push(max_width);
        }
        info!(
            "Loaded {} peaks across {} chromosomes",
            total_peaks,
            peaks_by_chr.len()
        );
    }

    /////////////////////////////////////
    //  2. Optional barcode whitelist  //
    /////////////////////////////////////
    let mut barcode_idx: HashMap<Box<str>, u64> = Default::default();
    let mut col_names: Vec<Box<str>> = Vec::new();
    let whitelist_only = args.barcodes.is_some();
    if let Some(bc_file) = args.barcodes.as_ref() {
        info!("Loading barcode whitelist from {}", bc_file);
        let buf = open_buf_reader(bc_file.as_ref())?;
        for line in buf.lines() {
            let line = line?;
            let s = line.trim();
            if s.is_empty() || s.starts_with('#') {
                continue;
            }
            // 10x barcodes.tsv may have a "-1" suffix; keep as-is to match the
            // fragments file convention.
            let bc: Box<str> = s.into();
            if !barcode_idx.contains_key(bc.as_ref()) {
                let id = col_names.len() as u64;
                col_names.push(bc.clone());
                barcode_idx.insert(bc, id);
            }
        }
        info!("Loaded {} whitelist barcodes", col_names.len());
        if col_names.is_empty() {
            anyhow::bail!("barcode whitelist is empty: {}", bc_file);
        }
    }

    //////////////////////////////////////////////////
    //  3. Stream fragments and accumulate triplets //
    //////////////////////////////////////////////////
    let mut triplet_map: HashMap<(u64, u64), f32> = Default::default();
    let mut n_total: u64 = 0;
    let mut n_skipped_bc: u64 = 0;
    let mut n_no_overlap: u64 = 0;
    let mut n_malformed: u64 = 0;

    if args.preload_data {
        run_fragments_preload_parallel(
            args,
            use_peaks,
            bin_size,
            whitelist_only,
            &peak_chr_idx,
            &peaks_by_chr,
            &peak_max_width,
            &mut barcode_idx,
            &mut col_names,
            &mut bin_row_idx,
            &mut bin_display_chr,
            &mut row_names,
            &mut triplet_map,
            &mut n_total,
            &mut n_skipped_bc,
            &mut n_no_overlap,
            &mut n_malformed,
        )?;
    } else {
        info!("Streaming fragments from {}", args.fragments);
        let reader = open_fragments_reader(args.fragments.as_ref())?;
        let prog_bar = streaming_fragments_progress();

        for line in reader.lines() {
            let line = line?;
            // skip headers / blanks
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let mut it = line.splitn(6, '\t');
            let chr = match it.next() {
                Some(c) if !c.is_empty() => c,
                _ => {
                    n_malformed += 1;
                    continue;
                }
            };
            let start: i64 = match it.next().and_then(|s| s.parse().ok()) {
                Some(v) => v,
                None => {
                    n_malformed += 1;
                    continue;
                }
            };
            let end: i64 = match it.next().and_then(|s| s.parse().ok()) {
                Some(v) => v,
                None => {
                    n_malformed += 1;
                    continue;
                }
            };
            let barcode = match it.next() {
                Some(b) if !b.is_empty() => b,
                _ => {
                    n_malformed += 1;
                    continue;
                }
            };
            if end <= start {
                n_malformed += 1;
                continue;
            }
            let cnt: f32 = if args.use_count {
                it.next().and_then(|s| s.parse::<f32>().ok()).unwrap_or(1.0)
            } else {
                1.0
            };

            n_total += 1;
            if n_total.is_multiple_of(1_000_000) {
                prog_bar.set_position(n_total);
            }

            // resolve barcode -> column index
            let col_idx = if let Some(&i) = barcode_idx.get(barcode) {
                i
            } else if whitelist_only {
                n_skipped_bc += 1;
                continue;
            } else {
                let i = col_names.len() as u64;
                let bc: Box<str> = barcode.into();
                col_names.push(bc.clone());
                barcode_idx.insert(bc, i);
                i
            };

            // resolve overlapping feature row(s) — match on the stripped chr key
            let ckey = chr_key(chr);
            if start < 0 {
                n_malformed += 1;
                continue;
            }
            if use_peaks {
                let mut hit = false;
                if let Some(&chr_i) = peak_chr_idx.get(ckey) {
                    let peaks = &peaks_by_chr[chr_i];
                    let max_w = peak_max_width[chr_i];
                    // peaks at idx < lower cannot overlap because their full
                    // [start, start+max_w] fits before frag.start
                    let lower = peaks.partition_point(|p| p.0 + max_w <= start);
                    // peaks at idx >= upper have p.start >= end → cannot overlap
                    let upper = peaks.partition_point(|p| p.0 < end);
                    for &(_, pe, ridx) in &peaks[lower..upper] {
                        if pe > start {
                            *triplet_map.entry((ridx, col_idx)).or_insert(0.0) += cnt;
                            hit = true;
                        }
                    }
                }
                if !hit {
                    n_no_overlap += 1;
                }
            } else {
                // bin mode: lookup-or-insert (stripped_chr, bin_idx) → row_idx
                let chr_map = if bin_row_idx.contains_key(ckey) {
                    bin_row_idx.get_mut(ckey).unwrap()
                } else {
                    bin_row_idx.insert(ckey.into(), Default::default());
                    bin_display_chr.insert(ckey.into(), chr.into());
                    bin_row_idx.get_mut(ckey).unwrap()
                };
                let display = bin_display_chr.get(ckey).map(|s| s.as_ref()).unwrap_or(chr);
                let first = start / bin_size;
                let last = (end - 1) / bin_size;
                for b in first..=last {
                    let ridx = if let Some(&r) = chr_map.get(&b) {
                        r
                    } else {
                        let r = row_names.len() as u64;
                        let bs = b * bin_size;
                        let be = (b + 1) * bin_size;
                        row_names.push(format!("{}:{}-{}", display, bs, be).into_boxed_str());
                        chr_map.insert(b, r);
                        r
                    };
                    *triplet_map.entry((ridx, col_idx)).or_insert(0.0) += cnt;
                }
            }
        }
        prog_bar.finish_with_message(format!("{} fragments", n_total));
    }

    info!(
        "Streamed {} fragments — {} cells, {} features, {} skipped (barcode), {} no peak overlap, {} malformed",
        n_total,
        col_names.len(),
        row_names.len(),
        n_skipped_bc,
        n_no_overlap,
        n_malformed
    );

    let nrows = row_names.len();
    let ncols = col_names.len();
    if nrows == 0 {
        anyhow::bail!("no features produced; check --peaks / --bin-size and the fragments file");
    }
    if ncols == 0 {
        anyhow::bail!("no cells produced; check the barcode whitelist or input file");
    }

    let triplets: Vec<(u64, u64, f32)> = triplet_map
        .into_iter()
        .map(|((r, c), v)| (r, c, v))
        .collect();
    let nnz = triplets.len();
    info!("Built {} triplets in {} x {} matrix", nnz, nrows, ncols);

    let mut out = create_sparse_from_triplets_owned(
        triplets,
        (nrows, ncols, nnz),
        Some(&backend_file),
        Some(&backend),
    )?;
    info!("Created sparse matrix: {}", backend_file);

    out.register_row_names_vec(&row_names);
    out.register_column_names_vec(&col_names);

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

////////////////////////////////////////////////////////////////////////
//  Parallel fragments path: preload entire (decompressed) TSV into   //
//  memory, split at newline boundaries, parse + aggregate per-thread //
//  with rayon, then merge into the global triplet/row/col state.     //
//                                                                    //
//  Output is identical to the serial path. First-appearance ordering //
//  of barcodes (and bin rows) is preserved by iterating threads in   //
//  chunk order during the merge phase.                               //
////////////////////////////////////////////////////////////////////////

/// Per-thread parse/accumulate state. Triplet keys are interpreted as:
/// - peaks + whitelist: (global_peak_row, global_col_idx)
/// - peaks + open:      (global_peak_row, local_col_idx)
/// - bin   + whitelist: (local_bin_row, global_col_idx)
/// - bin   + open:      (local_bin_row, local_col_idx)
///
/// Local indices are remapped to global indices during the merge phase.
#[derive(Default)]
struct FragLocalAccum {
    bc_map: rustc_hash::FxHashMap<Box<str>, u32>,
    bc_names: Vec<Box<str>>,
    bin_map: rustc_hash::FxHashMap<Box<str>, rustc_hash::FxHashMap<i64, u32>>,
    chr_display: rustc_hash::FxHashMap<Box<str>, Box<str>>,
    bin_keys: Vec<(Box<str>, i64)>,
    triplets: rustc_hash::FxHashMap<(u64, u64), f32>,
    n_total: u64,
    n_skipped_bc: u64,
    n_no_overlap: u64,
    n_malformed: u64,
}

/// Split `bytes` into approximately `n_chunks` slices, with each cut
/// placed at the first '\n' boundary at or after the target byte
/// position. The newline character itself ends up in the *preceding*
/// chunk, so each chunk (except possibly the last) ends with '\n'.
fn fragment_chunk_ranges(bytes: &[u8], n_chunks: usize) -> Vec<(usize, usize)> {
    if bytes.is_empty() || n_chunks <= 1 {
        return vec![(0, bytes.len())];
    }
    let target = bytes.len() / n_chunks;
    let mut cuts = vec![0usize];
    for i in 1..n_chunks {
        let probe = i * target;
        if probe >= bytes.len() {
            break;
        }
        match bytes[probe..].iter().position(|&b| b == b'\n') {
            Some(off) => {
                let cut = probe + off + 1;
                if cut > *cuts.last().unwrap() && cut < bytes.len() {
                    cuts.push(cut);
                }
            }
            None => break,
        }
    }
    cuts.push(bytes.len());
    cuts.windows(2).map(|w| (w[0], w[1])).collect()
}

/// Parse a single fragments line (no trailing newline).
#[allow(clippy::type_complexity)]
fn parse_fragments_line(
    line: &[u8],
    use_count: bool,
) -> Result<Option<(&str, i64, i64, &str, f32)>, ()> {
    if line.is_empty() || line[0] == b'#' {
        return Ok(None);
    }
    // strip trailing '\r' (CRLF inputs)
    let line = if let Some(&b'\r') = line.last() {
        &line[..line.len() - 1]
    } else {
        line
    };
    if line.is_empty() {
        return Ok(None);
    }
    let mut it = line.split(|&b| b == b'\t');
    let chr_b = it.next().ok_or(())?;
    let start_b = it.next().ok_or(())?;
    let end_b = it.next().ok_or(())?;
    let barcode_b = it.next().ok_or(())?;
    let count_b = it.next();

    if chr_b.is_empty() || barcode_b.is_empty() {
        return Err(());
    }
    let chr = std::str::from_utf8(chr_b).map_err(|_| ())?;
    let barcode = std::str::from_utf8(barcode_b).map_err(|_| ())?;
    let start: i64 = std::str::from_utf8(start_b)
        .map_err(|_| ())?
        .parse()
        .map_err(|_| ())?;
    let end: i64 = std::str::from_utf8(end_b)
        .map_err(|_| ())?
        .parse()
        .map_err(|_| ())?;
    if end <= start || start < 0 {
        return Err(());
    }
    let cnt: f32 = if use_count {
        count_b
            .and_then(|s| std::str::from_utf8(s).ok())
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(1.0)
    } else {
        1.0
    };
    Ok(Some((chr, start, end, barcode, cnt)))
}

/// Drive one rayon chunk: parse and accumulate triplets locally.
#[allow(clippy::too_many_arguments)]
fn process_fragments_chunk(
    chunk: &[u8],
    use_peaks: bool,
    bin_size: i64,
    use_count: bool,
    whitelist_only: bool,
    peak_chr_idx: &rustc_hash::FxHashMap<Box<str>, usize>,
    peaks_by_chr: &[Vec<(i64, i64, u64)>],
    peak_max_width: &[i64],
    global_bc_idx: &rustc_hash::FxHashMap<Box<str>, u64>,
) -> FragLocalAccum {
    let mut acc = FragLocalAccum::default();

    for raw in chunk.split(|&b| b == b'\n') {
        let parsed = match parse_fragments_line(raw, use_count) {
            Ok(Some(t)) => t,
            Ok(None) => continue,
            Err(()) => {
                // empty/header lines were handled by the Ok(None) branch;
                // anything else that fails parsing counts as malformed,
                // but skip purely-empty bytes (a trailing \n leaves an
                // empty slice in the iterator).
                if !raw.is_empty() && raw != b"\r" {
                    acc.n_malformed += 1;
                }
                continue;
            }
        };
        let (chr, start, end, barcode, cnt) = parsed;

        acc.n_total += 1;

        // resolve barcode -> column index (global if whitelist, else local)
        let col_idx: u64 = if whitelist_only {
            match global_bc_idx.get(barcode) {
                Some(&i) => i,
                None => {
                    acc.n_skipped_bc += 1;
                    continue;
                }
            }
        } else {
            match acc.bc_map.get(barcode) {
                Some(&i) => i as u64,
                None => {
                    let i = acc.bc_names.len() as u32;
                    let owned: Box<str> = barcode.into();
                    acc.bc_names.push(owned.clone());
                    acc.bc_map.insert(owned, i);
                    i as u64
                }
            }
        };

        let ckey = chr_key(chr);

        if use_peaks {
            let mut hit = false;
            if let Some(&chr_i) = peak_chr_idx.get(ckey) {
                let peaks = &peaks_by_chr[chr_i];
                let max_w = peak_max_width[chr_i];
                let lower = peaks.partition_point(|p| p.0 + max_w <= start);
                let upper = peaks.partition_point(|p| p.0 < end);
                for &(_, pe, ridx) in &peaks[lower..upper] {
                    if pe > start {
                        *acc.triplets.entry((ridx, col_idx)).or_insert(0.0) += cnt;
                        hit = true;
                    }
                }
            }
            if !hit {
                acc.n_no_overlap += 1;
            }
        } else {
            // bin mode: per-thread bin dict with local row indices
            let chr_map = if acc.bin_map.contains_key(ckey) {
                acc.bin_map.get_mut(ckey).unwrap()
            } else {
                let ck: Box<str> = ckey.into();
                acc.chr_display
                    .entry(ck.clone())
                    .or_insert_with(|| chr.into());
                acc.bin_map.entry(ck).or_default()
            };
            let first = start / bin_size;
            let last = (end - 1) / bin_size;
            for b in first..=last {
                let local_row: u32 = match chr_map.get(&b) {
                    Some(&r) => r,
                    None => {
                        let r = acc.bin_keys.len() as u32;
                        acc.bin_keys.push((ckey.into(), b));
                        chr_map.insert(b, r);
                        r
                    }
                };
                *acc.triplets
                    .entry((local_row as u64, col_idx))
                    .or_insert(0.0) += cnt;
            }
        }
    }

    acc
}

/// Build a progress bar matching the workspace style for the serial
/// streaming path (unknown total → spinner with fragment count).
fn streaming_fragments_progress() -> indicatif::ProgressBar {
    use indicatif::{ProgressBar, ProgressStyle};
    let prog_bar = ProgressBar::new_spinner();
    prog_bar.set_style(
        ProgressStyle::with_template("{spinner} streamed {pos} fragments ({per_sec})")
            .unwrap()
            .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ "),
    );
    prog_bar.enable_steady_tick(std::time::Duration::from_millis(200));
    prog_bar
}

/// Preload-and-parse path: read decompressed bytes into memory, split
/// at newline boundaries, parse/aggregate per chunk with rayon, then
/// merge results into the caller's global state.
#[allow(clippy::too_many_arguments)]
fn run_fragments_preload_parallel(
    args: &FromFragmentsArgs,
    use_peaks: bool,
    bin_size: i64,
    whitelist_only: bool,
    peak_chr_idx: &rustc_hash::FxHashMap<Box<str>, usize>,
    peaks_by_chr: &[Vec<(i64, i64, u64)>],
    peak_max_width: &[i64],
    barcode_idx: &mut rustc_hash::FxHashMap<Box<str>, u64>,
    col_names: &mut Vec<Box<str>>,
    bin_row_idx: &mut rustc_hash::FxHashMap<Box<str>, rustc_hash::FxHashMap<i64, u64>>,
    bin_display_chr: &mut rustc_hash::FxHashMap<Box<str>, Box<str>>,
    row_names: &mut Vec<Box<str>>,
    triplet_map: &mut rustc_hash::FxHashMap<(u64, u64), f32>,
    n_total: &mut u64,
    n_skipped_bc: &mut u64,
    n_no_overlap: &mut u64,
    n_malformed: &mut u64,
) -> anyhow::Result<()> {
    use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
    use rayon::prelude::*;
    use std::io::Read;

    //////////////////////////////////////
    //  Decompress entire file to bytes //
    //////////////////////////////////////
    info!("Preloading fragments from {}", args.fragments);
    let dec_pb = ProgressBar::new_spinner();
    dec_pb.set_style(
        ProgressStyle::with_template("{spinner} decompressing... {bytes} ({bytes_per_sec})")
            .unwrap()
            .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ "),
    );
    dec_pb.enable_steady_tick(std::time::Duration::from_millis(200));
    let mut reader = open_fragments_reader(args.fragments.as_ref())?;
    let mut buf: Vec<u8> = Vec::with_capacity(1 << 24);
    // small read loop so the progress bar advances during long decompress
    let mut tmp = [0u8; 1 << 20];
    loop {
        let n = reader.read(&mut tmp)?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
        dec_pb.set_position(buf.len() as u64);
    }
    dec_pb.finish_and_clear();
    info!("Decompressed {} bytes", buf.len());

    /////////////////////////////////////////
    //  Chunk & parallel parse+accumulate  //
    /////////////////////////////////////////
    let n_threads = rayon::current_num_threads().max(1);
    let ranges = fragment_chunk_ranges(&buf, n_threads);
    info!(
        "Parsing with {} chunks across {} rayon threads",
        ranges.len(),
        n_threads
    );
    let pb_tmpl = "{bar:40} {pos}/{len} chunks ({eta})";
    let prog_bar = ProgressBar::new(ranges.len() as u64).with_style(
        ProgressStyle::with_template(pb_tmpl)
            .unwrap()
            .progress_chars("##-"),
    );
    let locals: Vec<FragLocalAccum> = ranges
        .par_iter()
        .progress_with(prog_bar.clone())
        .map(|&(lo, hi)| {
            process_fragments_chunk(
                &buf[lo..hi],
                use_peaks,
                bin_size,
                args.use_count,
                whitelist_only,
                peak_chr_idx,
                peaks_by_chr,
                peak_max_width,
                barcode_idx,
            )
        })
        .collect();
    prog_bar.finish_and_clear();
    drop(buf);

    /////////////////////////////////
    //  Merge: barcodes (col idx)  //
    /////////////////////////////////
    let mut col_remaps: Vec<Vec<u64>> = Vec::with_capacity(locals.len());
    if whitelist_only {
        // Local triplet keys already use global col idx; no remap.
        for _ in &locals {
            col_remaps.push(Vec::new());
        }
    } else {
        for local in &locals {
            let mut remap = Vec::with_capacity(local.bc_names.len());
            for bc in &local.bc_names {
                let gid = if let Some(&i) = barcode_idx.get(bc.as_ref()) {
                    i
                } else {
                    let i = col_names.len() as u64;
                    col_names.push(bc.clone());
                    barcode_idx.insert(bc.clone(), i);
                    i
                };
                remap.push(gid);
            }
            col_remaps.push(remap);
        }
    }

    //////////////////////////////
    //  Merge: bins (row idx)   //
    //////////////////////////////
    let mut row_remaps: Vec<Vec<u64>> = Vec::with_capacity(locals.len());
    if use_peaks {
        for _ in &locals {
            row_remaps.push(Vec::new());
        }
    } else {
        for local in &locals {
            let mut remap = Vec::with_capacity(local.bin_keys.len());
            for (ckey, bin_idx) in &local.bin_keys {
                // bin_display_chr keeps the first-seen display name across
                // threads, matching the serial path's behavior.
                bin_display_chr.entry(ckey.clone()).or_insert_with(|| {
                    local
                        .chr_display
                        .get(ckey.as_ref())
                        .cloned()
                        .unwrap_or_else(|| ckey.clone())
                });
                let chr_map = bin_row_idx.entry(ckey.clone()).or_default();
                let gid = if let Some(&r) = chr_map.get(bin_idx) {
                    r
                } else {
                    let r = row_names.len() as u64;
                    let display = bin_display_chr
                        .get(ckey.as_ref())
                        .map(|s| s.as_ref())
                        .unwrap_or_else(|| ckey.as_ref());
                    let bs = bin_idx * bin_size;
                    let be = (bin_idx + 1) * bin_size;
                    row_names.push(format!("{}:{}-{}", display, bs, be).into_boxed_str());
                    chr_map.insert(*bin_idx, r);
                    r
                };
                remap.push(gid);
            }
            row_remaps.push(remap);
        }
    }

    //////////////////////////////
    //  Merge: triplets + stats //
    //////////////////////////////
    for (t, local) in locals.iter().enumerate() {
        *n_total += local.n_total;
        *n_skipped_bc += local.n_skipped_bc;
        *n_no_overlap += local.n_no_overlap;
        *n_malformed += local.n_malformed;

        let col_remap = &col_remaps[t];
        let row_remap = &row_remaps[t];
        for (&(r_or_local, c_or_local), &v) in local.triplets.iter() {
            let global_col = if whitelist_only {
                c_or_local
            } else {
                col_remap[c_or_local as usize]
            };
            let global_row = if use_peaks {
                r_or_local
            } else {
                row_remap[r_or_local as usize]
            };
            *triplet_map.entry((global_row, global_col)).or_insert(0.0) += v;
        }
    }

    Ok(())
}

/// Strip an optional `chr` / `Chr` / `CHR` prefix to get the canonical
/// chromosome key. This matches `genomic_data::coordinates::chr_stripped`
/// behavior so a peaks BED using `1` and a fragments file using `chr1`
/// (or vice versa) resolve to the same internal key.
fn chr_key(s: &str) -> &str {
    if s.len() >= 3 && s.as_bytes()[..3].eq_ignore_ascii_case(b"chr") {
        &s[3..]
    } else {
        s
    }
}

/// Open a fragments-like text file with bgzip-safe gzip support.
///
/// bgzip stores data as a series of concatenated gzip blocks, so the
/// standard `GzDecoder` would stop after the first block. `MultiGzDecoder`
/// reads all blocks and also handles ordinary single-block .gz files.
fn open_fragments_reader(path: &str) -> anyhow::Result<Box<dyn std::io::BufRead>> {
    use std::io::BufReader;
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|x| x.to_str());
    match ext {
        Some("gz") | Some("bgz") => {
            let f = std::fs::File::open(path)?;
            let dec = flate2::read::MultiGzDecoder::new(f);
            Ok(Box::new(BufReader::new(dec)))
        }
        _ => {
            let f = std::fs::File::open(path)?;
            Ok(Box::new(BufReader::new(f)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    ////////////////////////////////////////
    //  chr_key prefix-stripping (units)  //
    ////////////////////////////////////////

    #[test]
    fn chr_key_strips_lowercase_prefix() {
        assert_eq!(chr_key("chr1"), "1");
        assert_eq!(chr_key("chr10"), "10");
        assert_eq!(chr_key("chrX"), "X");
        assert_eq!(chr_key("chrMT"), "MT");
    }

    #[test]
    fn chr_key_strips_mixed_case_prefix() {
        assert_eq!(chr_key("Chr1"), "1");
        assert_eq!(chr_key("CHR1"), "1");
        assert_eq!(chr_key("ChR21"), "21");
    }

    #[test]
    fn chr_key_passes_through_bare_names() {
        assert_eq!(chr_key("1"), "1");
        assert_eq!(chr_key("X"), "X");
        assert_eq!(chr_key("MT"), "MT");
    }

    #[test]
    fn chr_key_leaves_non_chr_prefixes_alone() {
        // "ch" is not the prefix we strip
        assert_eq!(chr_key("ch1"), "ch1");
        // contigs whose names happen to contain "chr" later on
        assert_eq!(chr_key("scaffold_1"), "scaffold_1");
        assert_eq!(chr_key("HLA-DRB1"), "HLA-DRB1");
        // short strings (<3 chars) can't have the prefix
        assert_eq!(chr_key("c"), "c");
        assert_eq!(chr_key("ch"), "ch");
        assert_eq!(chr_key(""), "");
    }

    ////////////////////////////////////////////////////
    //  End-to-end: peaks with mixed chr/no-chr names //
    ////////////////////////////////////////////////////

    /// Peaks BED uses bare `1`/`2`, fragments file uses `chr1`/`chr2`.
    /// Without normalization, zero fragments would overlap any peak.
    #[test]
    fn from_fragments_normalizes_chr_prefix_across_peaks_and_fragments() {
        use std::io::Write;

        let tmp = tempfile::tempdir().unwrap();
        let frags_path = tmp.path().join("frags.tsv");
        let peaks_path = tmp.path().join("peaks.bed");
        let output_stem = tmp.path().join("out");
        let output_stem_str = output_stem.to_str().unwrap().to_string();

        // Fragments use UCSC naming.
        {
            let mut f = std::fs::File::create(&frags_path).unwrap();
            writeln!(f, "# header line should be skipped").unwrap();
            writeln!(f, "chr1\t100\t250\tBC_A\t1").unwrap();
            writeln!(f, "chr1\t200\t450\tBC_A\t1").unwrap();
            writeln!(f, "chr1\t5100\t5300\tBC_B\t2").unwrap();
            writeln!(f, "chr2\t7000\t7400\tBC_A\t1").unwrap();
            writeln!(f, "chr2\t7000\t7400\tBC_C\t1").unwrap();
        }

        // Peaks use Ensembl-style bare chromosome names.
        {
            let mut f = std::fs::File::create(&peaks_path).unwrap();
            writeln!(f, "1\t0\t500").unwrap();
            writeln!(f, "1\t5000\t5500").unwrap();
            writeln!(f, "2\t7000\t8000").unwrap();
        }

        let args = FromFragmentsArgs {
            fragments: frags_path.to_str().unwrap().into(),
            backend: SparseIoBackend::Zarr,
            output: output_stem_str.clone().into(),
            zip: false,
            peaks: Some(peaks_path.to_str().unwrap().into()),
            bin_size: 0,
            barcodes: None,
            use_count: false,
            preload_data: false,
            do_squeeze: false,
            row_nnz_cutoff: 1,
            column_nnz_cutoff: 1,
            block_size: None,
        };

        run_build_from_fragments(&args).unwrap();

        // Open the backend that was just written and check shape + names.
        let backend_path = format!("{}.zarr", output_stem_str);
        let data = open_sparse_matrix(&backend_path, &SparseIoBackend::Zarr).unwrap();

        assert_eq!(data.num_rows(), Some(3));
        assert_eq!(data.num_columns(), Some(3));

        let row_names = data.row_names().unwrap();
        // Row display names come straight from the BED file (no `chr`).
        assert_eq!(row_names[0].as_ref(), "1:0-500");
        assert_eq!(row_names[1].as_ref(), "1:5000-5500");
        assert_eq!(row_names[2].as_ref(), "2:7000-8000");

        let col_names = data.column_names().unwrap();
        assert_eq!(col_names[0].as_ref(), "BC_A");
        assert_eq!(col_names[1].as_ref(), "BC_B");
        assert_eq!(col_names[2].as_ref(), "BC_C");

        // Read full matrix and verify counts. Per fragments above:
        //   peak `1:0-500`     ← BC_A: 2 (two overlapping chr1 frags)
        //   peak `1:5000-5500` ← BC_B: 1 (chr1 5100-5300)
        //   peak `2:7000-8000` ← BC_A: 1, BC_C: 1
        let mat = data.read_columns_dmatrix(vec![0, 1, 2]).unwrap();
        assert_eq!(mat[(0, 0)], 2.0); // BC_A @ 1:0-500
        assert_eq!(mat[(0, 1)], 0.0);
        assert_eq!(mat[(0, 2)], 0.0);
        assert_eq!(mat[(1, 0)], 0.0);
        assert_eq!(mat[(1, 1)], 1.0); // BC_B @ 1:5000-5500
        assert_eq!(mat[(1, 2)], 0.0);
        assert_eq!(mat[(2, 0)], 1.0); // BC_A @ 2:7000-8000
        assert_eq!(mat[(2, 1)], 0.0);
        assert_eq!(mat[(2, 2)], 1.0); // BC_C @ 2:7000-8000
    }

    /// Symmetric case: peaks BED uses `chr1`, fragments use `1` — the same
    /// counts should fall out.
    #[test]
    fn from_fragments_normalizes_chr_prefix_reverse_direction() {
        use std::io::Write;

        let tmp = tempfile::tempdir().unwrap();
        let frags_path = tmp.path().join("frags.tsv");
        let peaks_path = tmp.path().join("peaks.bed");
        let output_stem = tmp.path().join("out");
        let output_stem_str = output_stem.to_str().unwrap().to_string();

        {
            let mut f = std::fs::File::create(&frags_path).unwrap();
            writeln!(f, "1\t100\t250\tBC_A").unwrap();
            writeln!(f, "1\t200\t450\tBC_A").unwrap();
        }
        {
            let mut f = std::fs::File::create(&peaks_path).unwrap();
            writeln!(f, "chr1\t0\t500").unwrap();
        }

        let args = FromFragmentsArgs {
            fragments: frags_path.to_str().unwrap().into(),
            backend: SparseIoBackend::Zarr,
            output: output_stem_str.clone().into(),
            zip: false,
            peaks: Some(peaks_path.to_str().unwrap().into()),
            bin_size: 0,
            barcodes: None,
            use_count: false,
            preload_data: false,
            do_squeeze: false,
            row_nnz_cutoff: 1,
            column_nnz_cutoff: 1,
            block_size: None,
        };

        run_build_from_fragments(&args).unwrap();

        let backend_path = format!("{}.zarr", output_stem_str);
        let data = open_sparse_matrix(&backend_path, &SparseIoBackend::Zarr).unwrap();
        assert_eq!(data.num_rows(), Some(1));
        assert_eq!(data.num_columns(), Some(1));

        let mat = data.read_columns_dmatrix(vec![0]).unwrap();
        assert_eq!(mat[(0, 0)], 2.0);
    }

    /// Bin mode: when fragments mix `chr1` and `1`, both should land in the
    /// same bin row (because the canonical key strips the prefix).
    #[test]
    fn from_fragments_bin_mode_merges_mixed_naming() {
        use std::io::Write;

        let tmp = tempfile::tempdir().unwrap();
        let frags_path = tmp.path().join("frags.tsv");
        let output_stem = tmp.path().join("out");
        let output_stem_str = output_stem.to_str().unwrap().to_string();

        {
            let mut f = std::fs::File::create(&frags_path).unwrap();
            // Same bin (chr1, 0-1000) accessed under two naming conventions.
            writeln!(f, "chr1\t100\t200\tBC_A").unwrap();
            writeln!(f, "1\t300\t400\tBC_A").unwrap();
        }

        let args = FromFragmentsArgs {
            fragments: frags_path.to_str().unwrap().into(),
            backend: SparseIoBackend::Zarr,
            output: output_stem_str.clone().into(),
            zip: false,
            peaks: None,
            bin_size: 1000,
            barcodes: None,
            use_count: false,
            preload_data: false,
            do_squeeze: false,
            row_nnz_cutoff: 1,
            column_nnz_cutoff: 1,
            block_size: None,
        };

        run_build_from_fragments(&args).unwrap();

        let backend_path = format!("{}.zarr", output_stem_str);
        let data = open_sparse_matrix(&backend_path, &SparseIoBackend::Zarr).unwrap();
        // Single canonical bin, regardless of mixed naming.
        assert_eq!(data.num_rows(), Some(1));
        assert_eq!(data.num_columns(), Some(1));

        // Display name uses the first chr string we saw (chr1).
        let row_names = data.row_names().unwrap();
        assert_eq!(row_names[0].as_ref(), "chr1:0-1000");

        let mat = data.read_columns_dmatrix(vec![0]).unwrap();
        assert_eq!(mat[(0, 0)], 2.0);
    }

    /////////////////////////////////////////////////////////////////
    //  --preload-data parity: parallel path must produce exactly  //
    //  the same row names, column names, and matrix entries as    //
    //  the serial path.                                           //
    /////////////////////////////////////////////////////////////////

    fn run_pair_and_compare(args_serial: FromFragmentsArgs, args_parallel: FromFragmentsArgs) {
        let out_serial = format!("{}.zarr", &args_serial.output);
        let out_parallel = format!("{}.zarr", &args_parallel.output);
        run_build_from_fragments(&args_serial).unwrap();
        run_build_from_fragments(&args_parallel).unwrap();
        let a = open_sparse_matrix(&out_serial, &SparseIoBackend::Zarr).unwrap();
        let b = open_sparse_matrix(&out_parallel, &SparseIoBackend::Zarr).unwrap();
        assert_eq!(a.num_rows(), b.num_rows());
        assert_eq!(a.num_columns(), b.num_columns());
        assert_eq!(a.row_names().unwrap(), b.row_names().unwrap());
        assert_eq!(a.column_names().unwrap(), b.column_names().unwrap());
        let ncols = a.num_columns().unwrap();
        let cols: Vec<usize> = (0..ncols).collect();
        let ma = a.read_columns_dmatrix(cols.clone()).unwrap();
        let mb = b.read_columns_dmatrix(cols).unwrap();
        assert_eq!(ma, mb);
    }

    #[test]
    fn from_fragments_preload_matches_serial_peaks_mode() {
        use std::io::Write;
        let tmp = tempfile::tempdir().unwrap();
        let frags_path = tmp.path().join("frags.tsv");
        let peaks_path = tmp.path().join("peaks.bed");
        let out_s = tmp.path().join("serial").to_str().unwrap().to_string();
        let out_p = tmp.path().join("parallel").to_str().unwrap().to_string();

        // Enough rows to exercise multiple rayon chunks.
        {
            let mut f = std::fs::File::create(&frags_path).unwrap();
            writeln!(f, "# header line").unwrap();
            let barcodes = ["BC_A", "BC_B", "BC_C", "BC_D", "BC_E"];
            for i in 0..2000 {
                let chr = if i % 2 == 0 { "chr1" } else { "chr2" };
                let s = (i as i64) * 37;
                let e = s + 150;
                let bc = barcodes[i % barcodes.len()];
                writeln!(f, "{}\t{}\t{}\t{}", chr, s, e, bc).unwrap();
            }
        }
        {
            let mut f = std::fs::File::create(&peaks_path).unwrap();
            writeln!(f, "1\t0\t10000").unwrap();
            writeln!(f, "1\t10000\t20000").unwrap();
            writeln!(f, "2\t0\t10000").unwrap();
            writeln!(f, "2\t10000\t20000").unwrap();
        }

        let base = FromFragmentsArgs {
            fragments: frags_path.to_str().unwrap().into(),
            backend: SparseIoBackend::Zarr,
            output: out_s.clone().into(),
            zip: false,
            peaks: Some(peaks_path.to_str().unwrap().into()),
            bin_size: 0,
            barcodes: None,
            use_count: false,
            preload_data: false,
            do_squeeze: false,
            row_nnz_cutoff: 1,
            column_nnz_cutoff: 1,
            block_size: None,
        };
        let par = FromFragmentsArgs {
            output: out_p.into(),
            preload_data: true,
            fragments: base.fragments.clone(),
            backend: base.backend.clone(),
            zip: base.zip,
            peaks: base.peaks.clone(),
            bin_size: base.bin_size,
            barcodes: base.barcodes.clone(),
            use_count: base.use_count,
            do_squeeze: base.do_squeeze,
            row_nnz_cutoff: base.row_nnz_cutoff,
            column_nnz_cutoff: base.column_nnz_cutoff,
            block_size: base.block_size,
        };
        run_pair_and_compare(base, par);
    }

    #[test]
    fn from_fragments_preload_matches_serial_bin_mode() {
        use std::io::Write;
        let tmp = tempfile::tempdir().unwrap();
        let frags_path = tmp.path().join("frags.tsv");
        let out_s = tmp.path().join("serial").to_str().unwrap().to_string();
        let out_p = tmp.path().join("parallel").to_str().unwrap().to_string();

        {
            let mut f = std::fs::File::create(&frags_path).unwrap();
            let barcodes = ["BC_A", "BC_B", "BC_C"];
            for i in 0..1500 {
                let chr = if i % 3 == 0 { "chr1" } else { "chr2" };
                let s = (i as i64) * 100;
                let e = s + 200;
                let bc = barcodes[i % barcodes.len()];
                writeln!(f, "{}\t{}\t{}\t{}", chr, s, e, bc).unwrap();
            }
        }

        let base = FromFragmentsArgs {
            fragments: frags_path.to_str().unwrap().into(),
            backend: SparseIoBackend::Zarr,
            output: out_s.clone().into(),
            zip: false,
            peaks: None,
            bin_size: 1000,
            barcodes: None,
            use_count: false,
            preload_data: false,
            do_squeeze: false,
            row_nnz_cutoff: 1,
            column_nnz_cutoff: 1,
            block_size: None,
        };
        let par = FromFragmentsArgs {
            output: out_p.into(),
            preload_data: true,
            fragments: base.fragments.clone(),
            backend: base.backend.clone(),
            zip: base.zip,
            peaks: base.peaks.clone(),
            bin_size: base.bin_size,
            barcodes: base.barcodes.clone(),
            use_count: base.use_count,
            do_squeeze: base.do_squeeze,
            row_nnz_cutoff: base.row_nnz_cutoff,
            column_nnz_cutoff: base.column_nnz_cutoff,
            block_size: base.block_size,
        };
        run_pair_and_compare(base, par);
    }
}
