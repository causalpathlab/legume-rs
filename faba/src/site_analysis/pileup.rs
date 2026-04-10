use arrow::array::{Float32Array, Int64Array, StringArray, UInt64Array};
use clap::Args;
use data_beans::hdf5_io::resolve_backend_file;
use data_beans::sparse_io::open_sparse_matrix;
use log::info;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rustc_hash::FxHashMap;
use std::io::Write;

#[derive(clap::ValueEnum, Clone, Debug)]
pub enum PileupSignal {
    /// Sum of values across all cells at each position
    Sum,
    /// Number of non-zero cells at each position
    Nnz,
    /// log10(1 + sum) at each position
    Log10Sum,
}

impl PileupSignal {
    fn name(&self) -> &'static str {
        match self {
            PileupSignal::Sum => "sum",
            PileupSignal::Nnz => "nnz",
            PileupSignal::Log10Sum => "log10-sum",
        }
    }
}

#[derive(clap::ValueEnum, Clone, Debug)]
pub enum SiteSignal {
    /// Number of sites per bin
    Count,
    /// Sum of wild-type base coverage per bin
    WtCoverage,
    /// Sum of mutant base coverage per bin
    MutCoverage,
    /// Sum of -log10(p-value) per bin
    NegLog10Pv,
}

impl SiteSignal {
    fn name(&self) -> &'static str {
        match self {
            SiteSignal::Count => "count",
            SiteSignal::WtCoverage => "wt-coverage",
            SiteSignal::MutCoverage => "mut-coverage",
            SiteSignal::NegLog10Pv => "-log10(pv)",
        }
    }
}

#[derive(Args, Debug)]
pub struct PileupArgs {
    /// Sparse matrix file (zarr or h5) from faba output
    pub data_file: Box<str>,

    /// Gene name or ID substring to match (case-insensitive)
    #[arg(short = 'q', long = "gene", required = true)]
    gene_query: Box<str>,

    /// Site-level parquet file (from dartseq or atoi output)
    #[arg(short = 's', long = "sites")]
    site_file: Option<Box<str>>,

    /// Signal aggregation mode for sparse matrix track
    #[arg(long, value_enum, default_value = "sum")]
    signal: PileupSignal,

    /// Signal for site-level parquet track
    #[arg(long = "site-signal", value_enum, default_value = "wt-coverage")]
    site_signal: SiteSignal,

    /// Number of bins along the gene body
    #[arg(short = 'n', long = "bins", default_value_t = 80)]
    num_bins: usize,

    /// Height of ASCII plot in terminal rows (per track)
    #[arg(long = "height", default_value_t = 20)]
    plot_height: usize,

    /// Output TSV file path (optional)
    #[arg(short, long)]
    output: Option<Box<str>>,

    /// Suppress ASCII plot
    #[arg(long)]
    quiet: bool,
}

/// Parse row name like "ENSG00000139618_BRCA2/m6A/chr13:32350000"
/// Returns (gene_part, chr, position) if parseable.
fn parse_row_name(name: &str) -> Option<(&str, &str, i64)> {
    let mut parts = name.splitn(3, '/');
    let gene_part = parts.next()?;
    let _modality = parts.next()?;
    let detail = parts.next()?;
    let (chr, pos_str) = detail.split_once(':')?;
    let pos = pos_str.parse::<i64>().ok()?;
    Some((gene_part, chr, pos))
}

/// Case-insensitive ASCII substring match without allocation.
fn contains_ignore_ascii_case(haystack: &str, needle: &str) -> bool {
    haystack
        .as_bytes()
        .windows(needle.len())
        .any(|w| w.eq_ignore_ascii_case(needle.as_bytes()))
}

struct PosAgg {
    sum: f64,
    nnz: usize,
}

struct BinnedPileup {
    gene: Box<str>,
    chr: Box<str>,
    bins: Vec<f64>,
    min_pos: i64,
    max_pos: i64,
    num_sites: usize,
    track_label: &'static str,
    signal_name: &'static str,
}

struct MatrixGeneData {
    gene: Box<str>,
    chr: Box<str>,
    positions: Vec<(i64, f64)>,
}

/// Gene boundaries and per-site signals from the sites parquet.
struct SiteAnnotation {
    gene_start: i64,
    gene_stop: i64,
    positions: Vec<(i64, f64)>,
    num_sites: usize,
}

fn read_site_annotation(
    site_file: &str,
    gene_query: &str,
    site_signal: &SiteSignal,
) -> anyhow::Result<SiteAnnotation> {
    let file = std::fs::File::open(site_file)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    let mut gene_start: Option<i64> = None;
    let mut gene_stop: Option<i64> = None;
    let mut positions: Vec<(i64, f64)> = Vec::new();
    let mut distinct_genes: FxHashMap<Box<str>, usize> = FxHashMap::default();

    // Resolve signal column names once
    let signal_cols: &[&str] = match site_signal {
        SiteSignal::WtCoverage => &["wt_a", "wt_t", "wt_g", "wt_c"],
        SiteSignal::MutCoverage => &["mut_a", "mut_t", "mut_g", "mut_c"],
        _ => &[],
    };

    for batch in reader {
        let batch = batch?;

        let gene_col = batch
            .column_by_name("gene")
            .ok_or_else(|| anyhow::anyhow!("missing 'gene' column in {}", site_file))?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow::anyhow!("'gene' column is not a string array"))?;

        let pos_col = batch
            .column_by_name("primary_pos")
            .ok_or_else(|| anyhow::anyhow!("missing 'primary_pos' column in {}", site_file))?
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| anyhow::anyhow!("'primary_pos' column is not Int64"))?;

        let start_col = batch
            .column_by_name("gene_start")
            .and_then(|c| c.as_any().downcast_ref::<Int64Array>());

        let stop_col = batch
            .column_by_name("gene_stop")
            .and_then(|c| c.as_any().downcast_ref::<Int64Array>());

        // Resolve signal columns once per batch
        let pv_col = batch
            .column_by_name("pv")
            .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

        let base_cols: Vec<Option<&UInt64Array>> = signal_cols
            .iter()
            .map(|name| {
                batch
                    .column_by_name(name)
                    .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
            })
            .collect();

        for i in 0..batch.num_rows() {
            let gene_val = gene_col.value(i);
            if !contains_ignore_ascii_case(gene_val, gene_query) {
                continue;
            }

            *distinct_genes.entry(gene_val.into()).or_insert(0) += 1;

            let pos = pos_col.value(i);

            // Update gene boundaries
            if let (Some(sc), Some(tc)) = (start_col, stop_col) {
                let gs = sc.value(i);
                let gt = tc.value(i);
                gene_start = Some(gene_start.map_or(gs, |v: i64| v.min(gs)));
                gene_stop = Some(gene_stop.map_or(gt, |v: i64| v.max(gt)));
            }

            let value = match site_signal {
                SiteSignal::Count => 1.0,
                SiteSignal::WtCoverage | SiteSignal::MutCoverage => {
                    let mut total = 0u64;
                    for col in base_cols.iter().flatten() {
                        total += col.value(i);
                    }
                    total as f64
                }
                SiteSignal::NegLog10Pv => {
                    if let Some(pv) = pv_col {
                        let p = pv.value(i);
                        if p > 0.0 {
                            -(p as f64).log10()
                        } else {
                            300.0
                        }
                    } else {
                        0.0
                    }
                }
            };

            positions.push((pos, value));
        }
    }

    if positions.is_empty() {
        return Err(anyhow::anyhow!(
            "no sites matching gene '{}' in {}",
            gene_query,
            site_file
        ));
    }

    if distinct_genes.len() > 1 {
        let mut genes: Vec<_> = distinct_genes.into_iter().collect();
        genes.sort_by(|a, b| b.1.cmp(&a.1));
        let list: Vec<String> = genes
            .iter()
            .take(10)
            .map(|(g, n)| format!("  {} ({} sites)", g, n))
            .collect();
        return Err(anyhow::anyhow!(
            "ambiguous gene query '{}' matched {} genes in parquet:\n{}{}",
            gene_query,
            genes.len(),
            list.join("\n"),
            if genes.len() > 10 { "\n  ..." } else { "" }
        ));
    }

    if positions.is_empty() {
        return Err(anyhow::anyhow!(
            "no sites matching gene '{}' in {}",
            gene_query,
            site_file
        ));
    }

    positions.sort_unstable_by_key(|(pos, _)| *pos);

    let gs = gene_start.unwrap_or_else(|| positions.first().unwrap().0);
    let gt = gene_stop.unwrap_or_else(|| positions.last().unwrap().0);

    info!(
        "loaded {} sites from parquet, gene boundaries: {}-{}",
        positions.len(),
        gs,
        gt
    );

    let num_sites = positions.len();
    Ok(SiteAnnotation {
        gene_start: gs,
        gene_stop: gt,
        positions,
        num_sites,
    })
}

fn read_matrix_positions(
    data_file: &str,
    gene_query: &str,
    signal: &PileupSignal,
) -> anyhow::Result<MatrixGeneData> {
    let (backend, resolved_path) = resolve_backend_file(data_file, None)?;
    let data = open_sparse_matrix(&resolved_path, &backend)?;

    let row_names = data.row_names()?;

    let mut matched_rows: Vec<(usize, &str, &str, i64)> = Vec::new();
    let mut distinct_genes: FxHashMap<&str, usize> = FxHashMap::default();

    for (idx, name) in row_names.iter().enumerate() {
        if let Some((gene_part, chr, pos)) = parse_row_name(name) {
            if contains_ignore_ascii_case(gene_part, gene_query) {
                *distinct_genes.entry(gene_part).or_insert(0) += 1;
                matched_rows.push((idx, gene_part, chr, pos));
            }
        }
    }

    if matched_rows.is_empty() {
        return Err(anyhow::anyhow!(
            "no rows matching gene '{}' in {}",
            gene_query,
            data_file
        ));
    }

    if distinct_genes.len() > 1 {
        let mut genes: Vec<_> = distinct_genes.into_iter().collect();
        genes.sort_by(|a, b| b.1.cmp(&a.1));
        let list: Vec<String> = genes
            .iter()
            .take(10)
            .map(|(g, n)| format!("  {} ({} sites)", g, n))
            .collect();
        return Err(anyhow::anyhow!(
            "ambiguous gene query '{}' matched {} genes:\n{}{}",
            gene_query,
            genes.len(),
            list.join("\n"),
            if genes.len() > 10 { "\n  ..." } else { "" }
        ));
    }

    let gene: Box<str> = matched_rows[0].1.into();
    let chr: Box<str> = matched_rows[0].2.into();

    info!(
        "matched {} sites for gene {} on {}",
        matched_rows.len(),
        gene,
        chr
    );

    let local_to_pos: Vec<i64> = matched_rows.iter().map(|(_, _, _, pos)| *pos).collect();
    let row_indices: Vec<usize> = matched_rows.iter().map(|(idx, _, _, _)| *idx).collect();
    let (_nrow, _ncol, triplets) = data.read_triplets_by_rows(row_indices)?;

    let mut pos_agg: FxHashMap<i64, PosAgg> = FxHashMap::default();

    for &pos in &local_to_pos {
        pos_agg.entry(pos).or_insert(PosAgg { sum: 0.0, nnz: 0 });
    }

    for (row, _col, val) in &triplets {
        let local_idx = *row as usize;
        if local_idx < local_to_pos.len() && *val != 0.0 {
            let pos = local_to_pos[local_idx];
            let agg = pos_agg.get_mut(&pos).unwrap();
            agg.sum += *val as f64;
            agg.nnz += 1;
        }
    }

    let mut positions: Vec<(i64, f64)> = pos_agg
        .iter()
        .map(|(&pos, agg)| {
            let value = match signal {
                PileupSignal::Sum | PileupSignal::Log10Sum => agg.sum,
                PileupSignal::Nnz => agg.nnz as f64,
            };
            (pos, value)
        })
        .collect();
    positions.sort_unstable_by_key(|(pos, _)| *pos);

    Ok(MatrixGeneData {
        gene,
        chr,
        positions,
    })
}

fn bin_positions_with_extent(
    positions: &[(i64, f64)],
    num_bins: usize,
    min_pos: i64,
    max_pos: i64,
    log_transform: bool,
) -> Vec<f64> {
    let span = (max_pos - min_pos).max(1) as u64;
    let mut bins = vec![0.0f64; num_bins];

    for &(pos, val) in positions {
        if pos < min_pos || pos > max_pos {
            continue;
        }
        let rel = (pos - min_pos) as u64;
        let bin = (rel * num_bins as u64 / span).min(num_bins as u64 - 1) as usize;
        bins[bin] += val;
    }

    if log_transform {
        for val in bins.iter_mut() {
            *val = (1.0 + *val).log10();
        }
    }

    bins
}

fn print_vertical_histogram(pileup: &BinnedPileup, height: usize) {
    let header = format!(
        "  {}  {}:{}-{}  [{}] signal: {}  sites: {}",
        pileup.gene,
        pileup.chr,
        pileup.min_pos,
        pileup.max_pos,
        pileup.track_label,
        pileup.signal_name,
        pileup.num_sites
    );

    let max_val = pileup.bins.iter().cloned().fold(0.0f64, f64::max);
    if max_val <= 0.0 {
        eprintln!("{}", header);
        eprintln!("  (no signal)");
        return;
    }

    eprintln!();
    eprintln!("{}", header);
    eprintln!();

    let max_label = format!("{:.1}", max_val);
    let label_width = max_label.len().max(4);

    for row in (1..=height).rev() {
        let threshold = max_val * row as f64 / height as f64;

        let label = if row == height {
            format!("{:>w$.1}", max_val, w = label_width)
        } else if row == height / 2 {
            format!("{:>w$.1}", max_val / 2.0, w = label_width)
        } else if row == 1 {
            format!("{:>w$.1}", max_val / height as f64, w = label_width)
        } else {
            " ".repeat(label_width)
        };

        let mut bar = String::with_capacity(pileup.bins.len());
        for &val in &pileup.bins {
            if val >= threshold {
                bar.push('#');
            } else {
                bar.push(' ');
            }
        }

        eprintln!("{} |{}", label, bar);
    }

    let label_pad = " ".repeat(label_width);
    eprintln!("{} +{}", label_pad, "-".repeat(pileup.bins.len()));

    let left_label = format!("{}", pileup.min_pos);
    let right_label = format!("{}", pileup.max_pos);
    let gap = pileup
        .bins
        .len()
        .saturating_sub(left_label.len() + right_label.len());
    eprintln!(
        "{} {}{}{}",
        " ".repeat(label_width + 1),
        left_label,
        " ".repeat(gap),
        right_label
    );
    eprintln!();
}

fn write_pileup_tsv(tracks: &[&BinnedPileup], output: &str) -> anyhow::Result<()> {
    let mut writer = matrix_util::common_io::open_buf_writer(output)?;

    for pileup in tracks {
        writeln!(
            writer,
            "#track={}\tgene={}\tchr={}\tmin_pos={}\tmax_pos={}\tsignal={}\tnum_sites={}",
            pileup.track_label,
            pileup.gene,
            pileup.chr,
            pileup.min_pos,
            pileup.max_pos,
            pileup.signal_name,
            pileup.num_sites
        )?;
        writeln!(writer, "bin\tgenomic_start\tgenomic_stop\tvalue")?;

        let span = (pileup.max_pos - pileup.min_pos).max(1);
        let num_bins = pileup.bins.len();

        for (i, &val) in pileup.bins.iter().enumerate() {
            let bin_start = pileup.min_pos + (i as i64 * span / num_bins as i64);
            let bin_stop = pileup.min_pos + ((i + 1) as i64 * span / num_bins as i64);
            writeln!(writer, "{}\t{}\t{}\t{:.4}", i, bin_start, bin_stop, val)?;
        }
    }

    writer.flush()?;
    Ok(())
}

pub fn run_pileup(args: &PileupArgs) -> anyhow::Result<()> {
    // Read sparse matrix data
    let mtx = read_matrix_positions(&args.data_file, &args.gene_query, &args.signal)?;
    let matrix_num_sites = mtx.positions.len();

    // Determine x-axis extent and build site track if parquet provided
    let site_annotation = args
        .site_file
        .as_ref()
        .map(|sf| read_site_annotation(sf, &args.gene_query, &args.site_signal))
        .transpose()?;

    let (min_pos, max_pos) = if let Some(ref sa) = site_annotation {
        (sa.gene_start, sa.gene_stop)
    } else {
        let lo = mtx.positions.first().map(|p| p.0).unwrap_or(0);
        let hi = mtx.positions.last().map(|p| p.0).unwrap_or(0);
        (lo, hi)
    };

    let max_sites = site_annotation
        .as_ref()
        .map_or(matrix_num_sites, |sa| matrix_num_sites.max(sa.num_sites));
    let effective_bins = args.num_bins.min(max_sites.max(1));
    let is_log = matches!(args.signal, PileupSignal::Log10Sum);

    let matrix_bins =
        bin_positions_with_extent(&mtx.positions, effective_bins, min_pos, max_pos, is_log);

    let site_pileup = site_annotation.as_ref().map(|sa| BinnedPileup {
        gene: mtx.gene.clone(),
        chr: mtx.chr.clone(),
        bins: bin_positions_with_extent(&sa.positions, effective_bins, min_pos, max_pos, false),
        min_pos,
        max_pos,
        num_sites: sa.num_sites,
        track_label: "sites",
        signal_name: args.site_signal.name(),
    });

    // Build matrix pileup after site_pileup to move ownership
    let matrix_pileup = BinnedPileup {
        gene: mtx.gene,
        chr: mtx.chr,
        bins: matrix_bins,
        min_pos,
        max_pos,
        num_sites: matrix_num_sites,
        track_label: "matrix",
        signal_name: args.signal.name(),
    };

    if !args.quiet {
        print_vertical_histogram(&matrix_pileup, args.plot_height);
        if let Some(ref sp) = site_pileup {
            print_vertical_histogram(sp, args.plot_height);
        }
    }

    if let Some(ref output) = args.output {
        let mut tracks: Vec<&BinnedPileup> = vec![&matrix_pileup];
        if let Some(ref sp) = site_pileup {
            tracks.push(sp);
        }
        write_pileup_tsv(&tracks, output)?;
        info!("wrote pileup TSV to {}", output);
    }

    Ok(())
}
