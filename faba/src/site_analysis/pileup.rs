use crate::data::cell_membership::CellMembership;
use crate::site_analysis::miami::bin::BinEdges;
use crate::site_analysis::miami::depth::read_depth_binned;
use crate::site_analysis::miami::genemodel::{load_gene_models, models_extent};
use crate::site_analysis::miami::render::{render_miami, FigOpts, PanelData};
use arrow::array::{Float32Array, Int64Array, StringArray, UInt64Array};
use auxiliary_data::feature_names::FeatureNameKind;
use clap::Args;
use data_beans::hdf5_io::resolve_backend_file;
use data_beans::sparse_io::open_sparse_matrix;
use genomic_data::bed::Bed;
use genomic_data::coordinates::chr_eq;
use genomic_data::sam::CellBarcode;
use log::info;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use plot_utils::palette::Palette;
use rustc_hash::FxHashMap;
use std::io::Write;
use std::sync::Arc;

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

/// Figure output format when only one is wanted. Omit to get the default
/// SVG + PDF pair.
#[derive(clap::ValueEnum, Clone, Debug)]
pub enum FigFormat {
    Svg,
    Pdf,
}

#[derive(Args, Debug)]
pub struct PileupArgs {
    /// Sparse matrix file(s) (zarr or h5) from faba output. Multiple files
    /// (e.g. replicates via a shell glob) are aggregated per genomic
    /// position into a single track.
    #[arg(required = true, num_args = 1..)]
    pub data_files: Vec<Box<str>>,

    /// Genes to pile up: comma-separated symbols (`MYCBP,GNA15`) or
    /// Ensembl IDs, case-insensitive. Uses the auxiliary-data relaxed
    /// gene-name scheme; all matched genes are aggregated into one pileup.
    #[arg(
        short = 'q',
        long = "genes",
        visible_alias = "gene",
        value_delimiter = ','
    )]
    genes: Vec<Box<str>>,

    /// Genomic regions to pile up: comma-separated `chr:lb-ub`
    /// (`chr17:1000-2000,chr1:50-99`). Selects rows by position, with or
    /// without `--genes`. At least one of `--genes`/`--regions` is required.
    #[arg(long = "regions", visible_alias = "region", value_delimiter = ',')]
    regions: Vec<Box<str>>,

    /// Site-level parquet file (from dartseq or atoi output) for the
    /// second track
    #[arg(short = 's', long = "sites-parquet")]
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

    // ----- Miami figure mode -----
    // Providing any of `--gtf`, `--bam`, `--format`, `--svg`, or `--png`
    // switches `pileup` from the ASCII histogram to a faceted Miami plot
    // (epi sites up / gene model / read depth down, one panel per cell
    // type). Otherwise the existing ASCII/TSV behavior is unchanged.
    /// Gene annotation GTF/GFF for the middle gene-model track (exons,
    /// introns, strand). Enables figure mode.
    #[arg(long = "gtf")]
    gtf: Option<Box<str>>,

    /// BAM file(s) for the bottom read-depth track. Repeatable; replicates
    /// are pooled. Enables figure mode.
    #[arg(long = "bam", num_args = 1..)]
    bam_files: Vec<Box<str>>,

    /// Cell barcode -> cell type membership (TSV/CSV/Parquet). Panels are
    /// faceted by cell type; without it the figure is a single "all cells"
    /// panel.
    #[arg(long = "cell-membership", visible_alias = "membership")]
    cell_membership_file: Option<Box<str>>,

    /// Column index of the cell barcode in the membership file
    #[arg(long = "membership-barcode-col", default_value_t = 0)]
    membership_barcode_col: usize,

    /// Column index of the cell type in the membership file
    #[arg(long = "membership-celltype-col", default_value_t = 1)]
    membership_celltype_col: usize,

    /// Require exact barcode matching (default: membership barcodes match
    /// as prefixes of BAM/matrix barcodes, handling "-1" suffixes)
    #[arg(long = "exact-barcode-match", default_value_t = false)]
    exact_barcode_match: bool,

    /// BAM tag holding the cell barcode (read-depth track)
    #[arg(long = "cell-barcode-tag", default_value = "CB")]
    cell_barcode_tag: Box<str>,

    /// Restrict the top track to these modalities (e.g. `m6A,A-to-I`).
    /// Empty = all matrix rows. Matched case-insensitively against the
    /// `gene/MODALITY/detail` row name.
    #[arg(long = "top-modality", value_delimiter = ',')]
    top_modality: Vec<Box<str>>,

    /// Output prefix for figure files (`<prefix>.miami.{svg,pdf,png}`).
    /// Defaults to the gene label.
    #[arg(long = "out")]
    out: Option<Box<str>>,

    /// Emit only this format. Omit for the default SVG + PDF.
    #[arg(long = "format", value_enum)]
    format: Option<FigFormat>,

    /// Also write the SVG (always written unless `--format pdf`)
    #[arg(long = "svg", default_value_t = false)]
    svg: bool,

    /// Also write a flattened PNG
    #[arg(long = "png", default_value_t = false)]
    png: bool,

    /// Skip PDF output
    #[arg(long = "no-pdf", default_value_t = false)]
    no_pdf: bool,

    /// Figure width in inches
    #[arg(long = "fig-width", default_value_t = 8.0)]
    fig_width: f32,

    /// Figure resolution (dots per inch)
    #[arg(long = "dpi", default_value_t = 300)]
    dpi: u32,

    /// Qualitative color palette for cell-type panels
    #[arg(long = "palette", value_enum, default_value = "auto")]
    palette: Palette,

    /// Rasterize the per-site dot layer once a panel exceeds this many
    /// sites (keeps SVG/PDF size bounded; axes/areas stay vector)
    #[arg(long = "raster-threshold", default_value_t = 300)]
    raster_threshold: usize,
}

/// Parse a faba row name `gene_key/modality/detail`. `detail` is either
/// `chr:pos` (site output, e.g. `ENSG00000139618_BRCA2/m6A/chr13:32350000`)
/// or a bare component ordinal (mixture output, e.g.
/// `ENSG00000139618_BRCA2/m6A/0`). Returns `(gene_part, chr, x)` where
/// `chr` is empty for mixture rows and `x` is the genomic position or the
/// component ordinal used as the pileup x-coordinate.
/// Returns `(gene, modality, chr, pos)`. `modality` is the middle
/// `/`-delimited token (e.g. `m6A`), used by the figure's
/// `--top-modality` filter; `chr` is empty for mixture (component) rows.
fn parse_row_name_full(name: &str) -> Option<(&str, &str, &str, i64)> {
    let mut parts = name.splitn(3, '/');
    let gene_part = parts.next()?;
    let modality = parts.next()?;
    let detail = parts.next()?;
    if let Some((chr, pos_str)) = detail.split_once(':') {
        let pos = pos_str.parse::<i64>().ok()?;
        Some((gene_part, modality, chr, pos))
    } else {
        // Mixture rows carry a component ordinal, not a chromosome.
        let component = detail.parse::<i64>().ok()?;
        Some((gene_part, modality, "", component))
    }
}

/// Relaxed gene matching, consistent with the auxiliary-data
/// `FeatureNameKind::Gene` canonicalization used for cross-file row
/// alignment. A row matches when its `gene_part` shares any `_`-split
/// component with the query, or agrees on the canonical gene symbol
/// (last `_`-delimited component, Cell Ranger feature-type suffix
/// stripped) — so both a symbol (`MYCBP`) and an Ensembl ID
/// (`ENSG00000139618`) query resolve the same row. Case-insensitive.
/// `query_sym` is the pre-canonicalized query symbol.
fn gene_matches(query: &str, query_sym: &str, gene_part: &str) -> bool {
    // Allocation-free component check first — it directly covers symbol and
    // Ensembl-ID queries (and subsumes a full-composite match). Fall back to
    // the suffix-stripping canonicalizer only when the components miss.
    gene_part.split('_').any(|c| c.eq_ignore_ascii_case(query))
        || FeatureNameKind::Gene { delim: '_' }
            .canonicalize(gene_part)
            .eq_ignore_ascii_case(query_sym)
}

/// Canonical query symbol used by [`gene_matches`].
fn query_symbol(query: &str) -> Box<str> {
    FeatureNameKind::Gene { delim: '_' }.canonicalize(query)
}

/// A `chr:lb-ub` genomic window. Bounds are inclusive.
struct Region {
    chr: Box<str>,
    lb: i64,
    ub: i64,
}

/// Parse `chr:lb-ub` (e.g. `chr17:1000-2000`). Reversed bounds are
/// swapped so `lb <= ub` always holds.
fn parse_region(spec: &str) -> anyhow::Result<Region> {
    let bad = || anyhow::anyhow!("region '{}' must be formatted chr:lb-ub", spec);
    let (chr, range) = spec.split_once(':').ok_or_else(bad)?;
    let (lb, ub) = range.split_once('-').ok_or_else(bad)?;
    let lb: i64 = lb.trim().parse().map_err(|_| bad())?;
    let ub: i64 = ub.trim().parse().map_err(|_| bad())?;
    let (lb, ub) = if lb <= ub { (lb, ub) } else { (ub, lb) };
    let chr = chr.trim();
    if chr.is_empty() {
        return Err(bad());
    }
    Ok(Region {
        chr: chr.into(),
        lb,
        ub,
    })
}

/// Combined gene + region row selector. A row is selected when it
/// matches any requested gene OR falls inside any requested region
/// (union), so callers can pass either or both.
pub(crate) struct Selector {
    genes: Vec<Box<str>>,
    gene_syms: Vec<Box<str>>,
    regions: Vec<Region>,
}

impl Selector {
    pub(crate) fn build(genes: &[Box<str>], regions: &[Box<str>]) -> anyhow::Result<Self> {
        let genes: Vec<Box<str>> = genes
            .iter()
            .map(|g| g.trim())
            .filter(|g| !g.is_empty())
            .map(Into::into)
            .collect();
        let gene_syms: Vec<Box<str>> = genes.iter().map(|g| query_symbol(g)).collect();
        let regions: Vec<Region> = regions
            .iter()
            .map(|r| r.trim())
            .filter(|r| !r.is_empty())
            .map(parse_region)
            .collect::<anyhow::Result<_>>()?;
        if genes.is_empty() && regions.is_empty() {
            anyhow::bail!("provide at least one of --genes or --regions");
        }
        Ok(Self {
            genes,
            gene_syms,
            regions,
        })
    }

    pub(crate) fn matches_gene(&self, gene_part: &str) -> bool {
        self.genes
            .iter()
            .zip(&self.gene_syms)
            .any(|(g, sym)| gene_matches(g, sym, gene_part))
    }

    fn matches_region(&self, chr: &str, pos: i64) -> bool {
        // Mixture rows carry no chromosome (empty `chr`); they can never sit
        // inside a region, so guard explicitly rather than relying on
        // `parse_region` having rejected empty region chromosomes.
        !chr.is_empty()
            && self
                .regions
                .iter()
                .any(|r| chr_eq(&r.chr, chr) && pos >= r.lb && pos <= r.ub)
    }

    /// Row is kept when it matches any gene or any region.
    fn selects(&self, gene_part: &str, chr: &str, pos: i64) -> bool {
        self.matches_gene(gene_part) || self.matches_region(chr, pos)
    }

    /// Short description of the active selection for log/error messages.
    fn describe(&self) -> String {
        let mut parts = Vec::new();
        if !self.genes.is_empty() {
            let g: Vec<&str> = self.genes.iter().map(|g| g.as_ref()).collect();
            parts.push(format!("genes [{}]", g.join(",")));
        }
        if !self.regions.is_empty() {
            let r: Vec<String> = self
                .regions
                .iter()
                .map(|r| format!("{}:{}-{}", r.chr, r.lb, r.ub))
                .collect();
            parts.push(format!("regions [{}]", r.join(",")));
        }
        parts.join(" + ")
    }
}

/// Human-readable label for the set of matched genes. A single gene is
/// shown verbatim; multiple matches collapse to `N genes: a,b,...`.
fn summarize_genes(distinct: &FxHashMap<Box<str>, usize>) -> Box<str> {
    if distinct.len() == 1 {
        return distinct.keys().next().unwrap().clone();
    }
    let mut names: Vec<&str> = distinct.keys().map(|k| k.as_ref()).collect();
    names.sort_unstable();
    let shown = names.iter().take(5).copied().collect::<Vec<_>>().join(",");
    format!(
        "{} genes: {}{}",
        names.len(),
        shown,
        if names.len() > 5 { ",..." } else { "" }
    )
    .into()
}

/// Label for the chromosome axis. `component` when matched rows carry no
/// chromosome (mixture output), the single chromosome when all matches
/// agree, else `*` for a multi-chromosome aggregate.
fn summarize_chr(matched_chrs: &[Box<str>]) -> Box<str> {
    let mut chrs: Vec<&str> = matched_chrs
        .iter()
        .map(|c| c.as_ref())
        .filter(|c| !c.is_empty())
        .collect();
    chrs.sort_unstable();
    chrs.dedup();
    match chrs.len() {
        0 => "component".into(),
        1 => chrs[0].into(),
        _ => "*".into(),
    }
}

struct PosAgg {
    sum: f64,
    nnz: usize,
}

struct BinnedPileup {
    gene: Box<str>,
    chr: Box<str>,
    bins: Vec<f64>,
    /// Distinct site coordinates (genomic order) backing the bins; used to
    /// place `+` axis markers and the right-side location list.
    sites: Vec<i64>,
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
    selector: &Selector,
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

        // Optional: enables `--regions` filtering on the parquet track.
        let chr_col = batch
            .column_by_name("chr")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());

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
            let pos = pos_col.value(i);
            let chr_val = chr_col.map(|c| c.value(i)).unwrap_or("");
            if !selector.selects(gene_val, chr_val, pos) {
                continue;
            }

            *distinct_genes.entry(gene_val.into()).or_insert(0) += 1;

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
            "no sites matching {} in {}",
            selector.describe(),
            site_file
        ));
    }

    // Aggregate every matched gene's sites; gene boundaries below widen to
    // the span across all of them rather than erroring on ambiguity.
    if distinct_genes.len() > 1 {
        info!(
            "{} matched {} genes in parquet; aggregating all sites",
            selector.describe(),
            distinct_genes.len()
        );
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

/// Matrix positions grouped by cell type (the figure's per-panel top
/// track). With `membership = None` every cell falls into a single
/// synthetic `""` group, which is exactly the all-cells aggregate the
/// ASCII path uses — so [`read_matrix_positions`] is a thin wrapper.
struct GroupedMatrix {
    gene: Box<str>,
    chr: Box<str>,
    /// celltype label -> sorted `(pos, value)`
    by_group: FxHashMap<Box<str>, Vec<(i64, f64)>>,
}

fn read_matrix_positions_grouped(
    data_files: &[Box<str>],
    selector: &Selector,
    signal: &PileupSignal,
    membership: Option<&CellMembership>,
    top_modality: &[Box<str>],
) -> anyhow::Result<GroupedMatrix> {
    // Per group: pos -> aggregate. Multiple input files (e.g. replicates)
    // merge per genomic position; gene/chr labels reflect the union.
    let mut by_group: FxHashMap<Box<str>, FxHashMap<i64, PosAgg>> = FxHashMap::default();
    let mut distinct_genes: FxHashMap<Box<str>, usize> = FxHashMap::default();
    let mut matched_chrs: Vec<Box<str>> = Vec::new();
    let mut total_matched = 0usize;

    let modality_filter: Option<Vec<String>> = if top_modality.is_empty() {
        None
    } else {
        Some(
            top_modality
                .iter()
                .map(|m| m.to_ascii_lowercase())
                .collect(),
        )
    };

    for data_file in data_files {
        let (backend, resolved_path) = resolve_backend_file(data_file, None)?;
        let data = open_sparse_matrix(&resolved_path, &backend)?;

        let row_names = data.row_names()?;

        let mut matched_rows: Vec<(usize, i64)> = Vec::new();
        for (idx, name) in row_names.iter().enumerate() {
            if let Some((gene_part, modality, chr, pos)) = parse_row_name_full(name) {
                if let Some(ref mf) = modality_filter {
                    if !mf.contains(&modality.to_ascii_lowercase()) {
                        continue;
                    }
                }
                if selector.selects(gene_part, chr, pos) {
                    *distinct_genes.entry(gene_part.into()).or_insert(0) += 1;
                    matched_chrs.push(chr.into());
                    matched_rows.push((idx, pos));
                }
            }
        }

        if matched_rows.is_empty() {
            continue;
        }
        total_matched += matched_rows.len();

        // Column index -> cell type (None to drop). Only needed when
        // stratifying; the all-cells path skips reading column names.
        let col_groups: Option<Vec<Option<Box<str>>>> = match membership {
            None => None,
            Some(m) => {
                let col_names = data.column_names()?;
                Some(
                    col_names
                        .iter()
                        .map(|bc| m.matches_barcode(&CellBarcode::Barcode(Arc::from(bc.as_ref()))))
                        .collect(),
                )
            }
        };

        let local_to_pos: Vec<i64> = matched_rows.iter().map(|(_, pos)| *pos).collect();
        let row_indices: Vec<usize> = matched_rows.iter().map(|(idx, _)| *idx).collect();
        let (_nrow, _ncol, triplets) = data.read_triplets_by_rows(row_indices)?;

        // All-cells: pre-seed every matched position so zero-signal sites
        // still appear (axis markers), matching the legacy behavior.
        if membership.is_none() {
            let g = by_group.entry("".into()).or_default();
            for &pos in &local_to_pos {
                g.entry(pos).or_insert(PosAgg { sum: 0.0, nnz: 0 });
            }
        }

        for (row, col, val) in &triplets {
            let local_idx = *row as usize;
            if local_idx >= local_to_pos.len() || *val == 0.0 {
                continue;
            }
            let group: Box<str> = match &col_groups {
                None => "".into(),
                Some(cg) => match cg.get(*col as usize).and_then(|o| o.clone()) {
                    Some(ct) => ct,
                    None => continue,
                },
            };
            let pos = local_to_pos[local_idx];
            let agg = by_group
                .entry(group)
                .or_default()
                .entry(pos)
                .or_insert(PosAgg { sum: 0.0, nnz: 0 });
            agg.sum += *val as f64;
            agg.nnz += 1;
        }
    }

    if total_matched == 0 {
        return Err(anyhow::anyhow!(
            "no rows matching {} in {} file(s)",
            selector.describe(),
            data_files.len()
        ));
    }

    // The selection can span several genes (and chromosomes); rather than
    // erroring on ambiguity we pile them together and label the aggregate.
    let gene: Box<str> = summarize_genes(&distinct_genes);
    let chr: Box<str> = summarize_chr(&matched_chrs);

    info!(
        "matched {} rows across {} gene(s) in {} file(s) for {} [{}]",
        total_matched,
        distinct_genes.len(),
        data_files.len(),
        selector.describe(),
        gene
    );

    let by_group = by_group
        .into_iter()
        .map(|(grp, pos_agg)| {
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
            (grp, positions)
        })
        .collect();

    Ok(GroupedMatrix {
        gene,
        chr,
        by_group,
    })
}

fn read_matrix_positions(
    data_files: &[Box<str>],
    selector: &Selector,
    signal: &PileupSignal,
) -> anyhow::Result<MatrixGeneData> {
    let grouped = read_matrix_positions_grouped(data_files, selector, signal, None, &[])?;
    // membership = None yields exactly one synthetic "" group.
    let positions = grouped
        .by_group
        .into_iter()
        .next()
        .map(|(_, p)| p)
        .unwrap_or_default();
    Ok(MatrixGeneData {
        gene: grouped.gene,
        chr: grouped.chr,
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
    BinEdges::new(min_pos, max_pos, num_bins).bin(positions, log_transform)
}

/// Distinct coordinates from a position/value list (already sorted by
/// position), for the axis markers and right-side location list.
fn distinct_positions(positions: &[(i64, f64)]) -> Vec<i64> {
    let mut out: Vec<i64> = positions.iter().map(|(p, _)| *p).collect();
    out.dedup();
    out
}

/// Group digits in threes for readability: `26781984` -> `26,781,984`.
pub(crate) fn fmt_thousands(n: i64) -> String {
    let digits = n.unsigned_abs().to_string();
    let bytes = digits.as_bytes();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    if n < 0 {
        out.push('-');
    }
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(*b as char);
    }
    out
}

/// Map a coordinate to its bin column under the same binning rule as
/// [`bin_positions_with_extent`].
fn pos_to_col(pos: i64, min_pos: i64, max_pos: i64, num_bins: usize) -> usize {
    BinEdges::new(min_pos, max_pos, num_bins).col_of(pos)
}

/// Right-side legend listing each site location top-to-bottom (genomic
/// order, mirroring the `+` marks left-to-right). First line is a title;
/// the list is capped to the rows available, with an overflow note.
fn build_site_legend(pileup: &BinnedPileup, height: usize) -> Vec<String> {
    if pileup.sites.is_empty() {
        return Vec::new();
    }
    // Mixture rows have no chromosome — the "positions" are component ordinals.
    let kind = if pileup.chr.is_empty() || pileup.chr.as_ref() == "component" {
        "components".to_string()
    } else {
        format!("sites @ {}", pileup.chr)
    };
    let mut out = vec![format!("{} ({}):", kind, pileup.sites.len())];

    let capacity = height.saturating_sub(1); // rows left under the title
    let n = pileup.sites.len();
    if n <= capacity {
        out.extend(pileup.sites.iter().map(|&p| fmt_thousands(p)));
    } else {
        let shown = capacity.saturating_sub(1);
        out.extend(pileup.sites.iter().take(shown).map(|&p| fmt_thousands(p)));
        out.push(format!("... (+{} more)", n - shown));
    }
    out
}

fn print_vertical_histogram(pileup: &BinnedPileup, height: usize) {
    let header = format!(
        "  {}  {}:{}-{}  [{}] signal: {}  sites: {}",
        pileup.gene,
        pileup.chr,
        fmt_thousands(pileup.min_pos),
        fmt_thousands(pileup.max_pos),
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

    // Locations listed down the right of the plot (one per row), so the
    // x-axis only needs `+` markers rather than crowded text.
    let legend = build_site_legend(pileup, height);

    for (i, row) in (1..=height).rev().enumerate() {
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

        let mut line = format!("{label} |{bar}");
        if let Some(entry) = legend.get(i) {
            line.push_str("   ");
            line.push_str(entry);
        }
        eprintln!("{}", line.trim_end());
    }

    // Axis line: `+` at every column that holds a site, `-` elsewhere.
    let mut axis = vec!['-'; pileup.bins.len()];
    for &pos in &pileup.sites {
        axis[pos_to_col(pos, pileup.min_pos, pileup.max_pos, pileup.bins.len())] = '+';
    }
    let axis: String = axis.into_iter().collect();
    eprintln!("{} +{}", " ".repeat(label_width), axis);
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
    let selector = Selector::build(&args.genes, &args.regions)?;

    // Figure mode is triggered by any figure-only input/output flag.
    // Otherwise fall through to the original ASCII / TSV path unchanged.
    let figure_mode = args.gtf.is_some()
        || !args.bam_files.is_empty()
        || args.format.is_some()
        || args.svg
        || args.png;
    if figure_mode {
        return run_miami_figure(args, &selector);
    }

    // Read sparse matrix data
    let mtx = read_matrix_positions(&args.data_files, &selector, &args.signal)?;
    let matrix_num_sites = mtx.positions.len();

    // Determine x-axis extent and build site track if parquet provided
    let site_annotation = args
        .site_file
        .as_ref()
        .map(|sf| read_site_annotation(sf, &selector, &args.site_signal))
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
        sites: distinct_positions(&sa.positions),
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
        sites: distinct_positions(&mtx.positions),
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

/// Faceted Miami-plot path: stratified matrix epi sites (top), GTF gene
/// model (middle), and BAM read depth (bottom), one panel per cell type.
fn run_miami_figure(args: &PileupArgs, selector: &Selector) -> anyhow::Result<()> {
    // Optional cell-type membership for faceting. allow_prefix = !exact.
    let membership = match &args.cell_membership_file {
        Some(p) => Some(CellMembership::from_file(
            p,
            args.membership_barcode_col,
            args.membership_celltype_col,
            !args.exact_barcode_match,
        )?),
        None => None,
    };

    // Top track: stratified matrix epi sites.
    let grouped = read_matrix_positions_grouped(
        &args.data_files,
        selector,
        &args.signal,
        membership.as_ref(),
        &args.top_modality,
    )?;

    // Optional parquet refines gene bounds in figure mode.
    let site_annotation = args
        .site_file
        .as_ref()
        .map(|sf| read_site_annotation(sf, selector, &args.site_signal))
        .transpose()?;

    // Middle track: gene model(s) from GTF.
    let models = match &args.gtf {
        Some(gtf) => load_gene_models(gtf, selector)?,
        None => Vec::new(),
    };

    // Shared extent = union of gene-model footprint, all matrix sites, and
    // parquet bounds — so the whole model and every site are visible.
    let mut lo = i64::MAX;
    let mut hi = i64::MIN;
    if let Some((_, mlo, mhi)) = models_extent(&models) {
        lo = lo.min(mlo);
        hi = hi.max(mhi);
    }
    for positions in grouped.by_group.values() {
        for &(p, _) in positions {
            lo = lo.min(p);
            hi = hi.max(p);
        }
    }
    if let Some(sa) = &site_annotation {
        lo = lo.min(sa.gene_start);
        hi = hi.max(sa.gene_stop);
    }
    if lo > hi {
        anyhow::bail!(
            "no genomic coordinates to plot for {} (need positional matrix rows, a GTF, or a sites parquet)",
            selector.describe()
        );
    }

    let edges = BinEdges::new(lo, hi, args.num_bins);

    // Region chromosome: prefer the matrix chr (same origin as the BAM),
    // else the GTF gene's chr.
    let region_chr: Box<str> = if grouped.chr.as_ref() != "*"
        && grouped.chr.as_ref() != "component"
        && !grouped.chr.is_empty()
    {
        grouped.chr.clone()
    } else if let Some(m) = models.first() {
        m.chr.clone()
    } else {
        grouped.chr.clone()
    };

    // Bottom track: read depth from BAM, stratified by cell type.
    let depth_by_group = if args.bam_files.is_empty() {
        FxHashMap::default()
    } else {
        let region = Bed {
            chr: region_chr.clone(),
            start: lo,
            stop: hi + 1,
        };
        read_depth_binned(
            &args.bam_files,
            &region,
            &edges,
            &args.cell_barcode_tag,
            membership.as_ref(),
        )?
    };

    // Panel order: membership cell types (sorted) or one all-cells panel.
    let celltypes: Vec<Box<str>> = match &membership {
        Some(m) => {
            let mut v = m.cell_types();
            v.sort();
            v
        }
        None => vec!["".into()],
    };

    let is_log = matches!(args.signal, PileupSignal::Log10Sum);
    let mut panels: Vec<PanelData> = Vec::with_capacity(celltypes.len());
    for ct in &celltypes {
        let raw = grouped.by_group.get(ct).cloned().unwrap_or_default();
        let epi_sites = if is_log {
            raw.into_iter()
                .map(|(p, v)| (p, (1.0 + v).log10()))
                .collect()
        } else {
            raw
        };
        let depth_bins = depth_by_group
            .get(ct)
            .cloned()
            .unwrap_or_else(|| vec![0.0; edges.num_bins]);
        panels.push(PanelData {
            celltype: ct.clone(),
            epi_sites,
            depth_bins,
        });
    }

    // Output formats: SVG always (unless `--format pdf`), PDF default-on.
    let (want_svg, want_pdf, want_png) = match &args.format {
        Some(FigFormat::Svg) => (true, false, args.png),
        Some(FigFormat::Pdf) => (args.svg, true, args.png),
        None => (true, !args.no_pdf, args.png),
    };

    let out_prefix: Box<str> = args
        .out
        .clone()
        .unwrap_or_else(|| slug(&grouped.gene).into());

    let top_label: Box<str> = if args.top_modality.is_empty() {
        "epi sites".into()
    } else {
        args.top_modality
            .iter()
            .map(|m| m.as_ref())
            .collect::<Vec<_>>()
            .join("/")
            .into()
    };

    let title: Box<str> = format!(
        "{}  {}:{}-{}",
        grouped.gene,
        region_chr,
        fmt_thousands(lo),
        fmt_thousands(hi)
    )
    .into();

    let opts = FigOpts {
        out_prefix,
        width_in: args.fig_width,
        dpi: args.dpi,
        palette: args.palette.clone(),
        want_svg,
        want_png,
        want_pdf,
        raster_threshold: args.raster_threshold,
        title,
        top_label,
    };

    let n = render_miami(&panels, &models, &edges, &opts)?;
    info!(
        "rendered Miami plot: {} panel(s), {} file(s)",
        panels.len(),
        n
    );
    Ok(())
}

/// Filesystem-safe slug for a default figure output prefix.
fn slug(s: &str) -> String {
    let out: String = s
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    if out.is_empty() {
        "miami".to_string()
    } else {
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_site_and_mixture_rows() {
        // site output: gene/modality/chr:pos
        assert_eq!(
            parse_row_name_full("ENSG00000139618_BRCA2/m6A/chr13:32350000"),
            Some(("ENSG00000139618_BRCA2", "m6A", "chr13", 32350000))
        );
        // mixture output: gene/modality/component
        assert_eq!(
            parse_row_name_full("ENSG00000060558_GNA15/m6A/0"),
            Some(("ENSG00000060558_GNA15", "m6A", "", 0))
        );
        // count rows (detail neither chr:pos nor an integer) don't parse
        assert_eq!(parse_row_name_full("gene_0/count/spliced"), None);
    }

    #[test]
    fn relaxed_gene_matching() {
        let gp = "ENSG00000060558_GNA15";
        // symbol, any case
        assert!(gene_matches("GNA15", &query_symbol("GNA15"), gp));
        assert!(gene_matches("gna15", &query_symbol("gna15"), gp));
        // Ensembl ID
        assert!(gene_matches(
            "ENSG00000060558",
            &query_symbol("ENSG00000060558"),
            gp
        ));
        // full composite
        assert!(gene_matches(gp, &query_symbol(gp), gp));
        // partial substring must NOT match (consistent with aux-data scheme)
        assert!(!gene_matches("GNA", &query_symbol("GNA"), gp));
        assert!(!gene_matches(
            "RPL",
            &query_symbol("RPL"),
            "ENSG00000063177_RPL18"
        ));
    }

    #[test]
    fn region_parsing_and_matching() {
        let r = parse_region("chr17:1000-2000").unwrap();
        assert_eq!((r.chr.as_ref(), r.lb, r.ub), ("chr17", 1000, 2000));
        // reversed bounds are normalized
        let r = parse_region("17:2000-1000").unwrap();
        assert_eq!((r.lb, r.ub), (1000, 2000));
        // malformed specs error
        assert!(parse_region("chr17").is_err());
        assert!(parse_region("chr17:1000").is_err());
        assert!(parse_region(":1-2").is_err());

        // chr-prefix tolerant (shared genomic_data::chr_eq)
        assert!(chr_eq("chr17", "17"));
        assert!(chr_eq("17", "17"));
        assert!(!chr_eq("chr1", "chr2"));
    }

    #[test]
    fn selector_gene_or_region_union() {
        let sel = Selector::build(&["GNA15".into()], &["chr17:100-200".into()]).unwrap();
        // gene branch (mixture row: no chr, component ordinal)
        assert!(sel.selects("ENSG00000060558_GNA15", "", 0));
        // region branch (different gene, but inside the window)
        assert!(sel.selects("ENSG1_OTHER", "chr17", 150));
        // outside both
        assert!(!sel.selects("ENSG1_OTHER", "chr17", 999));
        assert!(!sel.selects("ENSG1_OTHER", "chr9", 150));

        // at least one selector is required
        assert!(Selector::build(&[], &[]).is_err());
        // region-only is fine
        assert!(Selector::build(&[], &["chr1:1-9".into()]).is_ok());
    }

    #[test]
    fn thousands_and_axis_mapping() {
        assert_eq!(fmt_thousands(26781984), "26,781,984");
        assert_eq!(fmt_thousands(767), "767");
        assert_eq!(fmt_thousands(0), "0");
        assert_eq!(fmt_thousands(-1234), "-1,234");

        // first site -> first column, last site -> last column
        assert_eq!(pos_to_col(100, 100, 200, 10), 0);
        assert_eq!(pos_to_col(200, 100, 200, 10), 9);
        assert_eq!(pos_to_col(150, 100, 200, 10), 5);
        // degenerate single-position extent collapses to column 0
        assert_eq!(pos_to_col(100, 100, 100, 10), 0);
    }

    #[test]
    fn distinct_positions_dedups_sorted() {
        let p = [(10, 1.0), (10, 2.0), (20, 0.5), (30, 0.0)];
        assert_eq!(distinct_positions(&p), vec![10, 20, 30]);
    }

    #[test]
    fn aggregate_labels() {
        let mut one: FxHashMap<Box<str>, usize> = FxHashMap::default();
        one.insert("ENSG1_GNA15".into(), 2);
        assert_eq!(summarize_genes(&one).as_ref(), "ENSG1_GNA15");

        let mut many: FxHashMap<Box<str>, usize> = FxHashMap::default();
        many.insert("ENSG1_A".into(), 1);
        many.insert("ENSG2_B".into(), 3);
        let label = summarize_genes(&many);
        assert!(label.starts_with("2 genes: "), "got {label}");

        // chr label: empty -> component, single -> that chr, mixed -> *
        let mix: Vec<Box<str>> = vec!["".into(), "".into()];
        assert_eq!(summarize_chr(&mix).as_ref(), "component");
        let single: Vec<Box<str>> = vec!["chr13".into()];
        assert_eq!(summarize_chr(&single).as_ref(), "chr13");
        let multi: Vec<Box<str>> = vec!["chr1".into(), "chr2".into()];
        assert_eq!(summarize_chr(&multi).as_ref(), "*");
    }
}
