//! Long-format eQTL summary-statistics reader.
//!
//! One row of input is one (variant, gene, celltype) observation. Column
//! names are configurable; a leading `#` on the header's first column is
//! tolerated. Files that cannot be opened or decoded — the target data has
//! corrupt gz members — are skipped with a warning and counted, never
//! partially ingested.
//!
//! Variants are interned GLOBALLY as a canonical locus — `chromosome`
//! and `position` folded by [`canon_locus`], the workspace's locus rule, so
//! `chr5:1000` and `5:1000` are one name and a fagioli variant id is spelled
//! like a locus anywhere else in the workspace. The same locus therefore
//! carries one identity across every gene it was tested against.
//! Selection needs that: a variant chosen for one gene keeps its rows for
//! all the other genes in the window, and those rows are the gene-swap
//! negative pool.
//!
//! Future stage-1 formats (differently reported fine-mapping posteriors)
//! land here as reader-level importers, not as model changes: downstream
//! code sees only [`QtlEntry`] records.

use anyhow::{anyhow, ensure, Result};
use auxiliary_data::feature_names::FeatureNameKind;
use genomic_data::gff::parse_ensembl_id;
use log::{info, warn};
use matrix_util::membership::canon_locus;
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;
use std::fmt::Write as _;
use std::io::{BufRead, BufReader};

/// Input column names, matched exactly against the header fields after a
/// leading `#` on the first field is stripped.
#[derive(Debug, Clone)]
pub struct QtlColumns {
    pub gene: String,
    pub celltype: String,
    pub chromosome: String,
    pub position: String,
    pub beta: String,
    pub se: String,
    /// Optional nonnegative per-row weight (PIP-like). Rows from files
    /// without this column carry no weight.
    pub pip: String,
}

/// One (variant, gene, celltype) record after column mapping. Both the
/// variant and the celltype index are global.
#[derive(Debug, Clone)]
pub struct QtlEntry {
    pub variant: u32,
    pub celltype: u32,
    pub beta: f32,
    pub se: f32,
    /// Any nonnegative per-row weight; `None` when the source file had no
    /// such column or the value was unreadable.
    pub pip_weight: Option<f32>,
}

/// All retained rows of one gene.
#[derive(Debug, Default)]
pub struct GeneTable {
    /// Sorted by (variant, celltype); at most one entry per pair.
    pub entries: Vec<QtlEntry>,
}

/// Everything read: gene-major tables plus the hygiene counters.
#[derive(Debug)]
pub struct QtlData {
    pub genes: Vec<Box<str>>,
    pub celltypes: Vec<Box<str>>,
    /// Global variant keys, canonical loci (`chromosome_position`).
    pub variants: Vec<Box<str>>,
    pub tables: Vec<GeneTable>,
    pub n_files_read: usize,
    pub n_files_skipped: usize,
    pub n_rows_dropped: usize,
    pub n_rows_duplicate: usize,
    pub n_genes_dropped: usize,
}

impl QtlData {
    /// Total retained rows across all genes.
    pub fn n_rows(&self) -> usize {
        self.tables.iter().map(|t| t.entries.len()).sum()
    }
}

/// A parsed row of one file, held locally until the whole file decodes; a
/// truncated gz member must not leave half a file behind.
///
/// The three ids are FILE-LOCAL. Names repeat across nearly every row — one
/// gene appears once per (variant, celltype) — so interning per file and
/// translating whole dictionaries afterwards costs one allocation per
/// distinct name instead of three per row.
struct FileRow {
    gene: u32,
    celltype: u32,
    variant: u32,
    beta: f32,
    se: f32,
    pip_weight: Option<f32>,
}

/// One file's rows, its local dictionaries, and the count dropped by hygiene.
struct ParsedFile {
    genes: Vec<Box<str>>,
    celltypes: Vec<Box<str>>,
    variants: Vec<Box<str>>,
    rows: Vec<FileRow>,
    n_dropped: usize,
}

/// Name -> dense id. The map owns the only copy of each name; `into_names`
/// inverts it at the end, so a name is allocated once rather than twice.
#[derive(Default)]
struct Interner {
    map: HashMap<Box<str>, u32>,
}

impl Interner {
    fn intern(&mut self, name: &str) -> u32 {
        if let Some(&idx) = self.map.get(name) {
            return idx;
        }
        let idx = self.map.len() as u32;
        self.map.insert(Box::from(name), idx);
        idx
    }

    fn len(&self) -> usize {
        self.map.len()
    }

    fn into_names(self) -> Vec<Box<str>> {
        let mut names: Vec<Box<str>> = vec![Box::from(""); self.map.len()];
        for (name, idx) in self.map {
            names[idx as usize] = name;
        }
        names
    }
}

/// Compact an id space to the entries actually used.
///
/// Returns the surviving names and an old -> new remap carrying `u32::MAX`
/// where an entity was dropped.
fn compact(names: Vec<Box<str>>, used: &[bool]) -> (Vec<Box<str>>, Vec<u32>) {
    let mut remap = vec![u32::MAX; used.len()];
    let mut kept: Vec<Box<str>> = Vec::new();
    for (old, name) in names.into_iter().enumerate() {
        if used[old] {
            remap[old] = kept.len() as u32;
            kept.push(name);
        }
    }
    (kept, remap)
}

/// Gene identity under the workspace's shared naming rule, plus the Ensembl
/// annotation version.
///
/// Two independent things can spell one gene two ways, and the workspace
/// already answers each separately:
///
/// - **composite names** — `ENSG00000000002_SYM1` vs `SYM1`. That is
///   [`FeatureNameKind`], the same rule `senna` exposes as
///   `--feature-name-kind`, so a gene axis here means what it means there.
/// - **annotation version** — `ENSG00000000001.17` vs `.12`, from two GENCODE
///   releases. [`parse_ensembl_id`] is the workspace helper for that, and it
///   is applied only to `ENS`-prefixed ids: clone-based SYMBOLS such as
///   `CLONE1.4` also carry a dot but are DISTINCT genes.
fn canonical_gene(name: &str, kind: &FeatureNameKind) -> Box<str> {
    let name = name.trim();
    let versionless = if name.starts_with("ENS") {
        parse_ensembl_id(name).unwrap_or(name)
    } else {
        name
    };
    kind.canonicalize(versionless)
}

/// Read one or more TSV(.gz) files into gene-major tables.
///
/// Row hygiene: rows with unparseable numbers, non-finite beta or SE,
/// SE <= 0, or SE above `max_se` are dropped and counted. Duplicate
/// (variant, gene, celltype) rows keep the entry with the larger weight.
/// Genes observed in fewer than `min_celltypes` cell types are dropped.
///
/// `name_kind` is the workspace's shared feature-naming rule
/// ([`FeatureNameKind`]); variants are always canonicalized as loci.
pub fn read_qtl_files(
    files: &[String],
    cols: &QtlColumns,
    name_kind: &FeatureNameKind,
    min_celltypes: usize,
    max_se: Option<f64>,
) -> Result<QtlData> {
    let mut genes = Interner::default();
    let mut celltypes = Interner::default();
    let mut variants = Interner::default();
    let mut tables: Vec<GeneTable> = Vec::new();
    // Per gene: (variant, celltype) -> index into entries, for deduplication.
    let mut entry_maps: Vec<HashMap<(u32, u32), usize>> = Vec::new();

    let mut n_files_read = 0usize;
    let mut n_files_skipped = 0usize;
    let mut n_rows_dropped = 0usize;
    let mut n_rows_duplicate = 0usize;
    let mut n_rows_kept = 0usize;

    // Parsing (decompress + float scan) dominates the cost and is independent
    // per file, so it runs across cores; interning and deduplication share
    // mutable state and stay serial. Batching bounds peak memory: only one
    // batch of parsed rows is resident at a time.
    const PARSE_BATCH: usize = 64;

    for batch in files.chunks(PARSE_BATCH) {
        let parsed: Vec<(&String, Result<ParsedFile>)> = batch
            .par_iter()
            .map(|path| (path, parse_file(path, cols, name_kind, max_se)))
            .collect();

        for (path, outcome) in parsed {
            let file = match outcome {
                Ok(parsed) => parsed,
                Err(e) => {
                    warn!("Skipping unreadable file {}: {}", path, e);
                    n_files_skipped += 1;
                    continue;
                }
            };
            n_files_read += 1;
            n_rows_dropped += file.n_dropped;

            // Translate this file's dictionaries once, then every row is
            // integer indexing.
            let gmap: Vec<u32> = file.genes.iter().map(|n| genes.intern(n)).collect();
            let cmap: Vec<u32> = file.celltypes.iter().map(|n| celltypes.intern(n)).collect();
            let vmap: Vec<u32> = file.variants.iter().map(|n| variants.intern(n)).collect();
            while tables.len() < genes.len() {
                tables.push(GeneTable::default());
                entry_maps.push(HashMap::default());
            }

            for row in file.rows {
                let g = gmap[row.gene as usize] as usize;
                let v = vmap[row.variant as usize];
                let k = cmap[row.celltype as usize];

                let entry = QtlEntry {
                    variant: v,
                    celltype: k,
                    beta: row.beta,
                    se: row.se,
                    pip_weight: row.pip_weight,
                };
                match entry_maps[g].entry((v, k)) {
                    std::collections::hash_map::Entry::Vacant(slot) => {
                        slot.insert(tables[g].entries.len());
                        tables[g].entries.push(entry);
                        n_rows_kept += 1;
                    }
                    std::collections::hash_map::Entry::Occupied(slot) => {
                        n_rows_duplicate += 1;
                        let held = &mut tables[g].entries[*slot.get()];
                        if entry.pip_weight.unwrap_or(-1.0) > held.pip_weight.unwrap_or(-1.0) {
                            *held = entry;
                        }
                    }
                }
            }
        }
    }
    drop(entry_maps);

    if n_files_skipped > 0 {
        warn!(
            "Skipped {} of {} files as unreadable or truncated",
            n_files_skipped,
            files.len()
        );
    }

    // ── Gene filter: enough cell types ───────────────────────────────────
    let n_celltypes = celltypes.len();
    let n_variants = variants.len();
    let mut kept_genes: Vec<Box<str>> = Vec::new();
    let mut kept_tables: Vec<GeneTable> = Vec::new();
    let mut n_genes_dropped = 0usize;
    let mut ct_seen = vec![false; n_celltypes];
    let mut ct_used = vec![false; n_celltypes];

    for (name, table) in genes.into_names().into_iter().zip(tables) {
        ct_seen.iter_mut().for_each(|s| *s = false);
        for e in &table.entries {
            ct_seen[e.celltype as usize] = true;
        }
        let n_ct = ct_seen.iter().filter(|&&s| s).count();
        if n_ct < min_celltypes || table.entries.is_empty() {
            n_genes_dropped += 1;
            continue;
        }
        for (used, &seen) in ct_used.iter_mut().zip(&ct_seen) {
            *used |= seen;
        }
        kept_genes.push(name);
        kept_tables.push(table);
    }

    // Sorting is the dominant per-gene cost and every gene is independent.
    kept_tables
        .par_iter_mut()
        .for_each(|t| t.entries.sort_unstable_by_key(|e| (e.variant, e.celltype)));

    let mut variant_used = vec![false; n_variants];
    for table in &kept_tables {
        for e in &table.entries {
            variant_used[e.variant as usize] = true;
        }
    }

    ensure!(
        !kept_genes.is_empty(),
        "No usable gene survived reading: {} files read, {} skipped, \
         {} rows dropped. Check the --col-* names against the input header.",
        n_files_read,
        n_files_skipped,
        n_rows_dropped,
    );

    // ── Compact cell types and variants to those used by kept genes ──────
    let (kept_celltypes, ct_remap) = compact(celltypes.into_names(), &ct_used);
    let (kept_variants, variant_remap) = compact(variants.into_names(), &variant_used);
    kept_tables.par_iter_mut().for_each(|table| {
        for e in &mut table.entries {
            e.celltype = ct_remap[e.celltype as usize];
            e.variant = variant_remap[e.variant as usize];
        }
    });

    info!(
        "Read {} rows: {} genes x {} variants x {} cell types ({} files read, {} skipped, \
         {} rows dropped, {} duplicates collapsed, {} genes below --min-celltypes {})",
        n_rows_kept,
        kept_genes.len(),
        kept_variants.len(),
        kept_celltypes.len(),
        n_files_read,
        n_files_skipped,
        n_rows_dropped,
        n_rows_duplicate,
        n_genes_dropped,
        min_celltypes,
    );

    Ok(QtlData {
        genes: kept_genes,
        celltypes: kept_celltypes,
        variants: kept_variants,
        tables: kept_tables,
        n_files_read,
        n_files_skipped,
        n_rows_dropped,
        n_rows_duplicate,
        n_genes_dropped,
    })
}

/// Open plain or gzipped input. The real eQTL blocks are MULTI-member gzip
/// (one member per write flush), which the plain single-member decoder
/// silently truncates to its first member — here, the header line alone.
fn open_qtl_reader(path: &str) -> Result<Box<dyn BufRead>> {
    let file = std::fs::File::open(path)?;
    if path.ends_with(".gz") {
        let decoder = flate2::read::MultiGzDecoder::new(file);
        Ok(Box::new(BufReader::new(decoder)))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

// ── Column resolution ───────────────────────────────────────────────────
//
// Same shape as pinto's coordinate reader (`read_one_coord_file`): accept
// several delimiters, resolve columns BY NAME, and fall back to a known
// convention rather than failing. Here the fallback is a per-field alias
// list instead of fixed indices, because summary statistics are named
// tables and their column ORDER is not conventional the way Visium's
// `tissue_positions` is.
//
// Streaming stays hand-rolled rather than going through
// `matrix_util::common_io::read_lines_of_words_delim`: that returns every
// row as an allocated `Vec<Box<str>>`, which is exactly the per-row
// allocation this reader exists to avoid at tens of millions of rows.

/// Spellings accepted for each column, most conventional first.
/// Drawn from GTEx, eQTL Catalogue, plain SuSiE output and in-house
/// fine-mapping pipelines.
///
/// **The first entry of each list is that column's `--col-*` default**, and
/// that is load-bearing: aliases are tried only when the caller left the flag
/// alone. An explicit `--col-beta effect_size` that finds nothing is a
/// mistake worth reporting, not a reason to guess. Same contract as pinto's
/// coordinate reader, which falls back only when no index was requested.
const GENE_ALIASES: &[&str] = &[
    "gene",
    "gene_id",
    "phenotype_id",
    "molecular_trait_id",
    "trait",
];
const CELLTYPE_ALIASES: &[&str] = &[
    "celltype",
    "cell_type",
    "cell",
    "context",
    "tissue",
    "condition",
];
const CHROMOSOME_ALIASES: &[&str] = &["chromosome", "chr", "chrom", "#chrom", "seqnames"];
const POSITION_ALIASES: &[&str] = &[
    "physical.pos",
    "position",
    "pos",
    "bp",
    "start",
    "variant_pos",
];
const BETA_ALIASES: &[&str] = &["beta", "b", "effect", "slope", "effect_size", "es"];
const SE_ALIASES: &[&str] = &[
    "se",
    "standard_error",
    "stderr",
    "slope_se",
    "beta_se",
    "sd",
];
const PIP_ALIASES: &[&str] = &["alpha", "pip", "posterior_prob", "weight"];

/// The delimiter this file uses, chosen as the one splitting the header
/// into the most fields. A single-field header means one column, and tab
/// is as good a guess as any.
fn detect_delimiter(header: &str) -> char {
    ['\t', ',', ';', ' ']
        .into_iter()
        .max_by_key(|&d| header.split(d).count())
        .unwrap_or('\t')
}

/// Case-insensitive exact match on a header field.
fn find(fields: &[&str], name: &str) -> Option<usize> {
    fields.iter().position(|f| f.eq_ignore_ascii_case(name))
}

/// Aliases apply only while the caller is on the default spelling.
fn find_any(fields: &[&str], requested: &str, aliases: &[&str]) -> Option<usize> {
    let defaulted = aliases
        .first()
        .is_some_and(|d| requested.eq_ignore_ascii_case(d));
    find(fields, requested).or_else(|| {
        defaulted
            .then(|| aliases.iter().find_map(|n| find(fields, n)))
            .flatten()
    })
}

/// Resolve one required column: the caller's `--col-*` name first, then the
/// known aliases. A failure names the role, what was tried, and the header
/// that was actually seen, so the fix is visible without opening the file.
fn need(fields: &[&str], requested: &str, aliases: &[&str], role: &str) -> Result<usize> {
    find_any(fields, requested, aliases).ok_or_else(|| {
        anyhow!(
            "no {} column: tried '{}' and {:?}, header has {:?}",
            role,
            requested,
            aliases,
            fields
        )
    })
}

/// Parse one file completely, or fail as a whole. Names are interned into
/// file-local dictionaries; the caller translates them to global ids.
fn parse_file(
    path: &str,
    cols: &QtlColumns,
    name_kind: &FeatureNameKind,
    max_se: Option<f64>,
) -> Result<ParsedFile> {
    let reader = open_qtl_reader(path)?;
    let mut lines = reader.lines();

    // First non-empty line is the header.
    let header = loop {
        match lines.next() {
            Some(line) => {
                let line = line?;
                if !line.trim().is_empty() {
                    break line;
                }
            }
            None => return Err(anyhow!("empty file")),
        }
    };

    let delim = detect_delimiter(&header);
    let mut fields: Vec<&str> = header.split(delim).map(str::trim).collect();
    // Tolerate a leading '#' on the first header column.
    fields[0] = fields[0].trim_start_matches('#');

    let idx_gene = need(&fields, &cols.gene, GENE_ALIASES, "gene")?;
    let idx_celltype = need(&fields, &cols.celltype, CELLTYPE_ALIASES, "cell type")?;
    let idx_chromosome = need(&fields, &cols.chromosome, CHROMOSOME_ALIASES, "chromosome")?;
    let idx_position = need(&fields, &cols.position, POSITION_ALIASES, "position")?;
    let idx_beta = need(&fields, &cols.beta, BETA_ALIASES, "effect size")?;
    let idx_se = need(&fields, &cols.se, SE_ALIASES, "standard error")?;
    let idx_pip = find_any(&fields, &cols.pip, PIP_ALIASES);

    let min_fields = [
        idx_gene,
        idx_celltype,
        idx_chromosome,
        idx_position,
        idx_beta,
        idx_se,
    ]
    .into_iter()
    .chain(idx_pip)
    .max()
    .unwrap_or(0)
        + 1;
    let max_se = max_se.unwrap_or(f64::INFINITY) as f32;

    let mut rows: Vec<FileRow> = Vec::new();
    let mut n_dropped = 0usize;
    let mut genes = Interner::default();
    let mut celltypes = Interner::default();
    let mut variants = Interner::default();
    // Reused across rows: the variant key is only allocated when it is new.
    let mut key = String::new();

    for line in lines {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(delim).map(str::trim).collect();
        if fields.len() < min_fields {
            n_dropped += 1;
            continue;
        }
        let parsed = (
            fields[idx_position].parse::<u64>(),
            fields[idx_beta].parse::<f32>(),
            fields[idx_se].parse::<f32>(),
        );
        let (Ok(position), Ok(beta), Ok(se)) = parsed else {
            n_dropped += 1;
            continue;
        };
        if !beta.is_finite() || !se.is_finite() || se <= 0.0 || se > max_se {
            n_dropped += 1;
            continue;
        }
        let pip_weight = idx_pip
            .and_then(|i| fields[i].parse::<f32>().ok())
            .filter(|p| p.is_finite())
            .map(|p| p.max(0.0));

        // A variant IS a locus, so it is spelled the way every other crate
        // spells one: `canon_locus` folds `chr5:1000000`, `5:1000000` and
        // `5_1000000` onto one name.
        key.clear();
        let _ = write!(key, "{}:{}", fields[idx_chromosome].trim(), position);

        rows.push(FileRow {
            gene: genes.intern(&canonical_gene(fields[idx_gene], name_kind)),
            celltype: celltypes.intern(fields[idx_celltype].trim()),
            variant: variants.intern(&canon_locus(&key)),
            beta,
            se,
            pip_weight,
        });
    }

    Ok(ParsedFile {
        genes: genes.into_names(),
        celltypes: celltypes.into_names(),
        variants: variants.into_names(),
        rows,
        n_dropped,
    })
}

#[cfg(test)]
#[path = "reader_tests.rs"]
mod tests;
