//! Canonical feature-row (sparse-matrix row) convention for every modality.
//!
//! NOT to be confused with the sibling [`crate::feature_names`], which is about a
//! different problem. This module fixes the row-name **grammar** a producer emits
//! and a consumer splits; `feature_names` **canonicalizes** an already-emitted
//! name so the same gene or locus matches across files that spell it differently
//! (`FeatureNameKind`). Rows here are built and parsed; names there are matched.
//!
//! It lives in `auxiliary-data` rather than beside its producers because the
//! grammar has readers on both sides of the BAM/model boundary: faba writes these
//! rows, senna's embedding and association steps split them back apart.
//!
//! All per-cell matrices name their rows
//!
//! ```text
//! {unit}/{modality}/{channel}              unit-level (no subunit)
//! {unit}/{modality}/{subunit}/{channel}    sub-unit (component or site)
//! ```
//!
//! - `unit` — the modelling unit. For every gene-resolution modality this is
//!   the gene, `{gene_id}_{gene_name}` (`gene_count::splice::format_gene_key`).
//!   [`BAF`] is the exception: a variant is a coordinate, not a gene. It does
//!   not belong to one, and two overlapping genes would otherwise give the same
//!   variant two row names, so its unit is the `{chr}:{pos}` locus.
//! - `modality` — the lowercase subcommand name: [`COUNT`] / [`M6A`] / [`ATOI`]
//!   / [`APA`], or [`BAF`].
//! - `subunit` — optional sub-gene id: a single-base `{chr}:{pos}` site (m6A and
//!   A-to-I sites are one base pair) or an EM mixture `{component}` index.
//!   Omitted for gene-level pooled rows. It sits **above** the channel: a
//!   component/site is a position cluster fit once per `(gene, modality)` and
//!   shared by both channels, so the channel nests inside it.
//! - `channel` — the innermost (last) field: the two read-states that modality
//!   contrasts (gene counts split [`SPLICED`]/[`UNSPLICED`]; m6A
//!   [`METHYLATED`]/[`UNMETHYLATED`]; ATOI [`EDITED`]/[`UNEDITED`]; APA
//!   [`PROXIMAL`]/[`DISTAL`]; BAF [`ALT`]/[`DEPTH`]). Omitted by the one
//!   producer whose contrast lives across the units rather than within the row —
//!   see [`unit_row`].
//!
//! Putting the channel last means a unit's two channel rows share a contiguous
//! prefix (the unit), and "strip the trailing field" recovers the unit.
//!
//! Every channelized modality keeps both states in ONE matrix rather than in a
//! pair of same-shaped files, so a ratio is a division within one unit's rows
//! and no consumer has to open two files and trust their row orders agree.
//!
//! Most channel pairs PARTITION the coverage — the two states are exclusive and
//! sum to the total. [`BAF`] is the exception: [`ALT`] is nested inside
//! [`DEPTH`] (`alt ≤ depth`), so BAF is `alt / depth` and NOT `alt / (alt +
//! depth)`. Any consumer that sums a unit's channels to recover coverage is
//! wrong on this modality alone.
//!
//! This module is the single source of truth. Consumers split rows with
//! [`parse_feature_row`]; producers build them with [`feature_row`] /
//! [`unit_row`] rather than hand-spelling the tokens. The gene-count, APA, SNP
//! and quant producers all go through it; the editing / mixture / pileup
//! producers still emit their rows inline and are the remaining migration.
//!
//! One consumer still parses by hand: `faba::quant::extract_gene_key` strips a
//! trailing `/count/{channel}` with `rfind`. That is safe *there* — it runs only
//! over faba's own gene matrices, and its job is to group every row of a gene
//! (including the pooled `total` track) under one key for QC, which is what its
//! callers want. Contrast `senna::gem::rows`, which must additionally decide
//! WHICH track a row is: there the same shortcut put `total` in the spliced
//! bucket and double-counted the gene, so that one goes through
//! [`parse_feature_row`].
//!
//! The unit of a parsed row is [`FeatureRow::unit`], and the gene is
//! [`FeatureRow::gene`] — read those fields rather than re-splitting the string.
//! `unit.split('/').next()` USED to recover the gene and no longer does: a unit
//! may itself contain `/`, because gene symbols do (see [`parse_feature_row`]),
//! so that recipe truncates such a gene at its first slash.

///////////////////////////////
// modality tokens (field 1) //
///////////////////////////////
pub const COUNT: &str = "count";
pub const M6A: &str = "m6a";
pub const ATOI: &str = "atoi";
pub const APA: &str = "apa";
/// Per-cell allele frequency at a called variant locus. Named for what the
/// matrix measures (B-allele frequency), not for the calling step that chose the
/// positions: the call set — genotype, GQ, rsid — is `snp_sites.parquet` /
/// `snp_sites.vcf.gz`, and a row here carries none of it, only two read counts.
pub const BAF: &str = "baf";

//////////////////////////////
// channel tokens (field 2) //
//////////////////////////////
pub const SPLICED: &str = "spliced";
pub const UNSPLICED: &str = "unspliced";
/// Gene-count total (spliced + unspliced) — used by the pooled gene-QC track.
pub const TOTAL: &str = "total";
pub const METHYLATED: &str = "methylated";
pub const UNMETHYLATED: &str = "unmethylated";
pub const EDITED: &str = "edited";
pub const UNEDITED: &str = "unedited";
/// APA channels come from the 2-site PDUI decomposition (proximal vs distal
/// poly-A in the 3'UTR). The K-component poly-A *mixture* is a separate count
/// matrix, NOT channelized — it does not follow this convention.
pub const PROXIMAL: &str = "proximal";
pub const DISTAL: &str = "distal";
/// BAF numerator: reads carrying the called alt allele.
pub const ALT: &str = "alt";
/// BAF denominator: ALL reads over the locus, alt included. The only channel
/// pair that nests rather than partitions — see the module docs.
pub const DEPTH: &str = "depth";

/// Format a feature row. Pass `subunit = None` for a gene-level (pooled) row
/// `{gene}/{modality}/{channel}`, or `Some(site_or_component)` for a sub-gene row
/// `{gene}/{modality}/{subunit}/{channel}` (channel innermost). The `subunit` must
/// not contain `/` (sites use the single-base `chr:pos`, components are integers),
/// so the row round-trips through [`parse_feature_row`].
pub fn feature_row(gene: &str, modality: &str, channel: &str, subunit: Option<&str>) -> Box<str> {
    match subunit {
        Some(s) => format!("{gene}/{modality}/{s}/{channel}").into(),
        None => format!("{gene}/{modality}/{channel}").into(),
    }
}

/// Format a channel-less UNIT row `{gene}/{modality}/{subunit}`.
///
/// One producer names a unit with no channel, because its contrast lives ACROSS
/// the units rather than within the row: the APA poly-A mixture,
/// `{gene}/apa/{component}` — usage is relative across the components of a gene,
/// so no component has a counterpart channel.
///
/// SNP allele counts were the second such producer, splitting alt and depth
/// across two same-shaped matrices that a consumer had to open together. They
/// are now [`BAF`], channelized on [`ALT`]/[`DEPTH`] inside one matrix.
///
/// Such a row is indistinguishable from a gene-level one under
/// [`parse_feature_row`]: both are three fields, and the subunit lands in the
/// `channel` slot. A consumer has to know which matrix it is reading. Prefer
/// [`feature_row`] wherever the modality does have two channels.
pub fn unit_row(gene: &str, modality: &str, subunit: &str) -> Box<str> {
    format!("{gene}/{modality}/{subunit}").into()
}

/// A feature row split into its fields, borrowing from the source string.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FeatureRow<'a> {
    pub gene: &'a str,
    pub modality: &'a str,
    pub channel: &'a str,
    pub subunit: Option<&'a str>,
}

impl FeatureRow<'_> {
    /// The modelling unit of this row: the bare gene at gene resolution, or
    /// `{gene}/{modality}/{subunit}` at sub-gene resolution.
    ///
    /// The gene is NOT recoverable by splitting this on `/` — a gene symbol may
    /// contain one. Use [`Self::gene`], which is already the parsed field.
    pub fn unit(&self) -> Box<str> {
        match self.subunit {
            Some(s) => format!("{}/{}/{}", self.gene, self.modality, s).into(),
            None => self.gene.into(),
        }
    }
}

/// The closed modality vocabulary, used to LOCATE the modality field rather
/// than to validate it — see [`parse_feature_row`].
const MODALITIES: [&str; 5] = [COUNT, M6A, ATOI, APA, BAF];

/// Split a feature row into its fields. The channel is the innermost (last)
/// field, so a 3-field row is gene-level (`{gene}/{modality}/{channel}`) and a
/// 4-field row carries a subunit before the channel
/// (`{gene}/{modality}/{subunit}/{channel}`).
///
/// # A unit may contain `/`
///
/// Counting fields alone is not enough, because **real gene symbols contain
/// slashes** — standard human references ship at least one. Such a gene's count
/// row has four fields and used to parse as `gene = {id}_GENE1`,
/// `modality = GENE1B`, `subunit = count`: not an error, just a different gene,
/// so the two channel rows of that gene stopped pairing and nothing said so.
///
/// So the modality is located by NAME, scanned from the right against
/// [`MODALITIES`], and whatever precedes it is the unit however many slashes it
/// contains. The subunit and channel still may not contain `/` — they are a
/// `chr:pos`, a component index, and a fixed token.
///
/// The two candidate positions are tried nearest-first, so a row whose unit ENDS
/// in a modality token reads as the gene-level form: `A/count/count/spliced` is
/// the gene `A/count`, not the gene `A` with a subunit called `count`. That
/// ambiguity is unreachable in practice — a subunit is a `chr:pos` or a component
/// index, never a modality name.
///
/// When NEITHER candidate position holds a known modality the old positional
/// rule applies unchanged, so the producers that still emit their rows inline
/// with a modality token outside the constant list (see the module docs) keep
/// parsing exactly as they did. The vocabulary can only make a row parse
/// BETTER, never make a row that parsed stop parsing.
///
/// Returns `None` for anything with fewer than three fields, or more than four
/// when no known modality locates the split.
pub fn parse_feature_row(name: &str) -> Option<FeatureRow<'_>> {
    let is_modality = |s: &str| MODALITIES.contains(&s);

    // Peel the channel, then look one and two fields further left for the
    // modality. `rsplit_once` keeps every field a slice of `name`, so a
    // multi-field unit costs no allocation.
    if let Some((head, channel)) = name.rsplit_once('/') {
        if let Some((left, mid)) = head.rsplit_once('/') {
            if is_modality(mid) && !left.is_empty() {
                return Some(FeatureRow {
                    gene: left,
                    modality: mid,
                    channel,
                    subunit: None,
                });
            }
            if let Some((gene, modality)) = left.rsplit_once('/') {
                if is_modality(modality) && !gene.is_empty() {
                    return Some(FeatureRow {
                        gene,
                        modality,
                        channel,
                        subunit: Some(mid),
                    });
                }
            }
        }
    }

    let parts: Vec<&str> = name.split('/').collect();
    match parts.as_slice() {
        [gene, modality, channel] => Some(FeatureRow {
            gene,
            modality,
            channel,
            subunit: None,
        }),
        [gene, modality, subunit, channel] => Some(FeatureRow {
            gene,
            modality,
            channel,
            subunit: Some(subunit),
        }),
        _ => None,
    }
}

///////////////////////////////////////////////
// gene-count rows, interned to a gene axis  //
///////////////////////////////////////////////

/// Split a gene-level count row `{gene}/count/{spliced|unspliced}` into its gene
/// key and whether it is the **nascent** (unspliced) track. `None` when the row is
/// not a gene-level count row at all.
///
/// Goes through [`parse_feature_row`] rather than matching on `/count/` directly,
/// because a bare `rsplit_once` **cannot tell "spliced" apart from "not a count
/// row"** — both fall to the same branch. It used to, and the consequence was
/// silent: `GENE1/m6a/methylated` became a mature gene literally named
/// `GENE1/m6a/methylated`, and the sub-gene form `{gene}/count/{site}/{channel}`
/// became a mature row of the right gene.
///
/// [`TOTAL`] is rejected along with everything else: it already IS
/// `spliced + unspliced`, so interning it as a third track would count the gene
/// twice. A `subunit` is rejected because a per-site or per-component row is not
/// a thing that pairs across tracks at gene resolution.
#[must_use]
pub fn split_count_row(name: &str) -> Option<(&str, bool)> {
    let row = parse_feature_row(name)?;
    if row.modality != COUNT || row.subunit.is_some() {
        return None;
    }
    match row.channel {
        SPLICED => Some((row.gene, false)),
        UNSPLICED => Some((row.gene, true)),
        _ => None,
    }
}

/// [`CountRowMap::row_to_gene`] entry for a row that was left off the gene axis.
/// Only ever produced under [`UnparsedRowPolicy::Reject`].
pub const NO_GENE: u32 = u32::MAX;

/// What [`intern_count_rows`] does with a row that is not
/// `{gene}/count/{spliced|unspliced}`.
///
/// The two consumers want opposite things, and neither is more correct: a
/// gene-keyed model can carry a stray row harmlessly as its own single-track
/// gene, while a consumer that POOLS the two tracks cannot — it would have to
/// decide which track the stray row is, and every answer is wrong.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UnparsedRowPolicy {
    /// Give the row its own single-track gene id, keyed on the whole row name,
    /// so every index still lines up with the matrix and the row can never be
    /// paired with a real gene. Ids stay assigned in row order across both
    /// kinds. The caller is expected to warn.
    OwnGene,
    /// Leave the row off the gene axis: [`NO_GENE`] in `row_to_gene`, and its
    /// index in [`CountRowMap::unparsed`]. The caller decides whether that is
    /// fatal.
    Reject,
}

/// A count matrix's feature axis interned onto a dense gene axis.
///
/// Row order is the matrix's own and is never permuted; gene ids are assigned in
/// first-seen row order, so `gene_names` is stable for a given input.
pub struct CountRowMap {
    /// `row_to_gene[r]` = gene id of row `r`, or [`NO_GENE`].
    pub row_to_gene: Vec<u32>,
    /// `row_is_nascent[r]` = true when row `r` is the unspliced track. Always
    /// `false` for an unparsed row under either policy — such a row is not a
    /// nascent row, it is not a count row.
    pub row_is_nascent: Vec<bool>,
    /// Gene keys in id order.
    pub gene_names: Vec<Box<str>>,
    /// Indices of rows that are not `{gene}/count/{spliced|unspliced}`, in row
    /// order. Empty on a well-formed gene-count matrix.
    pub unparsed: Vec<usize>,
}

impl CountRowMap {
    #[must_use]
    pub fn n_genes(&self) -> usize {
        self.gene_names.len()
    }

    #[must_use]
    pub fn n_rows(&self) -> usize {
        self.row_to_gene.len()
    }

    /// Rows on the nascent track. Zero on a spliced-only matrix — a legitimate
    /// input that simply identifies no nascent-minus-mature contrast.
    #[must_use]
    pub fn n_nascent_rows(&self) -> usize {
        self.row_is_nascent.iter().filter(|&&n| n).count()
    }
}

/// Intern a count matrix's feature axis onto a dense gene axis, pairing a gene's
/// two channel rows under one id.
#[must_use]
pub fn intern_count_rows(feature_names: &[Box<str>], policy: UnparsedRowPolicy) -> CountRowMap {
    let mut ids: rustc_hash::FxHashMap<Box<str>, u32> = rustc_hash::FxHashMap::default();
    let mut row_to_gene = Vec::with_capacity(feature_names.len());
    let mut row_is_nascent = Vec::with_capacity(feature_names.len());
    let mut gene_names: Vec<Box<str>> = Vec::new();
    let mut unparsed: Vec<usize> = Vec::new();

    for (r, name) in feature_names.iter().enumerate() {
        let Some((gene, is_nascent)) = split_count_row(name) else {
            unparsed.push(r);
            match policy {
                UnparsedRowPolicy::Reject => row_to_gene.push(NO_GENE),
                UnparsedRowPolicy::OwnGene => {
                    let g = gene_names.len() as u32;
                    ids.insert(name.clone(), g);
                    gene_names.push(name.clone());
                    row_to_gene.push(g);
                }
            }
            row_is_nascent.push(false);
            continue;
        };
        let gid = match ids.get(gene) {
            Some(&g) => g,
            None => {
                let g = gene_names.len() as u32;
                ids.insert(gene.into(), g);
                gene_names.push(gene.into());
                g
            }
        };
        row_to_gene.push(gid);
        row_is_nascent.push(is_nascent);
    }

    CountRowMap {
        row_to_gene,
        row_is_nascent,
        gene_names,
        unparsed,
    }
}

#[cfg(test)]
mod tests;
