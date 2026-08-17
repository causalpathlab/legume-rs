//! Canonical feature-row (sparse-matrix row) convention for every faba modality.
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
//! This module is the intended single source of truth: consumers (e.g. the gem
//! channel arm) split rows with [`parse_feature_row`], and producers are being
//! migrated onto [`feature_row`] so the tokens are no longer hand-spelled at call
//! sites (the editing / mixture / pileup producers still emit them inline today).
//! The unit is always recoverable from a parsed row's [`FeatureRow::unit`] via
//! `unit.split('/').next()`.

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
    /// `{gene}/{modality}/{subunit}` at sub-gene resolution. The gene stays
    /// recoverable as `unit.split('/').next()` at any resolution.
    pub fn unit(&self) -> Box<str> {
        match self.subunit {
            Some(s) => format!("{}/{}/{}", self.gene, self.modality, s).into(),
            None => self.gene.into(),
        }
    }
}

/// Split a feature row into its fields. The channel is the innermost (last)
/// field, so a 3-field row is gene-level (`{gene}/{modality}/{channel}`) and a
/// 4-field row carries a subunit before the channel
/// (`{gene}/{modality}/{subunit}/{channel}`). Returns `None` for rows with fewer
/// than three or more than four `/`-fields (gene + modality + channel mandatory;
/// neither subunit nor channel may contain `/`).
pub fn parse_feature_row(name: &str) -> Option<FeatureRow<'_>> {
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

#[cfg(test)]
mod tests;
