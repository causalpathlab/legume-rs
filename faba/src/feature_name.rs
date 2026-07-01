//! Canonical feature-name (sparse-matrix row) convention for every faba modality.
//!
//! All per-cell matrices name their rows
//!
//! ```text
//! {gene}/{modality}/{channel}              gene-level (no subunit)
//! {gene}/{modality}/{subunit}/{channel}    sub-gene (component or site)
//! ```
//!
//! - `gene`     — `{gene_id}_{gene_name}` (`gene_count::splice::format_gene_key`);
//!                the modelling unit at gene resolution.
//! - `modality` — the lowercase subcommand name: [`COUNT`] / [`M6A`] / [`ATOI`] /
//!                [`APA`] / [`SNP`].
//! - `subunit`  — optional sub-gene id: a `{chr}:{start}-{stop}` site or an EM
//!                mixture `{component}` index. Omitted for gene-level pooled rows.
//!                It sits **above** the channel: a component/site is a position
//!                cluster fit once per `(gene, modality)` and shared by both
//!                channels, so the channel nests inside it.
//! - `channel`  — the innermost (last) field: the two read-states that modality
//!                contrasts (gene counts split [`SPLICED`]/[`UNSPLICED`]; m6A
//!                [`METHYLATED`]/[`UNMETHYLATED`]; ATOI [`EDITED`]/[`UNEDITED`];
//!                APA [`PROXIMAL`]/[`DISTAL`]; SNP [`ALT`]/[`REF`]).
//!
//! Putting the channel last means a unit's two channel rows share a contiguous
//! prefix (the unit), and "strip the trailing field" recovers the unit.
//!
//! This module is the intended single source of truth: consumers (e.g. the gem
//! channel arm) split rows with [`parse_feature_row`], and producers are being
//! migrated onto [`feature_row`] so the tokens are no longer hand-spelled at call
//! sites (the editing / mixture / pileup producers still emit them inline today).
//! The gene is always recoverable from a parsed row's [`FeatureRow::unit`] via
//! `unit.split('/').next()`.

// ----- modality tokens (field 1) -----
pub const COUNT: &str = "count";
pub const M6A: &str = "m6a";
pub const ATOI: &str = "atoi";
pub const APA: &str = "apa";
pub const SNP: &str = "snp";

// ----- channel tokens (field 2) -----
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
pub const ALT: &str = "alt";
pub const REF: &str = "ref";

/// Format a feature row. Pass `subunit = None` for a gene-level (pooled) row
/// `{gene}/{modality}/{channel}`, or `Some(site_or_component)` for a sub-gene row
/// `{gene}/{modality}/{subunit}/{channel}` (channel innermost). The `subunit` must
/// not contain `/` (sites use `chr:start-stop`, components are integers), so the
/// row round-trips through [`parse_feature_row`].
pub fn feature_row(gene: &str, modality: &str, channel: &str, subunit: Option<&str>) -> Box<str> {
    match subunit {
        Some(s) => format!("{gene}/{modality}/{s}/{channel}").into(),
        None => format!("{gene}/{modality}/{channel}").into(),
    }
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
