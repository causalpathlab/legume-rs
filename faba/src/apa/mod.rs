pub mod cell_assign;
pub mod em;
pub mod fragment;
pub mod likelihood;
pub mod pdui;
pub mod pipeline;
/// The `faba apa` run. Binary entry: [`run::run_apa`].
pub mod run;
pub mod site_discovery;
pub mod utr_region;

/////////////////////
// poly-A site ids //
/////////////////////
// A poly-A `site_id` is a channel-less unit row (`faba::feature_name::unit_row`):
// `{gene}/apa/{component}` for the EM mixture, `{gene}/apa/{chr}:{start}` for the
// simple pileup. It is both a `{batch}_apa_mixture` row name and the `site_id`
// column of `apa_components.parquet`, so the two join on it — and several callers
// split it back apart to recover the gene. Format and parse live together here so
// they cannot drift; they used to be a `/pA/` literal spelled at seven call sites,
// which is how the emitted token came to disagree with `feature_name::APA`.

/// The infix separating a poly-A `site_id`'s gene from its subunit.
pub(crate) const SITE_INFIX: &str = "/apa/";

/// Format a poly-A `site_id` for `gene`, keyed by a component index or `chr:pos`.
pub(crate) fn site_id(gene: &str, subunit: &str) -> Box<str> {
    faba::feature_name::unit_row(gene, faba::feature_name::APA, subunit)
}

/// Recover the gene from a poly-A `site_id`, or the whole id when it has no infix.
pub(crate) fn site_gene(site_id: &str) -> &str {
    site_id.split_once(SITE_INFIX).map_or(site_id, |(g, _)| g)
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;

#[cfg(test)]
pub mod simulate;
