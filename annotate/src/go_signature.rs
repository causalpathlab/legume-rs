//! GO/GMT signature helpers for `annotate-by-enrichment`: load + reconcile
//! gene-sets against a gene dictionary, and write a per-group top-N signature
//! TSV from a (group × term) effect matrix.

use auxiliary_data::gene_sets::{read_gaf, read_gmt, GafOpts};
use auxiliary_data::ontology::Ontology;
use data_beans::utilities::name_matching::GeneIndex;
use log::info;
use std::io::Write;

/// Coverage floor below which enrichment is meaningless — fail loudly rather
/// than emit an empty signature.
const MIN_COVERAGE_FRAC: f32 = 0.1;
const MIN_COVERAGE_TERMS: usize = 5;
/// Top terms reported per group in the signature TSV.
const TOP_N: usize = 10;

/// Reconciled GO/GMT gene-sets ready for scoring.
pub struct GeneSetInputs {
    pub onto: Ontology,
    /// `(term id, member rows into the supplied `gene_names`)`, size-windowed,
    /// sorted by id.
    pub terms: Vec<(Box<str>, Vec<usize>)>,
    /// All matched annotated rows (the background universe).
    pub universe: Vec<usize>,
}

/// Load GO/GMT gene-sets, reconcile to `gene_names` (+ size window + coverage
/// gate). Exactly one of `gaf`/`gmt` must be `Some` (the caller validates this).
#[allow(clippy::too_many_arguments)]
pub fn load_go_gene_sets(
    obo: &str,
    gaf: Option<&str>,
    gmt: Option<&str>,
    no_iea: bool,
    min_gene_set: usize,
    max_gene_set: usize,
    gene_names: &[Box<str>],
) -> anyhow::Result<GeneSetInputs> {
    let onto = Ontology::load_obo(obo)?;
    info!("loaded ontology: {} terms from {obo}", onto.len());

    let gene_sets = if let Some(gaf) = gaf {
        info!("reading GAF gene-sets from {gaf} (no_iea={no_iea})");
        read_gaf(gaf, &GafOpts { no_iea })?.into_gene_sets(Some(&onto))
    } else {
        let gmt = gmt.expect("exactly one of --gaf/--gmt is required");
        info!("reading GMT gene-sets from {gmt}");
        read_gmt(gmt)?
    };
    info!(
        "gene-sets: {} terms, {} genes, {} annotations",
        gene_sets.n_terms(),
        gene_sets.n_genes(),
        gene_sets.n_annotations()
    );

    let idx = GeneIndex::build(gene_names);
    let rec = gene_sets.reconcile(&idx, min_gene_set, Some(max_gene_set));
    rec.log_coverage();
    rec.ensure_coverage(MIN_COVERAGE_FRAC, MIN_COVERAGE_TERMS)?;
    let universe = rec.universe;
    let mut terms: Vec<(Box<str>, Vec<usize>)> = rec.term_rows.into_iter().collect();
    terms.sort_by(|a, b| a.0.cmp(&b.0));
    info!("scoring {} size-windowed terms", terms.len());

    Ok(GeneSetInputs {
        onto,
        terms,
        universe,
    })
}

/// Write a per-group top-`N` GO signature TSV from a `group × term` effect
/// matrix, each group's terms ranked by descending positive effect. `term_ids`
/// indexes the matrix columns and aligns 1:1 with `terms` (for the gene count).
pub fn write_go_signature(
    path: &str,
    onto: &Ontology,
    effect_kt: &enrichment::Mat,
    term_ids: &[Box<str>],
    terms: &[(Box<str>, Vec<usize>)],
    group_axis: &str,
    group_names: &[Box<str>],
) -> anyhow::Result<()> {
    let n_terms = term_ids.len();
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "{group_axis}\trank\tterm_id\tterm_name\teffect\tn_genes")?;
    for (k, gname) in group_names.iter().enumerate() {
        let mut ranked: Vec<(usize, f32)> = (0..n_terms)
            .map(|t| (t, effect_kt[(k, t)]))
            .filter(|&(_, e)| e > 0.0)
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (rank, &(t, e)) in ranked.iter().take(TOP_N).enumerate() {
            let id = &term_ids[t];
            let name = onto.name(id).unwrap_or(id.as_ref());
            writeln!(
                f,
                "{gname}\t{}\t{id}\t{name}\t{e:.4}\t{}",
                rank + 1,
                terms[t].1.len()
            )?;
        }
    }
    info!("wrote {path}");
    Ok(())
}
