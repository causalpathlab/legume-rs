//! Bridge from the concrete Cell Ontology loader to the generic TreeBH core.
//!
//! The calling/tree/TreeBH math is generic and lives in `enrichment`
//! ([`enrichment::annotate_ontology_core`]) with **no** `auxiliary-data`
//! dependency — ontology access is injected as closures. This module is the one
//! place that binds that core to the concrete `auxiliary_data::ontology::Ontology`
//! OBO loader: load the OBO + the curated `label→CL` map, build the closures,
//! and run the core. Shared by the term-ORA projection path here and by
//! `senna annotate-ontology` / `-by-enrichment` (which delegate to it), so the
//! OBO glue is written once.

use anyhow::Result;
use auxiliary_data::ontology::Ontology;
use enrichment::{
    annotate_ontology_core, parse_label_map, Mat, OntologyAccess, OntologyParams, OntologyScore,
};
use log::info;

/// Load the Cell Ontology + curated `label<TAB>CL:id` map, inject them into the
/// generic TreeBH core as closures, and run it. Returns the
/// `(ontology_assignment.tsv, ontology_node_mass.parquet)` paths.
#[allow(clippy::too_many_arguments)]
pub fn annotate_ontology_from_obo(
    out: &str,
    label_cl: &str,
    obo: &str,
    fdr_q: f64,
    by: bool,
    score: OntologyScore<'_>,
    q: Option<&Mat>,
    cluster_names: &[Box<str>],
    celltype_names: &[Box<str>],
) -> Result<(String, String)> {
    let label_to_id = parse_label_map(label_cl, |id| id.starts_with("CL:"))?;
    let onto = Ontology::load_obo(obo)?;
    info!("loaded Cell Ontology: {} terms from {obo}", onto.len());

    // Bind the closures to locals so the borrows in `OntologyAccess` outlive the
    // call.
    let ancestors_or_self = |id: &str| onto.ancestors_or_self(id);
    let name_of = |id: &str| onto.name(id).map(Box::from);
    let contains = |id: &str| onto.contains(id);
    let access = OntologyAccess {
        ancestors_or_self: &ancestors_or_self,
        name_of: &name_of,
        contains: &contains,
    };

    annotate_ontology_core(
        &OntologyParams { out, fdr_q, by },
        score,
        q,
        &label_to_id,
        &access,
        cluster_names,
        celltype_names,
    )
}
