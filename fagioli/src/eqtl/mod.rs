//! Variant x gene x context eQTL embedding: a CP hyperedge over three
//! entity types, trained by InfoNCE against certified-absent negatives.
//!
//! ```text
//! score(variant j, gene g, context k) = sum_h u[j,h] * v[g,h] * c[k,h]
//! ```
//!
//! Contexts are the real cell types plus one pseudo-context, `ubiquitous`,
//! whose gate is fixed at one — there, the score is the context-free anchor
//! `<u_j, v_g>`. A second pseudo-candidate, `empty`, scores exactly zero and
//! owns no parameters, so every real candidate is judged against an
//! absolute threshold rather than only against its siblings.
//!
//! The pipeline is five stages, one submodule each:
//!
//! - `reader` ([`read_qtl_files`]) parses long-format summary statistics —
//!   one row per (variant, gene, celltype) — interning variants globally as
//!   `chromosome:position`.
//! - `select` ([`select_top_variants`]) keeps the top-K variants of every
//!   gene, ranked ONCE per gene, and then keeps every gene those variants
//!   were tested against.
//! - `ubiquity` ([`loo_meta`]) builds the leave-one-out meta-analysis that
//!   defines the ubiquitous context, and the model-free [`ubiquity_index`].
//! - `evidence` ([`classify_states`]) classifies each (variant, context,
//!   gene) cell as an edge, a certified absence, or unknown. Unknown cells
//!   are sampled in NEITHER class, which is what keeps statistical power out
//!   of the geometry.
//! - `model` ([`train`]) fits the three embeddings and reports the held-out
//!   separation, the gate's effective rank, and the group structure.
//!
//! `report` ([`specificity_rows`]) then turns a fit back into the per-pair
//! answer people ask for: ubiquitous, or specific to which cell type.
//!
//! ## Why this design
//!
//! Validated on real single-cell eQTL summary statistics: held-out edges
//! separate from certified absences well above chance, the gate's
//! effective rank settles below H, and contexts sharing a label prefix
//! have more similar gates than contexts that do not. Permuting the
//! edge/absent labels collapses all three.
//!
//! Two earlier architectures failed on the same data, with
//! no recoverable structure: free per-tuple deviation vectors, and
//! additive scalar deviations. The three-way multiplicative form with an
//! explicit cell-type embedding is what works.

pub(crate) mod evidence;
pub(crate) mod model;
pub(crate) mod reader;
pub(crate) mod report;
pub(crate) mod select;
pub(crate) mod ubiquity;

pub use evidence::{classify_states, EvidenceTable, Observation, PairRange, State, UBIQUITOUS};
pub use model::{train, EntityMap, EqtlFit, EqtlModelConfig, FitMetrics};
pub use reader::{read_qtl_files, GeneTable, QtlColumns, QtlData, QtlEntry};
pub use report::{specificity_rows, SpecificityRow};
pub use select::{select_top_variants, Pair, PairObs, Selection};
pub use ubiquity::{loo_meta, ubiquity_index, UbiquityRow};
