//! `faba gem` — **Ge**odesic **E**mbedding + **M**otion: a joint cell-feature embedding
//! over the shared `graph_embedding_util` engine. Motion is the local velocity δ (the
//! tangent); the lineage is the geodesic path it traces. The engine is modality-agnostic;
//! it is fed gene counts (spliced + unspliced) today, but embeds any per-feature count.
//!
//! Each feature row `{gene}/count/{spliced|unspliced}` maps to its gene, so a
//! gene's spliced and unspliced tracks embed identically as `β_g` (β-sharing).
//! A single Poisson likelihood on counts: cell **identity** is the spliced
//! projection θ (written raw), and the **velocity** is the raw analytic increment
//! δ — a Poisson-MAP shift fit to the unspliced edges with θ held fixed (‖δ‖ =
//! speed) — tracking the spliced↔unspliced dynamics rather than a second (binomial)
//! likelihood.

pub mod args;
pub mod common;
pub mod sample_id;
