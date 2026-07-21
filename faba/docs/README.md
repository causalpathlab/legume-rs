# `faba` design & methods notes

Separated by **status**, because these are not the same kind of document and reading a plan as if
it were a description of the code is how people end up debugging things that were never built.

## Methods — describes code that exists

| doc | what it is |
|---|---|
| [`annotation-methods.md`](annotation-methods.md) | The full marker-annotation method: type prototypes, over-representation with a permutation null, the stability bootstrap, set-valued ("mixed") annotation, and the marker-panel bias guard. Every default is the shipped default; every measurement is attributed. Drives `faba annotate` and `faba lineage --markers`. |
| [`profiling-methods.md`](profiling-methods.md) | **BAM → per-cell features.** DART-seq m6A, A-to-I editing, alternative polyadenylation, gene counts and cell calling, SNP genotyping — the test, the null and the thresholds for each. Also lists where the code and its own `--help` text disagree. |
| [`annotation-grouping.md`](annotation-grouping.md) | **A negative result, kept deliberately.** Why the annotation pools cells into *coarse* clusters, and the measurements showing that a clustering-free per-cell-neighbourhood variant (the Milo device) is *worse than not testing at all*. Referenced from `graph-embedding-util/src/type_annotation/term_ora.rs` so the constraint is not quietly re-broken. |

## Plans — design notes for code that does **not** exist

| doc | what it is |
|---|---|
| [`annotation-ontology-plan.md`](annotation-ontology-plan.md) | Reference-free cluster annotation onto a gene-annotated Cell Ontology DAG. Design only; no code. |
| [`lineage-rooting.md`](lineage-rooting.md) | Eliciting expert knowledge for `faba lineage` rooting. Design only; no code. |

---

**Before trusting any annotation output**, check that the marker panel is on the embedding's
trained feature axis: `n_live / n_markers` in `{out}.panel_null.tsv` (or the "marker liveness" log
line) should be near 100%. If it is not, the panel's genes were projected rather than fitted, and
every downstream statistic is uninterpretable — including the ones designed to detect exactly
that. At gem's default `--n-hvg 0` every gene is trained, so this holds by construction; if you
run `--n-hvg > 0`, pass `faba gem --markers` pointing at *the same marker file* the annotation
uses. See `annotation-methods.md` §1.
