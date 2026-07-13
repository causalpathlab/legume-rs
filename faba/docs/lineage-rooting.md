# Eliciting Expert Knowledge in Lineage Rooting

Planning notes for `faba lineage` rooting. Not yet implemented — captures the
design directions from the 2026-07-08 discussion for a future update.

## Why rooting deserves expert knowledge (the motivation)

The root is not a cosmetic choice — it **determines the actual deliverables** of
the downstream `faba assoc` between-branch analysis, and automatic (data-only)
rooting is unreliable in an underfit embedding. Evidence gathered on the
cord-blood WT m6A data (15,315 cells, 6 samples):

- **Root reshapes pseudotime, it does not merely reverse it.** Holding the gem
  embedding *fixed* and changing only the root (velocity/`--root-from-gem` vs
  `--root-type Cycling_Progenitor`), per-cell pseudotime correlated at
  **Pearson r = −0.05** — essentially independent, not `−1` (a clean flip).
  Slingshot builds lineages as root→leaf paths, so a new root re-fits the curves
  → new pseudotime values **and** new branch assignments.
- **That flows straight into the permutation test.** The between-branch contrast
  bins by pseudotime and permutes branch labels *within* bins; when the bins and
  branches change, the null and the statistic both change.
- **So the significant hits do not agree across roots** (same embedding):

  | test | criterion | root A | root B | overlap | Jaccard |
  |------|-----------|--------|--------|---------|---------|
  | between-branch | BH-FDR q<0.1 | 0 | 0 | 0 | vacuous |
  | between-branch | WY-FWER<0.1 | 1 (RAN) | 2 (UBAC1,MR1) | **0** | **0.00** |
  | between-branch | raw p<0.01 | 16 | 15 | 4 | 0.15 |
  | within-branch | FSR lfsr<0.1 | 156 | 182 | 104 | 0.44 |
  | within-branch | FSR lfsr<0.05 | 109 | 130 | 52 | 0.28 |

  Multiple-testing correction controls error *within a fixed test family*; it does
  **not** provide cross-root stability (the families differ). Stricter thresholds
  make between-run overlap *worse*, not better (the extreme tail is the most
  root-sensitive part).
- **Automatic signals fail or are ambiguous here.** Velocity *direction* inverted
  (rooted at CD4_Memory_T, a terminal fate); velocity *magnitude* is usable
  (terminals low |v|: CD8/CD4-T ≈ 0.75–0.89, progenitors high: Early_GMP 1.62,
  Cycling_Progenitor 1.34) but does not by itself name the apex; the
  biologically-correct `--root-type HSC_MPP` **failed** because the embedding never
  resolves an HSC_MPP cluster (rare cord-blood HSCs).

Conclusion: for a developmental "where does differentiation start" question, the
root is a **biological prior**. Encode expert knowledge there rather than trying
to infer it from a shaky embedding.

## Current state (what exists today)

- Root priority in `resolve_root` (`faba/src/run_lineage.rs`):
  `--root-node > --root-cell > --root-type > --root-from-gem > velocity-flux > node 0`.
- `--root-type <TYPE>`: single marker-grounded type, matched to the
  highest-confidence node of that type via the marker ORA (`--markers`).
  **Fails silently** if the type is not resolved → falls through to velocity.
- Default (no flag) = **velocity-flux source** — velocity-based, so unreliable in
  exactly the regime we care about.
- Marker/node scoring infrastructure already present:
  `graph_embedding_util::type_annotation` (`term_ora`, `cluster_term_softq`) and
  the `--markers` path in `run_lineage.rs` (`compute_node_calls`,
  `root_type_node`).

## Ideas to elicit expert knowledge (cheap → rich)

### 1. Ranked root-type list
`--root-type HSC_MPP,LMPP,Cycling_Progenitor` — try in priority order, take the
first type that matches a node above a confidence floor. Directly fixes the
silent fall-through that bit us (unresolved HSC_MPP → sensible degrade to the
next-most-primitive type the embedding *does* resolve). Smallest change; extends
the existing `--root-type` parse + `root_type_node`.

### 2. Expert stemness *signature* (recommended primary)
Score nodes directly by a curated progenitor gene set (e.g. HLF, CRHBP, AVP,
MLLT3, MEIS1, HOXA9…) and root at the argmax — **independent of whether the
clustering named an HSC cluster**. This is how a hematologist actually reasons
("which node most expresses the stem program"), and it is robust to the
annotation missing the type. Reuses the node-marker scoring already in
`type_annotation`. Input could be a `--root-signature <genes.tsv>` (or a named
entry inside the `--markers` file flagged as the root program).

### 3. Terminal-type exclusions
`--terminal-types CD4_Memory_T,B,Late_Erythroid,…` — forbid known-mature types
from root candidacy. Trivially cheap; directly kills the "rooted at a T cell"
failure mode even when only velocity/potential signals are available.

### 4. Directed lineage prior (richest)
Expert supplies a partial order, e.g. `HSC → MPP → {GMP, MEP, CLP}`, that
constrains DAG **orientation**, not just the single root node. Heavier (needs a
prior-consistent orientation step), but it stabilizes the whole branch topology,
not only the origin. Could dovetail with the `--marker-obo` / Cell-Ontology layer
already wired for annotation.

## Cross-check / confidence design (do this alongside any of the above)

Compute the candidate root under several independent definitions —
**expert signature/type**, **differentiation potential** (CytoTRACE-style
gene-diversity / transcriptional entropy), and **velocity magnitude** — and
**report their agreement**:

- all agree → high-confidence root;
- disagree → emit a "trajectory underdetermined" diagnostic (the honest signal —
  this is the case on the current cord-blood embedding).

So expert knowledge **anchors** the root and the data-driven signals
**cross-check** it, instead of either silently winning. Replaces the removed
`underfit` flag's intent with something actionable and root-specific.

## Caveats (so we don't over-promise)

- Expert rooting removes **the root** as a source of run-to-run variation, but the
  **branch topology and pseudotime still ride on the (non-deterministic, underfit)
  embedding**. It makes the *root* reproducible, not the *branch structure*.
- Therefore expert rooting and **firming the embedding** are complementary, not
  either/or: the first stops us rooting at a T cell; the second stops the branches
  from reshuffling between runs. A stable between-branch deliverable needs both.
- Orthogonal mitigation: a **root-free** trajectory parameterization (undirected
  tree segments + arc-length) for the parts of `assoc` that don't need a root —
  detection (`|effect|`, p, gene set) is root-invariant; only trend *sign* and
  fate polarity need orientation. See the "drop root claims" discussion:
  report root-invariant quantities as primary, direction as a provisional,
  robustly-oriented annotation.

## Suggested implementation path

1. **Phase 1 (cheap CLI):** ranked `--root-type` list (#1) + `--terminal-types`
   exclusion (#3). Small edits to arg parsing + `root_type_node`.
2. **Phase 2:** stemness-signature rooting (#2) — `--root-signature`, reuse
   `type_annotation` node scoring.
3. **Phase 3:** agreement flag — compute potential + velocity-magnitude roots,
   emit a confidence/agreement diagnostic in `lineage_qc.json` (root-specific,
   replacing the intent of the retired `underfit` flag).
4. **Phase 4 (optional, larger):** directed lineage prior (#4); and/or a root-free
   undirected-segment branch mode so `assoc` can run with the root set aside.

## Related

- `faba/src/run_lineage.rs` — `resolve_root`, `root_type_node`,
  `gem_root_node`, `compute_node_calls`.
- `graph-embedding-util/src/type_annotation/` — marker ORA node scoring
  (`term_ora`, `cluster_term_softq`) to reuse for signature rooting.
- `docs/annotation-ontology-plan.md` — the reference-free CL-DAG annotation plan
  (shares the marker/ontology machinery).
- `faba/src/assoc/` — the downstream that inherits the rooting choice; see the
  root-invariance decomposition (detection vs direction).
