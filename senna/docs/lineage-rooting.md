# Eliciting Expert Knowledge in Lineage Rooting

Planning notes for `senna lineage` rooting. Not yet implemented; captures
design directions for a future update.

## Why rooting deserves expert knowledge (the motivation)

The root is not a cosmetic choice: it determines the actual deliverables of
the downstream `senna dyn-assoc` between-branch analysis, and automatic
(data-only) rooting is unreliable in an underfit embedding.

- **Root reshapes pseudotime, it does not merely reverse it.** Holding the
  embedding fixed and changing only the root produces per-cell pseudotime
  that is close to uncorrelated between roots, not a clean sign flip.
  Slingshot builds lineages as root-to-leaf paths, so a new root re-fits the
  curves, producing new pseudotime values and new branch assignments.
- **That flows straight into the permutation test.** The between-branch
  contrast bins by pseudotime and permutes branch labels within bins; when
  the bins and branches change, the null and the statistic both change.
- **So the significant hits do not agree across roots, on the same
  embedding,** at any of the usual multiple-testing thresholds. Correction
  controls error within a fixed test family; it does not provide cross-root
  stability, since the families themselves differ across roots. Stricter
  thresholds make cross-root overlap worse, not better, because the extreme
  tail is the most root-sensitive part of the ranking.
- **Automatic signals fail or are ambiguous on their own.** Velocity
  direction can invert at a terminal fate; velocity magnitude is informative
  (low near terminal states, high near progenitors) but does not by itself
  name the apex; a marker-grounded root type can fail outright when the
  embedding never resolves that cluster, which is common for a rare
  population.

Conclusion: for a developmental "where does differentiation start" question,
the root is a biological prior. Encode expert knowledge there rather than
trying to infer it from a shaky embedding.

## Current state (what exists today)

- Root priority in `resolve_root_hint` (`lineage/src/lineage/root.rs`):
  `--root-node > --root-cell > --root-type > velocity-flux root > node 0`.
- `--root-type <TYPE>`: single marker-grounded type, matched to the
  highest-confidence node of that type via the marker ORA (`--markers`).
  Fails silently if the type is not resolved, falling through to the
  velocity-flux root.
- Default (no flag) is the velocity-flux source: velocity-based, so
  unreliable in exactly the regime we care about.
- Marker/node scoring infrastructure already present:
  `graph_embedding_util::type_annotation` (the `term_ora` scoring path) and
  the `--markers` path in `lineage/src/lineage/traj_annotation.rs`
  (`compute_node_calls`, consumed by `root_type_node`).

## Ideas to elicit expert knowledge (cheap to rich)

### 1. Ranked root-type list
`--root-type TYPE1,TYPE2,TYPE3`: try in priority order, take the first type
that matches a node above a confidence floor. Directly fixes the silent
fall-through today (an unresolved primary type degrades to the
next-most-primitive type the embedding does resolve). Smallest change;
extends the existing `--root-type` parse and `root_type_node`.

### 2. Expert stemness signature (recommended primary)
Score nodes directly by a curated progenitor gene set and root at the
argmax, independent of whether the clustering named that cluster. This
mirrors how an expert actually reasons about which node most expresses the
progenitor program, and it is robust to the annotation missing the type.
Reuses the node-marker scoring already in `type_annotation`. Input could be
a `--root-signature <genes.tsv>` (or a named entry inside the `--markers`
file flagged as the root program).

### 3. Terminal-type exclusions
`--terminal-types TYPE1,TYPE2,...`: forbid known-mature types from root
candidacy. Trivially cheap; directly kills the "rooted at a mature cell"
failure mode even when only velocity/potential signals are available.

### 4. Directed lineage prior (richest)
Expert supplies a partial order, for example a root type feeding one or more
intermediate types that in turn feed the terminal types, constraining DAG
orientation rather than just the single root node. Heavier (needs a
prior-consistent orientation step), but it stabilizes the whole branch
topology, not only the origin. Could dovetail with the `--marker-obo` /
Cell-Ontology layer already wired for annotation.

## Cross-check / confidence design (do this alongside any of the above)

Compute the candidate root under several independent definitions, expert
signature/type, differentiation potential (gene-diversity / transcriptional
entropy), and velocity magnitude, and report their agreement:

- all agree: high-confidence root;
- disagree: emit a "trajectory underdetermined" diagnostic, which is the
  honest signal in that case.

So expert knowledge anchors the root and the data-driven signals cross-check
it, instead of either silently winning. Replaces the intent of the earlier,
now-removed underfit flag with something actionable and root-specific.

## Caveats (so we don't over-promise)

- Expert rooting removes the root as a source of run-to-run variation, but
  the branch topology and pseudotime still ride on the (non-deterministic,
  underfit) embedding. It makes the root reproducible, not the branch
  structure.
- Therefore expert rooting and firming the embedding are complementary, not
  either/or: the first stops us rooting at the wrong cell; the second stops
  the branches from reshuffling between runs. A stable between-branch
  deliverable needs both.
- Orthogonal mitigation: a root-free trajectory parameterization (undirected
  tree segments plus arc-length) for the parts of `assoc` that don't need a
  root. Detection (`|effect|`, p, gene set) is root-invariant; only trend
  sign and fate polarity need orientation. Report root-invariant quantities
  as primary, direction as a provisional, robustly-oriented annotation.

## Suggested implementation path

1. **Phase 1 (cheap CLI):** ranked `--root-type` list (#1) plus
   `--terminal-types` exclusion (#3). Small edits to arg parsing and
   `root_type_node`.
2. **Phase 2:** stemness-signature rooting (#2), `--root-signature`, reuse
   `type_annotation` node scoring.
3. **Phase 3:** agreement flag; compute potential and velocity-magnitude
   roots, emit a confidence/agreement diagnostic alongside the run outputs,
   root-specific, replacing the intent of the retired underfit flag.
4. **Phase 4 (optional, larger):** directed lineage prior (#4); and/or a
   root-free undirected-segment branch mode so `assoc` can run with the root
   set aside.

## Related

- `lineage/src/lineage/root.rs`: `resolve_root_hint`, `root_type_node`.
- `lineage/src/lineage/traj_annotation.rs`: `compute_node_calls`.
- `graph-embedding-util/src/type_annotation/`: the marker ORA node scoring
  to reuse for signature rooting.
- `docs/annotation-ontology-plan.md`: the reference-free CL-DAG annotation
  plan (shares the marker/ontology machinery).
- `lineage/src/assoc/`: the downstream that inherits the rooting choice; see
  the root-invariance decomposition (detection vs direction).
