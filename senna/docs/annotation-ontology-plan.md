# Plan — Reference-free cluster annotation on a gene-annotated Cell Ontology DAG

*Working notes, 2026-06-23. Status: design, no code. Parked sibling: `[[annotate_sharpness_exploration]]`, `[[annotate_projection_coarsening]]`.*

---

## 1. The problem

Per-cell / per-cluster annotation posteriors are **flat**: the argmax cell type often wins at 0.1–0.25 over ~20 BMMC types. Confident only for transcriptionally distinct types (Late_Eryth 0.74, T_proliferating 0.83); confusable lineage leaves (CD8_Naive / CD8_T / CD4) split the probability mass. Affects **both** the enrichment and projection scorers, so it is upstream of either.

Root cause, established over a long design discussion:

- The dilution is a **collinearity / overlap** problem. Confusable types share lineage markers; any **symmetric** weighting (IDF `ln(C/df)`, empirical specificity) contributes the shared genes equally to all confusable types and **structurally cannot break the tie**.
- For some pairs (CD8_Naive vs CD8_T) the **discriminating evidence is not in the RNA assay at all** — it lives in TCR / surface protein (CD45RA/CCR7) / chromatin. No RNA-only method can recover it.

**Therefore the goal is not a sharper number.** The number *should* be flat when the evidence is flat. The goal is:

> **Report the finest level of the cell-type hierarchy at which the call is identifiable; abstain ("not enough resolution") below that; and surface what the data contains that the ontology cannot explain.**

---

## 2. Core idea

Treat annotation as **placing activity marks on a gene-annotated Cell Ontology (CL) DAG to explain each cluster's top-k genes**, under a submodular / information-theoretic objective.

- **Explaining-away** (a coarse mark covers shared lineage genes once → a leaf mark only pays off if it has *unique* present genes) = submodular diminishing returns. This kills the splitter dilution structurally.
- **Adaptive granularity**: marks land only at the depth that earns marginal coverage; otherwise the walk stops and reports the parent.
- **Abstention** = the refusal to break a sibling tie.
- **Disagreement** = the residual top-k genes no node explains.

This is the synthesis of six earlier framings (coarse-candidate recalibration, ESS prior covariance, adversarial "no" critic, counterfactual necessity, Rubin overlap, RAG/ontology back-off). They all reduce to "report the resolvable level"; this plan is the buildable mechanism.

---

## 3. Objects and the free lunch from an annotated ontology

- **DAG** `G = (V, is_a)` — CL restricted to the tissue. (Downloaded `cl-basic.obo`, release 2026-06-08: 3,335 non-obsolete terms, 4,664 `is_a` edges; **it is a DAG, 1,148 terms have >1 parent** — not a tree.)
- **Gene annotations** with the **true-path rule**: a gene annotated to a node is implicitly annotated to all `is_a` ancestors. `genes(t)` = genes at `t` or below (shrinks with depth).
- **Information content** `IC(t) = −log(|genes(t)| / |genes(root)|)` — the ontology's own annotation statistics. **This is the specificity weight IDF could never be**, and it is parameter-free.
- **Gene→mark affinity** = standard semantic similarity (Resnik):
  `a(g, t) = IC( MICA(d(g), t) )`, where `d(g)` = gene `g`'s deepest annotation(s), MICA = most informative common ancestor.
  - Multi-annotation genes (e.g. GZMB in NK *and* CD8-effector) contribute to both lineages → correctly *creates* a cross-lineage tie that the tie rules resolve.
- **Per cluster** `c`: top-k genes `G_c` with weights `w_{c,g}`. **k and the ranking metric are part of the model** (see §7).

Validated that CL structure supports the merge mechanism:
```
LCA(naive CD8, effector-mem CD8) = "CD8-positive, αβ T cell"   ← specific (splitters merge low)
LCA(naive CD8, CD4 αβ T cell)    = "mature αβ T cell" / "T cell" ← general  (separable merge high)
```

---

## 4. Algorithm

### Phase A — depth (resolution walk). Emits "not enough resolution".

Top-down from root; descend only while a single child dominates the evidence.

```
walk(node v, genes G_c):
    children = is_a-children of v supported by G_c
    for each child u: mass(u) = Σ_{g∈G_c} w_g · a(g, u)
    normalize mass over children → P(child | v);  h = entropy(P(child|v))
    u* = argmax mass
    if P(u*|v) ≥ τ and h ≤ h_max:     # one child dominates
        return walk(u*, G_c)           # commit deeper
    else:
        return STOP(v, candidates={u: mass(u) high}, residual_entropy=h)
```
Output = deepest unambiguous node + explicit "resolution-limited, unresolved among {children}" when it stops early. `h` at the stop = confidence-of-abstention.

### Phase B — multiplicity + disagreement (submodular residual).

After the primary path, explain the *uncovered* top-k genes; whatever survives is the disagreement output.

```
R = { g ∈ G_c : a(g, v*) low }
while R not empty and gain ≥ gmin:
    t* = argmax_t Σ_{g∈R} w_g · a(g,t)   # submodular marginal gain (facility location)
    place mark t*; remove explained genes from R
report remaining R as DATA-ONLY residual  # ontology can't explain → novel program / state
```

### Objective choices (all submodular)

- **Selection: info-weighted facility location** `f(S)=Σ_g w_g · max_{t∈S} a(g,t)` — soft, discrimination-aware, **provably submodular, no independence assumption**. Workhorse.
- **Depth: conditional entropy / MI** — submodular under naive-Bayes (genes ⊥ given type). Minimizing residual `H(Y|G_c)` ⇔ maximizing MI. Shared genes contribute ≈0 entropy reduction automatically — the discrimination falls out of the information measure.
- Coverage alone is **discrimination-blind** (covering a shared gene scores like covering a discriminative one) → do not use as the sole objective.
- Constraint: **DAG-antichain matroid** (a cluster → an antichain; no ancestor+descendant double-mark unless both earn gain) → greedy keeps guarantees.

Complexity: propagation + IC precompute once `O(|genes|·depth)`; per cluster cheap; **parallel over clusters (rayon)**.

---

## 5. Tie-breaking — most ties are NOT broken

A tie is the signal, not a nuisance. Categorize by location in the DAG:

1. **Sibling tie (same parent) — do NOT break. This *is* the abstention.** CD8_Naive vs CD8_T ~50/50 → stop and report the parent ("CD8 T cell, subtype unresolved"). `τ` is the only knob (how dominant a child must be to descend).
2. **Ancestor vs descendant — Occam + necessity, default shallower.** Prefer the more specific node *only if its extra specificity is free*: `score(t) = explained(t) − λ·(annotated_in(t) but absent_in G_c)`. The necessity penalty is the "say no" critic returning as the tie-breaker. Still tied → go shallower.
3. **Cross-lineage tie (different subtrees) — back off to LCA, or report multiplicity.** Informative LCA → report it ("T/NK ambiguous"); both independently supported by different genes → doublet/mixed (Phase B finds it); uninformative LCA → flag "cross-lineage ambiguous".
4. **Exact numerical ties — deterministic, reproducibility only.** Fixed key `(higher π_t, higher IC, lower CL id)`. Keep separate from the semantic rules so a deterministic tie-break never makes a biological decision.

**Unifying principle: bias every unresolved tie toward *claiming less* (shallower / higher LCA / abstain).** That single asymmetry makes the system honest — it commits to depth only when forced, and "not enough resolution" is the default.

---

## 6. Outputs

- **Assignment**: CL node per cluster at adaptive depth (+ ancestor path).
- **Abstention**: explicit "resolution-limited" flag + candidate children + residual entropy.
- **Disagreement (the novel byproduct)**: top-k genes uncovered by any node ("ontology cannot explain") + clusters that accept only a coarse mark ("definitionally distinct, empirically unresolved in this assay").
- **Provenance**: per assignment, the genes that grounded it (with IC), what was covered vs residual.
- **Annotation-depth instrumentation**: per node, how many genes support it (3 vs 300) — trust cap (see §9).

---

## 7. The k / gene-ranking interaction (do not skip)

The ranking that defines "top-k genes" determines which objective works:
- top-k by **expression magnitude** → dominated by shared lineage genes → coverage points coarse, discriminative signal never enters the budget → can't resolve subtypes.
- top-k by **specificity / DE-vs-rest** → discriminative genes survive → entropy has signal.

Plan: take top-k by a **specificity-aware** score so discriminative genes enter; let the entropy objective decide whether they actually disambiguate; if not, abstain.

---

## 8. Visualization / decision-support (UMAP + DAG)

Purpose: let the user **decide the resolution/confidence trade-off and judge abstentions** — the `τ`/`k` knobs become interactive, with consequences shown on the embedding. (UMAP lives in `matrix-util`; senna already has PCA-init UMAP layout.)

1. **Adaptive-depth coloring on UMAP**: cells colored by their *assigned node at the resolved depth* — resolved leaves get saturated colors, abstained cells get the **parent color, desaturated**. Low-resolution regions are *visible as desaturated blobs*.
2. **Entropy / confidence layer**: recolor UMAP by residual entropy → where the calls are ambiguous.
3. **Linked DAG panel**: click a CL node → highlight cells assigned at/below it; a **`τ` slider re-walks the DAG live and recolors UMAP** → the user *sees* over- vs under-resolution and picks the level. This is the "help users decide" core.
4. **Disagreement view**: highlight clusters with large uncovered residual → "ontology fails / novel program here".
5. **Per-cluster evidence card**: selected cluster → top-k genes, the CL node each supports (with IC), covered vs residual, the candidate siblings at the stop node (arc/pie). The "why this call / why abstained" panel.
6. Output as parquet (assignment + entropy + residual + path) → static report first; interactive later.

---

## 9. Prior art & honest novelty positioning

The pieces individually exist. The **combination** does not. Be precise about this.

**Abstention / internal-node "reject option" — EXISTS, not novel by itself:**
- **CellO** (Bernstein et al., *iScience* 2021) — hierarchical classification on CL; places cells at **internal nodes** when uncertain; corrects probs with the ontology. *But supervised (trained on reference atlas), isotonic-regression correction, not annotation/coverage-driven.* [link](https://www.cell.com/iscience/fulltext/S2589-0042(20)31110-X)
- **Uncertainty-aware annotation with a hierarchical reject option** (2023/24) — explicit "partial reject" (return internal node) vs "full reject" (below threshold). *Closest framing to our abstention; supervised.* [link](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10957513/)
- **GPTAnno** (2025) — ontology-tree-guided, automatic resolution selection, hierarchical, with uncertainty. *LLM-based.* [link](https://www.biorxiv.org/content/10.1101/2025.11.27.690951.full.pdf)
- **Hierarchical cross-entropy loss** (2025) — atlas-scale, visualizes on the CL DAG. [link](https://www.biorxiv.org/content/10.1101/2025.04.23.650210)

**Explaining-away on an ontology DAG — EXISTS, but in GO-enrichment land, not cell typing:**
- **MGSA** (Bauer, Gagneur, Robinson, *NAR* 2010) — Bayesian network selects a minimal set of categories that **explain** the gene list, accounting for overlap (explaining-away). *Conceptual twin; Bayesian-network inference, not submodular; for GO enrichment of a gene list, not cluster cell-typing.* [link](https://academic.oup.com/nar/article/38/11/3523/3100635)
- **topGO elim/weight** (Alexa et al. 2006) — DAG-conditional enrichment; de-redundifies ("a term looks enriched just because its children are"). [link](https://bioconductor.org/packages//release/bioc/vignettes/topGO/inst/doc/topGO_manual.html) ; **evoGO** (2025) redundancy minimization [link](https://www.biorxiv.org/content/10.1101/2025.02.24.639258.full.pdf)

**Submodular in genomics — EXISTS, for selection, not ontology assignment:**
- **Submodular assay-panel selection** (Wei/Libbrecht/Bilmes, *Genome Biology* 2016) — facility location. [link](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-016-1089-7)
- **scGeneFit** (*Nat Commun* 2021) — label-aware marker-gene selection, hierarchical labels. [link](https://www.nature.com/articles/s41467-021-21453-4)

**Marker-based hierarchical typing:**
- **Garnett** (Pliner et al., *Nat Methods* 2019) — interpretable marker markup, hierarchy + "unknown". *User must pre-specify the hierarchy and markers; elastic-net per node.* [link](https://cole-trapnell-lab.github.io/pdfs/papers/pliner_garnet_NM_2019.pdf)
- **OnClass** (Wang et al., *Nat Commun* 2021) — embeds CL `is_a` graph; zero-shot to unseen terms; can derive markers for unseen types. [link](https://www.biorxiv.org/content/10.1101/810234v2.full)

### Where this plan is actually new (the defensible gap)
1. **Reference-free**: annotation/marker-driven via the ontology's own gene annotations + IC — no labeled-cell reference (unlike CellO, OnClass, the reject-option work). Closest peers (Garnett) need a hand-built hierarchy; MGSA isn't cell typing.
2. **Submodular coverage / conditional-entropy as the explicit assignment objective** on a gene-annotated CL DAG, with greedy guarantees and the antichain matroid — MGSA is Bayesian-network, topGO is conditional testing, CellO/reject-option are trained classifiers. None frame cluster→ontology as submodular coverage with IC affinity.
3. **Data-vs-DB disagreement as a first-class output** — the uncovered residual / "definitionally distinct, empirically unresolved" flag. Prior methods abstain but do **not** report *what the data has that the ontology lacks*.
4. **Abstention = refusal to break sibling ties**, with a uniform "claim less when tied" rule, rather than a confidence threshold bolted on.

**Honest caveat for the writeup:** the *abstention/internal-node* contribution alone is well-trodden (CellO, hierarchical reject, GPTAnno). Lead with the **reference-free submodular/IC mechanism + the disagreement output**, not with "we can abstain."

---

## 10. Open decisions / risks / what to validate first

**Validate before building (the go/no-go):**
- **Method vs information**: do the discriminative genes (e.g., CD8_Naive vs CD8_T) even exist in the chosen gene→CL annotation set and show dynamic range in the data? If not, the method *correctly* abstains everywhere it matters — confirm that's acceptable (it is, per the goal) but know it up front.
- **Annotation coverage / depth** of the gene→CL map. CL's native gene axioms are sparse; may need a GO bridge or a CellMarker/PanglaoDB→CL projection to thicken. **Annotation completeness, not the algorithm, caps resolution** — instrument per-node support count from day one. Sparse annotations → conservative over-abstention (safe given the goal).

**Decisions:**
- Single blended objective (`α·facility-location + β·MI`, one knob) vs **two-stage** (facility-location selects antichain, entropy walks depth). → start two-stage (inspectable while affinity is unproven).
- Gene→node annotation source: CL native `expresses`/logical axioms vs GO bridge vs CellMarker/Panglao→CL. → likely a union, with provenance per edge.
- `τ` / `h_max` calibration (descend eagerness) → expose as the UMAP slider; calibrate against the confident-tail types that should stay deep.
- Tree-ify CL vs query the DAG via induced-LCA over the chosen leaf set → **induced-LCA over the label set** (never flatten 3,335 terms; the monocyte subtree shows the multi-axis thicket that breaks naive tree-ification).

**Risks:**
- Creates no information — uncovered residual stays uncovered (correct, but set expectations).
- `a(g,t)` quality is the whole ballgame; IC weighting helps but garbage annotations → garbage affinity.
- Greedy is approximate; with the necessity penalty use distorted-greedy (Harshaw et al.) for the regularized form.

---

## 11. Implementation sketch (legume-rs)

- Likely a `senna` subcommand (`annotate-ontology`?) reusing the `graph-embedding-util::type_annotation` plumbing (marker parsing, IDF→replace with IC, output writers) + the existing `coarsen.rs` (it is already a one-level version: Leiden communities → merge map → lexical label).
- Inputs: cluster × gene top-k table (or compute from embeddings/counts), `cl-basic.obo` + gene→CL annotations.
- Core: DAG parse + propagation + IC; per-cluster Phase A/B (rayon); outputs parquet (assignment, path, entropy, covered, residual).
- Viz: parquet → matrix-util UMAP coloring + linked DAG panel.
- Per-crate version bump on the new subcommand; `cargo fmt` before commit; omit Co-Authored-By lines.
