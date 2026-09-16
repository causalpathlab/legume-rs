# Plan — A text-anchored gene space: literature priors, a concept vocabulary, and grounded hypotheses for senna

*Working notes, 2026-09-15. Status: design, no code. Builds on `annotation-ontology-plan.md` (the depth axis) and the deep-research report of the same day (108 agents, 26 primary sources, 23/25 claims verified). Sibling memory: `[[pretrained-llm-leverage-brainstorm]]`.*

---

## 0. Stance

Do not compete with single-cell foundation models; take the one asset they cannot supply and we cannot learn from one dataset — **gene identity grounded outside the data** — and let everything else stay as it is: sparse, likelihood-based, pseudobulk-first. The 2025–26 pattern we copy is the LLM world's: a small specialised model, frozen outside knowledge, retrieval, and tools. Nothing in this plan imports a backbone.

## 1. The problem

senna learns everything from the run in front of it. The axes that stay weak are exactly the ones shared across every dataset:

- **Gene side.** ρ on multiome scored AUC 0.44 against 0.65 for a clean run; gene-pair NCE and module priors did not move it. A dictionary learned from one dataset's co-variation is the low-sample object.
- **Annotation.** Per-cell confidence is flat (~0.2); `annotate-by-projection` was indistinguishable from a shuffled marker panel on BMMNC (panel-null p > 0.14 for 7/8 types); the empty-marker magnet and the panel-must-match pathology are both gene-side failures — markers the run never learned.
- **Batch / query placement.** The indexed encoder cannot batch-correct cleanly; a frozen reference embedding does not batch-correct at all.

## 2. What the evidence settled (do not re-litigate)

**Dead.** Importing a foundation-model backbone for embedding quality — simple baselines match or beat every scFM including after fine-tuning (Ahlmann-Eltze, Huber & Anders, *Nat Methods* 2025; VCBench, bioRxiv 10.64898/2026.06.18.733146; Han et al., bioRxiv 10.64898/2026.04.17.719314; Arc VCC 2025). Expression-FM gene tables (Geneformer, scGPT) as a ρ prior — beat the mean, not PCA from the training data (Ahlmann-Eltze). Scale — pretraining saturates near 200k cells (DenAdel et al., *Nat Methods* 2026). Dense-input models.

**Confirmed.** Retrieval over our own reference is the cell-side state of the art (VCBench; scTOP, arXiv 2602.16696 — a parameter-free projection onto a *pseudobulk basis*). Calibration comes from **agreement across heterogeneous annotators**, not from any single score (popV, *Nat Genet* 2024). A gene-table + MLP encoder suffices for atlas-scale retrieval annotation (SCimilarity, *Nature* 2025; DenAdel).

**Open, and ours to test.** LLM *text* embeddings of gene descriptions beat expression-learned gene embeddings on gene-function tasks (OMIM AUROC 0.88 vs scGPT 0.74; bioRxiv 10.1101/2025.01.29.635607), work as an *additive* prior on an expression dictionary rather than a replacement (scGenePT), need functional description text — bare symbols are near-random (GEbench, bioRxiv 10.64898/2026.04.30.721875) — and are anisotropic (cross-gene cosine 0.74–0.84; scELMo, *Patterns* 2026). Open-weight encoders beat the original API embeddings (bioRxiv 10.64898/2026.04.16.718976). **Nobody has tested a text prior inside a Poisson/topic model, and nobody benchmarks robustness to count noise or per-cell calibration.**

## 3. Core idea — genes are a bilingual dictionary

Two languages. In the expression language a cell is a sentence over genes and the model places it at `z_c ∈ ℝ^H`. In the text language every concept with a description — gene, Cell Ontology term, pathway, disease, state, transcription factor — has a vector `k ∈ ℝ^{H_text}` from one text encoder. **Genes exist in both**, so they are a parallel lexicon, and the map between the languages is fitted on them (the MUSE setup: Conneau et al., ICLR 2018 — Procrustes on a bilingual lexicon aligns whole vocabularies; CSLS corrects hubness).

The gene dictionary becomes

```
ρ = diag(s) · (K_text · W₀ + U · V) + b_g        U: [D, r], V: [r, H], r ≪ H
```

- `K_text` frozen `[D, H_text]`, centred and whitened.
- `W₀: H_text → H` the translation map (PCA basis for the probe; trained adapter for the real version).
- `U·V` a LoRA-shaped, shrunk residual — `r = 0` is "fully prior", `r = H` is "learn from scratch", so the experiment is a rank sweep. LoRA on a gene table has no precedent in single-cell; adapters on transformer backbones do (scPEFT, *Nat Mach Intell* 2025) and beat full fine-tuning on small or shifted data.
- `diag(s)` and `b_g` are not optional: text tables carry no notion of abundance and a bilinear Poisson decoder needs per-gene magnitude (the DoRA magnitude/direction split, Liu et al., ICML 2024). The topic models currently carry no `b_g`.
- Unmatched genes (no text) keep a **free** row; they are not dropped.

Once the map exists, every text concept can be translated into the model's space, including ones with no expression counterpart (a disease name), and every model object can be translated out (a topic β as an expression-weighted sum of gene text vectors — scELMo's weighted-average mode).

## 4. What it buys, by layer

### 4.1 Gene side (the direct claim)
Anchored genes cannot be dragged into a modality-dominated geometry — the CITE-seq ADT collapse on record (14 rows, 68 % of counts) is the first test. ADT rows take their target gene's text. Convergence: the itopic "SVD ρ-init" accelerator, with better content.

### 4.2 Annotation — depth × breadth
- **Breadth: the vocabulary.** A cluster is described by a *set* of scored concepts across axes — identity (`CD8⁺ T cell`), state (`exhaustion`), program (`interferon-α response`), context (`bone marrow`), TFs — where a tree could only give one path. Two directions: data → text (weighted sum of gene text vectors; no training) and text → model (`s_v = k_v W₀`, scored by the model's own bilinear rule, i.e. **concepts as pseudo-genes**).
- **Depth: ontology levels in bge phase 1.** Cell Ontology depths as additional pseudobulk partitions (label-pure pseudobulks per depth) alongside the unsupervised pb tree. ρ must explain composition at every granularity; the per-level pb embeddings *are* CL-term embeddings in the cell/gene space; annotation walks levels and **stops at a sibling tie** (the abstention rule of `annotation-ontology-plan.md`, now with a trained geometry); a held-out fine type retreats to its ancestor instead of being dropped. This is SCimilarity's one real ingredient — its labels live on the CL — taken without its dense input, label-triplet loss, or `1/distance` OOD threshold.
- **Predicted division of labour.** Text resolves lineage and is blind to siblings (CD4⁺ and CD8⁺ T-cell descriptions are the same sentence); the expression-trained levels resolve siblings or abstain. The two cover each other's weakness.
- **`annotate-by-projection` revived.** It failed because it projected through a gene space that had not learned the markers. With anchored ρ, marker signatures are distinct from shuffled ones by construction, and the type-text route needs no panel at all — so the empty-marker magnet and panel-must-match cannot occur.

### 4.3 Multiome — regions as words with no text
Peaks have no description, but they share cells with anchored genes, so free peak rows settle next to the genes and concepts they co-vary with: a functional annotation of a region with no gene assigned by hand. Honest reading: co-placement is **program membership** (trans), not a cis link. The cis call belongs to chickpea (peak-to-gene + GhostKnockoff FDR); the anchored space supplies the *prior*. Optional partial anchor for peaks: motif content as a sentence of TF words, `k_peak = Σ_motifs k_TF`. Validation is expression-independent: CRISPRi enhancer–gene pairs (Fulco et al., *Nat Genet* 2019; Gasperini et al., *Cell* 2019; ENCODE-rE2G, Gschwind et al., bioRxiv 2023).

### 4.4 Hypothesis sentences — the last mile
Per cluster, the scored concept set with its abstentions is the evidence; a lightweight LLM **composes, never decides**. Structured JSON in; one or two sentences plus the list of concepts used out; a verifier rejects any sentence naming an entity outside the retrieved set (the RAG citation check). Abstentions are a required slot ("a T cell whose CD4/CD8 identity the data does not resolve"). What makes it a hypothesis rather than a caption is the proposed test — "*HAVCR2* and *LAG3* should be elevated relative to cluster 3" — which the tools can run. Local models (Qwen3-1.7B/4B-Instruct, Phi-4-mini via `candle-transformers`) are sufficient; the composer lives in the same thin binary as the text encoder, never in senna. Prior art to position against: GPTCelltype (Hou & Ji, *Nat Methods* 2024) and scChat hand a marker list to GPT-4; CellWhisperer trains a CLIP-style captioner on paired data. Ours needs no pairs — genes are the pairs — and the composer is sandboxed.

## 5. Ruled out on the way (keep out)

- SCimilarity as a model or index: dense 28k-gene input, label-triplet metric learning with a 0.05 margin, robustness only to noise present in its training atlas, and drift handled by *dropping* cells. Its architecture class (gene table + pooling + MLP) is bge's phase-2 encoder trained on labels instead of on the pb tree; bge additionally has a count likelihood and depth augmentation by construction.
- Expression-FM gene tables in the prior bank; Geneformer backbone via candle (the only licence-clean one, recorded for completeness); Stack-style in-context cells (parked, +1.8 % margin); remote embedding APIs; Census-embedding joins (parked, undecided reference).

## 6. Objects and seams that already exist

- `auxiliary-data::frozen_features::load_frozen_feature_host` — any `[D, H]` parquet with the gene name in row column 0, name canonicalisation (`ENSG…_TGFB1` ≡ `TGFB1`), strict intersection. Reached by `senna masked-topic --init-feature-embedding <prefix>` / `--freeze-feature-embedding <prefix>` via `run_manifest::resolve_feature_loading` (`{prefix}.feature_loading.parquet`). `topic` takes ρ through `--from <manifest>`; `bge` only through `senna update`'s parent manifest.
- `candle_util::frozen_features::{install_frozen_var_2d, trainable_vars}` — freeze vs. init are the two modes; the LoRA mode is the missing third.
- `graph-embedding-util::fit::projection::CellEncoders` (post PR #58) — the trunk that places query cells; the second attachment point for the same LoRA primitive (query-side adapter for batch; same self-supervised pb objective, no labels, retrieval-anchored term later).
- bge phase 1 (`fit/hier`) — exact two-level softmax over a stack of pb partitions; an ontology depth is one more partition.
- `annotate-by-{projection,enrichment}` with bootstrap + permutation — the shuffled-panel null is the test projection failed and must pass.
- `--poisson-thin` — the robustness benchmark needs nothing new.

Gaps: the LoRA primitive (~50–100 lines; `candle-lora` wraps `candle_nn::Linear`, ours are candle-util's own); `b_g` for the topic models; an `H_ext → H` adapter (PCA for the probe); "unmatched → free row" in the loader; `--init-feature-embedding` on `topic`/`bge` (one hook each, only if the probe earns it); a thin binary `gene-text-embed --model <hf-id>` (`hf-hub` + `candle-transformers` 0.10.1 + `tokenizers`, kept out of senna's build).

## 7. The probe (before any model code)

All steps in R (vignette convention) or the thin Rust binary; a single-run difference proves nothing on this codebase, so three seeds throughout.

1. **Tables → parquet.** BioConceptVec (PubMed word2vec/fastText — the literal "trained on PubMed" table), GenePT / scGenePT (NCBI, UniProt, GO text), the 2026 open-backbone GenePT release, scELMo. Align to the run's gene axis, report coverage, centre + whiten, PCA to H, pre-fill unmatched genes with the run's own init so nothing is dropped. Write `{prefix}.feature_loading.parquet`.
2. **Residual diagnostic on existing runs (no training).** Regress a trained bge ρ and a masked-topic ρ on each `K_text`: R², residual singular spectrum (→ the rank), and *which* genes are explained — expect a literature-attention bias (famous genes explained, obscure ones not); this decides whether the prior is about biology or fame.
3. **Init A/B on `masked-topic`** (flag exists, zero Rust). Random vs each text init. Two claims, measured separately: **convergence** (epochs to the random init's final ELBO) and **optimum** (held-out LL, gene-side AUC, topic coherence, ARI). Track `‖ρ_t − ρ₀‖/‖ρ₀‖` — an init that is washed out in a few epochs has only tested convergence.
   - **3b. Projection revived, on BMMNC** (the adversarial bed). Type signatures both ways (marker mean through anchored ρ; type text through `W₀`); the existing permutation null; agreement with `annotate-by-enrichment` and trusted labels. Expect lineage agreement to jump and sibling agreement to stay flat.
   - **3c. Training-free vocabulary.** Name an existing run's topics by nearest concepts; on a dataset with a known state (IFN-stimulated or tumour-infiltrated), do `interferon response` / `exhaustion` surface for the right cells — the result a hierarchy cannot produce.
4. **CITE-seq ADT check.** Anchored ρ on the multiome run that collapsed: does the gene AUC recover with the ADT rows on their target-gene text.
5. **Branch.** Convergence only → ship as an init option. Quality → build the anchored form (LoRA primitive, `b_g`, adapter), rank from step 2. Nothing → the idea is dead for the price of a day.

**Gene-side truth is expression-side** (held-out co-expression, STRING PPI, CITE-seq pairing) — never GO, which leaks from the text.

## 8. After the probe (order, with dependencies)

A. **Robustness benchmark** on current bge: Poisson-thin queries to 10–20 %, inject ambient at a few %, measure retrieved-label stability. The primary evaluation axis for everything below and an unpublished gap.
B. **Ontology levels** in phase 1 (backbone `is_a` path per term for distillation targets; rank-normalised CL cut; partial coverage per level is fine). Leave-one-term-out: a held-out `CD8⁺` cohort must stop at `T cell`, not become `CD4⁺` — under A's noise too.
C. **`annotate-by-retrieval` + agreement**: kNN vote over our own reference (impute core, IVF index) as one voter beside projection, enrichment, the scTOP-style pb-basis projection, and the ontology nodes; confidence = consensus count; OOD = distance *and* low agreement. Calibration curve; leave-one-type-out; BMMNC.
D. **LoRA primitive, second attachment**: query-side adapter on `CellEncoders` for batch (rank r regularises against absorbing biology; hold out a cell type and see whether it survives adaptation).
E. **Multiome**: anchored genes + free peak rows; enhancer–gene benchmark; prior handed to chickpea.
F. **Concept sets → hypothesis sentences**, with the template baseline as the control (a reader's ability to identify the cluster; fraction of proposed tests that turn out true).
G. **MCP server** over senna/pinto outputs — the tool layer the sentences' proposed tests run against; independent of A–F.

## 9. What would fool us

Fame not function (step 2); GO leakage (expression-side truth only); coverage — lncRNA and novel genes have no text, ADT rows use the target gene; hubness in text space (CSLS); siblings are text-blind by nature (depth axis handles them); PCA-to-H loses the prior (acceptable for the probe only); an init washed out by AdamW masquerading as "no effect"; single-seed A/Bs; and the multiome trap of reading co-placement as a cis link.

## 10. Decisions still open

1. Reference atlas for B/C: own labelled atlases (trusted labels, matched protocol) vs a Census subset (breadth, CL labels, mixed protocol) — D exists for the mismatch case.
2. Which existing run's ρ is the test bed for step 2 (BM1, panc8, the multiome run).
3. Whether the thin text binary is built before or after the probe (the probe does not need it).
