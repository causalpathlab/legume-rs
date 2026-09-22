# Chickpea

Chromatin Interactions Captured by Knitting Peaks with Expression Anchors


## TODO

* [x] download ABC / E2G reference data (Engreitz lab schema + parquet)

* [x] study ABC model.. what is the generative model? it's just activity co-occurrence

* [x] practical algorithm (no rSVD; no GhostKnockoff / LOCO-TMLE / SuSiE surgery):
    1. link peaks to genes on pb profiles: `--link-score pearson` (log1p correlation, default)
       or `abc` (Engreitz activity × contact with the 1 Mb pseudocount, ATAC only);
       `--top-k-per-gene` cuts each gene's ranked list (ATAC-only: ArchR gene activity is
       the RNA stand-in; circular under pearson, and the gene axis of the embed either way)
    2. train peak + gene + pseudobulk embeddings with the `graph-embedding-util` `fit/hier`
       trainer over a Vec of feature partitions: frozen pb units for every level of the pb
       tree, one sparse count axis and one module partition per feature kind (genes, peaks),
       one shared unit table; a linked peak starts in its strongest gene's module, unlinked
       peaks are k-means modules of their own; optional merge-only re-collapse from the
       frozen unit profiles (`--merge-every`, off by default). Replaced the FNE / SIMBA
       count-edge table (2026-09-21)
    3. cluster the finest pb rows (min-sample gate) → **refine peak→gene within each
       cluster** (pb-per-cluster Pearson / ABC)
    4. emit **E2G-like parquet**:
       - `peaks.parquet`: id, chromosome, start, end, class  (← enhancers)
       - `clusters.parquet`: id, name  (← cell_types)
       - `peak_gene/chr*.parquet`: id, score, target_gene_id, target_gene_name, target_gene_tss, enhancer_gene_distance, model, chromosome, enhancer_id, cell_type_id (= cluster)  (← enhancer_gene_predictions)
       plus `{out}.{peak,gene,pb}_embedding.parquet` and `{out}.pb_tree_embedding.parquet`

* [x] 2026-09-21, measured on the simulator and dropped (not in the tree):
  a bge feature-embedding affinity as the link evidence, an fne contrast with
  trans-permuted negatives, and an ABC-triplet logistic NCE against chance ABC.
  None reached the log1p Pearson; the sim's causal signal is cross-pseudobulk
  co-variation of RNA with the causal peaks, which Pearson measures directly and
  ABC-shaped targets exclude by construction. What the sim does NOT contain is a
  bystander that co-varies through shared cell state — the case Pearson is wrong
  about; a simulator knob for that is the prerequisite for any further evidence term.

* [x] 2026-09-21, `data-beans-sim multiome` at `--pve-cis` 0.3 and 0.8 (cis-window 50 kb,
  sort-dim 8, 2 levels, 32 modules per axis, 64 units/step): the hier embed takes seconds,
  and gene–peak cosine on the trained tables ranks causal peaks well above the Pearson link
  that seeds it (AUROC ≈ 0.85 at 10 epochs, ≈ 0.89 at 100, vs 0.64–0.70 for Pearson).
  The link/refine stage is now the weak stage, not the embedding.

* [ ] **fine-tune the links after the embed.** Today the trained gene / peak rows only
  reach the output as side parquets; the refine re-runs the same Pearson / ABC on each
  cluster's raw pb columns. Read the embedding in the link stage: gene–peak cosine (or a
  unit-gated score over a cluster's pb rows) as the evidence, or folded into the Pearson
  by product / rank average; re-measure on the sim above.

* [ ] **gene–peak co-embedding term.** Genes and peaks share a space only through the
  unit table; no loss term touches a gene and a peak directly, the cis links only seed
  the partitions. Add a coupling term (linked-pair attraction against sampled cis
  negatives, or a peak→gene aggregation in the gene axis) and check it does not
  reintroduce the FNE cost regime.

* [x] **cell-level embedding (phase 2)** (2026-09-21). A per-axis cold Poisson-MAP
  engine in `graph-embedding-util` (`fit/projection/block_sgd/axes.rs`): one partition
  and one intercept per feature axis, one shared latent, streamed from the per-modality
  backends in groups. chickpea projects every cell onto the frozen gene and peak
  dictionaries, writes `{out}.cell_embedding.parquet`, clusters cells, and labels each
  finest pb by the majority of its cells for the refine. ATAC-only projects on the peak
  axis alone; the pb-only clustering path is gone. `tracks.rs` / `polish_cells`
  untouched (senna follow-up: encoder-only vs solve-only, then route bge onto this engine).

* [ ] batch fold in phase 2 once the collapse exposes a per-cell δ (today the phase-2
  counts are raw; the pb-level δ-correction does not reach the cells).

* [ ] open question: how should we model multi-resolution Y (gene RNA/ATAC) ~ X (ATAC peaks, 1kb / 10kb / 100kb)?

* removed (e13492ab): rSVD ATAC embedding, SuSiE-RSS on embedding z/R, GhostKnockoff FDR,
  LOCO-TMLE, and their derivations under `docs/`



## longer term plan

What this crate can do... broadly not yet

1. peak to gene interaction mapping... 
    - cell type can be confounder... peak -> cell type <- gene
    - 

2. given peak to gene map (either we build here or ABC), 
    - we can estimate common embedding space for genes and peaks
	- using the embedding results, we can interpret GWAS
    - while fixing SNP embedding, can we identify trait embedding?

3. we want to use graph embedding util 



what is the goal? the goal is mediation... can we?

- snp/peak (genomic location) -> molecular trait -> trait

- how do we learn directional?
    - maybe we need to take care of unobserved confounders? 
    - predicting LD-driven snp snp interaction is about negative energy...



### Multi-Resolution Peak-Gene Linking and Embedding

X (ATAC peaks, 1k-100kb) -> Y (RNA genes)

#### Simulation scheme


* $A_{ir}$: ATAC for a peak $r \in [p]$ in a cell $i \in [n]$

* $X_{ig}$: gene expression for a gene $g \in [m]$ in a cell $i \in [n]$

1. generation of ATAC data

$$A_{ir} \sim \text{Poisson}\left(\rho_{i} \sum_{t} \theta_{it} \beta_{tr} \right)$$

where multiplicative noise $\ln \rho_{i} \sim \mathcal{N}\!\left(0,\sigma_{\rho}^{2}\right)$

2. generation of RNA data

$$X_{ig} \sim \text{Poisson}\left(\tau_{i} \sum_{t} \theta_{it} \sum_{r} \beta_{tr} M_{gr} \right)$$

where multiplicative noise $\ln \tau_{i} \sim \mathcal{N}\!\left(0,\sigma_{\tau}^{2}\right)$

* We have an indicator matrix $M_{gr}$ that maps a region $r$ to a target gene $g$.

* We can restrict to cis-regulatory regions per gene

* We can make $M_{gr}(t)$ per topic $t$. 

* Need to provide GTFs

* Think about consistent gene and peak naming across workspace
