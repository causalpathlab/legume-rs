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

* [ ] **fine-tune the links after the embed** (superseded in part by the module-network
  item below; the quadratic form is the same, read at module level). Today the trained
  gene / peak rows only reach the output as side parquets; the refine re-runs the same
  Pearson / ABC on each cluster's raw pb columns, and on both the sim and real data that
  score barely favors promoter-proximal peaks while the rows already rank causal peaks
  well. The embedding makes the link a quadratic form: with `ρ_g`, `ρ_p` the frozen rows
  and `Σ_S` the covariance of the cell latents `θ` over a cell set `S`,

      cov_S(g, p)  = ρ_gᵀ Σ_S ρ_p
      corr_S(g, p) = cov_S / sqrt(ρ_gᵀ Σ_S ρ_g · ρ_pᵀ Σ_S ρ_p)

  is the same statistic as today (a correlation) on the model's denoised rates, costs
  one H×H matrix per cell set plus a few dot products per cis pair, and needs no pb
  columns. Three shapes, in order:
  1. global link from the rows alone: cosine of `ρ_g`, `ρ_p` over the cis candidates
     (already measured: AUROC 0.89 vs 0.70 for Pearson on the sim); drop-in for the
     phase-1 link map, cell-type agnostic;
  2. per-cluster latent correlation (recommended next): `Σ_k` over the cells of cluster
     `k`, gated by activity so a link is reported where the peak is open in that cluster
     (`exp(⟨ρ_p, θ̄_k⟩ + b_p)`), distance decay kept as the prior. Within-cluster removes
     the between-lineage axis, which is the bystander-through-shared-state failure; what
     remains is the within-state co-variation the sim plants as its private signal;
  3. a gene–peak coupling term in phase 1 (linked pairs attracted against sampled cis
     negatives); only after 2 shows what the rows can already do.
  Plan: prototype 2 in R from the sim's parquets (gene / peak / cell embeddings +
  ground truth exist for pve-cis 0.3 and 0.8), compare Pearson, row cosine, global latent
  correlation, within-cluster latent correlation with the same top-k / AUROC; then a new
  scorer in `link_map.rs` / `refine.rs` taking rows and a covariance instead of pb
  matrices. Caveat: the sim has no bystander that co-varies through shared state, so 2
  can only be shown not to lose on the easy case until that simulator knob exists.

* [x] **phase 2 by distilled encoders, measured and rejected** (2026-09-22). Implemented
  the per-axis encoder path (one trunk per axis distilled onto the phase-1 pb tables,
  refined on the cells' likelihood summed over axes, no polish) and compared it with the
  cold per-cell MAP on the same PBMC cells. The encoder lost on every count: the B-cell
  island dissolved into the lymphoid and myeloid arcs, NK and T mixed, cluster agreement
  with the MAP was low (ARI 0.40, 15-NN overlap 0.14), cluster markers were less clean,
  and it was slower (23 min vs 16), because the likelihood refine runs on one core
  (candle's CPU elementwise and reduction ops over the 180k feature axis are
  single-threaded) while moving the loss by a tenth of a percent. Distillation alone
  reached held-out cosine 0.93 on pb targets in 3 min, but the pb tables are too coarse a
  target to place cells within a lineage. The cold solver stays; the encoder code is not
  kept. Same lesson for senna bge: the encoder's placement is not a substitute for the
  per-cell solve there either, so the thing to delete is the encoder, not the polish.

* [x] **phase 2 warm start** (2026-09-22). Every cell starts at its finest pseudobulk's
  phase-1 row, each axis intercept at its exact conditional MLE given that start, and
  the block runs on the polish step budget. The budget, not the tolerance, is what a good
  start saves: Adam's normalised step does not shrink because the start is close, so the
  relative-step test fires on the learning-rate schedule either way. Real data: the
  projection went from 12.5 min to about 3 min; agreement with the cold solve is
  moderate at the cluster level, and the per-edge deviance is a little higher, both of
  which are bounded by the dictionary (next item), not the solver.

* [x] **the dictionary was module-level** (2026-09-22). A feature row is `μ_module +
  r_feature`; at 10 epochs (50 SGD steps on 1.2k pseudobulk units) the residual sd per
  coordinate sat exactly at the init scale on both axes, so every row was its module's
  row plus noise, a third of each row's energy random. Finer partitions do not help
  (the partition is k-means on the same pseudobulk profiles); more steps help slowly
  (500 steps moved the peak residual to 0.14). Decision: **peaks carry no residual**
  (`HierConfig::module_only`): the row is the module's row and the bias the peak's
  closed-form share of the module's counts, `ln(total_p / total_m)`; the within-module
  softmax over 144k peaks disappears, and a phase-1 step fell from 450 ms to 68 ms.
  Genes keep module plus residual. Cost on the sim, until peaks get rows again: the
  gene–peak cosine link falls from AUROC 0.85 to 0.75 (cell clustering unchanged, ARI 1).

* [x] **cells as phase-1 units** (2026-09-22). `--phase1-cells-per-pb 16` (bge's rule
  and default, not to be tuned): at most 16 cells per pseudobulk at every level, union,
  read from the backends as one group over the gene and peak axes and appended to the
  unit table as a level of their own. On the real set the finest pseudobulks hold about
  a dozen cells, so this is nearly every cell; an ATAC-only run gives each cell an empty
  gene row, which the axis weight masks. Motivation: 1000 pseudobulk-only epochs left
  the loss flat while the gene residual kept creeping, i.e. the pseudobulk table starves
  the residuals; a feature only gets gradient from the units that draw its module.

* [x] **threaded phase-1 step** (2026-09-22). A step's units are split into slices, each
  slice draws its modules, builds its loss and runs its own backward on a thread, and
  the gradient stores are summed; exact, since the loss is a sum over units. About
  2× on ten threads (the step was single-core: small GEMMs and `exp` over padded
  within-module batches). Pseudobulk-only, 1000 epochs is now a few minutes.

* [ ] **phase 1 with cell units is still too slow for 1000 epochs** (next). Ten times the
  units means ten times the steps per epoch, and the threaded step has a serial floor
  (dense full-table gradients per slice merged and applied every step, autograd
  bookkeeping, rayon contention inside the slices). Two ways: a closed-form sparse
  gradient for the plain two-level softmax (no autograd; accumulate into the touched
  rows only; rayon over units), as the projection engine already does for its Poisson
  objective, keeping the autograd step for tracks / offsets / LoRA; or fewer draws per
  unit (`--modules-per-unit 4` halves the step at the same loss). Then phase 2.

* [ ] **peak-level rows by refinement, after gene and cell rows are settled.** With the
  unit rows (pseudobulk or cell) and gene rows frozen, a peak's row is a convex Poisson
  regression of its counts onto the unit table, warm-started at its module row: the
  per-axis projection engine with roles swapped (peaks as the cells, units as the
  dictionary, unit depth `ln N_u` as the bias). A first cut ran but did not move off
  the warm start on a planted test; parked until the inputs are settled.

* [ ] **links read a module network, not pairs.** With module-level peaks the network is
  `W[m_p, m_g]` between peak modules and gene modules (cosine of the module rows in the
  shared unit space; per cluster, `μ_{m_g}ᵀ Σ_k μ_{m_p}`), and a peak–gene score is a
  lookup of `W[m(p), m(g)]` inside the cis window with distance as the prior. Pair-level
  scoring at cell resolution is too slow to keep. Open: tie a linked peak's module to its
  gene module (one shared row) once `cos(μ_peak_m, μ_gene_m)` on the seeded pairs says so.

* [x] 2026-09-22: senna bge multiome re-verified after the shared trainer changes
  (module-only partitions, extra-axis residual made optional, warm-started per-axis
  projection): the sim run completes and clusters the cells correctly; senna's tests pass.

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

* [x] 2026-09-22, first real run on a public 10x PBMC multiome set (about 12k cells,
  37k genes, 144k peaks; two backends from one 10x h5 via `data-beans from-10x-matrix
  --select-row-type`): the cell embedding separates the expected lineages by marker
  genes. The per-cell projection is most of the wall time at real peak counts (blocks
  are sized by the activation budget, so a block is a few hundred cells at 180k
  features); the hier fit and the refine are minor.

* [ ] **write per-cell cluster labels.** `clusters.parquet` lists ids only; nothing maps a
  barcode to its cluster, so cluster → cell type has to be redone outside. Write
  `{out}.cell_clusters.parquet` (cell, cluster) next to the cell embedding.

* [ ] **link stage barely favors promoter-proximal peaks on real data.** The top-scoring
  peak per gene and cluster lies within 5 kb of the TSS only slightly more often than a
  random candidate does, and its median distance is ~200 kb. Same conclusion as the sim:
  the Pearson link is the weak stage; see the link-fine-tuning item above.

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
