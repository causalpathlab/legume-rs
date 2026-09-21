# Chickpea

Chromatin Interactions Captured by Knitting Peaks with Expression Anchors


## TODO

* [x] download ABC / E2G reference data (Engreitz lab schema + parquet)

* [x] study ABC model.. what is the generative model? it's just activity co-occurrence

* [x] practical algorithm (no rSVD; no GhostKnockoff / LOCO-TMLE / SuSiE surgery):
    1. link peaks to genes on pb profiles: `--link-score pearson` (log1p correlation, default)
       or `abc` (Engreitz activity × contact with the 1 Mb pseudocount, ATAC only);
       `--top-k-per-gene` cuts each gene's ranked list (ATAC-only: ArchR gene activity is
       the RNA stand-in; circular under pearson, projection-only under abc)
    2. train peak + gene + pseudobulk embeddings jointly with **`graph-embedding-util`** FNE:
       the link relation plus SIMBA-binned pb × feature relations for every level of the
       pb tree (`--num-levels`, `--context-bins`) and the tree's parent edges; this is what
       brings cellular context into the peak/gene geometry (2026-09-21)
    3. embed cells in that space → group into clusters (min-cell gate; prefer joint RNA+ATAC or RNA-led clusters) → **refine peak→gene within each cluster** (pb-per-cluster)
    4. emit **E2G-like parquet**:
       - `peaks.parquet`: id, chromosome, start, end, class  (← enhancers)
       - `clusters.parquet`: id, name  (← cell_types)
       - `peak_gene/chr*.parquet`: id, score, target_gene_id, target_gene_name, target_gene_tss, enhancer_gene_distance, model, chromosome, enhancer_id, cell_type_id (= cluster)  (← enhancer_gene_predictions)

* [x] 2026-09-21, measured on the simulator and dropped (not in the tree):
  a bge feature-embedding affinity as the link evidence, an fne contrast with
  trans-permuted negatives, and an ABC-triplet logistic NCE against chance ABC.
  None reached the log1p Pearson; the sim's causal signal is cross-pseudobulk
  co-variation of RNA with the causal peaks, which Pearson measures directly and
  ABC-shaped targets exclude by construction. What the sim does NOT contain is a
  bystander that co-varies through shared cell state — the case Pearson is wrong
  about; a simulator knob for that is the prerequisite for any further evidence term.

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
