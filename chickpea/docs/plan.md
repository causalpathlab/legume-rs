# chickpea p2g: rebuild on main's embedding engine

Branch `ypp/chickpea-afresh`, cut from `origin/main` at `aa308a11` (after #75, #76, and the
faba extraction in #77).

## Why a fresh start

- #74 (`ypp/chickpea-ditch-rsvd-knockoff`, tip `0374ecd5`) and #75 both branched
  from #73 and diverged inside `graph-embedding-util`: #75 replaced module-warm with
  `module_partition` + `RowCollapse` and removed the block-SGD track path, while #74
  built a two-axis hier engine (separate gene and peak axes, `train_partitions`,
  `module_warm`, module re-collapse) plus a per-axis cold cell projection.
- None of #74's 31 commits are on main. Merging gave 10 conflicting files across two
  designs of the same code, so #74 was closed (branch kept) and chickpea is rebuilt here.
- Rule for this branch: `0374ecd5` is **reference code only**. Nothing is merged from it;
  a piece is ported only when a step below needs it, against current main, with its tests.

## What carries forward from #74 (evidence, not code)

- The hier embedding ranks causal peaks well on the sim: gene-peak row cosine AUROC
  about 0.85 (10 epochs) to 0.89 (100), versus 0.64 to 0.70 for Pearson links.
  With module-only peaks (no per-peak row) it drops to about 0.75.
- The link stage is the weak stage. On PBMC 10k the top-1 Pearson peak lies within
  5 kb of the TSS for only 4.2% of genes (2.4% for all candidates).
- Phase 2 (cell projection) dominated runtime on real peak counts: 12.5 of 16 min.
- Measured and rejected: a distilled per-axis encoder for phase 2 (worse ARI, slower);
  bge feature affinity, fne contrast, and ABC-triplet NCE as link evidence.
- Missing output found on PBMC: per-cell cluster labels.

## Steps

Each step ends with a check before the next starts.

### 1. Gene-centric two-track embedding

Peaks are cis-regulatory evidence for genes, so ATAC enters as a gene feature.
Each gene gets an ATAC version, `a_gc = Σ_p w_gp x_pc` over its candidate cis
peaks, and the gene axis carries two tracks of the same genes:

- track 0: RNA gene counts (as in bge);
- track 1: ATAC aggregated onto genes by a fixed peak-to-gene weight matrix `W`.

Both tracks are scored against the same pseudobulk embeddings `e_u`, and
`TrackSpec.gene_of_row` ties a gene's two rows to one base row: track 1 is the base
row plus a rank-r offset under a ridge (`offset_rank`, `offset_l2`), which sets how
closely ATAC-gene must agree with RNA-gene. Each track has its own softmax support,
so RNA and ATAC keep separate normalisers. The axis is about 2 × G gene rows
instead of G + 144k features.

`W` (sparse gene × peak), selected by `--gene-score`:

- `abc` (first): contact-only kernel on peak-midpoint distance to the TSS, a power
  law `(d + pseudocount)^-γ`, normalised over each gene's candidate peaks. Activity
  is the per-cell accessibility `x_pc` itself, so it is not multiplied in again.
  Needs only the TSS: works with today's loaders and the sim's `gene_coords.tsv.gz`.
- `archr` (next): `exp(-d/5000) + exp(-1)` with `d` the distance to the gene body
  extended 5 kb upstream, 100 kb window, clipped at neighbouring genes, gene-size
  factor capped at 5 (#74's `gene_activity.rs` as reference). Each gene sums its own
  peaks, so a peak near two genes counts for both. Needs gene start/end: port the
  `GeneLoc` fields and release `legume-genomic-types` first.

Gene positions are a required input (`--gff-file`, or `--gene-coords` for the sim).
Log how many RNA genes found a position and how many peaks reached no gene.

`W` is fixed here; learning a sparse `W` (the links themselves) is item C.

Check: sim cell ARI; per-gene agreement of the two tracks' rows; the offset's share
of the row norm; wall time. The sim has TSS only and is imperfect, so treat it as a
smoke test and judge on PBMC.

### 2. Remove the old stages

Drop rSVD, knockoff, TMLE, and the SuSiE cascade (#74 removed them too). Keep input
loading and cell QC. Update `README.md` and `todo.md` to the new pipeline.

Check: `cargo test -p chickpea`, clippy clean.

### 3. Loci and candidate pairs

Folded into step 1: the cis candidates and gene positions are what build `W`.
Check here only that candidate counts and the distance distribution look sane.

### 4. Links: localized attention

Each gene attends over its own cis peaks. Nothing here is dense or per-peak: all
work runs over the precomputed cis-pair arrays (about 36k genes × at most 200
peaks, the same sparsity as `W`), one batched gather per step.

1. Peak rows, once. With the pseudobulk embeddings `E` [U × H] frozen, fold every
   peak in by one weighted least-squares step,
   `Φ = Z Ω E (Eᵀ Ω E + λI)⁻¹` (`Z` the sparse log-scale pb accessibility [P × U],
   `Ω` shared unit weights): one sparse × dense product plus one H × H solve, so
   1M peaks cost about the same as a single pass over the ATAC pseudobulks. It is
   the first IRLS step of the per-peak Poisson GLM; check it against the exact GLM
   on a few thousand peaks and add a second step if needed. One `φ_p` per peak,
   shared by every gene it is a candidate for.
2. Attention weights, a localized membership per gene (shares over its cis peaks):

       π_gp = softmax_{p ∈ cis(g)} [ log k_θ(d_gp) + ρ_gᵀ M φ_p ]

   `k_θ` is the ABC (later ArchR) distance kernel with its few constants learned
   globally; `M` is low-rank (r × H). No per-pair parameters: only `θ` and `M` train,
   with `ρ` and `φ` frozen.
3. Loss: the ATAC-derived gene row must agree with the RNA row, which is itself
   tied to the pseudobulk embeddings:

       L = Σ_g ‖ ρ_g − Σ_{p ∈ cis(g)} π_gp φ_p ‖²   (or a cosine version)

   Per epoch about nnz × H flops (7M × 128). Mixing rows is a log-scale stand-in for
   mixing peak rates; check it against the rate-space version on one chromosome.
4. Outputs: `π_gp` as the link score (the fraction of gene g's input from peak p);
   per cluster `k`, `c_gpk = π_gp λ_pk / Σ_q π_gq λ_qk`, the E2G-like tables
   (`peaks.parquet`, `clusters.parquet`, `peak_gene/chr*.parquet`). Fixed ABC per
   cluster is written alongside as the baseline to beat.
5. Optional: rebuild the ATAC-gene track as `π · A_peak` and refit the embedding
   once, if `π` moves well away from `W`.

Known limit: co-accessible bystander peaks have similar `φ`, so the content term
cannot separate them; only distance separates them here (later: within-cluster
fitting, single-effect sparsity).

Check: link AUROC and top-1 / top-3 against fixed ABC (sim as a smoke test);
promoter enrichment of top links and wall time on PBMC.

### 5. Outputs

Write per-cell cluster labels `(cell, cluster)` next to the cluster table.

### 6. Real data

Rerun PBMC 10k (public 10x multiome). The GENCODE GFF used before lived under the
removed `faba/` tree; download it again first. Compare with the #74 run: marker-coherent
clusters, wall time and peak RSS, promoter enrichment of top-1 links.

## Test-first order

Tests live in `chickpea/tests/`; chickpea is split into lib + bin first so they can
reach the modules. Each test is written before its code and fails first. Fixtures
use neutral names (`GENE1`, `chr1`, `CT1`).

0. lib + bin split (`src/lib.rs`, thin `main.rs`).
1. `cis_pairs.rs`: window membership; ABC kernel decreases with distance; per-gene
   weights sum to 1; a peak near two genes appears in both; peaks with no gene counted.
2. `atac_gene_track.rs`: `W · A_peak` on a tiny matrix equals hand-computed counts;
   cells unchanged; rows named `GENE/atac`; genes without cis peaks get no ATAC row.
3. `tracks.rs`: `TrackSpec` rows (`track_of_row`, `gene_of_row`); an RNA-only gene
   sits on track 0 only.
4. `embed_two_track.rs`: planted two-program fixture as tiny zarrs; after a short
   fit the programs separate and a gene's RNA and ATAC rows are closer than random pairs.
5. `peak_foldin.rs`: WLS fold-in recovers planted `φ`; agrees with the exact
   per-peak Poisson GLM on a small case.
6. `attention.rs`: shares sum to 1 over each gene's cis peaks; pooling is the weighted
   mean; autograd matches finite differences for `θ` and `M`; a planted gene built
   from one peak concentrates `π` on it; an identical-`φ` bystander farther away
   loses on distance.
7. `context_links.rs`: `c_gpk` sums to 1 per gene per cluster; fixed ABC per cluster
   matches a hand-computed toy.

After each group: `cargo test -p chickpea`, clippy, commit.

## Verification after touching `graph-embedding-util`

Re-run `senna bge --multiome` on the sim (ARI 1.0 at pve 0.8) and the
`graph-embedding-util` and `senna` tests, so bge stays as #75 left it.

## Next, each with its own plan when its turn comes

These follow step 6. Write a separate plan for each before coding.

### A. Context-aware link score

With frozen rows `ρ_g`, `ρ_p` and `Σ_k` the covariance of the cell embeddings in
cluster `k`: `cov_k(g, p) = ρ_gᵀ Σ_k ρ_p`, normalised to a correlation. One H×H
matrix per cluster plus a dot product per cis pair. Prototype in R on the sim first.

### B. Feature network as a prior on rows

An external network (cis distance, contact, motif to TF; never co-accessibility from
the same data, which leaks into the score) as a smoothness prior: peak row = module
row + a residual passed along the network from linked genes (SGC-style), or a
Laplacian penalty. The network says what can interact; the cells (through `Σ_k`)
say when. LoRA (`PresetMode::Lora`) is not the tool here: its shared low-rank
residual cannot express a link between one peak and one gene.
First test: a cis-distance kernel only; target is to recover the 0.75 to 0.85 gap.

### C. Sharper links

Step 4's localized attention is the learned `W`. Next levels, only if step 4 shows
the need: within-cluster fitting against shared cell-state co-variation, and a
single-effect (SuSiE-style) posterior over each gene's attention shares.

### Also open

- Phase 1 fast and accurate (about 1000 epochs at this size).
- Cells as phase-1 units (16 per pseudobulk, as bge).
- Phase 2 speed (the PBMC cost lever).
- Module-level link network `W[m_p, m_g]`; pair-level only as a cis lookup.
