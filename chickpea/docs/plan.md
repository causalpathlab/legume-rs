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

### 1. Embedding on main's engine

Replace the rSVD `p2g/embed.rs` with the `graph_embedding_util::fit` call that
`senna bge --multiome` uses: RNA and ATAC as two modalities of one feature axis,
peaks module-only. One pass gives gene rows, peak module rows, and cell embeddings.

Check: sim (`data-beans-sim multiome`, pve-cis 0.3 and 0.8) cell ARI 1.0 as on #74;
gene-peak row cosine AUROC recorded as the baseline for step 4; wall time.

### 2. Remove the old stages

Drop rSVD, knockoff, TMLE, and the SuSiE cascade (#74 removed them too). Keep input
loading and cell QC. Update `README.md` and `todo.md` to the new pipeline.

Check: `cargo test -p chickpea`, clippy clean.

### 3. Loci and candidate pairs

Keep cis-window candidate pairs from GFF gene loci. Port the `genomic-data` `GeneLoc`
start/end fields from #74 only if ArchR-style gene scores stay in the pipeline;
`genomic-data` is now published as `legume-genomic-types`, so that change needs a
new release before chickpea can be published.

Check: candidate counts and distance distribution match main's current output on the sim.

### 4. Link score, baseline

Score each candidate pair from the embedding (row cosine) and write the E2G-like
tables (`peaks.parquet`, `clusters.parquet`, `peak_gene/chr*.parquet`).

Check: sim top-1 / top-3 per ground-truth gene and AUROC against step 1's baseline.

### 5. Outputs

Write per-cell cluster labels `(cell, cluster)` next to the cluster table.

### 6. Real data

Rerun PBMC 10k (public 10x multiome). The GENCODE GFF used before lived under the
removed `faba/` tree; download it again first. Compare with the #74 run: marker-coherent
clusters, wall time and peak RSS, promoter enrichment of top-1 links.

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

### C. Gene-centric model

Per gene, its candidate cis peaks. ATAC merged onto genes becomes a second track of
the gene axis (existing `TrackSpec`; the rank-r track offset is the LoRA-like part).
Peak to gene is a sparse per-gene weight vector `w_gp`, and those weights are the
links. Shrinks the embedding from about 180k features to about 2 × 36k gene rows.
Risk: bystander peaks that share cell state. Counter with single-effect sparsity,
a distance prior, and within-cluster variation. The plain distance-weighted gene
activity score is a check, not the training target (that would be circular).
Stages: (1) fixed distance-kernel track in the existing engine, compare cell ARI and
per-gene track agreement; (2) learned sparse `w_gp`, scored on the sim ground truth.

### Also open

- Phase 1 fast and accurate (about 1000 epochs at this size).
- Cells as phase-1 units (16 per pseudobulk, as bge).
- Phase 2 speed (the PBMC cost lever).
- Module-level link network `W[m_p, m_g]`; pair-level only as a cis lookup.
