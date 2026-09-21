# Chickpea graph-embedding-util peak-to-gene Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire chickpea `peak-to-gene` to train peak/gene embeddings with `graph-embedding-util`, refine links within cell clusters, and write E2G-like parquet.

**Architecture:** Chickpea owns data loading, rough ABC/co-occurrence edges, cell clustering, within-cluster refine, and parquet I/O. All embedding training goes through `graph-embedding-util` (prefer FNE / simba-style typed graphs with `region` + `gene` node types — do not reimplement NCE/PBG). Interim CLI stub on branch `ypp/chickpea-ditch-rsvd-knockoff` is replaced incrementally.

**Tech Stack:** Rust, `graph-embedding-util`, `data-beans`, `genomic-data`, parquet (Arrow), clap.

**Spec:** [`chickpea/todo.md`](../../../chickpea/todo.md) (practical algorithm + E2G working example under `~/work/writing/paper-chickpea/data/e2g/`; ATAC fixture `10k_pbmc_ATACv2-qc.zarr.zip`).

## Global Constraints

- No rSVD / SuSiE / GhostKnockoff / LOCO-TMLE association path.
- Do not reimplement graph embedding in chickpea — call `graph-embedding-util`.
- Prefer pb-level training where ge-util supports it.
- Output shape mirrors E2G: `peaks.parquet`, `clusters.parquet`, `peak_gene/chr*.parquet`.
- Within-cluster refine uses hard clusters + pb-per-cluster; min-cell gate; prefer joint RNA+ATAC or RNA-led clusters.
- Clause-boundary line breaks for clap `long_about` / `long_help`.
- `cargo fmt` + `cargo clippy -p chickpea -- -D warnings` before every commit.

---

## File map

| File | Responsibility |
|------|----------------|
| [`chickpea/Cargo.toml`](../../../chickpea/Cargo.toml) | add `graph-embedding-util` (+ parquet when writing) |
| [`chickpea/src/p2g/abc_map.rs`](../../../chickpea/src/p2g/abc_map.rs) | rough pb co-occurrence / cis ABC-style peak–gene edges |
| [`chickpea/src/p2g/embed_ge.rs`](../../../chickpea/src/p2g/embed_ge.rs) | build typed graph + call ge-util train; return peak/gene (and cell) embeddings |
| [`chickpea/src/p2g/cluster.rs`](../../../chickpea/src/p2g/cluster.rs) | cell embed → clusters (min-cell gate) |
| [`chickpea/src/p2g/refine.rs`](../../../chickpea/src/p2g/refine.rs) | within-cluster peak→gene scores |
| [`chickpea/src/p2g/parquet_out.rs`](../../../chickpea/src/p2g/parquet_out.rs) | E2G-like three-table writer |
| [`chickpea/src/p2g/run.rs`](../../../chickpea/src/p2g/run.rs) | orchestrate stages; restore real CLI |
| [`chickpea/src/p2g/input.rs`](../../../chickpea/src/p2g/input.rs) | existing paired load (already scaffolding) |

---

### Task 1: Depend on graph-embedding-util + stage modules

**Files:**
- Modify: `chickpea/Cargo.toml`
- Modify: `chickpea/src/p2g/mod.rs`
- Create: empty/stub `abc_map.rs`, `embed_ge.rs`, `cluster.rs`, `refine.rs`, `parquet_out.rs` with module docs only
- Modify: `chickpea/src/p2g/run.rs` — keep bail, but list planned stages in the error / help

- [ ] Add `graph-embedding-util = { workspace = true }` (and cuda/metal feature passthrough mirroring senna/pinto if needed)
- [ ] Declare `mod` stubs
- [ ] `cargo clippy -p chickpea -- -D warnings`
- [ ] Commit: `chore(chickpea): depend on graph-embedding-util; stage p2g modules`

### Task 2: Rough ABC / co-occurrence edge builder

**Files:**
- Create/implement: `chickpea/src/p2g/abc_map.rs`
- Test: unit test on tiny synthetic pb matrices (known linked peak–gene ranks high)

- [ ] Write failing test: cis window + pb correlation / co-occurrence yields edge list `(peak_idx, gene_idx, weight)`
- [ ] Implement pb-level map (multiome: RNA gene × ATAC peak; ATAC-only: gene-activity proxy TBD — start multiome-only if simpler)
- [ ] Clippy + test
- [ ] Commit: `feat(chickpea): rough pb peak–gene co-occurrence map`

### Task 3: Train peak/gene embeddings via ge-util

**Files:**
- Implement: `chickpea/src/p2g/embed_ge.rs`
- Mirror patterns from `senna/src/fne/` / `senna/src/simba/` (typed `region`+`gene` graph → `graph_embedding_util::fne::train` or simba recipe)

- [ ] Write failing test or smoke: graph from Task 2 edges → finite embedding rows for peaks and genes
- [ ] Implement thin wrapper (no local NCE loop)
- [ ] Clippy + test
- [ ] Commit: `feat(chickpea): train peak/gene embeds via graph-embedding-util`

### Task 4: Cell embed → cluster → within-cluster refine

**Files:**
- Implement: `cluster.rs`, `refine.rs`
- Use ge-util cell projection / `postprocess::cell_clusters` where available

- [ ] Cluster cells; drop clusters below min-cell
- [ ] Per-cluster pb refine of peak–gene scores from embeddings + local co-occurrence
- [ ] Clippy + test
- [ ] Commit: `feat(chickpea): within-cluster peak–gene refine`

### Task 5: E2G-like parquet output + wire `run`

**Files:**
- Implement: `parquet_out.rs`
- Modify: `run.rs`, `main.rs` help
- Reference schemas: `~/work/writing/paper-chickpea/data/e2g/`

- [ ] Write `peaks.parquet`, `clusters.parquet`, `peak_gene/chr*.parquet`
- [ ] `run_peak_to_gene` runs full pipeline end-to-end on sim or tiny fixture
- [ ] Clippy + test
- [ ] Commit: `feat(chickpea): E2G-like parquet peak-to-gene output`

### Task 6: Dry-run notes on 10k PBMC ATAC QC

- [ ] Document in `todo.md` or README how to point at `10k_pbmc_ATACv2-qc.zarr.zip` (ATAC-only / gene-activity path may still be partial)
- [ ] Commit only if docs change

---

## Done when

- `chickpea peak-to-gene` trains with ge-util (not rSVD), refines within clusters, writes E2G-shaped parquet
- fmt/clippy/tests green on chickpea
