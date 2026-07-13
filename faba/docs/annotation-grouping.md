# Why marker annotation pools cells into *coarse* clusters

Measured on cord blood (15,315 cells) with the BoneMarrowMap broad panel, 2026-07-12.
Referenced from `graph-embedding-util/src/type_annotation/term_ora.rs`.

## The constraint

Term-ORA calls a cluster by its **most over-represented** marker type. Over-representation is a
*discovery* statistic: it ranks terms by how **surprising** a count is, not by how **likely** the
term is. Those two rankings agree only when the cluster is large.

- In a 700-cell cluster you need a lot of cells of a type to be surprising, so
  most-enriched ≈ most-abundant. The distinction is invisible.
- In a 43-cell group it inverts. A type with **4 cells in the entire dataset** has an expected
  count of 0.01, so **two** of them is more "enriched" than the 30 cells of the type that
  actually fills the group.

Anything that shrinks the groups walks into this: a high `--resolution`, or replacing the
partition with per-cell neighbourhoods.

## The negative control

The panel has 24 types; **10 of them are mature/terminal and are absent from cord blood**
(`bonemarrowmap_broad_cordblood.tsv` `#`-comments them out for exactly this reason). Uncommenting
them turns every cell they win into a *countable false positive*, with no ground truth needed.

| grouping | mature share | agrees with raw `argmin` | "phantom" cells¹ | real types found² |
|---|---|---|---|---|
| raw `argmin`, no test at all | 28.2% | 100% | 88 | 6 / 14 |
| **Leiden `--resolution 0.5`** | **12.2%** | 53.8% | **0** | 5 / 14 |
| per-cell kNN neighbourhood, k=30 | **38.1%** ❌ | 77.4% | 231 | 7 / 14 |
| per-cell kNN neighbourhood, k=480 (n_k ≈ Leiden's) | 26.4% | 62.7% | — | — |
| **Leiden 0.5 + `--bootstrap-markers`** | **2.4%** ✅ | — | — | 4 / 14 |
| neighbourhood k=30 + `--bootstrap-markers` | 14.9% | — | — | 4 / 14 |

¹ cells handed to one of the 13 types that `argmin` uses for **fewer than 20 cells combined**.
² of the 14 types cord blood actually contains, how many get ≥ 50 cells.

**The clustering-free variant was worse than the untested `argmin` it was meant to filter.** It
manufactured 226 cells for types with almost none — 25 cells of `CD4_Memory_T`, a type with 4 in
the whole dataset. Leiden gives every one of those types zero. Growing the neighbourhood to
Leiden's own pooling size (k=480, n_k≈637 vs 731) does not close the gap.

Calling by **plurality** instead of by enrichment (let the vote pick the label, let the p-value
only gate whether to call at all) fixes the phantoms — 231 → 7 cells — but leaves mature at 35.9%.
The phantoms were a symptom; the mature false positives are coherent *blobs* in the embedding, and
a 161-cell ball sits entirely inside one. Smoothing cannot remove them.

## Why Milo's device does not transfer

Milo (Dann et al., *Nat. Biotechnol.* 2022) tests kNN neighbourhoods for **differential
abundance**, and it works because its labels — condition, sample — are **external to the
embedding**. Ours are `argmin` over marker centroids in the *same* space that defines the
neighbourhoods, so a per-cell neighbourhood test partly re-tests the geometry against itself.
That is the disanalogy; do not lean on the precedent.

## What actually moves the number

Not the grouping. On the same negative control:

- **The panel bootstrap** (`--bootstrap-markers`, resample the marker panel *and* re-derive the
  clustering, ship the consensus): 28.2% → **2.4%** mature, at 38.2% abstention.
- **The embedding.** 19 of the 24 types have **under half their markers alive** in this gem run —
  the `--n-hvg` filter and the two-pass null cull dropped them, and the bias is type-dependent
  (HVG rewards variance, and a rare population's markers are high-variance by construction).
  No grouping and no statistic can rescue a centroid built from genes the model never trained on.
  Fix upstream: `faba gem --must-train-features <panel>`.

Nothing tried here calls more than 7 of the 14 types cord blood actually contains. The grouping
question is second-order until that is fixed.
