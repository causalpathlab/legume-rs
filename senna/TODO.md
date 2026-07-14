# TODO — `bge` undertrains the gene dictionary ρ

**Status:** diagnosed, not fixed. One hypothesis, with a concrete A/B that was never read out.

## The symptom

`senna bge` on HCA BM (BoneMarrowMap, 8 donors, 96,791 cells, 34,008 genes):

| | live ρ rows (`feature_qc`) |
|---|---|
| HCA BM, 200 epochs | **4.0%** |
| HCA BM, 1500 epochs | **0.2%** |
| 10k BMMNC, 200 epochs (reference) | **31.4%** |

The dead rows sit at `‖ρ_g‖² ≈ 2.81`. For `H=32` that **is** the random-init norm — they received
essentially no gradient and never moved. It is not slow convergence; it is no convergence.

**The cell side is fine.** kNN label accuracy on the cell embedding is **0.729** against 24-class
ground-truth labels (majority-class baseline 0.203); `senna topic` gets 0.757 on the same cells. So
the data trains — only the *gene* side collapses.

The genes that do survive are the **top 2% by abundance** (70× more abundant than the dead ones;
Spearman(‖ρ‖², gene abundance) = **+0.68**).

q## The hypothesis: one un-dampened sampling axis

`bge` draws sublinearly on every axis **except** the one that decides whether a *gene* gets gradient:

| axis | draw | exponent | where |
|---|---|---|---|
| pseudobulk | `pb_size^α` | 0.5 | `DEFAULT_STRATIFY_ALPHA_PB` |
| cell | `degree^α` | 0.5 | `DEFAULT_STRATIFY_ALPHA_CELL` |
| negatives | `count^α` | 0.75 | `alpha_neg`, `fit/mod.rs` |
| **feature, within a pb/cell** | `count · fisher(f)` | **1.0 — LINEAR** | `loss/feat.rs` |

`fit/config.rs` already *states the argument* for the other three — *"sublinear, mirrors the
`count^0.75` we use for negatives; gives rare cell types meaningful coverage without starving the
dominant strata"* — but it was never applied one level down, to the gene.

**Why HCA and not BMMNC.** HCA is shallow (988 genes/cell vs BMMNC's 2,191), so each cell's count
mass piles onto a few highly-expressed genes:

| | top 5% of genes take… | median positive draws: dead vs live gene |
|---|---|---|
| BMMNC | 43.9% of the budget | 23 vs 25 — **1×** (budget does not predict liveness) |
| HCA | **79.6%** | 40 vs 2,766 — **70×** (budget entirely predicts liveness) |

So it is not absolute starvation — HCA genes actually get *more* draws than BMMNC's. It is
**concentration**: the head absorbs ~80% of the positive draws and the tail never leaves init.

## ⚠️ The landmine — read this before touching the sampler

**`bge`'s cell axis is OFF by default.** `--phase1-cells-per-pb 0` means *"pure-pb: suppress the cell
axis entirely; `E_feat` shaped by pb aggregates only."*

So ρ is trained from the **pseudobulk** axis:

- ✅ `build_stratified_sampler` (`loss/feat.rs`) ← **this is the live path**
- ❌ `build_active_samplers` (`fit/samplers.rs`) ← cell axis, **does not run by default**

I patched the cell axis first. The A/B came back **4.3% → 4.4% — no change**, because that code path
never executes. Don't repeat this.

## The proposed fix (unvalidated)

Add an exponent to the positive feature draw, mirroring the other three axes:

```
q(f | pb) ∝ count^alpha_feat · fisher(f)        # alpha_feat = 0.5; 1.0 = today's linear draw
```

Sites to change (all three build the positive picker from `count · fisher`):

- `loss/feat.rs` — `build_stratified_sampler` (pb axis; **the one that matters**)
- `loss/feat.rs` — `gene_paired_entries` (faba's spliced/unspliced β-sharing)
- `fit/samplers.rs` — `build_active_samplers` (cell axis; only when `--phase1-cells-per-pb > 0`)

Thread it as `FitConfig::alpha_feat` + a `senna bge --alpha-feat` knob so it can be A/B'd, and keep
`1.0` as the escape hatch.

**Predicted effect** (from the count distribution, not yet measured end-to-end): at `alpha_feat=0.5`
the dead genes' draw budget rises ~7× (40 → 290) and the top-5% share falls 79.6% → 27.8%, while
BMMNC barely moves (it was never starved).

## The experiment that decides it

```bash
D=~/work/paper-senna/data/BoneMarrowMap
for A in 1.0 0.5; do
  senna bge $D/HCA_BM_BM{1..8}.zarr.zip -o out_$A \
      --device cuda --device-no 0 -i 200 --embedding-dim 32 --n-hvg 3000 \
      --alpha-feat $A --must-train-features panel.txt
done
# read out: feature_qc.parquet -> fraction live
```

**If the live fraction does not move at `alpha_feat=0.5`, the hypothesis is wrong.** Say so; don't
salvage it. In that case the next suspects, in order:

1. **Learning rate / gradient scale.** The surviving genes reach `‖ρ‖² ≈ 90` vs an init of 2.8 —
   a handful of genes *exploded* while the rest sat still. That pattern (bimodal, head runs away)
   is as consistent with an optimizer/LR problem as with a sampling one.
2. **`--n-hvg` and batching are already ruled out.** 3000 vs 5000 → 83 vs 78 live. With vs without
   batch files → 103 vs 83. Neither matters.
3. **More epochs makes it WORSE** (0.2% @1500 vs 4.0% @200), which is not a convergence story. Check
   whether that is real collapse or an artifact of `chi2_null_call` estimating σ̂² from the bulk —
   if more genes drift *slightly*, σ̂² rises and fewer clear the bar, which would shrink the "live"
   count without anything actually getting worse.

## Why it matters downstream

`annotate-by-projection` reads `outputs.feature_embedding`, which is the SIMBA co-embed *derived
from* ρ. A dead ρ row still gets a plausible unit-norm coordinate — parked at the **hub** of the cell
cloud, where it is close to every cell at once. On HCA, **86%** of the co-embed is hub-parked, and
the projection annotation scores **59.1%** against ground truth versus **87.5%** for the raw-count
enrichment path on the same cells.

`hub_call` + `--min-markers` (already landed) *detect* this and refuse rather than ship a confident
magnet — but the cure is upstream, here.

## Notes

- `feature_qc.parquet` is computed on `out.model.e_feat` — the **raw ρ**, at pass 1, before the
  co-embed. It tells you about ρ, **not** about `feature_embedding`.
- Data: `~/work/paper-senna/data/BoneMarrowMap/HCA_BM_BM{1..8}.zarr.zip` + `.cell_metadata.tsv.gz`
  (ground-truth `CellType_Broad`). Raw integer counts, verified.
- `git log` shows **no recent change** to the sampler — this is latent, not a regression.
