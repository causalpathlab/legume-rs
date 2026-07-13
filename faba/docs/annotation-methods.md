# Marker-based cell-type annotation in `faba` — methods

Source material for a paper Methods section. Every default quoted here is the shipped default;
every number attributed to "measured" was obtained on the cord-blood dataset described in §7.
Code: `graph-embedding-util/src/type_annotation/{term_ora,marker_bootstrap,panel_null}.rs`,
driven by `faba annotate` and `faba lineage --markers`.

---

## 1. Inputs and the space the call is made in

`faba gem` embeds cells and gene features jointly. Two of its outputs matter here:

| output | rows | meaning |
|---|---|---|
| `{out}.cell_embedding.parquet` | cells × H | the latent cell coordinate **θ_c** |
| `{out}.feature_embedding.parquet` | feature rows × H | the co-embedded gene vector **e_g** |

Feature rows are keyed `{gene}/count/{spliced,unspliced}`; annotation selects one modality and
re-keys by gene (`spliced` for the mature-identity track, `unspliced` for the velocity track).

**Annotation does not use the `β_g` dictionary**, and this is load-bearing. A Euclidean
nearest-centroid call is only meaningful if genes and cells inhabit one metric space. gem couples
β and θ through an *inner product*, which fixes their relative directions but not their relative
scale — and the fitted model exploits that freedom. Measured: β rows have median norm **0.00**
(most genes are untrained and their post-hoc projection fails its null test) and mean norm 0.21,
against cells at 30.1. At that ratio

    ‖θ − c‖² = ‖θ‖² − 2⟨θ, c⟩ + ‖c‖²

loses the ‖c‖² term, and ‖θ‖² is constant across candidate types, so `argmin` collapses to
`argmax ⟨θ, c⟩` — an unnormalized inner product in which a centroid's **norm** decides the winner
largely irrespective of its direction. Measured on an untrained panel, the rank correlation
between a type's centroid norm and the share of cells it captured was **+0.93**.

> **Prerequisite.** `faba gem` must be run with `--must-train-features <panel>` using *the same
> marker file* the annotation will use. Otherwise the panel's genes are not fitted, only
> projected. Measured with a mismatched panel (44% of markers trained): spurious assignment to
> types absent from the tissue ran at 59.7%; with the matching panel (97% trained) it fell to
> **18.9%**. Report `n_live / n_markers` (in `{out}.panel_null.tsv`, or the "marker liveness" log
> line) as a QC figure — if it is not near 100%, nothing downstream is interpretable.

---

## 2. Type prototypes and the per-cell call

For a marker panel assigning genes to types, each type *T* gets an **IDF-weighted centroid**

    e_T = ( Σ_{g ∈ markers(T)} w_g · e_g ) / ( Σ_{g ∈ markers(T)} w_g )

over its **live** markers only (a gene with an all-zero embedding row contributes nothing and is
excluded from *both* numerator and denominator; counting it in the denominator would shrink the
centroid toward the origin in proportion to a type's dead-marker fraction, and a short centroid is
not a weak competitor but a *magnet*). `w_g` is the inverse-document-frequency weight
down-weighting markers shared across many types (`--no-idf` disables).

Cells are assigned by nearest centroid, `t(c) = argmin_T ‖θ_c − e_T‖₂`. Zero-norm centroids are
excluded from the competition: they sit at constant distance `‖θ_c‖` from every cell and would
otherwise capture every cell nearer the origin than to any real prototype.

**QC prune.** Within each type, cells whose distance to their assigned centroid is a high-side
robust outlier (`> median + k·MAD`, `k = --assign-mad`, default **2.5**) are set to `unassigned`:
they took a type by `argmin` but do not actually sit near it (ambient RNA, doublets).

---

## 3. Over-representation within cell groups

A single cell's nearest-centroid call is close to a coin flip; pooling makes it testable. Cells
are grouped by **Leiden** community detection on their cosine kNN graph (`--knn`, default 30;
`--resolution`, default 1.0 — see §8 for why a *low* resolution is required).

For each (group *K*, type *T*) the count `a = #{c ∈ K : t(c) = T}` is tested against the
hypergeometric null with margins `(N, m_T, n_K)`, where **N and m_T are counted over the cells**
(each cell once), *not* summed from the contingency table. The statistic is `S = −ln P(X ≥ a)`.

`S` is then **calibrated against a permutation null**: the per-cell labels are shuffled with the
group memberships held fixed, and `S` is pooled across groups within a type (the statistic is
relabeling-invariant). `--num-perm` (default 500) draws; the pool is `n_perm × n_groups`, capped
at 10⁵ per type. The permutation p is Benjamini–Hochberg-adjusted across the types within each
group. A group is called by its top over-represented type if `q < --fdr-alpha` (default **0.1**),
else left uncalled; its cells inherit the call.

Calibration diagnostics (`{out}.null_calibration.tsv`): analytic-vs-permutation agreement, and a
genomic-inflation-style λ plus a Kolmogorov–Smirnov statistic on the permutation null itself. λ is
computed with **mid-p** tie-splitting, without which the discrete statistic's ties at the floor
pin λ at a constant 108.77 — an artifact of the numerical clamp, not inflation.

---

## 4. Stability bootstrap (default ON)

`argmin` always returns something, and returns it with no error bar. Two error sources are
therefore treated as zero: the **marker list can be wrong** (a listed gene is not specific to its
type in this dataset), and the **embedding can disagree with it** (a type's markers do not sit
together, so its centroid is a fiction).

Rather than posit a generative density, we resample the evidence. For each of `B` replicates
(`--n-boot`, default **200**):

1. **Resample the panel.** For each type, draw `|live(T)|` of its live markers *with replacement*
   and rebuild `e_T` from that multiset (Efron's bootstrap over the marker panel).
2. **Re-derive the grouping.** Re-run Leiden under a fresh seed (`faba annotate`), or refit the
   k-means MST nodes under a fresh seed (`faba lineage`).
3. Re-run §2–3 end to end and record each cell's resulting label.

The shipped label is the consensus; `label_support` is the fraction of replicates that agreed.

Three points of method:

- **The bootstrap perturbs exactly the decision variable**, so the jitter it induces is
  automatically on the scale of the decision. Nothing needs calibrating against the
  (incommensurable) spread of the cells.
- **The grouping must be resampled too, or the bootstrap has no teeth.** Holding the partition
  fixed and resampling only the panel measures almost nothing: a 2,000-cell cluster's `argmax`
  does not flip because a few markers were redrawn. Measured with the partition held fixed: **0%**
  of cells abstained, and support's ability to separate spurious calls fell from **AUC 0.93 to
  0.69**. The partition is where the instability lives. (This is not an optional stochasticity:
  the kNN graph is built by `hnsw_rs`, which seeds itself from OS entropy with no API to set it,
  and Leiden amplifies a 0.3% edge difference — four identical runs of the same binary on the same
  15,315 cells produced **990 / 132 / 137 / 138** communities.)
- **The kNN graph is built once and only Leiden is reseeded.** The cell embedding is identical on
  every replicate — the bootstrap resamples the *panel*, not the cells — so rebuilding the graph
  would only re-randomise an approximate-nearest-neighbour index's internal seed, which is
  implementation noise, not uncertainty about the data. Measured: 135 s → 4 s, with support
  correlation 0.96 and AUC 0.931 → 0.943.

**`label_support` is not a posterior.** It is the sampling variability of the pipeline's own
output: it sees **variance, not bias**. A systematically wrong call that every replicate agrees on
returns support 1.0. That is what §5 is for.

A type with fewer than 2 live markers is excluded from the competition entirely: a one-element
panel resampled with replacement always returns itself, so its centroid never moves and it appears
*perfectly* stable — the opposite of the truth.

---

## 5. Set-valued ("mixed") annotation

Collapsing the replicate distribution to `argmax` + a threshold discards the information the
bootstrap exists to produce. We instead report, for every cell, the **smallest set of types whose
replicate shares sum to `--set-coverage`** (default **0.8**), rendered in canonical order and
capped at `--max-set-size` (default **3**):

    label_set = HSPC/LMPP        label_set_size = 2        label_set_support = 0.87

The set is taken over *types*, while the shares remain shares of *all* replicates — so any mass
the replicates spent on "no call" makes the coverage harder to reach, and a cell its replicates
mostly declined to call falls out as uncalled rather than being given the nonsense set
`Erythroid/unassigned`.

The cap is a test, not a truncation: a cell needing a 4th type to reach coverage is reported
`unassigned`, because past that point a set stops narrowing anything down and would launder "we
don't know" as a finding.

Measured (7 compartments, 200 replicates, matched panel): **38.0% single, 33.4% two-way, 28.3%
three-way, 0.3% no call**, mean `label_set_support` 0.93. Under a single-label rule the two- and
three-way cells would all have been reported as "unassigned".

**Abstention rule for the single-label column.** Default `--min-support` (0.5): the top type must
win at least that share. We note that this bar is *not* scale-free — with `C` types, chance
agreement is `1/C`, so 0.5 sits at 3× chance for a 6-type panel and 12× chance for a 24-type one,
and the same setting is a different test on different panels. `--abstain-separable` offers a
scale-free alternative: an exact binomial sign test of the top type against the runner-up (among
the `m` replicates choosing one of the two, `n₁ ~ Binomial(m, ½)` under equal probability). It
resolves more cells but does not improve correctness — an abstention rule decides *when to stay
silent*, never *whether a call is right*.

---

## 6. Marker-panel permutation null (bias guard, `--panel-perm`)

The bootstrap is blind to bias. The panel null asks the complementary question: *is this answer
better than one a panel that means nothing would have given?*

For each type *T* and each of `P` draws: replace **only** *T*'s panel with `|live(T)|` genes drawn
at random from the pool of *live* marker genes, keeping *T*'s IDF weight multiset, and leave every
rival type's panel real. Rebuild `e_T`, re-run the assignment, and score

    bar[c][T] = min_{S ≠ T} ‖θ_c − e_S‖²          (the rivals — real, and fixed)
    cost(T | panel) = Σ_c min( ‖θ_c − e_T(panel)‖², bar[c][T] )
    p_T = P( cost(T | random genes) ≤ cost(T | T's own genes) )

Four design points, each of which we found to be necessary:

- **One type at a time.** Randomising every panel at once collapses all `C` null centroids onto
  the marker-pool mean; they become mutually indistinguishable and the null fails for reasons
  unrelated to any particular type. Holding the rivals real keeps the competition intact.
- **Same size ⇒ the winner's curse cancels.** A type with few live markers has a high-variance
  centroid, and a noisy prototype wins cells it should not (the maximum of a noisy score is biased
  upward). The null panel is drawn at the same size and is *equally* wobbly, so the advantage
  appears on both sides and divides out. A small panel is asked only whether *these* genes beat
  *any* genes.
- **Null genes are drawn from the live pool.** A random *untrained* gene carries no signal at all,
  so a null of dead genes would be trivially beatable and every type would look significant. This
  holds "is the gene trained?" fixed and isolates "are these the right genes?".
- **The statistic is assignment cost, not cell count.** Occupancy measures whether any rival is
  nearby, not whether the panel is right: on a cleanly separated synthetic panel a random draw
  captures *as many cells as the real one* (0.337 vs 0.333, p = 0.995), because once *T*'s real
  centroid leaves the competition its cells have no near rival and anything in the neighbourhood
  sweeps them up by elimination. Cost separates them; it also has no perverse optimum (a centroid
  capturing nothing pays the maximum `Σ bar`).

On a matched panel the null discriminates: the three compartments holding cells pass
(Erythroid p = 0.005, EoBasoMast p = 0.020, Granulo-Mono p = 0.060) and the four that do not, fail.
On a *mismatched* panel it finds nothing significant — which is itself the correct verdict.

---

## 7. Validation design

**Negative control without ground truth.** The BoneMarrowMap broad panel contains 10 mature /
terminal types that cord blood does not contain (they are `#`-commented out of the shipped panel
for exactly this reason). Re-enabling them turns every cell they capture into a *countable false
positive*, needing no labels. Reported as "mature share".

**Support calibration.** Bin cells by `label_support` and measure agreement with the point-estimate
label. Measured: support ≥ 0.8 → 96.2% agreement; support < 0.6 → 54.5%. As a detector of spurious
(mature) calls, support reached **AUC 0.83** with no ground truth.

**Headline.** Cord blood, 15,315 cells, 24-type panel: mature share 28.2% (point estimate) →
**2.4%** (bootstrap). Independently, on the panel-vs-embedding axis: 59.7% (44% of the panel
trained) → **18.9%** (97% trained).

---

## 8. Negative results (do not re-derive)

**Clustering-free / per-cell-neighbourhood ORA fails.** Replacing the Leiden partition with one
kNN neighbourhood per cell (the Milo device) was **worse than the untested `argmin`** — 38.1% vs
28.2% spurious — and manufactured cells for types with 4 in the entire dataset. Over-representation
ranks types by how **surprising** a count is, not how **likely**: a discovery statistic, not a
classifier. The two rankings coincide only when the group is large (in a 700-cell cluster you need
many cells to be surprising, so most-enriched ≈ most-abundant); in a 43-cell ball it inverts. The
pooling must stay coarse, and Milo's precedent does not transfer because *its* labels
(condition, sample) are external to the embedding whereas ours are `argmin` in the same space that
defines the neighbourhoods. Full write-up and tables: [`annotation-grouping.md`](annotation-grouping.md).

**Parametric Bayesian centroids fail.** A conjugate Gibbs sampler with the centroid as a latent
(`μ_T ~ N(m_T, (ω_T²/κ_T)·I)`) passed every synthetic control and then made the real negative
control *worse* (48% mature vs 28% baseline): the free per-type variance let one component
shrink-wrap a blob and take 33% of cells at entropy 0.005. Misspecification surfaces as false
confidence, not as uncertainty.

**High Leiden resolution is a bad operating point.** Over 50 replicates:

| resolution | communities | cells labelled | mean support |
|---|---|---|---|
| **0.5** | **30–36** | 86.3% | **0.83** |
| 1.0 | 41–48 | 86.1% | 0.80 |
| 2.0 | 62–68 | 79.2% | 0.77 |
| 8.0 | 128–1713 | 73.8% | 0.74 |

---

## 9. Parameters and outputs

| flag | default | §|
|---|---|---|
| `--knn` | 30 | 3 |
| `--resolution` | 1.0 (use **0.5**) | 3, 8 |
| `--assign-mad` | 2.5 | 2 |
| `--num-perm` | 500 (pool capped at 10⁵) | 3 |
| `--fdr-alpha` | 0.1 | 3 |
| `--no-bootstrap-markers` | off ⇒ **bootstrap is ON** | 4 |
| `--n-boot` | 200 | 4 |
| `--min-support` / `--abstain-separable` | 0.5 / off | 5 |
| `--set-coverage`, `--max-set-size` | 0.8, 3 | 5 |
| `--panel-perm` | 0 (off) | 6 |
| `--seed` | 42 | — |

| output | contents |
|---|---|
| `{out}.annot.parquet` | per cell: `coarse_label`, `label_set`, `label_set_size`, `label_set_support`, `label_support`, `label_entropy`, `fine_label`, `fine_distance`, `cluster_size`, `community` |
| `{out}.label_stability.parquet` | per cell × type: the full replicate distribution |
| `{out}.marker_support.parquet` | per (gene, type): the marker's deviation from its type's centroid, and whether it is live |
| `{out}.type_qc.tsv` | per type: `n_live`, `centroid_jitter`, `decision_gap`, `noise_ratio`, `occupancy` |
| `{out}.panel_null.tsv` | per type: `n_live`, `occupancy`, `cost`, `null_cost`, `p` |
| `{out}.null_calibration.tsv` | permutation-null diagnostics (λ, KS, analytic agreement) |
| `{out}.cluster_term_{p,q,softq}.parquet` | group × type test matrices |

`faba lineage --markers` runs the same core over the trajectory's MST nodes instead of Leiden
communities (the bootstrap's regroup step reseeds the k-means), writing `{out}.lineage_annot.*`
and `{out}.trajectory_annotation.parquet` (node → role → cell type → confidence). With the
bootstrap on, that `confidence` is the mean bootstrap support of the cells that voted for the
node's label — a reproducibility — rather than a softmaxed test statistic. This matters because
`--root-type` selects the trajectory root as the highest-confidence node of a named type, so the
entire trajectory hangs off that number.
