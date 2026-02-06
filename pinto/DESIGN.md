# Gene-Gene Interaction Modeling

## 1. Motivation

The current pipeline captures **cell-cell spatial interactions** through a KNN graph
over tissue coordinates. For each reciprocal cell pair `(i, j)`, we compute
per-gene features (raw expression, neighbor-adjusted delta, residual), collapse
pairs into pseudobulk samples, and fit independent per-gene Poisson-Gamma models.
SVD or a neural topic model then factorizes the gene × sample matrix into latent
topics.

The key limitation is that **genes are treated independently** within the
Poisson-Gamma framework. Co-expression structure, regulatory relationships, and
interaction programs across genes are captured only implicitly—by the downstream
SVD/topic decomposition—rather than being modeled as first-class objects.

An explicit sparse gene-gene interaction graph would:

- Expose co-regulatory programs directly in the model structure
- Allow prior biological knowledge (pathway databases, PPI networks) to
  constrain the gene-gene topology
- Produce interpretable gene-pair loadings on latent topics, making it possible
  to read off which gene-gene interactions drive each program

## 2. Current Approach Summary

### Pipeline steps

1. **Spatial KNN graph** (`SrtCellPairs::new` in `srt_cell_pairs.rs`):
   build a KD-tree over cell coordinates, find k-nearest neighbors,
   retain only reciprocal edges, and partition each pair's neighbor set
   into disjoint left-only / right-only subsets (`PairsNeighbours`).

2. **Per-pair features** (`collect_pair_stat_visitor` in `srt_collapse_pairs.rs`):
   for each cell pair `(i, j)`, read raw expression columns, impute via
   `impute_with_neighbours` using distance-weighted neighbor averages,
   and compute:
   - **raw**: direct sparse column from the expression matrix
   - **delta**: observed / imputed (deviation from neighborhood expectation)
   - **residual**: raw / (delta × sample size) (unexplained variance)

3. **Random projection & sample assignment** (`srt_random_projection.rs`):
   project log1p-normalized pair features onto a random Gaussian basis,
   then `binary_sort_columns` to assign pairs to discrete pseudobulk samples.

4. **Pseudobulk collapsing** (`SrtCollapsePairsOps::collapse_pairs`):
   accumulate raw/delta sums per gene per sample across all pairs assigned
   to each sample → dense gene × sample matrices.

5. **Poisson-Gamma optimization** (`SrtCollapsedStat::optimize`):
   fit `GammaMatrix` posteriors independently per gene for left, right,
   left_delta, right_delta, left_resid, right_resid.

6. **Dimensionality reduction**:
   - *SVD path* (`fit_srt_svd.rs`): concatenate `posterior_log_mean()` of
     the four delta/resid matrices horizontally, column-scale, apply RSVD →
     gene dictionary + pair latent codes via Nyström projection.
   - *Topic path* (`fit_srt_topic.rs`): `LogSoftmaxEncoder` +
     `TopicDecoder` trained as a VAE with Poisson-Gamma jittering on
     concatenated delta features.

### What is missing

Genes contribute independently to the latent topics. There is no mechanism for
two genes to jointly define an interaction feature that is then tracked across
cell pairs and samples. Any gene-gene structure must be recovered post hoc from
the topic loadings, rather than being estimated as part of the model.

## 3. Prior Art: Gene-Gene Interaction Modeling

### Synthetic lethality as a template

Synthetic lethality (SL) is the most mature framework for gene-gene interactions:
two genes are synthetic lethal when losing either alone is viable but losing both
is lethal. The core statistical idea transfers directly to our setting—measure
**deviation from independence**.

**Epsilon scores.** In combinatorial CRISPR screens the standard interaction
measure is the epsilon score:

```
ε_{g1,g2} = fitness_{g1,g2} - fitness_{g1} × fitness_{g2}
```

where fitness is the observed growth under knockout. ε ≈ 0 means the genes act
independently; ε < 0 is synergistic (synthetic sick/lethal); ε > 0 is
alleviating. This multiplicative-null model is the direct analog of what we want:
given two gene deltas, does their joint behavior deviate from what individual
deltas predict?

**GEMINI** (Zamanighomi et al., Genome Biology 2019) uses variational Bayes to
decompose observed log-fold changes from combinatorial screens into
sample-independent individual effects, sample-dependent individual effects, and a
**combination effect** that captures the interaction beyond additivity. The
separation of individual vs. combination effects is conceptually close to our
per-gene Poisson-Gamma (individual) plus gene-pair interaction (combination).

**Multi-view graph autoencoders.** SLMGAE (Hao et al., J-BHI 2021) uses GCN
encoders on multiple graph views—SL graph as the main view, PPI and Gene Ontology
as support views—with attentive merging of reconstructed graphs. The multi-view
idea suggests we can combine a data-driven gene graph with external prior graphs
(BioGRID) rather than choosing one or the other.

### Epistasis and interaction features

The epistasis literature offers several ways to parameterize gene-gene
interaction features:

| Method | Formula | Interpretation |
|---|---|---|
| Multiplicative deviation | `obs - E[g1] × E[g2]` | Epsilon-style; deviation from independence |
| Hadamard product | `x_{g1} ⊙ x_{g2}` | Element-wise product; used in tensor genetic models |
| Mutual information | `MI(g1; g2) = KL(P_{g1,g2} ‖ P_{g1} P_{g2})` | Non-parametric; captures non-linear dependence |
| Multifactor dim. reduction | model-free combinatorial binning | Good for discrete genotypes; less natural for continuous expression |

For continuous expression data in a Poisson-Gamma framework, the **product of
deltas** (Hadamard-style) is the most natural choice: it stays non-negative when
deltas are non-negative ratios, sums over cell pairs still approximate Poisson
counts, and it directly measures co-deviation from neighborhood expectation.

### BioGRID as external prior

BioGRID (thebiogrid.org) is an open-access database of curated protein, genetic,
and chemical interactions:

- **Scale**: ~2.9M protein and genetic interactions across species
- **Genetic interactions**: annotated with epsilon scores where available
  (ε < -0.08, p < 0.05 for significant SL)
- **Physical interactions**: experimentally validated protein-protein interactions
- **Access**: REST API, tab-delimited bulk downloads, PSI-MITAB format

For our purposes, BioGRID provides two things: (1) a prior edge set for the
gene-gene graph—known interacting gene pairs get edges regardless of expression
correlation; (2) epsilon scores as ground-truth validation for discovered
interaction features.

### Graphical lasso with priors

The weighted graphical lasso (wGLASSO) offers a principled way to combine
data-driven and prior-knowledge edges:

```
minimize  -log det(Θ) + tr(SΘ) + λ Σ_{g1,g2} w_{g1,g2} |θ_{g1,g2}|
```

where S is the sample covariance of gene × sample posterior means, and
w_{g1,g2} is set lower for gene pairs with BioGRID evidence (encouraging their
inclusion) and higher otherwise. Non-zero entries in the estimated precision
matrix Θ define conditional dependencies—a sparser and more interpretable graph
than thresholded marginal correlation.

## 4. Proposed Extension: Gene-Gene Interaction Graph

### Constructing the graph

We need a sparse gene-gene graph **G_genes** before computing interaction
features. The key insight is that we already have a gene × sample matrix from
the existing pipeline (step 5: Poisson-Gamma posterior means). This is the
natural starting point.

#### Path A: Data-driven gene KNN from gene × sample matrix

The existing pipeline produces `GammaMatrix` posteriors of shape
(n_genes × n_samples). We can construct a gene-gene KNN graph directly from
these posterior means, reusing the same `ColumnDict` + HNSW machinery that
builds the cell-cell spatial graph.

`ColumnDict<K>` (`matrix-util/src/knn_match.rs`) wraps `instant-distance`
HNSW for approximate nearest-neighbor search with Euclidean distance.
Currently `SrtCellPairs::new` uses it over spatial coordinates to find
cell neighbors; the same API works for genes in sample-expression space:

```
1. Compute M = posterior_log_mean()           [n_genes × n_samples]
2. Row-normalize M (center + unit-variance)   [Euclidean on normalized rows ∝ correlation distance]
3. let dict = ColumnDict::from_dmatrix(M, gene_names)
                                              [each gene = a point in R^n_samples; HNSW index built]
4. For each gene g:
     dict.search_others(&g, k_gene)           [approximate k_gene nearest neighbors]
5. Reciprocal filter (same DashMap pattern    [keep edge (g1,g2) iff g1 ∈ KNN(g2) AND g2 ∈ KNN(g1)]
   as srt_cell_pairs.rs lines 384-399)
6. Build CscMatrix from surviving edges
```

Row-normalization in step 2 is important: on zero-mean unit-variance vectors,
Euclidean distance is a monotone function of Pearson correlation
(`d² = 2n(1 - r)`), so the HNSW index effectively finds genes with the
highest absolute correlation—without needing a custom distance metric.

**Advantages**: fully data-driven; adapts to the specific tissue and condition;
reuses `ColumnDict`, `search_others`, and the reciprocal-filtering pattern
with no new dependencies.

**When it works best**: enough pseudobulk samples (≥ 50) to estimate
gene-gene correlation reliably.

#### Path B: BioGRID interaction network as prior graph

Import known genetic and physical interactions from BioGRID as edges:

```
1. Download BioGRID tab-delimited for the target species
2. Filter to genetic interactions (|ε| > 0.08) and/or physical PPI
3. Map gene symbols to row indices in our expression matrix
4. Construct CscMatrix adjacency from matched edges
5. (Optional) Add edge weights from ε scores or PPI confidence
```

**Advantages**: biologically grounded; no dependence on sample size;
captures interactions invisible to co-expression (e.g., synthetic lethal
pairs that are anti-correlated or uncorrelated in expression).

**When it works best**: well-studied organisms (human, mouse, yeast) with
high BioGRID coverage.

#### Combining both paths

Use wGLASSO (Section 3) or a simpler union-then-filter strategy:

- **Union**: take edges from Path A ∪ Path B; mark provenance
- **Intersection-weighted**: edges in both paths get higher confidence
- **wGLASSO**: Path B sets the penalty weights, Path A's sample covariance
  drives the graphical lasso → edges reflect both data and prior

### Interaction features: SL-style statistics on raw expression

A key simplification: we can **drop the cell-cell delta imputation** and work
directly with raw expression. The delta was designed to isolate per-gene
cell-specific deviation from neighborhood expectation. But when we compute
gene-gene interaction statistics, the interaction itself—the deviation from
what two genes would do independently—already captures the interesting
biology. The cell-cell neighbor imputation becomes unnecessary overhead.

For each gene pair `(g1, g2)` in G_genes and each cell `c`, define the
**epsilon interaction statistic** directly on raw (size-normalized) expression:

```
ε_{g1,g2}(c) = x_{g1}(c) × x_{g2}(c)  -  μ_{g1} × μ_{g2}
```

where `x_g(c)` is the (size-normalized) expression of gene g in cell c and
`μ_g` is the gene-level mean across cells. This is the direct analog of the
synthetic lethality epsilon score: observed joint behavior minus expected
under independence.

**Why this works without deltas:**

- The epsilon statistic already measures **deviation from independence** —
  exactly what deltas + products were trying to capture in two steps
- Positive ε: genes are co-activated beyond expectation (synergistic)
- Negative ε: genes are anti-correlated (antagonistic / SL-like)
- ε ≈ 0: genes act independently in this cell
- Spatial structure is recovered by the cell → sample assignment
  (cells in similar spatial contexts land in the same pseudobulk sample)

**Variant interaction statistics:**

| Statistic | Formula | Interpretation |
|---|---|---|
| **Epsilon (multiplicative)** | `x_{g1} × x_{g2} - μ_{g1} × μ_{g2}` | SL-style deviation from independence |
| **Log epsilon** | `log(x_{g1} + 1) × log(x_{g2} + 1) - E[log(x_{g1}+1)] × E[log(x_{g2}+1)]` | Variance-stabilized; less dominated by high-expressors |
| **Min co-activation** | `min(x_{g1}, x_{g2})` | Both genes must be on; robust to outliers |
| **Signed geometric mean** | `sign(ε) × sqrt(|x_{g1} × x_{g2}|)` | Scale-compressed concordance |

The **log epsilon** is the recommended default for count data: log1p
transformation stabilizes variance, and the centered product on log-scale
is a covariance-like statistic that sums naturally across cells.

The result is features indexed by `(gene_pair, cell)` — no cell-cell pairing
needed for the interaction computation itself. Cell-cell structure enters only
through the sample assignment step.

## 5. Expand-Then-Collapse Strategy

### Gene pairs as linearized attention

The gene-gene interaction features have the structure of a **sparse attention
matrix** over genes, computed per cell. In standard (quadratic) attention,
each token pair gets a score `Q_i · K_j`; here, each gene pair gets a score
`ε_{g1,g2}(c)` that captures their joint deviation from independence. The
KNN-sparsified gene graph plays the role of the attention mask—restricting
which pairs interact—while the epsilon statistic plays the role of the
attention logit.

This is rich information: each gene pair encodes a different interaction mode,
and these modes vary across cells and spatial contexts. We should **preserve
gene-pair granularity** through the pipeline and collapse only the cell axis
(into pseudobulk samples), not the gene-pair axis.

### Simplified pipeline

By working directly with raw expression and SL-style interaction statistics,
the pipeline no longer needs the cell-cell delta imputation step:

```
   raw expression × gene-gene graph
              │
              ▼
     compute ε per cell          (SL-style interaction statistics)
              │
              ▼
     assign cells to samples     (random projection + binary sort, or spatial)
              │
              ▼
     collapse cells → samples    (sum ε per gene pair per sample)
              │
              ▼
     gene-pair × sample          (the new data matrix)
              │
              ▼
     filter gene pairs           (post-collapse sparsity control)
              │
              ▼
     Poisson-Gamma → SVD / topic model
```

Note: cell-cell pairing is no longer required for the interaction computation.
Cells can be assigned to pseudobulk samples by spatial random projection
directly (existing `binary_sort_columns` on cell coordinates), without the
reciprocal-KNN + neighbor-imputation machinery. The cell-cell graph becomes
**optional**—useful if we want spatially-conditioned samples, but not a
prerequisite for the gene-gene interaction features.

### Steps

1. **Compute ε per cell**: for each cell c and each edge `(g1, g2)` in
   G_genes, compute `ε_{g1,g2}(c) = log1p(x_{g1}) × log1p(x_{g2}) - μ̃_{g1} × μ̃_{g2}`
   where `μ̃_g = E[log1p(x_g)]` across cells. This is a sparse operation:
   only |E(G_genes)| interactions per cell, not p².

2. **Assign cells to samples**: use random projection on cell features
   (expression or spatial coordinates) + `binary_sort_columns` to partition
   cells into pseudobulk samples. This reuses existing infrastructure.

3. **Collapse cells → samples**: sum ε values per gene pair across cells
   within each sample. Result: **gene_pair × sample** matrix—gene pairs
   remain as individual rows.

4. **Filter gene pairs** (see below): prune edges with weak or constant
   interaction signal across samples.

5. **Fit**: apply Poisson-Gamma + SVD/topic model on the gene_pair × sample
   matrix. Each latent topic is a weighted combination of gene-gene
   interaction modes, directly interpretable as an interaction program.

### Filtering gene pairs

The gene-gene graph already enforces sparsity (k_gene neighbors per gene), but
further filtering after collapsing keeps the matrix manageable and sharpens
the signal:

| Filter | Criterion | Effect |
|---|---|---|
| **Variance filter** | Drop gene pairs with low variance of ε across samples | Removes constitutive co-expression (no spatial modulation) |
| **Signal-to-noise** | Keep pairs where |mean(ε)| / sd(ε) exceeds threshold | Focuses on reproducible interactions |
| **Marginal independence** | Drop pairs where |mean(ε)| < threshold across samples | Removes non-interacting pairs (ε ≈ 0 everywhere) |
| **Top-k per gene** | Keep only the top-k interactors per gene by |mean(ε)| | Hard cap on matrix rows; ensures every gene is represented |

These filters are cheap on the collapsed gene_pair × sample matrix.

### Feature engineering for gene pairs

**Handling signed ε.** The epsilon statistic is naturally signed (positive =
synergistic, negative = antagonistic). Two approaches for the Poisson-Gamma
framework, which requires non-negative inputs:

- **Split into positive and negative channels**: accumulate
  `ε⁺ = max(ε, 0)` and `ε⁻ = max(-ε, 0)` separately per gene pair per
  sample. Each channel is non-negative and Poisson-Gamma compatible. The
  gene_pair × sample matrix has 2|E| rows. This is the recommended approach:
  it preserves the sign information and lets the SVD/topic model learn which
  gene pairs are synergistic vs. antagonistic in each program.

- **Squared epsilon**: use `ε²` as the interaction feature. Always
  non-negative, captures interaction strength regardless of direction.
  Loses sign but halves the row dimension.

**Multi-feature per gene pair.** We can track multiple interaction features
per gene pair, analogous to how the current pipeline concatenates left_delta,
right_delta, left_resid, right_resid:

- **ε⁺ / ε⁻**: signed interaction channels (synergistic / antagonistic)
- **Co-activation**: `min(log1p(x_{g1}), log1p(x_{g2}))` — both genes on
- **Differential**: `|log1p(x_{g1}) - log1p(x_{g2})|` — one gene on, other off

These become parallel columns in the gene_pair × sample matrix, concatenated
horizontally before SVD/topic model fitting.

## 6. Mathematical Formulation Sketch

### Current model (per-gene, independent)

For gene g, sample s:

```
Y_{g,s} ~ Poisson(λ_{g,s})
λ_{g,s} ~ Gamma(a_g, b_g)    (prior)
λ_{g,s} | Y_{g,s} ~ Gamma(a_g + Y_{g,s}, b_g + n_s)   (posterior)
```

SVD operates on log E[λ_{g,s} | Y] across g and s.

### Proposed model (gene-pair interactions)

For gene pair (g1, g2) ∈ E(G_genes), cell c, define:

```
ε_{g1,g2}(c) = log1p(x_{g1}(c)) × log1p(x_{g2}(c))  -  μ̃_{g1} × μ̃_{g2}
```

where `μ̃_g = (1/N) Σ_c log1p(x_g(c))` is the gene-level mean.

Aggregate into pseudobulk samples by summing over cells assigned to sample s:

```
ε⁺_{(g1,g2), s} = Σ_{c ∈ sample s}  max(ε_{g1,g2}(c), 0)
ε⁻_{(g1,g2), s} = Σ_{c ∈ sample s}  max(-ε_{g1,g2}(c), 0)
```

Each channel is non-negative and approximately Poisson-distributed (sum of
sparse non-negative terms), so the Poisson-Gamma conjugate framework applies:

```
ε⁺_{(g1,g2), s} ~ Poisson-Gamma
ε⁻_{(g1,g2), s} ~ Poisson-Gamma
```

Apply SVD or topic model on the gene_pair × sample posterior means:

```
log E[ε⁺ | data]             U⁺
                    ≈  [ U⁺ ; U⁻ ]  Σ  V^T
log E[ε⁻ | data]             U⁻
```

where U⁺ and U⁻ are concatenated vertically (one block per sign channel).

The dictionary rows are indexed by (gene_pair, sign), so each latent topic
is a weighted combination of synergistic and antagonistic gene-gene
interactions. The sample loadings V are directly comparable to the existing
per-sample latent codes.

### Connection to covariance decomposition

The aggregated epsilon has a natural interpretation. For a sample s with
n_s cells:

```
Σ_{c ∈ s} ε_{g1,g2}(c) = Σ_c log1p(x_{g1}) × log1p(x_{g2})  -  n_s × μ̃_{g1} × μ̃_{g2}
                        = n_s × [ Ĉov_s(log1p(x_{g1}), log1p(x_{g2})) + (μ̂_{g1,s} - μ̃_{g1})(μ̂_{g2,s} - μ̃_{g2}) ]
```

where Ĉov_s is the within-sample covariance and μ̂_{g,s} is the sample mean.
So the aggregated epsilon decomposes into **within-sample covariance** (local
co-regulation) and **mean-shift interaction** (sample-level co-expression
shift). Both are biologically meaningful: the first captures cell-level
co-regulation programs; the second captures tissue-region-level co-expression.

### Sparsity is critical

With p genes, there are p(p-1)/2 possible pairs. Even modest gene sets (p=1000)
give ~500K pairs. The gene-gene graph G_genes enforces sparsity: if each gene
has at most k_gene neighbors (e.g., k_gene=20), the number of gene pairs is at
most p · k_gene / 2 = 10K for p=1000—tractable.

## 7. Implementation Notes

### Reuse of existing building blocks

| Component | Crate / module | Current use | New use |
|---|---|---|---|
| `ColumnDict` + HNSW | `matrix_util::knn_match` | Cell-cell spatial KNN | Gene-gene KNN in sample space (Path A) |
| Reciprocal filtering | `srt_cell_pairs.rs` (DashMap pattern) | Symmetric cell graph | Symmetric gene graph |
| `CscMatrix` + COO construction | `nalgebra_sparse` via `srt_cell_pairs.rs` | Cell-cell spatial graph | Gene-gene graph storage |
| `GammaMatrix` | `matrix_param` | Per-gene Poisson-Gamma posteriors | Per-gene-pair Poisson-Gamma posteriors (ε⁺, ε⁻) |
| `binary_sort_columns` | `srt_random_projection.rs` | Assign cell pairs to samples | Assign cells to samples (no pairing needed) |
| `read_columns_csc` visitor | `data_beans::SparseIoVec` | Read cell-pair expression | Read per-cell expression for ε computation |
| RSVD | `matrix_util` | SVD on gene × sample | SVD on gene_pair × sample |
| `LogSoftmaxEncoder` / `TopicDecoder` | `candle_util` | VAE topic model on gene features | VAE topic model on gene-pair features |

Note: the cell-cell pairing (`SrtCellPairs`), neighbor imputation
(`impute_with_neighbours`), and delta computation are **not needed** in this
pipeline. The interaction statistic operates on raw per-cell expression.

### Integration point

The gene-gene interaction pipeline can be a **standalone subcommand or an
additional stage** after the existing per-gene pipeline. The flow:

1. **(Optional) Run existing per-gene pipeline** through pseudobulk collapsing
   and Poisson-Gamma fit — needed only if using Path A for gene graph
   construction (gene KNN from posterior means).
2. **Construct G_genes** via Path A (gene KNN from posterior means),
   Path B (BioGRID import), or both combined (Section 4).
3. **Compute ε per cell**: visit each cell's expression column, compute
   interaction statistics for all edges in G_genes (sparse inner product).
4. **Assign cells to samples**: random projection on expression or spatial
   coordinates + `binary_sort_columns` (reuses existing infrastructure;
   cell-cell pairing no longer required).
5. **Collapse**: sum ε⁺ and ε⁻ per gene pair across cells within each sample.
6. **Filter**: prune gene pairs with weak signal (Section 5).
7. **Fit Poisson-Gamma** on the gene_pair × sample matrix.
8. **RSVD or topic model** on posterior means.
9. Nyström projection / encoder evaluation on the gene-pair feature space.

### Data structures

```
SrtGenePairGraph {
    /// Sparse adjacency: gene indices → neighbor gene indices
    adjacency: CscMatrix<f32>,
    /// Edge list for iteration: Vec<(g1, g2)>
    edges: Vec<(usize, usize)>,
    /// Edge weights (correlation from Path A, ε/PPI confidence from Path B)
    weights: Vec<f32>,
    /// Provenance per edge: DataDriven, BioGRID, or Both
    source: Vec<EdgeSource>,
}

SrtGenePairCollapsedStat {
    /// Positive interaction sums ε⁺: (n_gene_pairs × n_samples)
    eps_pos_ds: Mat,
    /// Negative interaction sums ε⁻: (n_gene_pairs × n_samples)
    eps_neg_ds: Mat,
    /// Number of cells per sample
    size_s: DVec,
}
```

## 8. Computational Considerations

### Scaling

| Quantity | Typical range | Notes |
|---|---|---|
| Genes (p) | 500 – 5,000 | After filtering to HVGs |
| Gene-gene edges | p × k_gene / 2 ≈ 5K – 50K | With k_gene = 20 |
| Cells (N) | 10K – 500K | Depends on tissue size |
| Samples | 50 – 500 | After binary sorting |
| Gene-pair × sample matrix | 50K × 500 × 2 = 50M entries | ×2 for ε⁺/ε⁻ channels; fits in memory |

### Bottleneck: per-cell ε computation

For each cell, we read its sparse expression vector and compute |E(G_genes)|
interaction statistics. With 100K cells and 10K gene edges, that is ~1 billion
multiply-accumulate operations — but this is actually **cheaper** than the
current cell-pair pipeline (which reads two cells + their neighbors per pair).

The computation is embarrassingly parallel across cells and I/O-bound
(one `read_columns_csc` per cell chunk). Each cell's expression vector is
sparse (typically 1K–5K non-zero entries), and the gene-gene edge list is
pre-sorted, so the ε computation is a sparse inner-product-like operation.

Mitigations for very large datasets:

- **Chunked visitor pattern**: reuse the existing `read_columns_csc` batching
  to amortize I/O. Process cells in chunks of 100–1000.
- **Subsample cells**: use a random subset of cells for ε accumulation
  (pseudobulk averaging smooths out per-cell noise anyway).
- **Random projection on gene-pair features**: if |E| > 50K, project the
  ε vector (length |E|) onto a random basis of dimension d_proj << |E|
  before accumulating into the sample matrix.

### Memory

The gene_pair × sample matrix is the primary new allocation. At 50K gene pairs
× 500 samples × 2 channels × 4 bytes = 200 MB, this is modest. The transient
per-chunk ε vectors are bounded by chunk_size × |E| × 4 bytes.
