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

## 3. Proposed Extension: Gene-Gene Interaction Graph

### Constructing the graph

Build a sparse gene-gene graph **G_genes** analogously to the cell-cell graph
**G_cells**. Possible construction strategies (not mutually exclusive):

| Strategy | Source | Pros | Cons |
|---|---|---|---|
| Co-expression correlation | Collapsed pseudobulk matrix | Data-driven, no external data needed | Noisy in low-sample regime |
| Pathway / PPI databases | KEGG, STRING, Reactome | Biologically grounded | Incomplete, species-specific |
| Learned jointly | Gradient-based during topic/SVD fit | Adapts to dataset | Harder to optimize; identifiability |

A practical starting point is **thresholded absolute Pearson correlation** across
collapsed samples (after Poisson-Gamma smoothing), keeping the top-k edges per
gene. This mirrors the reciprocal-KNN strategy used for cells but in gene space.

### Interaction features

For each gene pair `(g1, g2)` in G_genes and each cell pair `(i, j)` in G_cells,
define interaction features that capture the joint expression pattern. For example:

- **Product delta**: `delta_{g1}(i,j) * delta_{g2}(i,j)` — captures concordant
  or discordant deviation from neighborhood expectation
- **Absolute difference**: `|delta_{g1}(i,j) - delta_{g2}(i,j)|` — highlights
  differential regulation
- **Min / geometric mean**: preserves co-activation signal while being robust
  to single-gene outliers

The result is a **bipartite interaction structure**: cells × cells AND
genes × genes, producing features indexed by `(gene_pair, cell_pair)`.

## 4. Expand-Then-Collapse Strategy

The pipeline naturally extends the existing expand → collapse → fit pattern to
two collapsing axes:

```
         cell-cell pairs × gene-gene pairs
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             │             ▼
   collapse           │        collapse
   cell-cell          │        gene-gene
        │             │             │
        ▼             ▼             ▼
  gene-pair ×    (full tensor)   cell-sample ×
   sample                        gene-pair
        │                           │
        └───────────┬───────────────┘
                    ▼
           gene-pair × sample
           (doubly collapsed)
```

### Steps

1. **Expand**: for each cell pair `(i, j)`, compute gene-gene interaction
   features for all edges `(g1, g2)` in G_genes. This is a sparse outer
   product restricted to the edges of G_genes—not a full n_genes² operation.

2. **Collapse cell-cell**: aggregate gene-gene interaction patterns across cell
   pairs within each pseudobulk sample (reuse existing `CollapsingOps` /
   sample-assignment machinery). Result: gene_pair × sample matrix.

3. **Collapse gene-gene** (optional): further aggregate gene pairs into
   gene-pair-level summaries—e.g., by grouping correlated gene pairs into
   gene-pair communities. This is analogous to the pseudobulk step but in
   gene-pair space.

4. **Fit**: apply Poisson-Gamma + SVD/topic model on the (doubly) collapsed
   gene_pair × sample matrix.

## 5. Mathematical Formulation Sketch

### Current model

For gene g, sample s:

```
Y_{g,s} ~ Poisson(λ_{g,s})
λ_{g,s} ~ Gamma(a_g, b_g)    (prior)
λ_{g,s} | Y_{g,s} ~ Gamma(a_g + Y_{g,s}, b_g + n_s)   (posterior)
```

SVD operates on log E[λ_{g,s} | Y] across g and s.

### Proposed model

For gene pair (g1, g2), sample s, define the interaction statistic:

```
Z_{(g1,g2), s} = Σ_{(i,j) ∈ sample s}  f(delta_{g1}(i,j), delta_{g2}(i,j))
```

where f is the chosen interaction function (product, abs-diff, etc.) and the sum
runs over cell pairs assigned to sample s.

Then model:

```
Z_{(g1,g2), s} ~ Poisson-Gamma   (same conjugate framework)
```

and apply SVD or topic model on the gene_pair × sample posterior means:

```
log E[Z | data]  ≈  U  Σ  V^T
```

The dictionary U now has rows indexed by gene pairs, so each latent topic is a
weighted combination of gene-gene interactions. The sample loadings V are
directly comparable to the existing per-sample latent codes.

### Sparsity is critical

With p genes, there are p(p-1)/2 possible pairs. Even modest gene sets (p=1000)
give ~500K pairs. The gene-gene graph G_genes enforces sparsity: if each gene
has at most k_gene neighbors (e.g., k_gene=20), the number of gene pairs is at
most p · k_gene / 2 = 10K for p=1000—tractable.

## 6. Implementation Notes

### Reuse of existing building blocks

| Component | Crate / module | Current use | New use |
|---|---|---|---|
| `CscMatrix` + COO construction | `nalgebra_sparse` via `srt_cell_pairs.rs` | Cell-cell spatial graph | Gene-gene co-expression graph |
| `GammaMatrix` | `matrix_param` | Per-gene Poisson-Gamma posteriors | Per-gene-pair Poisson-Gamma posteriors |
| `collapse_pairs` / visitor | `srt_collapse_pairs.rs` | Cell-pair → pseudobulk aggregation | Gene-pair × cell-pair → pseudobulk |
| `binary_sort_columns` | `srt_random_projection.rs` | Assign cell pairs to samples | Unchanged (sample assignment is cell-side) |
| RSVD | `matrix_util` | SVD on gene × sample | SVD on gene_pair × sample |
| `LogSoftmaxEncoder` / `TopicDecoder` | `candle_util` | VAE topic model on gene features | VAE topic model on gene-pair features |

### Integration point

The gene-gene extension adds **new stages between collapsing and fitting**, not
new subcommands. The modified flow:

1. Run existing pipeline through pseudobulk collapsing and Poisson-Gamma fit.
2. **New**: construct G_genes from posterior means of the collapsed gene × sample
   matrix (thresholded correlation).
3. **New**: re-visit cell pairs (`collect_pair_stat_visitor`-style), computing
   gene-pair interaction features for edges in G_genes, accumulating into a
   gene_pair × sample matrix.
4. **New**: fit Poisson-Gamma on the gene_pair × sample matrix.
5. Apply RSVD or topic model on gene_pair × sample posterior means.
6. Nyström projection / encoder evaluation proceeds as before, but on the
   gene-pair feature space.

### Data structures

```
SrtGenePairGraph {
    /// Sparse adjacency: gene indices → neighbor gene indices
    adjacency: CscMatrix<f32>,
    /// Edge list for iteration: Vec<(g1, g2)>
    edges: Vec<(usize, usize)>,
    /// Optional edge weights (correlation, PPI confidence, etc.)
    weights: Option<Vec<f32>>,
}

SrtGenePairCollapsedStat {
    /// Gene-pair interaction sums: (n_gene_pairs × n_samples)
    interaction_sum_ds: Mat,
    /// Sample sizes (shared with cell-pair collapsing)
    size_s: DVec,
}
```

## 7. Computational Considerations

### Scaling

| Quantity | Typical range | Notes |
|---|---|---|
| Genes (p) | 500 – 5,000 | After filtering to HVGs |
| Gene-gene edges | p × k_gene / 2 ≈ 5K – 50K | With k_gene = 20 |
| Cell pairs | 10K – 500K | Depends on tissue size and k_cell |
| Samples | 50 – 500 | After binary sorting |
| Gene-pair × sample matrix | 50K × 500 = 25M entries | Fits in memory as dense |

### Bottleneck: pair-level expansion

Computing gene-pair features for every cell pair is the most expensive step.
For each cell pair, we touch `|E(G_genes)|` gene pairs. With 100K cell pairs
and 10K gene edges, that is 1 billion multiply-accumulate operations—heavy but
embarrassingly parallel across cell pairs.

Mitigations:

- **Random projection on gene-pair features**: project the sparse gene-pair
  vector (length |E|) onto a random basis of dimension d_proj << |E| before
  accumulating into the sample matrix. This parallels the existing random
  projection on raw gene features.
- **Chunked visitor pattern**: reuse the existing `read_columns_csc` batching
  in the visitor to amortize I/O.
- **Subsample cell pairs**: use a random subset of cell pairs for gene-pair
  feature accumulation (the pseudobulk collapsing already averages out noise).

### Memory

The gene_pair × sample matrix is the primary new allocation. At 50K gene pairs
× 500 samples × 4 bytes = 100 MB, this is modest. The transient per-chunk
gene-pair feature vectors are bounded by chunk_size × |E| × 4 bytes.
