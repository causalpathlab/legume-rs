# Stochastic data Embedding with Nearest Neighbourhood Adjustment

## `senna indexed-topic` — Embedded Topic Model

The per-cell top-K feature window training (`indexed-topic` / `itopic`) factorizes the topic-feature distribution β through a shared per-gene embedding ρ — Dieng et al. (2020)'s ETM, on packed top-K inputs.

**Parameters**
- ρ ∈ ℝ^{D×H}: per-gene embedding. **Shared between encoder and decoder.**
- α ∈ ℝ^{K×H}: per-topic embedding (decoder).

**Model**

```text
encoder (input pooling):
    v_norm  = anscombe_lite(values, batch_null, μ_d)                   # [N, K]
    h       = Σ_k v_norm[k] · ρ[idx[k]]                                # [N, H]
    z       ∼ Gaussian(μ_z(h), σ_z(h)),    log θ = log_softmax(z)      # [N, K_topics]

decoder (ETM factorization):
    β_kd    = log_softmax_d(α_k · ρ_dᵀ)                                # [K_topics, D]
    log x̂_n = log(Σ_k θ_nk · exp β_k:)                                 # multinomial recon
```

**Per-batch slice (no [K, D] materialization).** For the union of top-K feature ids across the minibatch, only `ρ_S = ρ[S, :]` is gathered:

```text
β̃_kS = log_softmax_S(α · ρ_Sᵀ − log q_S)
```

where `log q_S` is the Jean et al. (2015) importance correction for sampled softmax (`q_s` = per-feature selection frequency). Per-batch cost is O(K·S·H) on the gradient path; the full β is only built on demand for output / evaluation.

**Why share ρ.** Each gradient step lands on the same `Var` from two paths — the encoder's bag-of-features pool *and* the decoder's α·ρᵀ logits — so feature embeddings are densely supervised even though only top-K features fire per cell. Parameter count drops from K·D to K·H + D·H (H ≪ D in typical configurations).

**Outputs.** In addition to the standard topic-model artifacts (`.dictionary.parquet`, `.latent.parquet`, `.safetensors`, etc.), `indexed-topic` writes:

- `{out}.feature_embedding.parquet` — D × H learned ρ. Directly usable for gene-gene similarity, clustering into programs, or initializing downstream models.

## `senna masked-vae` — masked imputation with an unconstrained latent (BERT-style)

The continuous-latent sibling of `masked-topic` (alias `bert`). Same masked-imputation
pipeline — PB-collapse training, shared per-gene embedding ρ, NB ETM head, encoder-only
cell inference — but the encoder emits a raw `z` with no simplex softmax. It is
deterministic and KL-free, like `masked-topic` and `masked-sbp`: the masking is the
regularizer.

**Model**

```text
encoder (masked, visible-pooled):
    h       = Σ_{k∈visible} v_norm[k] · ρ[idx[k]]            # bag-of-visible-genes
    z       = clamp(W_z · h)                                 # raw latent, NO softmax

decoder (NB ETM head, reused unchanged):
    θ       = softmax(z)                                     # gene-axis mixture weights
    β_kg    = softmax_d(α_k · ρ_dᵀ)                          # per-topic gene dist (full-vocab Zₖ)
    μ_g     = ℓ · Σ_t θ_t · β_{t,g}
    x_g     ∼ NB(μ_g, φ_g)        for masked g

loss = − Σ_{masked g} log NB(x_g | μ_g, φ_g)
```

**Latent vs masked-topic.** Both are deterministic and KL-free, and both hand the
decoder `log_softmax` of the encoder's logits, so they train the same model with the
same gradients. The difference is what is written out: `masked-topic` stores `log θ`,
`masked-vae` stores the raw pre-softmax `z`, which is the more natural input for a
downstream kNN or linear model. NB objective only. Outputs otherwise match
`masked-topic` (`.dictionary`, `.feature_embedding`, `.dispersion`, `.latent`, …).

> A KL bottleneck (`--kl-weight`) used to sit on top of the masked objective here. At
> its default weight it drove `z` to zero and every cell's θ to uniform, so it was
> removed along with the flag; a model recorded with `kl_weight` still replays, the
> value is ignored.

## `senna vae` — scVI-style Gaussian VAE

The continuous-latent sibling of `senna topic`. Same data pipeline (batch-aware
pseudobulk collapse → multilevel hierarchy → dense VAE), but the latent is an
**unconstrained Gaussian** `z` instead of a simplex `θ`, paired with a gene-axis
softmax NB decoder (the scVI parameterization). Outputs are continuous **factors**
(cell × factor) and **loadings** (gene × factor) — not topic proportions and a
topic-gene dictionary.

**Model**

```text
encoder:
    z ∼ Gaussian(μ_z(x), σ_z(x))                       # [N, K] raw latent (no softmax)

decoder (scVI):
    π_nd = softmax_d(z_n · W + b)                       # gene distribution, sums to 1 over D
    μ_nd = library_n · π_nd
    x_nd ∼ NB(μ_nd, φ_d)                                # per-gene dispersion φ
```

**Why a separate decoder front.** A Gaussian `z` is not on the simplex, so it
cannot drive the topic decoders' mixture `logsumexp_k(log θ_k + log β_kd)` (that
assumes `θ` sums to 1). The gene-axis softmax `softmax_d(z·W)` is the matched
likelihood: it turns an arbitrary real vector into a valid gene distribution.

**Shared training loop.** The dense topic trainer is reused verbatim — with
`topic_smoothing = 0` the simplex smoothing step becomes a no-op, so the raw `z`
flows straight into the decoder's own NB likelihood. The topic-specific machinery
(anchor prior, NB-Fisher weighting, ambient mixture, empirical dictionary, feature
coarsening) does not apply to a continuous-factor model and is skipped. Inference
is encoder-only (no decoder refinement); `senna predict` recognizes the `vae`
model type and runs the encoder over held-out cells.

**Outputs.** `{out}.latent.parquet` (cell × factor `z`), `{out}.dictionary.parquet`
(gene × factor loadings `W`), `{out}.feature_mean.parquet` (per-gene mean rate
`μ_d`), plus the standard `.safetensors` / `.model.json` / `.senna.json`.
