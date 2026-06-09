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

## `senna masked-vae` — masked-imputation Gaussian VAE (BERT-style)

The Gaussian-latent sibling of `masked-topic` (alias `bert`). Same masked-imputation
pipeline — PB-collapse training, shared per-gene embedding ρ, NB ETM head, encoder-only
cell inference — but the latent is a **reparameterized Gaussian** `z` (no simplex
softmax) regularized by a KL term: a genuine variational bottleneck on top of the
masked objective.

**Model**

```text
encoder (masked, visible-pooled):
    h       = Σ_{k∈visible} v_norm[k] · ρ[idx[k]]            # bag-of-visible-genes
    z       ∼ Gaussian(μ_z(h), σ_z(h))                       # raw latent, NO softmax

decoder (NB ETM head, reused unchanged):
    β_kg    = softmax_d(α_k · ρ_dᵀ)                          # per-topic gene dist (full-vocab Zₖ)
    μ_g     = ℓ · Σ_t exp(z_t) · β_{t,g}                     # exp(z) = log-normal intensities
    x_g     ∼ NB(μ_g, φ_g)        for masked g

loss = − Σ_{masked g} log NB(x_g | μ_g, φ_g)  +  β · KL(z ‖ N(0, I))
```

**Why `exp(z)`, not `softmax_d(z·α·ρᵀ)`.** A genuine gene-axis softmax over a Gaussian
`z` needs the full-vocab partition `Σ_d exp(z·α·ρ_d)`, which does *not* decompose per
topic (z couples topics inside the logsumexp) — expensive in the masked/indexed setting.
Using `exp(z)` as **non-negative log-normal topic intensities** keeps the cheap
per-topic partition `Zₖ` the masked head already computes, so the NB decoder
(`impute_masked_nb`) is reused **verbatim** — `z` simply takes the place of `log θ`,
and `exp(z)` the place of the simplex weights. The only new code is the encoder's
Gaussian masked forward (reparameterize + KL).

**Latent vs masked-topic.** `masked-topic` is deterministic (`θ = softmax(z_mean)`,
no KL) — the masking *is* the regularizer. `masked-vae` adds the KL bottleneck back
(reparameterization + `β·KL`), so the masking *and* the KL regularize; the latent is
unconstrained continuous factors rather than topic proportions. `--kl-weight` tunes
β (the masked-NB signal is weaker than a full reconstruction, so β < 1 often helps).
NB objective only. Outputs match `masked-topic` (`.dictionary`, `.feature_embedding`,
`.dispersion`, `.latent`, …).

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
