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
