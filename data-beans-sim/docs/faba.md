# `faba` — RNA modification + processing simulator (counts + m6A + A-to-I + APA)

Generate sparse per-track count data shaped like a `faba all` run: one
`.zarr.zip` per RNA-level track — expression counts, m6A methylation,
A-to-I editing, alternative polyadenylation — with rows named
`{gene}/{track}/{detail}`, plus a full set of ground-truth parquets
describing every latent.

The generative model encodes well-documented coupling between m6A /
A-to-I editing / APA (shared substrate axes, shared writer/editor
programs) so methods see data with the same structural correlations
real RNA-level multi-track assays produce.

## Biological background

| coupling                | how it shows up in the simulator                       |
|-------------------------|--------------------------------------------------------|
| m6A ↔ APA               | both load on the same `utr_length` substrate axis      |
| A-to-I ↔ Alu/dsRNA      | A-to-I loads on a separate `alu` substrate axis        |
| writer/editor programs  | one cell-state topic `k` can activate ≥ 2 modalities   |
|   shared across tracks  | via non-zero `A_{m, k}` in multiple m                  |

What is **NOT** in v1 (opt-in later):
- m6A → APA position bias (Mettl3-KO ⇒ 3′UTR shortening).
- m6A — A-to-I local antagonism (steric exclusion at adjacent sites).
- Writer/editor co-induction beyond the generic shared-topics path.

## Generative model

### Cell-state (shared with writer/editor activity)

```
θ_{k, j}  ~  Dirichlet (softened one-hot, controlled by pve_topic)
                                              [K × N]
```

`K = --k-topics`. The same θ drives both the mRNA pool (via β_topic)
and the writer/editor activity (via A) — a single cell-state axis.

### Substrate (per gene × modality)

Each gene has an `S`-dim latent structural score (default S = 3:
`utr_length`, `drach`, `alu`). Each modality's substrate mask is a
sigmoid-Bernoulli on a linear projection of that score:

```
s_g            ~ N(0, I_S)                    [G × S]
w_m            ∈ ℝ^S, configurable per modality (defaults below)
intercept_m    binary-searched so E[φ_{g,m}] = π_meas[m]
φ_{g, m}       ~ Bernoulli( σ( s_g · w_m + intercept_m ) )
```

Default weights `w_m`:

| modality | utr_length | drach | alu |
|----------|-----------:|------:|----:|
| count    | 0          | 0     | 0   | (φ_count = 1 always)
| m6A      | +0.7       | +1.0  | 0   |
| A2I      | 0          | 0     | +1.2|
| pA       | +1.0       | 0     | 0   |

Consequence: substrate coverage of m6A and pA is positively correlated
through `utr_length`; A-to-I is largely independent (loads on `alu`).

### Writer/editor program coupling

```
A_{m, k}  ~  Bern(π_A) · N(0, σ_A²)           [M × K]
                                              (count row = 0)
```

Column `k` of A is "what cell-state topic k looks like across
modalities". Many columns have non-zero entries in ≥ 2 modalities ⇒
shared programs that activate multiple modifications together; some
have non-zero entries in only one modality ⇒ modality-specific
programs.

### Gene response to programs

```
ξ_{g, k}  ~ Bernoulli(π_z)
z_{g, k}  = ξ_{g, k} · N(0, σ_z²)             [G × K]
```

`z_g` is gene g's response profile to each program k.

### mRNA pool (drives count modality, couples modifiers)

```
log μ_{g, j}  =  β_g
              +  log( (β_topic · θ)_{g, j} )
              +  δ_{g, B(j)}                  (batch effect)
```

`β_topic[G × K]` is drawn via the existing
`core::sample_lognormal_dictionary` (PVE-controlled log-normal with
`E[β] = 1`); `β_g ~ N(0, σ_β²)` is a per-gene baseline.

### Modification rate (per gene × modality × cell)

```
log r_{g, m, j}  =  base_{g, m}
                 +  φ_{g, m} · Σ_k z_{g, k} · A_{m, k} · θ_{k, j}
                 +  γ_m · δ_{g, B(j)}         (modality batch coupling)
```

`base_{g, m} ~ N(0, σ_b²)` is a per-(g, m) intercept. The substrate
gate `φ_{g, m}` zeros the program-driven term for genes that don't
have the structural feature — keeps the rate at base + batch.

### Mixture split (per gene × modality)

```
α_{g, m}  ~  Dir( α_mix · 1_{C_m} )           Σ_c α_{g,m,c} = 1
```

Flat Dirichlet over `C_m` components. `C_count = 2` (spliced,
unspliced); `C_modifier = --components-per-modifier` (default 3).
All components within a single (g, m) share the same modification
rate and mixture weights; they differ only via the per-component
fraction `α_{g,m,c}`.

### Final Poisson rates

```
λ_{g, count, c, j}    =  α_{g, count, c} · μ_{g, j}              · depth_count
λ_{g, m,     c, j}    =  α_{g, m, c}     · μ_{g, j} · r_{g,m,j}  · depth_m
                          if φ_{g, m} = 1; else no row emitted

y_{g, m, c, j}  ~  Poisson( λ_{g, m, c, j} )
```

Modifier reads **multiply through μ** — silent genes can't be
modified. This is the biological reason "count-weighted positive
sampling" works on real data.

### Held-out (g, m) pairs

`--held-out-frac` zeroes out a subset of substrate-positive (g, m)
pairs (no rows emitted). The mask is exported so imputation harnesses
can compare a learner's prediction for the held-out pair against the
simulator's ground-truth (ρ_g, z_g, A_{m, :}, etc.) without leaking
through the training data.

## Outputs

### Per-modality `.zarr.zip` (drop-in for `senna bge --multiome` etc.)

| file                       | rows                                                |
|----------------------------|-----------------------------------------------------|
| `{out}.genes.zarr.zip`     | `gene_i/count/{spliced,unspliced}`                  |
| `{out}.dartseq.zarr.zip`   | `gene_i/m6A/component_c` (substrate-positive g)     |
| `{out}.atoi.zarr.zip`      | `gene_i/A2I/component_c` (substrate-positive g)     |
| `{out}.apa.zarr.zip`       | `gene_i/pA/component_c`  (substrate-positive g)     |

Columns: `cell_0..cell_{N-1}`, identical across files.

### Ground-truth parquets

| file                                        | contents                          |
|---------------------------------------------|-----------------------------------|
| `{out}.substrate.parquet`                   | s_g  `[G × S]` (named axes)       |
| `{out}.substrate_weights.parquet`           | w_m  `[M × S]`                    |
| `{out}.substrate_mask.parquet`              | φ    `[G × M]` (binary)           |
| `{out}.held_out_mask.parquet`               | held-out `[G × M]` (binary)       |
| `{out}.program_writer_editor.parquet`       | A    `[M × K]`                    |
| `{out}.gene_program_loadings.parquet`       | z    `[G × K]`                    |
| `{out}.gene_baseline.parquet`               | β_g  `[G × 1]`                    |
| `{out}.modality_base.parquet`               | base `[G × M]`                    |
| `{out}.topic_dictionary.parquet`            | β_topic `[G × K]`                 |
| `{out}.topic_proportions.parquet`           | θ    `[N × K]`                    |
| `{out}.alpha_{count,m6A,A2I,pA}.parquet`    | α    `[G × C_m]` per modality     |
| `{out}.intercepts.tsv.gz`                   | per-modality intercept + realised coverage |
| `{out}.barcodes.txt`                        | cell barcodes (shared)            |
| `{out}.batch.gz`, `.ln_batch_{m}.parquet`   | only if `--batches > 1`           |

## Knobs

| flag                          | effect                                       |
|-------------------------------|----------------------------------------------|
| `--n-genes` / `--n-cells`     | G, N                                         |
| `--k-topics`                  | K (shared cell-state / writer-editor axis)   |
| `--n-substrate-features`      | S (default 3)                                |
| `--pi-measured`               | per-modality target coverage `count,m6A,A2I,pA` |
| `--components-per-modifier`   | C_m for modifiers (default 3)                |
| `--alpha-mix`                 | Dirichlet concentration on α                 |
| `--sigma-z` / `--sigma-a` / `--sigma-b` / `--sigma-beta` | SDs on the latents |
| `--pi-z` / `--pi-a`           | spike-and-slab sparsity                      |
| `--depth-count` / `--depth-modifier` | per-modality target library size      |
| `--pve-topic` / `--pve-batch` | topic / batch variance shares                |
| `--batches`                   | B                                            |
| `--held-out-frac`             | substrate-positive (g, m) pairs held out     |
| `--rseed`                     | seed                                         |

## Code map

`faba::run_faba` → `faba::latents::{sample_all, sample_held_out}` →
`faba::sample::{sample_count_modality, sample_modifier_modality}` →
`faba::output::write_all`. Reuses:
- `core::sample_lognormal_dictionary` for β_topic,
- `core::sample_log_batch_effects` for batch δ,
- `multiome::sample_nested_topic_proportions` (K_sub = 1) for θ,
- `multiome::sample_poisson_from_logits` for the actual Poisson draw.
