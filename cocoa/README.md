# COunterfactual COnfounder Adjustment

## Simulation DAGs

### `simulate-one` (single cell type)

No cell-type heterogeneity. Individual-level confounding only.

```
    V
   / \
  v   v
  X   |
  |   |
  v   v
  Y_g
```

| Edge | Parameter | Description |
|------|-----------|-------------|
| V → X | `pve_covar_exposure` | Confounder drives exposure assignment via `logit(X) ~ V*α + ε` |
| X → Y | `pve_exposure_gene` | Causal effect of exposure on gene expression (causal genes only) |
| V → Y | `pve_covar_gene` | Confounder directly affects gene expression via `V*γ` |

Generative model:
```
V_i ~ N(0, I)                                          [individual confounders]
X_i ~ Cat(softmax(V_i * α * pve + ε * (1-pve)))        [exposure assignment]
log μ_{ig} = β_g * X_i * √pve_xg + V_i * γ_g * √pve_vg + ε * √(1-pve_xg-pve_vg)
Y_{ijg} ~ Poisson(ρ_j * exp(log μ_{ig}))               [cell-level counts]
ρ_j ~ Gamma(a, b)                                      [cell depth]
```

### `simulate-collider` (multiple cell types)

Cell-type assignment A is a collider on the V→X→Y path.
U is a cell-level confounder that affects both A and Y.

```
    V           U
   / \         / \
  v   v       v   v
  X   |       A   |
  |\ _|_ ___/|   |
  | \ | /    |   |
  v  vvv     v   v
  Y_g(A)
```

More precisely:

```
V ──→ X ──→ Y
│     │      ↑
│     ↓      │
│     A ←── U
│     │      │
│     ↓      ↓
└───→ Y      Y
```

| Edge | Parameter | Description |
|------|-----------|-------------|
| V → X | `pve_covar_exposure` | Individual confounder drives exposure |
| X → A | `pve_exposure_celltype` | Exposure shifts cell-type composition (collider) |
| U → A | `pve_cell_covar_celltype` | Cell-level confounder drives cell-type assignment (collider) |
| X → Y | `pve_exposure_gene` | Causal exposure effect on expression (causal genes only) |
| V → Y | `pve_covar_gene` | Individual confounder directly affects expression |
| U → Y | `pve_cell_covar_gene` | Cell-level confounder directly affects expression |
| A → Y | `celltype_effect_size` | Cell-type DE (different baseline expression per type) |

Generative model:
```
V_i ~ N(0, I)                                          [individual confounders]
X_i ~ Cat(softmax(V_i * α * √pve_vx + ε * √(1-pve_vx)))
U_j ~ N(0, I)                                          [cell-level confounders]
A_{ij} ~ Cat(softmax(U_j * δ * √pve_ua + X_i * η * √pve_xa + ε * √(1-pve_ua-pve_xa)))
log μ_{ijg} = Δ_{g,A} + β_g * X_i * √pve_xg + V_i * γ_g * √pve_vg + U_j * ξ_g * √pve_ug + ε * √(1-...)
Y_{ijg} ~ Poisson(ρ_j * exp(log μ))                    [cell-level counts]
```

### Collider bias

Conditioning on cell type A opens the path X → A ← U → Y:

- When X affects A (composition shift), and U also affects A,
  conditioning on A induces a spurious association between X and U.
- Since U → Y, this creates a non-causal path X ↔ U → Y,
  inflating the apparent effect of X on Y even for non-causal genes.

CoCoA addresses this by matching cells across exposure groups using KNN,
constructing counterfactual controls (y0) for each treated cell (y1).
The topic-weighted matching (`z_matched(k)`) ensures matches are within
the same cell type, but residual bias remains when composition shifts
alter the within-type cell-state distribution.

