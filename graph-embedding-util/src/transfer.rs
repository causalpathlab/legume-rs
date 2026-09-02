//! Gene-axis alignment: carrying a trained gene side onto a new dataset's gene
//! axis with a stated status and provenance per gene.
//!
//! Every gene of the UNION of the model's axis and the new data's axis ends up
//! in exactly one of four states:
//!
//! * `Matched` — in both; the trained row, bias and membership verbatim.
//! * `Missing` — in the model, absent from the data; the trained row is kept
//!   (it stays in the partition and its rate can be predicted), nothing is
//!   observed for it.
//! * `Initialized` — in the data, absent from the model; a row placed by
//!   membership: `π̂_g` is the similarity-weighted mean of the membership rows
//!   of the `k` matched genes whose count profiles are closest, `ρ̂_g = π̂_g μ`,
//!   and the bias is set later by moment matching against pass-1 latents.
//! * `Dropped` — in the data, absent from the model, and no way to place it
//!   (no profiles were given, or the model has neither modules nor a usable
//!   neighbour).
//!
//! Pure functions over matrices: no names, no manifests, no files. The caller
//! owns name matching (`new_to_train`), the profile matrix, and where the
//! alignment is written. An initialized row is a PRIOR, never a measurement,
//! and the provenance carried here is what lets every consumer say so.

use nalgebra::DMatrix;
use rayon::prelude::*;

/// Where a union gene came from and what it carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeneStatus {
    Matched,
    Missing,
    Initialized,
    Dropped,
}

/// The trained module tables, when the model has them.
pub struct ModuleTables<'a> {
    /// `[D_train × M]` membership.
    pub pi: &'a DMatrix<f32>,
    /// `[M × H]` module dictionary.
    pub mu: &'a DMatrix<f32>,
}

/// Inputs to [`align_gene_axis`].
pub struct AlignInputs<'a> {
    /// Trained composed rows `[D_train × H]`.
    pub rho: &'a DMatrix<f32>,
    /// Trained per-gene bias `[D_train]`.
    pub b_feat: &'a [f32],
    /// Module tables; `None` for a free model, in which case an unseen gene is
    /// placed on the similarity-weighted mean of its neighbours' ROWS instead.
    pub modules: Option<ModuleTables<'a>>,
    /// For each NEW-data gene, the training row it matched by name, or `None`.
    pub new_to_train: &'a [Option<usize>],
    /// Count profiles of the NEW data's genes over its pseudobulks `[G_new × S]`;
    /// `None` makes every unseen gene `Dropped`.
    pub profiles_new: Option<&'a DMatrix<f32>>,
    /// Neighbours used for an initialized gene.
    pub k: usize,
    /// Below this best cosine similarity a gene is placed on the module-average
    /// (or row-average) prior and flagged `diffuse`.
    pub similarity_floor: f32,
}

/// How an `Initialized` gene was placed.
#[derive(Clone, Debug, PartialEq)]
pub struct Provenance {
    /// Training rows of the neighbours used, best first. Empty when diffuse.
    pub neighbours: Vec<usize>,
    /// Cosine similarity of the best neighbour (`0` when none).
    pub best_similarity: f32,
    /// Placed on the average prior because no neighbour reached the floor.
    pub diffuse: bool,
}

/// The aligned gene side on the union axis: training genes first, in training
/// order, then the new-only genes in the new data's order.
pub struct GeneAlignment {
    pub rows: DMatrix<f32>,
    pub bias: Vec<f32>,
    pub status: Vec<GeneStatus>,
    /// `[G_union × M]` when the model has modules: trained rows for `Matched` /
    /// `Missing`, `π̂` for `Initialized`, zeros for `Dropped`.
    pub membership: Option<DMatrix<f32>>,
    /// Per union gene; `None` unless `Initialized`.
    pub provenance: Vec<Option<Provenance>>,
    /// Union gene → training row.
    pub union_to_train: Vec<Option<usize>>,
    /// Union gene → new-data gene.
    pub union_to_new: Vec<Option<usize>>,
}

impl GeneAlignment {
    pub fn n_union(&self) -> usize {
        self.status.len()
    }

    /// Union genes with a given status.
    pub fn with_status(&self, s: GeneStatus) -> Vec<usize> {
        (0..self.n_union())
            .filter(|&g| self.status[g] == s)
            .collect()
    }
}

/// Build the alignment. See the module doc for the four states.
///
/// Initialized genes get bias `0.0` here: the bias is a scale the membership
/// says nothing about, and [`moment_matched_bias`] sets it once the caller has
/// pass-1 latents.
pub fn align_gene_axis(inputs: &AlignInputs) -> GeneAlignment {
    let n_train = inputs.rho.nrows();
    let h = inputs.rho.ncols();
    let n_new = inputs.new_to_train.len();
    assert_eq!(
        inputs.b_feat.len(),
        n_train,
        "b_feat must match the training rows"
    );
    let n_modules = inputs.modules.as_ref().map(|m| m.pi.ncols());
    if let Some(m) = &inputs.modules {
        assert_eq!(m.pi.nrows(), n_train, "π must match the training rows");
        assert_eq!(m.mu.ncols(), h, "μ must match the embedding width");
    }

    // Training row → the new gene that matched it (first wins on a duplicate).
    let mut train_to_new: Vec<Option<usize>> = vec![None; n_train];
    for (n, t) in inputs.new_to_train.iter().enumerate() {
        if let Some(t) = t {
            assert!(*t < n_train, "new_to_train points past the training axis");
            if train_to_new[*t].is_none() {
                train_to_new[*t] = Some(n);
            }
        }
    }
    let new_only: Vec<usize> = (0..n_new)
        .filter(|&n| inputs.new_to_train[n].is_none())
        .collect();
    let n_union = n_train + new_only.len();

    let mut rows = DMatrix::<f32>::zeros(n_union, h);
    let mut bias = vec![0f32; n_union];
    let mut status = vec![GeneStatus::Dropped; n_union];
    let mut membership = n_modules.map(|m| DMatrix::<f32>::zeros(n_union, m));
    let mut provenance: Vec<Option<Provenance>> = vec![None; n_union];
    let mut union_to_train: Vec<Option<usize>> = vec![None; n_union];
    let mut union_to_new: Vec<Option<usize>> = vec![None; n_union];

    // Training genes: verbatim, matched or missing.
    for t in 0..n_train {
        rows.set_row(t, &inputs.rho.row(t));
        bias[t] = inputs.b_feat[t];
        if let (Some(pm), Some(m)) = (membership.as_mut(), inputs.modules.as_ref()) {
            pm.set_row(t, &m.pi.row(t));
        }
        union_to_train[t] = Some(t);
        union_to_new[t] = train_to_new[t];
        status[t] = if train_to_new[t].is_some() {
            GeneStatus::Matched
        } else {
            GeneStatus::Missing
        };
    }
    for (i, &n) in new_only.iter().enumerate() {
        union_to_new[n_train + i] = Some(n);
    }
    let Some(profiles) = inputs.profiles_new else {
        return GeneAlignment {
            rows,
            bias,
            status,
            membership,
            provenance,
            union_to_train,
            union_to_new,
        };
    };
    assert_eq!(
        profiles.nrows(),
        n_new,
        "profiles must match the new gene axis"
    );

    // Unit-norm, centred log profiles — the same reading of a profile the module
    // warm start uses, so "similar" means the same thing in both places.
    let unit = unit_profile_rows(profiles);
    // The matched genes are the only ones whose membership / row is known.
    let matched_new: Vec<usize> = (0..n_new)
        .filter(|&n| inputs.new_to_train[n].is_some())
        .collect();

    // Average priors for a gene that resembles nothing.
    let avg_membership: Option<Vec<f32>> = inputs.modules.as_ref().map(|m| {
        (0..m.pi.ncols())
            .map(|c| m.pi.column(c).iter().sum::<f32>() / n_train.max(1) as f32)
            .collect()
    });
    let avg_row: Vec<f32> = (0..h)
        .map(|j| inputs.rho.column(j).iter().sum::<f32>() / n_train.max(1) as f32)
        .collect();

    let placed: Vec<(Vec<f32>, Option<Vec<f32>>, Provenance)> = new_only
        .par_iter()
        .map(|&n| {
            // Cosine against every matched gene, best first.
            let q = &unit[n];
            let mut sims: Vec<(f32, usize)> = matched_new
                .iter()
                .map(|&mn| {
                    let s: f32 = q.iter().zip(&unit[mn]).map(|(a, b)| a * b).sum();
                    (s, inputs.new_to_train[mn].expect("matched"))
                })
                .collect();
            sims.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            sims.truncate(inputs.k.max(1));
            let best = sims.first().map_or(0.0, |s| s.0);
            let weights: Vec<f32> = sims.iter().map(|s| s.0.max(0.0)).collect();
            let wsum: f32 = weights.iter().sum();
            if sims.is_empty() || best < inputs.similarity_floor || wsum <= 0.0 {
                let row = match (&avg_membership, &inputs.modules) {
                    (Some(pm), Some(m)) => membership_to_row(pm, m.mu),
                    _ => avg_row.clone(),
                };
                return (
                    row,
                    avg_membership.clone(),
                    Provenance {
                        neighbours: Vec::new(),
                        best_similarity: best.max(0.0),
                        diffuse: true,
                    },
                );
            }
            let neighbours: Vec<usize> = sims.iter().map(|s| s.1).collect();
            let (row, pm) = match &inputs.modules {
                Some(m) => {
                    let mut pm = vec![0f32; m.pi.ncols()];
                    for (&w, &t) in weights.iter().zip(&neighbours) {
                        for (c, v) in pm.iter_mut().enumerate() {
                            *v += w * m.pi[(t, c)] / wsum;
                        }
                    }
                    (membership_to_row(&pm, m.mu), Some(pm))
                }
                None => {
                    let mut row = vec![0f32; h];
                    for (&w, &t) in weights.iter().zip(&neighbours) {
                        for (j, v) in row.iter_mut().enumerate() {
                            *v += w * inputs.rho[(t, j)] / wsum;
                        }
                    }
                    (row, None)
                }
            };
            (
                row,
                pm,
                Provenance {
                    neighbours,
                    best_similarity: best,
                    diffuse: false,
                },
            )
        })
        .collect();

    for (i, (row, pm, prov)) in placed.into_iter().enumerate() {
        let g = n_train + i;
        for (j, v) in row.iter().enumerate() {
            rows[(g, j)] = *v;
        }
        if let (Some(table), Some(pm)) = (membership.as_mut(), pm) {
            for (c, v) in pm.iter().enumerate() {
                table[(g, c)] = *v;
            }
        }
        status[g] = GeneStatus::Initialized;
        provenance[g] = Some(prov);
    }

    GeneAlignment {
        rows,
        bias,
        status,
        membership,
        provenance,
        union_to_train,
        union_to_new,
    }
}

/// `π̂ μ` for one membership row.
fn membership_to_row(pm: &[f32], mu: &DMatrix<f32>) -> Vec<f32> {
    (0..mu.ncols())
        .map(|j| pm.iter().enumerate().map(|(m, &w)| w * mu[(m, j)]).sum())
        .collect()
}

/// Depth-normalize the pseudobulk columns to a common total, `log1p`, centre
/// each gene's row and scale it to unit norm, so a dot product is a cosine on
/// the SHAPE of the profile. A constant row (no shape) becomes the zero vector
/// and is similar to nothing.
fn unit_profile_rows(profiles: &DMatrix<f32>) -> Vec<Vec<f32>> {
    let (d, s) = (profiles.nrows(), profiles.ncols());
    let col_tot: Vec<f32> = (0..s)
        .map(|j| profiles.column(j).iter().sum::<f32>().max(1e-8))
        .collect();
    let mean_tot = col_tot.iter().sum::<f32>() / s.max(1) as f32;
    (0..d)
        .into_par_iter()
        .map(|i| {
            let mut r: Vec<f32> = (0..s)
                .map(|j| (profiles[(i, j)] * mean_tot / col_tot[j]).ln_1p())
                .collect();
            let mean = r.iter().sum::<f32>() / s.max(1) as f32;
            for x in &mut r {
                *x -= mean;
            }
            let norm = r.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-6 {
                for x in &mut r {
                    *x /= norm;
                }
            } else {
                r.fill(0.0);
            }
            r
        })
        .collect()
}

/// Moment-matched bias for initialized rows: with pass-1 latents `theta [N × H]`
/// and cell biases `b_cell [N]` fixed, the bias that makes each gene's total
/// predicted count equal its total observed count,
///
/// ```text
///   â_g = log Σ_c x_cg − log Σ_c exp(ρ̂_g · θ_c + b_c)
/// ```
///
/// `rows` is `[G × H]` (the initialized rows), `observed_total` is `[G]`. A gene
/// with no observed counts gets `fallback`.
pub fn moment_matched_bias(
    rows: &DMatrix<f32>,
    theta: &DMatrix<f32>,
    b_cell: &[f32],
    observed_total: &[f32],
    fallback: f32,
) -> Vec<f32> {
    let (g_n, h) = (rows.nrows(), rows.ncols());
    let n = theta.nrows();
    assert_eq!(theta.ncols(), h, "latent width must match the rows");
    assert_eq!(b_cell.len(), n, "one cell bias per latent row");
    assert_eq!(observed_total.len(), g_n, "one observed total per row");
    (0..g_n)
        .into_par_iter()
        .map(|g| {
            let obs = f64::from(observed_total[g]);
            if obs <= 0.0 {
                return fallback;
            }
            let pred: f64 = (0..n)
                .map(|c| {
                    let s: f32 =
                        (0..h).map(|j| rows[(g, j)] * theta[(c, j)]).sum::<f32>() + b_cell[c];
                    f64::from(s).exp()
                })
                .sum();
            (obs.ln() - pred.max(1e-300).ln()) as f32
        })
        .collect()
}

#[cfg(test)]
mod tests;
