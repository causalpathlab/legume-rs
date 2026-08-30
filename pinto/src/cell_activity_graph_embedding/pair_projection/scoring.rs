//! Predictive scoring for a projected pair.
//!
//! The solver estimates the multinomial partition by importance sampling because
//! it pays that cost once per Adam step. Scoring pays it once per pair, so it
//! sums the active axis exactly — one exhaustive pass costs what a handful of
//! sampled steps do, and an approximate denominator here would put sampling
//! noise straight into the number being reported.

use super::{PairDictionary, SCORE_CLAMP};
use matrix_util::agreement::{pearson_log1p, spearman, CellAgreement};

/// The gene axis a run scores over, resolved once.
pub struct EvalAxis {
    /// Positions the CORRELATIONS run over; `None` leaves them `NaN`. Also the
    /// marker for "the user restricted the axis" — when absent, everything active
    /// is scored by the likelihood but nothing is correlated, because a sort per
    /// pair is not a cost to pay by default.
    correlated: Option<Vec<u32>>,
    /// Positions the likelihood normalises over.
    over: Vec<u32>,
    /// Membership by active-list position — a bitmap, not a hash set.
    is_scored: Vec<bool>,
    /// `ln Σ exp(b_g)` over `over`; constant for the run.
    z_null: f32,
}

impl EvalAxis {
    #[must_use]
    fn scores(&self, local: usize) -> bool {
        self.is_scored.get(local).copied().unwrap_or(false)
    }

    /// Streaming log-sum-exp of `f` over the scored positions.
    fn log_partition(&self, f: impl Fn(usize) -> f32) -> f32 {
        log_sum_exp(self.over.iter().map(|&g| f(g as usize)))
    }
}

/// One-pass log-sum-exp: rescales the accumulator when a new maximum arrives,
/// so nothing overflows and the input is walked once.
fn log_sum_exp(values: impl Iterator<Item = f32>) -> f32 {
    let (mut max, mut acc) = (f32::NEG_INFINITY, 0f32);
    for v in values {
        if v > max {
            acc *= (max - v).exp();
            max = v;
        }
        acc += (v - max).exp();
    }
    max + acc.ln()
}

/// One pair's held-out score, on the same multinomial nats/count scale senna's
/// `predictive.parquet` reports — which is the point: a cell pair pools two
/// cells, but nats per observed count does not care how many cells went in, so
/// the two commands' columns can sit in one table.
#[derive(Clone, Copy, Debug)]
pub struct PairScore {
    pub llik: f32,
    /// The same likelihood under `b_g` alone — the gene abundances with no
    /// pair-specific embedding. Every pair scores against this floor, so the
    /// difference is what the latent bought.
    pub null_llik: f32,
    pub total: f32,
    pub agreement: CellAgreement,
}

impl Default for PairScore {
    fn default() -> Self {
        Self {
            llik: 0.0,
            null_llik: 0.0,
            total: 0.0,
            agreement: CellAgreement {
                spearman: f32::NAN,
                pearson_log1p: f32::NAN,
            },
        }
    }
}

impl PairDictionary {
    /// Score one pair's observed profile against the prediction its latent implies.
    ///
    /// `eval` restricts the whole score — likelihood, null and correlations — to a
    /// fixed set of active-list positions. Restricting the likelihood too is what
    /// makes it comparable with `senna predict`: renormalising over the scored
    /// genes turns it into the conditional multinomial "given a count landed in
    /// this gene set, which gene is it", which is exactly what senna reports. Two
    /// commands answering the same question is worth more here than each
    /// answering its own.
    #[must_use]
    pub fn score(&self, obs: &[(u32, f32)], theta: &[f32], axis: &EvalAxis) -> PairScore {
        let local = self.to_local(obs);
        if local.is_empty() {
            return PairScore::default();
        }

        let d = self.d;
        let mut log_rate = vec![0f32; self.b.len()];
        for (g, lr) in log_rate.iter_mut().enumerate() {
            let row = &self.feat[g * d..(g + 1) * d];
            let dot: f32 = row.iter().zip(theta).map(|(&e, &t)| e * t).sum();
            *lr = (dot + self.b[g]).clamp(-SCORE_CLAMP, SCORE_CLAMP);
        }
        let z_model = axis.log_partition(|g| log_rate[g]);

        let mut llik = 0f64;
        let mut null_llik = 0f64;
        let mut total = 0f32;
        for &(l, x) in &local {
            let l = l as usize;
            if !axis.scores(l) {
                continue;
            }
            total += x;
            llik += f64::from(x) * f64::from(log_rate[l] - z_model);
            null_llik += f64::from(x) * f64::from(self.b[l] - axis.z_null);
        }
        if !total.is_finite() || total <= 0.0 {
            return PairScore::default();
        }

        PairScore {
            llik: llik as f32,
            null_llik: null_llik as f32,
            total,
            agreement: self.agreement(&local, &log_rate, z_model, total, axis),
        }
    }

    /// Observed against predicted over the evaluation axis.
    ///
    /// The observed side is densified onto that axis rather than the sparse
    /// profile being correlated directly: a held-out profile is mostly zeros, and
    /// those zeros are data — a model that puts mass on an unobserved gene has to
    /// be charged for it.
    fn agreement(
        &self,
        local: &[(u32, f32)],
        log_rate: &[f32],
        z_model: f32,
        total: f32,
        axis: &EvalAxis,
    ) -> CellAgreement {
        let Some(axis) = axis.correlated.as_deref() else {
            return CellAgreement {
                spearman: f32::NAN,
                pearson_log1p: f32::NAN,
            };
        };
        let mut obs = vec![0f32; self.b.len()];
        for &(l, x) in local {
            obs[l as usize] += x;
        }
        let o: Vec<f32> = axis.iter().map(|&g| obs[g as usize]).collect();
        let p: Vec<f32> = axis
            .iter()
            .map(|&g| (log_rate[g as usize] - z_model).exp() * total)
            .collect();
        CellAgreement {
            spearman: spearman(&o, &p),
            pearson_log1p: pearson_log1p(&o, &p),
        }
    }

    /// Build the scoring axis once for a whole run.
    ///
    /// Everything here was previously rebuilt inside every pair's `score`: the
    /// membership set (a SipHash `HashSet` per pair), and the null's
    /// log-partition, which depends only on `b` and the axis. With hundreds of
    /// thousands of pairs that was `|eval|` hashed inserts and `n_active`
    /// `exp()` calls per pair, for two values that never change.
    #[must_use]
    pub fn eval_axis(&self, positions: Option<Vec<u32>>) -> EvalAxis {
        let n_active = self.b.len();
        let mut is_scored = vec![positions.is_none(); n_active];
        if let Some(p) = positions.as_deref() {
            for &g in p {
                if let Some(slot) = is_scored.get_mut(g as usize) {
                    *slot = true;
                }
            }
        }
        let over: Vec<u32> = match positions.as_deref() {
            Some(p) => p.to_vec(),
            None => (0..n_active as u32).collect(),
        };
        let z_null = log_sum_exp(over.iter().map(|&g| self.b[g as usize]));
        EvalAxis {
            correlated: positions,
            over,
            is_scored,
            z_null,
        }
    }

    /// Map feature names to active-list positions for `--eval-features`.
    #[must_use]
    pub fn eval_positions(
        &self,
        gene_names: &[Box<str>],
        wanted: &std::collections::HashSet<&str>,
    ) -> Vec<u32> {
        gene_names
            .iter()
            .enumerate()
            .filter(|(_, n)| wanted.contains(n.as_ref()))
            .filter_map(|(g, _)| {
                let l = *self.local_of_gene.get(g)?;
                (l != u32::MAX).then_some(l)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests;
