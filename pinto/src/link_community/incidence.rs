//! Latent edge incidence (frozen, variational-Bayes Poisson–Gamma block term).
//!
//! Treats each edge as a "visible" unit with a community label `z_e ∈ {0..K-1}`
//! and each incident vertex (cell) as a "hidden" unit with propensity vector
//! `θ[v, ·] ∈ Δ^K` (already produced by `compute_node_membership`). The K×K
//! incidence matrix `B[k, k']` is fitted **once** from the post-V-cycle edge
//! labels under a conjugate `Gamma(a, b)` prior and **frozen** for the
//! duration of the final EM-Gibbs and greedy passes.
//!
//! The cached value is the proper variational-Bayes expected log:
//!
//!     E_q[log B[k, k']] = ψ(a + S[k, k']) − log(b + W[k'])
//!
//! (digamma minus log of rate), where
//!
//!     S[k, k'] = Σ_{e: z_e=k} (θ_L[k'] + θ_R[k'])
//!     W[k']    = Σ_e          (θ_L[k'] + θ_R[k'])     (k-independent)
//!
//! The score gains an additive term
//!
//!     Σ_{k'} (θ_L(e)[k'] + θ_R(e)[k']) · E_q[log B[k, k']]
//!
//! computed by `add_incidence_to_log_probs`. There is no per-move
//! bookkeeping: `B` is a global read-only weight table, exactly the
//! "RBM-style frozen coupling" picture.

use special::Gamma as SpecialGamma;

/// Compute frozen variational `E_q[log B[k, k']]` from the current edge
/// labels and vertex propensity. Output is a length-`k*k` row-major matrix
/// ready to be borrowed read-only by `IncidenceConfig`.
///
/// `propensity` is row-major `[n_cells × k]`.
pub fn fit_log_incidence(
    edges: &[(usize, usize)],
    membership: &[usize],
    propensity: &[f64],
    k: usize,
    a: f64,
    b: f64,
) -> Vec<f64> {
    debug_assert_eq!(edges.len(), membership.len());
    let mut s = vec![0.0f64; k * k];
    let mut w = vec![0.0f64; k];

    for (e, &(i, j)) in edges.iter().enumerate() {
        let c = membership[e];
        debug_assert!(c < k);
        let prop_i = &propensity[i * k..(i + 1) * k];
        let prop_j = &propensity[j * k..(j + 1) * k];
        let base = c * k;
        for kp in 0..k {
            let delta = prop_i[kp] + prop_j[kp];
            s[base + kp] += delta;
            w[kp] += delta;
        }
    }

    let mut log_b = vec![0.0f64; k * k];
    for c in 0..k {
        let base = c * k;
        for kp in 0..k {
            // E_q[log B] = ψ(a + S) − log(b + W)
            log_b[base + kp] = SpecialGamma::digamma(a + s[base + kp]) - (b + w[kp]).ln();
        }
    }
    log_b
}

/// Pack a `[n_cells × k]` row-major `Vec<f64>` propensity from a
/// column-major `nalgebra::DMatrix<f32>`. The flat layout lets the hot
/// loop index a length-K cell row as a contiguous slice.
pub fn pack_propensity_row_major(propensity_mat: &crate::util::common::Mat) -> Vec<f64> {
    let n = propensity_mat.nrows();
    let k = propensity_mat.ncols();
    let mut out = vec![0.0f64; n * k];
    for i in 0..n {
        for kp in 0..k {
            out[i * k + kp] = propensity_mat[(i, kp)] as f64;
        }
    }
    out
}

/// Add the (frozen) incidence contribution to `log_probs` (delta form
/// against the current community), to be called immediately after
/// `compute_log_probs_for_edge`. Pure read of `log_incidence`.
#[inline]
pub fn add_incidence_to_log_probs(
    e: usize,
    edges: &[(usize, usize)],
    propensity: &[f64],
    log_incidence: &[f64],
    k: usize,
    current_c: usize,
    log_probs: &mut [f64],
) {
    let (i, j) = edges[e];
    let prop_i = &propensity[i * k..(i + 1) * k];
    let prop_j = &propensity[j * k..(j + 1) * k];

    let cur_base = current_c * k;
    let mut current_contrib = 0.0f64;
    for kp in 0..k {
        current_contrib += (prop_i[kp] + prop_j[kp]) * log_incidence[cur_base + kp];
    }

    for (t, lp) in log_probs.iter_mut().enumerate().take(k) {
        if t == current_c {
            continue;
        }
        let tgt_base = t * k;
        let mut tgt = 0.0f64;
        for kp in 0..k {
            tgt += (prop_i[kp] + prop_j[kp]) * log_incidence[tgt_base + kp];
        }
        *lp += tgt - current_contrib;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two well-separated cell blocks, edges placed within blocks: the
    /// fitted `log B` should be diagonally dominant.
    #[test]
    fn diagonal_dominance_on_two_blocks() {
        let k = 2usize;
        let n_cells = 6usize;
        let mut prop = vec![0.0f64; n_cells * k];
        for i in 0..3 {
            prop[i * k] = 1.0;
        }
        for i in 3..6 {
            prop[i * k + 1] = 1.0;
        }
        let edges = vec![(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)];
        let membership = vec![0, 0, 0, 1, 1, 1];

        let log_b = fit_log_incidence(&edges, &membership, &prop, k, 1.0, 1.0);

        assert!(log_b[0] > log_b[1]);
        assert!(log_b[k + 1] > log_b[k]);
    }

    /// Moving a within-block edge to the wrong community should be
    /// penalised by the incidence term.
    #[test]
    fn add_to_log_probs_penalises_wrong_block() {
        let k = 2usize;
        let n_cells = 6usize;
        let mut prop = vec![0.0f64; n_cells * k];
        for i in 0..3 {
            prop[i * k] = 1.0;
        }
        for i in 3..6 {
            prop[i * k + 1] = 1.0;
        }
        let edges = vec![(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)];
        let membership = vec![0, 0, 0, 1, 1, 1];
        let log_b = fit_log_incidence(&edges, &membership, &prop, k, 1.0, 1.0);

        let mut log_probs = vec![0.0f64; k];
        add_incidence_to_log_probs(0, &edges, &prop, &log_b, k, 0, &mut log_probs);
        assert!(
            log_probs[1] < 0.0,
            "moving block-0 edge to community 1 should be penalised; got {:?}",
            log_probs
        );
    }
}
