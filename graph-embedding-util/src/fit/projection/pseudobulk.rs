//! Per-**pseudobulk** phase-2 velocity readout (gem β-sharing / lineage-DAG path):
//! identity `θ_pb` + velocity `δ_pb` for every pb node of every collapse level,
//! via the same block Poisson-MAP SGD ([`block_sgd`]) that projects cells — a pb
//! node is just a dense "cell". The per-cell sibling is [`super::cells`].

use super::block_sgd;
use crate::data::UnifiedData;
use candle_util::candle_core::Device;

/// Per-level pseudobulk phase-2 velocity readout: the analytic identity `θ_pb`
/// and velocity increment `δ_pb` for every pb node of one collapse level, each
/// flattened `[n_pb × h]` row-major. Produced by [`project_pbs_phase2`] for the
/// lineage-DAG path — `δ_pb` orients the pb-DAG structure term, `θ_pb` are the
/// latent landmarks the phase-2 cell lift attaches to.
pub struct PbLevelVelocity {
    pub n_pb: usize,
    /// Identity `θ_pb`, `[n_pb × h]` row-major (raw spliced Poisson-MAP).
    pub theta: Vec<f32>,
    /// Velocity `δ_pb`, `[n_pb × h]` row-major; zero rows where undefined.
    pub delta: Vec<f32>,
}

/// Phase-2 **pseudobulk** velocity readout (gem β-sharing / lineage-DAG path).
/// Re-projects every pb node of every level onto the frozen feature dictionary —
/// identity `θ_pb` from its spliced aggregate, velocity `δ_pb` from its unspliced
/// aggregate with `θ_pb` fixed — **through the same block SGD as the per-cell
/// projection** ([`super::cells::project_cells_phase2`]): a pb node is fed as a
/// (dense) "cell", so the readout gets the identified full-log-partition MAP the
/// per-node Newton solve gave up on, on `dev`, not a cache-hostile per-node Gram.
///
/// The pb aggregates (`pb_blobs[level].triplets`) are already batch-corrected, so
/// no batch divisor is applied; `gauge_fix` is **off**, so `θ_pb` stays in the raw
/// as-trained frame the cell-lift differences cells against (pb landmarks are never
/// co-embedded, so there is no common mode to remove). `frozen_e` is row-major
/// `[n_features × h]`.
///
/// Returns one [`PbLevelVelocity`] per level, in `pb_blobs` order (coarsest→finest).
pub(crate) fn project_pbs_phase2(
    frozen_e: &[f32],
    frozen_b: &[f32],
    h: usize,
    pb_blobs: &[UnifiedData],
    unspliced_rows: &[bool],
    lambda: f64,
    dev: &Device,
) -> anyhow::Result<Vec<PbLevelVelocity>> {
    let mut out = Vec::with_capacity(pb_blobs.len());
    for pb in pb_blobs {
        let n_pb = pb.n_cells();
        // Group the pb's (feature, count) edges by pb-node id — the exact shape the
        // per-cell projection consumes, so the block SGD solves a pb node like a
        // (dense) cell: one shared frozen Eᵀ, the full log-partition over every
        // feature (which the observed-only Newton path gave up on), on `dev`.
        let mut feats: Vec<Vec<u32>> = vec![Vec::new(); n_pb];
        let mut counts: Vec<Vec<f32>> = vec![Vec::new(); n_pb];
        for t in &pb.triplets {
            feats[t.cell as usize].push(t.feature);
            counts[t.cell as usize].push(t.count);
        }
        let nodes: Vec<(u32, &[u32], &[f32])> = (0..n_pb)
            .map(|p| (p as u32, feats[p].as_slice(), counts[p].as_slice()))
            .collect();
        // No batch divisor — pb aggregates are already batch-corrected. `gauge_fix`
        // off keeps pb θ raw: the cell-lift differences cells against raw pb θ, and
        // pb landmarks are never co-embedded, so there is no common mode to remove.
        let res = block_sgd::project_cells(
            &block_sgd::Phase2Input {
                feat: frozen_e,
                b_feat: frozen_b,
                h,
                n_cells: n_pb,
                lambda,
                dev,
                label: "pb velocity readout",
                gauge_fix: false,
                joint: false,
            },
            &nodes,
            None,
            Some(unspliced_rows),
        )?;
        out.push(PbLevelVelocity {
            n_pb,
            theta: res.theta,
            delta: res.velocity.unwrap_or_else(|| vec![0f32; n_pb * h]),
        });
    }
    Ok(out)
}

#[cfg(test)]
#[path = "pseudobulk_tests.rs"]
mod tests;
