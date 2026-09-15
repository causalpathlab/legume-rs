//! The feature axis as **tracks**: the rows of one track, the split of a cell's
//! folded counts into per-track rows, and the set of encoders a fit trains —
//! one per COUNT track, each pooling only its own track's rows.
//!
//! # Why one encoder per track and not one over the axis
//!
//! A track is a whole measurement: its rows carry their own depth, their own
//! zero rate and their own bias row. Pooling every track's rows into one read
//! makes a cell's placement a function of whichever track happens to dominate
//! its counts, and a cell missing a track is then read as a cell with low
//! counts on it. One trunk per track keeps each read in its own units; the
//! placement is the **mean over the tracks the cell actually has counts on**,
//! so an absent track falls back to the others instead of dragging the cell
//! toward the origin.
//!
//! # The one-track case is the previous code
//!
//! `senna bge` has one count track over the whole axis. Every entry point here
//! short-circuits on that: [`split_rows_by_track`] borrows the folded rows
//! untouched, and [`CellEncoders::theta_block`] / [`CellEncoders::encode_edges`]
//! call the single trunk's forward with no mask, no mean and no extra tensor.

use super::{block_sgd, densify_mapped, null_intercept, CellEncoder, FoldedRow};
use crate::fit::config::TrackSpec;
use crate::fit::projection::FrozenProjection;
use candle_util::candle_core::Tensor;
use rayon::prelude::*;
use std::borrow::Cow;

/// `local_of_row` entry for a global feature row that this track does not hold.
const NOT_IN_TRACK: u32 = u32::MAX;

/////////////////////
// Rows of a track //
/////////////////////

/// One track's rows of the feature axis, with the inverse map.
pub(crate) struct TrackRows {
    pub track: u32,
    /// This track's global feature rows, ascending.
    pub rows: Vec<u32>,
    /// Global feature row → its position within this track; [`NOT_IN_TRACK`]
    /// for a row of another track. Length is the whole feature axis.
    pub local_of_row: Vec<u32>,
}

impl TrackRows {
    /// Every track's rows, in track order, in one pass over the axis.
    pub(crate) fn all(tracks: &TrackSpec) -> Vec<Self> {
        let n_features = tracks.track_of_row.len();
        let mut out: Vec<Self> = (0..tracks.n_tracks())
            .map(|t| Self {
                track: t as u32,
                rows: Vec::new(),
                local_of_row: vec![NOT_IN_TRACK; n_features],
            })
            .collect();
        for (row, &t) in tracks.track_of_row.iter().enumerate() {
            let tr = &mut out[t as usize];
            tr.local_of_row[row] = tr.rows.len() as u32;
            tr.rows.push(row as u32);
        }
        out
    }
}

/// Global feature row → position within a track built from `rows`;
/// [`NOT_IN_TRACK`] elsewhere.
fn local_map(rows: &[u32], n_features: usize) -> Vec<u32> {
    let mut map = vec![NOT_IN_TRACK; n_features];
    for (local, &row) in rows.iter().enumerate() {
        map[row as usize] = local as u32;
    }
    map
}

/// Split every cell's folded counts into one row per track, relabelled to that
/// track's LOCAL feature ids.
///
/// At `T = 1` the single track is the whole axis and its local ids are the
/// global rows (a one-track spec is the plain gene axis), so the rows come back
/// **borrowed**: no copy, nothing allocated per cell. At `T > 1` every edge is
/// copied exactly once — the split walks each row a single time and hands each
/// edge to its own track.
pub(crate) fn split_rows_by_track<'a>(
    rows: &'a [FoldedRow],
    tracks: &TrackSpec,
) -> Vec<Cow<'a, [FoldedRow]>> {
    if tracks.n_tracks() == 1 {
        return vec![Cow::Borrowed(rows)];
    }
    let tr = TrackRows::all(tracks);
    let n_t = tr.len();
    let per_row: Vec<Vec<FoldedRow>> = rows
        .par_iter()
        .map(|r| {
            let mut split: Vec<FoldedRow> = (0..n_t)
                .map(|_| FoldedRow::new(Vec::new(), Vec::new()))
                .collect();
            for (&f, &c) in r.feats.iter().zip(&r.counts) {
                let t = tracks.track_of_row[f as usize] as usize;
                split[t].feats.push(tr[t].local_of_row[f as usize]);
                split[t].counts.push(c);
            }
            split
        })
        .collect();
    let mut by_track: Vec<Vec<FoldedRow>> =
        (0..n_t).map(|_| Vec::with_capacity(rows.len())).collect();
    for split in per_row {
        for (t, fr) in split.into_iter().enumerate() {
            by_track[t].push(fr);
        }
    }
    by_track.into_iter().map(Cow::Owned).collect()
}

///////////////////////
// The encoder set   //
///////////////////////

/// One count track's trained trunk, with the rows it reads.
pub struct TrackEncoder {
    /// Position of this track in the fit's [`TrackSpec`].
    pub track: u32,
    /// The track's name, as the spec gives it.
    pub name: Box<str>,
    /// The track's global feature rows, ascending — the dictionary rows the
    /// trunk was built on.
    pub rows: Vec<u32>,
    pub encoder: CellEncoder,
}

/// The maps a fit placed its cells by: one [`TrackEncoder`] per COUNT track,
/// ascending. A non-count track has no encoder (it contributes to the
/// likelihood but carries no read).
///
/// A one-track fit holds exactly one, which is what `senna bge` persists and
/// what [`Self::single`] hands back.
pub struct CellEncoders {
    encoders: Vec<TrackEncoder>,
    /// Rows of the whole feature axis, across every track.
    n_features: usize,
}

impl CellEncoders {
    pub(crate) fn new(encoders: Vec<TrackEncoder>, n_features: usize) -> Self {
        Self {
            encoders,
            n_features,
        }
    }

    /// The trained trunks, ascending by track.
    #[must_use]
    pub fn iter(&self) -> &[TrackEncoder] {
        &self.encoders
    }

    /// `Some` iff there is exactly ONE encoder — the single file `senna bge`
    /// persists and `senna predict` reloads.
    ///
    /// One encoder is **not** the same as one track: an axis can carry one
    /// count track beside non-count ones, and then this is `Some` while the
    /// encoder reads only part of the axis. Anything that depends on the
    /// encoder covering every feature row must ask [`Self::spans_axis`].
    #[must_use]
    pub fn single(&self) -> Option<&CellEncoder> {
        match self.encoders.as_slice() {
            [only] => Some(&only.encoder),
            _ => None,
        }
    }

    /// One encoder whose rows ARE the whole feature axis — the `senna bge`
    /// shape, and the only one where a node's global feature ids can be fed to
    /// the encoder unsplit. The rows of a track are ascending and distinct, so
    /// a count equal to the axis width means the identity map.
    fn spans_axis(&self) -> bool {
        match self.encoders.as_slice() {
            [only] => only.rows.len() == self.n_features,
            _ => false,
        }
    }

    /// Rebuild a saved set on the run's dictionary: `paths` names one saved
    /// trunk per COUNT track, and `tracks` says which rows each of them reads.
    pub fn load(
        feat: &[f32],
        b_feat: &[f32],
        h: usize,
        tracks: &TrackSpec,
        paths: &[(u32, String)],
        dev: &candle_util::candle_core::Device,
    ) -> anyhow::Result<Self> {
        let mut ordered: Vec<(u32, &str)> = paths.iter().map(|(t, p)| (*t, p.as_str())).collect();
        ordered.sort_by_key(|&(t, _)| t);
        // Every count track, exactly once. A short list would silently place
        // cells by a subset of the maps that trained them, and a list missing
        // track 0 would report another track's intercept as track 0's.
        let named: Vec<usize> = ordered.iter().map(|&(t, _)| t as usize).collect();
        let expected = tracks.count_tracks();
        anyhow::ensure!(
            named == expected,
            "the saved encoders name track(s) {named:?}, but this axis's count track(s) are \
             {expected:?} — one encoder per count track is required"
        );
        let mut encoders = Vec::with_capacity(ordered.len());
        for (t, path) in ordered {
            let info = tracks.tracks.get(t as usize).ok_or_else(|| {
                anyhow::anyhow!("track {t} is past this axis's {} tracks", tracks.n_tracks())
            })?;
            anyhow::ensure!(
                info.is_count,
                "track {t} (`{}`) carries no counts, so it has no encoder to load",
                info.name
            );
            let rows = tracks.rows_of_track(t as usize);
            let encoder = CellEncoder::load_on_rows(feat, b_feat, h, &rows, path, dev)?;
            encoders.push(TrackEncoder {
                track: t,
                name: info.name.clone(),
                rows,
                encoder,
            });
        }
        Ok(Self {
            encoders,
            n_features: tracks.track_of_row.len(),
        })
    }

    /// Nodes to hand one [`Self::encode_edges`] call. Sized from the WHOLE
    /// feature axis: the per-track dense blocks together are `n × n_features`,
    /// the same activation the single-dictionary block was.
    #[must_use]
    pub fn group_nodes(&self) -> usize {
        block_sgd::block_cells(self.n_features)
    }

    /// Subtract `shift [h]` from EVERY encoder's output, so the combined
    /// placement — a mean of the encoders that fired — moves by exactly
    /// `−shift` too. See [`CellEncoder::shift_output`] for why the gauge has to
    /// reach the persisted maps at all.
    pub(crate) fn shift_output(&self, shift: &[f32]) -> anyhow::Result<()> {
        for te in &self.encoders {
            te.encoder.shift_output(shift)?;
        }
        Ok(())
    }

    /// The combine rule for one block: `θ = Σ_t has_t·θ^t / max(1, Σ_t has_t)`
    /// over the count tracks, `has_t` being "this node has counts on track t".
    ///
    /// `xs[i]` / `totals[i]` are the dense block and row totals of
    /// `self.iter()[i]`'s track.
    ///
    /// With one encoder that spans the whole axis this is its forward and
    /// nothing else — no mask, no division, no extra tensor — which is what
    /// keeps the `senna bge` graph exactly what it was. One encoder over PART
    /// of the axis still goes through the mask: a node with no counts on its
    /// track has nothing to average and belongs at the origin.
    pub(super) fn theta_block(
        &self,
        xs: &[Tensor],
        totals: &[&[f32]],
        train: bool,
    ) -> anyhow::Result<Tensor> {
        // The `senna bge` shape only: one encoder over the WHOLE axis, where
        // the combine is the identity and a node can have counts nowhere else.
        if self.spans_axis() {
            let enc = &self.encoders[0].encoder;
            return Ok(enc
                .encoder
                .forward(&xs[0], None, Some(&enc.mean_1d), None, train)?);
        }
        anyhow::ensure!(!self.encoders.is_empty(), "no count track to encode with");
        let dev = xs[0].device();
        let n = xs[0].dim(0)?;
        let mut present = vec![0f32; n];
        let mut sum: Option<Tensor> = None;
        for (i, te) in self.encoders.iter().enumerate() {
            let enc = &te.encoder;
            let theta = enc
                .encoder
                .forward(&xs[i], None, Some(&enc.mean_1d), None, train)?;
            let has: Vec<f32> = totals[i]
                .iter()
                .map(|&v| if v > 0.0 { 1.0 } else { 0.0 })
                .collect();
            for (c, &h) in present.iter_mut().zip(&has) {
                *c += h;
            }
            let mask = Tensor::from_vec(has, (n, 1), dev)?;
            let masked = theta.broadcast_mul(&mask)?;
            sum = Some(match sum {
                None => masked,
                Some(acc) => (acc + masked)?,
            });
        }
        // A node with counts on nothing stays at the origin rather than being
        // divided by zero — the same "no information" position an unseen cell
        // gets from the solver.
        let denom: Vec<f32> = present.iter().map(|&c| c.max(1.0)).collect();
        let denom = Tensor::from_vec(denom, (n, 1), dev)?;
        let sum = sum.ok_or_else(|| anyhow::anyhow!("no count track to encode with"))?;
        Ok(sum.broadcast_div(&denom)?)
    }

    /// Place nodes given as `(id, feature ids, counts)` on the fit's feature
    /// axis — the predict-time entry. `θ` is the mean over the count tracks the
    /// node has counts on; `b_node` is track 0's conditional intercept at that
    /// `θ`, so the reported pair is self-consistent.
    pub fn encode_edges(
        &self,
        nodes: &[(u32, &[u32], &[f32])],
    ) -> anyhow::Result<FrozenProjection> {
        // One encoder over the WHOLE axis: the single-dictionary call,
        // unchanged — a node's global feature ids are that encoder's own ids.
        // One encoder over PART of the axis must NOT come here: its dictionary
        // is narrower than the feature ids the caller hands in.
        if self.spans_axis() {
            return self.encoders[0].encoder.encode_edges(nodes);
        }
        let first = self
            .encoders
            .first()
            .ok_or_else(|| anyhow::anyhow!("no count track to encode with"))?;
        let (h, dev) = (first.encoder.dict.h, first.encoder.mean_1d.device());
        let maps: Vec<Vec<u32>> = self
            .encoders
            .iter()
            .map(|te| local_map(&te.rows, self.n_features))
            .collect();
        let group = self.group_nodes();
        let mut latent = vec![0f32; nodes.len() * h];
        let mut b_node = vec![0f32; nodes.len()];
        for (b, block) in nodes.chunks(group).enumerate() {
            let start = b * group;
            let mut xs = Vec::with_capacity(self.encoders.len());
            let mut totals = Vec::with_capacity(self.encoders.len());
            for (i, te) in self.encoders.iter().enumerate() {
                let d_t = te.encoder.dict.d();
                let (x, tot) = densify_mapped(block, &maps[i], d_t);
                xs.push(Tensor::from_vec(x, (block.len(), d_t), dev)?);
                totals.push(tot);
            }
            let refs: Vec<&[f32]> = totals.iter().map(Vec::as_slice).collect();
            let theta = self.theta_block(&xs, &refs, false)?;
            let c = null_intercept(&first.encoder.dict, &theta, &totals[0])?;
            latent[start * h..(start + block.len()) * h]
                .copy_from_slice(&theta.flatten_all()?.to_vec1::<f32>()?);
            b_node[start..start + block.len()].copy_from_slice(&c);
        }
        Ok(FrozenProjection {
            theta: latent,
            b_node,
        })
    }
}
