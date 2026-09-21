//! Row-grammar track assignment for `senna gem`.
//!
//! A gem feature axis is one gene axis carrying several **tracks**: the base
//! gene count (`{gene}/count/spliced`, optionally `{gene}/count/unspliced`)
//! plus, for every co-measured modality passed with `--modality`, that
//! modality's two channel rows (`{gene}/m6a/{methylated,unmethylated}`,
//! `{gene}/atoi/{edited,unedited}`, `{gene}/apa/{proximal,distal}`). Every
//! row is `{gene}/{modality}/{channel}`, split by
//! [`data_beans::aux::feature_rows::parse_feature_row`] — the modality and
//! channel are read from the row itself, never from a file name or load
//! order.
//!
//! [`assign_tracks`] is the single place this grammar is enforced: it turns
//! a feature-name axis into a [`TrackPlan`], which is both gem's own view of
//! the axis (per-row track/gene, per-gene HVG pooling — see
//! [`super::hvg::gem_hvg_row_weights`]) and, via [`TrackPlan::to_ge`], the
//! `graph_embedding_util::fit::TrackSpec` the hierarchical trainer consumes.
//!
//! No heuristics: a row that does not fit the grammar, carries a subunit
//! (gem tracks are gene-level only), names a modality outside
//! `{count, m6a, atoi, apa}`, or is a `count` row with a channel other than
//! `spliced`/`unspliced` (so the pooled `{gene}/count/total` track some
//! producers also write is an ERROR here, not a silent third gene) is
//! rejected, naming up to ten offending rows and the total.

use std::collections::BTreeSet;

use data_beans::aux::feature_rows::{
    parse_feature_row, APA, ATOI, COUNT, DISTAL, EDITED, M6A, METHYLATED, PROXIMAL, SPLICED,
    UNEDITED, UNMETHYLATED, UNSPLICED,
};
use graph_embedding_util as ge;
use rustc_hash::FxHashMap;

/// One track: a `(modality, channel)` pair on the gem feature axis. Every row
/// of a gene on this track shares the gene's loading; a track other than 0
/// adds a per-track offset to it (`--offset-l2`).
#[derive(Clone, Debug)]
pub(crate) struct Track {
    pub id: u32,
    pub modality: Box<str>,
    pub channel: Box<str>,
    /// Read by [`TrackPlan::to_ge`] (`ge::fit::TrackInfo::is_count`).
    pub is_count: bool,
}

/// The row-grammar-derived plan for one gem feature axis. `Clone` so
/// `senna gem`'s driver call can pass one copy into [`crate::bge::driver::EmbedPlan::tracks`]
/// and keep the original to build [`super::contrast::write_contrast`]'s
/// `after_fit` closure from.
#[derive(Clone, Debug)]
pub(crate) struct TrackPlan {
    /// `tracks[0]` is always `(count, spliced)` — the base track every gene
    /// must have a row on. `tracks[1]` is `(count, unspliced)` when the axis
    /// has one; the rest are sorted by `(modality, channel)`.
    pub tracks: Vec<Track>,
    /// Track id per feature row, in `feature_names` order.
    pub row_track: Vec<u32>,
    /// Dense gene id per row, over ALL tracks. Ids are assigned in first-seen
    /// row order (scanning every track together, not track 0 alone).
    pub row_gene: Vec<u32>,
    /// Gene keys in id order.
    pub gene_names: Vec<Box<str>>,
    /// `base_rows[r]` iff row `r` is on track 0 (`count/spliced`).
    pub base_rows: Vec<bool>,
}

/// Assign every row of a gem feature axis to a track and a gene, from the row
/// grammar alone. See the module docs for the rejected shapes.
pub(crate) fn assign_tracks(feature_names: &[Box<str>]) -> anyhow::Result<TrackPlan> {
    let mut bad_parse: Vec<usize> = Vec::new();
    let mut bad_subunit: Vec<usize> = Vec::new();
    let mut bad_modality: Vec<usize> = Vec::new();
    let mut bad_count_channel: Vec<usize> = Vec::new();
    let mut parsed: Vec<Option<data_beans::aux::feature_rows::FeatureRow<'_>>> =
        Vec::with_capacity(feature_names.len());

    for (r, name) in feature_names.iter().enumerate() {
        let Some(row) = parse_feature_row(name) else {
            bad_parse.push(r);
            parsed.push(None);
            continue;
        };
        if row.subunit.is_some() {
            bad_subunit.push(r);
            parsed.push(None);
            continue;
        }
        if !matches!(row.modality, COUNT | M6A | ATOI | APA) {
            bad_modality.push(r);
            parsed.push(None);
            continue;
        }
        if row.modality == COUNT && !matches!(row.channel, SPLICED | UNSPLICED) {
            bad_count_channel.push(r);
            parsed.push(None);
            continue;
        }
        parsed.push(Some(row));
    }

    if !bad_parse.is_empty() {
        return Err(rows_error(
            "do not parse as `{gene}/{modality}/{channel}` feature rows",
            &bad_parse,
            feature_names,
        ));
    }
    if !bad_subunit.is_empty() {
        return Err(rows_error(
            "carry a subunit; gem tracks are gene-level only",
            &bad_subunit,
            feature_names,
        ));
    }
    if !bad_modality.is_empty() {
        return Err(rows_error(
            "name a modality outside {count, m6a, atoi, apa}",
            &bad_modality,
            feature_names,
        ));
    }
    if !bad_count_channel.is_empty() {
        return Err(rows_error(
            "are `count` rows with a channel outside {spliced, unspliced}",
            &bad_count_channel,
            feature_names,
        ));
    }

    let has_base = parsed
        .iter()
        .flatten()
        .any(|row| row.modality == COUNT && row.channel == SPLICED);
    anyhow::ensure!(
        has_base,
        "no base rows: at least one `{{gene}}/count/spliced` row is required"
    );

    // Track ids: 0 = count/spliced, 1 = count/unspliced when present, the
    // rest sorted ascending by (modality, channel).
    let mut pairs: BTreeSet<(Box<str>, Box<str>)> = BTreeSet::new();
    for row in parsed.iter().flatten() {
        pairs.insert((row.modality.into(), row.channel.into()));
    }
    let base_key: (Box<str>, Box<str>) = (COUNT.into(), SPLICED.into());
    let unspliced_key: (Box<str>, Box<str>) = (COUNT.into(), UNSPLICED.into());

    let mut tracks: Vec<Track> = vec![Track {
        id: 0,
        modality: COUNT.into(),
        channel: SPLICED.into(),
        is_count: true,
    }];
    let mut track_id_of: FxHashMap<(Box<str>, Box<str>), u32> = FxHashMap::default();
    track_id_of.insert(base_key.clone(), 0);

    if pairs.contains(&unspliced_key) {
        let id = tracks.len() as u32;
        tracks.push(Track {
            id,
            modality: COUNT.into(),
            channel: UNSPLICED.into(),
            is_count: true,
        });
        track_id_of.insert(unspliced_key.clone(), id);
    }

    for (modality, channel) in pairs
        .iter()
        .filter(|k| **k != base_key && **k != unspliced_key)
    {
        let id = tracks.len() as u32;
        tracks.push(Track {
            id,
            modality: modality.clone(),
            channel: channel.clone(),
            // The count modality only ever contributes (count, spliced) /
            // (count, unspliced) — both already claimed above — so every
            // track reached here is a non-count modality track.
            is_count: false,
        });
        track_id_of.insert((modality.clone(), channel.clone()), id);
    }

    // Genes: first-seen row order, scanning every track together.
    let mut gene_ids: FxHashMap<Box<str>, u32> = FxHashMap::default();
    let mut gene_names: Vec<Box<str>> = Vec::new();
    let mut row_track: Vec<u32> = Vec::with_capacity(feature_names.len());
    let mut row_gene: Vec<u32> = Vec::with_capacity(feature_names.len());
    let mut base_rows: Vec<bool> = Vec::with_capacity(feature_names.len());

    for row in parsed.iter().flatten() {
        let gid = *gene_ids.entry(row.gene.into()).or_insert_with(|| {
            let g = gene_names.len() as u32;
            gene_names.push(row.gene.into());
            g
        });
        let tid = track_id_of[&(
            Box::<str>::from(row.modality),
            Box::<str>::from(row.channel),
        )];
        row_gene.push(gid);
        row_track.push(tid);
        base_rows.push(tid == 0);
    }

    Ok(TrackPlan {
        tracks,
        row_track,
        row_gene,
        gene_names,
        base_rows,
    })
}

/// Build an error naming up to ten offending rows and the total count.
fn rows_error(what: &str, rows: &[usize], feature_names: &[Box<str>]) -> anyhow::Error {
    let total = rows.len();
    let preview: Vec<&str> = rows
        .iter()
        .take(10)
        .map(|&r| feature_names[r].as_ref())
        .collect();
    let more = total.saturating_sub(preview.len());
    if more > 0 {
        anyhow::anyhow!("{total} feature row(s) {what}: {preview:?} (+{more} more)")
    } else {
        anyhow::anyhow!("{total} feature row(s) {what}: {preview:?}")
    }
}

/// The cell-encoder safetensors suffix `senna bge`'s driver writes for one
/// track: the bare name for track 0 (what `predict` reads by default), else
/// namespaced by the ge track `name` (`{modality}/{channel}`) with `/`
/// replaced by `.`. Takes the raw `(track id, name)` a
/// `graph_embedding_util::TrackEncoder` carries, rather than gem's own
/// [`Track`], since the driver saves whatever the engine handed back.
pub(crate) fn encoder_suffix_for(track: u32, name: &str) -> String {
    if track == 0 {
        "cell_encoder.safetensors".to_string()
    } else {
        format!("cell_encoder.{}.safetensors", name.replace('/', "."))
    }
}

/// The two channels a modality contrasts, `(numerator, denominator)`, for
/// `{out}.feature_contrast.parquet`. `None` for a modality this axis does not
/// recognize. Fixed by the row grammar's own channel vocabulary
/// (`data_beans::aux::feature_rows`), never guessed from what is on the axis.
pub(crate) fn contrast_channels(modality: &str) -> Option<(&'static str, &'static str)> {
    match modality {
        COUNT => Some((UNSPLICED, SPLICED)),
        M6A => Some((METHYLATED, UNMETHYLATED)),
        ATOI => Some((EDITED, UNEDITED)),
        APA => Some((PROXIMAL, DISTAL)),
        _ => None,
    }
}

impl TrackPlan {
    /// The `graph_embedding_util::fit::TrackSpec` this plan describes: track
    /// names are `{modality}/{channel}`, `is_count` iff the modality is
    /// `count`. What `senna gem`'s driver call passes as `FitConfig.tracks`.
    pub(crate) fn to_ge(&self) -> ge::fit::TrackSpec {
        ge::fit::TrackSpec {
            track_of_row: self.row_track.clone(),
            gene_of_row: self.row_gene.clone(),
            tracks: self
                .tracks
                .iter()
                .map(|t| ge::fit::TrackInfo {
                    name: format!("{}/{}", t.modality, t.channel).into(),
                    is_count: t.is_count,
                })
                .collect(),
        }
    }

    /// The track id of `(modality, channel)`, when this plan has one. Used
    /// by [`super::contrast::contrast_rows`] to locate a modality's two
    /// contrast channels.
    pub(crate) fn track_of(&self, modality: &str, channel: &str) -> Option<u32> {
        self.tracks
            .iter()
            .find(|t| t.modality.as_ref() == modality && t.channel.as_ref() == channel)
            .map(|t| t.id)
    }

    /// Row indices on `track`, ascending.
    pub(crate) fn rows_of(&self, track: u32) -> Vec<usize> {
        self.row_track
            .iter()
            .enumerate()
            .filter(|&(_, &t)| t == track)
            .map(|(r, _)| r)
            .collect()
    }
}

#[cfg(test)]
#[path = "tracks/tests.rs"]
mod tests;
