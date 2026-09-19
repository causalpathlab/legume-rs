//! `--{freeze,init,lora}-feature-embedding` for `senna gem`: an earlier
//! run's feature table read onto gem's row grammar.
//!
//! A bare name (a `senna bge` or `senna fne` table) is the gene's
//! `count/spliced` row; a name in the grammar (an earlier `senna gem` table)
//! is that row. Genes match by canonical key, as everywhere. The matched
//! base rows (track 0) become the engine's preset on the gene axis, under
//! the flag's mode; a matched row on another track this run has, whose gene
//! also has a given base row, becomes that track's given offset for the
//! gene, `δ₀ = row − base row` (`graph_embedding_util::PresetOffsets`, pinned
//! under freeze, a start otherwise); a track row whose gene has no base row
//! is ignored with a warning, since an offset is relative to the base row.
//! The rows that match nothing are carried through as for every engine
//! ([`senna::carried_rows::CarriedRows`]), a gene under its lifted name.

use crate::gem::tracks::TrackPlan;
use auxiliary_data::feature_rows::{feature_row, parse_feature_row, COUNT, SPLICED};
use graph_embedding_util as ge;
use graph_embedding_util::PresetMode;
use log::{info, warn};
use rustc_hash::FxHashMap;
use senna::carried_rows::CarriedRows;
use std::collections::BTreeMap;

/// What a given table resolves to on a gem axis.
pub(crate) struct GemPreset {
    /// The base rows, ids on the plan's gene axis; `None` when no table was given.
    pub base: Option<ge::PresetRows>,
    /// One entry per non-base track the table had rows on.
    pub offsets: Vec<ge::PresetOffsets>,
    pub carried: Option<CarriedRows>,
}

/// A source name onto the row grammar: a bare name is the gene's
/// `count/spliced` row; a name already in the grammar is itself.
pub(crate) fn row_name_of(name: &str) -> Box<str> {
    if parse_feature_row(name).is_some() {
        name.into()
    } else {
        feature_row(name, COUNT, SPLICED, None)
    }
}

pub(crate) fn resolve_gem_preset(
    resolved: Option<(&str, PresetMode)>,
    feature_names: &[Box<str>],
    plan: &TrackPlan,
) -> anyhow::Result<GemPreset> {
    let Some((prefix, mode)) = resolved else {
        return Ok(GemPreset {
            base: None,
            offsets: Vec::new(),
            carried: None,
        });
    };
    let flag = crate::feature_embedding_args::flag_name(mode);
    let kind = ge::FeatureNameKind::Gene { delim: '_' };
    let (rows, carried) = crate::feature_preset::load_preset_rows(
        prefix,
        mode,
        feature_names,
        &kind,
        Some(&row_name_of),
    )?;
    let h = rows.width();

    let mut base_ids: Vec<u32> = Vec::new();
    let mut base_rows: Vec<f32> = Vec::new();
    let mut base_index: FxHashMap<u32, usize> = FxHashMap::default();
    let mut by_track: BTreeMap<u32, Vec<(u32, usize)>> = BTreeMap::new();
    for (i, &row) in rows.ids.iter().enumerate() {
        let r = row as usize;
        let (t, g) = (plan.row_track[r], plan.row_gene[r]);
        if plan.base_rows[r] {
            base_index.insert(g, base_ids.len());
            base_ids.push(g);
            base_rows.extend_from_slice(&rows.rows[i * h..(i + 1) * h]);
        } else {
            by_track.entry(t).or_default().push((g, i));
        }
    }
    anyhow::ensure!(
        !base_ids.is_empty(),
        "{flag} {prefix}: no `count/spliced` row of this axis has a row in the table \
         (a bare gene name is read as that row)"
    );
    let mut offsets: Vec<ge::PresetOffsets> = Vec::new();
    let mut n_without_base = 0usize;
    for (t, entries) in by_track {
        let mut ids: Vec<u32> = Vec::new();
        let mut delta: Vec<f32> = Vec::new();
        for (g, i) in entries {
            match base_index.get(&g) {
                Some(&j) => {
                    ids.push(g);
                    delta.extend((0..h).map(|k| rows.rows[i * h + k] - base_rows[j * h + k]));
                }
                None => n_without_base += 1,
            }
        }
        if !ids.is_empty() {
            offsets.push(ge::PresetOffsets {
                track: t,
                ids,
                rows: delta,
            });
        }
    }
    if n_without_base > 0 {
        warn!(
            "{flag}: {n_without_base} given track rows belong to genes with no given base row and are \
             ignored: a track's offset is relative to the gene's base row"
        );
    }
    let n_track_rows: usize = offsets.iter().map(|o| o.ids.len()).sum();
    info!(
        "{flag}: {} base rows {}; {n_track_rows} track rows on {} track(s) {}",
        base_ids.len(),
        mode.describe(),
        offsets.len(),
        if matches!(mode, PresetMode::Freeze) {
            "pinned as given"
        } else {
            "as the start of the track's offset"
        }
    );
    Ok(GemPreset {
        base: Some(ge::PresetRows {
            ids: base_ids,
            rows: base_rows,
            mode,
        }),
        offsets,
        carried,
    })
}

#[cfg(test)]
#[path = "preset/tests.rs"]
mod tests;
