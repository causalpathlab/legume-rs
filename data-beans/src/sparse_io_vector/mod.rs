#![allow(dead_code)]

use crate::sparse_io::*;

use indicatif::{ParallelProgressIterator, ProgressBar, ProgressIterator, ProgressStyle};
use log::info;
use matrix_util::knn_match::ColumnDict;
use matrix_util::knn_match::MakeVecPoint;
use matrix_util::traits::*;
use matrix_util::utils::*;
use rayon::prelude::*;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::borrow::Cow;
use std::ops::Index;
use std::sync::Arc;

// `impl SparseIoVec` is split across sibling modules by concern. Inherent
// impls compose across modules, so every method stays callable at the same
// path; siblings see the struct's private fields as descendant modules and
// inherit this module's imports via `use super::*`.
mod batch;
mod groups;
mod matched;
mod push;
mod read;

/// Where a single global cell's nonzeros come from in one of the
/// underlying [`SparseIo`] backends. Under
/// [`ColumnAlignment::Disjoint`] each global column has exactly one
/// `BackendLocation`. Under [`ColumnAlignment::Union`] a global column
/// can carry one entry per backend that observed the cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackendLocation {
    /// Index into [`SparseIoVec::data_vec`].
    pub backend: u32,
    /// Local column index within that backend.
    pub local_col: u32,
}

type SparseData = dyn SparseIo<IndexIter = Vec<usize>>;

/// Optional canonicalizer applied to every backend row name during
/// `push`. Two raw names that produce the same canonical form are
/// treated as the same row. The displayed `row_names_by_global` stores
/// the canonical form, so downstream readers see a single consistent
/// label for each row across all backends. Default = `None` preserves
/// exact-match behavior.
pub type RowNameCanonicalizer = Arc<dyn Fn(&str) -> Box<str> + Send + Sync>;

/// Optional canonicalizer applied to every backend *column* (barcode)
/// name during `push` under [`ColumnAlignment::Union`]. Mirror of
/// [`RowNameCanonicalizer`]. Two raw barcodes whose canonical forms
/// match are treated as the same biological cell — backends share a
/// single global column id, and `read_columns_*` merges their nonzeros
/// into one output column. Default = `None` preserves exact-match
/// behavior.
pub type ColumnNameCanonicalizer = Arc<dyn Fn(&str) -> Box<str> + Send + Sync>;

pub struct SparseIoVec {
    data_vec: Vec<Arc<SparseData>>,
    /// Per global column → one or more [`BackendLocation`] entries.
    /// Under [`ColumnAlignment::Disjoint`] each inner `Vec` always
    /// has length 1. Under [`ColumnAlignment::Union`] a global
    /// column can be observed by multiple backends — one entry per
    /// backend that contributes triplets for that cell.
    col_to_data: Vec<Vec<BackendLocation>>,
    data_to_cols: HashMap<usize, Vec<usize>>,
    offset: usize,
    // Row-name alignment across backends. Each backend may have a
    // different subset/ordering of rows; we keep only the intersection
    // and remap local indices through `data_local_to_global_row` and
    // `global_to_compact_row` at read time.
    row_canonicalizer: Option<RowNameCanonicalizer>,
    row_name_position: HashMap<Box<str>, usize>,
    row_names_by_global: Vec<Box<str>>,
    data_local_to_global_row: Vec<Vec<usize>>,
    /// Per-backend inverse of `data_local_to_global_row[didx]`.
    /// Built once at `push` time, never mutated thereafter, so row reads
    /// don't have to rebuild it per call.
    data_global_to_local_row: Vec<HashMap<usize, usize>>,
    /// `true` for datasets where `local_to_global` is non-injective —
    /// i.e. the canonicalizer collapsed two or more local rows to the
    /// same global row. Read paths use this to switch from a fast
    /// pass-through emit to a `(row, col) → sum` merge so the resulting
    /// COO doesn't carry duplicate entries (which break CSC builders).
    data_has_intra_row_merges: Vec<bool>,
    row_count_by_global: Vec<usize>,
    global_to_compact_row: Vec<Option<usize>>,
    /// Inverse of `global_to_compact_row` restricted to compact rows.
    /// `compact_to_global_row[c]` is the raw-global index for compact `c`.
    /// Kept in sync with `global_to_compact_row` whenever it changes.
    compact_to_global_row: Vec<usize>,
    column_names_with_data_tag: Vec<Box<str>>,
    /// `Union` mode only: canonical barcode → global column id. Populated
    /// at push time so subsequent pushes can match cells across backends.
    /// Unused (empty) under `Disjoint`.
    col_name_position: HashMap<Box<str>, u32>,
    col_to_group: Option<HashMap<usize, usize>>,
    group_to_cols: Option<Vec<Vec<usize>>>,
    group_keys: Option<Vec<Box<str>>>,
    batch_knn_lookup: Option<Vec<ColumnDict<usize>>>,
    col_to_batch: Option<Vec<usize>>,
    batch_to_cols: Option<Vec<Vec<usize>>>,
    batch_idx_to_name: Option<Vec<Box<str>>>,
    between_batch_proximity: Option<Vec<Vec<usize>>>,
    cached_num_rows: usize,
    cached_num_columns: usize,
    /// How row names align across pushed backends. Default
    /// [`RowAlignment::Union`] keeps every row from any backend, with
    /// cells from a backend that doesn't contain row `g` implicitly
    /// observing zero at that position — used for multi-modal data
    /// where features (e.g. peaks vs genes) are disjoint.
    /// [`RowAlignment::Intersect`] reverts to the historical
    /// "common rows only" semantics.
    row_alignment: RowAlignment,
    /// How column (cell) names align across pushed backends. Default
    /// [`ColumnAlignment::Disjoint`] preserves the historical
    /// concatenate-cells semantics, with `@<basename>` suffixing when
    /// multiple files are loaded. [`ColumnAlignment::Union`] matches
    /// cells by canonical barcode across backends and lets one global
    /// column carry triplets from multiple backends — the foundation
    /// for patchy multi-modal (multiome) integration.
    column_alignment: ColumnAlignment,
    /// Optional canonicalizer for barcode matching under `Union` mode.
    /// See [`ColumnNameCanonicalizer`].
    column_canonicalizer: Option<ColumnNameCanonicalizer>,
    /// Optional per-backend feature-name (row) suffix, indexed by push
    /// order (`didx`). When set, every row of backend `b` is renamed
    /// `{canon(row)}/{suffix[b]}` *after* canonicalization — so the
    /// canonical gene/locus rule still applies to the bare name, and the
    /// suffix only namespaces it onto a modality-specific row. Two
    /// backends sharing a raw name (e.g. spliced vs unspliced `TSPAN6`)
    /// thus stay separate, while the same name + same suffix (same
    /// modality across donors) still merges. `None` = no suffixing.
    per_backend_row_suffix: Option<Vec<Box<str>>>,
}

/// Strategy for aligning row names across multiple pushed backends.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RowAlignment {
    /// Keep every row from any backend; non-observing backends emit zero
    /// at that position. The default — strictly more permissive than
    /// intersection: it reduces to the same result when all files share
    /// their row set, and enables multi-modal load (e.g. peaks ∪ genes)
    /// when they don't.
    #[default]
    Union,
    /// Keep only rows present in every backend. Opt-in for callers that
    /// want strict single-modality semantics.
    Intersect,
}

/// Strategy for aligning column (cell) names across multiple pushed
/// backends.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ColumnAlignment {
    /// Today's behavior. Each push appends `ncol_data` brand-new global
    /// cells; raw barcodes get `@<basename>` suffixed for disambiguation
    /// when more than one backend is pushed. Two backends naming the
    /// same biological cell stay disjoint.
    #[default]
    Disjoint,
    /// Match each pushed column's canonical barcode against existing
    /// global cells. New barcodes extend the cell pool; matched barcodes
    /// share one global column across backends (the column carries
    /// triplets from every backend that observes the cell). No
    /// `@<basename>` suffix. Foundation for patchy multi-modal
    /// (multiome) integration.
    Union,
}

pub struct TripletsMatched {
    pub shape: (usize, usize),
    pub triplets: Vec<(u64, u64, f32)>,
    pub source_columns: Vec<usize>,
    pub matched_columns: Vec<usize>,
    pub distances: Vec<f32>,
}

impl Index<usize> for SparseIoVec {
    type Output = Arc<SparseData>;
    fn index(&self, idx: usize) -> &Self::Output {
        &self.data_vec[idx]
    }
}

impl Default for SparseIoVec {
    /// an empty sparse io vector for horizontal data integration
    fn default() -> Self {
        Self::new()
    }
}

impl SparseIoVec {
    /// an empty sparse io vector for horizontal data integration
    pub fn new() -> Self {
        Self {
            data_vec: vec![],
            col_to_data: vec![],
            data_to_cols: HashMap::default(),
            offset: 0,
            row_canonicalizer: None,
            row_name_position: HashMap::default(),
            row_names_by_global: vec![],
            data_local_to_global_row: vec![],
            data_global_to_local_row: vec![],
            data_has_intra_row_merges: vec![],
            row_count_by_global: vec![],
            global_to_compact_row: vec![],
            compact_to_global_row: vec![],
            column_names_with_data_tag: vec![],
            col_name_position: HashMap::default(),
            col_to_group: None,
            group_to_cols: None,
            group_keys: None,
            batch_knn_lookup: None,
            col_to_batch: None,
            batch_to_cols: None,
            batch_idx_to_name: None,
            between_batch_proximity: None,
            cached_num_rows: 0,
            cached_num_columns: 0,
            row_alignment: RowAlignment::default(),
            column_alignment: ColumnAlignment::default(),
            column_canonicalizer: None,
            per_backend_row_suffix: None,
        }
    }

    /// Switch row-name alignment between intersection (default) and union.
    /// Must be called BEFORE any push — the row-mapping recompute uses
    /// the current value of `row_alignment`. Errors if any backend has
    /// already been added.
    pub fn with_row_alignment(mut self, mode: RowAlignment) -> anyhow::Result<Self> {
        anyhow::ensure!(
            self.data_vec.is_empty(),
            "row alignment must be set before any push"
        );
        self.row_alignment = mode;
        Ok(self)
    }

    /// Install a row-name canonicalizer for fuzzy cross-backend row
    /// alignment. Must be called BEFORE the first [`push`](Self::push) —
    /// returns an error if any backend has already been added (the
    /// existing `row_names_by_global` would otherwise mix raw and
    /// canonicalized forms).
    ///
    /// Typical use: pass a `GeneIndexResolver`-style canonicalizer so
    /// `ENSG00000000003_TSPAN6` (file A) and `TSPAN6` (file B) collapse
    /// to a single row, instead of being silently dropped from the
    /// shared-row intersection.
    pub fn with_row_canonicalizer(
        mut self,
        canon: impl Fn(&str) -> Box<str> + Send + Sync + 'static,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            self.data_vec.is_empty(),
            "row canonicalizer must be set before any push"
        );
        self.row_canonicalizer = Some(Arc::new(canon));
        Ok(self)
    }

    /// Install a per-backend feature-name suffix (one entry per backend,
    /// in push order). Each backend `b`'s rows are renamed
    /// `{canon(row)}/{suffix[b]}`, so files sharing raw feature names stay
    /// on separate rows unless they also share the suffix (same modality).
    /// Must be called BEFORE the first [`push`](Self::push). The vec length
    /// must match the number of backends that will be pushed; `push` errors
    /// if `didx` is out of range.
    pub fn with_per_backend_row_suffix(mut self, suffix: Vec<Box<str>>) -> anyhow::Result<Self> {
        anyhow::ensure!(
            self.data_vec.is_empty(),
            "per-backend row suffix must be set before any push"
        );
        self.per_backend_row_suffix = Some(suffix);
        Ok(self)
    }

    /// Switch column (cell) alignment between disjoint concatenation
    /// (default) and barcode-keyed union. Must be called BEFORE any
    /// push — the push branches off the current value. Errors if any
    /// backend has already been added.
    pub fn with_column_alignment(mut self, mode: ColumnAlignment) -> anyhow::Result<Self> {
        anyhow::ensure!(
            self.data_vec.is_empty(),
            "column alignment must be set before any push"
        );
        self.column_alignment = mode;
        Ok(self)
    }

    /// Install a column-name canonicalizer for fuzzy cross-backend
    /// barcode matching under [`ColumnAlignment::Union`]. Must be
    /// called BEFORE the first [`push`](Self::push) — returns an error
    /// if any backend has already been added. Has no effect under
    /// [`ColumnAlignment::Disjoint`] (barcodes are never compared
    /// across backends in that mode).
    pub fn with_column_canonicalizer(
        mut self,
        canon: impl Fn(&str) -> Box<str> + Send + Sync + 'static,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            self.data_vec.is_empty(),
            "column canonicalizer must be set before any push"
        );
        self.column_canonicalizer = Some(Arc::new(canon));
        Ok(self)
    }

    /// number of data sets
    pub fn len(&self) -> usize {
        self.data_vec.len()
    }

    /// check if the vector is empty
    pub fn is_empty(&self) -> bool {
        self.data_vec.is_empty()
    }

    pub fn num_rows(&self) -> usize {
        self.cached_num_rows
    }

    /// Number of canonical rows observed by **at least** `k` of the
    /// pushed backends. Useful for detecting multi-modal-shaped inputs
    /// (`num_rows_in_at_least(n_backends)` is the strict intersection
    /// size; comparing it against per-backend row counts reveals how
    /// disjoint the feature axes are).
    pub fn num_rows_in_at_least(&self, k: usize) -> usize {
        self.row_count_by_global.iter().filter(|&&c| c >= k).count()
    }

    /// Current column-alignment mode. Mirrors [`Self::row_alignment`].
    pub fn column_alignment(&self) -> ColumnAlignment {
        self.column_alignment
    }

    pub fn num_non_zeros(&self) -> anyhow::Result<usize> {
        let mut ret = 0;
        for dat in self.data_vec.iter() {
            let nnz = dat
                .num_non_zeros()
                .ok_or(anyhow::anyhow!("can't figure out the number of non-zeros"))?;
            ret += nnz;
        }
        Ok(ret)
    }

    /// total number of columns across all data files
    pub fn num_columns(&self) -> usize {
        self.cached_num_columns
    }

    /// Structural clone for an independent projection / collapse pass.
    ///
    /// The backend cannot derive `Clone` because `batch_knn_lookup` holds
    /// non-`Clone` HNSW indices. This copies the data handles (matrices are
    /// `Arc`-shared, so only the index `Vec`s/`HashMap`s are duplicated —
    /// cheap) and the row/column alignment state, while **dropping** the
    /// derived batch / group / HNSW caches. Those caches are re-registered
    /// from scratch by `register_batch_membership` / the collapse, so a fresh
    /// clone is the correct pre-collapse state.
    ///
    /// Intended use: `let mut spliced = vec.clone_for_collapse();
    /// spliced.mask_rows(&spliced_keep)?;` — gives a spliced-only view that
    /// drives RP + collapse + refinement without disturbing the full backend
    /// (still needed at all rows for per-modality aggregation).
    pub fn clone_for_collapse(&self) -> Self {
        Self {
            data_vec: self.data_vec.clone(),
            col_to_data: self.col_to_data.clone(),
            data_to_cols: self.data_to_cols.clone(),
            offset: self.offset,
            row_canonicalizer: self.row_canonicalizer.clone(),
            row_name_position: self.row_name_position.clone(),
            row_names_by_global: self.row_names_by_global.clone(),
            data_local_to_global_row: self.data_local_to_global_row.clone(),
            data_global_to_local_row: self.data_global_to_local_row.clone(),
            data_has_intra_row_merges: self.data_has_intra_row_merges.clone(),
            row_count_by_global: self.row_count_by_global.clone(),
            global_to_compact_row: self.global_to_compact_row.clone(),
            compact_to_global_row: self.compact_to_global_row.clone(),
            column_names_with_data_tag: self.column_names_with_data_tag.clone(),
            col_name_position: self.col_name_position.clone(),
            // Derived caches dropped — re-registered by projection / collapse.
            col_to_group: None,
            group_to_cols: None,
            group_keys: None,
            batch_knn_lookup: None,
            col_to_batch: None,
            batch_to_cols: None,
            batch_idx_to_name: None,
            between_batch_proximity: None,
            cached_num_rows: self.cached_num_rows,
            cached_num_columns: self.cached_num_columns,
            row_alignment: self.row_alignment,
            column_alignment: self.column_alignment,
            column_canonicalizer: self.column_canonicalizer.clone(),
            per_backend_row_suffix: self.per_backend_row_suffix.clone(),
        }
    }

    /// Exclude rows (genes) from the working set. `keep[compact_row]`
    /// is `true` for rows to keep, `false` for rows to exclude.
    /// The compact row indices are renumbered after filtering.
    /// This affects all downstream operations (projection, collapse,
    /// training, inference).
    pub fn mask_rows(&mut self, keep: &[bool]) -> anyhow::Result<()> {
        let n_compact = self.cached_num_rows;
        if keep.len() != n_compact {
            return Err(anyhow::anyhow!(
                "mask_rows: keep.len()={} != num_rows={}",
                keep.len(),
                n_compact
            ));
        }
        // Build old_compact → new_compact mapping
        let mut old_to_new: Vec<Option<usize>> = vec![None; n_compact];
        let mut next = 0usize;
        for (old, &k) in keep.iter().enumerate() {
            if k {
                old_to_new[old] = Some(next);
                next += 1;
            }
        }
        // Update global_to_compact_row
        for entry in self.global_to_compact_row.iter_mut() {
            *entry = entry.and_then(|old_compact| old_to_new[old_compact]);
        }
        self.cached_num_rows = next;
        self.compact_to_global_row.clear();
        self.compact_to_global_row.resize(next, 0);
        for (g, &c_opt) in self.global_to_compact_row.iter().enumerate() {
            if let Some(c) = c_opt {
                self.compact_to_global_row[c] = g;
            }
        }
        log::info!(
            "mask_rows: {} → {} rows ({} excluded)",
            n_compact,
            next,
            n_compact - next
        );
        Ok(())
    }

    /// Exclude columns (cells) from the working set. `keep[global_col]`
    /// is `true` for cells to keep, `false` to exclude. Global column
    /// indices are renumbered after filtering. This affects all
    /// downstream operations (projection, collapse, training, inference).
    ///
    /// Cell-axis mirror of [`Self::mask_rows`]. MUST be called before
    /// batch/group registration: `register_batch_membership` and group
    /// assignment index by global column id and would be corrupted by a
    /// renumber, so we `debug_assert!` they are unset and defensively
    /// clear them. Dropped cells leave their backend-local columns in
    /// place but unmapped (`usize::MAX` in `data_to_cols`), which the
    /// row-wise read path (`rows_triplets`) skips.
    pub fn mask_columns(&mut self, keep: &[bool]) -> anyhow::Result<()> {
        let n = self.cached_num_columns;
        if keep.len() != n {
            return Err(anyhow::anyhow!(
                "mask_columns: keep.len()={} != num_columns={}",
                keep.len(),
                n
            ));
        }
        debug_assert!(
            self.col_to_batch.is_none() && self.col_to_group.is_none(),
            "mask_columns must be called before batch/group registration"
        );

        // old_global → Some(new_global) | None (dropped)
        let mut old_to_new: Vec<Option<usize>> = vec![None; n];
        let mut next = 0usize;
        for (old, &k) in keep.iter().enumerate() {
            if k {
                old_to_new[old] = Some(next);
                next += 1;
            }
        }

        // Compact col_to_data + column_names_with_data_tag in one pass.
        let mut new_col_to_data: Vec<Vec<BackendLocation>> = Vec::with_capacity(next);
        let mut new_names: Vec<Box<str>> = Vec::with_capacity(next);
        for (old, &k) in keep.iter().enumerate() {
            if k {
                new_col_to_data.push(std::mem::take(&mut self.col_to_data[old]));
                new_names.push(self.column_names_with_data_tag[old].clone());
            }
        }
        self.col_to_data = new_col_to_data;
        self.column_names_with_data_tag = new_names;

        // Remap data_to_cols in place. It is a positional
        // local-col → global-col map per backend (consumed by
        // `rows_triplets` / `take_backend_columns`); dropped cells map to
        // `usize::MAX`, which both consumers skip.
        for cols in self.data_to_cols.values_mut() {
            for c in cols.iter_mut() {
                *c = old_to_new[*c].unwrap_or(usize::MAX);
            }
        }

        // Rebuild the Union-mode canonical-barcode → global lookup
        // (empty / no-op under Disjoint).
        if !self.col_name_position.is_empty() {
            let mut pos: HashMap<Box<str>, u32> =
                HashMap::with_capacity_and_hasher(next, Default::default());
            for (g, name) in self.column_names_with_data_tag.iter().enumerate() {
                let canon: Box<str> = match self.column_canonicalizer.as_ref() {
                    Some(c) => c(name),
                    None => name.clone(),
                };
                let g_u32: u32 = g
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("global col overflows u32"))?;
                pos.insert(canon, g_u32);
            }
            self.col_name_position = pos;
        }

        self.offset = next;
        self.cached_num_columns = next;

        // Any cell-indexed membership is now stale (asserted unset above).
        self.col_to_batch = None;
        self.batch_to_cols = None;
        self.batch_idx_to_name = None;
        self.batch_knn_lookup = None;
        self.between_batch_proximity = None;
        self.col_to_group = None;
        self.group_to_cols = None;
        self.group_keys = None;

        log::info!(
            "mask_columns: {} → {} cells ({} excluded)",
            n,
            next,
            n - next
        );
        Ok(())
    }

    pub fn row_names(&self) -> anyhow::Result<Vec<Box<str>>> {
        let ntot = self.num_rows();
        let mut ret = vec![Box::from(""); ntot];
        for (raw_global, name) in self.row_names_by_global.iter().enumerate() {
            if let Some(compact) = self
                .global_to_compact_row
                .get(raw_global)
                .copied()
                .flatten()
            {
                ret[compact] = name.clone();
            }
        }
        Ok(ret)
    }
}
