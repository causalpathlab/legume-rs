//! Cell-cell interaction counts: a graph over cells paired with the counts
//! those cells came from.
//!
//! [`CellPairs`] borrows a [`SparseIoVec`] and owns the [`KnnGraph`] whose
//! edges are the interacting pairs. Nothing here knows about spatial
//! coordinates — a graph built from tissue positions and one built from an
//! expression embedding are the same object at this level. Front-ends that
//! do carry geometry wrap this and add it themselves (e.g. pinto's
//! `SrtCellPairs`).
//!
//! Originally lived in `pinto/src/util/cell_pairs.rs`; moved here so senna /
//! faba / chickpea can share one cell-pair representation instead of each
//! growing an inline edge list.

use data_beans::sparse_data_visitors::styled_progress_bar;
use data_beans::sparse_io_vector::SparseIoVec;
use indicatif::ParallelProgressIterator;
use matrix_util::knn_graph::KnnGraph;
use matrix_util::parquet::{write_named_table, Column};
use matrix_util::utils::generate_minibatch_intervals;
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;
use std::sync::{Arc, Mutex};

/// A set of interacting pairs over some node indexing, plus optional per-pair
/// weights.
///
/// Deliberately says nothing about *what* the endpoints are. Cells indexed into
/// a [`SparseIoVec`], pseudobulk samples indexed into a dense profile matrix,
/// and the coarser levels produced by [`collapse_pairs`] are all the same shape
/// at this level — which is what lets one collapsing routine serve all of them.
#[derive(Clone, Debug, Default)]
pub struct PairSet {
    pub pairs: Vec<(usize, usize)>,
    /// Per-pair weight, parallel to `pairs` — kNN distance at the cell level,
    /// `None` once collapsed (see [`collapse_pairs`]).
    pub weights: Option<Vec<f32>>,
}

impl PairSet {
    pub fn new(pairs: Vec<(usize, usize)>) -> Self {
        Self {
            pairs,
            weights: None,
        }
    }

    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// See [`collapse_pairs`].
    pub fn collapse_by(&self, node_labels: &[usize]) -> (PairSet, Vec<usize>) {
        collapse_pairs(&self.pairs, node_labels)
    }
}

/// Collapse pairs by node label: every pair whose endpoints carry the same
/// unordered label pair becomes one coarse pair.
///
/// Returns the coarse set and `fine_to_coarse[e]`, the coarse index of fine
/// pair `e`. Coarse ids are handed out in first-seen order, so the result is
/// deterministic in the input order and independent of hasher iteration.
///
/// Pairs whose endpoints share a label collapse to a self-loop `(l, l)`; that
/// is deliberate — a within-group interaction is still an interaction.
///
/// Weights are dropped rather than aggregated: summing distances, averaging
/// them, and counting multiplicity are all defensible and the caller knows
/// which one it wants.
///
/// This is the pair-level analogue of the column-level collapsing in
/// [`crate::collapse_data`], and it composes: the coarse [`PairSet`] can be
/// collapsed again to build a multi-level hierarchy.
pub fn collapse_pairs(pairs: &[(usize, usize)], node_labels: &[usize]) -> (PairSet, Vec<usize>) {
    let mut key_to_coarse: HashMap<(usize, usize), usize> = HashMap::default();
    let mut coarse: Vec<(usize, usize)> = Vec::new();
    let mut fine_to_coarse = Vec::with_capacity(pairs.len());

    for &(i, j) in pairs {
        let (li, lj) = (node_labels[i], node_labels[j]);
        let key = (li.min(lj), li.max(lj));
        let c = *key_to_coarse.entry(key).or_insert_with(|| {
            let next = coarse.len();
            coarse.push(key);
            next
        });
        fine_to_coarse.push(c);
    }

    (PairSet::new(coarse), fine_to_coarse)
}

/// Interacting cell pairs paired with the counts their endpoints index into.
///
/// Borrows the pair list rather than owning it, so it costs nothing to build
/// over a [`KnnGraph`]'s edges (see [`CellPairs::from_graph`]) or over a
/// [`PairSet`] returned by [`collapse_pairs`]. It holds no graph, because
/// nothing here needs adjacency — callers that do keep the graph themselves.
pub struct CellPairs<'a> {
    /// Counts the endpoints index into: features × cells.
    pub data: &'a SparseIoVec,
    pairs: &'a [(usize, usize)],
    weights: Option<&'a [f32]>,
}

impl<'a> CellPairs<'a> {
    pub fn new(data: &'a SparseIoVec, pairs: &'a [(usize, usize)]) -> Self {
        Self {
            data,
            pairs,
            weights: None,
        }
    }

    /// As [`Self::new`], plus a per-pair weight written as the `distance`
    /// column by [`Self::to_parquet`].
    pub fn with_weights(
        data: &'a SparseIoVec,
        pairs: &'a [(usize, usize)],
        weights: &'a [f32],
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            weights.len() == pairs.len(),
            "{} weights for {} pairs",
            weights.len(),
            pairs.len()
        );
        Ok(Self {
            data,
            pairs,
            weights: Some(weights),
        })
    }

    /// Take a kNN graph's edges and distances. The graph stays with the caller.
    pub fn from_graph(data: &'a SparseIoVec, graph: &'a KnnGraph) -> Self {
        Self {
            data,
            pairs: &graph.edges,
            weights: Some(&graph.distances),
        }
    }

    /// Interacting cell pairs `(left, right)`, canonically `left < right`.
    pub fn pairs(&self) -> &[(usize, usize)] {
        self.pairs
    }

    /// Per-pair weights, parallel to [`Self::pairs`], when the caller supplied any.
    pub fn weights(&self) -> Option<&[f32]> {
        self.weights
    }

    pub fn num_pairs(&self) -> usize {
        self.pairs.len()
    }

    pub fn num_features(&self) -> usize {
        self.data.num_rows()
    }

    /// visit cell pairs by regular-sized block
    ///
    /// A visitor function takes
    /// - `(lb,ub)` `(usize,usize)`
    /// - data itself
    /// - `shared_input`
    /// - `shared_out` (`Arc(Mutex())`)
    pub fn visit_pairs_by_block<Visitor, SharedIn, SharedOut>(
        &self,
        visitor: &Visitor,
        shared_in: &SharedIn,
        shared_out: &mut SharedOut,
        block_size: Option<usize>,
    ) -> anyhow::Result<()>
    where
        Visitor: Fn(
                (usize, usize),
                &CellPairs,
                &SharedIn,
                Arc<Mutex<&mut SharedOut>>,
            ) -> anyhow::Result<()>
            + Sync
            + Send,
        SharedIn: Sync + Send + ?Sized,
        SharedOut: Sync + Send,
    {
        let ntot = self.num_pairs();
        let jobs = generate_minibatch_intervals(ntot, self.num_features(), block_size);
        let arc_shared_out = Arc::new(Mutex::new(shared_out));

        let prog_bar = styled_progress_bar(jobs.len() as u64, "blocks");
        let result = jobs
            .par_iter()
            .progress_with(prog_bar.clone())
            .map(|&(lb, ub)| -> anyhow::Result<()> {
                visitor((lb, ub), self, shared_in, arc_shared_out.clone())
            })
            .collect::<anyhow::Result<()>>();
        prog_bar.finish_and_clear();
        result
    }

    /// Write all the cell pairs into a `.parquet` file.
    ///
    /// Columns are `left_cell`, `right_cell`, then `extra` in the order
    /// given, then `distance` when weights were supplied. `extra` lets a
    /// caller splice in per-pair values — endpoint coordinates, say — without
    /// this module having to know what they mean; each must carry one value
    /// per pair.
    ///
    /// * `file_path`: destination file name (try to include a recognizable
    ///   extension in the end, e.g., `.parquet`)
    pub fn to_parquet(
        &self,
        file_path: &str,
        extra: &[(Box<str>, Column<'_>)],
    ) -> anyhow::Result<()> {
        let num_pairs = self.num_pairs();
        let cell_names = self.data.column_names()?;

        let left: Vec<Box<str>> = self
            .pairs()
            .iter()
            .map(|&(left, _)| cell_names[left].clone())
            .collect();
        let right: Vec<Box<str>> = self
            .pairs()
            .iter()
            .map(|&(_, right)| cell_names[right].clone())
            .collect();

        let mut columns: Vec<(Box<str>, Column<'_>)> = vec![
            ("left_cell".into(), Column::Str(&left)),
            ("right_cell".into(), Column::Str(&right)),
        ];

        for (name, col) in extra {
            // Name the offending column; parquet's own row-count check fires
            // later and cannot say which one is short.
            let len = match col {
                Column::Str(d) => d.len(),
                Column::F32(d) => d.len(),
                Column::I32(d) => d.len(),
            };
            if len != num_pairs {
                return Err(anyhow::anyhow!(
                    "column `{}` carries {} values for {} pairs",
                    name,
                    len,
                    num_pairs
                ));
            }
            columns.push((
                name.clone(),
                match col {
                    Column::Str(d) => Column::Str(d),
                    Column::F32(d) => Column::F32(d),
                    Column::I32(d) => Column::I32(d),
                },
            ));
        }

        if let Some(w) = self.weights {
            columns.push(("distance".into(), Column::F32(w)));
        }

        let row_names: Vec<Box<str>> = (0..num_pairs)
            .map(|i| i.to_string().into_boxed_str())
            .collect();

        write_named_table(file_path, "cell_pair", &row_names, &columns)
    }
}
