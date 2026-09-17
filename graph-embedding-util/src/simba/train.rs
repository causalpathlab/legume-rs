//! SIMBA's training as a two-type run of the [`crate::fne`] engine: cells and
//! genes are the node types, each expression level is one relation with PBG's
//! `linspace(1, 5)` weight, and the seeded init, fused softmax step, RowAdagrad
//! and stochastic weight decay are the engine's own.

use super::graph::{EdgeList, RelationTable};
use super::{EpochStats, SimbaConfig};
use crate::fne::{self, FneConfig, NodeTypeTable, Relation, TypedEdgeList};
use candle_util::candle_core::Tensor;

/// The two cell/gene node types, in the order the engine lays them out.
pub const CELL_TYPE: &str = "e_cell";
pub const GENE_TYPE: &str = "e_gene";

pub struct TrainOutput {
    /// `[N, D]` on the CPU, detached.
    pub e_cell: Tensor,
    /// `[G, D]` on the CPU, detached.
    pub e_gene: Tensor,
    pub epochs: Vec<EpochStats>,
    pub relations: RelationTable,
    /// The weight decay actually used (auto or pinned).
    pub wd: f64,
    pub n_train_edges: usize,
    pub n_eval_edges: usize,
}

/// Train both tables on `edges` (consumed).
pub fn train(edges: EdgeList, cfg: &SimbaConfig) -> anyhow::Result<TrainOutput> {
    anyhow::ensure!(!edges.is_empty(), "simba: no edges to train on");
    let rel = RelationTable::from_levels(&edges.levels_present());
    let types = NodeTypeTable::new(&[(CELL_TYPE, edges.n_cells), (GENE_TYPE, edges.n_genes)])?;
    let relations = fne::RelationTable::new(
        rel.levels
            .iter()
            .zip(&rel.weights)
            .map(|(&level, &weight)| Relation {
                name: format!("level_{level}").into_boxed_str(),
                lhs_type: 0,
                rhs_type: 1,
                weight,
                undirected: false,
            })
            .collect(),
        &types,
    )?;
    let gene_offset = types.range(1).start;
    let typed = TypedEdgeList {
        rel: edges.level.iter().map(|&l| rel.rel(l) as u16).collect(),
        rhs: edges.gene.iter().map(|&g| g + gene_offset).collect(),
        lhs: edges.cell,
        weight: None,
    };
    let out = fne::train(
        typed,
        types,
        relations,
        &FneConfig {
            dim: cfg.dim,
            epochs: cfg.epochs,
            lr: cfg.lr,
            batch_size: cfg.batch_size,
            num_batch_negs: cfg.num_batch_negs,
            num_uniform_negs: cfg.num_uniform_negs,
            wd: cfg.wd,
            wd_interval: cfg.wd_interval,
            eval_fraction: cfg.eval_fraction,
            // PBG's `int(n · f)` per relation, no floor.
            eval_min_per_relation: 0,
            relation_repeats: Vec::new(),
            // Gene-local indices become global ids behind the cell block.
            preset: cfg.preset_genes.as_ref().map(|p| fne::PresetRows {
                node: p.node.iter().map(|&g| g + gene_offset).collect(),
                rows: p.rows.clone(),
                mode: p.mode,
            }),
            seed: cfg.seed,
            device: cfg.device.clone(),
        },
    )?;
    let (n_cells, n_genes) = (out.node_types.n_nodes(0), out.node_types.n_nodes(1));
    Ok(TrainOutput {
        e_cell: out.embedding.narrow(0, 0, n_cells)?,
        e_gene: out.embedding.narrow(0, n_cells, n_genes)?,
        epochs: out.epochs,
        relations: rel,
        wd: out.wd,
        n_train_edges: out.n_train_edges,
        n_eval_edges: out.n_eval_edges,
    })
}

#[cfg(test)]
#[path = "train_tests.rs"]
mod train_tests;
