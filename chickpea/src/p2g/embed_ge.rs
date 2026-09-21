//! Peak/gene (and cell) embeddings via `graph-embedding-util`.
//!
//! Thin wrapper: typed region+gene graph → ge-util train. No local NCE/PBG.
//! Not implemented yet.

#![allow(dead_code)]

// Keep the workspace dependency live until the trainer is wired.
use graph_embedding_util as _;
