pub mod counter;
pub mod pipeline;
/// The `faba genes` run. Binary entry: [`run::run_gene_count`].
pub mod run;
pub mod splice;

pub use counter::collect_all_gene_counts;
