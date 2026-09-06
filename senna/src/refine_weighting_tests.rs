use super::*;
use clap::Parser;
use data_beans_alg::collapse_data::BelowEdge;

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    collapse: CollapseArgs,
}

#[test]
fn the_tree_is_the_default_and_marginal_switches_it_off() {
    let cli = Cli::try_parse_from(["senna"]).unwrap();
    let params = cli.collapse.pb_tree_params().expect("refined by default");
    assert!((params.edge_margin - 0.10).abs() < 1e-6);
    assert_eq!(params.below_edge, BelowEdge::Keep);
    assert_eq!(params.max_genes, 2000);
    assert_eq!(params.min_cells_to_split, 8);
    let reassign = params.reassign_cells.expect("cells are reassigned first");
    assert_eq!(reassign.num_gibbs, 3);

    let cli = Cli::try_parse_from(["senna", "--pb-tree", "marginal"]).unwrap();
    assert!(cli.collapse.pb_tree_params().is_none());
}
