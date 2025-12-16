use crate::common::*;

#[derive(Parser, Debug)]
pub struct SimColliderArgs {
    #[arg(
        short = 'r',
        required = true,
        help = "number of rows/genes",
        long_help = "Number of rows/genes/features"
    )]
    n_genes: usize,

    #[arg(
        short = 'c',
        required = true,
        help = "number of columns/cells",
        long_help = "Number of columns/cells"
    )]
    n_cells: usize,
}

fn run_sim_collider_data(args: &SimColliderArgs) -> anyhow::Result<()> {

    unimplemented!("");

    Ok(())
}
