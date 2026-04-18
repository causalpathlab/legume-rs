//! `senna visualize phate` — raw-gene-space PB landmarks laid out
//! with PHATE diffusion embedding over the log1p-CPM PB features.
//! Cells placed by cheap Nyström in proj space.

use super::fit_visualize_common::{finalize_viz, prepare_viz, PhateCliArgs, VisualizeCommonArgs};
use crate::embed_common::*;
use crate::geometry::phate::phate_layout_2d;

#[derive(Args, Debug)]
pub struct VisualizePhateArgs {
    #[clap(flatten)]
    common: VisualizeCommonArgs,

    #[clap(flatten)]
    phate: PhateCliArgs,
}

pub fn fit_visualize_phate(args: &VisualizePhateArgs) -> anyhow::Result<()> {
    let prep = prepare_viz(&args.common)?;
    let pb_coords = phate_layout_2d(&prep.log_cpm_pb, &(&args.phate).into());
    finalize_viz(&args.common, &prep, &pb_coords)
}
