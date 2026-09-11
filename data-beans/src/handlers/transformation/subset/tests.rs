//! Every subcommand that writes a backend produces the same thing by default.
//!
//! The four handlers here stage a working `.zarr` directory and re-zip it at the
//! end, so before this they silently left a directory where every `from-*`
//! subcommand leaves a `.zarr.zip`. A pipeline that feeds one subcommand's
//! output to the next then breaks on the path alone.

use super::super::{ReorderRowsArgs, RunSqueezeArgs, SubsetColumnsArgs, SubsetRowsArgs};
use clap::Parser;

#[derive(Parser)]
struct Cols {
    #[command(flatten)]
    args: SubsetColumnsArgs,
}
#[derive(Parser)]
struct Rows {
    #[command(flatten)]
    args: SubsetRowsArgs,
}
#[derive(Parser)]
struct Reorder {
    #[command(flatten)]
    args: ReorderRowsArgs,
}
#[derive(Parser)]
struct Squeeze {
    #[command(flatten)]
    args: RunSqueezeArgs,
}

#[test]
fn an_edit_handler_zips_its_output_like_every_builder_does() {
    assert!(Cols::parse_from(["x", "in.zarr.zip", "-o", "out"]).args.zip);
    assert!(Rows::parse_from(["x", "in.zarr.zip", "-o", "out"]).args.zip);
    assert!(
        Reorder::parse_from(["x", "in.zarr.zip", "-r", "rows.tsv", "-o", "out"])
            .args
            .zip
    );
    assert!(Squeeze::parse_from(["x", "in.zarr.zip"]).args.zip);
}

#[test]
fn no_zip_still_asks_for_a_directory() {
    assert!(
        !Cols::parse_from(["x", "in.zarr.zip", "-o", "out", "--no-zip"])
            .args
            .zip
    );
    assert!(
        !Rows::parse_from(["x", "in.zarr.zip", "-o", "out", "--no-zip"])
            .args
            .zip
    );
}
