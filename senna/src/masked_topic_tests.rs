//! The masked family's argument surface.

use super::MaskedTopicArgs;
use clap::CommandFactory;

#[derive(clap::Parser)]
struct Cli {
    #[command(flatten)]
    args: MaskedTopicArgs,
}

/// `senna predict` tells a residual-trained model that `--adj-method batch`
/// removes its null mismatch, so the flag has to be where a user can find it.
#[test]
fn adj_method_is_listed_in_help() {
    let cmd = Cli::command();
    let arg = cmd
        .get_arguments()
        .find(|a| a.get_id() == "adj_method")
        .expect("the masked family accepts --adj-method");
    assert!(!arg.is_hide_set(), "--adj-method must not be hidden from --help");
}
