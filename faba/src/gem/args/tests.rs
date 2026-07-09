use super::GemArgs;
use clap::Parser;

/// `GemArgs` is an `Args` group, not a `Parser`; wrap it to parse standalone.
#[derive(Parser)]
struct Wrap {
    #[command(flatten)]
    gem: GemArgs,
}

fn parse(argv: &[&str]) -> GemArgs {
    Wrap::try_parse_from(argv).expect("parse").gem
}

#[test]
fn positional_genes_are_space_separated() {
    let a = parse(&["gem", "a.zarr.zip", "b.zarr.zip", "-o", "out"]);
    let g = a.genes().unwrap();
    assert_eq!(g.len(), 2);
    assert_eq!(&*g[0], "a.zarr.zip");
    assert_eq!(&*g[1], "b.zarr.zip");
}

#[test]
fn positional_genes_also_accept_commas() {
    let a = parse(&["gem", "a.zarr.zip,b.zarr.zip", "-o", "out"]);
    assert_eq!(a.genes().unwrap().len(), 2);
}

/// The pre-existing spelling must keep working for scripts already in the wild.
#[test]
fn legacy_genes_flag_still_works() {
    let a = parse(&["gem", "--genes", "a.zarr.zip,b.zarr.zip", "-o", "out"]);
    let g = a.genes().unwrap();
    assert_eq!(g.len(), 2);
    assert_eq!(&*g[1], "b.zarr.zip");
}

/// Silently preferring one form would hide a typo; the intent is ambiguous.
#[test]
fn both_forms_together_is_an_error() {
    let a = parse(&["gem", "a.zarr.zip", "--genes", "b.zarr.zip", "-o", "out"]);
    let err = a.genes().unwrap_err().to_string();
    assert!(err.contains("both positionally"), "{err}");
}

#[test]
fn no_genes_at_all_is_an_error() {
    let a = parse(&["gem", "-o", "out"]);
    let err = a.genes().unwrap_err().to_string();
    assert!(err.contains("no gene matrices given"), "{err}");
}
