use super::*;
use clap::Parser;

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    modules: GeneModuleArgs,
}

fn parse(args: &[&str]) -> GeneModuleArgs {
    Cli::parse_from(std::iter::once("x").chain(args.iter().copied())).modules
}

#[test]
fn default_on_cli_trains_modules_unless_told_otherwise() {
    let cfg = parse(&[]).resolve(Some(128)).unwrap().unwrap();
    assert_eq!(cfg.n_modules, 128);
    assert!(parse(&["--no-gene-modules"])
        .resolve(Some(128))
        .unwrap()
        .is_none());
    let cfg = parse(&["--gene-modules", "32"])
        .resolve(Some(128))
        .unwrap()
        .unwrap();
    assert_eq!(cfg.n_modules, 32);
}

#[test]
fn opt_in_cli_stays_off_without_the_flag() {
    assert!(parse(&[]).resolve(None).unwrap().is_none());
    assert_eq!(
        parse(&["--gene-modules", "16"])
            .resolve(None)
            .unwrap()
            .unwrap()
            .n_modules,
        16
    );
}

#[test]
fn the_two_flags_conflict() {
    assert!(Cli::try_parse_from(["x", "--gene-modules", "8", "--no-gene-modules"]).is_err());
}

#[test]
fn validation_rejects_bad_knobs() {
    assert!(parse(&["--gene-modules", "1"]).resolve(None).is_err());
    assert!(parse(&["--gene-modules", "8", "--gene-dropout", "1.0"])
        .resolve(None)
        .is_err());
}

#[test]
fn the_uniformity_knobs_default_to_off() {
    let cfg = parse(&[]).resolve(Some(128)).unwrap().unwrap();
    assert_eq!(cfg.lambda_uniform, 0.0);
    assert_eq!(cfg.uniform_temp, 2.0);
    let cfg = parse(&["--module-uniform", "0.1"])
        .resolve(Some(128))
        .unwrap()
        .unwrap();
    assert_eq!(cfg.lambda_uniform, 0.1);
}

#[test]
fn negative_uniformity_and_a_zero_temperature_are_rejected() {
    assert!(parse(&["--module-uniform=-0.1"])
        .resolve(Some(128))
        .is_err());
    assert!(parse(&["--module-uniform-temp", "0"])
        .resolve(Some(128))
        .is_err());
}
