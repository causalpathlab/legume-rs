use super::{LoraSpec, PresetMode};

#[test]
fn only_init_leaves_the_rows_free() {
    assert!(!PresetMode::Init.pins());
    assert!(PresetMode::Freeze.pins());
    assert!(PresetMode::Lora(LoraSpec {
        rank: 2,
        lr_ratio: 1.0,
        ridge: 0.0
    })
    .pins());
}

#[test]
fn the_rank_must_be_strictly_between_zero_and_h() {
    let ok = PresetMode::Lora(LoraSpec {
        rank: 2,
        lr_ratio: 16.0,
        ridge: 0.0,
    });
    assert!(ok.validate(4).is_ok());
    assert!(PresetMode::Lora(LoraSpec {
        rank: 0,
        lr_ratio: 1.0,
        ridge: 0.0
    })
    .validate(4)
    .is_err());
    assert!(PresetMode::Lora(LoraSpec {
        rank: 4,
        lr_ratio: 1.0,
        ridge: 0.0
    })
    .validate(4)
    .is_err());
    assert!(PresetMode::Lora(LoraSpec {
        rank: 1,
        lr_ratio: 0.0,
        ridge: 0.0
    })
    .validate(4)
    .is_err());
    assert!(PresetMode::Freeze.validate(1).is_ok());
    assert_eq!(
        ok.lora(),
        Some(super::LoraSpec {
            rank: 2,
            lr_ratio: 16.0,
            ridge: 0.0
        })
    );
    assert!(PresetMode::Lora(LoraSpec {
        rank: 1,
        lr_ratio: 1.0,
        ridge: -1.0
    })
    .validate(4)
    .is_err());
    assert_eq!(PresetMode::Init.lora(), None);
}

mod lora_args {
    use super::super::{LoraArgs, LoraSpec};
    use clap::Parser;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        lora: LoraArgs,
    }

    fn parse(argv: &[&str]) -> LoraArgs {
        Cli::try_parse_from(std::iter::once("x").chain(argv.iter().copied()))
            .unwrap()
            .lora
    }

    #[test]
    fn nothing_given_is_the_default_spec_and_passes_either_way() {
        let a = parse(&[]);
        assert!(a.refuse_unless_selected(false, "--x").is_ok());
        assert!(a.refuse_unless_selected(true, "--x").is_ok());
        assert_eq!(a.spec(), LoraSpec::default());
    }

    #[test]
    fn a_knob_is_refused_unless_its_selector_is_in_effect_naming_both() {
        let a = parse(&["--lora-ridge", "1"]);
        assert!(a.refuse_unless_selected(true, "--x").is_ok());
        let err = a
            .refuse_unless_selected(false, "--x")
            .unwrap_err()
            .to_string();
        assert!(err.contains("--lora-ridge") && err.contains("--x"), "{err}");
    }

    #[test]
    fn each_knob_overrides_its_default_alone() {
        let a = parse(&["--lora-rank", "4"]);
        assert_eq!(
            a.spec(),
            LoraSpec {
                rank: 4,
                ..LoraSpec::default()
            }
        );
        let b = parse(&["--lora-lr-ratio", "1", "--lora-ridge", "0"]);
        assert_eq!(
            b.spec(),
            LoraSpec {
                lr_ratio: 1.0,
                ridge: 0.0,
                ..LoraSpec::default()
            }
        );
    }
}
