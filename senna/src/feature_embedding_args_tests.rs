use super::{flag_name, FeatureEmbeddingArgs};
use clap::Parser;
use graph_embedding_util::{LoraSpec, PresetMode};

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    fe: FeatureEmbeddingArgs,
}

fn parse(argv: &[&str]) -> Result<FeatureEmbeddingArgs, clap::Error> {
    Cli::try_parse_from(std::iter::once("x").chain(argv.iter().copied())).map(|c| c.fe)
}

#[test]
fn each_flag_resolves_to_its_mode_and_lora_carries_its_knobs() {
    assert_eq!(parse(&[]).unwrap().resolve().unwrap(), None);
    assert_eq!(
        parse(&["--freeze-feature-embedding", "a"])
            .unwrap()
            .resolve()
            .unwrap(),
        Some(("a", PresetMode::Freeze))
    );
    assert_eq!(
        parse(&["--init-feature-embedding", "b"])
            .unwrap()
            .resolve()
            .unwrap(),
        Some(("b", PresetMode::Init))
    );
    assert_eq!(
        parse(&["--lora-feature-embedding", "c"])
            .unwrap()
            .resolve()
            .unwrap(),
        Some((
            "c",
            PresetMode::Lora(LoraSpec {
                rank: 16,
                lr_ratio: 4.0,
                ridge: 0.05
            })
        ))
    );
    assert_eq!(
        parse(&[
            "--lora-feature-embedding",
            "c",
            "--lora-rank",
            "4",
            "--lora-lr-ratio",
            "1"
        ])
        .unwrap()
        .resolve()
        .unwrap(),
        Some((
            "c",
            PresetMode::Lora(LoraSpec {
                rank: 4,
                lr_ratio: 1.0,
                ridge: 0.05
            })
        ))
    );
}

#[test]
fn the_three_flags_exclude_each_other_and_the_knobs_need_lora() {
    assert!(parse(&[
        "--freeze-feature-embedding",
        "a",
        "--init-feature-embedding",
        "b"
    ])
    .is_err());
    assert!(parse(&[
        "--freeze-feature-embedding",
        "a",
        "--lora-feature-embedding",
        "b"
    ])
    .is_err());
    assert!(parse(&[
        "--init-feature-embedding",
        "a",
        "--lora-feature-embedding",
        "b"
    ])
    .is_err());
    // The knobs parse anywhere (the group is shared with models that select
    // LoRA another way) and are refused at resolution without the flag.
    for knob in [
        ["--lora-rank", "4"],
        ["--lora-lr-ratio", "2"],
        ["--lora-ridge", "1"],
    ] {
        let err = parse(&knob).unwrap().resolve().unwrap_err().to_string();
        assert!(
            err.contains(knob[0]) && err.contains("--lora-feature-embedding"),
            "{err}"
        );
        assert!(
            parse(&[&["--freeze-feature-embedding", "a"][..], &knob[..]].concat())
                .unwrap()
                .resolve()
                .is_err()
        );
    }
    assert_eq!(
        parse(&["--lora-feature-embedding", "c", "--lora-ridge", "2.5"])
            .unwrap()
            .resolve()
            .unwrap()
            .and_then(|(_, m)| m.lora())
            .map(|l| l.ridge),
        Some(2.5)
    );
    assert_eq!(
        flag_name(PresetMode::Lora(LoraSpec {
            rank: 1,
            lr_ratio: 1.0,
            ridge: 0.0
        })),
        "--lora-feature-embedding"
    );
}
