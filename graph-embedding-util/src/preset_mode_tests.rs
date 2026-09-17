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
