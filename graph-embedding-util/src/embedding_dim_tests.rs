use super::EmbeddingDim;

#[test]
fn parses_a_width_or_auto_and_refuses_zero() {
    assert_eq!(
        "128".parse::<EmbeddingDim>().unwrap(),
        EmbeddingDim::Fixed(128)
    );
    assert_eq!("auto".parse::<EmbeddingDim>().unwrap(), EmbeddingDim::Auto);
    assert_eq!("Auto".parse::<EmbeddingDim>().unwrap(), EmbeddingDim::Auto);
    assert!("0".parse::<EmbeddingDim>().is_err());
    assert!("wide".parse::<EmbeddingDim>().is_err());
    assert_eq!(EmbeddingDim::Fixed(16).to_string(), "16");
    assert_eq!(EmbeddingDim::Auto.to_string(), "auto");
}

#[test]
fn auto_takes_the_table_and_a_fixed_width_must_agree() {
    assert_eq!(EmbeddingDim::Auto.resolve(Some(64)).unwrap(), Some(64));
    assert_eq!(EmbeddingDim::Fixed(64).resolve(Some(64)).unwrap(), Some(64));
    let err = EmbeddingDim::Fixed(16)
        .resolve(Some(64))
        .unwrap_err()
        .to_string();
    assert!(err.contains("16") && err.contains("64"), "{err}");
    assert_eq!(EmbeddingDim::Auto.resolve(None).unwrap(), None);
    assert_eq!(EmbeddingDim::Fixed(32).resolve(None).unwrap(), Some(32));
}

#[test]
fn records_as_a_number_or_the_word_and_reads_the_old_zero_as_auto() {
    assert_eq!(
        serde_json::to_string(&EmbeddingDim::Fixed(128)).unwrap(),
        "128"
    );
    assert_eq!(
        serde_json::to_string(&EmbeddingDim::Auto).unwrap(),
        "\"auto\""
    );
    assert_eq!(
        serde_json::from_str::<EmbeddingDim>("128").unwrap(),
        EmbeddingDim::Fixed(128)
    );
    assert_eq!(
        serde_json::from_str::<EmbeddingDim>("\"auto\"").unwrap(),
        EmbeddingDim::Auto
    );
    assert_eq!(
        serde_json::from_str::<EmbeddingDim>("0").unwrap(),
        EmbeddingDim::Auto
    );
    assert!(serde_json::from_str::<EmbeddingDim>("\"wide\"").is_err());
}
