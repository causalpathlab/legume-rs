use super::build_gene_index;

fn rows(names: &[&str]) -> Vec<Box<str>> {
    names.iter().map(|s| Box::from(*s)).collect()
}

#[test]
fn spliced_and_unspliced_pool_into_one_gene() {
    let names = rows(&[
        "GENE1/count/spliced",
        "GENE1/count/unspliced",
        "GENE2/count/spliced",
    ]);
    let (row_to_gene, genes) = build_gene_index(&names).expect("well-formed rows parse");
    assert_eq!(genes.len(), 2, "two genes, not three rows");
    assert_eq!(&*genes[0], "GENE1");
    assert_eq!(&*genes[1], "GENE2");
    assert_eq!(
        row_to_gene[0], row_to_gene[1],
        "both GENE1 tracks pool together"
    );
    assert_ne!(row_to_gene[0], row_to_gene[2], "GENE2 is a distinct gene");
}

#[test]
fn a_row_that_does_not_parse_is_an_error_naming_the_row() {
    let names = rows(&["GENE1/count/spliced", "not-a-feature-row"]);
    let err =
        build_gene_index(&names).expect_err("an unparseable row must error, not silently pass");
    assert!(
        err.to_string().contains("not-a-feature-row"),
        "the error must name the offending row: {err}"
    );
}
