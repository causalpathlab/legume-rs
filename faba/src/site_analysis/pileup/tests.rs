use super::*;

#[test]
fn parse_site_and_mixture_rows() {
    // site output: gene/modality/chr:pos
    assert_eq!(
        parse_row_name_full("ENSG00000139618_BRCA2/m6A/chr13:32350000"),
        Some(("ENSG00000139618_BRCA2", "m6A", "chr13", 32350000))
    );
    // mixture output: gene/modality/component
    assert_eq!(
        parse_row_name_full("ENSG00000060558_GNA15/m6A/0"),
        Some(("ENSG00000060558_GNA15", "m6A", "", 0))
    );
    // count rows (detail neither chr:pos nor an integer) don't parse
    assert_eq!(parse_row_name_full("gene_0/count/spliced"), None);
}

#[test]
fn relaxed_gene_matching() {
    let gp = "ENSG00000060558_GNA15";
    // symbol, any case
    assert!(gene_matches("GNA15", &query_symbol("GNA15"), gp));
    assert!(gene_matches("gna15", &query_symbol("gna15"), gp));
    // Ensembl ID
    assert!(gene_matches(
        "ENSG00000060558",
        &query_symbol("ENSG00000060558"),
        gp
    ));
    // full composite
    assert!(gene_matches(gp, &query_symbol(gp), gp));
    // partial substring must NOT match (consistent with aux-data scheme)
    assert!(!gene_matches("GNA", &query_symbol("GNA"), gp));
    assert!(!gene_matches(
        "RPL",
        &query_symbol("RPL"),
        "ENSG00000063177_RPL18"
    ));
}

#[test]
fn region_parsing_and_matching() {
    let r = parse_region("chr17:1000-2000").unwrap();
    assert_eq!((r.chr.as_ref(), r.lb, r.ub), ("chr17", 1000, 2000));
    // reversed bounds are normalized
    let r = parse_region("17:2000-1000").unwrap();
    assert_eq!((r.lb, r.ub), (1000, 2000));
    // malformed specs error
    assert!(parse_region("chr17").is_err());
    assert!(parse_region("chr17:1000").is_err());
    assert!(parse_region(":1-2").is_err());

    // chr-prefix tolerant (shared genomic_data::chr_eq)
    assert!(chr_eq("chr17", "17"));
    assert!(chr_eq("17", "17"));
    assert!(!chr_eq("chr1", "chr2"));
}

#[test]
fn selector_gene_or_region_union() {
    let sel = Selector::build(&["GNA15".into()], &["chr17:100-200".into()]).unwrap();
    // gene branch (mixture row: no chr, component ordinal)
    assert!(sel.selects("ENSG00000060558_GNA15", "", 0));
    // region branch (different gene, but inside the window)
    assert!(sel.selects("ENSG1_OTHER", "chr17", 150));
    // outside both
    assert!(!sel.selects("ENSG1_OTHER", "chr17", 999));
    assert!(!sel.selects("ENSG1_OTHER", "chr9", 150));

    // at least one selector is required
    assert!(Selector::build(&[], &[]).is_err());
    // region-only is fine
    assert!(Selector::build(&[], &["chr1:1-9".into()]).is_ok());
}

#[test]
fn thousands_and_axis_mapping() {
    assert_eq!(fmt_thousands(26781984), "26,781,984");
    assert_eq!(fmt_thousands(767), "767");
    assert_eq!(fmt_thousands(0), "0");
    assert_eq!(fmt_thousands(-1234), "-1,234");

    // first site -> first column, last site -> last column
    assert_eq!(pos_to_col(100, 100, 200, 10), 0);
    assert_eq!(pos_to_col(200, 100, 200, 10), 9);
    assert_eq!(pos_to_col(150, 100, 200, 10), 5);
    // degenerate single-position extent collapses to column 0
    assert_eq!(pos_to_col(100, 100, 100, 10), 0);
}

#[test]
fn distinct_positions_dedups_sorted() {
    let p = [(10, 1.0), (10, 2.0), (20, 0.5), (30, 0.0)];
    assert_eq!(distinct_positions(&p), vec![10, 20, 30]);
}

#[test]
fn aggregate_labels() {
    let mut one: FxHashMap<Box<str>, usize> = FxHashMap::default();
    one.insert("ENSG1_GNA15".into(), 2);
    assert_eq!(summarize_genes(&one).as_ref(), "ENSG1_GNA15");

    let mut many: FxHashMap<Box<str>, usize> = FxHashMap::default();
    many.insert("ENSG1_A".into(), 1);
    many.insert("ENSG2_B".into(), 3);
    let label = summarize_genes(&many);
    assert!(label.starts_with("2 genes: "), "got {label}");

    // chr label: empty -> component, single -> that chr, mixed -> *
    let mix: Vec<Box<str>> = vec!["".into(), "".into()];
    assert_eq!(summarize_chr(&mix).as_ref(), "component");
    let single: Vec<Box<str>> = vec!["chr13".into()];
    assert_eq!(summarize_chr(&single).as_ref(), "chr13");
    let multi: Vec<Box<str>> = vec!["chr1".into(), "chr2".into()];
    assert_eq!(summarize_chr(&multi).as_ref(), "*");
}
