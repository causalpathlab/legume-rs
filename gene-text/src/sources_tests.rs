use super::*;
use legume_numeric::matrix::common_io::file_stem;
use std::io::Write;

fn tmp(contents: &str, suffix: &str) -> tempfile::NamedTempFile {
    let mut f = tempfile::Builder::new().suffix(suffix).tempfile().unwrap();
    f.write_all(contents.as_bytes()).unwrap();
    f.flush().unwrap();
    f
}

fn gene_kind() -> FeatureNameKind {
    FeatureNameKind::Gene { delim: '_' }
}

#[test]
fn generic_rows_merge_by_type_and_feature_and_a_later_source_only_fills_gaps() {
    let f = tmp(
        "feature\ttype\tname\ttext\nTP53\tgene\ttumor protein p53\t\nENSG1_TP53\tgene\t\tGuardian of the genome.\nGO:1\tterm\tapoptosis\tProgrammed death.\nTP53\tterm\t\tnot the gene\n",
        ".tsv",
    );
    let mut c = Corpus::new(gene_kind());
    c.add_generic_tsv(f.path().to_str().unwrap()).unwrap();
    assert_eq!(
        c.docs().len(),
        3,
        "gene TP53 merged with its ENSG alias; term TP53 is separate"
    );
    let tp53 = &c.docs()[0];
    assert_eq!(tp53.ty.as_ref(), "gene");
    assert_eq!(tp53.name.as_ref(), "tumor protein p53");
    assert_eq!(tp53.text.as_ref(), "Guardian of the genome.");
    assert_eq!(
        tp53.sentence(),
        "tumor protein p53. Guardian of the genome."
    );
    assert_eq!(c.docs()[2].sentence(), "not the gene");
}

#[test]
fn uniprot_rows_strip_evidence_tags_and_fan_out_over_primary_symbols() {
    let f = tmp(
        "Entry\tGene Names (primary)\tProtein names\tFunction [CC]\n\
         P04637\tTP53\tCellular tumor antigen p53\tFUNCTION: Acts as a tumor suppressor {ECO:0000269|PubMed:1}. Binds DNA {ECO:0000305}.\n\
         Q00000\tGENEA; GENEB\tTwin protein\t\n\
         Q11111\t\tOrphan\tFUNCTION: nothing\n",
        ".tsv",
    );
    let mut c = Corpus::new(gene_kind());
    c.add_uniprot_tsv(f.path().to_str().unwrap()).unwrap();
    assert_eq!(
        c.docs().len(),
        3,
        "TP53, GENEA, GENEB; the row without a symbol is skipped"
    );
    assert_eq!(
        c.docs()[0].text.as_ref(),
        "Acts as a tumor suppressor . Binds DNA ."
    );
    assert_eq!(c.docs()[1].name.as_ref(), "Twin protein");
    assert_eq!(c.docs()[2].feature.as_ref(), "GENEB");
}

#[test]
fn gene_info_rows_give_the_full_name_and_synonyms() {
    let f = tmp(
        "#tax_id\tGeneID\tSymbol\tLocusTag\tSynonyms\tdbXrefs\tchromosome\tmap_location\tdescription\ttype_of_gene\n\
         9606\t7157\tTP53\t-\tBCC7|LFS1|P53\tMIM:191170\t17\t17p13.1\ttumor protein p53\tprotein-coding\n\
         9606\t1\tA1BG\t-\t-\t-\t19\t19q13.43\talpha-1-B glycoprotein\tprotein-coding\n",
        ".tsv",
    );
    let mut c = Corpus::new(gene_kind());
    c.add_ncbi_gene_info(f.path().to_str().unwrap()).unwrap();
    assert_eq!(c.docs().len(), 2);
    assert_eq!(c.docs()[0].name.as_ref(), "tumor protein p53");
    assert_eq!(c.docs()[0].text.as_ref(), "Also known as BCC7, LFS1, P53.");
    assert_eq!(c.docs()[1].text.as_ref(), "");
}

#[test]
fn gmt_sets_use_the_description_unless_it_is_a_url_and_obo_terms_bring_definitions() {
    let gmt = tmp(
        "HALLMARK_APOPTOSIS\thttp://msigdb/x\tTP53\tBAX\nCUSTOM_SET\tGenes I like\tMYC\n",
        ".gmt",
    );
    let obo = tmp(
        "format-version: 1.2\n\n[Term]\nid: GO:1\nname: apoptotic process\ndef: \"Programmed death.\" [x]\n\n[Term]\nid: GO:2\nname: nameless def\n\n[Term]\nid: GO:3\nis_obsolete: true\n",
        ".obo",
    );
    let mut c = Corpus::new(gene_kind());
    c.add_gmt(gmt.path().to_str().unwrap()).unwrap();
    c.add_obo(obo.path().to_str().unwrap()).unwrap();
    let mut docs: Vec<(String, String, String)> = c
        .docs()
        .iter()
        .map(|d| {
            (
                d.feature.to_string(),
                d.name.to_string(),
                d.text.to_string(),
            )
        })
        .collect();
    docs.sort();
    assert_eq!(
        docs,
        vec![
            (
                "CUSTOM_SET".into(),
                "CUSTOM SET".into(),
                "Genes I like".into()
            ),
            (
                "GO:1".into(),
                "apoptotic process".into(),
                "Programmed death.".into()
            ),
            ("GO:2".into(), "nameless def".into(), String::new()),
            (
                "HALLMARK_APOPTOSIS".into(),
                "HALLMARK APOPTOSIS".into(),
                String::new()
            ),
        ]
    );
    assert_eq!(c.retain_with_text(), 0);
}

#[test]
fn gmt_and_gaf_memberships_ride_along_with_the_text_and_gaf_propagates_through_the_obo() {
    let gmt = tmp("SET_A\thttp://x\tTP53\tBAX\nSET_B\tdesc\tMYC\n", ".gmt");
    let obo = tmp(
        "format-version: 1.2\n\n[Term]\nid: GO:1\nname: root\n\n[Term]\nid: GO:2\nname: leaf\ndef: \"A leaf.\" [x]\nis_a: GO:1 ! root\n",
        ".obo",
    );
    let mut row = vec![""; 17];
    row[0] = "UniProtKB";
    row[1] = "P1";
    row[2] = "TP53";
    row[3] = "involved_in";
    row[4] = "GO:2";
    row[5] = "PMID:1";
    row[6] = "IDA";
    row[8] = "P";
    row[10] = "TP53";
    row[11] = "protein";
    row[12] = "taxon:9606";
    row[13] = "20200101";
    row[14] = "UniProt";
    let gaf = tmp(&format!("!gaf-version: 2.2\n{}\n", row.join("\t")), ".gaf");
    let mut c = Corpus::new(gene_kind());
    c.add_gmt(gmt.path().to_str().unwrap()).unwrap();
    c.add_obo(obo.path().to_str().unwrap()).unwrap();
    c.add_gaf(gaf.path().to_str().unwrap(), false).unwrap();
    let mut m: Vec<(String, String)> = c
        .memberships()
        .iter()
        .map(|m| (m.gene.to_string(), m.term.to_string()))
        .collect();
    m.sort();
    // GMT: 3 rows; GAF: TP53 → GO:2 and, propagated, GO:1.
    assert_eq!(
        m,
        vec![
            ("BAX".into(), "SET_A".into()),
            ("MYC".into(), "SET_B".into()),
            ("TP53".into(), "GO:1".into()),
            ("TP53".into(), "GO:2".into()),
            ("TP53".into(), "SET_A".into()),
        ]
    );
    let src: std::collections::BTreeSet<&str> =
        c.memberships().iter().map(|m| m.source.as_ref()).collect();
    assert_eq!(src.len(), 2, "one source per file");
    let has = |f: &str| {
        c.docs()
            .iter()
            .position(|d| d.feature.as_ref() == f)
            .unwrap()
    };
    assert!(
        !c.has_description(has("SET_A")),
        "URL description is no description"
    );
    assert!(c.has_description(has("SET_B")));
    assert!(c.has_description(has("GO:2")) && !c.has_description(has("GO:1")));
    assert_eq!(
        file_stem("/a/b/c2.all.v2026.1.Hs.symbols.gmt"),
        "c2.all.v2026.1.Hs.symbols"
    );
    assert_eq!(file_stem("goa_human.gaf.gz"), "goa_human");
}
