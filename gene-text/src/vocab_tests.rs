use super::*;

fn opts(stem: bool) -> TokenizeOpts {
    TokenizeOpts::new(3, stem, None).unwrap()
}

fn occ(words: &[&str]) -> Vec<Occurrence> {
    words
        .iter()
        .enumerate()
        .map(|(i, w)| Occurrence {
            word: (*w).into(),
            start: i * 10,
            end: i * 10 + w.len(),
        })
        .collect()
}

#[test]
fn tokens_are_lowercased_filtered_by_length_digits_and_stopwords_and_keep_their_spans() {
    let t = tokenize(
        "The TP53 protein regulates Apoptosis in 2 steps; DNA-binding is the role.",
        &opts(false),
    );
    let words: Vec<&str> = t.iter().map(|o| o.word.as_ref()).collect();
    // "the" (stopword), "protein" (filler), "in" (short/stop), "2" (digits),
    // "is" (stop), "role" (filler) go; "tp53" stays (alphabetic content).
    assert_eq!(
        words,
        vec!["tp53", "regulates", "apoptosis", "steps", "dna", "binding"]
    );
    let first = &t[0];
    assert_eq!(&"The TP53 protein"[first.start..first.end], "TP53");
    let apoptosis = &t[2];
    assert_eq!(
        &"The TP53 protein regulates Apoptosis in 2 steps; DNA-binding is the role."
            [apoptosis.start..apoptosis.end],
        "Apoptosis"
    );
}

#[test]
fn stemming_folds_inflections_after_the_stopword_test() {
    let t = tokenize(
        "kinases phosphorylate proteins; regulates regulation",
        &opts(true),
    );
    let words: Vec<&str> = t.iter().map(|o| o.word.as_ref()).collect();
    assert_eq!(words, vec!["kinas", "phosphoryl", "regul", "regul"]);
}

#[test]
fn extra_stopwords_come_from_a_file() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("extra.txt");
    std::fs::write(&p, "# mine\nKinase\n\n").unwrap();
    let o = TokenizeOpts::new(3, false, Some(p.to_str().unwrap())).unwrap();
    let t = tokenize("kinase activity phosphorylates", &o);
    let words: Vec<&str> = t.iter().map(|w| w.word.as_ref()).collect();
    assert_eq!(words, vec!["phosphorylates"]);
}

/// 100 documents: `everywhere` in all, `common` in 60, `mid` in 20,
/// `rare` in 2, and a unique word per document.
fn planted_corpus() -> Vec<Vec<Occurrence>> {
    (0..100)
        .map(|i| {
            let mut w = vec!["everywhere"];
            if i < 60 {
                w.push("common");
            }
            if i < 20 {
                w.push("mid");
            }
            if i < 2 {
                w.push("rare");
            }
            let unique = format!("unique{i}");
            let mut o = occ(&w);
            o.push(Occurrence {
                word: unique.into(),
                start: 900,
                end: 909,
            });
            // a repeated word counts once per document
            o.push(o[0].clone());
            o
        })
        .collect()
}

#[test]
fn df_counts_once_per_document_and_both_tails_are_cut_by_quantile() {
    let docs = planted_corpus();
    // 104 words: 100 singletons, rare(2), mid(20), common(60), everywhere(100).
    let v = Vocabulary::build(
        &docs,
        &DfQcOpts {
            lower_quantile: 0.05,
            upper_quantile: 0.99,
            min_df: 0,
            max_df_frac: 1.0,
        },
    );
    assert_eq!(v.n_docs, 100);
    assert_eq!(v.entries.len(), 104);
    let verdict = |w: &str| {
        v.entries
            .iter()
            .find(|e| e.word.as_ref() == w)
            .unwrap()
            .verdict
    };
    let df = |w: &str| v.entries.iter().find(|e| e.word.as_ref() == w).unwrap().df;
    assert_eq!(
        df("everywhere"),
        100,
        "the duplicate occurrence counted once"
    );
    // The 5 % quantile of 104 ascending dfs is a singleton → df ≤ 1 dropped.
    assert_eq!(v.lower_cut, Some(1));
    assert_eq!(verdict("unique7"), Verdict::Rare);
    assert_eq!(
        verdict("rare"),
        Verdict::Kept,
        "df 2 is above the singleton cut"
    );
    // The 99 % quantile: index 102 of 104 → `common` (60); ≥ 60 dropped.
    assert_eq!(v.upper_cut, Some(60));
    assert_eq!(verdict("common"), Verdict::Common);
    assert_eq!(verdict("everywhere"), Verdict::Common);
    assert_eq!(verdict("mid"), Verdict::Kept);
    assert_eq!(v.kept.len(), 2, "rare and mid");
    assert!(v.idf(v.index["rare"]) > v.idf(v.index["mid"]));
}

#[test]
fn absolute_limits_tighten_but_never_loosen_the_quantile_cuts() {
    let docs = planted_corpus();
    let v = Vocabulary::build(
        &docs,
        &DfQcOpts {
            lower_quantile: 0.0,
            upper_quantile: 1.0,
            min_df: 3,
            max_df_frac: 0.5,
        },
    );
    let verdict = |w: &str| {
        v.entries
            .iter()
            .find(|e| e.word.as_ref() == w)
            .unwrap()
            .verdict
    };
    assert_eq!(v.lower_cut, None);
    assert_eq!(v.upper_cut, None);
    assert_eq!(verdict("rare"), Verdict::Rare, "df 2 < min-df 3");
    assert_eq!(verdict("common"), Verdict::Common, "df 60 > 50 % of 100");
    assert_eq!(verdict("mid"), Verdict::Kept);
    // Quantile on, absolute off: the same corpus keeps `rare`.
    let loose = Vocabulary::build(
        &docs,
        &DfQcOpts {
            lower_quantile: 0.05,
            upper_quantile: 1.0,
            min_df: 0,
            max_df_frac: 1.0,
        },
    );
    assert_eq!(
        loose
            .entries
            .iter()
            .find(|e| e.word.as_ref() == "rare")
            .unwrap()
            .verdict,
        Verdict::Kept
    );
}

#[test]
fn the_vocabulary_round_trips_through_its_tsv() {
    let docs = planted_corpus();
    let v = Vocabulary::build(
        &docs,
        &DfQcOpts {
            lower_quantile: 0.05,
            upper_quantile: 0.99,
            min_df: 0,
            max_df_frac: 1.0,
        },
    );
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("v.tsv");
    v.write_tsv(p.to_str().unwrap()).unwrap();
    let text = std::fs::read_to_string(&p).unwrap();
    assert!(
        text.starts_with("word\tdf\tdf_frac\tkept\treason\neverywhere\t100\t1.000000\t0\tcommon\n")
    );
    let back = Vocabulary::read_tsv(p.to_str().unwrap(), 100).unwrap();
    assert_eq!(back.kept, v.kept);
    assert_eq!(back.entries.len(), v.entries.len());
    assert!((back.idf(back.index["mid"]) - v.idf(v.index["mid"])).abs() < 1e-12);
}

#[test]
fn abbreviations_and_unit_tokens_fall_with_the_short_words() {
    let t = tokenize(
        "kinases, e.g. MAPK, i.e. the 5.8S rRNA, E.coli strains, TP53",
        &opts(false),
    );
    let words: Vec<&str> = t.iter().map(|o| o.word.as_ref()).collect();
    assert_eq!(
        words,
        vec!["kinases", "mapk", "rrna", "e.coli", "strains", "tp53"]
    );
}

#[test]
fn msigdb_collection_prefixes_and_direction_tags_are_filler() {
    let t = tokenize(
        "HALLMARK INTERFERON GAMMA RESPONSE UP; REACTOME apoptosis DN; KEGG glycolysis",
        &opts(false),
    );
    let words: Vec<&str> = t.iter().map(|o| o.word.as_ref()).collect();
    assert_eq!(
        words,
        vec!["interferon", "gamma", "apoptosis", "glycolysis"]
    );
}
