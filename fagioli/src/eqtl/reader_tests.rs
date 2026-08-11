//! Tests for the long-format eQTL reader: column mapping, file hygiene,
//! global variant identity, and the gene/celltype filters.

use super::*;
use matrix_util::common_io::open_buf_writer;
use std::io::Write;
use tempfile::TempDir;

/// The rule the subcommand defaults to: `--feature-name-kind auto`.
fn gene_rule() -> FeatureNameKind {
    auxiliary_data::feature_names::FeatureNameKindArg::Auto.resolve_or_gene()
}

fn default_columns() -> QtlColumns {
    QtlColumns {
        gene: "gene".into(),
        celltype: "celltype".into(),
        chromosome: "chromosome".into(),
        position: "physical.pos".into(),
        beta: "beta".into(),
        se: "se".into(),
        pip: "alpha".into(),
    }
}

fn write_plain(dir: &TempDir, name: &str, lines: &[&str]) -> String {
    let path = dir.path().join(name).to_str().unwrap().to_string();
    std::fs::write(&path, lines.join("\n")).unwrap();
    path
}

fn write_gz(dir: &TempDir, name: &str, lines: &[&str]) -> String {
    let path = dir.path().join(name).to_str().unwrap().to_string();
    let mut w = open_buf_writer(&path).unwrap();
    for line in lines {
        writeln!(w, "{}", line).unwrap();
    }
    w.flush().unwrap();
    path
}

/// A wide header: extra columns present, first column carrying a
/// leading '#'.
const WIDE_HEADER: &str =
    "#chromosome\tphysical.pos\tlevels\tgene\tcelltype\talpha\tmean\tsd\tlbf\tz\tlodds\tlfsr\tbeta\tse\tn\tp.val";

fn wide_row(chr: &str, pos: u64, gene: &str, ct: &str, alpha: f32, beta: f32, se: f32) -> String {
    format!("{chr}\t{pos}\t1\t{gene}\t{ct}\t{alpha}\t0\t0\t0\t0\t0\t0\t{beta}\t{se}\t100\t0.5")
}

#[test]
fn wide_header_with_leading_hash_parses_with_default_columns() {
    let dir = TempDir::new().unwrap();
    let rows = [
        wide_row("5", 100, "GENE1", "CT1", 0.5, 1.0, 0.1),
        wide_row("5", 100, "GENE1", "CT2", 0.4, 0.9, 0.2),
        wide_row("5", 100, "GENE1", "CT3", 0.1, 0.8, 0.3),
        wide_row("5", 200, "GENE1", "CT1", 0.5, -1.0, 0.1),
    ];
    let row_refs: Vec<&str> = std::iter::once(WIDE_HEADER)
        .chain(rows.iter().map(|s| s.as_str()))
        .collect();
    let path = write_gz(&dir, "block.txt.gz", &row_refs);

    let data = read_qtl_files(&[path], &default_columns(), &gene_rule(), 3, None).unwrap();

    assert_eq!(data.genes.len(), 1);
    assert_eq!(data.genes[0].as_ref(), "GENE1");
    assert_eq!(data.celltypes.len(), 3);
    assert_eq!(data.tables[0].entries.len(), 4);
    assert_eq!(data.n_files_read, 1);
    assert_eq!(data.n_files_skipped, 0);

    // Variant keys are canonical loci: `chromosome_position`.
    assert_eq!(data.variants.len(), 2);
    assert!(data.variants.iter().any(|v| v.as_ref() == "5_100"));
    assert!(data.variants.iter().any(|v| v.as_ref() == "5_200"));

    let ast = data
        .celltypes
        .iter()
        .position(|c| c.as_ref() == "CT1")
        .unwrap() as u32;
    let v100 = data
        .variants
        .iter()
        .position(|v| v.as_ref() == "5_100")
        .unwrap() as u32;
    let e = data.tables[0]
        .entries
        .iter()
        .find(|e| e.celltype == ast && e.variant == v100)
        .unwrap();
    assert!((e.beta - 1.0).abs() < 1e-6);
    assert!((e.se - 0.1).abs() < 1e-6);
    assert!((e.pip_weight.unwrap() - 0.5).abs() < 1e-6);
}

/// The same locus tested against two genes must intern to ONE variant id:
/// selection carries a chosen variant across every gene it touches, which
/// is only possible with a global identity.
#[test]
fn the_same_locus_is_one_variant_across_genes() {
    let dir = TempDir::new().unwrap();
    let rows = [
        wide_row("5", 100, "GENE_A", "CT1", 0.5, 1.0, 0.1),
        wide_row("5", 100, "GENE_A", "CT2", 0.5, 1.0, 0.1),
        wide_row("5", 100, "GENE_B", "CT1", 0.5, 0.2, 0.1),
        wide_row("5", 100, "GENE_B", "CT2", 0.5, 0.2, 0.1),
        wide_row("5", 300, "GENE_B", "CT1", 0.5, 0.2, 0.1),
        wide_row("5", 300, "GENE_B", "CT2", 0.5, 0.2, 0.1),
    ];
    let refs: Vec<&str> = std::iter::once(WIDE_HEADER)
        .chain(rows.iter().map(|s| s.as_str()))
        .collect();
    let path = write_gz(&dir, "shared.txt.gz", &refs);

    let data = read_qtl_files(&[path], &default_columns(), &gene_rule(), 2, None).unwrap();
    assert_eq!(data.genes.len(), 2);
    assert_eq!(data.variants.len(), 2);

    let v100 = data
        .variants
        .iter()
        .position(|v| v.as_ref() == "5_100")
        .unwrap() as u32;
    for table in &data.tables {
        assert!(table.entries.iter().any(|e| e.variant == v100));
    }
}

#[test]
fn column_overrides_map_a_renamed_header() {
    let dir = TempDir::new().unwrap();
    let path = write_plain(
        &dir,
        "renamed.tsv",
        &[
            "chrom\tbp\tsymbol\tcontext\teffect\tstderr\tpp",
            "1\t10\tGENE1\tA\t0.5\t0.1\t0.9",
            "1\t10\tGENE1\tB\t0.4\t0.1\t0.9",
            "1\t10\tGENE1\tC\t0.3\t0.1\t0.9",
        ],
    );
    let cols = QtlColumns {
        gene: "symbol".into(),
        celltype: "context".into(),
        chromosome: "chrom".into(),
        position: "bp".into(),
        beta: "effect".into(),
        se: "stderr".into(),
        pip: "pp".into(),
    };

    let data = read_qtl_files(&[path], &cols, &gene_rule(), 3, None).unwrap();
    assert_eq!(data.genes.len(), 1);
    assert_eq!(data.celltypes.len(), 3);
    assert_eq!(data.tables[0].entries.len(), 3);
}

#[test]
fn corrupt_and_truncated_gz_files_are_skipped_and_counted() {
    let dir = TempDir::new().unwrap();
    let good_rows = [
        wide_row("1", 10, "G1", "A", 0.9, 0.5, 0.1),
        wide_row("1", 10, "G1", "B", 0.9, 0.5, 0.1),
        wide_row("1", 10, "G1", "C", 0.9, 0.5, 0.1),
    ];
    let good_refs: Vec<&str> = std::iter::once(WIDE_HEADER)
        .chain(good_rows.iter().map(|s| s.as_str()))
        .collect();
    let good = write_gz(&dir, "good.txt.gz", &good_refs);

    // Not gzip at all.
    let garbage = dir
        .path()
        .join("garbage.txt.gz")
        .to_str()
        .unwrap()
        .to_string();
    std::fs::write(&garbage, b"this is not a gzip stream").unwrap();

    // A valid gz member cut off halfway.
    let mut bytes = std::fs::read(&good).unwrap();
    bytes.truncate(bytes.len() / 2);
    let truncated = dir
        .path()
        .join("truncated.txt.gz")
        .to_str()
        .unwrap()
        .to_string();
    std::fs::write(&truncated, &bytes).unwrap();

    let files = vec![good, garbage, truncated];
    let data = read_qtl_files(&files, &default_columns(), &gene_rule(), 3, None).unwrap();

    assert_eq!(data.n_files_read, 1);
    assert_eq!(data.n_files_skipped, 2);
    assert_eq!(data.genes.len(), 1);
    assert_eq!(data.tables[0].entries.len(), 3);
}

/// The real input blocks are multi-member gzip whose FIRST member holds only
/// the header line; a single-member decoder sees zero data rows. This test
/// re-creates that layout byte-for-byte.
#[test]
fn multi_member_gz_is_read_past_the_first_member() {
    let dir = TempDir::new().unwrap();
    let header = write_gz(&dir, "header.gz", &[WIDE_HEADER]);
    let rows = [
        wide_row("1", 10, "G1", "A", 0.9, 0.5, 0.1),
        wide_row("1", 10, "G1", "B", 0.9, 0.5, 0.1),
        wide_row("1", 10, "G1", "C", 0.9, 0.5, 0.1),
    ];
    let row_refs: Vec<&str> = rows.iter().map(|s| s.as_str()).collect();
    let body = write_gz(&dir, "body.gz", &row_refs);

    let mut bytes = std::fs::read(&header).unwrap();
    bytes.extend(std::fs::read(&body).unwrap());
    let multi = dir
        .path()
        .join("multi.txt.gz")
        .to_str()
        .unwrap()
        .to_string();
    std::fs::write(&multi, &bytes).unwrap();

    let data = read_qtl_files(&[multi], &default_columns(), &gene_rule(), 3, None).unwrap();
    assert_eq!(data.n_files_read, 1);
    assert_eq!(data.genes.len(), 1);
    assert_eq!(data.tables[0].entries.len(), 3);
}

#[test]
fn min_celltypes_filter_drops_and_counts_genes() {
    let dir = TempDir::new().unwrap();
    let rows = [
        wide_row("1", 10, "WIDE", "A", 0.9, 0.5, 0.1),
        wide_row("1", 10, "WIDE", "B", 0.9, 0.5, 0.1),
        wide_row("1", 10, "WIDE", "C", 0.9, 0.5, 0.1),
        wide_row("1", 20, "NARROW", "A", 0.9, 0.5, 0.1),
        wide_row("1", 20, "NARROW", "Zonly", 0.9, 0.5, 0.1),
    ];
    let refs: Vec<&str> = std::iter::once(WIDE_HEADER)
        .chain(rows.iter().map(|s| s.as_str()))
        .collect();
    let path = write_gz(&dir, "mix.txt.gz", &refs);

    let data = read_qtl_files(
        std::slice::from_ref(&path),
        &default_columns(),
        &gene_rule(),
        3,
        None,
    )
    .unwrap();
    assert_eq!(data.genes.len(), 1);
    assert_eq!(data.genes[0].as_ref(), "WIDE");
    assert_eq!(data.n_genes_dropped, 1);
    // The dropped gene's private cell type and variant are compacted away.
    assert!(data.celltypes.iter().all(|c| c.as_ref() != "Zonly"));
    assert_eq!(data.variants.len(), 1);
    assert_eq!(data.variants[0].as_ref(), "1_10");

    let relaxed = read_qtl_files(&[path], &default_columns(), &gene_rule(), 2, None).unwrap();
    assert_eq!(relaxed.genes.len(), 2);
    assert_eq!(relaxed.n_genes_dropped, 0);
    assert!(relaxed.celltypes.iter().any(|c| c.as_ref() == "Zonly"));
    assert_eq!(relaxed.variants.len(), 2);
}

#[test]
fn row_hygiene_drops_bad_se_and_unparseable_rows() {
    let dir = TempDir::new().unwrap();
    let mut rows = vec![
        wide_row("1", 10, "G1", "A", 0.9, 0.5, 0.1),
        wide_row("1", 10, "G1", "B", 0.9, 0.5, 0.1),
        wide_row("1", 10, "G1", "C", 0.9, 0.5, 0.1),
        // se = 0 must be dropped, not divided by.
        wide_row("1", 20, "G1", "A", 0.9, 0.5, 0.0),
        // se above --max-se.
        wide_row("1", 30, "G1", "A", 0.9, 0.5, 50.0),
    ];
    // Unparseable beta.
    rows.push("1\t40\t1\tG1\tA\t0.9\t0\t0\t0\t0\t0\t0\tNA\t0.1\t100\t0.5".to_string());
    let refs: Vec<&str> = std::iter::once(WIDE_HEADER)
        .chain(rows.iter().map(|s| s.as_str()))
        .collect();
    let path = write_gz(&dir, "hygiene.txt.gz", &refs);

    let data = read_qtl_files(&[path], &default_columns(), &gene_rule(), 3, Some(10.0)).unwrap();
    assert_eq!(data.n_rows_dropped, 3);
    assert_eq!(data.tables[0].entries.len(), 3);
    // Dropped rows must not have minted variants.
    assert_eq!(data.variants.len(), 1);
}

#[test]
fn duplicate_rows_keep_the_larger_weight() {
    let dir = TempDir::new().unwrap();
    let rows = [
        wide_row("1", 10, "G1", "A", 0.2, 0.5, 0.1),
        wide_row("1", 10, "G1", "A", 0.8, 0.7, 0.2),
        wide_row("1", 10, "G1", "B", 0.9, 0.5, 0.1),
        wide_row("1", 10, "G1", "C", 0.9, 0.5, 0.1),
    ];
    let refs: Vec<&str> = std::iter::once(WIDE_HEADER)
        .chain(rows.iter().map(|s| s.as_str()))
        .collect();
    let path = write_gz(&dir, "dup.txt.gz", &refs);

    let data = read_qtl_files(&[path], &default_columns(), &gene_rule(), 3, None).unwrap();
    assert_eq!(data.n_rows_duplicate, 1);
    assert_eq!(data.tables[0].entries.len(), 3);
    let a = data
        .celltypes
        .iter()
        .position(|c| c.as_ref() == "A")
        .unwrap() as u32;
    let e = data.tables[0]
        .entries
        .iter()
        .find(|e| e.celltype == a)
        .unwrap();
    assert!((e.pip_weight.unwrap() - 0.8).abs() < 1e-6);
    assert!((e.beta - 0.7).abs() < 1e-6);
}

#[test]
fn missing_pip_column_yields_unweighted_entries() {
    let dir = TempDir::new().unwrap();
    let path = write_plain(
        &dir,
        "nopip.tsv",
        &[
            "#chromosome\tphysical.pos\tgene\tcelltype\tbeta\tse",
            "1\t10\tG1\tA\t0.5\t0.1",
            "1\t10\tG1\tB\t0.5\t0.1",
            "1\t10\tG1\tC\t0.5\t0.1",
        ],
    );

    let data = read_qtl_files(&[path], &default_columns(), &gene_rule(), 3, None).unwrap();
    assert_eq!(data.tables[0].entries.len(), 3);
    assert!(data.tables[0]
        .entries
        .iter()
        .all(|e| e.pip_weight.is_none()));
}

#[test]
fn wrong_column_names_skip_the_file_with_a_clear_failure() {
    let dir = TempDir::new().unwrap();
    let path = write_plain(
        &dir,
        "wrong.tsv",
        &[
            "#chromosome\tphysical.pos\tgene\tcelltype\tbeta\tse",
            "1\t10\tG1\tA\t0.5\t0.1",
        ],
    );
    let mut cols = default_columns();
    cols.beta = "effect_size".into();

    // The only file is skipped for the missing column, so nothing survives.
    let err = read_qtl_files(&[path], &cols, &gene_rule(), 1, None).unwrap_err();
    assert!(err.to_string().contains("No usable gene"));
}

// ── Canonicalization ────────────────────────────────────────────────────

/// An Ensembl id carries an annotation version. Two files built against two
/// GENCODE releases name the same gene differently, and without stripping it
/// the gene interns twice and its rows never meet.
#[test]
fn ensembl_version_suffixes_collapse_to_one_gene() {
    assert_eq!(canonical_gene_str("ENSG00000000001.17"), "ENSG00000000001");
    assert_eq!(canonical_gene_str("ENSG00000000001.9"), "ENSG00000000001");
    assert_eq!(
        canonical_gene_str("  ENSG00000000001.17  "),
        "ENSG00000000001"
    );
}

/// Clone-based SYMBOLS contain a dot too, but the suffix distinguishes real
/// genes — collapsing them would merge distinct loci.
#[test]
fn symbol_names_keep_their_dotted_suffix() {
    assert_eq!(canonical_gene_str("CLONE1.4"), "CLONE1.4");
    assert_eq!(canonical_gene_str("CLONE1.1"), "CLONE1.1");
    assert_eq!(canonical_gene_str("GENE1"), "GENE1");
    assert_eq!(canonical_gene_str(" GENE1 "), "GENE1");
}

/// Variant keys go through the workspace locus rule, so a fagioli variant id
/// is spelled like a locus anywhere else. Whitespace is trimmed by the reader
/// before the key is built; `canon_locus` itself only folds prefix and
/// separators.
#[test]
fn variant_keys_use_the_workspace_locus_rule() {
    assert_eq!(canon_locus_str("chr5:1000000"), "5_1000000");
    assert_eq!(canon_locus_str("5:1000000"), "5_1000000");
    assert_eq!(canon_locus_str("5_1000000"), "5_1000000");
    assert_eq!(canon_locus_str("chrX:1000"), "X_1000");
}

/// End to end: the same locus written two ways in two files is ONE variant
/// of ONE gene, so its rows land in a single table.
#[test]
fn two_annotation_styles_reach_the_same_entities() {
    let dir = TempDir::new().unwrap();
    let mut old = vec![WIDE_HEADER.to_string()];
    let mut new = vec![WIDE_HEADER.to_string()];
    for (k, ct) in ["CT1", "CT3", "CT2"].iter().enumerate() {
        let beta = 0.4 + k as f32 * 0.01;
        old.push(format!(
            "5\t1000000\tL1\tENSG00000000001.12\t{ct}\t0.5\t0\t0\t0\t0\t0\t0\t{beta}\t0.05\t100\t0"
        ));
        new.push(format!(
            "chr5\t1000000\tL1\tENSG00000000001.17\t{ct}_2\t0.5\t0\t0\t0\t0\t0\t0\t{beta}\t0.05\t100\t0"
        ));
    }
    let old: Vec<&str> = old.iter().map(|s| s.as_str()).collect();
    let new: Vec<&str> = new.iter().map(|s| s.as_str()).collect();
    let a = write_plain(&dir, "old.tsv", &old);
    let b = write_plain(&dir, "new.tsv", &new);

    let data = read_qtl_files(&[a, b], &default_columns(), &gene_rule(), 1, None).unwrap();
    assert_eq!(data.genes.len(), 1, "the version suffix split the gene");
    assert_eq!(data.genes[0].as_ref(), "ENSG00000000001");
    assert_eq!(data.variants.len(), 1, "the chr prefix split the locus");
    assert_eq!(data.variants[0].as_ref(), "5_1000000");
    // Six cell types, three per file, all against the one (gene, variant).
    assert_eq!(data.celltypes.len(), 6);
    assert_eq!(data.tables[0].entries.len(), 6);
}

/// `canonical_gene` returns an owned name; these helpers keep the assertions
/// readable.
fn canonical_gene_str(name: &str) -> String {
    canonical_gene(name, &gene_rule()).to_string()
}

fn canon_locus_str(name: &str) -> String {
    canon_locus(name).to_string()
}

// ── Multiple input formats ──────────────────────────────────────────────

/// Same table, three delimiters. The header picks the delimiter, so a CSV
/// export does not have to be converted before it can be read.
#[test]
fn tsv_csv_and_space_delimited_files_read_alike() {
    let dir = TempDir::new().unwrap();
    let cols = [
        "gene",
        "celltype",
        "chromosome",
        "physical.pos",
        "beta",
        "se",
    ];
    let rows = [
        ["G1", "CT1", "5", "100", "0.5", "0.05"],
        ["G1", "CT3", "5", "100", "0.4", "0.05"],
        ["G1", "CT2", "5", "200", "0.3", "0.05"],
    ];
    let mut counts = Vec::new();
    for (name, sep) in [("t.tsv", '\t'), ("c.csv", ','), ("s.txt", ' ')] {
        let mut lines = vec![cols.join(&sep.to_string())];
        for r in &rows {
            lines.push(r.join(&sep.to_string()));
        }
        let refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
        let path = write_plain(&dir, name, &refs);
        let data = read_qtl_files(&[path], &default_columns(), &gene_rule(), 1, None).unwrap();
        counts.push((data.genes.len(), data.celltypes.len(), data.n_rows()));
    }
    assert_eq!(counts[0], (1, 3, 3));
    assert!(
        counts.iter().all(|c| *c == counts[0]),
        "delimiters disagreed: {counts:?}"
    );
}

/// A header using another project's spelling resolves through the alias
/// list, so GTEx-style output needs no --col-* flags.
#[test]
fn alias_spellings_resolve_without_col_flags() {
    let dir = TempDir::new().unwrap();
    let path = write_plain(
        &dir,
        "gtex.tsv",
        &[
            "phenotype_id\tcell_type\tchr\tpos\tslope\tslope_se",
            "G1\tAst\t5\t100\t0.5\t0.05",
            "G1\tMic\t5\t100\t0.4\t0.05",
        ],
    );
    let data = read_qtl_files(&[path], &default_columns(), &gene_rule(), 1, None).unwrap();
    assert_eq!(data.n_files_read, 1);
    assert_eq!(data.genes.len(), 1);
    assert_eq!(data.celltypes.len(), 2);
    assert_eq!(data.variants[0].as_ref(), "5_100");
}

/// Column names differing only in case still resolve.
#[test]
fn header_matching_ignores_case() {
    let dir = TempDir::new().unwrap();
    let path = write_plain(
        &dir,
        "upper.tsv",
        &[
            "GENE\tCELLTYPE\tCHROM\tPOS\tBETA\tSE",
            "G1\tAst\t5\t100\t0.5\t0.05",
            "G1\tMic\t5\t100\t0.4\t0.05",
        ],
    );
    let data = read_qtl_files(&[path], &default_columns(), &gene_rule(), 1, None).unwrap();
    assert_eq!(data.n_files_read, 1);
    assert_eq!(data.n_rows(), 2);
}

/// A header with no usable column must say what it tried and what it saw.
/// The old message named one column and left the rest to guesswork.
#[test]
fn an_unresolvable_header_reports_what_it_tried() {
    let dir = TempDir::new().unwrap();
    let path = write_plain(&dir, "wrong.tsv", &["a\tb\tc", "1\t2\t3"]);
    let err = match parse_file(&path, &default_columns(), &gene_rule(), None) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("a header with no usable column must not parse"),
    };
    assert!(err.contains("gene"), "{err}");
    assert!(err.contains("phenotype_id"), "aliases not listed: {err}");
    assert!(
        err.contains('a') && err.contains('b'),
        "header absent: {err}"
    );
}
