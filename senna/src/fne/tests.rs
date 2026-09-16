//! `senna fne` end to end: edge files of both shapes become one typed
//! graph, the artifacts carry every node type, and the manifest records
//! the fit.

use super::graph::{NodeText, TypedGraphBuilder};
use super::{fit_fne, FneArgs};
use crate::embed_common::Mat;
use crate::run_manifest::{RunKind, RunManifest};
use auxiliary_data::feature_names::FeatureNameKind;
use auxiliary_data::gene_sets::{read_gaf, read_gmt, GafOpts};
use auxiliary_data::ontology::Ontology;
use clap::Parser;
use matrix_util::parquet::read_parquet_string_columns_by_name;
use matrix_util::traits::IoOps;
use std::path::Path;

#[derive(Parser)]
struct Wrap<A: clap::Args> {
    #[command(flatten)]
    args: A,
}

fn parse_args<A: clap::Args>(argv: &[&str]) -> A {
    Wrap::<A>::parse_from(argv).args
}

fn write(dir: &Path, name: &str, body: &str) -> String {
    let p = dir.join(name);
    std::fs::write(&p, body).unwrap();
    p.to_string_lossy().into_owned()
}

fn gene_kind() -> FeatureNameKind {
    FeatureNameKind::Gene { delim: '_' }
}

#[test]
fn clap_defaults_are_the_published_recipe_at_the_workspace_dimension() {
    let a: FneArgs = parse_args(&["fne", "-o", "x"]);
    assert_eq!(a.embedding_dim, 128);
    assert_eq!(a.epochs, 10);
    assert_eq!(a.learning_rate, 0.1);
    assert_eq!(a.batch_size, 1000);
    assert_eq!(a.num_batch_negs, 50);
    assert_eq!(a.num_uniform_negs, 50);
    assert_eq!(a.weight_decay, None);
    assert_eq!(a.wd_interval, 50);
    assert_eq!(a.eval_fraction, 0.05);
    assert!(a.networks.is_empty());
    assert!(a.edges.is_empty());
    let b: FneArgs = parse_args(&[
        "fne",
        "ppi.tsv,string.tsv",
        "--edges",
        "a.tsv,b.tsv",
        "--relation-weight",
        "gene:word=0.5,gene:gene/ppi=2",
        "--lr",
        "0.05",
        "--feature-name-exact",
        "-o",
        "x",
    ]);
    assert_eq!(b.networks.len(), 2);
    assert_eq!(b.edges.len(), 2);
    assert_eq!(b.relation_weight.len(), 2);
    assert_eq!(b.learning_rate, 0.05);
    assert!(matches!(b.name_kind(), FeatureNameKind::Exact));
    // The serde default (for manifests missing a field) is the clap default.
    let d: FneArgs = serde_json::from_str("{}").unwrap();
    assert_eq!(d.embedding_dim, 128);
}

#[test]
fn the_relation_stem_drops_known_extensions_but_keeps_dots_inside_the_name() {
    use super::graph::file_stem;
    assert_eq!(file_stem("/x/y/biogrid.tsv"), "biogrid");
    assert_eq!(
        file_stem("BIOGRID-Homo_sapiens-5.0.256.unique_pairs.protein_coding.tsv.gz"),
        "BIOGRID-Homo_sapiens-5.0.256.unique_pairs.protein_coding"
    );
    assert_eq!(file_stem("goa_human.GAF.GZ"), "goa_human");
    assert_eq!(
        file_stem("c2.cp.reactome.v2025.1.Hs.symbols.gmt"),
        "c2.cp.reactome.v2025.1.Hs.symbols"
    );
    assert_eq!(file_stem("noext"), "noext");
    assert_eq!(file_stem(".tsv"), ".tsv", "a bare extension is a name");
}

#[test]
fn a_pair_file_becomes_one_undirected_gene_relation_with_weights_and_canonical_names() {
    let dir = tempfile::tempdir().unwrap();
    let p = write(
        dir.path(),
        "biogrid.tsv",
        "# comment\nTP53\tMDM2\t2.0\nMDM2\tTP53\t0.5\nENSG0001_TP53\tTP53\nTP53\tBAX\nonly_one\n",
    );
    let mut b = TypedGraphBuilder::new(gene_kind());
    b.add_pair_file(&p).unwrap();
    let g = b.finish().unwrap();
    assert_eq!(g.types.len(), 1);
    assert_eq!(g.types.name(0), "gene");
    assert_eq!(
        g.node_names,
        vec![Box::from("TP53"), Box::from("MDM2"), Box::from("BAX")]
    );
    assert_eq!(g.relations.len(), 1);
    let r = g.relations.get(0);
    assert_eq!(r.name.as_ref(), "gene:gene/biogrid");
    assert!(r.undirected);
    // TP53–MDM2 kept once at its larger weight; the ENSG alias was a self-loop.
    assert_eq!(g.edges.len(), 2);
    let w = g
        .edges
        .weight
        .as_ref()
        .expect("a non-unit weight was given");
    let mut pairs: Vec<(u32, u32, f32)> = (0..2)
        .map(|i| (g.edges.lhs[i], g.edges.rhs[i], w[i]))
        .collect();
    pairs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(pairs, vec![(0, 1, 2.0), (0, 2, 1.0)]);
}

#[test]
fn a_typed_file_groups_rows_by_type_pair_and_matches_non_gene_names_exactly() {
    let dir = tempfile::tempdir().unwrap();
    let p = write(
        dir.path(),
        "typed.tsv",
        "gene\tTP53\tterm\tGO:1\t0.9\n\
         gene\tENSG_TP53\tterm\tGO:2\n\
         gene\tBAX\tcell_type\tT cell\n\
         gene\tBAX\tcell_type\tT cell\n\
         term\tGO:1\tterm\tGO:2\n\
         term\tGO:2\tterm\tGO:1\n\
         word\tapoptosis\tgene\tTP53\t0.3\n\
         short\trow\n",
    );
    let mut b = TypedGraphBuilder::new(gene_kind());
    b.add_typed_file(&p).unwrap();
    assert!(b.set_relation_weight("gene:term=2").is_ok());
    assert!(
        b.set_relation_weight("gene:pathway=2").is_err(),
        "unknown relation"
    );
    assert!(b.set_relation_weight("gene:term").is_err(), "no `=`");
    assert!(b.set_relation_weight("gene:term=-1").is_err(), "negative");
    let g = b.finish().unwrap();
    let names: Vec<&str> = g.types.names().iter().map(AsRef::as_ref).collect();
    assert_eq!(names, vec!["gene", "term", "cell_type", "word"]);
    let rels: Vec<(&str, bool, f32)> = g
        .relations
        .iter()
        .map(|r| (r.name.as_ref(), r.undirected, r.weight))
        .collect();
    assert_eq!(
        rels,
        vec![
            ("gene:term", false, 2.0),
            ("gene:cell_type", false, 1.0),
            ("term:term", true, 1.0),
            ("word:gene", false, 1.0),
        ]
    );
    // 2 gene:term + 1 gene:cell_type (repeat dropped) + 1 term:term (both
    // directions are one undirected edge) + 1 word:gene.
    assert_eq!(g.edges.counts_per_relation(4), vec![2, 1, 1, 1]);
    assert_eq!(
        g.node_names[g.types.range(2).start as usize].as_ref(),
        "T cell"
    );
    g.edges.validate(&g.types, &g.relations).unwrap();
}

/// Two gene cliques, each marking its own cell type; a typed file adds a
/// term per clique.
fn planted_inputs(dir: &Path) -> (String, String) {
    let mut ppi = String::new();
    for grp in 0..2 {
        for i in 0..5 {
            for j in (i + 1)..5 {
                ppi.push_str(&format!("G{}\tG{}\n", grp * 5 + i, grp * 5 + j));
            }
        }
    }
    ppi.push_str("G4\tG5\t0.1\n");
    let mut typed = String::new();
    for g in 0..10 {
        let grp = usize::from(g >= 5);
        typed.push_str(&format!("gene\tG{g}\tcell_type\tCT{grp}\n"));
        typed.push_str(&format!("gene\tG{g}\tterm\tGO:{grp}\n"));
    }
    (write(dir, "ppi.tsv", &ppi), write(dir, "annot.tsv", &typed))
}

#[test]
fn fne_writes_typed_artifacts_and_a_manifest_and_places_genes_with_their_own_type() {
    let dir = tempfile::tempdir().unwrap();
    let (ppi, typed) = planted_inputs(dir.path());
    let out = dir.path().join("run").to_string_lossy().into_owned();
    let args: FneArgs = parse_args(&[
        "fne",
        &ppi,
        "--edges",
        &typed,
        "--relation-weight",
        "gene:cell_type=2",
        "--embedding-dim",
        "8",
        "-i",
        "60",
        "--batch-size",
        "16",
        "--num-batch-negs",
        "4",
        "--num-uniform-negs",
        "4",
        "--weight-decay",
        "0",
        "--eval-fraction",
        "0.1",
        "--seed",
        "1",
        "-o",
        &out,
    ]);
    fit_fne(&args).unwrap();

    let emb = Mat::from_parquet(&format!("{out}.feature_embedding.parquet")).unwrap();
    assert_eq!(emb.mat.nrows(), 14, "10 genes + 2 cell types + 2 terms");
    assert_eq!(emb.mat.ncols(), 8);
    assert_eq!(emb.cols[0].as_ref(), "h0");
    let types = read_parquet_string_columns_by_name(
        &format!("{out}.feature_types.parquet"),
        &["feature", "type"],
    )
    .unwrap();
    assert_eq!(types[0], emb.rows, "types table is row-aligned");
    let type_of = |name: &str| -> &str {
        let i = emb.rows.iter().position(|r| r.as_ref() == name).unwrap();
        types[1][i].as_ref()
    };
    assert_eq!(type_of("G0"), "gene");
    assert_eq!(type_of("CT1"), "cell_type");
    assert_eq!(type_of("GO:0"), "term");

    let rels = read_parquet_string_columns_by_name(
        &format!("{out}.relations.parquet"),
        &["relation", "lhs_type", "rhs_type"],
    )
    .unwrap();
    assert_eq!(
        rels[0],
        vec![
            Box::from("gene:gene/ppi"),
            Box::from("gene:cell_type"),
            Box::from("gene:term")
        ]
    );
    assert_eq!(rels[2][1].as_ref(), "cell_type");
    let rel_num = Mat::from_parquet(&format!("{out}.relations.parquet")).unwrap();
    let col = |c: &str| rel_num.cols.iter().position(|x| x.as_ref() == c).unwrap();
    assert_eq!(rel_num.mat[(1, col("weight"))], 2.0);
    assert_eq!(rel_num.mat[(0, col("n_edges"))], 21.0);
    assert_eq!(
        rel_num.mat[(0, col("n_train"))] + rel_num.mat[(0, col("n_eval"))],
        21.0
    );
    for r in 0..3 {
        assert!(rel_num.mat[(r, col("train_loss"))].is_finite());
        assert!(rel_num.mat[(r, col("eval_loss"))].is_finite());
    }

    let ll = Mat::from_parquet(&format!("{out}.log_likelihood.parquet")).unwrap();
    assert_eq!(ll.mat.nrows(), 60);
    let lc = |c: &str| ll.cols.iter().position(|x| x.as_ref() == c).unwrap();
    assert!(ll.mat[(0, lc("train_loss"))] > ll.mat[(59, lc("train_loss"))]);
    assert!(ll.mat[(59, lc("eval_loss"))].is_finite());

    let (m, _dir) = RunManifest::load(Path::new(&format!("{out}.senna.json"))).unwrap();
    assert_eq!(m.kind, RunKind::Fne);
    let ta = m.train_args.as_ref().expect("train args recorded");
    assert_eq!(ta.args["embedding_dim"], 8);
    assert_eq!(ta.args["relation_weight"][0], "gene:cell_type=2");

    // Every gene scores its own cell type and term above the other's.
    let row = |name: &str| -> Vec<f32> {
        let i = emb.rows.iter().position(|r| r.as_ref() == name).unwrap();
        emb.mat.row(i).iter().copied().collect()
    };
    let dot = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
    for g in 0..10 {
        let grp = usize::from(g >= 5);
        let e = row(&format!("G{g}"));
        assert!(
            dot(&e, &row(&format!("CT{grp}"))) > dot(&e, &row(&format!("CT{}", 1 - grp))),
            "G{g} sits with its own cell type"
        );
        assert!(
            dot(&e, &row(&format!("GO:{grp}"))) > dot(&e, &row(&format!("GO:{}", 1 - grp))),
            "G{g} sits with its own term"
        );
    }
    assert!(!Path::new(&format!("{out}.feature_bias.parquet")).exists());
    assert!(!Path::new(&format!("{out}.gamma.parquet")).exists());
}

#[test]
fn fne_refuses_to_run_without_any_input_or_with_no_usable_edges() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("run").to_string_lossy().into_owned();
    let none: FneArgs = parse_args(&["fne", "-o", &out]);
    assert!(fit_fne(&none).is_err());
    let loops = write(dir.path(), "loops.tsv", "A\tA\nB\tB\n");
    let only_loops: FneArgs = parse_args(&["fne", &loops, "-o", &out]);
    assert!(fit_fne(&only_loops).is_err());
}

/// go-basic-shaped OBO: root → process → {apoptosis, proliferation};
/// apoptosis part_of "death programme"; one definition each on the leaves.
const OBO: &str = "format-version: 1.2\n\n\
[Term]\nid: GO:0\nname: biological_process\n\n\
[Term]\nid: GO:1\nname: cellular process\nis_a: GO:0 ! biological_process\n\n\
[Term]\nid: GO:2\nname: apoptotic process\ndef: \"A programmed cell death.\" [GOC:x]\nis_a: GO:1 ! cellular process\nrelationship: part_of GO:4 ! death\n\n\
[Term]\nid: GO:3\nname: cell proliferation\ndef: \"Cells multiply.\" [GOC:y]\nis_a: GO:1 ! cellular process\n\n\
[Term]\nid: GO:4\nname: death programme\n\n";

/// GAF rows (17 columns; 2 = accession, 3 = symbol, 4 = qualifier, 5 = GO id,
/// 7 = evidence, 11 = synonyms).
fn gaf_row(symbol: &str, go: &str, evidence: &str) -> String {
    let mut cols = vec![""; 17];
    cols[0] = "UniProtKB";
    cols[1] = "P00000";
    cols[2] = symbol;
    cols[3] = "involved_in";
    cols[4] = go;
    cols[5] = "PMID:1";
    cols[6] = evidence;
    cols[8] = "P";
    cols[10] = symbol;
    cols[11] = "protein";
    cols[12] = "taxon:9606";
    cols[13] = "20200101";
    cols[14] = "UniProt";
    format!("{}\n", cols.join("\t"))
}

#[test]
fn a_membership_file_becomes_a_gene_to_label_relation_named_after_the_file() {
    let dir = tempfile::tempdir().unwrap();
    let p = write(
        dir.path(),
        "markers.tsv",
        "gene\tcelltype\nCD3E\tT cell\nCD3D\tT cell\nMS4A1\tB cell\nCD3E\tT cell\n",
    );
    let mut b = TypedGraphBuilder::new(gene_kind());
    assert!(
        b.add_membership_file("gene", &p).is_err(),
        "labels cannot be genes"
    );
    assert!(b.add_membership_file("", &p).is_err());
    b.add_membership_file("cell_type", &p).unwrap();
    let g = b.finish().unwrap();
    let r = g.relations.get(0);
    assert_eq!(r.name.as_ref(), "gene:cell_type/markers");
    assert!(!r.undirected);
    assert_eq!(g.edges.len(), 3, "the repeated row is one edge");
    assert_eq!(g.types.n_nodes(g.types.index_of("cell_type").unwrap()), 2);
    assert_eq!(
        g.node_names[3].as_ref(),
        "T cell",
        "labels verbatim, spaces kept"
    );
    assert!(g.edges.weight.is_none());
}

#[test]
fn gene_sets_from_a_gaf_propagate_up_the_ontology_and_the_hierarchy_joins_as_term_edges() {
    let dir = tempfile::tempdir().unwrap();
    let obo = write(dir.path(), "go.obo", OBO);
    let gaf = write(
        dir.path(),
        "goa.gaf",
        &format!(
            "!gaf-version: 2.2\n{}{}{}{}",
            gaf_row("TP53", "GO:2", "IDA"),
            gaf_row("BAX", "GO:2", "IEA"),
            gaf_row("MYC", "GO:3", "IDA"),
            gaf_row("CCND1", "GO:3", "IDA"),
        ),
    );
    let onto = Ontology::load_obo(&obo).unwrap();
    let sets = read_gaf(&gaf, &GafOpts { no_iea: false })
        .unwrap()
        .into_gene_sets(Some(&onto));
    let mut b = TypedGraphBuilder::new(gene_kind());
    // Cap at 3 members: the root (4 genes) is dropped, GO:1 (4) too; the
    // leaves (2) and the part_of parent GO:4 (2) stay; the floor drops nothing.
    let kept = b.add_gene_sets(&sets, "goa", 1, 3);
    assert_eq!(kept, 3, "GO:2, GO:3 and GO:4");
    b.add_ontology(&onto);
    let g = b.finish().unwrap();
    let names: Vec<&str> = g.relations.iter().map(|r| r.name.as_ref()).collect();
    // 2 + 2 + 2 membership edges. GO:2 → GO:1 and GO:3 → GO:1 point at a
    // dropped term, so no is_a edge survives and that relation is pruned;
    // GO:2 part_of GO:4 does survive.
    assert_eq!(names, vec!["gene:term/goa", "term:term/part_of"]);
    assert_eq!(g.edges.counts_per_relation(2), vec![6, 1]);
    let term_t = g.types.index_of("term").unwrap();
    let term_names: Vec<&str> = (g.types.range(term_t))
        .map(|i| g.node_names[i as usize].as_ref())
        .collect();
    assert_eq!(term_names, vec!["GO:2", "GO:3", "GO:4"]);
    let texts: Vec<(&str, &NodeText)> = g
        .texts
        .iter()
        .map(|(i, t)| (g.node_names[*i as usize].as_ref(), t))
        .collect();
    assert_eq!(texts.len(), 3);
    assert_eq!(
        texts[0],
        (
            "GO:2",
            &NodeText {
                name: Some("apoptotic process".into()),
                text: Some("A programmed cell death.".into())
            }
        )
    );
    assert_eq!(texts[2].1.text, None, "GO:4 has a name but no definition");
}

#[test]
fn gmt_sets_carry_their_description_as_the_term_name() {
    let dir = tempfile::tempdir().unwrap();
    let gmt = write(
        dir.path(),
        "hallmark.gmt",
        "HALLMARK_A\tset A description\tTP53\tBAX\tMDM2\nHALLMARK_B\thttp://x\tMYC\nTINY\t\tA\n",
    );
    let sets = read_gmt(&gmt).unwrap();
    let mut b = TypedGraphBuilder::new(gene_kind());
    let kept = b.add_gene_sets(&sets, "hallmark", 2, 0);
    assert_eq!(kept, 1, "min 2 members, no cap");
    let g = b.finish().unwrap();
    assert_eq!(g.relations.get(0).name.as_ref(), "gene:term/hallmark");
    assert_eq!(g.edges.len(), 3);
    assert_eq!(g.texts.len(), 1);
    assert_eq!(g.texts[0].1.name.as_deref(), Some("set A description"));
}

#[test]
fn region_links_tile_onto_windows_and_carry_their_score() {
    let dir = tempfile::tempdir().unwrap();
    let p = write(
        dir.path(),
        "abc.tsv",
        "# region\tgene\tscore\nchr1:4000-6000\tTP53\t0.8\n1_5500\tTP53\nchrX:100-200\tMYC\t0.2\nnot_a_region\tMYC\n",
    );
    let mut b = TypedGraphBuilder::new(gene_kind());
    b.add_region_file(&p, 5000).unwrap();
    let g = b.finish().unwrap();
    let r = g.relations.get(0);
    assert_eq!(r.name.as_ref(), "region:gene/abc");
    assert_eq!(g.types.name(r.lhs_type as usize), "region");
    let region_t = g.types.index_of("region").unwrap();
    let windows: Vec<&str> = (g.types.range(region_t))
        .map(|i| g.node_names[i as usize].as_ref())
        .collect();
    assert_eq!(windows, vec!["1:0-5000", "1:5000-10000", "X:0-5000"]);
    // chr1:4000-6000 → two windows at 0.8; 1_5500 → the second window at 1.0,
    // which wins over 0.8 for that pair; chrX → one window at 0.2.
    let mut edges: Vec<(&str, &str, f32)> = (0..g.edges.len())
        .map(|i| {
            (
                g.node_names[g.edges.lhs[i] as usize].as_ref(),
                g.node_names[g.edges.rhs[i] as usize].as_ref(),
                g.edges.edge_weight(i),
            )
        })
        .collect();
    edges.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(
        edges,
        vec![
            ("1:0-5000", "TP53", 0.8),
            ("1:5000-10000", "TP53", 1.0),
            ("X:0-5000", "MYC", 0.2),
        ]
    );
}

#[test]
fn fne_takes_every_side_information_source_at_once_and_exports_the_text() {
    let dir = tempfile::tempdir().unwrap();
    let (ppi, _typed) = planted_inputs(dir.path());
    let markers = write(
        dir.path(),
        "markers.tsv",
        "G0\tA\nG1\tA\nG2\tA\nG5\tB\nG6\tB\nG7\tB\n",
    );
    let obo = write(dir.path(), "go.obo", OBO);
    let gaf = write(
        dir.path(),
        "goa.gaf",
        &(0..10)
            .map(|g| gaf_row(&format!("G{g}"), if g < 5 { "GO:2" } else { "GO:3" }, "IDA"))
            .collect::<String>(),
    );
    let regions = write(
        dir.path(),
        "eqtl.tsv",
        "chr1:1000\tG0\t0.5\nchr1:2000\tG1\nchr2:1000\tG5\n",
    );
    let out = dir.path().join("run").to_string_lossy().into_owned();
    let text = dir.path().join("text.tsv").to_string_lossy().into_owned();
    let args: FneArgs = parse_args(&[
        "fne",
        &ppi,
        "--membership",
        &format!("cell_type={markers}"),
        "--gaf",
        &gaf,
        "--obo",
        &obo,
        "--min-gene-set",
        "2",
        "--max-gene-set",
        "8",
        "--region-gene",
        &regions,
        "--region-window",
        "5000",
        "--export-text",
        &text,
        "--embedding-dim",
        "8",
        "-i",
        "5",
        "--batch-size",
        "16",
        "--num-batch-negs",
        "4",
        "--num-uniform-negs",
        "4",
        "--weight-decay",
        "0",
        "--eval-fraction",
        "0",
        "-o",
        &out,
    ]);
    fit_fne(&args).unwrap();
    let types = read_parquet_string_columns_by_name(
        &format!("{out}.feature_types.parquet"),
        &["feature", "type"],
    )
    .unwrap();
    let mut kinds: Vec<&str> = types[1].iter().map(AsRef::as_ref).collect();
    kinds.sort_unstable();
    kinds.dedup();
    assert_eq!(kinds, vec!["cell_type", "gene", "region", "term"]);
    let rels =
        read_parquet_string_columns_by_name(&format!("{out}.relations.parquet"), &["relation"])
            .unwrap();
    let rel_names: Vec<&str> = rels[0].iter().map(AsRef::as_ref).collect();
    // GO:2 and GO:3 (5 genes each) and GO:1 (10 > 8) — the leaves stay, so do
    // GO:4 (5) ; is_a edges to the dropped GO:1 vanish, the part_of edge stays.
    assert_eq!(
        rel_names,
        vec![
            "gene:gene/ppi",
            "gene:cell_type/markers",
            "gene:term/goa",
            "term:term/part_of",
            "region:gene/eqtl"
        ]
    );
    let exported = std::fs::read_to_string(&text).unwrap();
    let lines: Vec<&str> = exported.lines().collect();
    assert_eq!(lines[0], "feature\ttype\tname\ttext");
    assert!(lines.contains(&"GO:2\tterm\tapoptotic process\tA programmed cell death."));
    assert!(lines.contains(&"GO:4\tterm\tdeath programme\t"));
    assert_eq!(lines.len(), 4, "header + three terms with text");
    let (m, _dir) = RunManifest::load(Path::new(&format!("{out}.senna.json"))).unwrap();
    assert!(m.data.input.iter().any(|p| p.ends_with("goa.gaf")));
    assert!(m.data.input.iter().any(|p| p.ends_with("eqtl.tsv")));

    // --gaf without --obo is refused up front.
    let bad: FneArgs = parse_args(&["fne", "--gaf", &gaf, "-o", &out]);
    assert!(fit_fne(&bad).is_err());
}
