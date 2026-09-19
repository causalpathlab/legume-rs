//! `senna fne` end to end: edge files of both shapes become one typed
//! graph, the artifacts carry every node type, and the manifest records
//! the fit.

use super::graph::{NodeText, PpiOpts, TypedGraphBuilder};
use super::{fit_fne, FneArgs};
use auxiliary_data::feature_names::FeatureNameKind;
use auxiliary_data::gene_sets::{read_gaf, read_gmt, GafOpts};
use auxiliary_data::ontology::Ontology;
use clap::Parser;
use matrix_util::parquet::read_parquet_string_columns_by_name;
use matrix_util::traits::IoOps;
use senna::embed_common::Mat;
use senna::run_manifest::{RunKind, RunManifest};
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
    assert_eq!(
        a.embedding_dim,
        graph_embedding_util::EmbeddingDim::Fixed(128)
    );
    assert_eq!(a.train.epochs, 10);
    assert_eq!(a.train.learning_rate, 0.1);
    assert_eq!(a.train.batch_size, 1000);
    assert_eq!(a.train.num_batch_negs, 50);
    assert_eq!(a.train.num_uniform_negs, 50);
    assert_eq!(a.train.weight_decay, None);
    assert_eq!(a.train.wd_interval, 50);
    assert_eq!(a.train.eval_fraction, 0.05);
    assert!(a.networks.is_empty());
    assert!(a.named_pairs.is_empty());
    assert!(a.relation_polarity.is_empty());
    assert!(a.edges.is_empty());
    let b: FneArgs = parse_args(&[
        "fne",
        "ppi.tsv,string.tsv",
        "--named-pairs",
        "mixed.tsv",
        "--edges",
        "a.tsv,b.tsv",
        "--relation-weight",
        "gene:word=0.5,gene:gene/ppi=2",
        "--relation-polarity",
        "gene:gene/genetic_sl=enemy",
        "--ppi-max-degree",
        "50",
        "--ppi-snn-k",
        "2",
        "--no-ppi-ppr",
        "--lr",
        "0.05",
        "--feature-name-kind",
        "exact",
        "-o",
        "x",
    ]);
    assert_eq!(b.networks.len(), 2);
    assert_eq!(b.named_pairs.len(), 1);
    assert_eq!(b.edges.len(), 2);
    assert_eq!(b.relation_weight.len(), 2);
    assert_eq!(b.relation_polarity.len(), 1);
    assert_eq!(b.train.learning_rate, 0.05);
    assert_eq!((b.ppi_max_degree, b.ppi_snn_k, b.no_ppi_ppr), (50, 2, true));
    assert_eq!(b.ppi_ppr_restart, 0.15);
    // Derived relations are on by default; the QC prunes are off.
    assert!(!a.no_ppi_snn && !a.no_ppi_ppr);
    assert_eq!(
        (a.ppi_snn_k, a.ppi_ppr_k, a.ppi_snn_min_shared),
        (10, 10, 1)
    );
    assert_eq!(
        (
            a.ppi_min_shared_neighbors,
            a.ppi_max_degree,
            a.ppi_min_degree
        ),
        (0, 0, 0)
    );
    assert!(matches!(b.name_kind(), FeatureNameKind::Exact));
    // The serde default (for manifests missing a field) is the clap default.
    let d: FneArgs = serde_json::from_str("{}").unwrap();
    assert_eq!(
        d.embedding_dim,
        graph_embedding_util::EmbeddingDim::Fixed(128)
    );
}

#[test]
fn the_relation_stem_drops_known_extensions_but_keeps_dots_inside_the_name() {
    use matrix_util::common_io::file_stem;
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
    b.add_pair_file(&p, &PpiOpts::default()).unwrap();
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
fn named_pairs_split_into_relations_and_polarity_overrides_take_full_ids() {
    use graph_embedding_util::fne::RelationPolarity;
    let dir = tempfile::tempdir().unwrap();
    let p = write(
        dir.path(),
        "mixed.tsv",
        "TP53\tMDM2\tphysical\t2.0\nA\tB\tgenetic_sl\nA\tC\tgenetic_sl\t0.5\n",
    );
    let mut b = TypedGraphBuilder::new(gene_kind());
    b.add_named_pair_file(&p).unwrap();
    b.set_relation_polarity("gene:gene/genetic_sl=enemy")
        .unwrap();
    assert!(b.set_relation_polarity("gene:gene/nope=enemy").is_err());
    assert!(b.set_relation_polarity("gene:gene/physical=maybe").is_err());
    let g = b.finish().unwrap();
    let names: Vec<&str> = g.relations.iter().map(|r| r.name.as_ref()).collect();
    assert_eq!(names, vec!["gene:gene/physical", "gene:gene/genetic_sl"]);
    assert_eq!(g.relations.get(0).polarity, RelationPolarity::Friend);
    assert_eq!(g.relations.get(1).polarity, RelationPolarity::Enemy);
    assert_eq!(g.edges.counts_per_relation(2), vec![1, 2]);
}

/// Two triangles A,B,C and D,E,F joined by C–D, a pendant G on A, and a
/// lone noisy edge H–I: the shape every PPI QC rule bites on.
fn ppi_fixture(dir: &Path) -> String {
    write(
        dir,
        "ppi.tsv",
        "A\tB\nA\tC\nB\tC\nC\tD\nD\tE\nD\tF\nE\tF\nA\tG\nH\tI\n",
    )
}

#[test]
fn ppi_qc_prunes_uncorroborated_edges_and_the_derived_relations_carry_their_weights() {
    let dir = tempfile::tempdir().unwrap();
    let p = ppi_fixture(dir.path());
    // Shared-neighbour QC at 1: A–G, C–D and H–I have no common partner.
    let mut b = TypedGraphBuilder::new(gene_kind());
    b.add_pair_file(
        &p,
        &PpiOpts {
            min_shared_neighbors: 1,
            ..PpiOpts::default()
        },
    )
    .unwrap();
    let g = b.finish().unwrap();
    assert_eq!(g.relations.len(), 1);
    assert_eq!(
        g.edges.len(),
        6,
        "two triangles survive, the three bridges do not"
    );

    // No QC, second-order and diffusion relations on top of the raw one.
    let mut b = TypedGraphBuilder::new(gene_kind());
    b.add_pair_file(
        &p,
        &PpiOpts {
            snn_k: 10,
            snn_min_shared: 1,
            ppr_k: 2,
            ppr_restart: 0.15,
            ..PpiOpts::default()
        },
    )
    .unwrap();
    let g = b.finish().unwrap();
    let names: Vec<&str> = g.relations.iter().map(|r| r.name.as_ref()).collect();
    assert_eq!(
        names,
        vec!["gene:gene/ppi", "gene:gene/ppi/snn", "gene:gene/ppi/ppr"]
    );
    assert!(g.relations.get(1).undirected && g.relations.get(2).undirected);
    let counts = g.edges.counts_per_relation(3);
    assert_eq!(counts[0], 9, "raw edges untouched");
    // SNN pairs: (B,G),(C,G) via A; (A,D),(B,D) via C; (C,E),(C,F) via D.
    assert_eq!(counts[1], 6);
    let name = |i: u32| g.node_names[i as usize].as_ref();
    let w = g.edges.weight.as_ref().expect("weighted relations");
    for (&rel, &wt) in g.edges.rel.iter().zip(w) {
        match rel {
            1 => assert!(
                wt > 0.0 && wt < 1.0,
                "Jaccard overlap of two distinct neighbourhoods: {wt}"
            ),
            2 => assert!(
                wt > 0.0 && wt <= 1.0,
                "ppr weights relative to the strongest target"
            ),
            _ => assert_eq!(wt, 1.0),
        }
    }
    // Every node with an edge has PPR targets; H and I only reach each other,
    // so the undirected fold leaves one pair for them.
    let ppr_pairs: Vec<(&str, &str)> = (0..g.edges.len())
        .filter(|&i| g.edges.rel[i] == 2)
        .map(|i| (name(g.edges.lhs[i]), name(g.edges.rhs[i])))
        .collect();
    assert!(ppr_pairs.contains(&("H", "I")) || ppr_pairs.contains(&("I", "H")));
    assert!(ppr_pairs
        .iter()
        .any(|&(a, b)| (a == "A" && b == "B") || (a == "B" && b == "A")));
    assert!(
        counts[2] >= 8,
        "at least one pair per connected node: {}",
        counts[2]
    );
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
    assert!(b.set_relation_repeat("gene:term=4").is_ok());
    assert!(
        b.set_relation_repeat("gene:term=0").is_err(),
        "at least one pass"
    );
    assert!(
        b.set_relation_repeat("gene:nope=2").is_err(),
        "unknown relation"
    );
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
    assert_eq!(g.relation_repeats, vec![4, 1, 1, 1]);
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
        "--relation-repeat",
        "gene:term=3",
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
        &["relation", "lhs_type", "rhs_type", "polarity"],
    )
    .unwrap();
    assert_eq!(
        rels[0],
        vec![
            Box::from("gene:gene/ppi"),
            Box::from("gene:gene/ppi/snn"),
            Box::from("gene:gene/ppi/ppr"),
            Box::from("gene:cell_type"),
            Box::from("gene:term")
        ]
    );
    assert!(rels[3].iter().all(|p| p.as_ref() == "friend"));
    assert_eq!(rels[2][3].as_ref(), "cell_type");
    let rel_num = Mat::from_parquet(&format!("{out}.relations.parquet")).unwrap();
    let col = |c: &str| rel_num.cols.iter().position(|x| x.as_ref() == c).unwrap();
    assert_eq!(rel_num.mat[(3, col("weight"))], 2.0);
    assert_eq!(rel_num.mat[(4, col("repeat"))], 3.0);
    assert_eq!(rel_num.mat[(0, col("repeat"))], 1.0);
    assert_eq!(rel_num.mat[(0, col("n_edges"))], 21.0);
    assert_eq!(
        rel_num.mat[(0, col("n_train"))] + rel_num.mat[(0, col("n_eval"))],
        21.0
    );
    for r in 0..5 {
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
            "gene:gene/ppi/snn",
            "gene:gene/ppi/ppr",
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

/// Freezing to an earlier run's table: the matched gene rows come out of a
/// second run exactly as the first run wrote them, at the first run's H even
/// with `--embedding-dim auto`, while the other nodes still train.
#[test]
fn fne_pins_gene_rows_to_an_earlier_runs_feature_embedding() {
    let dir = tempfile::tempdir().unwrap();
    let (ppi, typed) = planted_inputs(dir.path());
    let first = dir.path().join("first").to_string_lossy().into_owned();
    let second = dir.path().join("second").to_string_lossy().into_owned();
    let common = [
        "--batch-size",
        "16",
        "--num-batch-negs",
        "4",
        "--num-uniform-negs",
        "4",
        "--eval-fraction",
        "0",
    ];
    let mut argv = vec![
        "fne",
        &ppi,
        "--edges",
        &typed,
        "--embedding-dim",
        "6",
        "-i",
        "3",
    ];
    argv.extend_from_slice(&common);
    argv.extend_from_slice(&["-o", &first]);
    let a: FneArgs = parse_args(&argv);
    fit_fne(&a).unwrap();
    let mut argv = vec![
        "fne",
        &ppi,
        "--edges",
        &typed,
        "--freeze-feature-embedding",
        &first,
        "--embedding-dim",
        "auto",
        "-i",
        "5",
        "--seed",
        "7",
    ];
    argv.extend_from_slice(&common);
    argv.extend_from_slice(&["-o", &second]);
    let b: FneArgs = parse_args(&argv);
    fit_fne(&b).unwrap();

    let e1 = Mat::from_parquet(&format!("{first}.feature_embedding.parquet")).unwrap();
    let e2 = Mat::from_parquet(&format!("{second}.feature_embedding.parquet")).unwrap();
    assert_eq!(e2.mat.ncols(), 6, "H taken from the table");
    let row = |e: &matrix_util::traits::MatWithNames<Mat>, name: &str| -> Vec<f32> {
        let i = e.rows.iter().position(|r| r.as_ref() == name).unwrap();
        e.mat.row(i).iter().copied().collect()
    };
    for g in 0..10 {
        let name = format!("G{g}");
        assert_eq!(row(&e1, &name), row(&e2, &name), "{name} is pinned");
    }
    assert_ne!(row(&e1, "CT0"), row(&e2, "CT0"), "a cell type still trains");
    let m: RunManifest =
        serde_json::from_str(&std::fs::read_to_string(format!("{second}.senna.json")).unwrap())
            .unwrap();
    let recorded = &m.train_args.as_ref().unwrap().args;
    assert_eq!(
        recorded["freeze_feature_embedding"].as_str(),
        Some(first.as_str())
    );
}

/// `--lora-feature-embedding`: the matched gene rows come out of a second run
/// as the first run's rows plus a shared residual of the given rank, at the
/// first run's H, while the other nodes train freely; the manifest records
/// the flag and its knobs.
#[test]
fn fne_anchors_gene_rows_with_a_low_rank_residual() {
    let dir = tempfile::tempdir().unwrap();
    let (ppi, typed) = planted_inputs(dir.path());
    let first = dir.path().join("first").to_string_lossy().into_owned();
    let second = dir.path().join("second").to_string_lossy().into_owned();
    let common = [
        "--batch-size",
        "16",
        "--num-batch-negs",
        "4",
        "--num-uniform-negs",
        "4",
        "--eval-fraction",
        "0",
    ];
    let mut argv = vec![
        "fne",
        &ppi,
        "--edges",
        &typed,
        "--embedding-dim",
        "6",
        "-i",
        "3",
    ];
    argv.extend_from_slice(&common);
    argv.extend_from_slice(&["-o", &first]);
    let a: FneArgs = parse_args(&argv);
    fit_fne(&a).unwrap();
    let mut argv = vec![
        "fne",
        &ppi,
        "--edges",
        &typed,
        "--lora-feature-embedding",
        &first,
        "--lora-rank",
        "1",
        "--lora-lr-ratio",
        "4",
        "--embedding-dim",
        "auto",
        "-i",
        "5",
        "--seed",
        "7",
    ];
    argv.extend_from_slice(&common);
    argv.extend_from_slice(&["-o", &second]);
    let b: FneArgs = parse_args(&argv);
    fit_fne(&b).unwrap();

    let e1 = Mat::from_parquet(&format!("{first}.feature_embedding.parquet")).unwrap();
    let e2 = Mat::from_parquet(&format!("{second}.feature_embedding.parquet")).unwrap();
    assert_eq!(e2.mat.ncols(), 6, "H taken from the table");
    let row = |e: &matrix_util::traits::MatWithNames<Mat>, name: &str| -> Vec<f32> {
        let i = e.rows.iter().position(|r| r.as_ref() == name).unwrap();
        e.mat.row(i).iter().copied().collect()
    };
    let mut resid = Mat::zeros(10, 6);
    for g in 0..10 {
        let name = format!("G{g}");
        let (r1, r2) = (row(&e1, &name), row(&e2, &name));
        for k in 0..6 {
            resid[(g, k)] = r2[k] - r1[k];
        }
    }
    let sv = resid.singular_values();
    assert!(sv[0] > 1e-6, "the residual never moved");
    assert!(sv[1] <= 1e-4 * sv[0], "the residual is not rank 1: {sv}");
    assert_ne!(row(&e1, "CT0"), row(&e2, "CT0"), "a cell type still trains");
    let m: RunManifest =
        serde_json::from_str(&std::fs::read_to_string(format!("{second}.senna.json")).unwrap())
            .unwrap();
    let recorded = &m.train_args.as_ref().unwrap().args;
    assert_eq!(
        recorded["lora_feature_embedding"].as_str(),
        Some(first.as_str())
    );
    assert_eq!(recorded["lora_rank"].as_u64(), Some(1));
}

/// A source table wider than the graph: the rows no node matched — a gene
/// the graph lacks and a term — come out after the graph's own rows,
/// unchanged, and the types table keeps the graph's own node types.
#[test]
fn fne_carries_the_unmatched_rows_of_a_pinned_table_through() {
    use crate::feature_preset::test_support::{assert_carried, widen, EXTRA};
    let dir = tempfile::tempdir().unwrap();
    let (ppi, typed) = planted_inputs(dir.path());
    let first = dir.path().join("first").to_string_lossy().into_owned();
    let plus = dir.path().join("plus").to_string_lossy().into_owned();
    let second = dir.path().join("second").to_string_lossy().into_owned();
    let common = [
        "--batch-size",
        "16",
        "--num-batch-negs",
        "4",
        "--num-uniform-negs",
        "4",
        "--eval-fraction",
        "0",
        "-i",
        "2",
    ];
    let mut argv = vec!["fne", &ppi, "--edges", &typed, "--embedding-dim", "6"];
    argv.extend_from_slice(&common);
    argv.extend_from_slice(&["-o", &first]);
    fit_fne(&parse_args(&argv)).unwrap();
    let extra = widen(&format!("{first}.feature_embedding.parquet"), &plus);

    for mode in ["--freeze-feature-embedding", "--lora-feature-embedding"] {
        let out = format!("{second}-{}", &mode[2..6]);
        let mut argv = vec![
            "fne",
            &ppi,
            "--edges",
            &typed,
            mode,
            &plus,
            "--embedding-dim",
            "auto",
        ];
        if mode.starts_with("--lora") {
            argv.extend_from_slice(&["--lora-rank", "1"]);
        }
        argv.extend_from_slice(&common);
        argv.extend_from_slice(&["-o", &out]);
        fit_fne(&parse_args(&argv)).unwrap();
        let e1 = Mat::from_parquet(&format!("{first}.feature_embedding.parquet")).unwrap();
        let rho = format!("{out}.feature_embedding.parquet");
        assert_carried(&out, &rho, e1.rows.len(), &extra);
        let types = auxiliary_data::feature_types::read_feature_types(&out)
            .unwrap()
            .unwrap();
        let ct0 = types.iter().find(|(n, _)| n.as_ref() == "CT0").unwrap();
        assert_eq!(
            ct0.1.as_ref(),
            "cell_type",
            "the graph's own types are kept"
        );
        // The source's own CT0 / GO:0 rows are superseded by this run's: one row each.
        let e2 = Mat::from_parquet(&rho).unwrap();
        for n in ["CT0", "GO:0", EXTRA[0].0] {
            assert_eq!(e2.rows.iter().filter(|r| r.as_ref() == n).count(), 1, "{n}");
        }
    }
}
