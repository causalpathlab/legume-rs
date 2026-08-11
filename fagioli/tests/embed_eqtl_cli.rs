//! End-to-end check of the `embed-eqtl` command line on synthetic input:
//! the five outputs appear, the census is populated, and the specificity
//! table carries one column per fitted context.

use std::io::Read;
use std::path::PathBuf;
use std::process::Command;

const HEADER: &str = "#chromosome\tphysical.pos\tgene\tcelltype\tbeta\tse";

fn celltypes() -> Vec<String> {
    (0..4)
        .map(|k| format!("L1-{k}"))
        .chain((0..4).map(|k| format!("L2-{k}")))
        .collect()
}

/// Twelve genes, six variants each. The lead variant carries a group-wide
/// effect; the rest are precise nulls, and every variant is also tested
/// against the neighbouring gene so the gene-swap pool is non-empty.
fn synthetic_table() -> String {
    let cts = celltypes();
    let mut out = String::from(HEADER);
    out.push('\n');
    for gene in 0..12u32 {
        let first_group = gene % 2 == 0;
        for variant in 0..6u32 {
            let pos = gene * 1000 + variant * 10;
            for (k, ct) in cts.iter().enumerate() {
                let hot = if first_group { k < 4 } else { k >= 4 };
                let beta = if variant == 0 && hot { 0.5 } else { 0.0 };
                out.push_str(&format!("1\t{pos}\tG{gene}\t{ct}\t{beta}\t0.05\n"));
                // The same locus against the next gene: a null it does not
                // affect, which is what a gene-swap negative looks like.
                let other = (gene + 1) % 12;
                out.push_str(&format!("1\t{pos}\tG{other}\t{ct}\t0.0\t0.05\n"));
            }
        }
    }
    out
}

fn read_gz(path: &PathBuf) -> String {
    let file = std::fs::File::open(path).unwrap();
    let mut decoder = flate2::read::MultiGzDecoder::new(file);
    let mut text = String::new();
    decoder.read_to_string(&mut text).unwrap();
    text
}

#[test]
fn embed_eqtl_writes_every_output_from_a_synthetic_table() {
    let tmp = tempfile::TempDir::new().unwrap();
    let input = tmp.path().join("block.tsv");
    std::fs::write(&input, synthetic_table()).unwrap();
    let out = tmp.path().join("cli").to_str().unwrap().to_string();

    let status = Command::new(env!("CARGO_BIN_EXE_fagioli"))
        .args(["embed-eqtl", "--qtl-files"])
        .arg(&input)
        .args([
            "--top-k",
            "2",
            "--embedding-dim",
            "4",
            "--num-iterations",
            "300",
            "--batch-size",
            "32",
            "--output",
            &out,
        ])
        .status()
        .unwrap();
    assert!(status.success(), "embed-eqtl exited nonzero");

    for suffix in [
        "variant_embedding.parquet",
        "gene_embedding.parquet",
        "context_embedding.parquet",
        "specificity.tsv.gz",
        "parameters.json",
    ] {
        let path = PathBuf::from(format!("{out}.{suffix}"));
        assert!(path.is_file(), "missing output {path:?}");
        assert!(path.metadata().unwrap().len() > 0, "empty output {path:?}");
    }

    let params: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{out}.parameters.json")).unwrap())
            .unwrap();
    let census = &params["state_census"];
    assert!(census["edge"].as_u64().unwrap() > 0);
    assert!(census["certified_absent"].as_u64().unwrap() > 0);
    // Eight cell types plus the ubiquitous pseudo-context.
    assert_eq!(census["n_contexts"].as_u64().unwrap(), 9);
    // Top-2 of six variants per gene.
    assert!(params["selection"]["n_pairs_dropped"].as_u64().unwrap() > 0);
    assert!(params["diagnostics"]["auc_heldout"].as_f64().unwrap() > 0.5);
    // FitMetrics is serialized whole, so the counts land beside the
    // diagnostics rather than in a section of their own.
    assert!(params["diagnostics"]["n_positives"].as_u64().unwrap() > 0);

    let specificity = read_gz(&PathBuf::from(format!("{out}.specificity.tsv.gz")));
    let mut lines = specificity.lines();
    let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
    assert_eq!(header[0], "variant");
    assert_eq!(header[1], "gene");
    assert_eq!(header[2], "anchor");
    // Nine fixed columns, then one per real context.
    assert_eq!(header.len(), 9 + 8);
    assert!(header[9..]
        .iter()
        .all(|c| c.starts_with("L1-") || c.starts_with("L2-")));
    assert!(lines.next().is_some(), "no specificity rows written");
}
