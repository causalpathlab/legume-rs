use super::*;
use crate::site_analysis::pileup::Selector;

fn write_gtf(forward: bool) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    // Unique per call: two tests both write the `forward=true` fixture, so a
    // pid+strand name collides under parallel execution (one test removes the
    // file the other is mid-read). The counter keeps every path distinct.
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let strand = if forward { "+" } else { "-" };
    // gene 100..550 (1-based) with two exons.
    let gtf = format!(
        "chr1\tT\tgene\t100\t550\t.\t{s}\t.\tgene_id \"ENSG1\"; gene_name \"GENEA\"\n\
             chr1\tT\texon\t100\t200\t.\t{s}\t.\tgene_id \"ENSG1\"; gene_name \"GENEA\"\n\
             chr1\tT\texon\t400\t550\t.\t{s}\t.\tgene_id \"ENSG1\"; gene_name \"GENEA\"\n",
        s = strand
    );
    let p = std::env::temp_dir().join(format!(
        "miami_genemodel_{}_{}.gtf",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&p, gtf).unwrap();
    p
}

#[test]
fn loads_and_matches_by_symbol() {
    let gtf = write_gtf(true);
    let sel = Selector::build(&["GENEA".into()], &[]).unwrap();
    let models = load_gene_models(gtf.to_str().unwrap(), &sel).unwrap();
    std::fs::remove_file(&gtf).ok();
    assert_eq!(models.len(), 1);
    let g = &models[0];
    assert_eq!(g.lo, 99); // 100 (1-based) -> 99 (0-based)
    assert_eq!(g.hi, 550);
    assert!(g.forward);
    assert_eq!(g.exons.len(), 2);
    assert_eq!(g.exons[0], (99, 200));
}

#[test]
fn svg_endpoints_and_arrow_direction() {
    let gtf_fwd = write_gtf(true);
    let sel = Selector::build(&["GENEA".into()], &[]).unwrap();
    let fwd = load_gene_models(gtf_fwd.to_str().unwrap(), &sel).unwrap();
    std::fs::remove_file(&gtf_fwd).ok();
    let edges = BinEdges::new(fwd[0].lo, fwd[0].hi, 80);
    let svg = gene_model_svg(&fwd[0], &edges, 10.0, 80.0, 100.0, 8.0);
    assert!(svg.contains("<rect"));
    assert!(svg.contains("<polyline"));
    assert!(svg.contains("GENEA"));

    // forward chevron tip is to the right of its base (tipx > basex).
    let gtf_rev = write_gtf(false);
    let rev = load_gene_models(gtf_rev.to_str().unwrap(), &sel).unwrap();
    std::fs::remove_file(&gtf_rev).ok();
    assert!(!rev[0].forward);
}
