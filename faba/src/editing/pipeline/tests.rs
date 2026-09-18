use super::sort_sites;
use crate::data::dna::{Dna, DnaBaseCount};
use crate::editing::ConversionSite;
use dashmap::DashMap;
use genomic_data::gff::GeneId;
use genomic_data::sam::Strand;

fn base_count(entries: &[(Dna, usize)]) -> DnaBaseCount {
    let mut c = DnaBaseCount::new();
    for (b, n) in entries {
        c.add(Some(b), *n);
    }
    c
}

/// A forward-strand m6A candidate: converted reads are T, unconverted are C.
fn fwd_site(
    pos: i64,
    wt_t: usize,
    wt_c: usize,
    mu_t: usize,
    mu_c: usize,
    pv: f32,
) -> ConversionSite {
    ConversionSite::M6A {
        m6a_pos: pos,
        conversion_pos: pos + 1,
        wt_freq: base_count(&[(Dna::T, wt_t), (Dna::C, wt_c)]),
        mut_freq: base_count(&[(Dna::T, mu_t), (Dna::C, mu_c)]),
        pv,
    }
}

/// A reverse-strand m6A candidate: converted reads are A, unconverted are G.
fn rev_site(
    pos: i64,
    wt_a: usize,
    wt_g: usize,
    mu_a: usize,
    mu_g: usize,
    pv: f32,
) -> ConversionSite {
    ConversionSite::M6A {
        m6a_pos: pos,
        conversion_pos: pos - 1,
        wt_freq: base_count(&[(Dna::A, wt_a), (Dna::G, wt_g)]),
        mut_freq: base_count(&[(Dna::A, mu_a), (Dna::G, mu_g)]),
        pv,
    }
}

fn atoi_site(pos: i64, counts: &[(Dna, usize)]) -> ConversionSite {
    ConversionSite::AtoI {
        editing_pos: pos,
        wt_freq: base_count(counts),
        mut_freq: DnaBaseCount::default(),
        pv: 0.5,
    }
}

fn gene_id(name: &str) -> GeneId {
    GeneId::Ensembl(name.into())
}

/// Discovery does not decide anything: every putative site comes back, with
/// its p-value untouched, whatever the p-value, odds ratio or control depth.
/// The four sites below used to land in four different buckets (selected,
/// pvalue, odds_ratio, low_control); now they are four rows.
#[test]
fn discovery_keeps_every_putative_site_with_its_statistics() {
    let sites: DashMap<GeneId, Vec<ConversionSite>> = DashMap::new();
    sites.insert(
        gene_id("G"),
        vec![
            fwd_site(400, 1, 99, 1, 99, 0.5),  // flat in both arms
            fwd_site(100, 20, 0, 0, 50, 1e-4), // strong
            fwd_site(300, 20, 0, 0, 0, 0.9),   // no control at all
            fwd_site(200, 20, 0, 0, 50, 0.9),  // weak p
        ],
    );
    let out = sort_sites(sites);
    let g = out.get(&gene_id("G")).unwrap();
    assert_eq!(g.len(), 4, "nothing is dropped or partitioned");
    let pvs: Vec<f32> = g.iter().map(|s| s.pv()).collect();
    assert_eq!(pvs, vec![1e-4, 0.9, 0.9, 0.5]);
}

/// Sites within a gene are position-sorted so the parquet and the matrices are
/// reproducible whatever order the parallel scan produced them in.
#[test]
fn sort_sites_orders_by_position_within_gene() {
    let sites: DashMap<GeneId, Vec<ConversionSite>> = DashMap::new();
    sites.insert(
        gene_id("A"),
        vec![
            rev_site(700, 1, 1, 1, 1, 0.5),
            rev_site(50, 1, 1, 1, 1, 0.5),
            rev_site(300, 1, 1, 1, 1, 0.5),
        ],
    );
    let out = sort_sites(sites);
    let pos: Vec<i64> = out
        .get(&gene_id("A"))
        .unwrap()
        .iter()
        .map(|s| s.primary_pos())
        .collect();
    assert_eq!(pos, vec![50, 300, 700]);
}

/// The parquet's `coverage` / `converted` / `control_*` columns come from
/// `signal_counts` / `control_counts`, which must read the same base as
/// `contrast_counts` on each strand, for both modalities.
#[test]
fn count_columns_follow_strand_and_modality() {
    let f = fwd_site(100, 20, 5, 2, 50, 0.1);
    assert_eq!(f.signal_counts(Strand::Forward), (20, 5));
    assert_eq!(f.control_counts(Strand::Forward), (2, 50));
    let (a_w, u_w, a_m, u_m) = f.contrast_counts(Strand::Forward).unwrap();
    assert_eq!((a_w, u_w, a_m, u_m), (20, 5, 2, 50));

    let r = rev_site(100, 7, 3, 1, 9, 0.1);
    assert_eq!(r.signal_counts(Strand::Backward), (7, 3));
    assert_eq!(r.control_counts(Strand::Backward), (1, 9));

    // A-to-I: A→G forward, T→C reverse; no control arm.
    let a = atoi_site(5, &[(Dna::A, 30), (Dna::G, 4)]);
    assert_eq!(a.signal_counts(Strand::Forward), (4, 30));
    assert_eq!(a.control_counts(Strand::Forward), (0, 0));
    assert!(a.contrast_counts(Strand::Forward).is_none());
    let t = atoi_site(5, &[(Dna::T, 30), (Dna::C, 6)]);
    assert_eq!(t.signal_counts(Strand::Backward), (6, 30));
}
