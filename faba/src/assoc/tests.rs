//! Cross-cutting integration tests for the `assoc` module: calibration of the Bayesian
//! between-branch contrast under a pseudotime confounder, and agreement of the within-branch
//! estimators. Per-function tests live in each submodule's `tests.rs`.

use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Binomial, Distribution};

use super::contrast_bayes::{run_contrasts_bayes, BayesContrastConfig};
use super::io::{Lineage, Site};
use super::trend::{run_trends, TrendConfig};
use super::trend_bayes::{run_trends_bayes, BayesTrendConfig};
use super::Modality;

/// Null data with a pseudotime confounder: the editing rate rises with the pseudotime
/// bin but is independent of branch, while the branches occupy *different* bins (so
/// branch and rate are marginally associated). A test that ignored pseudotime would
/// fire constantly; the within-bin permutation null must stay calibrated.
fn confounded_null(n_sites: usize, seed: u64) -> (Lineage, Vec<Site>) {
    let (nbr, nbin, per) = (4usize, 5usize, 40usize);
    let cov = 20u32;
    let rate = |bin: usize| 0.15 + 0.13 * bin as f32; // 0.15 → 0.67, the confounder

    // Each branch b lives in bins {b, b+1}: adjacent branches share a bin (so the
    // permutation has within-bin contrast), yet branches differ in composition.
    let (mut names, mut pt, mut branch, mut bin_of) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for b in 0..nbr {
        for &bin in &[b.min(nbin - 1), (b + 1).min(nbin - 1)] {
            for j in 0..per {
                names.push(format!("c_{b}_{bin}_{j}").into_boxed_str());
                branch.push(b);
                pt.push(bin as f32);
                bin_of.push(bin);
            }
        }
    }
    let ncell = names.len();
    let lin = Lineage {
        cell_names: names,
        pseudotime: pt,
        branch,
        n_branches: nbr,
    };

    // Editing depends only on the bin (+ binomial noise), never on branch → null.
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut sites = Vec::with_capacity(n_sites);
    for s in 0..n_sites {
        let mut k = vec![0u32; ncell];
        let n = vec![cov; ncell];
        for c in 0..ncell {
            k[c] = Binomial::new(cov as u64, rate(bin_of[c]) as f64)
                .unwrap()
                .sample(&mut rng) as u32;
        }
        sites.push(Site {
            gene: format!("G{s}").into_boxed_str(),
            subunit: "chr1:1".into(),
            k,
            n,
        });
    }
    (lin, sites)
}

#[test]
fn between_branch_null_is_calibrated_under_confounding() {
    let (lin, sites) = confounded_null(80, 11);
    let cfg = BayesContrastConfig {
        n_bins: 5,
        min_total_coverage: 50,
        min_cells: 10,
        prior_sd: 3.0,
        n_samples: 400,
        warmup: 200,
        seed: 3,
    };
    let res = run_contrasts_bayes(&sites, &lin, &cfg);
    assert!(!res.is_empty(), "some (site, branch) should pass QC");

    // Editing depends only on the pseudotime bin, never on branch. The per-bin baseline α_b
    // conditions out that confounder, so the branch effect β must sit at ≈ 0 — small in
    // magnitude and with no systematic direction. (lfsr can still be small under high
    // coverage because the *sign* of a ~0 effect is well-determined; calibration here is
    // about magnitude.) A leaked confounder would inflate |effect| and bias its mean.
    let n = res.len() as f32;
    let mean_effect = res.iter().map(|r| r.effect).sum::<f32>() / n;
    let frac_big = res.iter().filter(|r| r.effect.abs() > 1.0).count() as f32 / n;
    assert!(
        mean_effect.abs() < 0.2,
        "systematic branch effect under the null (confounder leak?): mean={mean_effect:.3}"
    );
    assert!(
        frac_big < 0.1,
        "too many large spurious effects: {frac_big:.3} of {n} with |effect| > 1"
    );
}

/// A clear within-branch rise should be flagged by both the frequentist and Bayesian
/// estimators, with the same (positive) direction.
#[test]
fn within_branch_estimators_agree_on_direction() {
    let (per, cov) = (80usize, 30f32);
    let (mut names, mut pt, branch) = (Vec::new(), Vec::new(), vec![0usize; per]);
    for j in 0..per {
        names.push(format!("c{j}").into_boxed_str());
        pt.push(j as f32 / (per as f32 - 1.0));
    }
    let lin = Lineage {
        cell_names: names,
        pseudotime: pt.clone(),
        branch,
        n_branches: 1,
    };
    let mut k = vec![0u32; per];
    let n = vec![cov as u32; per];
    for j in 0..per {
        let t = pt[j] as f64;
        let p = 1.0 / (1.0 + (-(-2.5 + 5.0 * t)).exp());
        k[j] = (cov as f64 * p).round() as u32;
    }
    let sites = vec![Site {
        gene: "G".into(),
        subunit: "chr1:1".into(),
        k,
        n,
    }];

    let freq = run_trends(
        &sites,
        &lin,
        &TrendConfig {
            n_knots: 5,
            min_total_coverage: 30,
            min_cells: 10,
            overdispersion: true,
        },
    );
    let bayes = run_trends_bayes(
        &sites,
        &lin,
        &BayesTrendConfig {
            n_knots: 5,
            min_total_coverage: 30,
            min_cells: 10,
            prior_sd: 3.0,
            n_samples: 600,
            warmup: 200,
            seed: 1,
        },
    );
    assert_eq!(freq.len(), 1);
    assert_eq!(bayes.len(), 1);
    assert!(freq[0].effect > 0.0, "freq +effect, got {}", freq[0].effect);
    assert!(
        bayes[0].effect > 0.0,
        "bayes +effect, got {}",
        bayes[0].effect
    );
    assert!(
        freq[0].p_value < 0.05,
        "freq calls it, p={}",
        freq[0].p_value
    );
    assert!(
        bayes[0].lfsr < 0.05,
        "bayes confident, lfsr={}",
        bayes[0].lfsr
    );
}

/// The cell-type report re-pools cells across lineages: cell type "A" spans two branches,
/// both hyper-edited at site 0. Grouping by cell type and running the between-cell-type
/// contrast must recover A's positive excess (pooled over its two lineages) at site 0 and
/// stay ≈ 0 at the flat site 1.
#[test]
fn celltype_repool_recovers_shared_effect() {
    let (lin, sites, ctg) = super::test_util::celltype_shared_panel();
    // Cell-type-level Lineage: the per-cell group id is the cell type, n_branches = n_types.
    let ct_lin = Lineage {
        cell_names: lin.cell_names.clone(),
        pseudotime: lin.pseudotime.clone(),
        branch: ctg.ids.clone(),
        n_branches: ctg.names.len(),
    };
    let cfg = BayesContrastConfig {
        n_bins: 4,
        min_total_coverage: 50,
        min_cells: 10,
        prior_sd: 3.0,
        n_samples: 400,
        warmup: 200,
        seed: 5,
    };
    let res = run_contrasts_bayes(&sites, &ct_lin, &cfg);
    assert!(!res.is_empty(), "some (site, cell type) should pass QC");

    // Site 0, cell type A (id 0) — pooled over branches 0 & 1, ~80% vs ~20% rest.
    let a0 = res
        .iter()
        .find(|r| r.site == 0 && r.branch == 0)
        .expect("A / site 0 result");
    assert!(
        a0.effect > 1.0,
        "pooled cell-type A excess should be large +, got {}",
        a0.effect
    );
    assert!(a0.lfsr < 0.05, "expected confident sign, lfsr={}", a0.lfsr);
    // Site 1 is flat everywhere → A's effect ≈ 0.
    let a1 = res
        .iter()
        .find(|r| r.site == 1 && r.branch == 0)
        .expect("A / site 1 result");
    assert!(
        a1.effect.abs() < 0.5,
        "flat site should give ≈0 effect, got {}",
        a1.effect
    );
}

/// `load_celltypes`: parse a `cell<TAB>label` TSV, assign contiguous sorted ids with the
/// `unassigned` bucket last, and bucket missing / explicitly-unassigned cells there.
#[test]
fn load_celltypes_maps_sorts_and_buckets() {
    use std::io::Write;
    let mut f = tempfile::NamedTempFile::new().expect("temp file");
    write!(f, "a1\tBcell\na2\tBcell\nm1\tAcell\nu1\tunassigned\n").expect("write temp tsv");
    f.flush().expect("flush");
    let cells: Vec<Box<str>> = ["m1", "a1", "u1", "x1", "a2"]
        .iter()
        .map(|s| (*s).into())
        .collect();
    let ctg = super::io::load_celltypes(f.path().to_str().unwrap(), &cells).expect("load");

    // Assigned labels sorted (Acell, Bcell); unassigned forced last.
    assert_eq!(
        ctg.names.iter().map(|s| s.as_ref()).collect::<Vec<_>>(),
        vec!["Acell", "Bcell", "unassigned"]
    );
    // ids aligned to `cells`: m1→Acell(0), a1→Bcell(1), u1→unassigned(2),
    // x1 (absent)→unassigned(2), a2→Bcell(1).
    assert_eq!(ctg.ids, vec![0, 1, 2, 2, 1]);
    assert_eq!(ctg.unassigned_id, Some(2));
}

/// A bare-barcode annotation still joins `{barcode}@{sample}`-tagged lineage cells via
/// Membership's `@` base-key fallback, and no `unassigned` bucket appears when all match.
#[test]
fn load_celltypes_joins_at_tagged_barcodes() {
    use std::io::Write;
    let mut f = tempfile::NamedTempFile::new().expect("temp file");
    write!(f, "bc1\tAcell\nbc2\tBcell\n").expect("write temp tsv");
    f.flush().expect("flush");
    // Lineage cells are `@sample`-tagged; the annotation lists bare barcodes.
    let cells: Vec<Box<str>> = ["bc1@rep1", "bc2@rep2"]
        .iter()
        .map(|s| (*s).into())
        .collect();
    let ctg = super::io::load_celltypes(f.path().to_str().unwrap(), &cells).expect("load");

    assert_eq!(
        ctg.names.iter().map(|s| s.as_ref()).collect::<Vec<_>>(),
        vec!["Acell", "Bcell"]
    );
    assert_eq!(ctg.ids, vec![0, 1]);
    assert_eq!(
        ctg.unassigned_id, None,
        "all cells matched → no unassigned bucket"
    );
}

#[test]
fn modality_tokens_and_channels_are_distinct() {
    for m in [Modality::M6a, Modality::Atoi, Modality::Apa] {
        let (pos, neg) = m.channels();
        assert_ne!(pos, neg, "{}: channels must differ", m.token());
        assert!(!m.token().is_empty());
    }
    assert_ne!(Modality::M6a.token(), Modality::Atoi.token());
}
