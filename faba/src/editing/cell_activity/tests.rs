use super::*;
use crate::data::dna::Dna;
use genomic_data::sam::CellBarcode;

fn cb(s: &str) -> CellBarcode {
    CellBarcode::Barcode(s.into())
}

fn tally(cells: &[(&str, u64, u64)]) -> ActivityTally {
    cells
        .iter()
        .map(|(c, e, n)| {
            (
                cb(c),
                CellActivity {
                    edited: *e,
                    covered: *n,
                    ..Default::default()
                },
            )
        })
        .collect()
}

/////////////////////////
// Method of moments   //
/////////////////////////

#[test]
fn mom_recovers_a_pure_binomial_as_zero_overdispersion() {
    // Every cell at exactly the same rate ⇒ no excess scatter ⇒ rho collapses to 0.
    let counts: Vec<(u64, u64)> = (0..50).map(|_| (10u64, 1000u64)).collect();
    let (mean, rho) = fit_betabinom_mom(&counts);
    assert!((mean - 0.01).abs() < 1e-9, "mean {mean}");
    assert_eq!(rho, 0.0, "identical rates cannot be overdispersed");
}

#[test]
fn mom_detects_overdispersion_when_rates_scatter() {
    // Same pooled mean as above, but split into two rate populations.
    let mut counts: Vec<(u64, u64)> = (0..25).map(|_| (0u64, 1000u64)).collect();
    counts.extend((0..25).map(|_| (20u64, 1000u64)));
    let (mean, rho) = fit_betabinom_mom(&counts);
    assert!((mean - 0.01).abs() < 1e-9, "mean {mean}");
    assert!(rho > 0.0, "scattered rates must give rho > 0, got {rho}");
}

#[test]
fn mom_is_degenerate_safe() {
    assert_eq!(fit_betabinom_mom(&[]), (0.0, 0.0));
    assert_eq!(fit_betabinom_mom(&[(0, 0), (0, 0)]), (0.0, 0.0));
    // All-zero numerator: a mean of 0 leaves no dispersion to estimate.
    assert_eq!(fit_betabinom_mom(&[(0, 10), (0, 10), (0, 10)]), (0.0, 0.0));
    // Fewer than 3 cells: mean is still well defined, rho is not.
    let (m, r) = fit_betabinom_mom(&[(1, 10), (3, 10)]);
    assert!((m - 0.2).abs() < 1e-9);
    assert_eq!(r, 0.0);
}

/////////////////////
// Quantile strata //
/////////////////////

#[test]
fn strata_are_equal_count_and_monotone_in_value() {
    let v: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let s = quantile_strata(&v, 10, 1);
    for k in 0..10 {
        assert_eq!(s.iter().filter(|&&x| x == k).count(), 10, "stratum {k}");
    }
    // Larger value ⇒ never a lower stratum.
    for i in 1..v.len() {
        assert!(s[i] >= s[i - 1]);
    }
}

#[test]
fn strata_collapse_when_there_are_too_few_cells_per_bin() {
    // 10 values, min 50 per stratum ⇒ a single stratum, not 10 tiny ones.
    let v: Vec<f64> = (0..10).map(|i| i as f64).collect();
    assert!(quantile_strata(&v, 10, 50).iter().all(|&s| s == 0));
    assert!(quantile_strata(&[], 10, 1).is_empty());
}

#[test]
fn strata_are_deterministic_under_ties() {
    let v = vec![1.0; 20];
    assert_eq!(quantile_strata(&v, 4, 1), quantile_strata(&v, 4, 1));
}

///////////////////////
// The null-cell QC  //
///////////////////////

/// Two clean populations: competent cells editing ~5%, null cells editing at the
/// control's ~0.5%. The cut should land exactly on that boundary.
fn planted() -> (ActivityTally, Vec<CellActivity>) {
    let mut wt = Vec::new();
    for i in 0..40 {
        wt.push((format!("hot{i:03}"), 50u64, 1000u64)); // 5%
    }
    for i in 0..60 {
        wt.push((format!("null{i:03}"), 5u64, 1000u64)); // 0.5% == control
    }
    let wt: ActivityTally = wt
        .into_iter()
        .map(|(c, e, n)| {
            (
                cb(&c),
                CellActivity {
                    edited: e,
                    covered: n,
                    ..Default::default()
                },
            )
        })
        .collect();
    let control: Vec<CellActivity> = (0..100)
        .map(|_| CellActivity {
            edited: 5,
            covered: 1000,
            ..Default::default()
        })
        .collect();
    (wt, control)
}

#[test]
fn qc_cut_separates_competent_cells_from_null_cells() {
    let (wt, control) = planted();
    // A tight tolerance pins the cut to the true boundary; see
    // `a_loose_tolerance_deliberately_leaves_signal_behind` for the trade.
    let opts = NullCallOpts {
        reject_tolerance: 1.05,
        ..NullCallOpts::default()
    };
    let call = call_competent_cells(&wt, &control, &opts);

    assert_eq!(
        call.n_selected(),
        40,
        "should keep exactly the 40 editing cells"
    );
    assert!(call
        .selected
        .iter()
        .all(|c| c.to_string().starts_with("hot")));
    // The QC statement: what was discarded looks like the dead-enzyme control.
    assert!(
        (call.rejected_over_control() - 1.0).abs() < 0.05,
        "rejected/control = {}",
        call.rejected_over_control()
    );
    // Planted separation is exactly 10x (5% vs 0.5%); recovering it confirms the
    // cut landed on the boundary rather than somewhere merely reasonable.
    assert!(
        (call.selected_rate / call.rejected_rate - 10.0).abs() < 1e-9,
        "selected/rejected = {}",
        call.selected_rate / call.rejected_rate
    );
}

#[test]
fn qc_cut_keeps_everything_when_no_cell_is_null() {
    // Every WT cell edits well above control ⇒ nothing should be thrown away.
    let wt: ActivityTally = (0..50)
        .map(|i| {
            (
                cb(&format!("hot{i:03}")),
                CellActivity {
                    edited: 50,
                    covered: 1000,
                    ..Default::default()
                },
            )
        })
        .collect();
    let control: Vec<CellActivity> = (0..50)
        .map(|_| CellActivity {
            edited: 5,
            covered: 1000,
            ..Default::default()
        })
        .collect();
    let call = call_competent_cells(&wt, &control, &NullCallOpts::default());
    assert_eq!(call.n_selected(), call.n_scored, "no cell is null here");
}

#[test]
fn a_deep_null_cell_is_not_mistaken_for_a_competent_one() {
    // A null cell with 100x the coverage of everyone else. On raw counts it has
    // by far the most converted reads; on rate it is exactly at control. Depth
    // stratification must not let its depth promote it.
    let mut wt = vec![(
        cb("deep_null"),
        CellActivity {
            edited: 500,
            covered: 100_000,
            ..Default::default()
        },
    )];
    for i in 0..40 {
        wt.push((
            cb(&format!("hot{i:03}")),
            CellActivity {
                edited: 50,
                covered: 1000,
                ..Default::default()
            },
        ));
    }
    for i in 0..40 {
        wt.push((
            cb(&format!("null{i:03}")),
            CellActivity {
                edited: 5,
                covered: 1000,
                ..Default::default()
            },
        ));
    }
    let wt: ActivityTally = wt.into_iter().collect();
    let control: Vec<CellActivity> = (0..80)
        .map(|_| CellActivity {
            edited: 5,
            covered: 1000,
            ..Default::default()
        })
        .collect();
    let call = call_competent_cells(&wt, &control, &NullCallOpts::default());
    assert!(
        !call.selected.contains(&cb("deep_null")),
        "a high-coverage cell at the control rate is still a null cell"
    );
}

#[test]
fn cells_below_the_coverage_floor_are_not_scored() {
    let wt = tally(&[("thin", 1, 5), ("hot", 50, 1000), ("null", 5, 1000)]);
    let control = vec![
        CellActivity {
            edited: 5,
            covered: 1000,
            ..Default::default()
        };
        3
    ];
    let call = call_competent_cells(&wt, &control, &NullCallOpts::default());
    assert_eq!(call.n_scored, 2, "the 5-read cell is unscorable");
    assert!(!call.selected.contains(&cb("thin")));
}

#[test]
fn a_dead_control_leaves_every_cell_selected() {
    // No control signal ⇒ no null to calibrate against ⇒ refuse to cut rather
    // than silently discard the whole WT arm.
    let wt = tally(&[("a", 50, 1000), ("b", 5, 1000)]);
    let control = vec![
        CellActivity {
            edited: 0,
            covered: 1000,
            ..Default::default()
        };
        3
    ];
    let call = call_competent_cells(&wt, &control, &NullCallOpts::default());
    assert_eq!(call.n_selected(), 2);
}

#[test]
fn the_score_is_continuous_where_a_p_value_would_underflow() {
    // A strongly editing deep cell: a tail probability would be exactly 0 here
    // and tie with every other strong cell, destroying the ranking the sweep
    // walks. The standardized deviate stays finite and ordered.
    let strong = CellActivity {
        edited: 5_000,
        covered: 10_000,
        ..Default::default()
    };
    let stronger = CellActivity {
        edited: 8_000,
        covered: 10_000,
        ..Default::default()
    };
    let a = stratum_score(&strong, 0.005, 1e-4);
    let b = stratum_score(&stronger, 0.005, 1e-4);
    assert!(a.is_finite() && b.is_finite(), "{a} {b}");
    assert!(b > a, "score must stay strictly ordered: {a} vs {b}");
}

///////////////////////////////////////
// Generality: A-to-I has no control //
///////////////////////////////////////

#[test]
fn channel_bases_follow_the_modality_and_strand() {
    use crate::editing::cell_activity::scan::channel_bases;
    use crate::editing::sifter::{M6aContrast, ModificationType};
    let m6a = ModificationType::M6A {
        check_r_site: true,
        contrast: M6aContrast {
            min_control_coverage: 1,
            min_delta: 0.0,
        },
    };
    // m6A is a C->U deamination: C->T read forward, G->A read on the reverse strand.
    assert_eq!(channel_bases(&m6a, true), (Dna::C, Dna::T));
    assert_eq!(channel_bases(&m6a, false), (Dna::G, Dna::A));
    // A-to-I is an A->I deamination: A->G forward, T->C reverse.
    assert_eq!(
        channel_bases(&ModificationType::AtoI, true),
        (Dna::A, Dna::G)
    );
    assert_eq!(
        channel_bases(&ModificationType::AtoI, false),
        (Dna::T, Dna::C)
    );
}

#[test]
fn a_loose_tolerance_deliberately_leaves_signal_behind() {
    // The tolerance is the whole knob: it says how much editing the DISCARDED
    // pool may still show. Loosen it and the cut moves earlier, sacrificing a
    // little real signal for a purer kept pool — a trade, not a bug. This pins
    // the direction so a future change cannot silently invert it.
    let (wt, control) = planted();
    let tight = call_competent_cells(
        &wt,
        &control,
        &NullCallOpts {
            reject_tolerance: 1.05,
            ..NullCallOpts::default()
        },
    );
    let loose = call_competent_cells(
        &wt,
        &control,
        &NullCallOpts {
            reject_tolerance: 1.5,
            ..NullCallOpts::default()
        },
    );
    assert!(
        loose.n_selected() < tight.n_selected(),
        "a looser tolerance must cut harder: {} vs {}",
        loose.n_selected(),
        tight.n_selected()
    );
    assert!(loose.rejected_over_control() <= 1.5);
    assert!(tight.rejected_over_control() <= 1.05);
}

#[test]
fn candidate_positions_would_shift_past_an_assembly_gap() {
    // Regression for the index/coordinate bug: `fetch_reference_seq` filter_maps
    // non-ACGT away, silently SHORTENING the vector, so `pos = lo + i` lands off
    // by the number of skipped bases after any `N`. `fetch_reference_bases` keeps
    // them as `None`. hg38 gene spans do overlap assembly gaps, so this is a real
    // input, and the failure is silent: the scan would tally at wrong positions.
    use crate::data::dna::Dna;
    let raw = b"AANACGT";
    let dropped: Vec<Dna> = raw
        .iter()
        .filter_map(|&b| Dna::from_byte(b.to_ascii_uppercase()))
        .collect();
    let kept: Vec<Option<Dna>> = raw
        .iter()
        .map(|&b| Dna::from_byte(b.to_ascii_uppercase()))
        .collect();
    assert_eq!(dropped.len(), 6, "filter_map drops the N");
    assert_eq!(kept.len(), 7, "length-preserving keeps it as None");
    // Truth at index 3 is 'A' (A A N A C G T). With the N dropped, index 3 reads
    // the base that really lives at index 4 — every coordinate past the gap is
    // shifted by the number of skipped bases.
    assert_eq!(
        kept[3],
        Some(Dna::A),
        "length-preserving keeps the coordinate"
    );
    assert_eq!(
        dropped[3],
        Dna::C,
        "filter_map shifts index 3 to the base at 4"
    );
}

///////////////////////////////////////
// The motif / background scan masks //
///////////////////////////////////////

/// A reference with one forward motif (`GAC` at 12), one reverse motif (`GTC` at
/// 60), a `TAC` at 52 that only `--no-check-r-site` admits, non-motif C's on both
/// sides of the 25 nt keep-out boundary, and a run of A's for A-to-I. Five
/// trailing pad bases keep the `stop + 2` context window inside the contig.
fn scan_reference() -> String {
    let mut s = String::new();
    s.push_str(&"T".repeat(10)); //   0..=9
    s.push_str("GAC"); //            10..=12  forward motif at 12
    s.push('C'); //                  13       non-motif C, 1 nt from the motif
    s.push_str(&"T".repeat(23)); //  14..=36
    s.push('C'); //                  37       exactly 25 nt out: still kept out
    s.push_str(&"C".repeat(12)); //  38..=49  38 is 26 nt out: the first eligible
    s.push_str("TAC"); //            50..=52  motif only without the R check
    s.push_str(&"T".repeat(7)); //   53..=59
    s.push_str("GTC"); //            60..=62  reverse motif at 60
    s.push_str(&"T".repeat(37)); //  63..=99
    s.push_str(&"G".repeat(10)); // 100..=109 reverse background candidates
    s.push_str(&"A".repeat(10)); // 110..=119
    s.push_str(&"T".repeat(15)); // 120..=134 (the last 5 are pad)
    s
}

fn scan_fasta(seq: &str) -> (tempfile::NamedTempFile, rust_htslib::faidx::Reader) {
    use std::io::Write;
    let mut f = tempfile::NamedTempFile::with_suffix(".fa").unwrap();
    writeln!(f, ">chr1").unwrap();
    writeln!(f, "{}", seq).unwrap();
    f.flush().unwrap();
    let path = f.path().to_str().unwrap().to_string();
    let reader = rust_htslib::faidx::Reader::from_path(&path).unwrap();
    (f, reader)
}

/// The pre-mask implementation, transcribed: two hash sets, `contains` per base,
/// and a 51-wide `contains` probe for the keep-out. The dense mask has to agree
/// with it position for position — including *which* eligible positions the
/// stride keeps, which depends on the sweep order.
fn hashed_reference_sets(
    faidx: &rust_htslib::faidx::Reader,
    start: i64,
    stop: i64,
    forward: bool,
    mod_type: &crate::editing::sifter::ModificationType,
) -> (
    std::collections::BTreeSet<i64>,
    std::collections::BTreeSet<i64>,
) {
    use crate::data::util_htslib::fetch_reference_bases;
    use crate::editing::cell_activity::scan::channel_bases;
    use crate::editing::sifter::ModificationType;
    use std::collections::BTreeSet;

    let lo = (start - 2).max(0);
    let (ref_base, _) = channel_bases(mod_type, forward);
    let (mut motif, mut bg) = (BTreeSet::new(), BTreeSet::new());
    let Ok(Some(seq)) = fetch_reference_bases(faidx, "chr1", lo, stop + 2) else {
        return (motif, bg);
    };
    for i in 2..seq.len().saturating_sub(2) {
        let pos = lo + i as i64;
        if pos < start || pos >= stop || seq[i] != Some(ref_base) {
            continue;
        }
        let hit = match mod_type {
            ModificationType::AtoI => true,
            ModificationType::M6A { check_r_site, .. } if forward => {
                seq[i - 1] == Some(Dna::A)
                    && (!check_r_site || matches!(seq[i - 2], Some(Dna::A) | Some(Dna::G)))
            }
            ModificationType::M6A { check_r_site, .. } => {
                seq[i + 1] == Some(Dna::T)
                    && (!check_r_site || matches!(seq[i + 2], Some(Dna::C) | Some(Dna::T)))
            }
        };
        if hit {
            motif.insert(pos);
        }
    }
    let mut eligible = 0usize;
    for (i, b) in seq.iter().enumerate() {
        let pos = lo + i as i64;
        if pos < start || pos >= stop || *b != Some(ref_base) || motif.contains(&pos) {
            continue;
        }
        if (pos - 25..=pos + 25).any(|p| motif.contains(&p)) {
            continue;
        }
        eligible += 1;
        if eligible.is_multiple_of(4) {
            bg.insert(pos);
        }
    }
    (motif, bg)
}

/// Read the two sets back out of the dense mask.
fn mask_sets(
    pos: &crate::editing::cell_activity::scan::GenePositions,
    start: i64,
    stop: i64,
) -> (
    std::collections::BTreeSet<i64>,
    std::collections::BTreeSet<i64>,
) {
    use crate::editing::cell_activity::scan::PositionClass;
    use std::collections::BTreeSet;
    let (mut motif, mut bg) = (BTreeSet::new(), BTreeSet::new());
    // One past each end, so a mask that spilled outside the span would show up.
    for p in (start - 1)..(stop + 1) {
        match pos.class_at(p) {
            PositionClass::Motif => {
                motif.insert(p);
            }
            PositionClass::Background => {
                bg.insert(p);
            }
            PositionClass::Neither => {}
        }
    }
    (motif, bg)
}

#[test]
fn the_dense_mask_reproduces_the_hashed_scan_exactly() {
    use crate::editing::cell_activity::scan::candidate_and_background;
    use crate::editing::sifter::{M6aContrast, ModificationType};
    let m6a = |check_r_site: bool| ModificationType::M6A {
        check_r_site,
        contrast: M6aContrast {
            min_control_coverage: 1,
            min_delta: 0.0,
        },
    };
    let (_f, faidx) = scan_fasta(&scan_reference());

    for mod_type in [m6a(true), m6a(false), ModificationType::AtoI] {
        for forward in [true, false] {
            // (0, 130) exercises the contig edge, where `lo` clamps to 0 and the
            // first two bases have no motif context but are still swept for
            // background; the others cut motifs and keep-out zones off-centre.
            for (start, stop) in [(0i64, 130i64), (5, 100), (12, 63), (40, 41), (40, 40)] {
                let want = hashed_reference_sets(&faidx, start, stop, forward, &mod_type);
                let got = mask_sets(
                    &candidate_and_background(&faidx, "chr1", start, stop, forward, &mod_type),
                    start,
                    stop,
                );
                assert_eq!(
                    got, want,
                    "{mod_type:?} forward={forward} span={start}..{stop}"
                );
            }
        }
    }
}

#[test]
fn the_dense_mask_agrees_on_references_nobody_designed() {
    // The designed reference pins what the rule *means*; this pins the rewrite
    // against inputs no one chose — `N` gaps, motifs abutting the span edge,
    // keep-out zones that overlap each other and run off both ends, and spans
    // that hang off the contig so the fetch comes back short. Seeded, so any
    // failure reproduces exactly.
    use crate::editing::cell_activity::scan::candidate_and_background;
    use crate::editing::sifter::{M6aContrast, ModificationType};
    let m6a = |check_r_site: bool| ModificationType::M6A {
        check_r_site,
        contrast: M6aContrast {
            min_control_coverage: 1,
            min_delta: 0.0,
        },
    };
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    // Three alphabets: uniform, then C-heavy and A-heavy so motifs crowd close
    // enough for their keep-out zones to merge.
    for alphabet in [b"ACGTN".as_slice(), b"CCCAGTN", b"AAACGTN"] {
        let n = 1_500;
        let seq: String = (0..n)
            .map(|_| alphabet[(next() % alphabet.len() as u64) as usize] as char)
            .collect();
        let (_f, faidx) = scan_fasta(&seq);
        for _ in 0..8 {
            let start = (next() % 40) as i64;
            let stop = start + (next() % (n as u64 + 40)) as i64;
            for mod_type in [m6a(true), m6a(false), ModificationType::AtoI] {
                for forward in [true, false] {
                    let want = hashed_reference_sets(&faidx, start, stop, forward, &mod_type);
                    let got = mask_sets(
                        &candidate_and_background(&faidx, "chr1", start, stop, forward, &mod_type),
                        start,
                        stop,
                    );
                    assert_eq!(
                        got, want,
                        "{mod_type:?} forward={forward} span={start}..{stop}"
                    );
                }
            }
        }
    }
}

#[test]
fn the_single_fetch_leaves_candidate_positions_unchanged() {
    // FIX: `candidate_and_background` used to re-fetch the identical window to
    // run the same classifier a second time. The wrapper it now shares must
    // still answer exactly what the mask holds.
    use crate::editing::cell_activity::scan::{candidate_and_background, candidate_positions};
    use crate::editing::sifter::{M6aContrast, ModificationType};
    let mod_type = ModificationType::M6A {
        check_r_site: true,
        contrast: M6aContrast {
            min_control_coverage: 1,
            min_delta: 0.0,
        },
    };
    let (_f, faidx) = scan_fasta(&scan_reference());
    for forward in [true, false] {
        let wrapper: std::collections::BTreeSet<i64> =
            candidate_positions(&faidx, "chr1", 0, 130, forward, &mod_type)
                .into_iter()
                .collect();
        let (motif, _) = mask_sets(
            &candidate_and_background(&faidx, "chr1", 0, 130, forward, &mod_type),
            0,
            130,
        );
        assert_eq!(wrapper, motif, "forward={forward}");
    }
}

#[test]
fn background_skips_the_keep_out_zone_and_then_takes_every_fourth() {
    // Hand-computed on `scan_reference`, so a bug shared by the transcription in
    // `hashed_reference_sets` cannot hide behind an equality test. Forward m6A:
    // the only motif C is 12; C at 13 is 1 nt away, inside the keep-out; the C
    // run at 38..=49 starts at exactly 26 nt out, so all 12 are eligible, then
    // 52 and 62 follow. Every 4th eligible position survives the stride.
    use crate::editing::cell_activity::scan::candidate_and_background;
    use crate::editing::sifter::{M6aContrast, ModificationType};
    let mod_type = ModificationType::M6A {
        check_r_site: true,
        contrast: M6aContrast {
            min_control_coverage: 1,
            min_delta: 0.0,
        },
    };
    let (_f, faidx) = scan_fasta(&scan_reference());
    let pos = candidate_and_background(&faidx, "chr1", 0, 130, true, &mod_type);
    assert!(pos.has_motif());
    let (motif, bg) = mask_sets(&pos, 0, 130);
    let want_motif: std::collections::BTreeSet<i64> = [12].into_iter().collect();
    let want_bg: std::collections::BTreeSet<i64> = [41, 45, 49].into_iter().collect();
    assert_eq!(motif, want_motif, "GAC, edit at the C");
    assert_eq!(
        bg, want_bg,
        "4th, 8th and 12th eligible; the 13th and 14th never complete a group"
    );

    // A-to-I has no motif, so every reference A is a candidate and every A is
    // therefore inside some keep-out zone: the background channel is empty by
    // construction, not by accident.
    let atoi = candidate_and_background(&faidx, "chr1", 0, 130, true, &ModificationType::AtoI);
    let (m, b) = mask_sets(&atoi, 0, 130);
    assert!(!m.is_empty() && b.is_empty(), "{m:?} {b:?}");
}

#[test]
fn feature_matching_survives_name_canonicalisation() {
    use crate::editing::cell_activity::feature_is_gene;
    // Raw matrix rows carry the Ensembl prefix...
    assert!(feature_is_gene(
        "ENSG00000198492_YTHDF2/count/spliced",
        "YTHDF2"
    ));
    // ...but `load_unified_data` rsplits on '_', leaving the bare symbol. Matching
    // only the prefixed form reported "none matched" for genes that were present.
    assert!(feature_is_gene("YTHDF2/count/spliced", "YTHDF2"));
    // Anchored, so a longer symbol sharing a prefix does not match.
    assert!(!feature_is_gene("METTL3L/count/spliced", "METTL3"));
    assert!(!feature_is_gene(
        "ENSG00000000000_RBM15B/count/spliced",
        "RBM15"
    ));
    assert!(!feature_is_gene("SOMETHING_ELSE/count/spliced", "YTHDF2"));
}
