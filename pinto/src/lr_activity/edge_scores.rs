//! Descriptive per-batch LR contact-association scores
//! (`pinto lra --edge-scores-only`).
//!
//! The estimand: the probability that a ligand and a receptor are
//! co-expressed across a physical contact of one link community, BEYOND
//! each side's independent activity. Per (batch, community, pair), every
//! contact contributes both orientations, and each instance is classified
//! by detection at its two endpoints into a 2x2 table:
//!
//! ```text
//!                receptor+ (2nd)   receptor- (2nd)
//! ligand+ (1st)      n11               n10
//! ligand- (1st)      n01               n00
//! ```
//!
//! The score is the posterior log odds ratio under a Jeffreys-style
//! Dirichlet(1/2) prior (the classic +1/2 in every cell), with its
//! posterior standard error:
//!
//! ```text
//! log_or    = ln[(n11+.5)(n00+.5) / ((n10+.5)(n01+.5))]
//! log_or_se = sqrt(1/(n11+.5) + 1/(n10+.5) + 1/(n01+.5) + 1/(n00+.5))
//! ```
//!
//! Both-orientation enumeration transposes the table under a ligand to
//! receptor swap, and the odds ratio is transpose-invariant, so
//! `log_or(L,R) = log_or(R,L)` is structural. The margins ship beside it
//! as `lig_rate` / `rec_rate`: each side's independent activity at these
//! contacts, which the downstream analysis needs as covariates to isolate
//! the interaction, and which are activity phenotypes in their own right.
//!
//! Detection (count > 0) deliberately replaces magnitude: on measured
//! split-half reliability the magnitude-weighted means came out near
//! noise while the detection log-odds held up, raw and with the margins
//! regressed out of both halves (numbers in the local record).
//!
//! Two honesty rules shape the outputs. A PRIOR-DOMINATED pair, with no
//! co-detection observed and less than half a co-detection expected
//! under independence, is NaN in both columns: such a table carries no
//! information about association, and the +1/2 prior alone would
//! otherwise push the estimate toward `ln(2n)` while the SE plateaus
//! instead of growing. And the SE takes the physical contact, not the
//! oriented instance, as its sampling unit; contacts sharing a cell are
//! still correlated, so `log_or_se` is a relative precision weight for
//! filtering and weighting, not a calibrated interval. No thresholds are
//! applied here, by policy.

use crate::lr_activity::fit::BATCH_LABEL_ALL;
use crate::lr_activity::orientation::CommunityStrata;
use crate::util::common::*;
use matrix_util::utils::generate_minibatch_intervals;
use rayon::prelude::*;

/// How one physical contact reads for one pair, by detection.
///
/// The ligand is ANNOTATED (the pair file names it), so orientation on a
/// contact is bookkeeping, not inference: a one-way contact identifies
/// its ligand-carrying cell outright. What a static snapshot cannot do
/// is turn that into causation; these are configuration facts.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ContactConfig {
    /// Ligand detected on exactly one side, receptor on the other.
    OneWay {
        /// True when the ligand sits on the edge's FIRST endpoint.
        ligand_first: bool,
    },
    /// Both orientations co-detected: no side to name.
    Mutual,
    /// Fewer than both roles present anywhere on the contact.
    Silent,
}

/// Classify a contact from the four detection facts.
pub fn classify_contact(l_u: bool, r_u: bool, l_v: bool, r_v: bool) -> ContactConfig {
    let fwd = l_u && r_v;
    let rev = l_v && r_u;
    match (fwd, rev) {
        (true, true) => ContactConfig::Mutual,
        (true, false) => ContactConfig::OneWay { ligand_first: true },
        (false, true) => ContactConfig::OneWay {
            ligand_first: false,
        },
        (false, false) => ContactConfig::Silent,
    }
}

/// One (batch, community, pair) score row.
pub struct EdgeScoreRow {
    pub batch: Box<str>,
    pub community: u32,
    pub ligand: Box<str>,
    pub receptor: Box<str>,
    /// Physical edges behind the score (each contributes two orientations).
    pub n_edges: u32,
    /// Mean `log1p` total count over the unique cells the edges touch.
    pub mean_log_depth: f32,
    /// Fraction of contact instances with the ligand detected at the
    /// first endpoint.
    pub lig_rate: f32,
    /// Fraction of contact instances with the receptor detected at the
    /// second endpoint.
    pub rec_rate: f32,
    /// Co-detected contacts where BOTH orientations hold: no side to
    /// name. `n_mutual + n_oneway` is the physical co-detected contact
    /// count (the 2x2's `n11` counts oriented instances, so
    /// `n11 = 2 * n_mutual + n_oneway`).
    pub n_mutual: u32,
    /// Co-detected contacts with the ligand on exactly one side: each
    /// identifies its ligand-carrying cell outright, since the role is
    /// annotated, not inferred.
    pub n_oneway: u32,
    /// How much this pair's active cells specialize into sender or
    /// receiver here: mean over cells touching a co-detected contact of
    /// `|sent - received| / (sent + received)`. 1 = every cell plays one
    /// role; 0 = every cell plays both equally. NaN when nothing is
    /// co-detected. Spot-level data mixes cells and deflates this by
    /// construction; compare across cores, not across platforms.
    pub role_purity: f32,
    /// Posterior log odds ratio of co-detection across a contact. NaN
    /// when the table is prior dominated (no co-detection observed and
    /// none expected): such a row is unmeasurable, not zero.
    pub log_or: f32,
    /// Its posterior standard error, with the physical contact as the
    /// sampling unit. NaN exactly when `log_or` is. A relative precision
    /// weight, not a calibrated interval: contacts sharing a cell are
    /// correlated.
    pub log_or_se: f32,
}

pub struct EdgeScoresInput<'a> {
    /// The tested (spatial) edge list `CommunityStrata` indexes into.
    pub edges: &'a [(usize, usize, u32, Option<Box<str>>)],
    pub strata: &'a CommunityStrata,
    /// `(ligand, receptor, gene_l, gene_r)` with GLOBAL gene ids.
    pub pairs: &'a [(Box<str>, Box<str>, usize, usize)],
    /// Global gene id → row of `x_lr`.
    pub gene_to_local: &'a HashMap<usize, usize>,
    /// Raw per-cell counts of the LR genes, `n_lr_genes × n_cells`.
    pub x_lr: &'a Mat,
    /// Per-cell `log1p` total count (all genes, not only the panel).
    pub log_depth: &'a [f32],
}

/// Compute every (batch, community, pair) row. Returns the rows and the
/// number of straddling edges dropped (endpoint batch labels differ; such
/// a contact belongs to no single batch and must not leak across two).
pub fn compute_edge_scores(input: &EdgeScoresInput<'_>) -> (Vec<EdgeScoreRow>, usize) {
    let EdgeScoresInput {
        edges,
        strata,
        pairs,
        gene_to_local,
        x_lr,
        log_depth,
    } = input;

    // Multi-batch iff any edge carries a label; with none on file the run
    // is single-batch and every edge belongs to the `all` pseudo-batch.
    let multi_batch = edges.iter().any(|e| e.3.is_some());

    let mut rows: Vec<EdgeScoreRow> = Vec::new();
    let mut n_straddling = 0usize;
    for s in 0..strata.n_strata() {
        let community = strata.community(s);

        // Oriented instance endpoints, grouped by batch. BTreeMap so row
        // order is a function of the labels, not of hashing.
        type Group = (Vec<usize>, Vec<usize>, Vec<(usize, usize)>);
        let mut by_batch: std::collections::BTreeMap<Box<str>, Group> = Default::default();
        for &(e, flipped) in strata.oriented(s) {
            let (i, j, _, ref b) = edges[e as usize];
            let label: Box<str> = if multi_batch {
                match b {
                    Some(b) => b.clone(),
                    None => {
                        // Once per edge, not per orientation.
                        if !flipped {
                            n_straddling += 1;
                        }
                        continue;
                    }
                }
            } else {
                BATCH_LABEL_ALL.into()
            };
            let (u, v) = if flipped { (j, i) } else { (i, j) };
            let slot = by_batch.entry(label).or_default();
            slot.0.push(u);
            slot.1.push(v);
            // The PHYSICAL contact once, for the configuration counts;
            // the oriented instances above serve the symmetric 2x2.
            if !flipped {
                slot.2.push((i, j));
            }
        }

        for (batch, (us, vs, contacts)) in by_batch {
            let n = us.len();
            let unique: HashSet<usize> = us.iter().copied().collect();
            let mean_log_depth =
                unique.iter().map(|&c| log_depth[c]).sum::<f32>() / unique.len().max(1) as f32;

            let batch_rows: Vec<EdgeScoreRow> = pairs
                .par_iter()
                .map(|(ligand, receptor, gl, gr)| {
                    let li = gene_to_local[gl];
                    let ri = gene_to_local[gr];
                    let mut n11 = 0usize;
                    let mut n_l = 0usize;
                    let mut n_r = 0usize;
                    for (&u, &v) in us.iter().zip(vs.iter()) {
                        let lp = x_lr[(li, u)] > 0.0;
                        let rp = x_lr[(ri, v)] > 0.0;
                        n_l += lp as usize;
                        n_r += rp as usize;
                        n11 += (lp && rp) as usize;
                    }
                    let (log_or, log_or_se) = jeffreys_log_odds(n, n11, n_l, n_r);

                    // Configuration facts per PHYSICAL contact. The role
                    // is annotated, so a one-way contact identifies its
                    // ligand-carrying cell outright; mutual contacts have
                    // no side. Per-cell tallies feed the purity summary.
                    let mut n_mutual = 0u32;
                    let mut n_oneway = 0u32;
                    let mut role: HashMap<usize, (u32, u32)> = HashMap::default();
                    for &(a, b) in contacts.iter() {
                        let cfg = classify_contact(
                            x_lr[(li, a)] > 0.0,
                            x_lr[(ri, a)] > 0.0,
                            x_lr[(li, b)] > 0.0,
                            x_lr[(ri, b)] > 0.0,
                        );
                        match cfg {
                            ContactConfig::Mutual => {
                                n_mutual += 1;
                                for c in [a, b] {
                                    let e = role.entry(c).or_default();
                                    e.0 += 1;
                                    e.1 += 1;
                                }
                            }
                            ContactConfig::OneWay { ligand_first } => {
                                n_oneway += 1;
                                let (snd, rcv) = if ligand_first { (a, b) } else { (b, a) };
                                role.entry(snd).or_default().0 += 1;
                                role.entry(rcv).or_default().1 += 1;
                            }
                            ContactConfig::Silent => {}
                        }
                    }
                    let role_purity = if role.is_empty() {
                        f32::NAN
                    } else {
                        role.values()
                            .map(|&(snd, rcv)| (snd as f32 - rcv as f32).abs() / (snd + rcv) as f32)
                            .sum::<f32>()
                            / role.len() as f32
                    };
                    EdgeScoreRow {
                        batch: batch.clone(),
                        community,
                        ligand: ligand.clone(),
                        receptor: receptor.clone(),
                        n_edges: (n / 2) as u32,
                        mean_log_depth,
                        lig_rate: n_l as f32 / n as f32,
                        rec_rate: n_r as f32 / n as f32,
                        n_mutual,
                        n_oneway,
                        role_purity,
                        log_or,
                        log_or_se,
                    }
                })
                .collect();
            rows.extend(batch_rows);
        }
    }
    (rows, n_straddling)
}

/// Posterior log odds ratio of the 2x2 contact table and its posterior
/// standard error, under a Dirichlet(1/2) prior (+1/2 in every cell).
///
/// The +1/2 is a PRIOR, not a convenience floor: it makes the estimate a
/// posterior mean, finite for empty cells. That cuts both ways, so a
/// prior-dominated table is refused instead of scored: with `n11 == 0`
/// and fewer than half a co-detection expected under independence
/// (`2 * n_l * n_r < n`), the likelihood never touches the association
/// dimension, and the +1/2 cells alone would push the estimate toward
/// `ln(2n)` while the SE plateaus near `sqrt(12)` rather than growing.
/// NaN is the honest value there. A zero-`n11` table that EXPECTED
/// co-detections stays scored, and comes out negative, which is correct.
///
/// The SE takes the physical contact as its sampling unit: `n` counts
/// oriented instances, two per contact, so the Woolf variance
/// `sum 1/(cell+1/2)` is doubled. Contacts sharing a cell remain
/// correlated beyond that, which is why the SE is documented as a
/// relative weight rather than a calibrated interval.
pub(crate) fn jeffreys_log_odds(n: usize, n11: usize, n_l: usize, n_r: usize) -> (f32, f32) {
    if n11 == 0 && 2 * n_l * n_r < n {
        return (f32::NAN, f32::NAN);
    }
    let a = n11 as f64 + 0.5;
    let b = (n_l - n11) as f64 + 0.5; // ligand only
    let c = (n_r - n11) as f64 + 0.5; // receptor only
                                      // Sum before subtracting: n + n11 >= n_l + n_r always (n00 >= 0),
                                      // but the left-to-right order would underflow usize.
    let d = (n + n11 - n_l - n_r) as f64 + 0.5; // neither
    let log_or = ((a * d) / (b * c)).ln();
    let se = (2.0 * (1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)).sqrt();
    (log_or as f32, se as f32)
}

/// Write `{out}.lr_scores.parquet`, long format: one row per
/// (batch, community, ligand, receptor).
pub fn write_edge_scores(out_prefix: &str, rows: &[EdgeScoreRow]) -> anyhow::Result<()> {
    use matrix_util::parquet::{write_named_table, Column};

    let batch: Vec<Box<str>> = rows.iter().map(|r| r.batch.clone()).collect();
    let community: Vec<i32> = rows.iter().map(|r| r.community as i32).collect();
    let ligand: Vec<Box<str>> = rows.iter().map(|r| r.ligand.clone()).collect();
    let receptor: Vec<Box<str>> = rows.iter().map(|r| r.receptor.clone()).collect();
    let n_edges: Vec<i32> = rows.iter().map(|r| r.n_edges as i32).collect();
    let mean_log_depth: Vec<f32> = rows.iter().map(|r| r.mean_log_depth).collect();
    let lig_rate: Vec<f32> = rows.iter().map(|r| r.lig_rate).collect();
    let rec_rate: Vec<f32> = rows.iter().map(|r| r.rec_rate).collect();
    let n_mutual: Vec<i32> = rows.iter().map(|r| r.n_mutual as i32).collect();
    let n_oneway: Vec<i32> = rows.iter().map(|r| r.n_oneway as i32).collect();
    let role_purity: Vec<f32> = rows.iter().map(|r| r.role_purity).collect();
    let log_or: Vec<f32> = rows.iter().map(|r| r.log_or).collect();
    let log_or_se: Vec<f32> = rows.iter().map(|r| r.log_or_se).collect();
    let row_names: Vec<Box<str>> = (0..rows.len())
        .map(|i| i.to_string().into_boxed_str())
        .collect();

    write_named_table(
        &format!("{out_prefix}.lr_scores.parquet"),
        "row",
        &row_names,
        &[
            ("batch".into(), Column::Str(&batch)),
            ("community".into(), Column::I32(&community)),
            ("ligand".into(), Column::Str(&ligand)),
            ("receptor".into(), Column::Str(&receptor)),
            ("n_edges".into(), Column::I32(&n_edges)),
            ("mean_log_depth".into(), Column::F32(&mean_log_depth)),
            ("lig_rate".into(), Column::F32(&lig_rate)),
            ("rec_rate".into(), Column::F32(&rec_rate)),
            ("n_mutual".into(), Column::I32(&n_mutual)),
            ("n_oneway".into(), Column::I32(&n_oneway)),
            ("role_purity".into(), Column::F32(&role_purity)),
            ("log_or".into(), Column::F32(&log_or)),
            ("log_or_se".into(), Column::F32(&log_or_se)),
        ],
    )
}

/// Per-cell `log1p` of the total count over ALL rows (the panel a pair is
/// scored on is a subset; depth is a property of the cell, not the panel).
pub fn per_cell_log1p_depth(
    data: &SparseIoVec,
    block_size: Option<usize>,
) -> anyhow::Result<Vec<f32>> {
    let n_cells = data.num_columns();
    let jobs = generate_minibatch_intervals(n_cells, data.num_rows(), block_size);
    let mut depth = vec![0.0f32; n_cells];
    // Jobs partition the columns, so each writes a disjoint slice; a fold
    // would allocate a full-length accumulator per worker for no benefit.
    let totals: Vec<(usize, Vec<f32>)> = jobs
        .par_iter()
        .map(|&(lb, ub)| -> anyhow::Result<(usize, Vec<f32>)> {
            let x = data.read_columns_csc(lb..ub)?;
            let mut t = vec![0.0f32; ub - lb];
            for (col, tc) in t.iter_mut().enumerate() {
                *tc = x.col(col).values().iter().sum();
            }
            Ok((lb, t))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    for (lb, t) in totals {
        for (k, v) in t.into_iter().enumerate() {
            depth[lb + k] = v.ln_1p();
        }
    }
    Ok(depth)
}
